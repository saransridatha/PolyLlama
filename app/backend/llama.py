"""llama.cpp server process adapter.

The gateway deliberately talks to llama.cpp's maintained HTTP server rather than
reimplementing inference. Each resident model gets its own loopback-only process,
which isolates crashes and allows per-model context/GPU configuration.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import PROJECT_ROOT, Settings
from ..logging_setup import redact

RESTRICTED_EXTRA_FLAGS = {
    "--host",
    "--port",
    "--model",
    "-m",
    "--alias",
    "-a",
    "--api-key",
    "--ssl-key-file",
    "--ssl-cert-file",
    "--path",
    "--log-file",
    "--no-webui",
    "--webui",
    "--no-ui",
    "--ui",
}


class BackendError(Exception):
    def __init__(self, message: str, *, log_tail: str = ""):
        super().__init__(message)
        self.message = message
        self.log_tail = log_tail


def find_llama_server(settings: Settings) -> Path | None:
    candidates: list[Path] = []
    if settings.llama_server_bin:
        candidates.append(Path(settings.llama_server_bin).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "vendor" / "llama-build" / "bin" / "llama-server",
            PROJECT_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server",
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
            Path("/usr/local/bin/llama-server"),
            Path("/opt/homebrew/bin/llama-server"),
            Path("/usr/bin/llama-server"),
        ]
    )
    on_path = shutil.which("llama-server")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def allocate_port(preferred: int | None = None) -> int:
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def llama_version(binary: Path) -> str:
    try:
        import subprocess  # nosec B404 - only executes the configured llama-server binary

        proc = subprocess.run(  # nosec B603 - fixed binary path, argument list, no shell
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=_clean_env(binary),
        )
        first = (proc.stdout or proc.stderr).strip().splitlines()
        return first[0][:200] if first else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _clean_env(binary: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MS_")
        and k
        not in {
            "OPENAI_API_KEY",
            "CLOUDFLARE_TUNNEL_TOKEN",
            "ANTHROPIC_API_KEY",
            "HF_TOKEN",
        }
    }
    bin_dir = str(binary.parent)
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{current}" if current else bin_dir
    env.setdefault("GGML_NATIVE", "1")
    return env


@dataclass(slots=True)
class BackendConfig:
    model_id: str
    model_path: str
    context_total: int
    slots: int
    gpu_layers: int
    threads: int
    alias: str
    batch_size: int = 2048
    ubatch_size: int = 512
    embeddings: bool = False
    extra_args: list[str] = field(default_factory=list)
    flash_attn: str = "auto"


class LlamaBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.binary = find_llama_server(settings)
        if not self.binary:
            raise BackendError(
                "llama-server binary not found. Run scripts/build_llama.sh or set "
                "MS_LLAMA_SERVER_BIN."
            )
        self.version = llama_version(self.binary)
        self.process: asyncio.subprocess.Process | None = None
        self.port: int | None = None
        self.config: BackendConfig | None = None
        self.log_path: Path | None = None
        self.started_at: float = 0.0
        self._log_handle: Any = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    @property
    def base_url(self) -> str:
        if not self.port:
            raise BackendError("Backend is not running")
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self, config: BackendConfig) -> None:
        if self.alive:
            return
        if not self.binary:
            raise BackendError("llama-server binary is unavailable")
        self.config = config
        self.port = allocate_port()
        self.log_path = self.settings.log_dir / f"llama-{config.model_id}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab", buffering=0)
        try:
            self.log_path.chmod(0o600)
        except OSError:
            pass
        extra = self._sanitize_extra_args(config.extra_args)
        args = [
            str(self.binary),
            "--model",
            config.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(config.context_total),
            "--parallel",
            str(config.slots),
            "--n-gpu-layers",
            str(config.gpu_layers),
            "--threads",
            str(config.threads),
            "--batch-size",
            str(config.batch_size),
            "--ubatch-size",
            str(config.ubatch_size),
            "--alias",
            config.alias,
            *(["--embeddings"] if config.embeddings else []),
            "--no-webui",
            "--metrics",
            "--jinja",
            "--flash-attn",
            config.flash_attn if config.flash_attn in ("on", "off", "auto") else "auto",
            "--log-file",
            str(self.log_path),
            *extra,
        ]
        header = f"\n=== ModelSwitch start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        self._log_handle.write(header.encode())
        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=self._log_handle,
                env=_clean_env(self.binary),
                start_new_session=True,
            )
        except OSError as exc:
            self._close_log()
            raise BackendError(f"Unable to start llama-server: {exc}") from exc
        self.started_at = time.time()
        try:
            await self._wait_ready()
        except BackendError:
            await self.stop(force=True)
            raise

    def _sanitize_extra_args(self, args: list[str]) -> list[str]:
        clean: list[str] = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            flag = arg.split("=", 1)[0]
            if flag in RESTRICTED_EXTRA_FLAGS:
                if "=" not in arg:
                    skip_next = True
                continue
            clean.append(arg)
        return clean[:32]

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.settings.backend_start_timeout
        last_error = ""
        while time.monotonic() < deadline:
            if not self.alive:
                raise BackendError(
                    "llama-server exited during startup", log_tail=self.read_log_tail()
                )
            try:
                resp = await self._client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    # Confirm the server speaks the expected API generation.
                    props = await self._client.get(f"{self.base_url}/props")
                    if props.status_code == 200:
                        return
                    last_error = f"/props returned {props.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(0.35)
        raise BackendError(
            f"llama-server did not become ready within {self.settings.backend_start_timeout:.0f}s"
            + (f" ({last_error})" if last_error else ""),
            log_tail=self.read_log_tail(),
        )

    async def stop(self, *, force: bool = False) -> None:
        proc = self.process
        if not proc:
            self._close_log()
            return
        if proc.returncode is None:
            try:
                proc.terminate()
                timeout = 2.0 if force else 10.0
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (ProcessLookupError, asyncio.TimeoutError):
                    pass
        self.process = None
        self.port = None
        self._close_log()

    def _close_log(self) -> None:
        if self._log_handle:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def read_log_tail(self, lines: int = 80) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        try:
            content = self.log_path.read_text(errors="replace").splitlines()
            return redact("\n".join(content[-lines:]))
        except OSError:
            return ""

    async def health(self) -> dict[str, Any]:
        if not self.alive:
            return {"healthy": False, "reason": "process not running"}
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=2.0)
            return {
                "healthy": resp.status_code == 200,
                "status_code": resp.status_code,
                "latency_ms": round(resp.elapsed.total_seconds() * 1000, 2),
            }
        except httpx.HTTPError as exc:
            return {"healthy": False, "reason": str(exc)}

    async def props(self) -> dict[str, Any]:
        resp = await self._client.get(f"{self.base_url}/props")
        resp.raise_for_status()
        return resp.json()

    def process_stats(self) -> dict[str, Any]:
        if not self.process:
            return {}
        try:
            import psutil

            proc = psutil.Process(self.process.pid)
            with proc.oneshot():
                return {
                    "pid": self.process.pid,
                    "rss_bytes": proc.memory_info().rss,
                    "cpu_percent": proc.cpu_percent(interval=None),
                    "threads": proc.num_threads(),
                    "created_at": proc.create_time(),
                }
        except Exception:
            return {"pid": self.process.pid if self.process else None}

    async def close(self) -> None:
        await self.stop(force=True)
        await self._client.aclose()
