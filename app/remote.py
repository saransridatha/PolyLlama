"""Optional Cloudflare Tunnel lifecycle.

The tunnel always targets the public inference listener only. The admin dashboard
and control API live on a separate loopback listener and are never proxied.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import PROJECT_ROOT, Settings
from .logging_setup import redact

log = logging.getLogger("modelswitch.remote")

URL_RE = re.compile(r"https://[a-z0-9][a-z0-9.-]*\.trycloudflare\.com")
TOKEN_RE = re.compile(r"^[A-Za-z0-9+/=_-]{20,4096}$")


class RemoteError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def find_cloudflared(settings: Settings) -> Path | None:
    candidates = []
    if settings.cloudflare_bin:
        candidates.append(Path(settings.cloudflare_bin).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "vendor" / "cloudflared",
            Path.home() / ".local" / "bin" / "cloudflared",
            Path("/usr/local/bin/cloudflared"),
            Path("/opt/homebrew/bin/cloudflared"),
            Path("/usr/bin/cloudflared"),
        ]
    )
    path = shutil.which("cloudflared")
    if path:
        candidates.append(Path(path))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


class CloudflareTunnel:
    def __init__(self, settings: Settings, target_port: int | None = None):
        self.settings = settings
        self.target_port = target_port or settings.public_port
        self.binary = find_cloudflared(settings)
        self.process: asyncio.subprocess.Process | None = None
        self.mode: str | None = None
        self.public_url: str | None = settings.cloudflare_public_url or None
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.health_task: asyncio.Task[None] | None = None
        self._last_config: dict[str, Any] | None = None
        self._token: str | None = None
        self._log_path = settings.log_dir / "cloudflared.log"
        self._log: deque[str] = deque(maxlen=200)
        self._healthy: bool | None = None
        self._last_check: float | None = None
        self._health_error: str | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def _target(self) -> str:
        return f"http://127.0.0.1:{self.target_port}"

    def _validate_public_url(self, value: str | None) -> str | None:
        if not value:
            return None
        value = value.strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise RemoteError("Public URL must be a valid https:// URL")
        if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
            raise RemoteError("Public URL must be externally reachable")
        return value

    async def start(
        self,
        *,
        mode: str = "quick",
        token: str | None = None,
        public_url: str | None = None,
        tunnel_name: str | None = None,
    ) -> dict[str, Any]:
        if mode not in ("quick", "token"):
            raise RemoteError("mode must be quick or token")
        if self.running:
            raise RemoteError("Tunnel is already running", 409)
        if not self.binary:
            raise RemoteError(
                "cloudflared is not installed. Install it from "
                "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ "
                "or set MS_CLOUDFLARE_BIN.",
                503,
            )
        token = token or self.settings.cloudflare_tunnel_token or None
        if mode == "token":
            if not token:
                raise RemoteError(
                    "A tunnel token is required. Create a named tunnel in the Cloudflare "
                    "Zero Trust dashboard and copy its connector token.",
                    400,
                )
            if not TOKEN_RE.match(token):
                raise RemoteError("Tunnel token format is invalid")
        if public_url:
            public_url = self._validate_public_url(public_url)
        else:
            public_url = self._validate_public_url(self.settings.cloudflare_public_url or None)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._log_path.chmod(0o600)
        except OSError:
            pass
        log_handle = self._log_path.open("ab", buffering=0)
        args = [str(self.binary), "tunnel", "--no-autoupdate", "--loglevel", "info"]
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "MS_KEY_PEPPER",
                "CLOUDFLARE_TUNNEL_TOKEN",
                "MS_CLOUDFLARE_TUNNEL_TOKEN",
                "OPENAI_API_KEY",
            }
        }
        if mode == "quick":
            args.extend(["--url", self._target()])
        else:
            env["TUNNEL_TOKEN"] = token  # never placed on the command line
            args.extend(["run"])
        args.extend(["--logfile", str(self._log_path)])
        self._token = token
        self.mode = mode
        self.public_url = public_url
        self.last_error = None
        self._healthy = None
        self._health_error = None
        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            raise RemoteError(f"Failed to start cloudflared: {exc}", 503) from exc
        finally:
            try:
                log_handle.close()
            except OSError:
                pass
        self.started_at = time.time()
        try:
            await self._await_ready(mode)
        except Exception:
            await self.stop()
            raise
        self._last_config = {
            "mode": mode,
            "public_url": self.public_url,
            "tunnel_name": tunnel_name,
            "token": token,
        }
        if not self.health_task:
            self.health_task = asyncio.create_task(self._health_loop(), name="cloudflared-health")
        log.info("cloudflare tunnel started", extra={"event": "tunnel.start", "status": mode})
        return await self.status()

    async def _await_ready(self, mode: str) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process and self.process.returncode is not None:
                raise RemoteError(
                    "cloudflared exited during startup: " + self._tail(), 503
                )
            self._refresh_log()
            if mode == "quick" and not self.public_url:
                match = URL_RE.search("\n".join(self._log))
                if match:
                    self.public_url = match.group(0)
            if self.public_url or mode == "token":
                if mode == "token":
                    # Named tunnels do not necessarily print their hostname; a
                    # custom public URL is supplied by the operator.
                    await asyncio.sleep(1.0)
                return
            await asyncio.sleep(0.5)
        raise RemoteError(
            "cloudflared started but no public hostname was discovered within 30s. "
            "Check the tunnel log and Cloudflare DNS configuration.",
            503,
        )

    def _refresh_log(self) -> None:
        try:
            content = self._log_path.read_text(errors="replace").splitlines()
            self._log = deque(content[-200:], maxlen=200)
        except OSError:
            pass

    def _tail(self, lines: int = 20) -> str:
        self._refresh_log()
        return redact("\n".join(list(self._log)[-lines:]))

    async def stop(self) -> dict[str, Any]:
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except (asyncio.CancelledError, Exception):
                pass
            self.health_task = None
        proc = self.process
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (ProcessLookupError, asyncio.TimeoutError):
                    pass
        self.process = None
        self.started_at = None
        self._healthy = False
        log.info("cloudflare tunnel stopped", extra={"event": "tunnel.stop"})
        return {"running": False, "status": "stopped"}

    async def restart(self) -> dict[str, Any]:
        if not self._last_config:
            raise RemoteError("No previous tunnel configuration to restart", 409)
        config = dict(self._last_config)
        await self.stop()
        return await self.start(
            mode=config["mode"],
            token=config.get("token"),
            public_url=config.get("public_url"),
            tunnel_name=config.get("tunnel_name"),
        )

    async def _health_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(20)
                await self.check_health()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._health_error = str(exc)

    async def check_health(self) -> dict[str, Any]:
        if not self.running:
            self._healthy = False
            return {"healthy": False, "error": "tunnel process is not running"}
        self._refresh_log()
        if not self.public_url:
            return {"healthy": None, "error": "public URL is not configured"}
        url = self.public_url.rstrip("/") + "/healthz"
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
                resp = await client.get(url)
            self._healthy = resp.status_code == 200
            self._health_error = None if self._healthy else f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            self._healthy = False
            self._health_error = str(exc)
        self._last_check = time.time()
        return {
            "healthy": self._healthy,
            "checked_at": self._last_check,
            "error": self._health_error,
        }

    async def status(self) -> dict[str, Any]:
        self._refresh_log()
        health = {
            "healthy": self._healthy,
            "checked_at": self._last_check,
            "error": self._health_error,
        }
        api_base = None
        if self.public_url:
            api_base = self.public_url.rstrip("/") + "/v1"
        return {
            "installed": bool(self.binary),
            "binary": str(self.binary) if self.binary else None,
            "supported": True,
            "running": self.running,
            "mode": self.mode,
            "public_url": self.public_url,
            "api_base_url": api_base,
            "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else None,
            "pid": self.process.pid if self.process else None,
            "health": health,
            "last_error": self.last_error or self._health_error,
            "log_tail": redact("\n".join(list(self._log)[-40:])),
            "target": self._target(),
            "admin_exposed": False,
            "note": (
                "The tunnel proxies only the public inference listener. The admin "
                "dashboard is bound to a separate loopback port."
            ),
            "prerequisites": {
                "quick_tunnel": "cloudflared installed; outbound HTTPS/QUIC access",
                "named_tunnel": (
                    "Cloudflare account, a domain on Cloudflare, a named tunnel, and a "
                    "public hostname routed to the local service"
                ),
            },
        }

    async def close(self) -> None:
        await self.stop()
