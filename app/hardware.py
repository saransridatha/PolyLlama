"""Hardware discovery and live resource accounting.

Works without optional vendor libraries. GPU details are best-effort and the
gateway always retains a CPU-only path.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess  # nosec B404 - fixed vendor diagnostic commands, no shell
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from .gguf import GGUFMetadata


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    vendor: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization: float | None = None
    temperature_c: float | None = None
    driver: str = ""
    cuda: bool = False
    backend: str = "cuda"

    @property
    def total_bytes(self) -> int:
        return self.total_mb * 1024 * 1024

    @property
    def free_bytes(self) -> int:
        return self.free_mb * 1024 * 1024


@dataclass(slots=True)
class HardwareInfo:
    os_name: str
    os_version: str
    arch: str
    python_version: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    total_ram_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    gpus: list[GPUInfo] = field(default_factory=list)
    cuda_available: bool = False
    apple_silicon: bool = False
    llama_backend: str = "cpu"
    detected_at: float = field(default_factory=time.time)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, dynamic: dict[str, Any] | None = None) -> dict[str, Any]:
        data = asdict(self)
        data["gpus"] = [asdict(g) for g in self.gpus]
        data["total_ram_gb"] = round(self.total_ram_bytes / 1024**3, 2)
        data["disk_free_gb"] = round(self.disk_free_bytes / 1024**3, 2)
        if dynamic:
            data.update(dynamic)
        return data


@dataclass(slots=True)
class MemorySnapshot:
    total_ram_bytes: int
    available_ram_bytes: int
    used_ram_bytes: int
    ram_percent: float
    swap_used_bytes: int
    disk_free_bytes: int
    gpu_free_bytes: int = 0
    gpu_total_bytes: int = 0
    gpu_used_bytes: int = 0
    gpu_utilization: float | None = None
    cpu_percent: float = 0.0
    load_1m: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ram_bytes": self.total_ram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "used_ram_bytes": self.used_ram_bytes,
            "ram_percent": round(self.ram_percent, 1),
            "swap_used_bytes": self.swap_used_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "disk_free_gb": round(self.disk_free_bytes / 1024**3, 2),
            "gpu_free_bytes": self.gpu_free_bytes,
            "gpu_total_bytes": self.gpu_total_bytes,
            "gpu_used_bytes": self.gpu_used_bytes,
            "gpu_utilization": self.gpu_utilization,
            "cpu_percent": round(self.cpu_percent, 1),
            "load_1m": round(self.load_1m, 2),
            "timestamp": self.timestamp,
        }


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        proc = subprocess.run(  # nosec B603 - fixed argument lists, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def _cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Darwin":
        value = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value:
            return value
    if system == "Windows":
        value = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ]
        )
        if value:
            return value.splitlines()[0]
    return platform.processor() or platform.machine()


def _nvidia_gpus() -> tuple[list[GPUInfo], str]:
    if not shutil.which("nvidia-smi"):
        return [], ""
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,"
            "temperature.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=4,
    )
    if not raw:
        return [], ""
    gpus = []
    warnings = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            total = int(float(parts[2]))
            used = int(float(parts[3]))
            free = int(float(parts[4]))
            util = float(parts[5]) if parts[5] not in ("N/A", "[N/A]") else None
            temp = float(parts[6]) if parts[6] not in ("N/A", "[N/A]") else None
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                vendor="NVIDIA",
                total_mb=total,
                used_mb=used,
                free_mb=free,
                utilization=util,
                temperature_c=temp,
                driver=parts[7],
                cuda=True,
                backend="cuda",
            )
        )
    return gpus, "; ".join(warnings)


def _mac_gpus() -> list[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=6)
    if not raw:
        return []
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    gpus = []
    for idx, item in enumerate(data.get("SPDisplaysDataType", [])):
        name = item.get("sppci_model") or item.get("_name") or "Apple GPU"
        vram = item.get("spdisplays_vram") or ""
        total_mb = 0
        if isinstance(vram, str):
            try:
                total_mb = int(vram.replace(" MB", "").replace(" GB", "").strip()) * (
                    1024 if "GB" in vram else 1
                )
            except ValueError:
                total_mb = 0
        gpus.append(
            GPUInfo(
                index=idx,
                name=name,
                vendor="Apple",
                total_mb=total_mb,
                used_mb=0,
                free_mb=total_mb,
                backend="metal",
            )
        )
    return gpus


def _windows_gpus() -> list[GPUInfo]:
    if platform.system() != "Windows":
        return []
    raw = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
        ],
        timeout=6,
    )
    if not raw:
        return []
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    gpus = []
    for idx, item in enumerate(data):
        name = str(item.get("Name") or "GPU")
        total = int(item.get("AdapterRAM") or 0) // (1024 * 1024)
        vendor = "NVIDIA" if "nvidia" in name.lower() else "AMD" if "amd" in name.lower() else "Unknown"
        gpus.append(
            GPUInfo(
                index=idx,
                name=name,
                vendor=vendor,
                total_mb=total,
                used_mb=0,
                free_mb=total,
                backend="vulkan" if total else "cpu",
            )
        )
    return gpus


def _linux_secondary_gpus() -> list[GPUInfo]:
    """Detect integrated GPUs when nvidia-smi is unavailable (no VRAM budget)."""
    if platform.system() != "Linux":
        return []
    raw = _run(["lspci"], timeout=2)
    gpus = []
    for line in raw.splitlines():
        lower = line.lower()
        if "vga compatible controller" not in lower and "3d controller" not in lower:
            continue
        name = line.split(":", 2)[-1].strip()
        vendor = (
            "NVIDIA"
            if "nvidia" in lower
            else "Intel"
            if "intel" in lower
            else "AMD"
            if "amd" in lower
            else "Unknown"
        )
        gpus.append(
            GPUInfo(
                index=len(gpus),
                name=name,
                vendor=vendor,
                total_mb=0,
                used_mb=0,
                free_mb=0,
                backend="cpu",
            )
        )
    return gpus


def detect_hardware() -> HardwareInfo:
    warnings: list[str] = []
    system = platform.system()
    if system == "Linux":
        try:
            os_version = (
                Path("/etc/os-release")
                .read_text()
                .split("PRETTY_NAME=", 1)[1]
                .splitlines()[0]
                .strip('"')
            )
        except (OSError, IndexError):
            os_version = platform.release()
    else:
        os_version = f"{platform.release()}"

    gpus, gpu_warn = _nvidia_gpus()
    if gpu_warn:
        warnings.append(gpu_warn)
    if not gpus:
        gpus = _mac_gpus()
    if not gpus:
        gpus = _windows_gpus()
    if not gpus:
        gpus = _linux_secondary_gpus()

    cuda = any(g.cuda for g in gpus)
    apple = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    if cuda:
        backend = "cuda"
    elif apple:
        backend = "metal"
    else:
        backend = "cpu"
    if not cuda and not apple:
        warnings.append("No CUDA/Metal GPU detected; inference will use the CPU backend.")

    mem = psutil.virtual_memory()
    disk = shutil.disk_usage(str(Path.home()))
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    logical = psutil.cpu_count(logical=True) or physical
    info = HardwareInfo(
        os_name=system,
        os_version=os_version,
        arch=platform.machine(),
        python_version=platform.python_version(),
        cpu_model=_cpu_model(),
        physical_cores=physical,
        logical_cores=logical,
        total_ram_bytes=mem.total,
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        gpus=gpus,
        cuda_available=cuda,
        apple_silicon=apple,
        llama_backend=backend,
        warnings=warnings,
    )
    return info


class SystemMonitor:
    def __init__(self, hardware: HardwareInfo | None = None):
        self.hardware = hardware or detect_hardware()
        self._last_cpu = 0.0
        # Prime psutil's CPU counter.
        psutil.cpu_percent(interval=None)

    def snapshot(self) -> MemorySnapshot:
        mem = psutil.virtual_memory()
        try:
            disk = shutil.disk_usage(str(Path.home()))
            disk_free = disk.free
        except OSError:
            disk_free = 0
        gpu_free = sum(g.free_bytes for g in self.hardware.gpus)
        gpu_total = sum(g.total_bytes for g in self.hardware.gpus)
        gpu_used = sum((g.used_mb or 0) * 1024 * 1024 for g in self.hardware.gpus)
        gpu_util = None
        utils = [g.utilization for g in self.hardware.gpus if g.utilization is not None]
        if utils:
            gpu_util = round(sum(utils) / len(utils), 1)
        cpu = psutil.cpu_percent(interval=None)
        self._last_cpu = cpu
        try:
            load = os.getloadavg()[0]
        except (AttributeError, OSError):
            load = 0.0
        return MemorySnapshot(
            total_ram_bytes=mem.total,
            available_ram_bytes=mem.available,
            used_ram_bytes=mem.used,
            ram_percent=mem.percent,
            swap_used_bytes=getattr(psutil.swap_memory(), "used", 0),
            disk_free_bytes=disk_free,
            gpu_free_bytes=gpu_free,
            gpu_total_bytes=gpu_total,
            gpu_used_bytes=gpu_used,
            gpu_utilization=gpu_util,
            cpu_percent=cpu,
            load_1m=load,
        )

    def available_memory_for_models(self, safety_factor: float, reserve_bytes: int) -> dict[str, int]:
        snap = self.snapshot()
        usable_ram = max(
            0,
            int(snap.available_ram_bytes * safety_factor) - reserve_bytes,
        )
        usable_vram = int(snap.gpu_free_bytes * safety_factor)
        return {
            "ram_bytes": usable_ram,
            "vram_bytes": usable_vram,
            "available_ram_bytes": snap.available_ram_bytes,
            "available_vram_bytes": snap.gpu_free_bytes,
        }

    def recommend(self) -> dict[str, Any]:
        hw = self.hardware
        threads = max(2, hw.physical_cores)
        if hw.cuda_available:
            vram = max((g.free_bytes for g in hw.gpus), default=0)
            # Reserve 1 GiB for display/context; offload as much as possible.
            usable = max(0, vram - 1024 * 1024 * 1024)
            context = 8192 if vram >= 8 * 1024**3 else 4096
            max_resident = 2 if vram >= 12 * 1024**3 else 1
            ngl = 999
            tier = "gpu-large" if vram >= 16 * 1024**3 else "gpu"
        elif hw.apple_silicon:
            context = 8192
            max_resident = 1
            ngl = 999
            tier = "apple-metal"
        else:
            usable = 0
            available_gb = self.snapshot().available_ram_bytes / 1024**3
            context = 4096 if available_gb < 8 else 8192
            max_resident = 1 if available_gb < 12 else 2
            ngl = 0
            tier = "cpu-limited" if available_gb < 8 else "cpu"
        return {
            "tier": tier,
            "threads": threads,
            "context_length": context,
            "max_resident_models": max_resident,
            "n_gpu_layers": ngl,
            "usable_gpu_bytes": usable,
            "recommended_model_size": (
                "up to ~7B Q4" if tier.startswith("gpu") else "up to ~3-4B Q4"
            ),
            "notes": (
                "No accelerator budget is assumed; the scheduler will keep a single "
                "small model resident and evict it when idle."
                if tier.startswith("cpu")
                else "Partial or full GPU offload is available."
            ),
        }

    def plan_offload(self, meta: GGUFMetadata, requested_ngl: int | None) -> int:
        if not self.hardware.cuda_available and not self.hardware.apple_silicon:
            return 0
        if requested_ngl is not None and requested_ngl >= 0:
            return requested_ngl
        if self.hardware.apple_silicon:
            return 999
        vram = max((g.free_bytes for g in self.hardware.gpus), default=0)
        usable = max(0, int(vram * 0.85) - 512 * 1024 * 1024)
        if not meta.block_count:
            return 999 if usable >= meta.file_size else 0
        per_layer = meta.file_size / max(1, meta.block_count)
        layers = int(usable // per_layer) if per_layer else 0
        return max(0, min(meta.block_count, layers))

    def system_metrics(self) -> dict[str, Any]:
        snap = self.snapshot()
        process = psutil.Process()
        with process.oneshot():
            rss = process.memory_info().rss
            threads = process.num_threads()
        return {
            "hardware": self.hardware.to_dict(),
            "resources": snap.to_dict(),
            "process": {
                "pid": process.pid,
                "rss_bytes": rss,
                "threads": threads,
                "uptime_s": round(time.time() - process.create_time(), 1),
            },
            "recommendation": self.recommend(),
        }
