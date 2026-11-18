"""Memory-aware model residency and lifecycle management.

Key invariants:

* at most ``max_resident_models`` models are loaded;
* a model is never unloaded while it has an active request or queued caller;
* concurrent loads for the same model are coalesced by a single lifecycle lock;
* admission control uses a conservative memory estimate plus live free memory;
* a crashed backend is isolated and can be reloaded without restarting the gateway.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend.llama import BackendConfig, BackendError, LlamaBackend
from .config import Settings
from .db import Database
from .gguf import GGUFMetadata
from .hardware import SystemMonitor
from .registry import ModelRegistry, RegistryError
from .scheduler import Lease, PriorityGate

log = logging.getLogger("modelswitch.lifecycle")


class ModelUnavailable(Exception):
    def __init__(self, message: str, *, status_code: int = 503, code: str = "model_unavailable"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


@dataclass(slots=True)
class ModelRuntime:
    model_id: str
    state: str = "unloaded"
    backend: LlamaBackend | None = None
    gate: PriorityGate | None = None
    loaded_at: float | None = None
    last_used: float | None = None
    ngl: int = 0
    threads: int = 0
    slots: int = 1
    context_length: int = 4096
    context_total: int = 4096
    estimate: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    log_path: str | None = None
    load_count: int = 0
    crash_count: int = 0
    config_fingerprint: tuple[Any, ...] | None = None

    @property
    def active(self) -> int:
        return self.gate.in_use if self.gate else 0

    @property
    def waiting(self) -> int:
        return self.gate.waiting if self.gate else 0

    def is_loaded(self) -> bool:
        return self.state == "loaded" and self.backend is not None and self.backend.alive


class ModelManager:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        registry: ModelRegistry,
        monitor: SystemMonitor,
    ):
        self.settings = settings
        self.db = db
        self.registry = registry
        self.monitor = monitor
        self.runtimes: dict[str, ModelRuntime] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._idle_task: asyncio.Task[None] | None = None
        self._shutting_down = False
        self.started_at = time.time()
        self.load_failures = 0

    # -- observability ---------------------------------------------------
    def runtime(self, model_id: str) -> ModelRuntime:
        runtime = self.runtimes.get(model_id)
        if not runtime:
            runtime = ModelRuntime(model_id=model_id)
            self.runtimes[model_id] = runtime
        return runtime

    def is_loaded(self, model_id: str) -> bool:
        runtime = self.runtimes.get(model_id)
        return bool(runtime and runtime.is_loaded())

    def loaded_models(self) -> list[str]:
        return [mid for mid, rt in self.runtimes.items() if rt.is_loaded()]

    def resident_estimate(self) -> dict[str, int]:
        ram = sum(int(rt.estimate.get("ram_bytes", 0)) for rt in self.runtimes.values() if rt.is_loaded())
        vram = sum(
            int(rt.estimate.get("vram_bytes", 0)) for rt in self.runtimes.values() if rt.is_loaded()
        )
        return {"ram_bytes": ram, "vram_bytes": vram}

    def runtime_card(self, model_id: str, public: bool = False) -> dict[str, Any]:
        rt = self.runtimes.get(model_id)
        if not rt:
            return {
                "state": "unloaded",
                "loaded": False,
                "active_requests": 0,
                "queued_requests": 0,
            }
        card: dict[str, Any] = {
            "state": rt.state,
            "loaded": rt.is_loaded(),
            "active_requests": rt.active,
            "queued_requests": rt.waiting,
            "loaded_at": rt.loaded_at,
            "last_used": rt.last_used,
            "context_length": rt.context_length,
            "context_total": rt.context_total,
            "slots": rt.slots,
            "gpu_layers": rt.ngl,
            "threads": rt.threads,
            "estimated_memory": rt.estimate,
            "load_count": rt.load_count,
            "crash_count": rt.crash_count,
            "error": rt.error,
        }
        if rt.gate:
            card["queue"] = rt.gate.snapshot()
        if rt.backend:
            card["backend"] = {**rt.backend.process_stats(), "port": rt.backend.port}
        if not public:
            card["log_path"] = rt.log_path
        return card

    # -- resources -------------------------------------------------------
    def _usable_memory(self) -> dict[str, int]:
        return self.monitor.available_memory_for_models(
            self.settings.memory_safety_factor,
            self.settings.min_free_ram_mb * 1024 * 1024,
        )

    async def _make_room(
        self, *, needed_ram: int, needed_vram: int, exclude: str, slots_needed: int = 1
    ) -> None:
        """Evict idle models until the requested residency is feasible.

        Caller holds the lifecycle lock.
        """
        if slots_needed > self.settings.max_resident_models:
            raise ModelUnavailable(
                f"Model concurrency configuration requires {slots_needed} resident slots, "
                f"but MS_MAX_RESIDENT_MODELS={self.settings.max_resident_models}",
                status_code=409,
            )
        while True:
            loaded = [rt for rt in self.runtimes.values() if rt.is_loaded() and rt.model_id != exclude]
            memory = self._usable_memory()
            over_count = len(loaded) + 1 > self.settings.max_resident_models
            over_ram = needed_ram > memory["ram_bytes"]
            over_vram = needed_vram > 0 and needed_vram > memory["vram_bytes"]
            if not (over_count or over_ram or over_vram):
                return
            candidates = [rt for rt in loaded if rt.active == 0 and rt.waiting == 0]
            if not candidates:
                reason = (
                    f"insufficient memory (need {needed_ram / 1024**3:.2f} GiB RAM, "
                    f"available {memory['ram_bytes'] / 1024**3:.2f} GiB)"
                    if over_ram
                    else "resident model limit reached and all loaded models are busy"
                )
                raise ModelUnavailable(reason, status_code=503)
            candidates.sort(
                key=lambda rt: (
                    1 if self._model_residency(rt.model_id) == "keep_loaded" else 0,
                    rt.last_used or 0,
                )
            )
            victim = candidates[0]
            self.db.add_event(
                "info",
                "model.evicted",
                f"Evicting {victim.model_id} to make room for {exclude}",
                victim.model_id,
                {"reason": "memory_or_residency_pressure"},
            )
            await self._unload_locked(victim.model_id, reason="eviction")
            await asyncio.sleep(0.25)

    def _model_residency(self, model_id: str) -> str:
        model = self.registry.get(model_id)
        return model.get("residency", "on_demand") if model else "on_demand"

    # -- load/unload -----------------------------------------------------
    async def load(self, model_id: str) -> ModelRuntime:
        model = self.registry.get(model_id)
        if not model:
            raise ModelUnavailable("Model not found", status_code=404, code="model_not_found")
        if not model["enabled"]:
            raise ModelUnavailable(
                f"Model '{model_id}' is disabled", status_code=409, code="model_disabled"
            )
        if self._shutting_down:
            raise ModelUnavailable("Gateway is shutting down", status_code=503)
        runtime = self.runtime(model_id)
        if runtime.is_loaded():
            return runtime
        async with self._lifecycle_lock:
            runtime = self.runtime(model_id)
            if runtime.is_loaded():
                return runtime
            if runtime.state == "loading":
                # Only possible if state was mutated outside the lock; treat as a
                # transient busy condition rather than spawning a second process.
                raise ModelUnavailable("Model is already loading", status_code=409)
            return await self._load_locked(model, runtime)

    def _fingerprint(self, model: dict[str, Any]) -> tuple[Any, ...]:
        return (
            model["path"],
            model["context_length"],
            model["max_concurrency"],
            model.get("n_gpu_layers"),
            model.get("threads"),
            tuple(model.get("extra_args", [])),
        )

    async def _load_locked(
        self, model: dict[str, Any], runtime: ModelRuntime
    ) -> ModelRuntime:
        availability = self.registry.availability(model)
        if not availability["available"]:
            raise ModelUnavailable(
                f"Model '{model['id']}' is unavailable: {availability['reason']}",
                status_code=503,
                code="model_file_missing",
            )
        try:
            meta = self.registry.metadata(Path(model["path"]))
        except RegistryError as exc:
            raise ModelUnavailable(exc.message, status_code=503, code="model_invalid") from exc
        slots = max(1, int(model.get("max_concurrency") or 1))
        context = min(
            int(model["context_length"]),
            int(model.get("max_context_length") or model["context_length"]),
        )
        context_total = context * slots
        requested_ngl = model.get("n_gpu_layers")
        if requested_ngl is None and self.settings.llama_n_gpu_layers >= 0:
            requested_ngl = self.settings.llama_n_gpu_layers
        ngl = self.monitor.plan_offload(meta, requested_ngl)
        if not self.monitor.hardware.cuda_available and not self.monitor.hardware.apple_silicon:
            ngl = 0
        threads = int(
            model.get("threads")
            or self.settings.llama_threads
            or self.monitor.hardware.physical_cores
            or 4
        )
        estimate = self.registry.estimate_memory(
            model, context_length=context, slots=slots, gpu_layers=ngl
        )
        runtime.state = "loading"
        runtime.error = None
        runtime.loaded_at = None
        runtime.estimate = estimate
        runtime.slots = slots
        runtime.context_length = context
        runtime.context_total = context_total
        runtime.ngl = ngl
        runtime.threads = threads
        runtime.config_fingerprint = self._fingerprint(model)
        try:
            await self._make_room(
                needed_ram=int(estimate["ram_bytes"]),
                needed_vram=int(estimate["vram_bytes"]),
                exclude=model["id"],
            )
            if runtime.backend:
                # A crashed process may still hold file descriptors/client pools.
                try:
                    await runtime.backend.close()
                except Exception:  # nosec B110 - cleanup must not mask the reload
                    pass
                runtime.backend = None
                runtime.gate = None
            backend = LlamaBackend(self.settings)
            runtime.backend = backend
            runtime.log_path = str(backend.log_path or self.settings.log_dir / f"llama-{model['id']}.log")
            config = BackendConfig(
                model_id=model["id"],
                model_path=model["path"],
                context_total=context_total,
                slots=slots,
                gpu_layers=ngl,
                threads=threads,
                alias=model["id"],
                batch_size=self.settings.llama_batch_size,
                ubatch_size=self.settings.llama_ubatch_size,
                embeddings="embeddings" in model.get("capabilities", []),
                extra_args=list(model.get("extra_args") or []),
                flash_attn=self.settings.flash_attn,
            )
            log.info(
                "loading model",
                extra={
                    "event": "model.load_start",
                    "model_id": model["id"],
                    "status": "loading",
                },
            )
            await backend.start(config)
            runtime.state = "loaded"
            runtime.loaded_at = time.time()
            runtime.last_used = time.time()
            runtime.load_count += 1
            runtime.gate = PriorityGate(
                slots,
                max_queue=max(self.settings.max_queue_size, slots),
            )
            runtime.estimate["actual_rss_bytes"] = backend.process_stats().get("rss_bytes", 0)
            self.db.add_event(
                "info",
                "model.loaded",
                f"Loaded {model['id']} ({meta.architecture}, {ngl} GPU layers)",
                model["id"],
                {
                    "context_total": context_total,
                    "slots": slots,
                    "gpu_layers": ngl,
                    "estimate": estimate,
                },
            )
            return runtime
        except (BackendError, ModelUnavailable, OSError, Exception) as exc:
            runtime.state = "failed"
            runtime.error = str(exc)
            runtime.loaded_at = None
            log_tail = exc.log_tail if isinstance(exc, BackendError) else ""
            self.load_failures += 1
            self.db.add_event(
                "error",
                "model.load_failed",
                f"Failed to load {model['id']}: {exc}",
                model["id"],
                {"log_tail": log_tail[-4000:] if log_tail else ""},
            )
            if isinstance(exc, ModelUnavailable):
                backend = runtime.backend
                if backend:
                    await backend.close()
                    runtime.backend = None
                raise
            backend = runtime.backend
            if backend:
                await backend.close()
                runtime.backend = None
            message = f"Model '{model['id']}' failed to load"
            if log_tail:
                message += f": {_summarize_backend_error(log_tail)}"
            raise ModelUnavailable(message, status_code=503, code="model_load_failed") from exc

    async def unload(self, model_id: str, *, force: bool = False) -> bool:
        async with self._lifecycle_lock:
            return await self._unload_locked(model_id, force=force)

    async def _unload_locked(
        self, model_id: str, *, force: bool = False, reason: str = "manual"
    ) -> bool:
        runtime = self.runtimes.get(model_id)
        if not runtime or (runtime.state == "unloaded" and not runtime.backend):
            return False
        if not force and (runtime.active > 0 or runtime.waiting > 0):
            raise ModelUnavailable(
                f"Model '{model_id}' is serving {runtime.active} request(s) with "
                f"{runtime.waiting} queued; refusing to unload",
                status_code=409,
                code="model_busy",
            )
        runtime.state = "unloading"
        backend = runtime.backend
        if backend:
            await backend.close()
        runtime.backend = None
        runtime.gate = None
        runtime.state = "unloaded"
        runtime.loaded_at = None
        runtime.estimate = {}
        self.db.add_event(
            "info", "model.unloaded", f"Unloaded {model_id} ({reason})", model_id
        )
        return True

    async def reload(self, model_id: str) -> ModelRuntime:
        await self.unload(model_id, force=True)
        return await self.load(model_id)

    async def acquire(
        self,
        model_id: str,
        *,
        priority: int = 5,
        key_id: str = "-",
        queue_timeout: float | None = None,
    ) -> tuple[ModelRuntime, Lease]:
        runtime = await self.load(model_id)
        if not runtime.gate:
            raise ModelUnavailable("Model gate is unavailable", status_code=503)
        timeout = queue_timeout if queue_timeout is not None else self.settings.queue_timeout
        try:
            lease = await runtime.gate.acquire(
                priority=priority, key_id=key_id, timeout=timeout
            )
        except Exception as exc:
            from .scheduler import QueueFullError, QueueTimeoutError

            if isinstance(exc, QueueFullError):
                raise ModelUnavailable(
                    "Server is at capacity; retry shortly",
                    status_code=503,
                    code="queue_full",
                ) from exc
            if isinstance(exc, QueueTimeoutError):
                raise ModelUnavailable(
                    "Timed out waiting for model capacity",
                    status_code=503,
                    code="queue_timeout",
                ) from exc
            raise
        runtime.last_used = time.time()
        return runtime, lease

    def config_changed(self, model_id: str) -> bool:
        model = self.registry.get(model_id)
        runtime = self.runtimes.get(model_id)
        if not model or not runtime or not runtime.is_loaded():
            return False
        return runtime.config_fingerprint != self._fingerprint(model)

    # -- background maintenance -----------------------------------------
    async def start(self) -> None:
        if not self._idle_task:
            self._idle_task = asyncio.create_task(self._idle_loop(), name="model-idle-loop")

    async def _idle_loop(self) -> None:
        while not self._shutting_down:
            try:
                await asyncio.sleep(self.settings.idle_check_interval)
                if self._shutting_down:
                    return
                await self._maintenance_once()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # maintenance must never kill the gateway
                log.exception("idle maintenance failed: %s", exc)

    async def _maintenance_once(self) -> None:
        now = time.time()
        async with self._lifecycle_lock:
            for runtime in list(self.runtimes.values()):
                if runtime.state == "loaded" and runtime.backend and not runtime.backend.alive:
                    runtime.state = "crashed"
                    runtime.crash_count += 1
                    runtime.error = "backend process exited unexpectedly"
                    self.db.add_event(
                        "error",
                        "model.crashed",
                        f"Backend for {runtime.model_id} exited unexpectedly",
                        runtime.model_id,
                        {"returncode": runtime.backend.process.returncode if runtime.backend.process else None},
                    )
                    await runtime.backend.close()
                    runtime.backend = None
                    runtime.gate = None
                    continue
            loaded = [rt for rt in self.runtimes.values() if rt.is_loaded()]
            loaded.sort(key=lambda rt: rt.last_used or 0)
            for runtime in loaded:
                if runtime.active or runtime.waiting:
                    continue
                idle_for = now - (runtime.last_used or runtime.loaded_at or now)
                residency = self._model_residency(runtime.model_id)
                if residency == "keep_loaded":
                    continue
                if idle_for >= self.settings.backend_idle_timeout:
                    await self._unload_locked(runtime.model_id, reason="idle_timeout")
            # Enforce residency limits if configuration was lowered at runtime.
            loaded = [rt for rt in self.runtimes.values() if rt.is_loaded()]
            while len(loaded) > self.settings.max_resident_models:
                idle = [rt for rt in loaded if rt.active == 0 and rt.waiting == 0]
                if not idle:
                    break
                idle.sort(key=lambda rt: rt.last_used or 0)
                await self._unload_locked(idle[0].model_id, reason="resident_limit")
                loaded = [rt for rt in self.runtimes.values() if rt.is_loaded()]

    async def health(self) -> list[dict[str, Any]]:
        results = []
        for runtime in list(self.runtimes.values()):
            if runtime.state == "loaded" and runtime.backend:
                health = await runtime.backend.health()
                results.append({"model_id": runtime.model_id, **health})
        return results

    async def shutdown(self) -> None:
        self._shutting_down = True
        if self._idle_task:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):
                pass
            self._idle_task = None
        async with self._lifecycle_lock:
            for model_id in list(self.runtimes):
                try:
                    await self._unload_locked(model_id, force=True, reason="shutdown")
                except Exception:  # nosec B110 - continue shutting down remaining models
                    pass


def _summarize_backend_error(log_tail: str) -> str:
    lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if any(
            marker in line.lower()
            for marker in ("error", "failed", "unknown model", "unsupported", "out of memory", "oom")
        )
    ]
    payload = interesting[-3:] if interesting else lines[-2:]
    import re

    text = " | ".join(payload)
    # Backend errors can contain absolute model paths; keep those in the admin
    # log only and return a path-free message to API callers.
    text = re.sub(r"(?<![\w])/(?:[^\s'\"]+/?)+", "<path>", text)
    return text[:500]
