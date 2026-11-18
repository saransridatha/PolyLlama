"""Shared application runtime wiring."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .backend.llama import find_llama_server, llama_version
from .config import Settings
from .db import Database
from .hardware import HardwareInfo, SystemMonitor, detect_hardware
from .lifecycle import ModelManager
from .metrics import Metrics, metrics as global_metrics
from .registry import ModelRegistry
from .remote import CloudflareTunnel
from .router.router import IntelligentRouter
from .security import KeyService, RateLimiter

log = logging.getLogger("modelswitch.runtime")


class AppRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_dirs()
        self.started_at = time.time()
        self.db = Database(settings.db_path)
        self.keys = KeyService(self.db, settings.key_pepper)
        self.registry = ModelRegistry(self.db, settings)
        self.hardware: HardwareInfo = detect_hardware()
        self.monitor = SystemMonitor(self.hardware)
        self.manager = ModelManager(settings, self.db, self.registry, self.monitor)
        self.router = IntelligentRouter(settings, self.db, self.registry, self.manager)
        self.limiter = RateLimiter()
        self.metrics = global_metrics
        self.http = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
            timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
            headers={"User-Agent": f"ModelSwitch/{settings.version}"},
        )
        self.remote = CloudflareTunnel(settings, target_port=settings.public_port)
        # Imported lazily to avoid a circular import with the API layer.
        from .inference import InferenceService

        self.inference = InferenceService(
            settings=settings,
            db=self.db,
            registry=self.registry,
            manager=self.manager,
            router=self.router,
            limiter=self.limiter,
            metrics=self.metrics,
            http=self.http,
        )
        self._retention_task: asyncio.Task[None] | None = None
        self.llama_binary = find_llama_server(settings)
        self.llama_version = llama_version(self.llama_binary) if self.llama_binary else None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.manager.start()
        try:
            await asyncio.to_thread(self.registry.warm_metadata_cache)
        except Exception:
            log.exception("metadata cache warmup failed")
        self._started = True
        self._retention_task = asyncio.create_task(self._retention_loop(), name="retention-loop")
        self.db.add_event(
            "info",
            "gateway.started",
            f"ModelSwitch {self.settings.version} started",
            metadata={
                "llama_binary": str(self.llama_binary) if self.llama_binary else None,
                "hardware_backend": self.hardware.llama_backend,
            },
        )

    async def _retention_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(3600)
                if self.settings.router_history_retention_days > 0:
                    self.db.purge(self.settings.router_history_retention_days)
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("retention loop failed")

    async def close(self) -> None:
        if self._retention_task:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except (asyncio.CancelledError, Exception):
                pass
            self._retention_task = None
        try:
            await self.remote.close()
        except Exception:
            log.exception("failed to stop tunnel")
        try:
            await self.manager.shutdown()
        except Exception:
            log.exception("failed to stop model backends")
        try:
            await self.http.aclose()
        except Exception:  # nosec B110 - HTTP client cleanup during shutdown
            pass
        try:
            self.db.add_event("info", "gateway.stopped", "ModelSwitch stopped")
        except Exception:  # nosec B110 - DB may already be closing
            pass
        self.db.close()
        self._started = False

    def readiness(self) -> dict[str, Any]:
        models = [m for m in self.registry.list(include_disabled=False)]
        available = [m for m in models if self.registry.availability(m)["available"]]
        return {
            "ready": bool(available) and self.llama_binary is not None,
            "llama_binary": bool(self.llama_binary),
            "llama_version": self.llama_version,
            "registered_models": len(models),
            "available_models": len(available),
            "loaded_models": self.manager.loaded_models(),
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    def overview(self) -> dict[str, Any]:
        stats = self.db.stats_summary()
        resources = self.monitor.snapshot().to_dict()
        models = self.registry.list()
        runtime_cards = []
        for model in models:
            rt = self.manager.runtime_card(model["id"])
            try:
                estimate = self.registry.estimate_memory(
                    model,
                    context_length=int(model["context_length"]),
                    slots=int(model.get("max_concurrency") or 1),
                    gpu_layers=0,
                )
            except Exception:
                estimate = {}
            runtime_cards.append(
                {
                    "id": model["id"],
                    "name": model["name"],
                    "enabled": model["enabled"],
                    "size_bytes": model.get("size_bytes"),
                    "quantization": model.get("quantization"),
                    "parameter_size": model.get("parameter_size"),
                    "context_length": model.get("context_length"),
                    "capabilities": model.get("capabilities", []),
                    "runtime": rt,
                    "estimate": estimate,
                }
            )
        tokens_recent = int(stats.get("tokens_last_5m") or 0)
        loaded = [rt for rt in runtime_cards if rt["runtime"].get("loaded")]
        active = sum(int(rt["runtime"].get("active_requests", 0)) for rt in runtime_cards)
        queued = sum(int(rt["runtime"].get("queued_requests", 0)) for rt in runtime_cards)
        return {
            "app": {
                "name": self.settings.app_name,
                "version": self.settings.version,
                "uptime_s": round(time.time() - self.started_at, 1),
                "started_at": self.started_at,
            },
            "health": {
                "status": "healthy" if self.llama_binary else "degraded",
                "llama_binary": bool(self.llama_binary),
                "llama_version": self.llama_version,
                "hardware_backend": self.hardware.llama_backend,
            },
            "stats": {
                **stats,
                "tokens_per_second_5m": round(tokens_recent / 300, 2),
                "active_requests": active,
                "queued_requests": queued,
                "models_registered": len(models),
                "models_loaded": len(loaded),
                "models_enabled": sum(1 for m in models if m["enabled"]),
                "max_resident_models": self.settings.max_resident_models,
            },
            "resources": resources,
            "hardware": {
                "cpu_model": self.hardware.cpu_model,
                "physical_cores": self.hardware.physical_cores,
                "logical_cores": self.hardware.logical_cores,
                "total_ram_bytes": self.hardware.total_ram_bytes,
                "gpus": [
                    {
                        "name": g.name,
                        "vendor": g.vendor,
                        "total_mb": g.total_mb,
                        "free_mb": g.free_mb,
                        "utilization": g.utilization,
                        "backend": g.backend,
                        "cuda": g.cuda,
                    }
                    for g in self.hardware.gpus
                ],
                "backend": self.hardware.llama_backend,
                "warnings": self.hardware.warnings,
                "os": self.hardware.os_name,
                "os_version": self.hardware.os_version,
                "arch": self.hardware.arch,
            },
            "models": runtime_cards,
            "loaded_models": self.manager.loaded_models(),
            "series": self.db.request_timeseries(30),
            "recent_requests": self.db.list_requests(12),
            "routing": self.router.config.to_dict(),
            "remote": {
                "running": self.remote.running,
                "public_url": self.remote.public_url,
                "api_base_url": (
                    self.remote.public_url.rstrip("/") + "/v1"
                    if self.remote.public_url
                    else None
                ),
                "installed": bool(self.remote.binary),
            },
            "recommendation": self.monitor.recommend(),
        }
