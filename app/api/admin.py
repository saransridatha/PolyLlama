"""Administrative control-plane listener and dashboard host.

This app is bound to a separate loopback port from the public inference API. The
Cloudflare tunnel targets only the public port, so these routes cannot be reached
through the public hostname.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ..api.deps import require_admin
from ..api.errors import APIError, install_exception_handlers
from ..api.schemas import (
    KeyCreate,
    KeyUpdate,
    ModelCreate,
    ModelUpdate,
    RouterConfigUpdate,
    TunnelStart,
)
from ..logging_setup import request_id_var
from ..middleware import BodyLimitMiddleware, RequestContextMiddleware
from ..runtime import AppRuntime
from ..security import AuthContext

CAPABILITIES = [
    "chat",
    "general",
    "reasoning",
    "coding",
    "code_generation",
    "debugging",
    "math",
    "summarization",
    "extraction",
    "translation",
    "creative",
    "json",
    "long_context",
    "classification",
    "vision",
    "embeddings",
    "instruct",
    "fast",
    "tool_use",
]


def create_admin_app(runtime: AppRuntime) -> FastAPI:
    app = FastAPI(
        title="ModelSwitch Admin API",
        version=runtime.settings.version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.runtime = runtime
    install_exception_handlers(app)
    app.add_middleware(BodyLimitMiddleware, max_bytes=runtime.settings.max_request_bytes)
    app.add_middleware(
        RequestContextMiddleware,
        security_headers=True,
        csp=(
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        ),
    )

    # ------------------------------------------------------------------
    # Auth / overview
    # ------------------------------------------------------------------
    @app.get("/admin/api/auth/me")
    async def auth_me(ctx: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        return {
            "key": {
                "id": ctx.key_id,
                "name": ctx.name,
                "role": ctx.role,
                "rate_limit_rpm": ctx.rate_limit_rpm,
                "rate_limit_tpm": ctx.rate_limit_tpm,
                "permissions": ctx.permissions,
                "prefix": ctx.prefix,
            },
            "app": {"name": runtime.settings.app_name, "version": runtime.settings.version},
        }

    @app.get("/admin/api/overview")
    async def overview(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        data = runtime.overview()
        data["readiness"] = runtime.readiness()
        data["metrics"] = {
            "requests_started_total": _metric_total(runtime, "modelswitch_requests_started_total"),
        }
        data["remote"]["health"] = {
            "healthy": runtime.remote._healthy,
            "checked_at": runtime.remote._last_check,
        }
        return data

    @app.get("/admin/api/system")
    async def system(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        return runtime.monitor.system_metrics()

    @app.get("/admin/api/readiness")
    async def readiness(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        return runtime.readiness()

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    @app.get("/admin/api/models")
    async def list_models(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        models = []
        for model in runtime.registry.list():
            card = runtime.registry.admin_card(model)
            card["runtime"] = runtime.manager.runtime_card(model["id"])
            card["restart_required"] = runtime.manager.config_changed(model["id"])
            card["performance"] = runtime.db.get_perf(model["id"])
            try:
                card["estimate"] = runtime.registry.estimate_memory(
                    model,
                    context_length=int(model["context_length"]),
                    slots=int(model.get("max_concurrency") or 1),
                    gpu_layers=0,
                )
            except Exception:
                card["estimate"] = {}
            models.append(card)
        return {
            "models": models,
            "capabilities": CAPABILITIES,
            "limits": {
                "max_resident_models": runtime.settings.max_resident_models,
                "max_context_length": runtime.settings.max_context_length,
                "default_context_length": runtime.settings.default_context_length,
                "model_roots": [str(p) for p in runtime.settings.model_root_paths],
                "allow_external_model_paths": runtime.settings.allow_external_model_paths,
                "llama_binary": str(runtime.llama_binary) if runtime.llama_binary else None,
            },
        }

    @app.get("/admin/api/models/discover")
    async def discover(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        return {
            "models": runtime.registry.scan(),
            "roots": [str(p) for p in runtime.settings.model_root_paths],
        }

    @app.post("/admin/api/models", status_code=201)
    async def add_model(
        payload: ModelCreate, _: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        model = runtime.registry.add_model(**payload.model_dump())
        card = runtime.registry.admin_card(model)
        card["runtime"] = runtime.manager.runtime_card(model["id"])
        return card

    @app.get("/admin/api/models/{model_id}")
    async def get_model(model_id: str, _: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        model = runtime.registry.get(model_id)
        if not model:
            raise APIError("Model not found", status_code=404, code="model_not_found")
        card = runtime.registry.admin_card(model)
        card["runtime"] = runtime.manager.runtime_card(model["id"])
        card["restart_required"] = runtime.manager.config_changed(model["id"])
        card["performance"] = runtime.db.get_perf(model["id"])
        return card

    @app.patch("/admin/api/models/{model_id}")
    async def update_model(
        model_id: str, payload: ModelUpdate, _: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        model = runtime.registry.update_model(model_id, payload.model_dump(exclude_none=True))
        card = runtime.registry.admin_card(model)
        card["runtime"] = runtime.manager.runtime_card(model["id"])
        card["restart_required"] = runtime.manager.config_changed(model["id"])
        return card

    @app.delete("/admin/api/models/{model_id}")
    async def delete_model(model_id: str, _: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        if runtime.manager.is_loaded(model_id):
            raise APIError(
                "Unload the model before removing its registration.",
                status_code=409,
                code="model_loaded",
            )
        runtime.registry.delete_model(model_id)
        runtime.manager.runtimes.pop(model_id, None)
        return {"deleted": True, "model_id": model_id, "file_preserved": True}

    @app.post("/admin/api/models/{model_id}/load")
    async def load_model(model_id: str, _: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        started = time.perf_counter()
        await runtime.manager.load(model_id)
        return {
            "loaded": True,
            "model_id": model_id,
            "load_ms": round((time.perf_counter() - started) * 1000, 1),
            "runtime": runtime.manager.runtime_card(model_id),
        }

    @app.post("/admin/api/models/{model_id}/unload")
    async def unload_model(model_id: str, _: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        unloaded = await runtime.manager.unload(model_id)
        return {"unloaded": unloaded, "model_id": model_id}

    @app.post("/admin/api/models/{model_id}/reload")
    async def reload_model(model_id: str, _: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        started = time.perf_counter()
        await runtime.manager.reload(model_id)
        return {
            "reloaded": True,
            "model_id": model_id,
            "load_ms": round((time.perf_counter() - started) * 1000, 1),
            "runtime": runtime.manager.runtime_card(model_id),
        }

    @app.get("/admin/api/models/{model_id}/log")
    async def model_log(
        model_id: str, lines: int = Query(default=120, ge=1, le=2000),
        _: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        rt = runtime.manager.runtimes.get(model_id)
        tail = rt.backend.read_log_tail(lines) if rt and rt.backend else ""
        return {"model_id": model_id, "log": tail, "lines": len(tail.splitlines())}

    @app.get("/admin/api/models/{model_id}/estimate")
    async def model_estimate(
        model_id: str,
        context_length: int | None = Query(default=None, ge=256, le=1_048_576),
        slots: int | None = Query(default=None, ge=1, le=16),
        _: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        model = runtime.registry.get(model_id)
        if not model:
            raise APIError("Model not found", status_code=404, code="model_not_found")
        return runtime.registry.estimate_memory(
            model,
            context_length=context_length or int(model["context_length"]),
            slots=slots or int(model.get("max_concurrency") or 1),
            gpu_layers=0,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    @app.get("/admin/api/routing")
    async def routing(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        models = []
        for model in runtime.registry.list():
            models.append(
                {
                    "id": model["id"],
                    "name": model["name"],
                    "capabilities": model.get("capabilities", []),
                    "parameter_size": model.get("parameter_size"),
                    "quantization": model.get("quantization"),
                    "priority": model.get("priority"),
                    "enabled": model["enabled"],
                    "loaded": runtime.manager.is_loaded(model["id"]),
                    "performance": runtime.db.get_perf(model["id"]),
                }
            )
        return {
            "config": runtime.router.config.to_dict(),
            "aliases": runtime.router.aliases(),
            "models": models,
            "history": runtime.db.list_routing_history(100),
            "perf": runtime.db.model_stats(),
            "strategies": ["balanced", "quality", "latency", "local"],
        }

    @app.patch("/admin/api/routing/config")
    async def update_routing(
        payload: RouterConfigUpdate, _: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            config = runtime.router.update_config(payload.model_dump(exclude_none=True))
        except ValueError as exc:
            raise APIError(str(exc), status_code=400, code="invalid_router_config") from exc
        return {"config": config.to_dict(), "aliases": runtime.router.aliases()}

    @app.get("/admin/api/routing/history")
    async def routing_history(
        limit: int = Query(default=100, ge=1, le=1000),
        _: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        return {"history": runtime.db.list_routing_history(limit)}

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------
    @app.get("/admin/api/keys")
    async def list_keys(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        keys = [_sanitize_key(k) for k in runtime.db.list_api_keys()]
        return {
            "keys": keys,
            "usage": runtime.db.usage_by_key(),
            "limits": {
                "default_rpm": runtime.settings.default_rate_limit_rpm,
                "default_tpm": runtime.settings.default_rate_limit_tpm,
                "admin_rpm": runtime.settings.admin_rate_limit_rpm,
            },
        }

    @app.post("/admin/api/keys", status_code=201)
    async def create_key(
        payload: KeyCreate, _: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        permissions = _validate_permissions(payload.permissions)
        _validate_expiry(payload.expires_at)
        rpm = payload.rate_limit_rpm or (
            runtime.settings.admin_rate_limit_rpm
            if payload.role == "admin"
            else runtime.settings.default_rate_limit_rpm
        )
        tpm = payload.rate_limit_tpm or (
            1_000_000 if payload.role == "admin" else runtime.settings.default_rate_limit_tpm
        )
        record, raw = runtime.keys.create_key(
            name=payload.name,
            role=payload.role,
            permissions=permissions,
            rate_limit_rpm=rpm,
            rate_limit_tpm=tpm,
            expires_at=payload.expires_at,
        )
        runtime.db.add_event(
            "warning",
            "key.created",
            f"Created {payload.role} API key '{payload.name}'",
            metadata={"key_id": record["id"], "role": payload.role},
        )
        return {
            "key": _sanitize_key(record),
            "api_key": raw,
            "warning": "This API key is shown once. Store it in a secret manager now.",
        }

    @app.patch("/admin/api/keys/{key_id}")
    async def update_key(
        key_id: str, payload: KeyUpdate, _: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        record = runtime.db.get_api_key(key_id)
        if not record:
            raise APIError("API key not found", status_code=404, code="key_not_found")
        updates = payload.model_dump(exclude_none=True)
        if "permissions" in updates:
            updates["permissions"] = _validate_permissions(updates["permissions"])
        if "expires_at" in updates:
            _validate_expiry(updates["expires_at"])
        if not updates:
            raise APIError("No fields to update", status_code=400)
        runtime.db.update_api_key(key_id, updates)
        updated = runtime.db.get_api_key(key_id)
        return {"key": _sanitize_key(updated)}

    @app.post("/admin/api/keys/{key_id}/rotate")
    async def rotate_key(key_id: str, _: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        try:
            record, raw = runtime.keys.rotate(key_id)
        except KeyError as exc:
            raise APIError("API key not found", status_code=404, code="key_not_found") from exc
        except ValueError as exc:
            raise APIError(str(exc), status_code=409) from exc
        runtime.db.add_event(
            "warning",
            "key.rotated",
            f"Rotated API key {key_id}",
            metadata={"old_key_id": key_id, "new_key_id": record["id"]},
        )
        return {
            "key": _sanitize_key(record),
            "api_key": raw,
            "warning": "The previous key was revoked. This replacement is shown once.",
        }

    @app.delete("/admin/api/keys/{key_id}")
    async def revoke_key(
        key_id: str, ctx: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        record = runtime.db.get_api_key(key_id)
        if not record:
            raise APIError("API key not found", status_code=404, code="key_not_found")
        if record.get("revoked_at"):
            return {"revoked": False, "already_revoked": True}
        if record["role"] == "admin" and runtime.db.count_admin_keys() <= 1:
            raise APIError(
                "Refusing to revoke the last active administrator key.",
                status_code=409,
                code="last_admin_key",
            )
        runtime.db.revoke_api_key(key_id)
        runtime.db.add_event(
            "warning",
            "key.revoked",
            f"Revoked API key '{record['name']}'",
            metadata={"key_id": key_id, "revoked_by": ctx.key_id},
        )
        return {"revoked": True, "key_id": key_id}

    @app.get("/admin/api/keys/usage")
    async def key_usage(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        return {"usage": runtime.db.usage_by_key()}

    # ------------------------------------------------------------------
    # Remote access
    # ------------------------------------------------------------------
    @app.get("/admin/api/remote")
    async def remote_status(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        return await runtime.remote.status()

    @app.post("/admin/api/remote/start")
    async def remote_start(
        payload: TunnelStart, _: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            status = await runtime.remote.start(
                mode=payload.mode,
                token=payload.token,
                public_url=payload.public_url,
                tunnel_name=payload.tunnel_name,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", 400)
            raise APIError(
                getattr(exc, "message", str(exc)),
                status_code=status_code,
                error_type="remote_access_error",
                code="tunnel_start_failed",
            ) from exc
        runtime.db.add_event(
            "info", "tunnel.started", f"Cloudflare tunnel started in {payload.mode} mode"
        )
        return status

    @app.post("/admin/api/remote/stop")
    async def remote_stop(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        await runtime.remote.stop()
        runtime.db.add_event("info", "tunnel.stopped", "Cloudflare tunnel stopped")
        return await runtime.remote.status()

    @app.post("/admin/api/remote/restart")
    async def remote_restart(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        try:
            status = await runtime.remote.restart()
        except Exception as exc:
            raise APIError(
                getattr(exc, "message", str(exc)),
                status_code=getattr(exc, "status_code", 409),
                code="tunnel_restart_failed",
            ) from exc
        return status

    @app.post("/admin/api/remote/check")
    async def remote_check(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        health = await runtime.remote.check_health()
        return {"health": health, "status": await runtime.remote.status()}

    # ------------------------------------------------------------------
    # Logs / diagnostics / metrics
    # ------------------------------------------------------------------
    @app.get("/admin/api/logs/requests")
    async def requests_log(
        limit: int = Query(default=100, ge=1, le=1000),
        status: str | None = Query(default=None),
        _: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        return {"requests": runtime.db.list_requests(limit, status)}

    @app.get("/admin/api/logs/requests/{request_id}")
    async def request_detail(
        request_id: str, _: AuthContext = Depends(require_admin)
    ) -> dict[str, Any]:
        row = runtime.db.get_request(request_id)
        if not row:
            raise APIError("Request not found", status_code=404, code="request_not_found")
        row["routing_history"] = runtime.db.get_routing_for_request(request_id)
        return row

    @app.get("/admin/api/logs/events")
    async def events_log(
        limit: int = Query(default=100, ge=1, le=1000),
        level: str | None = Query(default=None),
        _: AuthContext = Depends(require_admin),
    ) -> dict[str, Any]:
        return {"events": runtime.db.list_events(limit, level)}

    @app.get("/admin/api/diagnostics")
    async def diagnostics(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        health = await runtime.manager.health()
        return {
            "readiness": runtime.readiness(),
            "models": health,
            "llama_binary": str(runtime.llama_binary) if runtime.llama_binary else None,
            "llama_version": runtime.llama_version,
            "settings": {
                "public_base_url": runtime.settings.public_base_url,
                "admin_base_url": runtime.settings.admin_base_url,
                "max_resident_models": runtime.settings.max_resident_models,
                "max_request_bytes": runtime.settings.max_request_bytes,
                "queue_timeout": runtime.settings.queue_timeout,
                "request_timeout": runtime.settings.request_timeout,
                "idle_timeout": runtime.settings.backend_idle_timeout,
                "memory_safety_factor": runtime.settings.memory_safety_factor,
            },
            "counts": {
                "requests": runtime.db.stats_summary(),
                "models": len(runtime.registry.list()),
                "keys": runtime.db.count_api_keys(),
            },
        }

    @app.get("/metrics")
    async def metrics(_: AuthContext = Depends(require_admin)) -> Response:
        system = runtime.monitor.snapshot()
        extra: dict[str, Any] = {
            "modelswitch_cpu_percent": system.cpu_percent,
            "modelswitch_ram_used_bytes": system.used_ram_bytes,
            "modelswitch_ram_available_bytes": system.available_ram_bytes,
            "modelswitch_gpu_free_bytes": system.gpu_free_bytes,
            "modelswitch_gpu_used_bytes": system.gpu_used_bytes,
            "modelswitch_models_loaded": len(runtime.manager.loaded_models()),
            "modelswitch_resident_ram_estimate_bytes": runtime.manager.resident_estimate()["ram_bytes"],
            "modelswitch_resident_vram_estimate_bytes": runtime.manager.resident_estimate()["vram_bytes"],
        }
        for model_id, rt in runtime.manager.runtimes.items():
            runtime.metrics.set("modelswitch_model_loaded", 1 if rt.is_loaded() else 0, {"model": model_id})
            runtime.metrics.set(
                "modelswitch_model_active_requests", rt.active, {"model": model_id}
            )
            runtime.metrics.set(
                "modelswitch_model_queued_requests", rt.waiting, {"model": model_id}
            )
        return Response(
            runtime.metrics.render(extra),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/admin/api/info")
    async def info(_: AuthContext = Depends(require_admin)) -> dict[str, Any]:
        return {
            "version": runtime.settings.version,
            "public_api_base": runtime.settings.public_url("/v1"),
            "admin_url": runtime.settings.admin_url("/admin"),
            "dashboard_built": (runtime.settings.dashboard_dir / "index.html").is_file(),
            "docs": {
                "readme": "README.md",
                "architecture": "docs/ARCHITECTURE.md",
                "api": "docs/API.md",
                "operations": "docs/OPERATIONS.md",
            },
            "capabilities": CAPABILITIES,
        }

    # ------------------------------------------------------------------
    # Dashboard SPA
    # ------------------------------------------------------------------
    dashboard = runtime.settings.dashboard_dir
    assets = dashboard / "assets"
    if assets.is_dir():
        app.mount("/admin/assets", StaticFiles(directory=str(assets)), name="admin-assets")
    if (dashboard / "favicon.svg").is_file():
        @app.get("/admin/favicon.svg", include_in_schema=False)
        async def favicon() -> FileResponse:
            return FileResponse(dashboard / "favicon.svg", media_type="image/svg+xml")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/admin")

    @app.get("/admin", include_in_schema=False)
    @app.get("/admin/{path:path}", include_in_schema=False)
    async def dashboard_spa(path: str = "") -> Response:
        if path.startswith("api/"):
            raise APIError("Not found", status_code=404, code="not_found")
        index = dashboard / "index.html"
        if index.is_file():
            return FileResponse(
                index,
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )
        return HTMLResponse(
            _missing_dashboard_html(), status_code=503
        )

    return app


def _sanitize_key(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    return {
        "id": record["id"],
        "name": record["name"],
        "role": record["role"],
        "prefix": record["prefix"],
        "permissions": record.get("permissions") or {},
        "rate_limit_rpm": record["rate_limit_rpm"],
        "rate_limit_tpm": record["rate_limit_tpm"],
        "created_at": record["created_at"],
        "last_used_at": record.get("last_used_at"),
        "expires_at": record.get("expires_at"),
        "revoked_at": record.get("revoked_at"),
        "revoked": bool(record.get("revoked_at")),
    }


def _validate_permissions(permissions: dict[str, Any] | None) -> dict[str, Any]:
    permissions = dict(permissions or {})
    allowed = {"models", "allow_streaming", "allow_embeddings", "max_tokens", "priority"}
    unknown = set(permissions) - allowed
    if unknown:
        raise APIError(
            f"Unknown permission(s): {', '.join(sorted(unknown))}",
            status_code=400,
            code="invalid_permissions",
        )
    models = permissions.get("models")
    if models is not None:
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise APIError("permissions.models must be a list of model ids")
        if len(models) > 100:
            raise APIError("permissions.models is too large")
        permissions["models"] = [m[:128] for m in models]
    if "allow_streaming" in permissions and not isinstance(permissions["allow_streaming"], bool):
        raise APIError("permissions.allow_streaming must be boolean")
    if "allow_embeddings" in permissions and not isinstance(permissions["allow_embeddings"], bool):
        raise APIError("permissions.allow_embeddings must be boolean")
    if "max_tokens" in permissions:
        try:
            value = int(permissions["max_tokens"])
        except (TypeError, ValueError) as exc:
            raise APIError("permissions.max_tokens must be an integer") from exc
        permissions["max_tokens"] = max(1, min(131072, value))
    return permissions


def _validate_expiry(value: str | None) -> None:
    if not value:
        return
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise APIError("expires_at must be an ISO-8601 timestamp", status_code=400) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed <= datetime.now(timezone.utc):
        raise APIError("expires_at must be in the future", status_code=400)


def _metric_total(runtime: AppRuntime, name: str) -> int:
    total = 0
    for (metric_name, _), value in runtime.metrics.counters.items():
        if metric_name == name:
            total += int(value)
    return total


def _missing_dashboard_html() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>ModelSwitch dashboard not built</title>
<style>body{font:15px/1.6 ui-sans-serif,system-ui;background:#101312;color:#e8eee9;
max-width:760px;margin:8vh auto;padding:32px}code{background:#1d2420;padding:2px 6px;border-radius:4px}
pre{background:#1d2420;padding:16px;border-radius:8px;overflow:auto;border:1px solid #2c3630}</style>
</head><body><h1>Dashboard assets are not built</h1>
<p>The gateway is running, but the React dashboard has not been compiled. Run:</p>
<pre>cd frontend
npm install
npm run build</pre>
<p>Then reload this page. The inference API is available independently at
<code>/v1</code> on the public listener.</p></body></html>"""
