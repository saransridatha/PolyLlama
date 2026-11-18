"""Public inference listener (OpenAI-compatible)."""

from __future__ import annotations

import time

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..api.deps import require_auth
from ..api.errors import install_exception_handlers
from ..api.schemas import ChatCompletionRequest, CompletionRequest, EmbeddingRequest
from ..logging_setup import request_id_var
from ..middleware import BodyLimitMiddleware, RequestContextMiddleware
from ..runtime import AppRuntime
from ..security import AuthContext


def create_public_app(runtime: AppRuntime) -> FastAPI:
    app = FastAPI(
        title="ModelSwitch Inference API",
        version=runtime.settings.version,
        description=(
            "OpenAI-compatible local multi-model inference gateway. Use model "
            "`auto` for intelligent routing, or a registered model / alias."
        ),
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.runtime = runtime
    install_exception_handlers(app)
    app.add_middleware(
        BodyLimitMiddleware, max_bytes=runtime.settings.max_request_bytes
    )
    app.add_middleware(
        RequestContextMiddleware,
        security_headers=True,
        csp=None,
    )
    if runtime.settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=runtime.settings.cors_origin_list,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id"],
            expose_headers=[
                "X-Request-Id",
                "X-ModelSwitch-Model",
                "X-ModelSwitch-Task",
                "X-ModelSwitch-Routing-Ms",
                "X-ModelSwitch-Queue-Ms",
                "X-ModelSwitch-Load-Ms",
                "X-ModelSwitch-Fallback",
            ],
        )

    async def ensure_rate(ctx: AuthContext) -> None:
        await runtime.limiter.check(ctx, 0)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, object]:
        return {
            "service": "ModelSwitch",
            "version": runtime.settings.version,
            "api": "/v1",
            "docs": "/docs",
            "health": "/healthz",
        }

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "modelswitch",
            "version": runtime.settings.version,
            "uptime_s": round(time.time() - runtime.started_at, 1),
        }

    @app.get("/health", include_in_schema=False)
    async def health_alias() -> dict[str, object]:
        return await healthz()

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        status = runtime.readiness()
        return JSONResponse(status, status_code=200 if status["ready"] else 503)

    @app.get("/v1/models")
    async def list_models(ctx: AuthContext = Depends(require_auth)) -> dict[str, object]:
        await ensure_rate(ctx)
        allowed = ctx.allowed_models()
        data: list[dict[str, object]] = []
        now = int(time.time())
        for alias, description in (
            ("auto", "Intelligent task-aware routing"),
            ("fast", "Lowest-latency feasible model"),
            ("coding", "Best feasible coding model"),
            ("reasoning", "Best feasible reasoning model"),
        ):
            data.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": now,
                    "owned_by": "modelswitch",
                    "description": description,
                }
            )
        for model in runtime.registry.list(include_disabled=False):
            if allowed is not None and model["id"] not in allowed:
                continue
            if not runtime.registry.availability(model)["available"]:
                continue
            card = runtime.registry.public_card(model)
            data.append(card)
            for alias in model.get("aliases", []):
                data.append(
                    {
                        "id": alias,
                        "object": "model",
                        "created": now,
                        "owned_by": "modelswitch-local",
                        "root": model["id"],
                    }
                )
        return {"object": "list", "data": data}

    @app.get("/v1/models/{model_id}")
    async def get_model(
        model_id: str, ctx: AuthContext = Depends(require_auth)
    ) -> dict[str, object]:
        await ensure_rate(ctx)
        if model_id in ("auto", "fast", "coding", "reasoning", "default"):
            return {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "modelswitch",
            }
        model = runtime.registry.find(model_id)
        if not model or not model["enabled"] or not runtime.registry.availability(model)["available"]:
            from ..api.errors import APIError

            raise APIError(
                f"Model '{model_id}' not found",
                status_code=404,
                error_type="invalid_request_error",
                code="model_not_found",
            )
        allowed = ctx.allowed_models()
        if allowed is not None and model["id"] not in allowed:
            from ..api.errors import APIError

            raise APIError(
                "API key is not permitted to use this model",
                status_code=403,
                error_type="permission_error",
                code="permission_denied",
            )
        return runtime.registry.public_card(model)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        ctx: AuthContext = Depends(require_auth),
    ):
        request_id = request_id_var.get()
        return await runtime.inference.chat(payload, ctx, request_id=request_id)

    @app.post("/v1/completions")
    async def completions(
        payload: CompletionRequest,
        request: Request,
        ctx: AuthContext = Depends(require_auth),
    ):
        request_id = request_id_var.get()
        return await runtime.inference.completion(payload, ctx, request_id=request_id)

    @app.post("/v1/embeddings")
    async def embeddings(
        payload: EmbeddingRequest,
        request: Request,
        ctx: AuthContext = Depends(require_auth),
    ):
        request_id = request_id_var.get()
        return await runtime.inference.embeddings(payload, ctx, request_id=request_id)

    return app
