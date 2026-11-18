"""Uniform OpenAI-style error handling and auth dependencies."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..inference import UpstreamError
from ..lifecycle import ModelUnavailable
from ..logging_setup import request_id_var
from ..registry import RegistryError
from ..router.router import RoutingError
from ..security import AuthContext, AuthError, RateLimitError

log = logging.getLogger("modelswitch.errors")


def error_body(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


class APIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.headers = headers or {}


def install_exception_handlers(app: Any) -> None:
    @app.exception_handler(AuthError)
    async def auth_error(_: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.message, error_type="authentication_error", code="invalid_api_key"),
            status_code=exc.status_code,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(RateLimitError)
    async def rate_error(_: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.message, error_type="rate_limit_error", code="rate_limit_exceeded"),
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.exception_handler(RoutingError)
    async def routing_error(_: Request, exc: RoutingError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.message, error_type="invalid_request_error", code=exc.code),
            status_code=exc.status_code,
        )

    @app.exception_handler(ModelUnavailable)
    async def model_error(_: Request, exc: ModelUnavailable) -> JSONResponse:
        return JSONResponse(
            error_body(
                exc.message,
                error_type="service_unavailable" if exc.status_code >= 500 else "invalid_request_error",
                code=exc.code,
            ),
            status_code=exc.status_code,
        )

    @app.exception_handler(UpstreamError)
    async def upstream_error(_: Request, exc: UpstreamError) -> JSONResponse:
        headers = {"Retry-After": "2"} if exc.status_code in (503, 504) else {}
        return JSONResponse(
            error_body(exc.message, error_type=exc.error_type, code=exc.code),
            status_code=exc.status_code,
            headers=headers,
        )

    @app.exception_handler(APIError)
    async def api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.message, error_type=exc.error_type, code=exc.code, param=exc.param),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(RegistryError)
    async def registry_error(_: Request, exc: RegistryError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.message, error_type="invalid_request_error", code="invalid_model"),
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = [str(x) for x in first.get("loc", []) if x != "body"]
        param = ".".join(loc) or None
        message = first.get("msg", "Invalid request")
        if param:
            message = f"{param}: {message}"
        return JSONResponse(
            error_body(
                message,
                error_type="invalid_request_error",
                code="invalid_request",
                param=param,
            ),
            status_code=400,
        )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            error_body(detail, error_type="invalid_request_error", code=f"http_{exc.status_code}"),
            status_code=exc.status_code,
            headers=exc.headers or {},
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", None) or request_id_var.get()
        log.exception("unhandled error request_id=%s", rid)
        return JSONResponse(
            error_body(
                "Internal gateway error. Quote the request id when reporting this issue.",
                error_type="api_error",
                code="internal_error",
            ),
            status_code=500,
            headers={"X-Request-Id": rid},
        )
