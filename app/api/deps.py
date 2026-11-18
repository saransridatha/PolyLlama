"""FastAPI authentication dependencies for the public and admin apps."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from ..logging_setup import api_key_id_var
from ..security import AuthContext, AuthError, RateLimitError


def get_runtime(request: Request) -> Any:
    return request.app.state.runtime


def client_ip(request: Request) -> str:
    runtime = get_runtime(request)
    if runtime.settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
        real = request.headers.get("x-real-ip")
        if real:
            return real.strip()[:64]
    return request.client.host if request.client else "unknown"


def extract_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return None
    key = request.headers.get("x-api-key")
    return key.strip() if key else None


async def require_auth(request: Request) -> AuthContext:
    runtime = get_runtime(request)
    ip = client_ip(request)
    raw = extract_key(request)
    if not await runtime.limiter.auth_failure_allowed(ip):
        raise RateLimitError("Too many failed authentication attempts", 60)
    if not raw:
        await runtime.limiter.register_auth_failure(ip)
        raise AuthError("Missing API key. Use 'Authorization: Bearer <key>'.")
    try:
        ctx = runtime.keys.authenticate(raw)
    except AuthError:
        await runtime.limiter.register_auth_failure(ip)
        raise
    api_key_id_var.set(ctx.key_id)
    return ctx


async def require_admin(request: Request) -> AuthContext:
    runtime = get_runtime(request)
    ctx = await require_auth(request)
    if not ctx.is_admin:
        raise AuthError("Administrator access required", 403)
    # Admin keys carry their own configurable rate limits too. This prevents a
    # stolen admin credential from being used for unbounded dashboard requests.
    await runtime.limiter.check(ctx, 0)
    return ctx


async def optional_auth(request: Request) -> AuthContext | None:
    if not extract_key(request):
        return None
    return await require_auth(request)
