"""ASGI middleware: request correlation, body limits, and security headers."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Awaitable, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .logging_setup import api_key_id_var, request_id_var

log = logging.getLogger("modelswitch.http")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class BodyLimitMiddleware:
    """Reject oversized bodies, including chunked requests without Content-Length."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode(errors="replace") for k, v in scope.get("headers", [])}
        length = headers.get("content-length")
        if length:
            try:
                if int(length) > self.max_bytes:
                    await self._too_large(scope, send)
                    return
            except ValueError:
                await self._too_large(scope, send)
                return
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                await self._too_large(scope, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            # StreamingResponse (and other ASGI machinery) concurrently waits for
            # client disconnects. Delegate that wait; never fabricate a disconnect.
            return await receive()

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, send: Send) -> None:
        max_mb = self.max_bytes / 1024 / 1024
        payload = json.dumps(
            {
                "error": {
                    "message": f"Request body exceeds the {max_mb:.1f} MiB limit.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "request_too_large",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                    (b"x-content-type-options", b"nosniff"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, security_headers: bool = True, csp: str | None = None):
        self.app = app
        self.security_headers = security_headers
        self.csp = csp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode(errors="replace") for k, v in scope.get("headers", [])}
        supplied = headers.get("x-request-id", "")
        request_id = supplied if REQUEST_ID_RE.match(supplied) else "req_" + uuid.uuid4().hex[:20]
        token = request_id_var.set(request_id)
        api_token = api_key_id_var.set("-")
        started = time.perf_counter()
        status_code = 500

        async def wrapped_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = list(message.get("headers", []))
                if not any(k.lower() == b"x-request-id" for k, _ in response_headers):
                    response_headers.append((b"x-request-id", request_id.encode()))
                if self.security_headers:
                    response_headers.extend(
                        [
                            (b"x-content-type-options", b"nosniff"),
                            (b"x-frame-options", b"DENY"),
                            (b"referrer-policy", b"no-referrer"),
                            (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                        ]
                    )
                    if self.csp:
                        response_headers.append((b"content-security-policy", self.csp.encode()))
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            duration = (time.perf_counter() - started) * 1000
            log.info(
                "http request",
                extra={
                    "event": "http.request",
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_code,
                    "duration_ms": round(duration, 2),
                },
            )
            request_id_var.reset(token)
            api_key_id_var.reset(api_token)


class RateLimitHeadersMiddleware:
    """No-op extension point for future proxy-level headers."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)
