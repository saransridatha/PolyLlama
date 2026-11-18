"""OpenAI-compatible inference orchestration, streaming, and failover."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response, StreamingResponse

from .api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    Message,
)
from .config import Settings
from .db import Database
from .lifecycle import ModelManager, ModelUnavailable
from .logging_setup import request_id_var
from .metrics import Metrics
from .registry import ModelRegistry
from .router.classifier import TaskClassification, estimate_tokens, flatten_messages
from .router.router import IntelligentRouter, RoutingDecision, RoutingError
from .scheduler import Lease
from .security import AuthContext, RateLimiter

log = logging.getLogger("modelswitch.inference")


class UpstreamError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "upstream_error",
        error_type: str = "api_error",
        raw: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.error_type = error_type
        self.raw = raw


@dataclass(slots=True)
class Prepared:
    decision: RoutingDecision
    runtime: Any
    lease: Lease
    wait_ms: float
    queue_ms: float
    fallback_attempts: list[dict[str, Any]]

    @property
    def load_ms(self) -> float:
        return max(0.0, self.wait_ms - self.queue_ms)


class InferenceService:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        registry: ModelRegistry,
        manager: ModelManager,
        router: IntelligentRouter,
        limiter: RateLimiter,
        metrics: Metrics,
        http: httpx.AsyncClient,
    ):
        self.settings = settings
        self.db = db
        self.registry = registry
        self.manager = manager
        self.router = router
        self.limiter = limiter
        self.metrics = metrics
        self.http = http

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------
    async def chat(
        self, payload: ChatCompletionRequest, ctx: AuthContext, *, request_id: str
    ) -> Response:
        if payload.stream and not ctx.allow_streaming():
            raise _permission_error("Streaming is not permitted for this API key")
        max_tokens = self._effective_max_tokens(payload.effective_max_tokens, ctx)
        text, _, _ = flatten_messages([m.model_dump(exclude_none=True) for m in payload.messages])
        estimated_prompt = estimate_tokens(text)
        await self.limiter.check(ctx, estimated_prompt + max_tokens)
        self._start_request(request_id, ctx, "/v1/chat/completions", payload.model, payload.stream)
        started = time.perf_counter()
        try:
            decision = await self.router.route(
                requested_model=payload.model,
                messages=[m.model_dump(exclude_none=True) for m in payload.messages],
                tools=payload.tools,
                response_format=payload.response_format,
                max_tokens=max_tokens,
                allowed_models=ctx.allowed_models(),
                request_id=request_id,
            )
            prepared = await self._acquire(decision, ctx, request_id)
            upstream = payload.model_dump(exclude_none=True)
            upstream["max_tokens"] = max_tokens
            upstream.pop("max_completion_tokens", None)
            upstream["stream"] = payload.stream
            if not payload.stream:
                upstream.pop("stream_options", None)
            return await self._dispatch(
                request_id=request_id,
                ctx=ctx,
                prepared=prepared,
                endpoint="/v1/chat/completions",
                upstream=upstream,
                stream=payload.stream,
                started=started,
            )
        except Exception as exc:
            self._fail_request(request_id, exc)
            raise

    async def completion(
        self, payload: CompletionRequest, ctx: AuthContext, *, request_id: str
    ) -> Response:
        if payload.stream and not ctx.allow_streaming():
            raise _permission_error("Streaming is not permitted for this API key")
        max_tokens = self._effective_max_tokens(payload.max_tokens, ctx)
        text = _completion_text(payload.prompt)
        estimated_prompt = estimate_tokens(text)
        await self.limiter.check(ctx, estimated_prompt + max_tokens)
        self._start_request(request_id, ctx, "/v1/completions", payload.model, payload.stream)
        started = time.perf_counter()
        try:
            decision = await self.router.route(
                requested_model=payload.model,
                messages=None,
                prompt=text,
                max_tokens=max_tokens,
                allowed_models=ctx.allowed_models(),
                request_id=request_id,
            )
            prepared = await self._acquire(decision, ctx, request_id)
            upstream = payload.model_dump(exclude_none=True)
            upstream["max_tokens"] = max_tokens
            upstream["stream"] = payload.stream
            if not payload.stream:
                upstream.pop("stream_options", None)
            return await self._dispatch(
                request_id=request_id,
                ctx=ctx,
                prepared=prepared,
                endpoint="/v1/completions",
                upstream=upstream,
                stream=payload.stream,
                started=started,
            )
        except Exception as exc:
            self._fail_request(request_id, exc)
            raise

    async def embeddings(
        self, payload: EmbeddingRequest, ctx: AuthContext, *, request_id: str
    ) -> Response:
        if not ctx.allow_embeddings():
            raise _permission_error("Embeddings are not permitted for this API key")
        model = await self._select_embedding_model(payload.model, ctx)
        text = payload.input if isinstance(payload.input, str) else json.dumps(payload.input)
        await self.limiter.check(ctx, estimate_tokens(text))
        self._start_request(request_id, ctx, "/v1/embeddings", payload.model, False)
        started = time.perf_counter()
        lease = None
        acquire_started = time.perf_counter()
        try:
            runtime, lease = await self.manager.acquire(
                model["id"], priority=_priority(ctx), key_id=ctx.key_id
            )
            decision = RoutingDecision(
                requested_model=payload.model,
                selected_model=model["id"],
                task="embeddings",
                task_label="Embeddings",
                confidence=1.0,
                cap="embeddings",
                strategy="explicit" if payload.model not in ("auto", "default") else "embedding",
                explanation=f"Selected {model['id']} for embeddings.",
                alternatives=[],
                constraints={"model_context_tokens": model["context_length"]},
                candidate_ids=[model["id"]],
                routing_ms=0,
                classifier_source="embedding-selector",
                classification=TaskClassification(
                    category="embeddings",
                    confidence=1.0,
                    scores={"embeddings": 1.0},
                    signals=["embeddings endpoint"],
                    language="english",
                    estimated_tokens=estimate_tokens(text),
                    complexity="low",
                    long_context=False,
                    needs_json=False,
                    needs_tools=False,
                    needs_vision=False,
                    ambiguous=False,
                    rationale="Embeddings request.",
                ),
                request_id=request_id,
            )
            upstream = payload.model_dump(exclude_none=True)
            upstream["model"] = model["id"]
            resp = await self._request(
                runtime, "/v1/embeddings", upstream, stream=False
            )
            if resp.status_code >= 400:
                raise _upstream_error(resp)
            data = resp.json()
            data["model"] = model["id"]
            data.setdefault("object", "list")
            data["modelswitch"] = {"routing": decision.to_public()}
            tokens = int(data.get("usage", {}).get("prompt_tokens") or estimate_tokens(text))
            total_ms = (time.perf_counter() - started) * 1000
            self._complete_request(
                request_id,
                decision,
                status="completed",
                http_status=200,
                prompt_tokens=tokens,
                completion_tokens=0,
                queue_ms=lease.queue_ms,
                load_ms=(time.perf_counter() - acquire_started) * 1000 - lease.queue_ms,
                ttft_ms=None,
                total_ms=total_ms,
                ctx=ctx,
            )
            self.db.update_perf(
                model["id"], ttft_ms=None, tps=None, total_ms=total_ms, success=True
            )
            wait_ms = (time.perf_counter() - acquire_started) * 1000
            return JSONResponse(
                data,
                headers=self._headers(
                    decision, lease, request_id, max(0.0, wait_ms - lease.queue_ms)
                ),
            )
        except Exception as exc:
            self._fail_request(request_id, exc)
            raise
        finally:
            if lease:
                lease.release()

    # ------------------------------------------------------------------
    # Selection / routing
    # ------------------------------------------------------------------
    async def _select_embedding_model(self, requested: str, ctx: AuthContext) -> dict[str, Any]:
        allowed = ctx.allowed_models()
        requested_lower = requested.lower()
        if requested_lower not in ("auto", "default", "fast", "coding", "reasoning"):
            model = self.registry.find(requested)
            if not model:
                raise RoutingError(
                    f"Model '{requested}' is not registered",
                    status_code=404,
                    code="model_not_found",
                )
            if allowed is not None and model["id"] not in allowed:
                raise _permission_error("API key is not permitted to use this model")
            if "embeddings" not in model.get("capabilities", []):
                raise RoutingError(
                    f"Model '{model['id']}' does not advertise embeddings support",
                    status_code=400,
                    code="embeddings_not_supported",
                )
            return model
        candidates = [
            m
            for m in self.registry.list(include_disabled=False)
            if "embeddings" in m.get("capabilities", [])
            and m["enabled"]
            and (allowed is None or m["id"] in allowed)
            and self.registry.availability(m)["available"]
        ]
        if not candidates:
            raise RoutingError(
                "No embeddings-capable model is registered. Register an embedding GGUF "
                "or call a specific embedding model.",
                status_code=400,
                code="embeddings_not_supported",
            )
        candidates.sort(
            key=lambda m: (
                0 if self.manager.is_loaded(m["id"]) else 1,
                -int(m.get("priority", 50)),
                int(m.get("size_bytes", 0)),
            )
        )
        return candidates[0]

    async def _acquire(
        self,
        decision: RoutingDecision,
        ctx: AuthContext,
        request_id: str,
        exclude: set[str] | None = None,
    ) -> Prepared:
        candidates = decision.candidate_ids or ([decision.selected_model] if decision.selected_model else [])
        attempts: list[dict[str, Any]] = []
        last_error: ModelUnavailable | None = None
        acquire_started = time.perf_counter()
        for idx, model_id in enumerate(candidates):
            if not model_id:
                continue
            if exclude and model_id in exclude:
                attempts.append({"model": model_id, "outcome": "already_attempted"})
                continue
            if not ctx.allows_model(model_id):
                attempts.append({"model": model_id, "outcome": "not_permitted"})
                continue
            try:
                runtime, lease = await self.manager.acquire(
                    model_id,
                    priority=_priority(ctx),
                    key_id=ctx.key_id,
                    queue_timeout=self.settings.queue_timeout,
                )
            except ModelUnavailable as exc:
                last_error = exc
                attempts.append(
                    {"model": model_id, "outcome": "unavailable", "message": exc.message}
                )
                self.db.add_event(
                    "warning",
                    "route.fallback",
                    f"Model {model_id} unavailable: {exc.message}",
                    model_id,
                    {"request_id": request_id},
                )
                continue
            if idx > 0:
                decision.fallback_used = True
                decision.selected_model = model_id
                decision.explanation += (
                    f" Fallback: {model_id} selected after {len(attempts)} failed attempt(s)."
                )
                decision.constraints["fallback_attempts"] = attempts
                self.db.update_routing_decision(
                    request_id,
                    selected_model=model_id,
                    explanation=decision.explanation,
                    fallback=True,
                )
                self.metrics.inc(
                    "modelswitch_fallback_total",
                    {"model": model_id, "status": "used"},
                )
            wait_ms = (time.perf_counter() - acquire_started) * 1000
            return Prepared(
                decision=decision,
                runtime=runtime,
                lease=lease,
                wait_ms=wait_ms,
                queue_ms=lease.queue_ms,
                fallback_attempts=attempts,
            )
        if last_error:
            if last_error.status_code == 404:
                raise RoutingError(
                    last_error.message, status_code=404, code="model_not_found"
                ) from last_error
            raise last_error
        raise RoutingError(
            "No permitted model is available for this request",
            status_code=403,
            code="model_not_permitted",
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def _dispatch(
        self,
        *,
        request_id: str,
        ctx: AuthContext,
        prepared: Prepared,
        endpoint: str,
        upstream: dict[str, Any],
        stream: bool,
        started: float,
    ) -> Response:
        """Execute, retrying on a different feasible model for pre-stream failures."""
        tried: set[str] = set()
        last_error: Exception | None = None
        while True:
            selected = prepared.decision.selected_model or ""
            upstream["model"] = selected
            try:
                return await self._execute(
                    request_id=request_id,
                    ctx=ctx,
                    prepared=prepared,
                    endpoint=endpoint,
                    upstream=upstream,
                    stream=stream,
                    started=started,
                )
            except UpstreamError as exc:
                last_error = exc
                tried.add(selected)
                retryable = (
                    exc.status_code >= 500
                    and self.router.config.fallback_enabled
                    and prepared.decision.strategy != "explicit"
                    and len(tried) < 3
                )
                if not retryable:
                    raise
                self.metrics.inc(
                    "modelswitch_fallback_total",
                    {"model": selected, "status": "backend_error"},
                )
                self.db.add_event(
                    "warning",
                    "route.fallback",
                    f"Backend failure on {selected}; trying next candidate: {exc.message}",
                    selected,
                    {"request_id": request_id, "status": exc.status_code},
                )
                prepared = await self._acquire(
                    prepared.decision, ctx, request_id, exclude=tried
                )
                if not prepared:
                    if last_error:
                        raise last_error
                    raise RoutingError(
                        "No fallback model is available", status_code=503, code="no_feasible_model"
                    )

    async def _execute(
        self,
        *,
        request_id: str,
        ctx: AuthContext,
        prepared: Prepared,
        endpoint: str,
        upstream: dict[str, Any],
        stream: bool,
        started: float,
    ) -> Response:
        decision = prepared.decision
        runtime = prepared.runtime
        lease = prepared.lease
        headers = self._headers(decision, lease, request_id, prepared.load_ms)
        if stream:
            try:
                req = self.http.build_request(
                    "POST",
                    f"{runtime.backend.base_url}{endpoint}",
                    json=upstream,
                    headers={"x-request-id": request_id},
                )
                resp = await self.http.send(req, stream=True)
                if resp.status_code >= 400:
                    body = await resp.aread()
                    await resp.aclose()
                    raise _upstream_error(resp, body)
            except Exception:
                lease.release()
                raise
            generator = self._stream_generator(
                resp=resp,
                prepared=prepared,
                ctx=ctx,
                request_id=request_id,
                started=started,
                endpoint=endpoint,
            )
            self.metrics.set(
                "modelswitch_active_requests", self._active_count(), {"endpoint": endpoint}
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    **headers,
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )
        try:
            resp = await self._request(runtime, endpoint, upstream, stream=False)
            if resp.status_code >= 400:
                raise _upstream_error(resp)
            data = resp.json()
            data = self._normalize_response(data, decision.selected_model or "", endpoint)
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(
                json.dumps(upstream.get("messages") or upstream.get("prompt") or "")
            ))
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_ms = (time.perf_counter() - started) * 1000
            timings = data.pop("timings", None) or {}
            ttft_ms = _timing_ttft(timings)
            tps = _timing_tps(timings, completion_tokens, total_ms, ttft_ms)
            data["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
            }
            data["modelswitch"] = {
                "routing": decision.to_public(),
                "queue_ms": round(prepared.queue_ms, 2),
                "load_ms": round(prepared.load_ms, 2),
                "wait_ms": round(prepared.wait_ms, 2),
                "request_id": request_id,
            }
            self._complete_request(
                request_id,
                decision,
                status="completed",
                http_status=200,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                queue_ms=prepared.queue_ms,
                load_ms=prepared.load_ms,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                ctx=ctx,
            )
            self.db.update_perf(
                decision.selected_model,
                ttft_ms=ttft_ms,
                tps=tps,
                total_ms=total_ms,
                success=True,
            )
            self._observe(decision, endpoint, 200, ttft_ms, total_ms, prompt_tokens, completion_tokens)
            return JSONResponse(data, headers=headers)
        finally:
            lease.release()

    async def _stream_generator(
        self,
        *,
        resp: httpx.Response,
        prepared: Prepared,
        ctx: AuthContext,
        request_id: str,
        started: float,
        endpoint: str,
    ) -> AsyncIterator[bytes]:
        decision = prepared.decision
        lease = prepared.lease
        buffer = b""
        completion_tokens = 0
        prompt_tokens = 0
        ttft_ms: float | None = None
        first_content_seen = False
        generated_chars = 0
        status = "completed"
        http_status = 200
        error_message: str | None = None
        try:
            async with asyncio.timeout(self.settings.request_timeout):
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        parsed = _parse_sse_event(event)
                        if parsed:
                            usage = parsed.get("usage")
                            if usage:
                                prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens") or 0))
                                completion_tokens = max(
                                    completion_tokens, int(usage.get("completion_tokens") or 0)
                                )
                            generated_chars += _content_chars(parsed)
                            if not first_content_seen and _has_content(parsed):
                                first_content_seen = True
                                ttft_ms = (time.perf_counter() - started) * 1000
                            timings = parsed.get("timings") or {}
                            if timings:
                                completion_tokens = max(
                                    completion_tokens, int(timings.get("predicted_n") or 0)
                                )
                                prompt_tokens = max(
                                    prompt_tokens, int(timings.get("prompt_n") or 0)
                                )
                    yield chunk
            if not first_content_seen:
                ttft_ms = (time.perf_counter() - started) * 1000
            total_ms = (time.perf_counter() - started) * 1000
            if not prompt_tokens:
                prompt_tokens = prepared.decision.classification.estimated_tokens
            if not completion_tokens and generated_chars:
                completion_tokens = max(1, round(generated_chars / 3.6))
            self._complete_request(
                request_id,
                decision,
                status="completed",
                http_status=200,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                queue_ms=prepared.queue_ms,
                load_ms=prepared.load_ms,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                ctx=ctx,
            )
            tps = (
                completion_tokens / max(0.001, (total_ms - (ttft_ms or 0)) / 1000)
                if completion_tokens
                else None
            )
            self.db.update_perf(
                decision.selected_model,
                ttft_ms=ttft_ms,
                tps=tps,
                total_ms=total_ms,
                success=True,
            )
            self._observe(decision, endpoint, 200, ttft_ms, total_ms, prompt_tokens, completion_tokens)
        except asyncio.CancelledError:
            status = "cancelled"
            total_ms = (time.perf_counter() - started) * 1000
            if not prompt_tokens:
                prompt_tokens = prepared.decision.classification.estimated_tokens
            if not completion_tokens and generated_chars:
                completion_tokens = max(1, round(generated_chars / 3.6))
            self._complete_request(
                request_id,
                decision,
                status="cancelled",
                http_status=499,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                queue_ms=prepared.queue_ms,
                load_ms=prepared.load_ms,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                error="client disconnected",
                ctx=ctx,
            )
            self.db.update_perf(
                decision.selected_model,
                ttft_ms=None,
                tps=None,
                total_ms=total_ms,
                success=False,
            )
            raise
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            http_status = 502
            total_ms = (time.perf_counter() - started) * 1000
            log.warning("stream failed: %s", exc, extra={"request_id": request_id})
            try:
                payload = {
                    "error": {
                        "message": "The model backend stream failed before completion.",
                        "type": "api_error",
                        "param": None,
                        "code": "backend_stream_error",
                    }
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            except Exception:  # nosec B110 - best-effort terminal SSE write
                pass
            if not prompt_tokens:
                prompt_tokens = prepared.decision.classification.estimated_tokens
            if not completion_tokens and generated_chars:
                completion_tokens = max(1, round(generated_chars / 3.6))
            self._complete_request(
                request_id,
                decision,
                status=status,
                http_status=http_status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                queue_ms=prepared.queue_ms,
                load_ms=prepared.load_ms,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                error=error_message,
                ctx=ctx,
            )
            self.db.update_perf(
                decision.selected_model,
                ttft_ms=None,
                tps=None,
                total_ms=total_ms,
                success=False,
            )
        finally:
            self.metrics.set(
                "modelswitch_active_requests", self._active_count(), {"endpoint": endpoint}
            )
            try:
                await resp.aclose()
            except Exception:  # nosec B110 - closing a failed upstream stream
                pass
            lease.release()

    async def _request(
        self,
        runtime: Any,
        endpoint: str,
        payload: dict[str, Any],
        *,
        stream: bool,
    ) -> httpx.Response:
        try:
            return await self.http.post(
                f"{runtime.backend.base_url}{endpoint}",
                json=payload,
                timeout=httpx.Timeout(
                    self.settings.request_timeout,
                    connect=10.0,
                    read=self.settings.request_timeout,
                    write=30.0,
                    pool=10.0,
                ),
            )
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "The model backend timed out.",
                status_code=504,
                code="backend_timeout",
                error_type="timeout_error",
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "Unable to reach the model backend.",
                status_code=503,
                code="backend_unavailable",
            ) from exc

    # ------------------------------------------------------------------
    # Response helpers / persistence
    # ------------------------------------------------------------------
    def _effective_max_tokens(self, requested: int | None, ctx: AuthContext) -> int:
        value = int(requested or self.settings.default_max_tokens)
        cap = ctx.max_tokens_cap()
        if cap:
            value = min(value, cap)
        return max(1, min(value, 131072))

    def _start_request(
        self,
        request_id: str,
        ctx: AuthContext,
        endpoint: str,
        requested_model: str,
        stream: bool,
    ) -> None:
        self.db.start_request(
            request_id=request_id,
            api_key_id=ctx.key_id,
            endpoint=endpoint,
            requested_model=requested_model,
            stream=stream,
        )
        self.metrics.inc("modelswitch_requests_started_total", {"endpoint": endpoint})
        request_id_var.set(request_id)

    def _complete_request(
        self,
        request_id: str,
        decision: RoutingDecision,
        *,
        status: str,
        http_status: int,
        prompt_tokens: int,
        completion_tokens: int,
        queue_ms: float,
        load_ms: float = 0.0,
        ttft_ms: float | None = None,
        total_ms: float,
        ctx: AuthContext | None = None,
        error: str | None = None,
    ) -> None:
        try:
            self.db.finish_request(
                request_id,
                status=status,
                http_status=http_status,
                resolved_model=decision.selected_model,
                task_category=decision.task,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                queue_ms=queue_ms,
                load_ms=load_ms,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                error=error,
                routing=decision.to_public(),
            )
        except Exception:
            log.exception("failed to persist request completion")
        if ctx:
            asyncio.create_task(
                self.limiter.record_usage(ctx.key_id, prompt_tokens + completion_tokens)
            )

    def _fail_request(self, request_id: str, exc: Exception) -> None:
        try:
            current = self.db.get_request(request_id)
            if not current or current.get("status") != "running":
                return
            status_code = getattr(exc, "status_code", 500)
            self.db.finish_request(
                request_id,
                status="failed",
                http_status=status_code,
                error=str(exc)[:1000],
            )
            self.metrics.inc(
                "modelswitch_requests_total",
                {"endpoint": current["endpoint"], "status": "failed", "model": "none"},
            )
        except Exception:
            log.exception("failed to mark request failed")

    def _normalize_response(
        self, data: dict[str, Any], model: str, endpoint: str
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise UpstreamError("Backend returned a non-JSON response")
        now = int(time.time())
        data.setdefault("id", f"{'chatcmpl' if 'chat' in endpoint else 'cmpl'}-{int(time.time()*1000)}")
        data.setdefault("object", "chat.completion" if "chat" in endpoint else "text_completion")
        data.setdefault("created", now)
        data["model"] = model
        if not isinstance(data.get("choices"), list):
            raise UpstreamError("Backend returned an invalid completion payload")
        usage = data.get("usage")
        if not isinstance(usage, dict):
            data["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return data

    def _headers(
        self,
        decision: RoutingDecision,
        lease: Lease,
        request_id: str,
        load_ms: float = 0.0,
    ) -> dict[str, str]:
        return {
            "X-Request-Id": request_id,
            "X-ModelSwitch-Model": decision.selected_model or "",
            "X-ModelSwitch-Task": decision.task,
            "X-ModelSwitch-Routing-Ms": f"{decision.routing_ms:.1f}",
            "X-ModelSwitch-Queue-Ms": f"{lease.queue_ms:.1f}",
            "X-ModelSwitch-Load-Ms": f"{load_ms:.1f}",
            "X-ModelSwitch-Fallback": "1" if decision.fallback_used else "0",
        }

    def _observe(
        self,
        decision: RoutingDecision,
        endpoint: str,
        status: int,
        ttft_ms: float | None,
        total_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        model = decision.selected_model or "none"
        self.metrics.inc(
            "modelswitch_requests_total",
            {"endpoint": endpoint, "status": "success", "model": model},
        )
        self.metrics.observe(
            "modelswitch_request_duration_seconds", total_ms / 1000, {"model": model}
        )
        if ttft_ms is not None:
            self.metrics.observe("modelswitch_ttft_seconds", ttft_ms / 1000, {"model": model})
        self.metrics.observe(
            "modelswitch_routing_seconds",
            decision.routing_ms / 1000,
            {"task": decision.task},
        )
        self.metrics.inc(
            "modelswitch_tokens_total", {"model": model, "type": "prompt"}, prompt_tokens
        )
        self.metrics.inc(
            "modelswitch_tokens_total",
            {"model": model, "type": "completion"},
            completion_tokens,
        )

    def _active_count(self) -> int:
        return sum(rt.active for rt in self.manager.runtimes.values() if rt.is_loaded())


def _priority(ctx: AuthContext) -> int:
    return 10 if ctx.is_admin else 5


def _permission_error(message: str) -> RoutingError:
    return RoutingError(message, status_code=403, code="permission_denied")


def _upstream_error(resp: httpx.Response, body: bytes | None = None) -> UpstreamError:
    if body is None:
        try:
            body = resp.content
        except Exception:
            body = b""
    message = f"Model backend returned HTTP {resp.status_code}"
    code = "backend_error"
    error_type = "api_error"
    try:
        payload = json.loads(body)
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            message = str(err.get("message") or message)
            code = str(err.get("code") or code)
        elif isinstance(err, str):
            message = err
        elif isinstance(payload, dict) and payload.get("message"):
            message = str(payload["message"])
    except (json.JSONDecodeError, UnicodeDecodeError):
        if body:
            message += f": {body[:200].decode(errors='replace')}"
    status = 502 if resp.status_code >= 500 else resp.status_code
    if resp.status_code == 400:
        error_type = "invalid_request_error"
    if resp.status_code == 404:
        status = 400
        code = "model_not_found"
    if resp.status_code == 503:
        code = "backend_unavailable"
        error_type = "service_unavailable"
    return UpstreamError(
        message, status_code=status, code=code, error_type=error_type, raw=body[:2000]
    )


def _completion_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, int):
                parts.append(f"<{item}>")
            elif isinstance(item, list):
                parts.append(" ".join(str(t) for t in item))
        return "\n".join(parts)
    return str(prompt)


def _parse_sse_event(event: bytes) -> dict[str, Any] | None:
    data_lines = []
    for line in event.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return None
    payload = b"\n".join(data_lines)
    if payload == b"[DONE]":
        return None
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _content_chars(chunk: dict[str, Any]) -> int:
    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return 0
    delta = choices[0].get("delta") or {}
    text = delta.get("content") or delta.get("reasoning_content")
    if text:
        return len(str(text))
    text = choices[0].get("text")
    return len(str(text)) if text else 0


def _has_content(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    delta = choices[0].get("delta") or {}
    if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
        return True
    return bool(choices[0].get("text"))


def _timing_ttft(timings: dict[str, Any]) -> float | None:
    value = timings.get("prompt_ms")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timing_tps(
    timings: dict[str, Any],
    completion_tokens: int,
    total_ms: float,
    ttft_ms: float | None,
) -> float | None:
    try:
        if timings.get("predicted_per_second"):
            return float(timings["predicted_per_second"])
    except (TypeError, ValueError):
        pass
    if completion_tokens and total_ms > 0:
        generation_ms = max(1.0, total_ms - (ttft_ms or 0))
        return completion_tokens / (generation_ms / 1000)
    return None
