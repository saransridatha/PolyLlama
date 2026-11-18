"""Deterministic, explainable model-selection policy.

The policy combines task fit, capability metadata, context requirements, live
memory feasibility, measured latency/throughput, residency state, queue depth,
and operator priority. It never selects an infeasible model merely because it is
the largest.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import Settings
from ..db import Database
from ..gguf import GGUFMetadata
from ..lifecycle import ModelManager
from ..registry import ModelRegistry
from .classifier import (
    TASK_CAPABILITY,
    TASK_LABELS,
    TaskClassification,
    build_llm_prompt,
    classify_task,
    parse_llm_classification,
)

STRATEGIES = ("balanced", "quality", "latency", "local")
ROLE_ALIASES = ("auto", "default", "fast", "coding", "reasoning")


@dataclass(slots=True)
class RouterConfig:
    strategy: str = "balanced"
    llm_enabled: bool = False
    llm_model: str = ""
    llm_timeout_ms: int = 2500
    ambiguity_threshold: float = 0.55
    prioritize_loaded: bool = True
    fallback_enabled: bool = True
    allow_general_fallback: bool = True
    max_alternatives: int = 4

    def validate(self) -> "RouterConfig":
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        self.llm_timeout_ms = max(200, min(10_000, int(self.llm_timeout_ms)))
        self.ambiguity_threshold = max(0.0, min(1.0, float(self.ambiguity_threshold)))
        self.max_alternatives = max(1, min(10, int(self.max_alternatives)))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Candidate:
    model: dict[str, Any]
    score: float
    components: dict[str, float]
    loaded: bool
    feasible: bool
    reason: str
    estimate: dict[str, Any] = field(default_factory=dict)
    queue_depth: int = 0
    rejection: str | None = None


@dataclass(slots=True)
class RoutingDecision:
    requested_model: str
    selected_model: str | None
    task: str
    task_label: str
    confidence: float
    cap: str
    strategy: str
    explanation: str
    alternatives: list[dict[str, Any]]
    constraints: dict[str, Any]
    candidate_ids: list[str]
    routing_ms: float
    classifier_source: str
    classification: TaskClassification
    fallback_used: bool = False
    resolved_alias: str | None = None
    request_id: str = ""
    error: str | None = None

    def to_public(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "task_label": self.task_label,
            "confidence": self.confidence,
            "capability": self.cap,
            "strategy": self.strategy,
            "selected_model": self.selected_model,
            "explanation": self.explanation,
            "alternatives": self.alternatives,
            "constraints": self.constraints,
            "routing_ms": round(self.routing_ms, 2),
            "classifier_source": self.classifier_source,
            "fallback_used": self.fallback_used,
            "resolved_alias": self.resolved_alias,
            "classification": self.classification.to_dict(),
        }


class RoutingError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, code: str = "routing_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class IntelligentRouter:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        registry: ModelRegistry,
        manager: ModelManager,
    ):
        self.settings = settings
        self.db = db
        self.registry = registry
        self.manager = manager
        self.monitor = manager.monitor
        self.config = self._load_config()

    def _load_config(self) -> RouterConfig:
        stored = self.db.get_setting("router_config", {}) or {}
        config = RouterConfig(
            strategy=stored.get("strategy", self.settings.router_strategy),
            llm_enabled=stored.get("llm_enabled", self.settings.router_llm_enabled),
            llm_model=stored.get("llm_model", self.settings.router_llm_model),
            llm_timeout_ms=stored.get("llm_timeout_ms", self.settings.router_llm_timeout_ms),
            ambiguity_threshold=stored.get(
                "ambiguity_threshold", self.settings.router_ambiguity_threshold
            ),
            prioritize_loaded=stored.get(
                "prioritize_loaded", self.settings.router_prioritize_loaded
            ),
            fallback_enabled=stored.get("fallback_enabled", True),
            allow_general_fallback=stored.get("allow_general_fallback", True),
            max_alternatives=stored.get("max_alternatives", 4),
        )
        try:
            return config.validate()
        except ValueError:
            return RouterConfig()

    def update_config(self, patch: dict[str, Any]) -> RouterConfig:
        current = self.config.to_dict()
        for key in current:
            if key in patch and patch[key] is not None:
                current[key] = patch[key]
        config = RouterConfig(**current).validate()
        if config.llm_model:
            model = self.registry.find(config.llm_model)
            if not model:
                raise ValueError(f"Unknown router model '{config.llm_model}'")
            config.llm_model = model["id"]
        self.config = config
        self.db.set_setting("router_config", config.to_dict())
        return config

    def aliases(self) -> dict[str, Any]:
        role_aliases = {
            "auto": {
                "description": "Intelligent task-aware routing",
                "kind": "router",
            },
            "default": {"description": "Alias for auto", "kind": "router"},
            "fast": {
                "description": "Lowest expected latency feasible model",
                "kind": "role",
            },
            "coding": {
                "description": "Best feasible coding/debugging model",
                "kind": "role",
            },
            "reasoning": {
                "description": "Best feasible reasoning/math model",
                "kind": "role",
            },
        }
        model_aliases = {}
        for model in self.registry.list():
            for alias in model.get("aliases", []):
                model_aliases[alias] = {"description": f"Alias for {model['id']}", "kind": "model"}
        return {"role_aliases": role_aliases, "model_aliases": model_aliases}

    async def route(
        self,
        *,
        requested_model: str,
        messages: list[dict[str, Any]] | None,
        prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 0,
        allowed_models: list[str] | None = None,
        request_id: str = "",
    ) -> RoutingDecision:
        started = time.perf_counter()
        classification = classify_task(
            messages=messages,
            prompt=prompt,
            tools=tools,
            response_format=response_format,
            max_tokens=max_tokens,
        )
        requested = (requested_model or "auto").strip()
        role = requested.lower() if requested.lower() in ROLE_ALIASES else None
        explicit = self.registry.find(requested) if not role else None
        if not role and not explicit:
            raise RoutingError(
                f"Model '{requested}' is not registered", status_code=404, code="model_not_found"
            )
        if role in ("fast", "coding", "reasoning"):
            classification = _apply_role(classification, role)
        if (
            role in ("auto", "default", None)
            and self.config.llm_enabled
            and classification.ambiguous
            and classification.confidence < max(self.config.ambiguity_threshold, 0.95)
        ):
            classification = await self._llm_classify(classification, messages, prompt)

        if explicit:
            decision = self._explicit_decision(
                requested=requested,
                model=explicit,
                classification=classification,
                max_tokens=max_tokens,
                elapsed=(time.perf_counter() - started) * 1000,
                request_id=request_id,
                allowed_models=allowed_models,
            )
            return decision

        if role is None:
            raise RoutingError("Unable to resolve routing alias", status_code=400)
        decision = self._policy_decision(
            requested=requested,
            role=None if role in ("auto", "default") else role,
            classification=classification,
            max_tokens=max_tokens,
            allowed_models=allowed_models,
            started=started,
            request_id=request_id,
        )
        return decision

    def _explicit_decision(
        self,
        *,
        requested: str,
        model: dict[str, Any],
        classification: TaskClassification,
        max_tokens: int,
        elapsed: float,
        request_id: str = "",
        allowed_models: list[str] | None = None,
    ) -> RoutingDecision:
        availability = self.registry.availability(model)
        required = classification.estimated_tokens + max(1, max_tokens) + 64
        error = None
        error_status = 400
        error_code = "model_unavailable"
        if allowed_models is not None and model["id"] not in allowed_models:
            error = f"API key is not permitted to use model '{model['id']}'"
            error_status, error_code = 403, "model_not_permitted"
        elif not model["enabled"]:
            error = f"Model '{model['id']}' is disabled"
            error_status, error_code = 409, "model_disabled"
        elif not availability["available"]:
            error = f"Model '{model['id']}' file is unavailable"
            error_status, error_code = 503, "model_file_missing"
        elif int(model["context_length"]) < required:
            error = (
                f"Request needs ~{required} tokens but model context is "
                f"{model['context_length']}"
            )
            error_status, error_code = 400, "context_length_exceeded"
        elif not (set(model.get("capabilities", [])) & {"chat", "general"}):
            error = f"Model '{model['id']}' is not a text generation model"
            error_status, error_code = 400, "not_a_chat_model"
        runtime = self.manager.runtimes.get(model["id"])
        loaded = bool(runtime and runtime.is_loaded())
        explanation = (
            f"Explicit model '{model['id']}' requested; automatic selection bypassed. "
            f"Task classified as {classification.label.lower()} for observability. "
            f"Context requirement ~{required} tokens, model limit {model['context_length']}."
        )
        decision = RoutingDecision(
            requested_model=requested,
            selected_model=model["id"] if not error else None,
            task=classification.category,
            task_label=classification.label,
            confidence=classification.confidence,
            cap=classification.capability,
            strategy="explicit",
            explanation=explanation,
            alternatives=[],
            constraints={
                "required_context_tokens": required,
                "model_context_tokens": model["context_length"],
                "loaded": loaded,
                "explicit": True,
            },
            candidate_ids=[model["id"]] if not error else [],
            routing_ms=elapsed,
            classifier_source=classification.source,
            classification=classification,
            error=error,
            request_id=request_id,
        )
        self._record(decision)
        if error:
            raise RoutingError(error, status_code=error_status, code=error_code)
        return decision

    def _policy_decision(
        self,
        *,
        requested: str,
        role: str | None,
        classification: TaskClassification,
        max_tokens: int,
        allowed_models: list[str] | None,
        started: float,
        request_id: str = "",
    ) -> RoutingDecision:
        required = classification.estimated_tokens + max(1, max_tokens) + 64
        models = self.registry.list(include_disabled=False)
        allowed = set(allowed_models) if allowed_models else None
        candidates: list[Candidate] = []
        excluded: list[dict[str, str]] = []
        memory = self.monitor.available_memory_for_models(
            self.settings.memory_safety_factor, self.settings.min_free_ram_mb * 1024 * 1024
        )
        evictable_ram = 0
        evictable_vram = 0
        for runtime in self.manager.runtimes.values():
            if runtime.is_loaded() and runtime.active == 0 and runtime.waiting == 0:
                evictable_ram += int(runtime.estimate.get("ram_bytes", 0))
                evictable_vram += int(runtime.estimate.get("vram_bytes", 0))
        for model in models:
            if allowed is not None and model["id"] not in allowed:
                excluded.append({"model": model["id"], "reason": "not permitted for this API key"})
                continue
            availability = self.registry.availability(model)
            if not availability["available"]:
                excluded.append({"model": model["id"], "reason": "model file unavailable"})
                continue
            if int(model["context_length"]) < required:
                excluded.append(
                    {
                        "model": model["id"],
                        "reason": (
                            f"context {model['context_length']} < required ~{required} tokens"
                        ),
                    }
                )
                continue
            caps = set(model.get("capabilities", []))
            if not (caps & {"chat", "general"}):
                excluded.append(
                    {"model": model["id"], "reason": "not a text generation model"}
                )
                continue
            if classification.needs_vision and "vision" not in caps:
                excluded.append(
                    {"model": model["id"], "reason": "request contains image input but model lacks vision capability"}
                )
                continue
            candidate = self._score_candidate(
                model,
                classification,
                role,
                required,
                memory,
                evictable_ram,
                evictable_vram,
                strict_capability=role in ("coding", "reasoning"),
            )
            if candidate.rejection:
                excluded.append({"model": model["id"], "reason": candidate.rejection})
                continue
            if role in ("coding", "reasoning") and candidate.components["capability"] < 50:
                if self.config.allow_general_fallback:
                    candidate.components["capability"] = 35
                    candidate.components["total"] = _weighted(
                        candidate.components, "balanced", self.config.prioritize_loaded
                    )
                    candidate.score = candidate.components["total"]
                    candidate.reason += "; no specialist model available, using general fallback"
                else:
                    excluded.append(
                        {"model": model["id"], "reason": f"no {role} capability metadata"}
                    )
                    continue
            candidates.append(candidate)

        if not candidates:
            self._raise_no_candidate(classification, required, excluded, models)
        candidates.sort(key=lambda c: c.score, reverse=True)
        selected = candidates[0]
        alternatives = [
            {
                "model": c.model["id"],
                "score": round(c.score, 2),
                "loaded": c.loaded,
                "capability_match": c.components["capability"] >= 50,
                "estimated_load_ms": c.estimate.get("estimated_load_ms"),
                "reason": c.reason,
            }
            for c in candidates[1 : 1 + self.config.max_alternatives]
        ]
        explanation = self._explain(selected, classification, role, candidates)
        candidate_ids = [c.model["id"] for c in candidates[:5]] if self.config.fallback_enabled else [
            selected.model["id"]
        ]
        decision = RoutingDecision(
            requested_model=requested,
            selected_model=selected.model["id"],
            task=classification.category,
            task_label=classification.label,
            confidence=classification.confidence,
            cap=classification.capability,
            strategy=role or self.config.strategy,
            explanation=explanation,
            alternatives=alternatives,
            constraints={
                "required_context_tokens": required,
                "available_ram_bytes": memory["ram_bytes"],
                "available_vram_bytes": memory["vram_bytes"],
                "evictable_ram_bytes": evictable_ram,
                "max_resident_models": self.settings.max_resident_models,
                "resident_models": self.manager.loaded_models(),
                "excluded": excluded[:20],
                "excluded_count": len(excluded),
                "requested_alias": requested,
                "role": role,
            },
            candidate_ids=candidate_ids,
            routing_ms=(time.perf_counter() - started) * 1000,
            classifier_source=classification.source,
            classification=classification,
            resolved_alias=role,
            request_id=request_id,
        )
        self._record(decision)
        return decision

    def _score_candidate(
        self,
        model: dict[str, Any],
        classification: TaskClassification,
        role: str | None,
        required: int,
        memory: dict[str, int],
        evictable_ram: int,
        evictable_vram: int,
        strict_capability: bool,
    ) -> Candidate:
        runtime = self.manager.runtimes.get(model["id"])
        loaded = bool(runtime and runtime.is_loaded())
        try:
            meta = self.registry.metadata(Path(model["path"]))
            requested_ngl = model.get("n_gpu_layers")
            if requested_ngl is None and self.settings.llama_n_gpu_layers >= 0:
                requested_ngl = self.settings.llama_n_gpu_layers
            ngl = self.monitor.plan_offload(meta, requested_ngl)
            estimate = self.registry.estimate_memory(
                model,
                context_length=int(model["context_length"]),
                slots=int(model.get("max_concurrency") or 1),
                gpu_layers=ngl,
            )
        except Exception:
            return Candidate(
                model=model,
                score=0,
                components={},
                loaded=loaded,
                feasible=False,
                reason="metadata unavailable",
                rejection="metadata unavailable",
            )
        ram_need = int(estimate["ram_bytes"])
        vram_need = int(estimate.get("vram_bytes", 0))
        feasible = loaded or (
            ram_need <= memory["ram_bytes"] + evictable_ram
            and vram_need <= memory["vram_bytes"] + evictable_vram
        )
        rejection = None
        if not feasible:
            rejection = (
                f"insufficient memory: needs {ram_need / 1024**3:.2f} GiB, "
                f"usable {memory['ram_bytes'] / 1024**3:.2f} GiB"
            )
        capability = self._capability_score(model, classification, role)
        quality = self._quality_score(model, classification)
        latency, load_ms = self._latency_score(model, loaded, runtime)
        resource = self._resource_score(ram_need, memory["ram_bytes"], loaded)
        queue = 100.0
        if runtime and runtime.gate:
            snap = runtime.gate.snapshot()
            queue = max(0.0, 100 - snap["in_use"] * 18 - snap["waiting"] * 12)
        priority = float(model.get("priority", 50))
        components = {
            "capability": capability,
            "quality": quality,
            "latency": latency,
            "resource": resource,
            "queue": queue,
            "priority": priority,
            "loaded_bonus": 6.0 if loaded and self.config.prioritize_loaded else 0.0,
        }
        score = _weighted(components, role or self.config.strategy, self.config.prioritize_loaded)
        estimate["estimated_load_ms"] = load_ms
        reason = _candidate_reason(
            model, loaded, capability, quality, latency, resource, queue, load_ms
        )
        return Candidate(
            model=model,
            score=score,
            components=components,
            loaded=loaded,
            feasible=feasible,
            reason=reason,
            estimate=estimate,
            queue_depth=(runtime.waiting if runtime else 0),
            rejection=rejection,
        )

    @staticmethod
    def _capability_score(
        model: dict[str, Any], classification: TaskClassification, role: str | None
    ) -> float:
        caps = set(model.get("capabilities", []))
        category = classification.category
        if role == "coding":
            category = "coding"
        if role == "reasoning":
            category = "reasoning"
        required = TASK_CAPABILITY.get(category, "chat")
        if required in caps:
            return 100.0
        if category in ("coding", "code_generation", "debugging") and caps & {
            "coding",
            "code_generation",
            "debugging",
        }:
            return 90.0
        if category in ("reasoning", "math") and "reasoning" in caps:
            return 82.0
        if "chat" in caps:
            return 32.0 if category != "general" else 90.0
        if "general" in caps:
            return 25.0
        return 0.0

    @staticmethod
    def _quality_score(model: dict[str, Any], classification: TaskClassification) -> float:
        count = int(model.get("parameter_count") or 0)
        if not count:
            size = str(model.get("parameter_size") or "")
            m = size.rstrip("B")
            try:
                count = int(float(m) * 1e9)
            except ValueError:
                count = 3_000_000_000
        quality = 18 + 13 * math.log10(max(1, count / 1e9) + 1) * 1.18
        quality = min(96.0, max(10.0, quality))
        quant = str(model.get("quantization") or "").upper()
        if "Q2" in quant or "Q3" in quant or "IQ1" in quant or "IQ2" in quant:
            quality -= 7
        elif "Q4" in quant:
            quality -= 1
        elif "Q8" in quant or "F16" in quant or "BF16" in quant:
            quality += 3
        caps = set(model.get("capabilities", []))
        if classification.category in ("coding", "debugging", "code_generation"):
            if caps & {"coding", "code_generation", "debugging"}:
                quality += 6
        if classification.category in ("reasoning", "math") and "reasoning" in caps:
            quality += 6
        return max(0.0, min(100.0, quality))

    def _latency_score(
        self, model: dict[str, Any], loaded: bool, runtime: Any
    ) -> tuple[float, int]:
        perf = self.db.get_perf(model["id"])
        if perf and perf.get("avg_tps"):
            tps = float(perf["avg_tps"])
            ttft = float(perf.get("avg_ttft_ms") or 400)
            load_ms = 0 if loaded else 800
        else:
            count = float(model.get("parameter_count") or 3e9)
            size_b = count / 1e9
            backend = self.monitor.hardware.llama_backend
            if backend == "cuda":
                tps = max(4.0, 65.0 / max(1.0, size_b))
                ttft = 180
            elif backend == "metal":
                tps = max(3.0, 45.0 / max(1.0, size_b))
                ttft = 240
            else:
                tps = max(1.2, 62.0 / (size_b**1.22))
                ttft = 300 + int((model.get("size_bytes") or 0) / 1024**3 * 260)
            load_ms = 0 if loaded else int(900 + (model.get("size_bytes") or 0) / 1024**2 * 0.35)
        latency = min(100.0, (tps / 60.0) * 55 + max(0.0, 1 - ttft / 2500.0) * 45)
        if loaded:
            latency = min(100.0, latency + 8)
        else:
            latency = max(0.0, latency - min(25.0, load_ms / 300.0))
        if runtime:
            snap = runtime.gate.snapshot() if runtime.gate else {}
            latency = max(0.0, latency - snap.get("waiting", 0) * 6)
        return latency, load_ms

    @staticmethod
    def _resource_score(need: int, available: int, loaded: bool) -> float:
        if loaded:
            return 100.0
        if available <= 0:
            return 0.0
        ratio = need / max(1, available)
        return max(0.0, min(100.0, 100.0 - ratio * 85.0))

    def _explain(
        self,
        selected: Candidate,
        classification: TaskClassification,
        role: str | None,
        candidates: list[Candidate],
    ) -> str:
        m = selected.model
        parts = [
            f"Selected {m['id']} for {TASK_LABELS.get(classification.category, classification.category).lower()}",
            "because it " + selected.reason,
        ]
        if role:
            parts[0] += f" via the '{role}' role alias"
        runner_up = candidates[1] if len(candidates) > 1 else None
        if runner_up:
            parts.append(
                f"Runner-up {runner_up.model['id']} scored {runner_up.score:.1f} "
                f"vs {selected.score:.1f}"
            )
        if selected.loaded:
            parts.append("and it is already resident, avoiding load overhead")
        parts.append(
            f"(classifier confidence {classification.confidence:.2f}, "
            f"context need ~{classification.estimated_tokens} tokens)"
        )
        return "; ".join(parts) + "."

    def _raise_no_candidate(
        self,
        classification: TaskClassification,
        required: int,
        excluded: list[dict[str, str]],
        models: list[dict[str, Any]],
    ) -> None:
        if not models:
            raise RoutingError(
                "No enabled models are registered. Add a GGUF model in the dashboard.",
                status_code=503,
                code="no_models",
            )
        context_rejections = [e for e in excluded if "context" in e["reason"]]
        if context_rejections and len(context_rejections) == len(excluded):
            raise RoutingError(
                f"Request needs ~{required} tokens; no enabled model has sufficient context.",
                status_code=400,
                code="context_length_exceeded",
            )
        raise RoutingError(
            "No model is currently feasible for this request. "
            + (excluded[0]["reason"] if excluded else "Check model availability and resources."),
            status_code=503,
            code="no_feasible_model",
        )

    async def _llm_classify(
        self,
        classification: TaskClassification,
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
    ) -> TaskClassification:
        model_id = self.config.llm_model
        runtime = self.manager.runtimes.get(model_id) if model_id else None
        if not runtime or not runtime.is_loaded() or not runtime.gate:
            # Never load a model just to classify; that would defeat the point of
            # lightweight routing and could evict the model we are about to use.
            loaded = [
                rt
                for rt in self.manager.runtimes.values()
                if rt.is_loaded() and rt.gate is not None
            ]
            if not loaded:
                return classification
            loaded.sort(key=lambda rt: int(self.registry.get(rt.model_id).get("size_bytes", 0)))
            runtime = loaded[0]
        text = prompt or ""
        if messages:
            from .classifier import flatten_messages

            text = flatten_messages(messages)[1] or text
        if not text:
            return classification
        started = time.perf_counter()
        lease = None
        try:
            lease = await runtime.gate.acquire(priority=9, key_id="router", timeout=0.8)
            import httpx

            async with httpx.AsyncClient(timeout=self.config.llm_timeout_ms / 1000) as client:
                resp = await client.post(
                    f"{runtime.backend.base_url}/v1/chat/completions",
                    json={
                        "model": runtime.model_id,
                        "messages": [
                            {"role": "user", "content": build_llm_prompt(classification, text)}
                        ],
                        "max_tokens": 48,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "response_format": {"type": "json_object"},
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                if resp.status_code != 200:
                    return classification
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content") or ""
                )
            refined = parse_llm_classification(content, classification)
            if refined:
                refined.signals = refined.signals + [
                    f"router model {runtime.model_id} ({int((time.perf_counter() - started) * 1000)} ms)"
                ]
                return refined
        except Exception:
            return classification
        finally:
            if lease:
                lease.release()
        return classification

    def _record(self, decision: RoutingDecision) -> None:
        try:
            self.db.add_routing_history(
                request_id=decision.request_id or f"route-{int(time.time() * 1000)}-{id(decision) % 10000}",
                task=decision.task,
                confidence=decision.confidence,
                selected_model=decision.selected_model,
                explanation=decision.explanation,
                alternatives=decision.alternatives,
                constraints=decision.constraints,
                fallback=decision.fallback_used,
                routing_ms=decision.routing_ms,
            )
        except Exception:  # nosec B110 - routing history is diagnostic, never fatal
            pass


def _apply_role(classification: TaskClassification, role: str) -> TaskClassification:
    if role == "coding":
        category = "coding"
    elif role == "reasoning":
        category = "reasoning"
    else:
        return classification
    scores = dict(classification.scores)
    scores[category] = max(scores.get(category, 0), 6.0)
    return TaskClassification(
        category=category,
        confidence=1.0,
        scores=scores,
        signals=classification.signals + [f"explicit role alias '{role}'"],
        language=classification.language,
        estimated_tokens=classification.estimated_tokens,
        complexity=classification.complexity,
        long_context=classification.long_context,
        needs_json=classification.needs_json,
        needs_tools=classification.needs_tools,
        needs_vision=classification.needs_vision,
        ambiguous=False,
        source="role_alias",
        rationale=f"Role alias '{role}' requested explicit task specialization.",
        alternatives=classification.alternatives,
    )


def _weighted(components: dict[str, float], strategy: str, prioritize_loaded: bool) -> float:
    if strategy == "coding":
        weights = {
            "capability": 0.40,
            "quality": 0.25,
            "latency": 0.08,
            "resource": 0.12,
            "queue": 0.10,
            "priority": 0.05,
            "loaded_bonus": 0.0,
        }
    elif strategy == "reasoning":
        weights = {
            "capability": 0.32,
            "quality": 0.34,
            "latency": 0.08,
            "resource": 0.12,
            "queue": 0.08,
            "priority": 0.06,
            "loaded_bonus": 0.0,
        }
    elif strategy == "fast":
        weights = {
            "capability": 0.12,
            "quality": 0.08,
            "latency": 0.56,
            "resource": 0.12,
            "queue": 0.07,
            "priority": 0.05,
            "loaded_bonus": 0.0,
        }
        score = sum(components[k] * w for k, w in weights.items())
        if prioritize_loaded:
            score += components.get("loaded_bonus", 0.0)
        return score
    elif strategy == "quality":
        weights = {
            "capability": 0.38,
            "quality": 0.32,
            "latency": 0.10,
            "resource": 0.08,
            "queue": 0.06,
            "priority": 0.06,
            "loaded_bonus": 0.0,
        }
    elif strategy == "latency":
        weights = {
            "capability": 0.22,
            "quality": 0.10,
            "latency": 0.42,
            "resource": 0.12,
            "queue": 0.09,
            "priority": 0.05,
            "loaded_bonus": 0.0,
        }
    elif strategy == "local":
        weights = {
            "capability": 0.25,
            "quality": 0.10,
            "latency": 0.20,
            "resource": 0.30,
            "queue": 0.10,
            "priority": 0.05,
            "loaded_bonus": 0.0,
        }
    else:
        weights = {
            "capability": 0.36,
            "quality": 0.18,
            "latency": 0.18,
            "resource": 0.14,
            "queue": 0.08,
            "priority": 0.06,
            "loaded_bonus": 0.0,
        }
    score = sum(components[k] * w for k, w in weights.items())
    if prioritize_loaded:
        score += components.get("loaded_bonus", 0.0)
    return score


def _candidate_reason(
    model: dict[str, Any],
    loaded: bool,
    capability: float,
    quality: float,
    latency: float,
    resource: float,
    queue: float,
    load_ms: int,
) -> str:
    bits = []
    bits.append("matches the task capability" if capability >= 50 else "is a general fallback")
    bits.append(f"quality {quality:.0f}/100")
    bits.append("is already loaded" if loaded else f"cold load ~{load_ms / 1000:.1f}s")
    bits.append(f"latency score {latency:.0f}")
    bits.append(f"memory fit {resource:.0f}")
    if queue < 60:
        bits.append("has queued work")
    return ", ".join(bits)


def _meta_placeholder() -> GGUFMetadata:  # pragma: no cover - typing helper
    raise NotImplementedError
