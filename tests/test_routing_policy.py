from __future__ import annotations

from pathlib import Path

import pytest

from app.router.router import IntelligentRouter, RoutingError
from tests.conftest import TEST_CODER_MODEL, TEST_EMBED_MODEL, TEST_MODEL, model_available


@pytest.mark.asyncio
async def test_policy_routes_code_to_coder(runtime):
    if not model_available() or not model_available(TEST_CODER_MODEL):
        pytest.skip("routing fixtures unavailable")
    general = runtime.registry.add_model(path=str(TEST_MODEL), context_length=2048, priority=60)
    coder = runtime.registry.add_model(
        path=str(TEST_CODER_MODEL),
        name="Coder",
        context_length=2048,
        priority=70,
        capabilities=["chat", "coding", "code_generation", "debugging", "json"],
    )
    decision = await runtime.router.route(
        requested_model="auto",
        messages=[{"role": "user", "content": "Write a Rust function to parse a config file."}],
        max_tokens=64,
    )
    assert decision.task in ("code_generation", "coding")
    assert decision.selected_model == coder["id"]
    assert decision.alternatives
    assert decision.routing_ms >= 0
    assert decision.constraints["required_context_tokens"] >= 64


@pytest.mark.asyncio
async def test_policy_respects_model_permissions(runtime):
    if not model_available():
        pytest.skip("model unavailable")
    model = runtime.registry.add_model(path=str(TEST_MODEL), context_length=2048)
    with pytest.raises(RoutingError) as exc:
        await runtime.router.route(
            requested_model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            allowed_models=["different-model"],
        )
    assert exc.value.status_code in (403, 503)


@pytest.mark.asyncio
async def test_explicit_decision_and_context_validation(runtime):
    if not model_available():
        pytest.skip("model unavailable")
    model = runtime.registry.add_model(path=str(TEST_MODEL), context_length=2048)
    decision = await runtime.router.route(
        requested_model=model["id"],
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8,
    )
    assert decision.strategy == "explicit"
    assert decision.selected_model == model["id"]
    with pytest.raises(RoutingError):
        await runtime.router.route(
            requested_model=model["id"],
            messages=[{"role": "user", "content": "word " * 8000}],
            max_tokens=8,
        )


@pytest.mark.asyncio
async def test_fast_alias_prefers_resident(runtime):
    if not model_available():
        pytest.skip("model unavailable")
    model = runtime.registry.add_model(
        path=str(TEST_MODEL), context_length=2048, max_concurrency=1, priority=50
    )
    # Load is not required for a decision; just ensure the alias is explainable.
    decision = await runtime.router.route(
        requested_model="fast",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=8,
    )
    assert decision.resolved_alias == "fast"
    assert decision.selected_model == model["id"]


@pytest.mark.asyncio
async def test_disabled_model_is_not_routed(runtime):
    if not model_available():
        pytest.skip("model unavailable")
    model = runtime.registry.add_model(path=str(TEST_MODEL), context_length=2048, enabled=False)
    with pytest.raises(RoutingError) as exc:
        await runtime.router.route(
            requested_model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=8,
        )
    assert exc.value.code == "no_models"


@pytest.mark.asyncio
async def test_embedding_only_model_excluded_from_chat(runtime):
    if not model_available(TEST_EMBED_MODEL):
        pytest.skip("embedding model unavailable")
    embed = runtime.registry.add_model(
        path=str(TEST_EMBED_MODEL),
        context_length=2048,
        capabilities=["embeddings"],
    )
    with pytest.raises(RoutingError) as exc:
        await runtime.router.route(
            requested_model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=8,
        )
    assert exc.value.code == "no_feasible_model"


@pytest.mark.asyncio
async def test_router_config_updates(runtime):
    config = runtime.router.update_config({"strategy": "quality", "llm_enabled": False})
    assert config.strategy == "quality"
    assert runtime.db.get_setting("router_config")["strategy"] == "quality"
    with pytest.raises(ValueError):
        runtime.router.update_config({"strategy": "bogus"})
    with pytest.raises(ValueError):
        runtime.router.update_config({"llm_model": "missing-model"})


@pytest.mark.asyncio
async def test_router_history_recorded(runtime):
    if not model_available():
        pytest.skip("model unavailable")
    runtime.registry.add_model(path=str(TEST_MODEL), context_length=2048)
    await runtime.router.route(
        requested_model="auto",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8,
        request_id="req-unit-test",
    )
    history = runtime.db.list_routing_history(10)
    assert any(item["request_id"] == "req-unit-test" for item in history)
