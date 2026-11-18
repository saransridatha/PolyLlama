from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from app.lifecycle import ModelUnavailable
from tests.conftest import TEST_CODER_MODEL, TEST_EMBED_MODEL, model_available


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_auto_end_to_end(public_client, keys, loaded_model):
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Reply with the word READY."}],
            "max_tokens": 16,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["model"] == loaded_model["id"]
    assert payload["usage"]["total_tokens"] >= payload["usage"]["prompt_tokens"]
    assert payload["modelswitch"]["routing"]["selected_model"] == loaded_model["id"]
    assert response.headers["x-modelswitch-model"] == loaded_model["id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explicit_model_bypasses_router(public_client, keys, loaded_model):
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": loaded_model["id"],
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 8,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == loaded_model["id"]
    assert payload["modelswitch"]["routing"]["strategy"] == "explicit"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_sse(public_client, keys, loaded_model):
    chunks = []
    async with public_client.stream(
        "POST",
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": "fast",
            "messages": [{"role": "user", "content": "Count 1 to 4."}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        async for line in response.aiter_lines():
            if line:
                chunks.append(line)
    assert chunks[-1] == "data: [DONE]"
    events = []
    for line in chunks:
        if line.startswith("data: "):
            body = line[6:]
            if body != "[DONE]":
                events.append(json.loads(body))
    usage_events = [event for event in events if event.get("usage")]
    assert usage_events
    assert usage_events[-1]["usage"]["completion_tokens"] >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_alias_is_explicit(public_client, keys, loaded_model):
    alias = loaded_model["aliases"][0]
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": alias,
            "messages": [{"role": "user", "content": "Say alias."}],
            "max_tokens": 6,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["model"] == loaded_model["id"]
    assert response.json()["modelswitch"]["routing"]["strategy"] == "explicit"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embeddings_end_to_end(public_client, keys, runtime):
    if not model_available(TEST_EMBED_MODEL):
        pytest.skip("embedding GGUF unavailable")
    model = runtime.registry.add_model(
        path=str(TEST_EMBED_MODEL),
        name="Test Embeddings",
        context_length=2048,
        max_concurrency=1,
        capabilities=["embeddings"],
    )
    response = await public_client.post(
        "/v1/embeddings",
        headers=keys["inference_headers"],
        json={"model": "auto", "input": ["local routing", "hardware aware"]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == model["id"]
    assert len(payload["data"]) == 2
    assert len(payload["data"][0]["embedding"]) > 16
    assert payload["usage"]["prompt_tokens"] > 0
    await runtime.manager.unload(model["id"], force=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_model_cannot_handle_chat(public_client, keys, runtime, loaded_model):
    if not model_available(TEST_EMBED_MODEL):
        pytest.skip("embedding GGUF unavailable")
    embed = runtime.registry.add_model(
        path=str(TEST_EMBED_MODEL),
        name="Embed Only",
        context_length=2048,
        max_concurrency=1,
        capabilities=["embeddings"],
    )
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={"model": embed["id"], "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "not_a_chat_model"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embeddings_rejected_for_chat_model(public_client, keys, loaded_model):
    response = await public_client.post(
        "/v1/embeddings",
        headers=keys["inference_headers"],
        json={"model": loaded_model["id"], "input": "embed this"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "embeddings_not_supported"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completions_endpoint(public_client, keys, loaded_model):
    response = await public_client.post(
        "/v1/completions",
        headers=keys["inference_headers"],
        json={
            "model": "auto",
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests(public_client, keys, loaded_model):
    async def one(index: int):
        started = time.perf_counter()
        response = await public_client.post(
            "/v1/chat/completions",
            headers=keys["inference_headers"],
            json={
                "model": loaded_model["id"],
                "messages": [{"role": "user", "content": f"Reply with the number {index}."}],
                "max_tokens": 12,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.status_code, (time.perf_counter() - started) * 1000

    results = await asyncio.gather(*(one(i) for i in range(4)))
    assert all(status == 200 for status, _ in results)
    assert max(duration for _, duration in results) < 60_000


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_limit_rejected(public_client, keys, loaded_model):
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": loaded_model["id"],
            "messages": [{"role": "user", "content": "word " * 8000}],
            "max_tokens": 64,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] in ("model_unavailable", "context_length_exceeded")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_auto_routing_picks_coding_specialist(public_client, keys, runtime):
    if not model_available(TEST_CODER_MODEL):
        pytest.skip("coder model unavailable")
    general = runtime.registry.add_model(
        path=str(loaded_model_path()), context_length=2048, priority=60
    )
    coder = runtime.registry.add_model(
        path=str(TEST_CODER_MODEL),
        name="Coder 7B",
        context_length=2048,
        priority=70,
        capabilities=["chat", "general", "coding", "code_generation", "debugging", "json"],
    )
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python function to reverse a linked list. Return code only.",
                }
            ],
            "max_tokens": 24,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["modelswitch"]["routing"]["task"] == "code_generation"
    assert payload["modelswitch"]["routing"]["selected_model"] == coder["id"]
    await runtime.manager.unload(coder["id"], force=True)
    await runtime.manager.unload(general["id"], force=True)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upstream_timeout_is_504(public_client, keys, loaded_model):
    warm = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": loaded_model["id"],
            "messages": [{"role": "user", "content": "Say warm."}],
            "max_tokens": 4,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert warm.status_code == 200
    runtime = public_client._transport.app.state.runtime  # type: ignore[attr-defined]
    old = runtime.settings.request_timeout
    runtime.settings.request_timeout = 0.01
    try:
        response = await public_client.post(
            "/v1/chat/completions",
            headers=keys["inference_headers"],
            json={
                "model": loaded_model["id"],
                "messages": [{"role": "user", "content": "Write a long essay about networking."}],
                "max_tokens": 128,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    finally:
        runtime.settings.request_timeout = old
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "backend_timeout"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_classifier_arbitration(runtime, keys, loaded_model):
    await runtime.manager.load(loaded_model["id"])
    runtime.router.update_config({"llm_enabled": True, "llm_model": loaded_model["id"]})
    from app.router.classifier import classify_task

    base = classify_task(
        messages=[{"role": "user", "content": "Can you make a small utility for this?"}]
    )
    refined = await runtime.router._llm_classify(
        base, [{"role": "user", "content": "Write a Python function to reverse a string."}], None
    )
    assert refined.source == "llm"
    assert refined.category in ("coding", "code_generation", "debugging")
    runtime.router.update_config({"llm_enabled": False})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backend_crash_recovery(public_client, keys, loaded_model):
    first = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": loaded_model["id"],
            "messages": [{"role": "user", "content": "Say one."}],
            "max_tokens": 4,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert first.status_code == 200
    runtime = None
    # Recover the runtime through the app closure.
    app = public_client._transport.app  # type: ignore[attr-defined]
    runtime = app.state.runtime
    rt = runtime.manager.runtimes[loaded_model["id"]]
    assert rt.backend and rt.backend.process
    rt.backend.process.kill()
    await rt.backend.process.wait()
    await asyncio.sleep(0.5)
    second = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={
            "model": loaded_model["id"],
            "messages": [{"role": "user", "content": "Say two."}],
            "max_tokens": 4,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert second.status_code == 200, second.text
    assert runtime.manager.runtimes[loaded_model["id"]].state == "loaded"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unload_refused_while_busy(runtime, keys, loaded_model):
    await runtime.manager.load(loaded_model["id"])
    _, lease = await runtime.manager.acquire(loaded_model["id"], key_id=keys["inference"][0]["id"])
    try:
        with pytest.raises(ModelUnavailable) as exc:
            await runtime.manager.unload(loaded_model["id"])
        assert exc.value.status_code == 409
    finally:
        lease.release()
    assert await runtime.manager.unload(loaded_model["id"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_python_sdk_compatibility(runtime, keys, loaded_model):
    import httpx2
    from openai import AsyncOpenAI

    from app.api.public import create_public_app

    app = create_public_app(runtime)
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://sdk.test"
    ) as http_client:
        client = AsyncOpenAI(
            api_key=keys["inference"][1],
            base_url="http://sdk.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        models = await client.models.list()
        assert any(model.id == "auto" for model in models.data)
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Reply READY."}],
            max_tokens=8,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        assert response.choices[0].message.content is not None
        assert response.usage and response.usage.total_tokens > 0


def loaded_model_path():
    from tests.conftest import TEST_MODEL

    return TEST_MODEL
