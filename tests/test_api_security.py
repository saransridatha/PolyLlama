from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.api.public import create_public_app
from app.config import Settings


@pytest.mark.asyncio
async def test_health_is_public(public_client):
    response = await public_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-request-id"].startswith("req_")


@pytest.mark.asyncio
async def test_models_requires_auth(public_client):
    response = await public_client.get("/v1/models")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_invalid_key_rejected(public_client):
    response = await public_client.get(
        "/v1/models", headers={"Authorization": "Bearer msk_invalid_invalid_invalid"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_basic_auth_is_rejected(public_client, keys):
    response = await public_client.get(
        "/v1/models", headers={"Authorization": "Basic dXNlcjpwYXNz"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_x_api_key_header_supported(public_client, keys):
    response = await public_client.get(
        "/v1/models", headers={"X-API-Key": keys["inference"][1]}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_models_list_hides_paths(public_client, keys):
    response = await public_client.get("/v1/models", headers=keys["inference_headers"])
    assert response.status_code == 200
    data = response.json()
    ids = [item["id"] for item in data["data"]]
    assert "auto" in ids
    assert "fast" in ids
    for item in data["data"]:
        assert "path" not in item
        assert "key_hash" not in item


@pytest.mark.asyncio
async def test_admin_routes_are_not_on_public_listener(public_client):
    for path in ("/admin", "/admin/api/overview", "/admin/api/keys"):
        response = await public_client.get(path)
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_admin_key_rate_limit(admin_client, runtime):
    _, raw = runtime.keys.create_key(
        name="slow-admin", role="admin", rate_limit_rpm=1, rate_limit_tpm=1000
    )
    headers = {"Authorization": f"Bearer {raw}"}
    first = await admin_client.get("/admin/api/info", headers=headers)
    second = await admin_client.get("/admin/api/info", headers=headers)
    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.asyncio
async def test_admin_app_rejects_inference_role(admin_client, keys):
    response = await admin_client.get("/admin/api/overview", headers=keys["inference_headers"])
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_app_accepts_admin_role(admin_client, keys):
    response = await admin_client.get("/admin/api/overview", headers=keys["admin_headers"])
    assert response.status_code == 200
    payload = response.json()
    assert "stats" in payload
    assert "resources" in payload


@pytest.mark.asyncio
async def test_invalid_message_role_is_400(public_client, keys):
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={"model": "auto", "messages": [{"role": "hacker", "content": "hello"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_empty_messages_is_400(public_client, keys):
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={"model": "auto", "messages": []},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_missing_model_returns_openai_error(public_client, keys):
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_oversized_body_is_413(public_client, keys):
    huge = "x" * (3 * 1024 * 1024)
    response = await public_client.post(
        "/v1/chat/completions",
        headers=keys["inference_headers"],
        json={"model": "auto", "messages": [{"role": "user", "content": huge}]},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.asyncio
async def test_rate_limit_enforced_per_key(runtime, public_client):
    _, raw = runtime.keys.create_key(
        name="limited", role="inference", rate_limit_rpm=1, rate_limit_tpm=100_000
    )
    headers = {"Authorization": f"Bearer {raw}"}
    first = await public_client.get("/v1/models", headers=headers)
    second = await public_client.get("/v1/models", headers=headers)
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"]


@pytest.mark.asyncio
async def test_no_cors_header_when_disabled(public_client, keys):
    response = await public_client.get(
        "/v1/models",
        headers={**keys["inference_headers"], "Origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.asyncio
async def test_raw_key_never_logged(public_client, keys, caplog):
    caplog.set_level(logging.DEBUG)
    raw = keys["inference"][1]
    await public_client.get("/v1/models", headers={"Authorization": f"Bearer {raw}"})
    assert raw not in caplog.text


@pytest.mark.asyncio
async def test_admin_security_headers_and_csp(admin_client):
    response = await admin_client.get("/")
    assert response.status_code in (200, 307)
    headers = response.headers if response.status_code == 200 else response.next_request  # type: ignore[attr-defined]
    if response.status_code == 307:
        follow = await admin_client.get("/admin")
        assert "content-security-policy" in follow.headers
        assert follow.headers["x-frame-options"] == "DENY"


@pytest.mark.asyncio
async def test_cors_configured(tmp_path: Path):
    from app.api.public import create_public_app
    from app.runtime import AppRuntime

    settings = Settings(
        data_dir=tmp_path / "data",
        model_roots=str(tmp_path),
        allow_external_model_paths=True,
        cors_origins="https://app.example",
        default_context_length=2048,
    )
    settings.ensure_dirs()
    runtime = AppRuntime(settings)
    await runtime.start()
    try:
        _, raw = runtime.keys.create_key(name="c", role="inference")
        import httpx

        app = create_public_app(runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {raw}", "Origin": "https://app.example"},
            )
            assert response.headers["access-control-allow-origin"] == "https://app.example"
    finally:
        await runtime.close()
