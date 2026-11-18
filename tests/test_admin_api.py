from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import TEST_MODEL, model_available


@pytest.mark.asyncio
async def test_dashboard_index_served(admin_client, runtime):
    index = runtime.settings.dashboard_dir / "index.html"
    if not index.is_file():
        pytest.skip("dashboard not built")
    response = await admin_client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ModelSwitch" in response.text
    assert "content-security-policy" in response.headers


@pytest.mark.asyncio
async def test_info_endpoint(admin_client, keys):
    response = await admin_client.get("/admin/api/info", headers=keys["admin_headers"])
    assert response.status_code == 200
    data = response.json()
    assert data["public_api_base"].endswith("/v1")
    assert isinstance(data["dashboard_built"], bool)


@pytest.mark.asyncio
async def test_create_and_revoke_key(admin_client, keys, runtime):
    response = await admin_client.post(
        "/admin/api/keys",
        headers=keys["admin_headers"],
        json={
            "name": "integration-key",
            "role": "inference",
            "permissions": {"models": ["*"], "allow_streaming": True},
            "rate_limit_rpm": 10,
            "rate_limit_tpm": 5000,
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["api_key"].startswith("msk_")
    assert "key_hash" not in payload["key"]
    key_id = payload["key"]["id"]

    listing = await admin_client.get("/admin/api/keys", headers=keys["admin_headers"])
    assert listing.status_code == 200
    assert all("key_hash" not in item and "lookup_hash" not in item for item in listing.json()["keys"])

    revoke = await admin_client.delete(
        f"/admin/api/keys/{key_id}", headers=keys["admin_headers"]
    )
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True


@pytest.mark.asyncio
async def test_rotate_key(admin_client, keys, runtime):
    _, raw = runtime.keys.create_key(name="rotate-me", role="inference")
    key = next(k for k in runtime.db.list_api_keys() if k["name"] == "rotate-me")
    response = await admin_client.post(
        f"/admin/api/keys/{key['id']}/rotate", headers=keys["admin_headers"]
    )
    assert response.status_code == 200
    new_raw = response.json()["api_key"]
    assert new_raw != raw
    # Old key must now fail authentication
    public = await admin_client.get(
        "/admin/api/overview", headers={"Authorization": f"Bearer {raw}"}
    )
    assert public.status_code in (401, 403)


@pytest.mark.asyncio
async def test_cannot_revoke_last_admin(admin_client, keys, runtime):
    admin_id = keys["admin"][0]["id"]
    response = await admin_client.delete(
        f"/admin/api/keys/{admin_id}", headers=keys["admin_headers"]
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "last_admin_key"


@pytest.mark.asyncio
async def test_invalid_key_permissions_rejected(admin_client, keys):
    response = await admin_client.post(
        "/admin/api/keys",
        headers=keys["admin_headers"],
        json={"name": "bad", "role": "inference", "permissions": {"root": True}},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_admin_model_path_error_is_400_not_500(admin_client, keys):
    response = await admin_client.post(
        "/admin/api/models",
        headers=keys["admin_headers"],
        json={"path": "/etc/passwd"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model"


@pytest.mark.asyncio
async def test_model_register_and_delete(admin_client, keys, runtime):
    if not model_available():
        pytest.skip("model unavailable")
    response = await admin_client.post(
        "/admin/api/models",
        headers=keys["admin_headers"],
        json={"path": str(TEST_MODEL), "name": "Admin Test Model", "context_length": 2048},
    )
    assert response.status_code == 201
    model_id = response.json()["id"]
    listing = await admin_client.get("/admin/api/models", headers=keys["admin_headers"])
    assert any(m["id"] == model_id for m in listing.json()["models"])
    delete = await admin_client.delete(
        f"/admin/api/models/{model_id}", headers=keys["admin_headers"]
    )
    assert delete.status_code == 200
    assert TEST_MODEL.exists()


@pytest.mark.asyncio
async def test_routing_config_update(admin_client, keys):
    response = await admin_client.patch(
        "/admin/api/routing/config",
        headers=keys["admin_headers"],
        json={"strategy": "latency", "fallback_enabled": True},
    )
    assert response.status_code == 200
    assert response.json()["config"]["strategy"] == "latency"


@pytest.mark.asyncio
async def test_routing_invalid_config(admin_client, keys):
    response = await admin_client.patch(
        "/admin/api/routing/config",
        headers=keys["admin_headers"],
        json={"strategy": "nonsense"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_remote_status_is_admin_only(admin_client, keys):
    assert (await admin_client.get("/admin/api/remote")).status_code == 401
    response = await admin_client.get("/admin/api/remote", headers=keys["admin_headers"])
    assert response.status_code == 200
    payload = response.json()
    assert payload["admin_exposed"] is False
    assert "installed" in payload


@pytest.mark.asyncio
async def test_remote_rejects_http_public_url(admin_client, keys):
    response = await admin_client.post(
        "/admin/api/remote/start",
        headers=keys["admin_headers"],
        json={"mode": "token", "token": "x" * 40, "public_url": "http://localhost:8000"},
    )
    assert response.status_code in (400, 503)
    assert response.json()["error"]["type"] == "remote_access_error"


@pytest.mark.asyncio
async def test_metrics_requires_admin(admin_client, keys):
    assert (await admin_client.get("/metrics")).status_code == 401
    response = await admin_client.get("/metrics", headers=keys["admin_headers"])
    assert response.status_code == 200
    assert "modelswitch_" in response.text
    assert "modelswitch_models_loaded" in response.text


@pytest.mark.asyncio
async def test_request_logs_are_admin_only(admin_client, keys):
    response = await admin_client.get(
        "/admin/api/logs/requests", headers=keys["admin_headers"]
    )
    assert response.status_code == 200
    assert "requests" in response.json()
    events = await admin_client.get(
        "/admin/api/logs/events", headers=keys["admin_headers"]
    )
    assert events.status_code == 200


@pytest.mark.asyncio
async def test_diagnostics(admin_client, keys):
    response = await admin_client.get("/admin/api/diagnostics", headers=keys["admin_headers"])
    assert response.status_code == 200
    assert "readiness" in response.json()
    assert "settings" in response.json()
