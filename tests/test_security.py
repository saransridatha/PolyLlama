from __future__ import annotations

import asyncio

import pytest

from app.db import Database
from app.security import (
    AuthContext,
    AuthError,
    KeyService,
    RateLimitError,
    RateLimiter,
    lookup_fingerprint,
    verify_hashed_key,
)


def key_service(tmp_path):
    db = Database(tmp_path / "keys.db")
    return db, KeyService(db, "test-pepper")


def test_key_is_hashed_and_authenticates(tmp_path):
    db, service = key_service(tmp_path)
    record, raw = service.create_key(name="ci", role="inference")
    assert raw.startswith("msk_")
    assert raw not in str(record)
    stored = db.query_one("SELECT * FROM api_keys WHERE id=?", (record["id"],))
    assert raw not in stored["key_hash"]
    assert raw not in stored["lookup_hash"]
    assert stored["lookup_hash"] == lookup_fingerprint(raw, "test-pepper")
    assert verify_hashed_key(raw, stored["key_hash"])
    assert not verify_hashed_key(raw + "x", stored["key_hash"])
    ctx = service.authenticate(raw)
    assert ctx.key_id == record["id"]
    assert ctx.role == "inference"
    db.close()


def test_correct_prefix_but_wrong_key_is_rejected(tmp_path):
    db, service = key_service(tmp_path)
    _, raw = service.create_key(name="ci", role="inference")
    with pytest.raises(AuthError):
        service.authenticate(raw[:-2] + "AA")
    with pytest.raises(AuthError):
        service.authenticate("")
    with pytest.raises(AuthError):
        service.authenticate("sk-not-a-modelswitch-key")
    db.close()


def test_revocation_is_immediate_even_with_verification_cache(tmp_path):
    db, service = key_service(tmp_path)
    record, raw = service.create_key(name="ci", role="inference")
    assert service.authenticate(raw).key_id == record["id"]
    db.revoke_api_key(record["id"])
    with pytest.raises(AuthError, match="revoked"):
        service.authenticate(raw)
    db.close()


def test_expired_key_is_rejected(tmp_path):
    db, service = key_service(tmp_path)
    _, raw = service.create_key(
        name="expired", role="inference", expires_at="2020-01-01T00:00:00+00:00"
    )
    with pytest.raises(AuthError, match="expired"):
        service.authenticate(raw)
    db.close()


def test_rotation_revokes_old_key(tmp_path):
    db, service = key_service(tmp_path)
    record, raw = service.create_key(name="ci", role="inference")
    new_record, new_raw = service.rotate(record["id"])
    assert new_record["id"] != record["id"]
    assert new_raw != raw
    with pytest.raises(AuthError):
        service.authenticate(raw)
    assert service.authenticate(new_raw).key_id == new_record["id"]
    db.close()


def test_permissions_are_applied():
    admin = AuthContext(key_id="a", name="a", role="admin")
    assert admin.allows_model("anything")
    assert admin.allowed_models() is None
    inference = AuthContext(
        key_id="i",
        name="i",
        role="inference",
        permissions={"models": ["qwen"], "allow_streaming": False, "max_tokens": 32},
    )
    assert inference.allows_model("qwen")
    assert not inference.allows_model("other")
    assert inference.allowed_models() == ["qwen"]
    assert inference.allow_streaming() is False
    assert inference.max_tokens_cap() == 32


@pytest.mark.asyncio
async def test_rate_limiter_blocks_requests():
    limiter = RateLimiter()
    ctx = AuthContext(key_id="k", name="k", role="inference", rate_limit_rpm=2, rate_limit_tpm=1000)
    await limiter.check(ctx, 10)
    await limiter.check(ctx, 10)
    with pytest.raises(RateLimitError) as exc:
        await limiter.check(ctx, 10)
    assert exc.value.retry_after >= 1


@pytest.mark.asyncio
async def test_rate_limiter_blocks_tokens():
    limiter = RateLimiter()
    ctx = AuthContext(key_id="k", name="k", role="inference", rate_limit_rpm=100, rate_limit_tpm=100)
    with pytest.raises(RateLimitError, match="Token"):
        await limiter.check(ctx, 101)


@pytest.mark.asyncio
async def test_auth_failure_brake():
    limiter = RateLimiter()
    for _ in range(20):
        await limiter.register_auth_failure("1.2.3.4")
    with pytest.raises(RateLimitError):
        await limiter.register_auth_failure("1.2.3.4")
    assert await limiter.auth_failure_allowed("5.6.7.8")
