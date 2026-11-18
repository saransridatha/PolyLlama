"""Authentication, API keys, permissions, and abuse-resistant rate limiting.

Only two representations of a key are stored:

* a deterministic lookup hash (SHA-256 over an optional server pepper), allowing
  an indexed lookup without a plaintext key; and
* a salted scrypt hash used for constant-time verification.

The plaintext key exists only in the creation/rotation response.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .db import Database

KEY_PREFIX = "msk_"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


class AuthError(Exception):
    def __init__(self, message: str = "Invalid API key", status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


def _raw_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def _hash_key(raw: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        raw.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_hashed_key(raw: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = encoded.split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            raw.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def lookup_fingerprint(raw: str, pepper: str = "") -> str:
    return hashlib.sha256((pepper + raw).encode("utf-8")).hexdigest()


def key_prefix(raw: str) -> str:
    return raw[:12]


@dataclass(slots=True)
class AuthContext:
    key_id: str
    name: str
    role: str
    permissions: dict[str, Any] = field(default_factory=dict)
    prefix: str = ""
    rate_limit_rpm: int = 120
    rate_limit_tpm: int = 200_000

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def allows_model(self, model_id: str | None) -> bool:
        if self.is_admin:
            return True
        models = self.permissions.get("models", ["*"])
        if not models or "*" in models:
            return True
        return model_id in models if model_id else True

    def allowed_models(self) -> list[str] | None:
        if self.is_admin:
            return None
        models = self.permissions.get("models", ["*"])
        if not models or "*" in models:
            return None
        return list(models)

    def allow_streaming(self) -> bool:
        return self.is_admin or bool(self.permissions.get("allow_streaming", True))

    def max_tokens_cap(self) -> int | None:
        if self.is_admin:
            return None
        value = self.permissions.get("max_tokens")
        return int(value) if value else None

    def allow_embeddings(self) -> bool:
        return self.is_admin or bool(self.permissions.get("allow_embeddings", True))


class KeyService:
    def __init__(self, db: Database, pepper: str = ""):
        self.db = db
        self.pepper = pepper
        # The indexed SHA-256 lookup is always checked, so revocation and expiry
        # take effect immediately. Cache only the expensive scrypt verification
        # for a short window to keep per-request auth overhead sub-millisecond.
        self._verify_cache: dict[str, float] = {}
        self._verify_ttl = 300.0

    def create_key(
        self,
        *,
        name: str,
        role: str = "inference",
        permissions: dict[str, Any] | None = None,
        rate_limit_rpm: int = 120,
        rate_limit_tpm: int = 200_000,
        expires_at: str | None = None,
        rotated_from: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        if role not in ("admin", "inference"):
            raise ValueError("role must be admin or inference")
        raw = _raw_key()
        key_id = uuid.uuid4().hex
        record = self.db.create_api_key(
            key_id=key_id,
            name=name.strip() or "Unnamed key",
            role=role,
            key_hash=_hash_key(raw),
            lookup_hash=lookup_fingerprint(raw, self.pepper),
            prefix=key_prefix(raw),
            permissions=permissions or {},
            rate_limit_rpm=max(1, int(rate_limit_rpm)),
            rate_limit_tpm=max(100, int(rate_limit_tpm)),
            expires_at=expires_at,
            rotated_from=rotated_from,
        )
        return record, raw

    def authenticate(self, raw: str) -> AuthContext:
        if not raw or len(raw) < 16 or not raw.startswith(KEY_PREFIX):
            raise AuthError()
        record = self.db.find_api_key_by_lookup(lookup_fingerprint(raw, self.pepper))
        if not record:
            raise AuthError()
        if record.get("revoked_at"):
            raise AuthError("API key has been revoked")
        expires_at = record.get("expires_at")
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=timezone.utc)
                if expiry <= datetime.now(timezone.utc):
                    raise AuthError("API key has expired")
            except ValueError:
                raise AuthError("API key has an invalid expiry")
        now = time.time()
        cached_until = self._verify_cache.get(record["id"], 0)
        if cached_until < now:
            if not verify_hashed_key(raw, record["key_hash"]):
                raise AuthError()
            self._verify_cache[record["id"]] = now + self._verify_ttl
            if len(self._verify_cache) > 2048:
                self._verify_cache = {
                    key: expiry for key, expiry in self._verify_cache.items() if expiry > now
                }
        self.db.touch_api_key(record["id"])
        return AuthContext(
            key_id=record["id"],
            name=record["name"],
            role=record["role"],
            permissions=record.get("permissions") or {},
            prefix=record.get("prefix", ""),
            rate_limit_rpm=int(record.get("rate_limit_rpm") or 120),
            rate_limit_tpm=int(record.get("rate_limit_tpm") or 200_000),
        )

    def rotate(self, key_id: str) -> tuple[dict[str, Any], str]:
        old = self.db.get_api_key(key_id)
        if not old:
            raise KeyError(key_id)
        if old.get("revoked_at"):
            raise ValueError("Cannot rotate a revoked key")
        record, raw = self.create_key(
            name=old["name"],
            role=old["role"],
            permissions=old.get("permissions") or {},
            rate_limit_rpm=int(old["rate_limit_rpm"]),
            rate_limit_tpm=int(old["rate_limit_tpm"]),
            expires_at=old.get("expires_at"),
            rotated_from=key_id,
        )
        self.db.revoke_api_key(key_id)
        return record, raw

    def bootstrap_if_empty(self) -> list[tuple[str, str]]:
        """Create admin + inference keys only when no keys exist.

        Returns ``(label, plaintext_key)`` pairs for one-time display. Nothing is
        written to disk in plaintext.
        """
        if self.db.count_api_keys() > 0:
            return []
        admin, admin_raw = self.create_key(
            name="bootstrap-admin",
            role="admin",
            permissions={"inference": True},
            rate_limit_rpm=600,
            rate_limit_tpm=1_000_000,
        )
        inference, inference_raw = self.create_key(
            name="bootstrap-inference",
            role="inference",
            permissions={"models": ["*"], "allow_streaming": True, "allow_embeddings": True},
            rate_limit_rpm=120,
            rate_limit_tpm=200_000,
        )
        self.db.add_event(
            "info",
            "keys.bootstrap",
            "Generated initial administrator and inference API keys",
            metadata={"admin_key_id": admin["id"], "inference_key_id": inference["id"]},
        )
        return [("admin", admin_raw), ("inference", inference_raw)]


class RateLimiter:
    """Sliding-window RPM/TPM limiter plus per-IP auth failure brake."""

    def __init__(self) -> None:
        self._events: dict[str, deque[tuple[float, int]]] = {}
        self._auth_failures: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, ctx: AuthContext, estimated_tokens: int = 0) -> None:
        now = time.monotonic()
        async with self._lock:
            events = self._events.setdefault(ctx.key_id, deque())
            self._prune(events, now)
            if len(events) >= ctx.rate_limit_rpm:
                retry = max(1, int(60 - (now - events[0][0]))) if events else 60
                raise RateLimitError(
                    f"Rate limit exceeded: {ctx.rate_limit_rpm} requests/minute", retry
                )
            used = sum(tokens for _, tokens in events)
            if estimated_tokens and used + estimated_tokens > ctx.rate_limit_tpm:
                retry = max(1, int(60 - (now - events[0][0]))) if events else 60
                raise RateLimitError(
                    f"Token rate limit exceeded: {ctx.rate_limit_tpm} tokens/minute", retry
                )
            events.append((now, max(0, estimated_tokens)))

    async def record_usage(self, key_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        now = time.monotonic()
        async with self._lock:
            events = self._events.setdefault(key_id, deque())
            self._prune(events, now)
            if events:
                ts, prior = events[-1]
                events[-1] = (ts, prior + tokens)
            else:
                events.append((now, tokens))

    async def register_auth_failure(self, ip: str, limit: int = 20) -> None:
        now = time.monotonic()
        async with self._lock:
            bucket = self._auth_failures.setdefault(ip, deque())
            self._prune_plain(bucket, now, 60)
            bucket.append(now)
            if len(bucket) > limit:
                raise RateLimitError("Too many failed authentication attempts", 60)

    async def auth_failure_allowed(self, ip: str, limit: int = 20) -> bool:
        now = time.monotonic()
        async with self._lock:
            bucket = self._auth_failures.get(ip)
            if not bucket:
                return True
            self._prune_plain(bucket, now, 60)
            return len(bucket) < limit

    @staticmethod
    def _prune(events: deque[tuple[float, int]], now: float) -> None:
        while events and now - events[0][0] >= 60:
            events.popleft()

    @staticmethod
    def _prune_plain(events: deque[float], now: float, window: float) -> None:
        while events and now - events[0] >= window:
            events.popleft()

    def snapshot(self, key_id: str) -> dict[str, int]:
        events = self._events.get(key_id)
        if not events:
            return {"requests_1m": 0, "tokens_1m": 0}
        now = time.monotonic()
        recent = [(ts, tok) for ts, tok in events if now - ts < 60]
        return {"requests_1m": len(recent), "tokens_1m": sum(t for _, t in recent)}
