"""SQLite persistence.

A single connection guarded by a process-local lock is intentional: every query
is short, WAL mode allows readers, and the control-plane write volume is tiny.
Heavy inference never runs on the event loop; only metadata is persisted here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# Update statements are chosen from fixed maps; no request data ever becomes SQL
# text. This is both safer and easier to audit than dynamic column names.
KEY_UPDATE_SQL = {
    "name": "name=?",
    "rate_limit_rpm": "rate_limit_rpm=?",
    "rate_limit_tpm": "rate_limit_tpm=?",
    "permissions": "permissions=?",
    "expires_at": "expires_at=?",
    "key_hash": "key_hash=?",
    "lookup_hash": "lookup_hash=?",
    "prefix": "prefix=?",
    "last_used_at": "last_used_at=?",
}
MODEL_UPDATE_SQL = {
    "name": "name=?",
    "path": "path=?",
    "aliases": "aliases=?",
    "capabilities": "capabilities=?",
    "context_length": "context_length=?",
    "max_context_length": "max_context_length=?",
    "quantization": "quantization=?",
    "parameter_size": "parameter_size=?",
    "parameter_count": "parameter_count=?",
    "size_bytes": "size_bytes=?",
    "priority": "priority=?",
    "enabled": "enabled=?",
    "residency": "residency=?",
    "max_concurrency": "max_concurrency=?",
    "n_gpu_layers": "n_gpu_layers=?",
    "threads": "threads=?",
    "description": "description=?",
    "extra_args": "extra_args=?",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin','inference')),
    key_hash TEXT NOT NULL,
    lookup_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    permissions TEXT NOT NULL DEFAULT '{}',
    rate_limit_rpm INTEGER NOT NULL DEFAULT 120,
    rate_limit_tpm INTEGER NOT NULL DEFAULT 200000,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    expires_at TEXT,
    revoked_at TEXT,
    rotated_from TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_lookup ON api_keys(lookup_hash);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    aliases TEXT NOT NULL DEFAULT '[]',
    capabilities TEXT NOT NULL DEFAULT '[]',
    context_length INTEGER NOT NULL DEFAULT 4096,
    max_context_length INTEGER NOT NULL DEFAULT 4096,
    quantization TEXT NOT NULL DEFAULT 'unknown',
    parameter_size TEXT NOT NULL DEFAULT 'unknown',
    parameter_count INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 50,
    enabled INTEGER NOT NULL DEFAULT 1,
    residency TEXT NOT NULL DEFAULT 'on_demand',
    max_concurrency INTEGER NOT NULL DEFAULT 2,
    n_gpu_layers INTEGER,
    threads INTEGER,
    description TEXT NOT NULL DEFAULT '',
    extra_args TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    api_key_id TEXT,
    endpoint TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    resolved_model TEXT,
    task_category TEXT,
    status TEXT NOT NULL,
    http_status INTEGER,
    stream INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    queue_ms REAL NOT NULL DEFAULT 0,
    load_ms REAL NOT NULL DEFAULT 0,
    ttft_ms REAL,
    total_ms REAL,
    error TEXT,
    routing TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(resolved_model, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_key ON requests(api_key_id, created_at DESC);

CREATE TABLE IF NOT EXISTS routing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    task TEXT NOT NULL,
    confidence REAL NOT NULL,
    selected_model TEXT,
    explanation TEXT NOT NULL,
    alternatives TEXT NOT NULL DEFAULT '[]',
    constraints TEXT NOT NULL DEFAULT '{}',
    fallback INTEGER NOT NULL DEFAULT 0,
    routing_ms REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routing_created ON routing_history(created_at DESC);

CREATE TABLE IF NOT EXISTS backend_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT,
    level TEXT NOT NULL,
    event TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_created ON backend_events(created_at DESC);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_perf (
    model_id TEXT PRIMARY KEY,
    samples INTEGER NOT NULL DEFAULT 0,
    avg_ttft_ms REAL,
    avg_tps REAL,
    avg_total_ms REAL,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat(timespec="milliseconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=30
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _migrate(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(requests)").fetchall()
        }
        if "load_ms" not in columns:
            self._conn.execute(
                "ALTER TABLE requests ADD COLUMN load_ms REAL NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, [tuple(r) for r in rows])

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- API keys --------------------------------------------------------
    def create_api_key(
        self,
        *,
        key_id: str,
        name: str,
        role: str,
        key_hash: str,
        lookup_hash: str,
        prefix: str,
        permissions: dict[str, Any],
        rate_limit_rpm: int,
        rate_limit_tpm: int,
        expires_at: str | None = None,
        rotated_from: str | None = None,
    ) -> dict[str, Any]:
        now = iso()
        self.execute(
            """INSERT INTO api_keys
            (id,name,role,key_hash,lookup_hash,prefix,permissions,rate_limit_rpm,
             rate_limit_tpm,created_at,expires_at,rotated_from)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                key_id,
                name,
                role,
                key_hash,
                lookup_hash,
                prefix,
                json.dumps(permissions),
                rate_limit_rpm,
                rate_limit_tpm,
                now,
                expires_at,
                rotated_from,
            ),
        )
        return self.get_api_key(key_id)  # type: ignore[return-value]

    def get_api_key(self, key_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM api_keys WHERE id=?", (key_id,))
        return self._decode_key(row)

    def find_api_key_by_lookup(self, lookup_hash: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM api_keys WHERE lookup_hash=?", (lookup_hash,))
        return self._decode_key(row)

    @staticmethod
    def _decode_key(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row:
            row["permissions"] = json.loads(row.get("permissions") or "{}")
            row["revoked"] = bool(row.get("revoked_at"))
        return row

    def list_api_keys(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM api_keys ORDER BY created_at DESC")
        return [self._decode_key(r) for r in rows]  # type: ignore[list-item]

    def revoke_api_key(self, key_id: str) -> bool:
        cur = self.execute(
            "UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
            (iso(), key_id),
        )
        return cur.rowcount > 0

    def update_api_key(self, key_id: str, fields: dict[str, Any]) -> None:
        sets, values = [], []
        for key, value in fields.items():
            fragment = KEY_UPDATE_SQL.get(key)
            if not fragment:
                continue
            if key == "permissions":
                value = json.dumps(value)
            sets.append(fragment)
            values.append(value)
        if not sets:
            return
        values.append(key_id)
        self.execute(f"UPDATE api_keys SET {', '.join(sets)} WHERE id=?", values)  # nosec B608

    def touch_api_key(self, key_id: str) -> None:
        self.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (iso(), key_id))

    def count_admin_keys(self) -> int:
        row = self.query_one(
            "SELECT COUNT(*) AS n FROM api_keys WHERE role='admin' AND revoked_at IS NULL"
        )
        return int(row["n"]) if row else 0

    def count_api_keys(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS n FROM api_keys WHERE revoked_at IS NULL")
        return int(row["n"]) if row else 0

    # -- models ----------------------------------------------------------
    def create_model(self, model: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO models
            (id,name,path,aliases,capabilities,context_length,max_context_length,quantization,parameter_size,
             parameter_count,size_bytes,priority,enabled,residency,max_concurrency,
             n_gpu_layers,threads,description,extra_args,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model["id"],
                model["name"],
                model["path"],
                json.dumps(model.get("aliases", [])),
                json.dumps(model.get("capabilities", [])),
                model.get("context_length", 4096),
                model.get("max_context_length", 4096),
                model.get("quantization", "unknown"),
                model.get("parameter_size", "unknown"),
                model.get("parameter_count", 0),
                model.get("size_bytes", 0),
                model.get("priority", 50),
                int(model.get("enabled", True)),
                model.get("residency", "on_demand"),
                model.get("max_concurrency", 2),
                model.get("n_gpu_layers"),
                model.get("threads"),
                model.get("description", ""),
                json.dumps(model.get("extra_args", [])),
                model.get("created_at", iso()),
                model.get("updated_at", iso()),
            ),
        )

    def update_model(self, model_id: str, fields: dict[str, Any]) -> None:
        sets, values = [], []
        for key, value in fields.items():
            fragment = MODEL_UPDATE_SQL.get(key)
            if not fragment:
                continue
            if key in ("aliases", "capabilities", "extra_args"):
                value = json.dumps(value)
            if key == "enabled":
                value = int(bool(value))
            sets.append(fragment)
            values.append(value)
        if not sets:
            return
        sets.append("updated_at=?")
        values.extend([iso(), model_id])
        self.execute(f"UPDATE models SET {', '.join(sets)} WHERE id=?", values)  # nosec B608

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        return self._decode_model(self.query_one("SELECT * FROM models WHERE id=?", (model_id,)))

    def find_model(self, name: str) -> dict[str, Any] | None:
        row = self.query_one(
            "SELECT * FROM models WHERE id=? OR lower(name)=lower(?) ORDER BY enabled DESC LIMIT 1",
            (name, name),
        )
        if row:
            return self._decode_model(row)
        for model in self.list_models():
            if name.lower() in [str(a).lower() for a in model.get("aliases", [])]:
                return model
        return None

    @staticmethod
    def _decode_model(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        for field in ("aliases", "capabilities", "extra_args"):
            try:
                row[field] = json.loads(row.get(field) or "[]")
            except json.JSONDecodeError:
                row[field] = []
        row["enabled"] = bool(row.get("enabled"))
        return row

    def list_models(self) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM models ORDER BY priority DESC, name ASC")
        return [self._decode_model(r) for r in rows]  # type: ignore[list-item]

    def delete_model(self, model_id: str) -> bool:
        cur = self.execute("DELETE FROM models WHERE id=?", (model_id,))
        return cur.rowcount > 0

    # -- settings --------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM settings WHERE key=?", (key,))
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            """INSERT INTO settings(key,value,updated_at) VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, json.dumps(value), iso()),
        )

    # -- requests --------------------------------------------------------
    def start_request(
        self,
        *,
        request_id: str,
        api_key_id: str | None,
        endpoint: str,
        requested_model: str,
        stream: bool,
    ) -> None:
        self.execute(
            """INSERT INTO requests
            (request_id,api_key_id,endpoint,requested_model,status,stream,created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (request_id, api_key_id, endpoint, requested_model, "running", int(stream), iso()),
        )

    def finish_request(
        self,
        request_id: str,
        *,
        status: str,
        http_status: int | None = None,
        resolved_model: str | None = None,
        task_category: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        queue_ms: float = 0.0,
        load_ms: float = 0.0,
        ttft_ms: float | None = None,
        total_ms: float | None = None,
        error: str | None = None,
        routing: dict[str, Any] | None = None,
    ) -> None:
        self.execute(
            """UPDATE requests SET status=?,http_status=?,resolved_model=?,task_category=?,
            prompt_tokens=?,completion_tokens=?,queue_ms=?,load_ms=?,ttft_ms=?,total_ms=?,error=?,
            routing=?,completed_at=? WHERE request_id=?""",
            (
                status,
                http_status,
                resolved_model,
                task_category,
                prompt_tokens,
                completion_tokens,
                queue_ms,
                load_ms,
                ttft_ms,
                total_ms,
                error,
                json.dumps(routing) if routing else None,
                iso(),
                request_id,
            ),
        )

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM requests WHERE request_id=?", (request_id,))
        if row and row.get("routing"):
            try:
                row["routing"] = json.loads(row["routing"])
            except json.JSONDecodeError:
                pass
        return row

    def list_requests(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self.query(
                "SELECT * FROM requests WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = self.query("SELECT * FROM requests ORDER BY created_at DESC LIMIT ?", (limit,))
        for row in rows:
            if row.get("routing"):
                try:
                    row["routing"] = json.loads(row["routing"])
                except json.JSONDecodeError:
                    row["routing"] = None
        return rows

    # -- routing ---------------------------------------------------------
    def update_routing_decision(
        self,
        request_id: str,
        *,
        selected_model: str | None,
        explanation: str,
        fallback: bool,
    ) -> None:
        row = self.query_one(
            "SELECT id FROM routing_history WHERE request_id=? ORDER BY id DESC LIMIT 1",
            (request_id,),
        )
        if not row:
            return
        self.execute(
            "UPDATE routing_history SET selected_model=?, explanation=?, fallback=? WHERE id=?",
            (selected_model, explanation, int(fallback), row["id"]),
        )

    def add_routing_history(
        self,
        *,
        request_id: str,
        task: str,
        confidence: float,
        selected_model: str | None,
        explanation: str,
        alternatives: list[dict[str, Any]],
        constraints: dict[str, Any],
        fallback: bool,
        routing_ms: float,
    ) -> None:
        self.execute(
            """INSERT INTO routing_history
            (request_id,task,confidence,selected_model,explanation,alternatives,
             constraints,fallback,routing_ms,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                request_id,
                task,
                confidence,
                selected_model,
                explanation,
                json.dumps(alternatives),
                json.dumps(constraints),
                int(fallback),
                routing_ms,
                iso(),
            ),
        )

    def list_routing_history(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.query("SELECT * FROM routing_history ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            for field in ("alternatives", "constraints"):
                try:
                    row[field] = json.loads(row.get(field) or "null")
                except json.JSONDecodeError:
                    row[field] = None
            row["fallback"] = bool(row["fallback"])
        return rows

    def get_routing_for_request(self, request_id: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT * FROM routing_history WHERE request_id=? ORDER BY id DESC LIMIT 1",
            (request_id,),
        )
        if not rows:
            return None
        row = rows[0]
        for field in ("alternatives", "constraints"):
            try:
                row[field] = json.loads(row.get(field) or "null")
            except json.JSONDecodeError:
                row[field] = None
        return row

    # -- backend events --------------------------------------------------
    def add_event(
        self,
        level: str,
        event: str,
        message: str,
        model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.execute(
            """INSERT INTO backend_events(model_id,level,event,message,metadata,created_at)
            VALUES (?,?,?,?,?,?)""",
            (model_id, level, event, message, json.dumps(metadata or {}), iso()),
        )

    def list_events(self, limit: int = 100, level: str | None = None) -> list[dict[str, Any]]:
        if level:
            rows = self.query(
                "SELECT * FROM backend_events WHERE level=? ORDER BY id DESC LIMIT ?",
                (level, limit),
            )
        else:
            rows = self.query("SELECT * FROM backend_events ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            try:
                row["metadata"] = json.loads(row.get("metadata") or "{}")
            except json.JSONDecodeError:
                row["metadata"] = {}
        return rows

    # -- performance -----------------------------------------------------
    def get_perf(self, model_id: str) -> dict[str, Any] | None:
        return self.query_one("SELECT * FROM model_perf WHERE model_id=?", (model_id,))

    def update_perf(
        self,
        model_id: str,
        *,
        ttft_ms: float | None,
        tps: float | None,
        total_ms: float | None,
        success: bool,
    ) -> None:
        current = self.get_perf(model_id)
        now = iso()
        if not current:
            self.execute(
                """INSERT INTO model_perf
                (model_id,samples,avg_ttft_ms,avg_tps,avg_total_ms,success_count,
                 failure_count,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    model_id,
                    1 if success else 0,
                    ttft_ms,
                    tps,
                    total_ms,
                    int(success),
                    int(not success),
                    now,
                ),
            )
            return
        samples = int(current["samples"] or 0)
        alpha = 0.25 if samples else 1.0

        def ewma(old: float | None, new: float | None) -> float | None:
            if new is None:
                return old
            if old is None:
                return new
            return old * (1 - alpha) + new * alpha

        self.execute(
            """UPDATE model_perf SET samples=?,avg_ttft_ms=?,avg_tps=?,avg_total_ms=?,
            success_count=?,failure_count=?,updated_at=? WHERE model_id=?""",
            (
                samples + (1 if success else 0),
                ewma(current["avg_ttft_ms"], ttft_ms),
                ewma(current["avg_tps"], tps),
                ewma(current["avg_total_ms"], total_ms),
                int(current["success_count"]) + int(success),
                int(current["failure_count"]) + int(not success),
                now,
                model_id,
            ),
        )

    # -- aggregates ------------------------------------------------------
    def stats_summary(self) -> dict[str, Any]:
        cutoff_1m = iso(utcnow() - timedelta(minutes=1))
        cutoff_5m = iso(utcnow() - timedelta(minutes=5))
        cutoff_24h = iso(utcnow() - timedelta(hours=24))
        total = self.query_one("SELECT COUNT(*) n FROM requests") or {"n": 0}
        rpm = self.query_one(
            "SELECT COUNT(*) n FROM requests WHERE created_at>=?", (cutoff_1m,)
        ) or {"n": 0}
        recent = self.query_one(
            """SELECT COUNT(*) n, AVG(total_ms) avg_ms, SUM(prompt_tokens+completion_tokens) tokens
            FROM requests WHERE created_at>=? AND status='completed'""",
            (cutoff_5m,),
        ) or {}
        day = self.query_one(
            """SELECT COUNT(*) n, SUM(prompt_tokens) pt, SUM(completion_tokens) ct,
            SUM(CASE WHEN status NOT IN ('completed','running') THEN 1 ELSE 0 END) errors
            FROM requests WHERE created_at>=?""",
            (cutoff_24h,),
        ) or {}
        return {
            "total_requests": int(total["n"]),
            "requests_last_minute": int(rpm["n"]),
            "avg_latency_ms": _float(recent.get("avg_ms")),
            "tokens_last_5m": int(recent.get("tokens") or 0),
            "requests_last_24h": int(day.get("n") or 0),
            "prompt_tokens_24h": int(day.get("pt") or 0),
            "completion_tokens_24h": int(day.get("ct") or 0),
            "errors_24h": int(day.get("errors") or 0),
        }

    def request_timeseries(self, minutes: int = 30) -> list[dict[str, Any]]:
        """Return a dense minute series (including zero-traffic minutes).

        Dense buckets keep charts honest: an idle period is visible as zero
        traffic rather than collapsing the axis.
        """
        minutes = max(5, min(240, int(minutes)))
        now = utcnow().replace(second=0, microsecond=0)
        start = now - timedelta(minutes=minutes - 1)
        cutoff = start.isoformat(timespec="milliseconds")
        rows = self.query(
            """SELECT substr(created_at,1,16) bucket, COUNT(*) count,
            AVG(total_ms) avg_ms, SUM(completion_tokens) tokens
            FROM requests WHERE created_at>=? GROUP BY bucket ORDER BY bucket""",
            (cutoff,),
        )
        indexed = {row["bucket"]: row for row in rows}
        series: list[dict[str, Any]] = []
        for offset in range(minutes):
            moment = start + timedelta(minutes=offset)
            key = moment.strftime("%Y-%m-%dT%H:%M")
            row = indexed.get(key)
            series.append(
                {
                    "bucket": moment.strftime("%Y-%m-%dT%H:%M:00Z"),
                    "count": int(row["count"]) if row else 0,
                    "avg_ms": _float(row["avg_ms"]) if row else None,
                    "tokens": int(row["tokens"] or 0) if row else 0,
                }
            )
        return series

    def model_stats(self) -> list[dict[str, Any]]:
        cutoff = iso(utcnow() - timedelta(hours=24))
        rows = self.query(
            """SELECT resolved_model model_id, COUNT(*) requests,
            SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) completed,
            AVG(total_ms) avg_ms, SUM(prompt_tokens) prompt_tokens,
            SUM(completion_tokens) completion_tokens
            FROM requests WHERE created_at>=? AND resolved_model IS NOT NULL
            GROUP BY resolved_model ORDER BY requests DESC""",
            (cutoff,),
        )
        return rows

    def usage_by_key(self) -> list[dict[str, Any]]:
        rows = self.query(
            """SELECT k.id, k.name, k.role, k.prefix, k.rate_limit_rpm, k.rate_limit_tpm,
            k.last_used_at, k.revoked_at,
            COUNT(r.request_id) requests_24h,
            SUM(r.prompt_tokens+r.completion_tokens) tokens_24h
            FROM api_keys k LEFT JOIN requests r
              ON r.api_key_id=k.id AND r.created_at>=?
            GROUP BY k.id ORDER BY k.created_at DESC""",
            (iso(utcnow() - timedelta(hours=24)),),
        )
        return rows

    def purge(self, retention_days: int) -> dict[str, int]:
        cutoff = iso(utcnow() - timedelta(days=retention_days))
        requests = self.execute("DELETE FROM requests WHERE created_at<?", (cutoff,)).rowcount
        routing = self.execute(
            "DELETE FROM routing_history WHERE created_at<?", (cutoff,)
        ).rowcount
        events = self.execute("DELETE FROM backend_events WHERE created_at<?", (cutoff,)).rowcount
        return {"requests": requests, "routing": routing, "events": events}


def _float(value: Any) -> float | None:
    try:
        return round(float(value), 2) if value is not None else None
    except (TypeError, ValueError):
        return None
