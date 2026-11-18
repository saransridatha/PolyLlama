"""Structured logging with request correlation.

Secrets are never interpolated into log records. The redaction filter is a last
line of defence for authorization headers and API-key-shaped strings.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
api_key_id_var: ContextVar[str] = ContextVar("api_key_id", default="-")

_SECRET_PATTERNS = [
    (re.compile(r"msk_[A-Za-z0-9_\-]{8,}"), "msk_[REDACTED]"),
    (re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?[^\s\"',}]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(tunnel[_-]?token[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"), r"\1[REDACTED]"),
]


def redact(value: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        record.msg = redact(message)
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": redact(record.getMessage()),
            "request_id": request_id_var.get(),
        }
        if api_key_id_var.get() != "-":
            payload["api_key_id"] = api_key_id_var.get()
        for key in ("event", "model_id", "status", "duration_ms", "task", "method", "path"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, separators=(",", ":"), default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.now().strftime("%H:%M:%S")
        rid = request_id_var.get()
        prefix = f"{stamp} {record.levelname:<7} [{rid[:8]}] {record.name}: "
        return prefix + redact(record.getMessage())


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    handler.addFilter(RedactionFilter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, message: str, **fields: object) -> None:
    logger.info(message, extra=fields)
