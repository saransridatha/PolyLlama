#!/usr/bin/env python3
"""Live compatibility smoke test using the official OpenAI Python SDK.

Requires a running gateway and valid keys:
    MS_ADMIN_KEY=... MS_INFERENCE_KEY=... python scripts/verify_live.py
"""

from __future__ import annotations

import os
import sys

import httpx
from openai import OpenAI


def main() -> int:
    admin_key = os.environ.get("MS_ADMIN_KEY", "")
    inference_key = os.environ.get("MS_INFERENCE_KEY", "")
    base = os.environ.get("MS_BENCH_BASE_URL", "http://127.0.0.1:8000/v1")
    admin = os.environ.get("MS_BENCH_ADMIN_URL", "http://127.0.0.1:8001")
    if not admin_key or not inference_key:
        print("Set MS_ADMIN_KEY and MS_INFERENCE_KEY", file=sys.stderr)
        return 2

    client = OpenAI(api_key=inference_key, base_url=base, timeout=180)
    checks: list[tuple[str, bool, str]] = []

    health = httpx.get(base.replace("/v1", "") + "/healthz", timeout=10)
    checks.append(("healthz", health.status_code == 200, health.text[:80]))

    models = client.models.list()
    ids = [model.id for model in models.data]
    checks.append(("models.list", "auto" in ids, f"{len(ids)} model ids"))

    response = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Reply with exactly: SDK_OK"}],
        max_tokens=12,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content or ""
    checks.append(
        (
            "chat.completions",
            bool(content),
            f"model={response.model} tokens={response.usage.total_tokens if response.usage else '?'}",
        )
    )

    chunks = 0
    streamed = []
    with client.chat.completions.stream(
        model="fast",
        messages=[{"role": "user", "content": "Count 1 to 4."}],
        max_tokens=16,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
        },
    ) as stream:
        for event in stream:
            choices = getattr(getattr(event, "chunk", None), "choices", None)
            if choices:
                chunks += 1
                text = choices[0].delta.content
                if text:
                    streamed.append(text)
    checks.append(("chat.completions.stream", chunks > 0, f"{chunks} chunks, text={''.join(streamed)[:40]!r}"))

    admin_overview = httpx.get(
        f"{admin}/admin/api/overview",
        headers={"Authorization": f"Bearer {admin_key}"},
        timeout=20,
    )
    checks.append(
        (
            "admin.overview",
            admin_overview.status_code == 200,
            f"status={admin_overview.status_code}",
        )
    )

    public_admin = httpx.get(
        base.replace("/v1", "") + "/admin/api/overview",
        headers={"Authorization": f"Bearer {admin_key}"},
        timeout=10,
    )
    checks.append(("admin isolation", public_admin.status_code == 404, f"status={public_admin.status_code}"))

    failed = 0
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
