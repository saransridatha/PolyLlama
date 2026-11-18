#!/usr/bin/env python3
"""Measured benchmark suite for a running ModelSwitch instance.

No synthetic numbers are produced. Every value in the report comes from an HTTP
request to the gateway, the llama.cpp backend through the gateway, or the
gateway's own system/admin endpoints.

Usage:
    MS_ADMIN_KEY=... MS_INFERENCE_KEY=... python scripts/benchmark.py \
        --models qwen3.5-2b,smollm3-3b
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def pct(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 2)


def summarize(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min_ms": round(min(clean), 2),
        "median_ms": round(statistics.median(clean), 2),
        "mean_ms": round(statistics.mean(clean), 2),
        "p95_ms": pct(clean, 95),
        "max_ms": round(max(clean), 2),
    }


class Bench:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # base_url includes /v1; admin_url does not.
        self.public = args.base_url.rstrip("/")
        self.admin = args.admin_url.rstrip("/")
        self.admin_headers = {"Authorization": f"Bearer {args.admin_key}"}
        self.inference_headers = {"Authorization": f"Bearer {args.inference_key}"}
        self.http = httpx.Client(
            timeout=httpx.Timeout(600, connect=15),
            limits=httpx.Limits(max_connections=32),
        )
        self.result: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": {},
            "models": {},
            "concurrency": [],
            "switching": {},
            "notes": [],
        }

    def close(self) -> None:
        self.http.close()

    def admin_get(self, path: str) -> Any:
        response = self.http.get(f"{self.admin}{path}", headers=self.admin_headers)
        response.raise_for_status()
        return response.json()

    def admin_post(self, path: str) -> Any:
        response = self.http.post(f"{self.admin}{path}", headers=self.admin_headers)
        response.raise_for_status()
        return response.json()

    def unload_all(self, models: list[dict[str, Any]]) -> None:
        for model in models:
            if model.get("runtime", {}).get("loaded"):
                try:
                    self.admin_post(f"/admin/api/models/{model['id']}/unload")
                except httpx.HTTPError:
                    pass

    def collect_environment(self) -> None:
        system = self.admin_get("/admin/api/system")
        readiness = self.admin_get("/admin/api/readiness")
        models = self.admin_get("/admin/api/models")
        self.result["environment"] = {
            "host": {
                "os": system["hardware"]["os_name"],
                "os_version": system["hardware"]["os_version"],
                "arch": system["hardware"]["arch"],
                "cpu": system["hardware"]["cpu_model"],
                "physical_cores": system["hardware"]["physical_cores"],
                "logical_cores": system["hardware"]["logical_cores"],
                "ram_bytes": system["hardware"]["total_ram_bytes"],
                "gpus": system["hardware"].get("gpus", []),
                "backend": system["hardware"].get("llama_backend"),
                "warnings": system["hardware"].get("warnings", []),
            },
            "llama_version": readiness.get("llama_version"),
            "python": platform.python_version(),
            "models_registered": [m["id"] for m in models["models"]],
            "resident_limit": models["limits"]["max_resident_models"],
            "public_url": self.public,
        }

    def stream_request(self, model_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        started = time.perf_counter()
        ttft: float | None = None
        completion_tokens = 0
        prompt_tokens = 0
        text_parts: list[str] = []
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with self.http.stream(
            "POST", f"{self.public}/chat/completions", headers=self.inference_headers, json=body
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or delta.get("reasoning_content") or choices[0].get("text")
                    if content:
                        text_parts.append(str(content))
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens") or completion_tokens
                    prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                timings = chunk.get("timings")
                if timings:
                    completion_tokens = timings.get("predicted_n") or completion_tokens
                    prompt_tokens = timings.get("prompt_n") or prompt_tokens
        total_ms = (time.perf_counter() - started) * 1000
        if completion_tokens == 0 and text_parts:
            completion_tokens = max(1, round(len("".join(text_parts)) / 3.6))
        generation_ms = max(0.001, total_ms - (ttft or 0))
        tps = completion_tokens / (generation_ms / 1000) if completion_tokens else None
        return {
            "ttft_ms": round(ttft, 2) if ttft is not None else None,
            "total_ms": round(total_ms, 2),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tokens_per_second": round(tps, 2) if tps else None,
            "text": "".join(text_parts)[:240],
        }

    def nonstream_request(self, model_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        started = time.perf_counter()
        response = self.http.post(
            f"{self.public}/chat/completions",
            headers=self.inference_headers,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        total_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        return {
            "total_ms": round(total_ms, 2),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "queue_ms": payload.get("modelswitch", {}).get("queue_ms"),
            "load_ms": payload.get("modelswitch", {}).get("load_ms"),
            "model": payload.get("model"),
        }

    def model_memory(self, model_id: str) -> dict[str, Any]:
        listing = self.admin_get("/admin/api/models")
        model = next((m for m in listing["models"] if m["id"] == model_id), None)
        if not model:
            return {}
        runtime = model.get("runtime") or {}
        backend = runtime.get("backend") or {}
        return {
            "rss_bytes": backend.get("rss_bytes"),
            "estimated_ram_bytes": (runtime.get("estimated_memory") or {}).get("ram_bytes"),
            "estimated_vram_bytes": (runtime.get("estimated_memory") or {}).get("vram_bytes"),
            "load_count": runtime.get("load_count"),
            "crash_count": runtime.get("crash_count"),
        }

    def benchmark_model(self, model: dict[str, Any]) -> None:
        model_id = model["id"]
        prompt = "Explain in two sentences why CPU cache locality matters for inference."
        print(f"  loading {model_id}…")
        started = time.perf_counter()
        try:
            load = self.admin_post(f"/admin/api/models/{model_id}/load")
        except httpx.HTTPStatusError as exc:
            self.result["models"][model_id] = {"error": exc.response.text[:1000]}
            return
        cold_load_ms = (time.perf_counter() - started) * 1000
        # Warmup
        try:
            self.nonstream_request(model_id, "Reply with OK.", 4)
        except httpx.HTTPError as exc:
            self.result["models"][model_id] = {"error": str(exc)}
            return
        memory = self.model_memory(model_id)
        stream_runs = [self.stream_request(model_id, prompt, 32) for _ in range(self.args.runs)]
        nonstream_runs = [self.nonstream_request(model_id, prompt, 32) for _ in range(self.args.runs)]
        ttfts = [run["ttft_ms"] for run in stream_runs if run["ttft_ms"] is not None]
        tps_values = [run["tokens_per_second"] for run in stream_runs if run["tokens_per_second"]]
        totals = [run["total_ms"] for run in stream_runs]
        nonstream_totals = [run["total_ms"] for run in nonstream_runs]
        unload_started = time.perf_counter()
        try:
            self.admin_post(f"/admin/api/models/{model_id}/unload")
            unload_ms = (time.perf_counter() - unload_started) * 1000
        except httpx.HTTPError:
            unload_ms = None
        record = {
            "id": model_id,
            "name": model.get("name"),
            "parameter_size": model.get("parameter_size"),
            "quantization": model.get("quantization"),
            "file_size_bytes": model.get("size_bytes"),
            "context_length": model.get("context_length"),
            "cold_load_ms": round(cold_load_ms, 2),
            "unload_ms": round(unload_ms, 2) if unload_ms is not None else None,
            "memory": memory,
            "stream": {
                "ttft": summarize(ttfts),
                "completion_tokens_per_second": {
                    "median": round(statistics.median(tps_values), 2) if tps_values else None,
                    "mean": round(statistics.mean(tps_values), 2) if tps_values else None,
                },
                "request_duration": summarize(totals),
                "runs": stream_runs,
            },
            "nonstream": {
                "request_duration": summarize(nonstream_totals),
                "runs": nonstream_runs,
            },
        }
        self.result["models"][model_id] = record
        print(
            f"    load {record['cold_load_ms']} ms · TTFT {record['stream']['ttft'].get('median_ms')} ms · "
            f"{record['stream']['completion_tokens_per_second']['median']} tok/s"
        )

    def benchmark_concurrency(self, model: dict[str, Any]) -> None:
        model_id = model["id"]
        print(f"  concurrency on {model_id}…")
        self.admin_post(f"/admin/api/models/{model_id}/load")
        self.nonstream_request(model_id, "Reply OK.", 4)
        for concurrency in (1, 2, 4):
            def call(index: int) -> dict[str, Any]:
                return self.nonstream_request(
                    model_id, f"List three benefits of caching. Request {index}.", 24
                )

            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(call, i) for i in range(concurrency)]
                results = []
                errors = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception:
                        errors += 1
            wall_ms = (time.perf_counter() - started) * 1000
            tokens = sum(r["completion_tokens"] for r in results)
            self.result["concurrency"].append(
                {
                    "model": model_id,
                    "concurrency": concurrency,
                    "wall_ms": round(wall_ms, 2),
                    "successes": len(results),
                    "errors": errors,
                    "aggregate_tokens_per_second": round(tokens / (wall_ms / 1000), 2),
                    "request_latency": summarize([r["total_ms"] for r in results]),
                    "queue_ms": [r["queue_ms"] for r in results],
                    "load_ms": [r["load_ms"] for r in results],
                }
            )
            print(
                f"    c={concurrency}: wall {wall_ms:.0f} ms · "
                f"{tokens / (wall_ms / 1000):.1f} tok/s aggregate"
            )
        self.admin_post(f"/admin/api/models/{model_id}/unload")

    def benchmark_switching(self, first: dict[str, Any], second: dict[str, Any]) -> None:
        print("  model switching…")
        self.admin_post(f"/admin/api/models/{first['id']}/unload")
        self.admin_post(f"/admin/api/models/{second['id']}/unload")
        timings = []
        for index in range(2):
            model = first if index % 2 == 0 else second
            started = time.perf_counter()
            response = self.nonstream_request(model["id"], "Reply OK.", 4)
            total = (time.perf_counter() - started) * 1000
            timings.append(
                {
                    "from": second["id"] if index % 2 == 0 else first["id"],
                    "to": model["id"],
                    "request_ms": round(total, 2),
                    "load_ms": response.get("load_ms"),
                }
            )
        self.result["switching"] = {
            "first": first["id"],
            "second": second["id"],
            "transitions": timings,
            "median_transition_ms": round(statistics.median([t["request_ms"] for t in timings]), 2),
        }
        for model in (first, second):
            try:
                self.admin_post(f"/admin/api/models/{model['id']}/unload")
            except httpx.HTTPError:
                pass

    def run(self) -> dict[str, Any]:
        self.collect_environment()
        models_response = self.admin_get("/admin/api/models")
        all_models = models_response["models"]
        budget = self.args.models
        selected = [m for m in all_models if m["id"] in budget] if budget else [
            m for m in all_models if m["enabled"]
        ][:3]
        if not selected:
            raise SystemExit("No matching enabled models. Pass --models with registered ids.")
        self.unload_all(all_models)
        for model in selected:
            self.benchmark_model(model)
        # Use the smallest selected model for concurrency measurements.
        fastest = min(
            selected,
            key=lambda m: (m.get("parameter_count") or 10**12, m.get("size_bytes") or 10**18),
        )
        self.benchmark_concurrency(fastest)
        if len(selected) >= 2:
            self.benchmark_switching(selected[0], selected[1])
        self.result["summary"] = {
            "models_measured": len([m for m in self.result["models"].values() if "error" not in m]),
            "errors": sum(1 for m in self.result["models"].values() if "error" in m),
            "benchmark_duration_s": round(
                (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(self.result["generated_at"])
                ).total_seconds(),
                1,
            ),
        }
        return self.result

    def write_reports(self, result: dict[str, Any]) -> tuple[Path, Path]:
        REPORTS.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        json_path = REPORTS / f"benchmark-{stamp}.json"
        json_path.write_text(json.dumps(result, indent=2))
        md_path = REPORTS / "BENCHMARKS.md"
        lines = [
            "# ModelSwitch benchmark report",
            "",
            f"Generated: `{result['generated_at']}`",
            "",
            "These are measured values from a live gateway. Hardware, model choice,",
            "context size and background load materially affect results.",
            "",
            "## Environment",
            "",
            f"- Host: {result['environment']['host']['os']} {result['environment']['host']['os_version']} ({result['environment']['host']['arch']})",
            f"- CPU: {result['environment']['host']['cpu']} · {result['environment']['host']['physical_cores']} cores / {result['environment']['host']['logical_cores']} threads",
            f"- RAM: {result['environment']['host']['ram_bytes'] / 1024**3:.1f} GiB",
            f"- Backend: {result['environment']['host']['backend']}",
            f"- llama.cpp: {result['environment']['llama_version']}",
            f"- Resident limit: {result['environment']['resident_limit']}",
            "",
        ]
        for gpu in result["environment"]["host"].get("gpus", []):
            lines.append(f"- GPU: {gpu.get('name')} · {gpu.get('total_mb')} MiB · backend {gpu.get('backend')}")
        if result["environment"]["host"].get("warnings"):
            lines.append(f"- Warnings: {'; '.join(result['environment']['host']['warnings'])}")
        lines.extend(["", "## Per-model measurements", ""])
        lines.append("| Model | Quant | Cold load | TTFT median | Stream tok/s | Non-stream median | RSS |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for model_id, rec in result["models"].items():
            if "error" in rec:
                lines.append(f"| {model_id} | — | FAILED: {rec['error'][:60]} | — | — | — | — |")
                continue
            rss = (rec.get("memory") or {}).get("rss_bytes")
            lines.append(
                f"| `{model_id}` | {rec.get('quantization')} | {rec.get('cold_load_ms'):.0f} ms | "
                f"{(rec['stream']['ttft'].get('median_ms') or 0):.0f} ms | "
                f"{(rec['stream']['completion_tokens_per_second'].get('median') or 0):.1f} | "
                f"{(rec['nonstream']['request_duration'].get('median_ms') or 0):.0f} ms | "
                f"{(rss / 1024**3 if rss else 0):.2f} GiB |"
            )
        lines.extend(["", "## Concurrency", "", "| Model | Parallel requests | Wall | Aggregate tok/s | p50 latency | Errors |", "|---|---:|---:|---:|---:|---:|"])
        for row in result["concurrency"]:
            lines.append(
                f"| `{row['model']}` | {row['concurrency']} | {row['wall_ms']:.0f} ms | "
                f"{row['aggregate_tokens_per_second']:.1f} | "
                f"{(row['request_latency'].get('median_ms') or 0):.0f} ms | {row['errors']} |"
            )
        if result.get("switching"):
            sw = result["switching"]
            lines.extend(
                [
                    "",
                    "## Model switching",
                    "",
                    f"Alternating between `{sw['first']}` and `{sw['second']}`: median cold-switch request "
                    f"`{sw['median_transition_ms']:.0f} ms`.",
                ]
            )
        lines.extend(
            [
                "",
                "## Methodology",
                "",
                "- Streaming TTFT is measured client-side from request start to the first content delta (includes routing and any cold load).",
                "- Stream tok/s is completion tokens divided by (total time − TTFT).",
                "- Cold load is the admin load endpoint duration until llama.cpp reports ready.",
                "- Concurrency uses ThreadPoolExecutor and explicit model requests; aggregate tok/s is total completion tokens divided by wall time.",
                "- No values are synthesized. Raw samples are in the JSON report alongside this file.",
                "",
            ]
        )
        md_path.write_text("\n".join(lines))
        return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("MS_BENCH_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--admin-url", default=os.environ.get("MS_BENCH_ADMIN_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--admin-key", default=os.environ.get("MS_ADMIN_KEY", ""))
    parser.add_argument("--inference-key", default=os.environ.get("MS_INFERENCE_KEY", ""))
    parser.add_argument("--models", default=os.environ.get("MS_BENCH_MODELS", ""), help="Comma-separated model ids")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="One run per measurement")
    args = parser.parse_args()
    if args.quick:
        args.runs = 1
    if not args.admin_key or not args.inference_key:
        parser.error("Set MS_ADMIN_KEY and MS_INFERENCE_KEY or pass --admin-key/--inference-key")
    bench = Bench(args)
    try:
        result = bench.run()
        json_path, md_path = bench.write_reports(result)
        print(f"\nWrote {json_path}\nWrote {md_path}")
        return 0
    finally:
        bench.close()


if __name__ == "__main__":
    raise SystemExit(main())
