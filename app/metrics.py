"""Small Prometheus exposition registry without external instrumentation deps."""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Any

_LABEL_ORDER = ("model", "endpoint", "status", "task", "type", "method", "code")

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self.gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(
            lambda: [0.0, 0.0, *([0.0] * len(_BUCKETS))]
        )
        self.started = __import__("time").time()

    @staticmethod
    def _labels(labels: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
        if not labels:
            return ()
        return tuple(sorted((str(k), str(v)) for k, v in labels.items()))

    def inc(self, name: str, labels: dict[str, Any] | None = None, value: float = 1.0) -> None:
        key = (name, self._labels(labels))
        with self._lock:
            self.counters[key] += value

    def set(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        key = (name, self._labels(labels))
        with self._lock:
            self.gauges[key] = float(value)

    def observe(self, name: str, value: float, labels: dict[str, Any] | None = None) -> None:
        key = (name, self._labels(labels))
        with self._lock:
            hist = self.histograms[key]
            hist[0] += value
            hist[1] += 1
            for idx, bucket in enumerate(_BUCKETS):
                if value <= bucket:
                    hist[2 + idx] += 1

    def render(self, extra: dict[str, Any] | None = None) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self.counters)
            gauges = dict(self.gauges)
            histograms = {k: list(v) for k, v in self.histograms.items()}
        by_name: dict[str, list[tuple[tuple[tuple[str, str], ...], Any]]] = defaultdict(list)
        for (name, labels), value in counters.items():
            by_name[name].append((labels, value))
        for (name, labels), value in gauges.items():
            by_name[name].append((labels, value))
        for (name, labels), value in histograms.items():
            by_name.setdefault(name, []).append((labels, value))
        for name in sorted(by_name):
            if name in {k[0] for k in histograms}:
                metric_type = "histogram"
            elif name in {k[0] for k in gauges}:
                metric_type = "gauge"
            else:
                metric_type = "counter"
            lines.append(f"# TYPE {name} {metric_type}")
            for labels, value in sorted(by_name[name], key=lambda item: item[0]):
                label_text = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
                if isinstance(value, list):
                    total, count, *bucket_counts = value
                    cumulative = 0.0
                    for idx, bucket in enumerate(_BUCKETS):
                        cumulative = bucket_counts[idx]
                        suffix = (
                            f'{{{label_text},le="{bucket}"}}'
                            if label_text
                            else f'{{le="{bucket}"}}'
                        )
                        lines.append(f"{name}_bucket{suffix} {cumulative:.0f}")
                    inf_suffix = f'{{{label_text},le="+Inf"}}' if label_text else '{le="+Inf"}'
                    lines.append(f"{name}_bucket{inf_suffix} {count:.0f}")
                    lines.append(f"{name}_sum{_fmt_labels(label_text)} {total:.6f}")
                    lines.append(f"{name}_count{_fmt_labels(label_text)} {count:.0f}")
                else:
                    lines.append(f"{name}{_fmt_labels(label_text)} {value:.6f}")
        if extra:
            for name, value in extra.items():
                safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
                lines.append(f"# TYPE {safe} gauge")
                if isinstance(value, dict):
                    for label, number in value.items():
                        try:
                            lines.append(f'{safe}{{kind="{_escape(label)}"}} {float(number):.6f}')
                        except (TypeError, ValueError):
                            continue
                else:
                    try:
                        lines.append(f"{safe} {float(value):.6f}")
                    except (TypeError, ValueError):
                        continue
        return "\n".join(lines) + "\n"


def _fmt_labels(label_text: str) -> str:
    return f"{{{label_text}}}" if label_text else ""


metrics = Metrics()
