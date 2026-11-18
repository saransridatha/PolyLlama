"""Fair, priority-aware concurrency gates.

One gate exists per resident model. A request either gets a slot immediately or
waits in a bounded queue. Higher-privilege callers win, and within a priority
class the key with the fewest active requests is selected first. Waiting is
always bounded and cancellation-safe (client disconnects must not leak slots).
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field


class QueueFullError(Exception):
    pass


class QueueTimeoutError(Exception):
    pass


@dataclass(slots=True)
class Lease:
    gate: "PriorityGate"
    priority: int
    queue_ms: float
    key_id: str
    _released: bool = field(default=False)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.gate.release(self.key_id)

    async def __aenter__(self) -> "Lease":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()


class PriorityGate:
    def __init__(self, limit: int, max_queue: int = 64):
        self.limit = max(1, limit)
        self.max_queue = max(1, max_queue)
        self._in_use = 0
        self._active_by_key: dict[str, int] = {}
        self._waiters: list[tuple[int, int, str, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._lock = asyncio.Lock()
        self.acquired_total = 0
        self.timed_out_total = 0

    @property
    def in_use(self) -> int:
        return self._in_use

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    async def acquire(self, *, priority: int = 5, key_id: str = "-", timeout: float = 120) -> Lease:
        started = time.monotonic()
        async with self._lock:
            if self._in_use < self.limit:
                self._take(key_id)
                return Lease(self, priority, 0.0, key_id)
            if len(self._waiters) >= self.max_queue:
                raise QueueFullError("Model request queue is full")
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append((priority, next(self._seq), key_id, fut))
        try:
            await asyncio.wait_for(fut, timeout=max(0.1, timeout))
        except asyncio.TimeoutError as exc:
            async with self._lock:
                self._waiters = [w for w in self._waiters if w[3] is not fut]
                self.timed_out_total += 1
            raise QueueTimeoutError("Timed out waiting for model capacity") from exc
        except asyncio.CancelledError:
            async with self._lock:
                self._waiters = [w for w in self._waiters if w[3] is not fut]
            raise
        return Lease(self, priority, (time.monotonic() - started) * 1000, key_id)

    def _take(self, key_id: str) -> None:
        self._in_use += 1
        self._active_by_key[key_id] = self._active_by_key.get(key_id, 0) + 1
        self.acquired_total += 1

    def release(self, key_id: str) -> None:
        self._in_use = max(0, self._in_use - 1)
        count = self._active_by_key.get(key_id, 0) - 1
        if count > 0:
            self._active_by_key[key_id] = count
        else:
            self._active_by_key.pop(key_id, None)
        if not self._waiters:
            return
        # Prefer priority, then the least-served key at that priority, then FIFO.
        best_idx = 0
        best_key = None
        for idx, (priority, seq, waiter_key, fut) in enumerate(self._waiters):
            if fut.done():
                continue
            candidate = (
                -priority,
                self._active_by_key.get(waiter_key, 0),
                seq,
            )
            if best_key is None or candidate < best_key:
                best_key = candidate
                best_idx = idx
        if best_key is None:
            self._waiters.clear()
            return
        _, _, waiter_key, fut = self._waiters.pop(best_idx)
        self._take(waiter_key)
        if not fut.done():
            fut.set_result(None)

    def snapshot(self) -> dict[str, int]:
        return {
            "limit": self.limit,
            "in_use": self._in_use,
            "waiting": len(self._waiters),
            "acquired_total": self.acquired_total,
            "timed_out_total": self.timed_out_total,
        }
