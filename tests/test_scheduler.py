from __future__ import annotations

import asyncio

import pytest

from app.scheduler import PriorityGate, QueueFullError, QueueTimeoutError


@pytest.mark.asyncio
async def test_gate_enforces_concurrency():
    gate = PriorityGate(limit=2)
    first = await gate.acquire(key_id="a")
    second = await gate.acquire(key_id="b")
    assert gate.in_use == 2
    assert not first._released
    first.release()
    assert gate.in_use == 1
    second.release()
    assert gate.in_use == 0


@pytest.mark.asyncio
async def test_gate_priority_order():
    gate = PriorityGate(limit=1)
    held = await gate.acquire(priority=5, key_id="holder")
    order = []

    async def waiter(priority: int, name: str):
        lease = await gate.acquire(priority=priority, key_id=name, timeout=5)
        order.append(name)
        await asyncio.sleep(0.01)
        lease.release()

    low = asyncio.create_task(waiter(1, "low"))
    await asyncio.sleep(0.02)
    high = asyncio.create_task(waiter(9, "high"))
    await asyncio.sleep(0.02)
    held.release()
    await asyncio.gather(low, high)
    assert order[0] == "high"


@pytest.mark.asyncio
async def test_gate_fairness_within_priority():
    gate = PriorityGate(limit=1)
    held = await gate.acquire(priority=5, key_id="holder")
    order = []

    async def first():
        lease = await gate.acquire(priority=5, key_id="a", timeout=5)
        order.append("a")
        lease.release()

    async def second():
        await asyncio.sleep(0.02)
        lease = await gate.acquire(priority=5, key_id="b", timeout=5)
        order.append("b")
        lease.release()

    tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
    await asyncio.sleep(0.05)
    held.release()
    await asyncio.gather(*tasks)
    assert order == ["a", "b"]


@pytest.mark.asyncio
async def test_gate_timeout():
    gate = PriorityGate(limit=1)
    held = await gate.acquire()
    with pytest.raises(QueueTimeoutError):
        await gate.acquire(timeout=0.05)
    assert gate.waiting == 0
    held.release()


@pytest.mark.asyncio
async def test_gate_queue_full():
    gate = PriorityGate(limit=1, max_queue=1)
    held = await gate.acquire()
    waiter = asyncio.create_task(gate.acquire(timeout=2))
    await asyncio.sleep(0.02)
    with pytest.raises(QueueFullError):
        await gate.acquire(timeout=0.1)
    held.release()
    await waiter


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_slot():
    gate = PriorityGate(limit=1)
    held = await gate.acquire()
    waiter = asyncio.create_task(gate.acquire(timeout=5))
    await asyncio.sleep(0.02)
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass
    held.release()
    assert gate.in_use == 0
    lease = await gate.acquire(timeout=0.2)
    lease.release()
