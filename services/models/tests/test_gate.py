import asyncio
import time

import pytest

from models_svc.gate import Busy, GpuGate


def test_gate_serializes_work_and_rejects_beyond_the_wait_bound():
    async def scenario():
        gate = GpuGate(max_waiting=2)
        slow = [asyncio.create_task(gate.run(time.sleep, 0.2)) for _ in range(2)]
        await asyncio.sleep(0.05)
        with pytest.raises(Busy):
            await gate.run(time.sleep, 0)
        started = time.perf_counter()
        await asyncio.gather(*slow)
        return time.perf_counter() - started

    assert asyncio.run(scenario()) >= 0.3  # the two jobs ran one after the other


def test_G1_a_cancelled_caller_holds_the_gate_until_its_gpu_job_ends():
    """Proof G1 turned around: the thread of a cancelled caller's job still runs, so the next job waits."""
    log = []

    def job(name, seconds):
        log.append(("start", name))
        time.sleep(seconds)
        log.append(("end", name))
        return name

    async def scenario():
        gate = GpuGate(max_waiting=4)
        first = asyncio.create_task(gate.run(job, "first", 0.3))
        await asyncio.sleep(0.05)
        first.cancel()
        second = asyncio.create_task(gate.run(job, "second", 0.01))
        await asyncio.sleep(0.05)
        first.cancel()  # cancelled again while it waits for its thread: still holds the gate
        with pytest.raises(asyncio.CancelledError):
            await first
        return await second

    assert asyncio.run(scenario()) == "second"
    assert log == [("start", "first"), ("end", "first"), ("start", "second"), ("end", "second")]


def test_a_cancelled_caller_that_was_still_waiting_leaves_at_once():
    async def scenario():
        gate = GpuGate(max_waiting=4)
        running = asyncio.create_task(gate.run(time.sleep, 0.2))
        await asyncio.sleep(0.02)
        waiting = asyncio.create_task(gate.run(time.sleep, 0))
        await asyncio.sleep(0.02)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert gate._waiting == 1 and not running.done()
        await running

    asyncio.run(scenario())
