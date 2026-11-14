"""Serialize GPU work with a bounded queue: the running job and the waiting ones count toward max_waiting
(MODELS_MAX_WAITING), and callers beyond it get 503 at once.

A job runs in a worker thread (asyncio.to_thread), which cannot be stopped. A caller cancelled while its
job runs therefore keeps the gate until the job has finished, then lets the cancellation through, so the
next job never starts on the GPU beside one that is still running (docs/STATE_MACHINES.md, M14)."""

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Busy(Exception):
    pass


class GpuGate:
    def __init__(self, max_waiting: int):
        """max_waiting counts the running job and the waiting ones."""
        self._lock = asyncio.Lock()
        self._max_waiting = max_waiting
        self._waiting = 0

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        if self._waiting >= self._max_waiting:
            raise Busy()
        self._waiting += 1
        try:
            async with self._lock:
                job = asyncio.ensure_future(asyncio.to_thread(fn, *args))
                try:
                    return await asyncio.shield(job)
                except asyncio.CancelledError:
                    while not job.done():  # hold the gate until the thread is done, however often cancelled
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.wait({job})
                    if not job.cancelled():
                        job.exception()  # retrieved: the caller is gone and nobody else looks at it
                    raise
        finally:
            self._waiting -= 1
