"""Run parsing in a long-lived child process that is killed on timeout and replaced on a crash.

The child keeps Docling's models loaded between documents. A pathological file costs one timeout, or
one crashed child reported at once (not after the timeout), and a fresh child; never the worker itself.
"""

import multiprocessing as mp
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import psutil


class ParserCrashed(RuntimeError):
    pass


class IsolatedRunner:
    def __init__(self, max_tasks_per_child: int = 50, max_cache_mb: int = 1024):
        self._ctx = mp.get_context("spawn")
        self._max_tasks = max_tasks_per_child
        self._max_cache_bytes = max_cache_mb * 1024 * 1024
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def run(self, fn: Callable[..., Any], *args: Any, timeout: float) -> Any:
        """Blocking call; use from async code through asyncio.to_thread. `fn` must be a module-level
        function. Its own exceptions are re-raised here unchanged."""
        with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(1, mp_context=self._ctx, max_tasks_per_child=self._max_tasks)
            try:
                # An idle child can die between jobs: submit itself can raise BrokenProcessPool.
                future = self._pool.submit(fn, *args)
                return future.result(timeout)
            except TimeoutError:
                self._kill()
                raise TimeoutError(f"parsing took longer than {timeout:.0f} s") from None
            except BrokenProcessPool:
                self._kill()
                raise ParserCrashed("the parser process died (out of memory or a fault in a native library)") from None
            finally:
                # Release large native caches after both successful and failed parses, before
                # embedding/the next file. A successful result has already reached the parent.
                if self._pool is not None:
                    try:
                        retained = sum(psutil.Process(p.pid).memory_info().rss
                                       for p in (self._pool._processes or {}).values())
                    except psutil.NoSuchProcess:
                        retained = 0
                    if self._max_cache_bytes == 0 or retained > self._max_cache_bytes:
                        self._kill()

    def _kill(self) -> None:
        if self._pool is not None:
            for process in list((self._pool._processes or {}).values()):  # no public kill before Python 3.14
                process.kill()
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def close(self) -> None:
        pool = self._pool
        if pool is not None:  # a parse in progress holds the lock: kill its child so it fails at once
            for process in list((pool._processes or {}).values()):
                process.kill()
        with self._lock:
            self._kill()
