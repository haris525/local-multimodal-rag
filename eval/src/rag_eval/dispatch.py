"""Benchmark requests one at a time or several at once (--concurrency).

`dispatch` sends the jobs in the order given, keeps at most `concurrency` of them in flight and hands each result
to `done` in the caller's thread as it arrives, so run records, the stream file, progress lines and pass counts
are only ever touched by one thread. With concurrency 1 it is the plain loop the benchmark always ran: the next
request is sent when the previous one has finished.

Worker threads are daemons and the caller waits in short steps, so Ctrl+C stops the run at once (the reports for
the runs so far are written) and abandons the requests still in flight instead of waiting minutes for them. A
worker that raises stops the dispatch of new jobs; the requests already in flight still finish and reach `done`,
then the exception is raised in the caller, as the sequential loop would have raised it.
"""

import queue
import threading
from collections.abc import Callable, Sequence
from typing import TypeVar

J = TypeVar("J")
R = TypeVar("R")


def dispatch(jobs: Sequence[J], work: Callable[[J], R], done: Callable[[int, J, R, int], None], concurrency: int = 1,
             poll_s: float = 0.2) -> None:
    """Run work(job) for every job, at most `concurrency` at a time, in dispatch order. done(k, job, result, others)
    gets k, the job's 1-based dispatch position, and others, the number of other requests in flight when it was
    sent (always 0 with concurrency 1)."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if concurrency == 1:
        for k, job in enumerate(jobs, 1):
            done(k, job, work(job), 0)
        return

    results: queue.Queue = queue.Queue()
    running = 0
    failure: BaseException | None = None

    def worker(k: int, job: J, others: int) -> None:
        try:
            results.put((k, job, others, work(job), None))
        except BaseException as exc:  # noqa: BLE001  handed to the caller's thread, never lost in a worker
            results.put((k, job, others, None, exc))

    def collect() -> None:
        nonlocal running, failure
        while True:
            try:
                k, job, others, result, exc = results.get(timeout=poll_s)  # a timeout keeps Ctrl+C responsive
                break
            except queue.Empty:
                continue
        running -= 1
        if exc is not None:
            failure = failure or exc
            return
        done(k, job, result, others)

    for k, job in enumerate(jobs, 1):
        while running >= concurrency:
            collect()
        if failure is not None:
            break
        threading.Thread(target=worker, args=(k, job, running), daemon=True, name=f"bench-request-{k}").start()
        running += 1
    while running:
        collect()
    if failure is not None:
        raise failure
