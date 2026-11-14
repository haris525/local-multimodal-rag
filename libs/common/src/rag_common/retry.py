"""One retry after a backoff for transient failures of a backend call, logged with the request ID.

Transient: the backend answered 429, 500, 502, 503 or 504 (a busy GPU gate answers 503), or did not answer
in time or at all. Everything else (a 4xx about the request itself, a bad reply) is not retried: trying the
same request again would fail the same way. The first failure is logged at WARNING, a second at ERROR, and
the second error is raised to the caller, which decides what the user is told.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from rag_common.http import ServiceError
from rag_common.logging import log_fields

T = TypeVar("T")
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


def transient(exc: BaseException) -> bool:
    if isinstance(exc, ServiceError):
        return exc.status in TRANSIENT_STATUS
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_STATUS
    return isinstance(exc, httpx.TimeoutException | httpx.TransportError)


def describe(exc: BaseException) -> str:
    if isinstance(exc, ServiceError):
        return f"{exc.status}: {exc.detail}"[:300]
    return f"{type(exc).__name__}: {exc}"[:300]


async def retry_once(call: Callable[[], Awaitable[T]], *, what: str, backoff_s: float, log: logging.Logger,
                     should_retry: Callable[[BaseException], bool] = transient) -> T:
    """Run `call`; on a transient failure wait `backoff_s` and run it once more."""
    try:
        return await call()
    except Exception as exc:
        if not should_retry(exc):
            raise
        log_fields(log, f"{what} failed; retrying once", logging.WARNING, error=describe(exc), backoff_s=backoff_s)
    await asyncio.sleep(backoff_s)
    try:
        return await call()
    except Exception as exc:
        log_fields(log, f"{what} failed again after a retry", logging.ERROR, error=describe(exc))
        raise
