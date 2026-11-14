"""One retry after a backoff for transient failures only, logged at WARNING then ERROR."""

import asyncio
import logging

import httpx
import pytest

from rag_common.http import ServiceError
from rag_common.retry import retry_once, transient

log = logging.getLogger("test.retry")


def run(outcomes: list, **kwargs):
    calls = []

    async def call():
        calls.append(1)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return asyncio.run(retry_once(call, what="embedding", backoff_s=0, log=log, **kwargs)), len(calls)


def test_transient_failures_are_retried_once_and_logged(caplog):
    with caplog.at_level(logging.WARNING):
        assert run([ServiceError(503, "queue full"), "ok"]) == ("ok", 2)
    assert "embedding failed; retrying once" in caplog.text
    with caplog.at_level(logging.WARNING), pytest.raises(ServiceError):
        run([ServiceError(504, "slow"), ServiceError(502, "down")])
    assert "embedding failed again after a retry" in caplog.text


def test_request_errors_are_not_retried():
    with pytest.raises(ServiceError):
        run([ServiceError(400, "bad"), "never"])
    assert not transient(ServiceError(404, "gone")) and not transient(ValueError("x"))
    assert transient(httpx.ReadTimeout("slow")) and transient(ServiceError(429, "busy"))
    with pytest.raises(ServiceError):
        run([ServiceError(408, "query timed out"), "never"], should_retry=lambda e: e.status != 408)
