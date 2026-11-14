"""Requests from the sandbox network are refused before anything else, logged once a minute at most; /readyz answers
from a short cache, so callers share one probe of the backends."""

import asyncio
import logging

import httpx

import rag.api as api
from rag.config import Settings
from rag.guard import SubnetGuard


def call(app, client: tuple[str, int], path: str = "/healthz") -> httpx.Response:
    async def go():
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url="http://rag") as c:
            return await c.get(path)
    return asyncio.run(go())


def inner_app():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    return app, seen


def test_a_request_from_the_sandbox_network_never_reaches_rag():
    app, seen = inner_app()
    guard = SubnetGuard(app, subnet="172.31.241.0/24")
    for host in ("172.31.241.2", "172.31.241.254", "::ffff:172.31.241.7"):
        r = call(guard, (host, 40000), "/readyz")
        assert r.status_code == 403 and r.json() == {"detail": "forbidden"}
    assert seen == []
    assert call(guard, ("172.30.0.5", 40000)).status_code == 200 and seen == ["/healthz"]  # the gateway's network


def test_an_empty_subnet_turns_the_guard_off():
    app, seen = inner_app()
    assert call(SubnetGuard(app, subnet=""), ("172.31.241.2", 1)).status_code == 200 and seen == ["/healthz"]


def test_refusals_are_counted_and_logged_once_a_minute(caplog):
    app, _ = inner_app()
    now = [1000.0]
    guard = SubnetGuard(app, subnet="172.31.241.0/24", clock=lambda: now[0])
    with caplog.at_level(logging.WARNING, logger="rag.guard"):
        for _ in range(50):
            call(guard, ("172.31.241.9", 1))
        now[0] += 61
        call(guard, ("172.31.241.9", 1))
    lines = [r for r in caplog.records if r.name == "rag.guard"]
    assert len(lines) == 2  # the first refusal, then one line a minute later with the count since
    assert lines[0].fields["count"] == 1 and lines[1].fields["count"] == 50


def test_the_guard_is_rags_outermost_middleware():
    assert api.app.user_middleware[0].cls is SubnetGuard


def test_readiness_answers_from_its_cache_for_a_few_seconds(monkeypatch):
    probes = []

    async def probe():
        probes.append(1)
        return {"index": True}

    monkeypatch.setattr(api, "_probe", probe)

    async def go(cache_s: float):
        monkeypatch.setattr(api, "settings", Settings(ready_cache_s=cache_s))
        api._ready.update(loop=None)
        return [await api.readiness() for _ in range(5)] + list(await asyncio.gather(*[api.readiness() for _ in range(5)]))

    assert asyncio.run(go(5.0)) == [{"index": True}] * 10 and len(probes) == 1
    probes.clear()
    asyncio.run(go(0.0))
    assert len(probes) >= 5  # no cache: every sequential call probes
