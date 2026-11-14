"""POST /v1/execute: frames, caps, one run at a time, and a failed clean-up stops the server."""

import asyncio
import dataclasses
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_common.sandbox import MEDIA_TYPE, encode

import sandbox.api as api
from sandbox import executor


RAG = ("172.31.241.2", 40000)


@pytest.fixture
def client(limits, monkeypatch):
    monkeypatch.setattr(api, "limits", limits)
    monkeypatch.setattr(api, "_turn", asyncio.Lock())
    monkeypatch.setitem(api.state, "restarting", False)
    with TestClient(api.app, client=RAG) as c:  # rag's address on the sandbox network
        yield c


def post(client, code="print('hi')", timeout_s=30, files=()):
    body = b"".join(encode(code, timeout_s, list(files)))
    return client.post("/v1/execute", content=body, headers={"content-type": MEDIA_TYPE})


def test_a_run_answers_with_its_output(client):
    r = post(client, "import json; print(json.load(open('data/m.json'))['n'] + 1)", files=[("data/m.json", b'{"n": 41}')])
    assert r.status_code == 200
    body = r.json()
    assert body["exit_code"] == 0 and body["stdout"].strip() == "42" and body["files"] == [] and body["error"] is None
    assert set(body) == {"exit_code", "signal", "timed_out", "duration_ms", "timeout_s", "stdout", "stdout_bytes",
                         "stderr", "stderr_bytes", "files", "files_total", "error"}


@pytest.mark.parametrize("body, words", [
    (b"\x00\x00", "shorter"),
    (b"\x00\x00\x00\x05nojso", "not UTF-8 JSON"),
])
def test_a_malformed_frame_is_a_400(client, body, words):
    r = client.post("/v1/execute", content=body)
    assert r.status_code == 400 and words in r.json()["detail"]


def test_an_unsafe_path_is_a_400_and_nothing_runs(client, monkeypatch):
    import json
    import struct

    ran = []
    monkeypatch.setattr(executor, "execute", lambda *a: ran.append(a))
    header = json.dumps({"version": 1, "code": "print(1)", "timeout_s": 5,
                         "files": [{"path": "../../etc/x", "size": 1}]}).encode()
    r = client.post("/v1/execute", content=struct.pack(">I", len(header)) + header + b"x")
    assert r.status_code == 400 and "unsafe file path" in r.json()["detail"] and ran == []


def test_code_and_files_over_their_caps_are_a_413(client, monkeypatch, limits):
    monkeypatch.setattr(api, "limits", dataclasses.replace(limits, max_code_bytes=100, max_input_bytes=1000))
    r = post(client, "#" * 200)
    assert r.status_code == 413 and "SANDBOX_MAX_CODE_KB" in r.json()["detail"]
    r = post(client, files=[("data/a.bin", b"x" * 600), ("data/b.bin", b"x" * 600)])
    assert r.status_code == 413 and "SANDBOX_MAX_INPUT_MB" in r.json()["detail"]
    r = client.post("/v1/execute", content=b"x", headers={"content-length": str(10 ** 9)})
    assert r.status_code == 413


def test_a_time_limit_above_the_ceiling_is_refused_not_lowered(client):
    r = post(client, timeout_s=10_000)
    assert r.status_code == 400 and "SANDBOX_MAX_TIMEOUT_S" in r.json()["detail"]


def test_a_request_waits_for_the_run_before_it_then_gets_a_503(client, monkeypatch, limits):
    monkeypatch.setattr(api, "limits", dataclasses.replace(limits, queue_wait_s=0.2))
    lock = asyncio.Lock()
    monkeypatch.setattr(api, "_turn", lock)
    asyncio.run(lock.acquire())  # another run holds the turn
    r = post(client)
    assert r.status_code == 503 and "SANDBOX_QUEUE_WAIT_S" in r.json()["detail"]


def test_the_turn_is_held_until_the_run_ends_even_when_the_caller_gives_up(limits, monkeypatch):
    """rag timing out must never let a second run start next to one still running."""
    release = threading.Event()
    started = threading.Event()

    def slow(job, lim, control=None):
        started.set()
        release.wait(10)
        return {"duration_ms": 1, "exit_code": 0, "signal": None, "timed_out": False, "files_total": 0,
                "stdout_bytes": 0}

    monkeypatch.setattr(api, "limits", limits)
    monkeypatch.setattr(executor, "execute", slow)

    async def scenario():
        lock = asyncio.Lock()
        monkeypatch.setattr(api, "_turn", lock)
        transport = httpx.ASGITransport(app=api.app, client=("172.30.0.2", 40000))  # rag's address, not loopback
        async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as c:
            body = b"".join(encode("print(1)", 5, []))
            first = asyncio.ensure_future(c.post("/v1/execute", content=body))
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set(), first.result() if first.done() else "the run never started"
            first.cancel()  # the caller went away mid-run
            await asyncio.sleep(0.05)
            held = lock.locked()
            release.set()
            for _ in range(200):
                if not lock.locked():
                    break
                await asyncio.sleep(0.01)
            return held, lock.locked()

    held, after = asyncio.run(scenario())
    assert held is True and after is False


def test_a_failed_clean_up_stops_the_server_and_refuses_runs_meanwhile(client, monkeypatch):
    stopped = []

    def failing(job, lim, control=None):
        raise executor.CleanupFailed("/scratch still holds ['run-x']")

    monkeypatch.setattr(executor, "execute", failing)
    monkeypatch.setattr(api, "terminate", lambda: stopped.append(True))
    r = post(client)
    assert r.status_code == 500 and "restarts" in r.json()["detail"] and stopped == [True]
    r = post(client)
    assert r.status_code == 503 and "restarting" in r.json()["detail"]


def test_files_that_do_not_fit_the_scratch_space_are_a_507(client, monkeypatch):
    def full(job, lim, control=None):
        raise executor.ScratchFull("No space left on device")

    monkeypatch.setattr(executor, "execute", full)
    r = post(client)
    assert r.status_code == 507 and "SANDBOX_SCRATCH_SIZE" in r.json()["detail"]


def test_health_needs_nothing(client):
    assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize("client_addr, server_addr", [(("127.0.0.1", 5000), ("127.0.0.1", 8008)),
                                                      (("127.0.0.2", 5000), ("172.30.0.3", 8008)),
                                                      (("::ffff:127.0.0.5", 5000), ("172.30.0.3", 8008)),
                                                      (("172.30.0.3", 5000), ("172.30.0.3", 8008)),
                                                      (("::1", 5000), ("::1", 8008)),
                                                      (("testclient", 5000), ("172.30.0.3", 8008))])
def test_a_run_cannot_ask_for_a_run_of_its_own(limits, monkeypatch, client_addr, server_addr):
    """A connection from inside the container (loopback, or the container's own address) is refused before
    anything is read, so a run cannot queue code to run after its own clean-up."""
    monkeypatch.setattr(api, "limits", limits)
    monkeypatch.setattr(api, "_turn", asyncio.Lock())
    ran = []
    monkeypatch.setattr(executor, "execute", lambda *a: ran.append(a))

    async def scenario():
        async def app(scope, receive, send):
            scope = {**scope, "client": client_addr, "server": server_addr}
            await api.app(scope, receive, send)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sandbox") as c:
            return await c.post("/v1/execute", content=b"".join(encode("print(1)", 5, [])))

    r = asyncio.run(scenario())
    assert r.status_code == 403 and ran == []


def test_rags_address_is_not_inside(client):
    from starlette.requests import Request

    scope = {"type": "http", "client": ("172.30.0.2", 40000), "server": ("172.30.0.3", 8008), "headers": []}
    assert api.from_inside(Request(scope)) is False


def test_a_request_waiting_while_a_clean_up_fails_never_runs(limits, monkeypatch):
    """The run before it left something behind: the request that was waiting for the turn gets 503, not a run in
    a container that is not clean."""
    release, started, ran = threading.Event(), threading.Event(), []

    def failing(job, lim, control=None):
        if ran:
            raise AssertionError("a second run started after a failed clean-up")
        ran.append(job)
        started.set()
        release.wait(10)
        raise executor.CleanupFailed("/scratch still holds ['run-x']")

    monkeypatch.setattr(api, "limits", limits)
    monkeypatch.setattr(api, "terminate", lambda: None)
    monkeypatch.setitem(api.state, "restarting", False)
    monkeypatch.setattr(executor, "execute", failing)

    async def scenario():
        monkeypatch.setattr(api, "_turn", asyncio.Lock())
        transport = httpx.ASGITransport(app=api.app, client=("172.30.0.2", 40000))
        async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as c:
            body = b"".join(encode("print(1)", 5, []))
            first = asyncio.ensure_future(c.post("/v1/execute", content=body))
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            second = asyncio.ensure_future(c.post("/v1/execute", content=body))  # waits for the turn
            await asyncio.sleep(0.05)
            release.set()
            return (await first).status_code, (await second).status_code

    assert asyncio.run(scenario()) == (500, 503) and len(ran) == 1


def test_another_address_of_this_container_is_inside(monkeypatch):
    from starlette.requests import Request

    monkeypatch.setitem(api.state, "own", {api._address("172.31.241.3"), api._address("10.9.0.4")})
    scope = {"type": "http", "client": ("10.9.0.4", 40000), "server": ("172.31.241.3", 8008), "headers": []}
    assert api.from_inside(Request(scope)) is True


# --- cancel --------------------------------------------------------------------------------------------------------


def _scenario(limits, monkeypatch, fake):
    """An app with its own turn, `fake` as the executor; returns an async client factory."""
    monkeypatch.setattr(api, "limits", limits)
    monkeypatch.setattr(executor, "execute", fake)
    monkeypatch.setitem(api.state, "restarting", False)
    monkeypatch.setitem(api.state, "running", None)
    monkeypatch.setattr(api, "_cancelled", __import__("collections").OrderedDict())

    def client():
        transport = httpx.ASGITransport(app=api.app, client=RAG)
        return httpx.AsyncClient(transport=transport, base_url="http://sandbox")
    return client


def test_a_running_run_is_killed_on_cancel_and_the_answer_waits_for_its_end(limits, monkeypatch):
    started = threading.Event()

    def blocking(job, lim, control=None):
        started.set()
        for _ in range(500):  # stands for a child process; the real one is killed by Control.cancel
            if control.cancelled:
                break
            time.sleep(0.01)
        return {"duration_ms": 1, "exit_code": None, "signal": "SIGKILL", "timed_out": False, "files_total": 0,
                "stdout_bytes": 0, "error": executor.failure(False, -9, "SIGKILL", "", 5, lim, control.cancelled)}

    make = _scenario(limits, monkeypatch, blocking)

    async def go():
        monkeypatch.setattr(api, "_turn", asyncio.Lock())
        async with make() as c:
            body = b"".join(encode("print(1)", 5, []))
            run = asyncio.ensure_future(c.post("/v1/execute", content=body, headers={"x-run-id": "b" * 32}))
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            answer = (await c.post(f"/v1/runs/{'b' * 32}/cancel")).json()
            reply = (await run).json()
            return answer, reply, api._turn.locked(), api.state["running"]

    answer, reply, locked, running = asyncio.run(go())
    assert answer == {"ended": True, "started": True} and "cancelled" in reply["error"]
    assert locked is False and running is None


def test_a_run_cancelled_before_its_turn_never_runs(limits, monkeypatch):
    ran = []
    make = _scenario(limits, monkeypatch, lambda *a: ran.append(a))

    async def go():
        monkeypatch.setattr(api, "_turn", asyncio.Lock())
        async with make() as c:
            answer = (await c.post(f"/v1/runs/{'c' * 32}/cancel")).json()
            r = await c.post("/v1/execute", content=b"".join(encode("print(1)", 5, [])), headers={"x-run-id": "c" * 32})
            return answer, r.status_code

    answer, status = asyncio.run(go())
    assert answer == {"ended": True, "started": False} and status == 409 and ran == []


def test_cancel_needs_a_run_id_and_comes_from_rag_only(client):
    assert client.post("/v1/runs/not-an-id/cancel").status_code == 400

    async def inside():
        transport = httpx.ASGITransport(app=api.app, client=("127.0.0.1", 1))
        async with httpx.AsyncClient(transport=transport, base_url="http://sandbox") as c:
            return (await c.post(f"/v1/runs/{'d' * 32}/cancel")).status_code

    assert asyncio.run(inside()) == 403
