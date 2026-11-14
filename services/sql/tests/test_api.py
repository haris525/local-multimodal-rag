"""The sql service opens only the caller's tables, and only for rag."""

import duckdb
import pytest
from fastapi.testclient import TestClient

from conftest import KEYS, TABLES
from rag_common.testing import new_user_id

from sql_sandbox.api import app

A, B = new_user_id(), new_user_id()
DOC = "a" * 32


@pytest.fixture(scope="module")
def client():
    folder = TABLES / A
    folder.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(folder / f"{DOC}.duckdb")) as con:
        con.execute("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'North', 10.0)) t(_row, region, amount)")
    return TestClient(app)


def query(client, headers, doc_id=DOC):
    return client.post("/v1/query", json={"doc_id": doc_id, "sql": "SELECT region, amount FROM sales"}, headers=headers)


def test_a_full_sandbox_answers_503_fast_instead_of_queueing_past_rags_timeout(client, monkeypatch):
    import asyncio
    import time

    import sql_sandbox.api as api

    monkeypatch.setattr(api, "_running", asyncio.Semaphore(0))  # every slot taken
    monkeypatch.setattr(api, "QUEUE_WAIT_S", 0.05)
    started = time.perf_counter()
    r = query(client, KEYS.headers(A, issuer="rag"))
    assert r.status_code == 503 and "busy" in r.json()["detail"] and time.perf_counter() - started < 5


def test_a_slot_is_given_back_after_a_failed_query(client, monkeypatch):
    import asyncio

    import sql_sandbox.api as api

    monkeypatch.setattr(api, "_running", asyncio.Semaphore(1))
    monkeypatch.setattr(api, "QUEUE_WAIT_S", 0.5)
    bad = client.post("/v1/query", json={"doc_id": DOC, "sql": "SELECT nope FROM sales"}, headers=KEYS.headers(A, issuer="rag"))
    assert bad.status_code == 400 and query(client, KEYS.headers(A, issuer="rag")).status_code == 200


def test_the_owner_queries_their_tables_through_rag(client):
    r = query(client, KEYS.headers(A, issuer="rag"))
    assert r.status_code == 200 and r.json()["rows"] == [["North", 10.0]]


def test_another_users_document_does_not_exist(client):
    assert query(client, KEYS.headers(B, issuer="rag")).status_code == 404
    assert query(client, KEYS.headers(A, issuer="rag"), "b" * 32).status_code == 404


def test_only_rag_may_call_and_only_for_a_user(client):
    assert query(client, {}).status_code == 401
    assert query(client, KEYS.headers(A)).status_code == 401  # signed by auth: sql accepts rag only
    version, payload, signature = KEYS.token(A, issuer="rag").split(".")
    forged = ".".join([version, payload, signature[:10] + ("B" if signature[10] != "B" else "C") + signature[11:]])
    assert query(client, {"x-identity": forged}).status_code == 401
    token = KEYS.token(A, issuer="rag")
    assert query(client, [("x-identity", token), ("x-identity", token)]).status_code == 401  # one header only


def test_a_document_id_cannot_leave_the_owners_folder(client):
    r = client.post("/v1/query", json={"doc_id": "../" + A + "/" + DOC, "sql": "SELECT 1"},
                    headers=KEYS.headers(B, issuer="rag"))
    assert r.status_code == 422


def test_the_owner_gets_the_whole_tables_file_through_rag(client):
    r = client.get(f"/v1/tables/{DOC}", headers=KEYS.headers(A, issuer="rag"))
    assert r.status_code == 200 and r.content == (TABLES / A / f"{DOC}.duckdb").read_bytes()
    assert r.headers["content-type"] == "application/octet-stream" and int(r.headers["content-length"]) == len(r.content)


def test_the_tables_file_of_another_user_or_without_rag_does_not_exist(client):
    assert client.get(f"/v1/tables/{DOC}", headers=KEYS.headers(B, issuer="rag")).status_code == 404
    assert client.get(f"/v1/tables/{'b' * 32}", headers=KEYS.headers(A, issuer="rag")).status_code == 404
    assert client.get(f"/v1/tables/{DOC}", headers=KEYS.headers(A)).status_code == 401  # signed by auth
    assert client.get(f"/v1/tables/{DOC}").status_code == 401
    assert client.get("/v1/tables/..%2F" + DOC, headers=KEYS.headers(A, issuer="rag")).status_code in (404, 422)


def test_tables_file_streams_are_capped_and_given_back(client, monkeypatch):
    import asyncio

    import sql_sandbox.api as api

    monkeypatch.setattr(api, "_streams", asyncio.Semaphore(1))
    monkeypatch.setattr(api, "FILE_QUEUE_WAIT_S", 0.05)
    for _ in range(3):  # each finished stream gives its place back
        assert client.get(f"/v1/tables/{DOC}", headers=KEYS.headers(A, issuer="rag")).status_code == 200
    asyncio.run(api._streams.acquire())  # one stream in progress
    r = client.get(f"/v1/tables/{DOC}", headers=KEYS.headers(A, issuer="rag"))
    assert r.status_code == 503 and "SQL_MAX_FILE_STREAMS" in r.json()["detail"]


def test_a_stream_the_client_leaves_before_it_starts_closes_its_file(monkeypatch, tmp_path):
    import asyncio

    import sql_sandbox.api as api

    closed = []
    data = tmp_path / "f.bin"
    data.write_bytes(b"x" * 10)
    f = open(data, "rb")  # noqa: SIM115

    async def never():
        yield b""

    async def scenario():
        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            raise OSError("client gone")  # the client left before the first byte

        reply = api.ClosingStream(never(), lambda: (f.close(), closed.append(True)))
        try:
            await reply({"type": "http", "method": "GET", "path": "/", "headers": []}, receive, send)
        except OSError:
            pass

    asyncio.run(scenario())
    assert closed == [True] and f.closed


def test_one_stream_per_user_and_a_short_queue_before_503(monkeypatch):
    import asyncio

    import sql_sandbox.api as api

    monkeypatch.setattr(api, "_streams", asyncio.Semaphore(2))
    monkeypatch.setattr(api, "FILE_QUEUE_WAIT_S", 0.3)
    monkeypatch.setattr(api, "_user_streams", {})

    async def scenario():
        first = await api._stream_place(A)  # A streams
        other = await api._stream_place(B)  # B is not held up by A
        waiting = asyncio.ensure_future(api._stream_place(A))  # A's second waits in the queue
        await asyncio.sleep(0.1)
        assert not waiting.done()
        first()  # A's first ends: the second gets its place
        second = await waiting
        try:
            await api._stream_place(A)
            timed_out = False
        except Exception as exc:  # noqa: BLE001
            timed_out = getattr(exc, "status_code", None) == 503
        second()
        other()
        return timed_out, dict(api._user_streams), api._streams._value

    timed_out, users, free = asyncio.run(scenario())
    assert timed_out is True and users == {} and free == 2  # everything given back
