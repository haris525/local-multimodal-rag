"""Threads and memories per user: every store method and route treats another user's data as missing."""

import pytest
from fastapi.testclient import TestClient

from conftest import KEYS
from rag_common.contracts import NewMessage
from rag_common.testing import new_user_id

from threads.store import Store

A, B = new_user_id(), new_user_id()


def turn(q: str, a: str) -> list[NewMessage]:
    return [NewMessage(role="user", content=q), NewMessage(role="assistant", content=a)]


def test_every_store_method_returns_none_or_false_for_another_users_data(tmp_path):
    store = Store(tmp_path / "t.db")
    t = store.create(A, "A's chat", ["d" * 32]).thread_id
    store.append(A, t, turn("q", "a"), None)
    mem = store.add_memory(A, "A works in finance.", "user", t)

    assert store.recent(B, 10) == []
    assert store.get(B, t) is None and store.context(B, t) is None
    assert store.update(B, t, "renamed by B", ["e" * 32]) is None
    assert store.append(B, t, turn("injected", "x"), None) is None
    assert store.set_summary(B, t, "B's summary", 99) is False
    assert store.delete(B, t) is False
    assert store.memories(B) == []
    assert store.delete_memory(B, mem.memory_id) is False
    with pytest.raises(KeyError):
        store.replace_memory(B, mem.memory_id, "B rewrote it.", "user")

    detail = store.get(A, t)  # nothing of A's changed
    assert detail.thread.title == "A's chat" and [m.content for m in detail.messages] == ["q", "a"]
    assert detail.thread.summarized == 0 and [m.text for m in store.memories(A)] == ["A works in finance."]


def test_store_refuses_a_missing_or_malformed_user(tmp_path):
    store = Store(tmp_path / "t.db")
    for bad in (None, "", "legacy", "not-a-uuid"):
        with pytest.raises(ValueError):
            store.create(bad, "x", [])
        with pytest.raises(ValueError):
            store.memories(bad)


def test_duplicates_cap_and_eviction_are_per_user(tmp_path):
    store = Store(tmp_path / "t.db", max_memories=2)
    assert store.add_memory(A, "Uses metric units.", "auto", None)
    assert store.add_memory(B, "uses METRIC units", "auto", None)  # B may say what A says
    assert store.add_memory(A, "uses metric units.", "user", None) is None  # A's own duplicate
    store.add_memory(A, "Prefers tables.", "user", None)
    store.add_memory(A, "Works in Oslo.", "user", None)  # A is full: A's oldest automatic memory goes
    assert [m.text for m in store.memories(A)] == ["Prefers tables.", "Works in Oslo."]
    assert [m.text for m in store.memories(B)] == ["uses METRIC units"]  # never evicted by A
    with pytest.raises(ValueError, match="memory is full"):
        store.add_memory(A, "Third typed fact.", "user", None)
    assert store.add_memory(B, "Second fact of B.", "user", None)  # B has room of their own


def test_a_memory_never_points_at_another_users_thread(tmp_path):
    store = Store(tmp_path / "t.db")
    a_thread = store.create(A, "A's chat", []).thread_id
    b_thread = store.create(B, "B's chat", []).thread_id
    assert store.add_memory(B, "Likes tea.", "auto", a_thread).thread_id is None
    assert store.add_memory(B, "Likes coffee.", "auto", b_thread).thread_id == b_thread
    assert [m.thread_id for m in store.memories(B)] == [None, b_thread]


def test_replacing_checks_duplicates_only_within_the_user(tmp_path):
    store = Store(tmp_path / "t.db")
    store.add_memory(A, "Works in legal.", "user", None)
    b_mem = store.add_memory(B, "Works in finance.", "user", None)
    item, old = store.replace_memory(B, b_mem.memory_id, "Works in legal.", "auto")  # A's text is no obstacle
    assert (item.text, old) == ("Works in legal.", "Works in finance.")


def test_purge_user_deletes_only_that_user_and_can_be_repeated(tmp_path):
    store = Store(tmp_path / "t.db")
    for user in (A, B):
        t = store.create(user, "chat", []).thread_id
        store.append(user, t, turn("q", "a"), None)
        store.add_memory(user, "A fact.", "user", t)
    assert store.purge_user(A) == {"threads": 1, "messages": 2, "memories": 1, "search_tasks": 0}
    assert store.purge_user(A) == {"threads": 0, "messages": 0, "memories": 0, "search_tasks": 0}
    assert store.recent(A, 10) == [] and store.memories(A) == []
    assert len(store.recent(B, 10)) == 1 and len(store.memories(B)) == 1
    assert store.get(B, store.recent(B, 10)[0].thread_id).thread.message_count == 2


# --- API ------------------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.events = []

    async def record(self, action, subject=None, actor=None, *, user_id=None, **detail):
        self.events.append((action, user_id))

    async def aclose(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    from threads import api

    monkeypatch.setattr(api, "store", Store(tmp_path / "api.db", max_memories=50))
    monkeypatch.setattr(api, "audit", Recorder())
    return TestClient(api.app)


def test_another_users_thread_and_memory_are_404(client):
    ha, hb = KEYS.headers(A), KEYS.headers(B)
    t = client.post("/v1/threads", json={"title": "A's chat"}, headers=ha).json()["thread_id"]
    m = client.post("/v1/memories", json={"text": "A works in finance."}, headers=ha).json()["memory_id"]

    assert client.get("/v1/threads", headers=hb).json() == []
    assert client.get(f"/v1/threads/{t}", headers=hb).status_code == 404
    assert client.patch(f"/v1/threads/{t}", json={"title": "x"}, headers=hb).status_code == 404
    assert client.delete(f"/v1/threads/{t}", headers=hb).status_code == 404
    assert client.get("/v1/memories", headers=hb).json() == []
    assert client.delete(f"/v1/memories/{m}", headers=hb).status_code == 404
    assert client.put(f"/v1/memories/{m}", json={"text": "B's text."}, headers=hb).status_code == 404
    rb = KEYS.headers(B, issuer="rag")
    assert client.get(f"/v1/threads/{t}/context", headers=rb).status_code == 404
    assert client.post(f"/v1/threads/{t}/messages", json={"messages": [{"role": "user", "content": "x"}]},
                       headers=rb).status_code == 404
    assert client.put(f"/v1/threads/{t}/summary", json={"summary": "s", "upto": 5}, headers=rb).json() == {"updated": False}
    linked = client.post("/v1/memories", json={"text": "B likes tea.", "thread_id": t}, headers=hb)
    assert linked.status_code == 201 and linked.json()["thread_id"] is None
    # "already saved" never comes from another user's memory
    assert client.post("/v1/memories", json={"text": "A works in finance."}, headers=hb).json() is not None

    assert client.get(f"/v1/threads/{t}", headers=ha).json()["thread"]["title"] == "A's chat"
    assert [x["text"] for x in client.get("/v1/memories", headers=ha).json()] == ["A works in finance."]


def test_backend_routes_need_rags_signature(client):
    t = client.post("/v1/threads", json={}, headers=KEYS.headers(A)).json()["thread_id"]
    body = {"messages": [{"role": "user", "content": "hello"}]}
    as_gateway, as_rag = KEYS.headers(A), KEYS.headers(A, issuer="rag")
    assert client.post(f"/v1/threads/{t}/messages", json=body, headers=as_gateway).status_code == 403
    assert client.get(f"/v1/threads/{t}/context", headers=as_gateway).status_code == 403
    assert client.put(f"/v1/threads/{t}/summary", json={"summary": "s", "upto": 1}, headers=as_gateway).status_code == 403
    assert client.post(f"/v1/threads/{t}/messages", json=body, headers=as_rag).status_code == 201
    assert [m["content"] for m in client.get(f"/v1/threads/{t}/context", headers=as_rag).json()["messages"]] == ["hello"]


def test_routes_need_an_identity_and_audit_names_the_user(client):
    from threads import api

    assert client.get("/v1/threads").status_code == 401
    assert client.get("/v1/memories", headers={"x-identity": KEYS.token(A) + "x"}).status_code == 401
    assert client.get("/v1/threads", headers=KEYS.headers(A, role="system")).status_code == 401
    m = client.post("/v1/memories", json={"text": "Uses EUR."}, headers=KEYS.headers(A)).json()["memory_id"]
    client.delete(f"/v1/memories/{m}", headers=KEYS.headers(A))
    assert api.audit.events == [("memory.added", A), ("memory.deleted", A)]


def test_internal_routes_are_for_auth_system_calls(client):
    system = KEYS.headers(new_user_id(), role="system")
    client.post("/v1/memories", json={"text": "Uses EUR."}, headers=KEYS.headers(A))
    assert client.get("/internal/capabilities", headers=system).json() == {"service": "threads", "multi_user": 1,
                                                                           "schema": 3}
    assert client.get("/internal/capabilities", headers=KEYS.headers(A)).status_code == 403
    assert client.delete(f"/internal/users/{A}", headers=KEYS.headers(A, role="admin")).status_code == 403
    assert client.delete(f"/internal/users/{A}", headers=system).json()["memories"] == 1
    assert client.delete(f"/internal/users/{A}", headers=system).json()["memories"] == 0
    assert client.delete("/internal/users/not-a-user", headers=system).status_code == 422


# --- deleted accounts -----------------------------------------------------------------------


def test_nothing_can_be_created_for_a_deleted_account(tmp_path):
    from threads.store import OwnerDeleted

    store = Store(tmp_path / "t.db")
    t = store.create(A, "chat", []).thread_id
    store.purge_user(A)
    for write in (lambda: store.create(A, "late chat", []),
                  lambda: store.append(A, t, turn("late q", "late a"), None),
                  lambda: store.add_memory(A, "A late fact.", "auto", None),
                  lambda: store.adopt_legacy(A)):
        with pytest.raises(OwnerDeleted):
            write()
    assert store.recent(A, 10) == [] and store.memories(A) == []
    assert store.purge_user(A) == {"threads": 0, "messages": 0, "memories": 0, "search_tasks": 0}  # repeatable
    assert store.create(B, "B is fine", []).title == "B is fine" and store.add_memory(B, "B fact.", "user", None)


def test_a_v1_database_gets_the_tombstone_table(tmp_path):
    import sqlite3

    path = tmp_path / "t.db"
    Store(path)
    with sqlite3.connect(path) as db:  # what a v1 store looked like: no deleted_users
        db.execute("DROP TABLE deleted_users")
        db.execute("PRAGMA user_version = 1")
    store = Store(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
    store.purge_user(A)
    assert Store(path) and store.recent(A, 1) == []


def test_a_chat_authorized_before_the_purge_cannot_write_after_it(client):
    """rag keeps signing fresh identities from a chat that started before the account was deleted."""
    ha, ra = KEYS.headers(A), KEYS.headers(A, issuer="rag")
    t = client.post("/v1/threads", json={"title": "chat"}, headers=ha).json()["thread_id"]
    system = KEYS.headers(new_user_id(), role="system")
    assert client.delete(f"/internal/users/{A}", headers=system).status_code == 200
    gone = {"detail": "this account has been deleted"}
    r = client.post("/v1/threads", json={"title": "new chat"}, headers=ha)
    assert r.status_code == 403 and r.json() == gone
    r = client.post(f"/v1/threads/{t}/messages", json={"messages": [{"role": "user", "content": "x"}]}, headers=ra)
    assert r.status_code == 403
    r = client.post("/v1/memories", json={"text": "A remembers this."}, headers=ra)
    assert r.status_code == 403
    assert client.get("/v1/threads", headers=ha).json() == [] and client.get("/v1/memories", headers=ha).json() == []
    owner = {"owner_id": A, "dry_run": True}
    assert client.post("/internal/migrate-legacy", json=owner, headers=system).status_code == 409



def test_purge_checks_that_the_wal_was_truncated(tmp_path, monkeypatch, caplog):
    import logging

    from threads import store as store_module

    path = tmp_path / "t.db"
    store = Store(path)
    store.add_memory(A, "A takes medication X daily.", "user", None)
    store.purge_user(A)
    data = b"".join(p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists())
    assert b"medication X" not in data
    calls = []
    monkeypatch.setattr(store_module, "truncate_wal", lambda db, log, **f: calls.append(f) or log.warning("blocked"))
    with caplog.at_level(logging.WARNING, logger="threads.store"):
        store.purge_user(B)
    assert calls and any(r.name == "threads.store" for r in caplog.records)
