"""Audit per user: v1/v2 hash chain, user_seq, legacy adoption, crypto-shredding, what each caller may read."""

import hashlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from conftest import KEYS
from rag_common.testing import new_user_id

from audit.store import GENESIS, AuditStore, Conflict
from rag_common.identity import AUDIT_WRITER_ID

A, B = new_user_id(), new_user_id()
V0_SCHEMA = """
CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, service TEXT NOT NULL,
    action TEXT NOT NULL, actor TEXT NOT NULL, request_id TEXT, subject TEXT, detail TEXT NOT NULL,
    prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE INDEX events_action ON events (action, seq);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


def legacy_log(path, n: int = 3) -> None:
    """An audit.db written before accounts, hashed exactly as the old service did."""
    fields = ("time", "service", "action", "actor", "request_id", "subject", "detail")
    with sqlite3.connect(path) as db:
        db.executescript(V0_SCHEMA)
        prev = GENESIS
        for i in range(n):
            row = {"time": f"2026-01-0{i + 1}T00:00:00.000+00:00", "service": "rag", "action": "chat.answered",
                   "actor": "local", "request_id": f"r{i}", "subject": "d" * 32,
                   "detail": json.dumps({"question": f"old question {i}"}, sort_keys=True)}
            canonical = json.dumps({k: row[k] for k in fields}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            row["prev_hash"], row["hash"] = prev, hashlib.sha256((prev + canonical).encode()).hexdigest()
            db.execute("INSERT INTO events (time, service, action, actor, request_id, subject, detail, prev_hash, hash)"
                       " VALUES (:time, :service, :action, :actor, :request_id, :subject, :detail, :prev_hash, :hash)",
                       row)
            prev = row["hash"]


def event(store: AuditStore, user: str, action: str = "chat.answered", service: str = "rag", **detail) -> dict:
    return store.append(service, action, user, "rid", "d" * 32, detail or {"question": "q"}, user)


def tamper(path, sql: str) -> None:
    with sqlite3.connect(path) as db:  # someone with file access
        db.execute("DROP TRIGGER IF EXISTS events_no_update")
        db.execute(sql)


# --- chain ----------------------------------------------------------------------------------


def test_a_mixed_v1_and_v2_chain_verifies_before_and_after_the_upgrade(tmp_path):
    legacy_log(tmp_path / "a.db")
    store = AuditStore(tmp_path / "a.db")
    assert store.verify()["ok"] and store.verify()["events"] == 3
    event(store, A)
    event(store, B)
    result = store.verify()
    assert result["ok"] and result["events"] == 5 and result["first_bad_seq"] is None
    with sqlite3.connect(tmp_path / "a.db") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert db.execute("SELECT user_id FROM events ORDER BY seq").fetchall() == [
            ("legacy",), ("legacy",), ("legacy",), (A,), (B,)]
    AuditStore(tmp_path / "a.db")  # a second start changes nothing
    assert store.verify()["ok"]


@pytest.mark.parametrize("sql", [
    "UPDATE events SET user_id = 'legacy' WHERE seq = 2",  # hide a user's event among the legacy rows
    "UPDATE events SET user_id = '{B}' WHERE seq = 1",  # move an event to another user
    "UPDATE events SET user_seq = 7 WHERE seq = 2",
])
def test_relabelled_or_renumbered_rows_are_detected(tmp_path, sql):
    store = AuditStore(tmp_path / "a.db")
    event(store, A)
    event(store, A)
    tamper(tmp_path / "a.db", sql.format(B=B))
    assert not store.verify()["ok"]


def test_a_user_learns_where_the_log_was_altered_only_if_it_is_their_event(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    event(store, A)
    event(store, B)
    event(store, A)
    tamper(tmp_path / "a.db", "UPDATE events SET actor = 'x' WHERE seq = 3")  # A's second event
    assert store.verify_for(A) == {"ok": False, "events": 2, "first_bad": 2}
    assert store.verify_for(B) == {"ok": False, "events": 1, "first_bad": None}


# --- per-user reads -------------------------------------------------------------------------


def test_users_read_only_their_own_events_numbered_in_their_own_sequence(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    assert [event(store, u)["user_seq"] for u in (A, B, A, A, B)] == [1, 1, 2, 3, 2]
    mine = store.list_for(A, 10)
    assert [e["user_seq"] for e in mine] == [3, 2, 1]
    assert all(set(e) == {"user_seq", "time", "service", "action", "subject", "detail", "hash"} for e in mine)
    assert [e["user_seq"] for e in store.list_for(A, 10, before=3)] == [2, 1]
    assert store.verify_for(A) == {"ok": True, "events": 3, "first_bad": None}
    assert store.list_for(new_user_id(), 10) == []


def test_details_are_encrypted_at_rest_and_readable_by_their_user(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    event(store, A, question="What is our Q3 revenue?")
    with sqlite3.connect(tmp_path / "a.db") as db:
        stored = db.execute("SELECT detail FROM events").fetchone()[0]
    assert stored.startswith("enc1:") and "revenue" not in stored
    assert store.list_for(A, 1)[0]["detail"] == {"question": "What is our Q3 revenue?"}


def test_after_a_purge_the_chain_verifies_and_details_are_shredded(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    event(store, A, question="secret plans")
    event(store, B, question="B's question")
    assert store.purge_user(A) == {"events": 1, "key_shredded": True, "legacy_closed": False}
    assert store.purge_user(A) == {"events": 1, "key_shredded": False, "legacy_closed": False}  # repeatable
    assert store.verify()["ok"]
    assert store.list_for(A, 10)[0]["detail"] == {"_shredded": True}
    assert store.list_for(B, 10)[0]["detail"] == {"question": "B's question"}
    event(store, A, question="late event after the purge")  # e.g. background work that finished late
    assert store.list_for(A, 1)[0]["detail"] == {"_shredded": True} and store.verify()["ok"]
    with sqlite3.connect(tmp_path / "a.db") as db:
        assert "late event" not in " ".join(r[0] for r in db.execute("SELECT detail FROM events"))


def test_security_events_are_the_auth_service_rows_only(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    event(store, A, "chat.answered", question="private")
    event(store, A, "auth.login_succeeded", "auth", ip="10.0.0.1")
    store.append("auth", "auth.login_failed", "anonymous", None, None, {"username": "nosuch"}, "anonymous")
    rows = store.security_events(10)
    assert [(r["action"], r["user_id"]) for r in rows] == [("auth.login_failed", "anonymous"),
                                                         ("auth.login_succeeded", A)]
    assert rows[1]["detail"] == {"ip": "10.0.0.1"} and [r["action"] for r in store.security_events(10, rows[1]["seq"])] == []


def test_new_events_can_never_be_legacy(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    for bad in ("legacy", "", None):
        with pytest.raises(ValueError):
            store.append("rag", "chat.answered", "x", None, None, {}, bad)


# --- legacy adoption ------------------------------------------------------------------------


def test_legacy_rows_become_the_owners_first_events_once(tmp_path):
    legacy_log(tmp_path / "a.db", n=3)
    store = AuditStore(tmp_path / "a.db")
    assert event(store, A)["user_seq"] == 1  # e.g. auth.account_created, written before the adoption
    assert store.adopt_legacy(A, dry_run=True) == {"legacy_rows": 3, "owner_set": False, "dry_run": True}
    assert store.list_for(A, 10)[0]["user_seq"] == 1 and len(store.list_for(A, 10)) == 1
    assert store.adopt_legacy(A) == {"legacy_rows": 3, "owner_set": True, "dry_run": False}
    assert store.adopt_legacy(A) == {"legacy_rows": 3, "owner_set": False, "dry_run": False}
    with pytest.raises(Conflict):
        store.adopt_legacy(B)

    assert event(store, A)["user_seq"] == 5  # continues after the legacy rows and the earlier event
    mine = store.list_for(A, 10)
    assert [e["user_seq"] for e in mine] == [5, 4, 3, 2, 1]  # one gap-free sequence
    assert mine[-1]["detail"] == {"question": "old question 0"}  # legacy details stay readable
    assert [e["user_seq"] for e in store.list_for(A, 2, before=4)] == [3, 2]
    assert store.verify_for(A) == {"ok": True, "events": 5, "first_bad": None}
    assert store.list_for(B, 10) == [] and store.verify_for(B)["events"] == 0
    assert store.verify()["ok"]


def test_deleting_the_legacy_owner_closes_the_legacy_rows_for_good(tmp_path):
    legacy_log(tmp_path / "a.db", n=2)
    store = AuditStore(tmp_path / "a.db")
    store.adopt_legacy(A)
    assert len(store.list_for(A, 10)) == 2
    assert store.purge_user(A) == {"events": 0, "key_shredded": False, "legacy_closed": True}
    with sqlite3.connect(tmp_path / "a.db") as db:
        assert db.execute("SELECT value FROM meta WHERE key = 'legacy_owner'").fetchone()[0] != A
    for user in (A, B):  # neither the deleted id nor anyone else reads them
        assert store.list_for(user, 10) == [] and store.verify_for(user)["events"] == 0
        with pytest.raises(Conflict, match="belonged to an account that was deleted"):
            store.adopt_legacy(user)
    assert store.purge_user(A)["legacy_closed"] is False  # repeatable
    assert store.purge_user(B)["legacy_closed"] is False and store.verify()["ok"]


# --- API ------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    from audit import api

    monkeypatch.setattr(api, "store", AuditStore(tmp_path / "api.db"))
    return TestClient(api.app)


def post(client, body: dict, headers: dict | None = None):
    return client.post("/v1/events", json={"service": "rag", "action": "chat.answered", "detail": {"q": 1}} | body,
                       headers=headers or {})


RAG_A = KEYS.headers(A, issuer="rag")
AUTH_WRITER = KEYS.headers(AUDIT_WRITER_ID, role="system")  # how the auth service signs its event writes


def test_writes_name_their_user_and_a_signed_write_cannot_name_another(client):
    assert post(client, {"user_id": A, "service": "threads"}).json() == {"user_seq": 1}  # no key: unsigned
    assert post(client, {}, RAG_A).json() == {"user_seq": 2}  # the user from the signature
    assert post(client, {"user_id": B}, RAG_A).status_code == 403
    assert post(client, {"user_id": "legacy", "service": "threads"}).status_code == 422
    assert post(client, {"service": "threads"}).status_code == 201  # unsigned, no user: unattributed, read by nobody
    assert [e["user_seq"] for e in client.get("/v1/events", headers=KEYS.headers(A)).json()] == [2, 1]


def test_only_auth_writes_auth_events_and_only_rag_writes_rag_events(client):
    login = {"service": "auth", "action": "auth.login_succeeded", "user_id": A}
    assert post(client, login).status_code == 403  # unsigned: any container on the backend network
    assert post(client, login, KEYS.headers(A)).status_code == 403  # a user assertion from auth, replayed
    assert post(client, login, KEYS.headers(new_user_id(), role="system")).status_code == 403  # an /internal one
    assert post(client, login, RAG_A).status_code == 403  # signed, but by rag
    assert post(client, login, AUTH_WRITER).status_code == 201
    assert post(client, login | {"action": "auth.login_failed", "user_id": "anonymous"}, AUTH_WRITER).status_code == 201
    assert post(client, {"user_id": A}).status_code == 403  # an unsigned rag event
    assert post(client, {"user_id": A}, AUTH_WRITER).status_code == 403  # auth cannot write rag events
    assert post(client, {"user_id": A, "service": "ingest-worker", "action": "document.ready"}).status_code == 201
    assert post(client, {"service": "ingest-worker", "action": "document.ready"}, RAG_A).status_code == 403
    assert post(client, {"user_id": A, "service": "anything-else"}).status_code == 403
    system = KEYS.headers(new_user_id(), role="system")
    assert [(e["action"], e["user_id"]) for e in client.get("/internal/security-events", headers=system).json()] == [
        ("auth.login_failed", "anonymous"), ("auth.login_succeeded", A)]


def test_reads_are_the_callers_own_and_need_an_identity_from_auth(client):
    post(client, {"user_id": A, "subject": "a" * 32}, RAG_A)
    post(client, {"user_id": B, "subject": "b" * 32}, KEYS.headers(B, issuer="rag"))
    events = client.get("/v1/events", headers=KEYS.headers(B)).json()
    assert [e["subject"] for e in events] == ["b" * 32]
    assert all("seq" not in e and "prev_hash" not in e and "request_id" not in e for e in events)
    assert client.get("/v1/events", params={"subject": "a" * 32, "action": "*"}, headers=KEYS.headers(B)).json() == []
    assert client.get("/v1/verify", headers=KEYS.headers(B)).json() == {"ok": True, "events": 1, "first_bad": None}
    assert client.get("/v1/events").status_code == 401
    assert client.get("/v1/events", headers=KEYS.headers(B, issuer="rag")).status_code == 403
    assert client.get("/v1/verify", headers=KEYS.headers(B, role="system")).status_code == 401


def test_internal_routes_are_for_auth_system_calls(client):
    post(client, {"user_id": A}, RAG_A)
    post(client, {"user_id": A, "service": "auth", "action": "auth.login_succeeded"}, AUTH_WRITER)
    system, user = KEYS.headers(new_user_id(), role="system"), KEYS.headers(A)
    for method, path in (("GET", "/internal/capabilities"), ("GET", "/internal/verify"),
                         ("GET", "/internal/security-events"), ("DELETE", f"/internal/users/{A}")):
        assert client.request(method, path, headers=user).status_code == 403
    assert client.get("/internal/capabilities", headers=system).json() == {"service": "audit", "multi_user": 1,
                                                                           "schema": 1}
    assert client.get("/internal/verify", headers=system).json()["events"] == 2
    assert [e["action"] for e in client.get("/internal/security-events", headers=system).json()] == [
        "auth.login_succeeded"]
    assert client.delete(f"/internal/users/{A}", headers=system).json()["key_shredded"] is True
    assert client.get("/v1/events", headers=user).json()[0]["detail"] == {"_shredded": True}
    owner = new_user_id()
    assert client.post("/internal/migrate-legacy", json={"owner_id": owner}, headers=system).status_code == 200
    r = client.post("/internal/migrate-legacy", json={"owner_id": B}, headers=system)
    assert r.status_code == 409 and r.json() == {"detail": "the legacy audit rows already belong to another user"}
    client.delete(f"/internal/users/{owner}", headers=system)
    r = client.post("/internal/migrate-legacy", json={"owner_id": B}, headers=system)
    assert r.status_code == 409 and "belonged to an account that was deleted" in r.json()["detail"]



def test_purge_checks_that_the_wal_was_truncated(tmp_path, monkeypatch, caplog):
    import logging

    from audit import store as store_module

    store = AuditStore(tmp_path / "a.db")
    event(store, A)
    calls = []
    monkeypatch.setattr(store_module, "truncate_wal", lambda db, log, **f: calls.append(f) or log.warning("blocked"))
    with caplog.at_level(logging.WARNING, logger="audit.store"):
        store.purge_user(A)
    assert calls and any(r.name == "audit.store" for r in caplog.records)
