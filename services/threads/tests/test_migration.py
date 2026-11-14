"""Upgrading threads.db from before accounts (schema v0: global memories) and adopting its rows."""

import sqlite3

from fastapi.testclient import TestClient

from conftest import KEYS
from rag_common.testing import new_user_id

from threads.store import Store

V0_SCHEMA = """
CREATE TABLE threads (thread_id TEXT PRIMARY KEY, title TEXT NOT NULL, doc_ids TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '', summary_upto INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE messages (seq INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES threads (thread_id) ON DELETE CASCADE, role TEXT NOT NULL,
    content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE INDEX messages_thread ON messages (thread_id, seq);
CREATE TABLE memories (memory_id TEXT PRIMARY KEY, text TEXT NOT NULL, norm TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL, thread_id TEXT, created_at TEXT NOT NULL);
"""
THREAD = "a" * 32
MEMORIES = [("3" * 32, "Uses EUR.", "uses eur"), ("1" * 32, "Works in finance.", "works in finance"),
            ("2" * 32, "Likes tables.", "likes tables")]  # rowid order differs from id order on purpose


def legacy_db(path):
    with sqlite3.connect(path) as db:
        db.executescript(V0_SCHEMA)
        db.execute("INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, 'Old chat', 't', 't')",
                   (THREAD,))
        db.execute("INSERT INTO messages (thread_id, role, content, created_at) VALUES (?, 'user', 'old q', 't')",
                   (THREAD,))
        for memory_id, text, norm in MEMORIES:
            db.execute("INSERT INTO memories (memory_id, text, norm, source, thread_id, created_at)"
                       " VALUES (?, ?, ?, 'auto', ?, 't')", (memory_id, text, norm, THREAD))


def test_upgrade_keeps_memories_in_order_and_hides_everything_until_adopted(tmp_path):
    legacy_db(tmp_path / "t.db")
    store = Store(tmp_path / "t.db")
    owner = new_user_id()
    with sqlite3.connect(tmp_path / "t.db") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        rows = db.execute("SELECT memory_id, user_id FROM memories ORDER BY rowid").fetchall()
        assert rows == [(m[0], None) for m in MEMORIES]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert store.memories(owner) == [] and store.recent(owner, 10) == []

    assert store.adopt_legacy(owner, dry_run=True) == {"threads": 1, "memories": 3, "memories_skipped": 0,
                                                        "dry_run": True}
    assert store.memories(owner) == []
    assert store.adopt_legacy(owner) == {"threads": 1, "memories": 3, "memories_skipped": 0, "dry_run": False}
    assert store.adopt_legacy(owner) == {"threads": 0, "memories": 0, "memories_skipped": 0, "dry_run": False}
    assert [m.memory_id for m in store.memories(owner)] == [m[0] for m in MEMORIES]  # order kept
    detail = store.get(owner, THREAD)
    assert detail.thread.title == "Old chat" and [m.content for m in detail.messages] == ["old q"]
    assert store.memories(owner)[0].thread_id == THREAD
    Store(tmp_path / "t.db")  # a second start changes nothing
    assert len(store.memories(owner)) == 3


def test_a_legacy_memory_the_owner_already_has_stays_unowned(tmp_path):
    legacy_db(tmp_path / "t.db")
    store = Store(tmp_path / "t.db")
    owner = new_user_id()
    store.add_memory(owner, "uses EUR", "user", None)  # saved after the upgrade, before adoption
    assert store.adopt_legacy(owner)["memories_skipped"] == 1
    assert sorted(m.text for m in store.memories(owner)) == ["Likes tables.", "Works in finance.", "uses EUR"]
    assert store.adopt_legacy(owner)["memories"] == 0


def test_migrate_route_is_for_auth_system_calls(tmp_path, monkeypatch):
    from threads import api

    legacy_db(tmp_path / "t.db")
    monkeypatch.setattr(api, "store", Store(tmp_path / "t.db"))
    client = TestClient(api.app)
    owner = new_user_id()
    assert client.post("/internal/migrate-legacy", json={"owner_id": owner}, headers=KEYS.headers(owner)).status_code == 403
    r = client.post("/internal/migrate-legacy", json={"owner_id": owner}, headers=KEYS.headers(owner, role="system"))
    assert r.json() == {"threads": 1, "memories": 3, "memories_skipped": 0, "dry_run": False}
    assert len(client.get("/v1/memories", headers=KEYS.headers(owner)).json()) == 3
