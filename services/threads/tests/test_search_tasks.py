"""Custom search tasks per user (schema v3): CRUD with owner checks, the per-user limit, deletion with the account,
the routes and their plain refusals, and the upgrade of a v2 database."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from conftest import KEYS
from rag_common.testing import new_user_id

from threads.store import SCHEMA_VERSION, OwnerDeleted, Store

A, B = new_user_id(), new_user_id()
CONTRACTS = "Given a question about an agreement, retrieve the clauses that answer it"


def test_tasks_are_added_listed_in_order_edited_in_place_and_deleted(tmp_path):
    store = Store(tmp_path / "t.db")
    first = store.add_search_task(A, "Contracts", CONTRACTS)
    second = store.add_search_task(A, "Minutes", "Given a question about a meeting, retrieve what was decided")
    assert [t.task_id for t in store.search_tasks(A)] == [first.task_id, second.task_id]
    assert store.search_task(A, first.task_id) == first and len(first.task_id) == 32
    edited = store.update_search_task(A, first.task_id, "Agreements", "Given a question, retrieve the clauses")
    assert (edited.task_id, edited.name, edited.instruction, edited.created_at) == (
        first.task_id, "Agreements", "Given a question, retrieve the clauses", first.created_at)
    assert [t.name for t in store.search_tasks(A)] == ["Agreements", "Minutes"]  # its place is kept
    assert store.delete_search_task(A, first.task_id) is True
    assert store.delete_search_task(A, first.task_id) is False
    assert store.search_task(A, first.task_id) is None and [t.name for t in store.search_tasks(A)] == ["Minutes"]


def test_another_users_task_reads_like_a_missing_one(tmp_path):
    store = Store(tmp_path / "t.db")
    task = store.add_search_task(A, "Contracts", CONTRACTS)
    assert store.search_tasks(B) == [] and store.search_task(B, task.task_id) is None
    assert store.update_search_task(B, task.task_id, "B's", "B's instruction") is None
    assert store.delete_search_task(B, task.task_id) is False
    assert store.search_tasks(A) == [task]  # nothing of A's changed
    for bad in (None, "", "legacy", "not-a-uuid"):
        with pytest.raises(ValueError):
            store.search_tasks(bad)
        with pytest.raises(ValueError):
            store.add_search_task(bad, "n", "i")


def test_the_limit_is_per_user_and_a_deleted_task_frees_its_place(tmp_path):
    store = Store(tmp_path / "t.db", max_search_tasks=2)
    kept = [store.add_search_task(A, f"Task {i}", f"Instruction {i}") for i in range(2)]
    with pytest.raises(ValueError, match="you have 2 search tasks, the most allowed \\(2\\); delete one first"):
        store.add_search_task(A, "Third", "Instruction 3")
    assert store.add_search_task(B, "B's own", "Instruction")  # B has room of their own
    store.delete_search_task(A, kept[0].task_id)
    assert store.add_search_task(A, "Third", "Instruction 3")


def test_tasks_go_with_the_account_and_none_can_be_added_after(tmp_path):
    path = tmp_path / "t.db"
    store = Store(path)
    store.add_search_task(A, "Contracts", CONTRACTS)
    store.add_search_task(A, "Minutes", "Given a question about a meeting, retrieve what was decided")
    store.add_search_task(B, "B's", "Retrieve passages about the budget")
    assert store.purge_user(A) == {"threads": 0, "messages": 0, "memories": 0, "search_tasks": 2}
    assert store.purge_user(A)["search_tasks"] == 0  # repeatable
    assert store.search_tasks(A) == [] and len(store.search_tasks(B)) == 1
    with pytest.raises(OwnerDeleted):
        store.add_search_task(A, "Late", "A late instruction")
    data = b"".join(p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists())
    assert CONTRACTS.encode() not in data  # zeroed pages, truncated WAL


def test_a_v2_database_gets_the_search_tasks_table_and_keeps_everything(tmp_path):
    path = tmp_path / "t.db"
    old = Store(path)
    thread = old.create(A, "A's chat", []).thread_id
    old.add_memory(A, "Works in legal.", "user", None)
    with sqlite3.connect(path) as db:  # what a v2 store looked like: no search_tasks
        db.execute("DROP TABLE search_tasks")
        db.execute("PRAGMA user_version = 2")
    store = Store(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 3
        assert {r[1] for r in db.execute("PRAGMA table_info(search_tasks)")} == {
            "task_id", "user_id", "name", "instruction", "created_at", "updated_at"}
    assert store.get(A, thread).thread.title == "A's chat" and [m.text for m in store.memories(A)] == ["Works in legal."]
    assert store.add_search_task(A, "Contracts", CONTRACTS).name == "Contracts"
    Store(path)  # a second start changes nothing
    assert len(store.search_tasks(A)) == 1


# --- API ------------------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.events = []

    async def record(self, action, subject=None, actor=None, *, user_id=None, **detail):
        self.events.append((action, subject, user_id, detail))

    async def aclose(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    from threads import api

    monkeypatch.setattr(api, "store", Store(tmp_path / "api.db", max_search_tasks=2))
    monkeypatch.setattr(api, "audit", Recorder())
    return TestClient(api.app)


def test_the_routes_add_list_edit_and_delete_for_the_signed_in_user(client):
    from threads import api

    ha = KEYS.headers(A)
    r = client.post("/v1/search-tasks", json={"name": " Contracts ", "instruction": f" {CONTRACTS} "}, headers=ha)
    assert r.status_code == 201
    task = r.json()
    assert (task["name"], task["instruction"]) == ("Contracts", CONTRACTS)  # trimmed
    assert client.get("/v1/search-tasks", headers=ha).json() == [task]
    assert client.get(f"/v1/search-tasks/{task['task_id']}", headers=KEYS.headers(A, issuer="rag")).json() == task
    r = client.put(f"/v1/search-tasks/{task['task_id']}", json={"name": "Agreements", "instruction": "Find clauses"},
                   headers=ha)
    assert r.status_code == 200 and r.json()["name"] == "Agreements" and r.json()["task_id"] == task["task_id"]
    assert client.delete(f"/v1/search-tasks/{task['task_id']}", headers=ha).status_code == 204
    assert client.get("/v1/search-tasks", headers=ha).json() == []
    events = api.audit.events
    assert [(e[0], e[1], e[2]) for e in events] == [("search_task.added", task["task_id"], A),
                                                    ("search_task.updated", task["task_id"], A),
                                                    ("search_task.deleted", task["task_id"], A)]
    assert all(e[3] == {} for e in events)  # the audit names the task, never its name or instruction


def test_another_users_task_is_a_404_on_every_route(client):
    task = client.post("/v1/search-tasks", json={"name": "Contracts", "instruction": CONTRACTS},
                       headers=KEYS.headers(A)).json()["task_id"]
    for hb in (KEYS.headers(B), KEYS.headers(B, issuer="rag")):
        assert client.get("/v1/search-tasks", headers=hb).json() == []
        assert client.get(f"/v1/search-tasks/{task}", headers=hb).status_code == 404
        assert client.put(f"/v1/search-tasks/{task}", json={"name": "x", "instruction": "y"}, headers=hb).status_code == 404
        assert client.delete(f"/v1/search-tasks/{task}", headers=hb).status_code == 404
    assert client.get(f"/v1/search-tasks/{task}", headers=KEYS.headers(A)).json()["instruction"] == CONTRACTS
    assert client.get("/v1/search-tasks").status_code == 401
    assert client.get("/v1/search-tasks/not-an-id", headers=KEYS.headers(A)).status_code == 422


def test_refusals_say_what_is_wrong_in_plain_words(client):
    ha = KEYS.headers(A)
    for body, words in [({"name": "", "instruction": "x"}, "the name must not be empty"),
                        ({"name": "n" * 41, "instruction": "x"}, "the name must be at most 40 characters"),
                        ({"name": "n", "instruction": "two\nlines"}, "the instruction must be a single line"),
                        ({"name": "n", "instruction": "x" * 301}, "the instruction must be at most 300 characters"),
                        ({"name": "n", "instruction": " "}, "the instruction must not be empty")]:
        r = client.post("/v1/search-tasks", json=body, headers=ha)
        assert r.status_code == 422 and words in r.json()["detail"][0]["msg"], body
    for i in range(2):
        client.post("/v1/search-tasks", json={"name": f"T{i}", "instruction": "x"}, headers=ha)
    r = client.post("/v1/search-tasks", json={"name": "T3", "instruction": "x"}, headers=ha)
    assert r.status_code == 409 and r.json()["detail"] == ("you have 2 search tasks, the most allowed (2); "
                                                            "delete one first")


def test_a_deleted_account_cannot_add_a_task(client):
    system = KEYS.headers(new_user_id(), role="system")
    client.post("/v1/search-tasks", json={"name": "Contracts", "instruction": CONTRACTS}, headers=KEYS.headers(A))
    assert client.delete(f"/internal/users/{A}", headers=system).json()["search_tasks"] == 1
    r = client.post("/v1/search-tasks", json={"name": "Late", "instruction": "x"}, headers=KEYS.headers(A))
    assert r.status_code == 403 and r.json() == {"detail": "this account has been deleted"}


def test_an_update_racing_a_delete_returns_the_row_it_wrote_or_nothing_never_a_crash(tmp_path):
    """A delete can land right after the update statement; the update must not read the row again afterwards."""
    from contextlib import contextmanager

    class Cursor:
        def __init__(self, rows, rowcount):
            self.rows, self.rowcount = rows, rowcount

        def fetchall(self):
            return self.rows

        def fetchone(self):
            return self.rows[0] if self.rows else None

    class DeletesAfterUpdate:
        """A connection on which another request deletes the task the moment an update of it finishes."""

        def __init__(self, db):
            self.db = db

        def __getattr__(self, name):
            return getattr(self.db, name)

        def execute(self, sql, params=()):
            cursor = self.db.execute(sql, params)
            if sql.startswith("UPDATE search_tasks"):
                rows, count = cursor.fetchall(), cursor.rowcount
                self.db.execute("DELETE FROM search_tasks")
                return Cursor(rows, count)
            return cursor

    class Racing(Store):
        @contextmanager
        def _connect(self):
            with super()._connect() as db:
                yield DeletesAfterUpdate(db)

    store = Racing(tmp_path / "t.db")
    task = store.add_search_task(A, "Contracts", CONTRACTS)
    edited = store.update_search_task(A, task.task_id, "Agreements", "Find clauses")
    assert edited is not None and (edited.task_id, edited.name) == (task.task_id, "Agreements")
    assert store.search_tasks(A) == []  # deleted meanwhile
    assert store.update_search_task(A, task.task_id, "Again", "Find clauses") is None  # gone: a 404


def test_tasks_of_accounts_an_earlier_threads_purged_are_deleted_at_start(tmp_path):
    """After a rollback, the earlier threads purges an account but not its search tasks; the next start does."""
    path = tmp_path / "t.db"
    store = Store(path)
    store.add_search_task(A, "Contracts", CONTRACTS)
    store.add_search_task(B, "B's", "Retrieve passages about the budget")
    with sqlite3.connect(path) as db:  # what the earlier threads' purge left: the tombstone, the tasks still there
        db.execute("INSERT INTO deleted_users (user_id, deleted_at) VALUES (?, 't')", (A,))
    Store(path)
    assert store.search_tasks(A) == [] and len(store.search_tasks(B)) == 1
    data = b"".join(p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists())
    assert CONTRACTS.encode() not in data  # zeroed pages, truncated WAL
    Store(path)  # a second start finds nothing to do
    assert len(store.search_tasks(B)) == 1


def test_a_task_never_holds_a_credential_but_may_talk_about_one(client):
    ha = KEYS.headers(A)
    ok = client.post("/v1/search-tasks", json={"name": "Security",
                                               "instruction": "Retrieve passages about password reset policies"},
                     headers=ha)
    assert ok.status_code == 201
    refusal = {"detail": "search tasks never hold passwords, keys, tokens or other credentials"}
    for body in ({"name": "Keys", "instruction": "Use api key: sk-abcdefghijklmnopqrstuvwx to find invoices"},
                 {"name": "password is hunter2", "instruction": "Find invoices"},
                 {"name": "Db", "instruction": "Find rows in postgres://admin:s3cret@db.local/sales"}):
        r = client.post("/v1/search-tasks", json=body, headers=ha)
        assert (r.status_code, r.json()) == (422, refusal), body
        r = client.put(f"/v1/search-tasks/{ok.json()['task_id']}", json=body, headers=ha)
        assert (r.status_code, r.json()) == (422, refusal), body
    assert [t["name"] for t in client.get("/v1/search-tasks", headers=ha).json()] == ["Security"]


def test_a_name_needs_a_letter_or_digit_and_no_invisible_characters(client):
    ha = KEYS.headers(A)
    for name, words in [("---", "the name must hold at least one letter or digit"),
                        ("Con" + chr(0x200B) + "tracts", "the name must not hold invisible formatting characters"),
                        ("Contracts" + chr(0x202E), "the name must not hold invisible formatting characters"),
                        (chr(0xFEFF) + "Contracts", "the name must not hold invisible formatting characters")]:
        r = client.post("/v1/search-tasks", json={"name": name, "instruction": "Find clauses"}, headers=ha)
        assert r.status_code == 422 and words in r.json()["detail"][0]["msg"], name
    for name in ("Q3", "Vertr" + chr(0xE4) + "ge", chr(0x5951) + chr(0x7D04)):  # any letter or digit counts
        assert client.post("/v1/search-tasks", json={"name": name, "instruction": "Find clauses"},
                           headers=KEYS.headers(new_user_id())).status_code == 201, name  # the limit here is 2


def test_an_admin_sees_no_other_users_task_and_a_system_identity_none(client):
    task = client.post("/v1/search-tasks", json={"name": "Contracts", "instruction": CONTRACTS},
                       headers=KEYS.headers(A)).json()["task_id"]
    admin = KEYS.headers(new_user_id(), role="admin")
    assert client.get("/v1/search-tasks", headers=admin).json() == []
    assert client.get(f"/v1/search-tasks/{task}", headers=admin).status_code == 404
    assert client.put(f"/v1/search-tasks/{task}", json={"name": "x", "instruction": "y"}, headers=admin).status_code == 404
    assert client.delete(f"/v1/search-tasks/{task}", headers=admin).status_code == 404
    system = KEYS.headers(new_user_id(), role="system")
    assert client.get("/v1/search-tasks", headers=system).status_code == 401
    assert client.get(f"/v1/search-tasks/{task}", headers=system).status_code == 401
    assert client.get(f"/v1/search-tasks/{task}", headers=KEYS.headers(A)).json()["name"] == "Contracts"
