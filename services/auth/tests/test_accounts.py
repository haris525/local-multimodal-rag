"""The account machine (docs/STATE_MACHINES.md, M1, M2 and M21): one state derivation, one transactional
apply() with the allowed-from table, the CHECK constraints and their upgrade, the deletion that answers in
time and resumes, and the state and actions the Users panel renders. Proof A4 turned around."""

import asyncio
import sqlite3
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from auth.api import create
from auth.store import ALLOWED, SCHEMA_VERSION, AccountState, AuthStore, Refused, account_state
from conftest import PASSWORD, Api

STATES = list(AccountState)
COLUMNS = {  # (status, disabled, must_change_password) of each state
    AccountState.ACTIVE: ("active", 0, 0),
    AccountState.MUST_CHANGE: ("active", 0, 1),
    AccountState.DISABLED: ("active", 1, 0),
    AccountState.DELETING: ("deleting", 1, 0),
}
VALUES = {"change_password": {"password_hash": "h2", "expected_version": 0}, "reset": {"password_hash": "h2"},
          "set_role": {"role": "admin"}}


def in_state(store: AuthStore, name: str, state: AccountState):
    user = store.create_user(name, "user", "h", False, None)
    with store._connect() as db:
        db.execute("UPDATE users SET status = ?, disabled = ?, must_change_password = ? WHERE user_id = ?",
                   (*COLUMNS[state], user.user_id))
    return store.get_user(user.user_id)


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("action", sorted(ALLOWED))
def test_every_action_follows_the_allowed_table(tmp_path, state, action):
    store = AuthStore(tmp_path / "a.db")
    user = in_state(store, "bob", state)
    assert account_state(user) == state
    if state in ALLOWED[action]:
        after = store.apply(user.user_id, action, **VALUES.get(action, {}))
        assert after.credentials_version == user.credentials_version + (
            action in ("change_password", "reset", "disable", "enable", "delete"))
    else:
        with pytest.raises(Refused) as exc:
            store.apply(user.user_id, action, **VALUES.get(action, {}))
        assert exc.value.status == 409
        assert store.get_user(user.user_id) == user  # nothing changed


def test_the_state_follows_each_action(tmp_path):
    store = AuthStore(tmp_path / "a.db")
    bob = store.create_user("bob", "user", "h", True, None)
    state = lambda: account_state(store.get_user(bob.user_id))  # noqa: E731
    assert state() == AccountState.MUST_CHANGE
    store.apply(bob.user_id, "change_password", password_hash="h2", expected_version=0)
    assert state() == AccountState.ACTIVE
    store.apply(bob.user_id, "reset", password_hash="h3")
    assert state() == AccountState.MUST_CHANGE
    store.apply(bob.user_id, "disable")
    assert state() == AccountState.DISABLED
    store.apply(bob.user_id, "reset", password_hash="h4")
    assert state() == AccountState.DISABLED  # a reset keeps it disabled
    store.apply(bob.user_id, "enable")
    assert state() == AccountState.MUST_CHANGE
    store.apply(bob.user_id, "delete", actor_id="11111111-1111-4111-8111-111111111111")
    assert state() == AccountState.DELETING
    assert store.get_user(bob.user_id).deletion_requested_by == "11111111-1111-4111-8111-111111111111"
    with pytest.raises(Refused):
        store.apply("22222222-2222-4222-8222-222222222222", "disable")


def test_A4_reset_and_unlock_are_refused_while_deleting(api, services):
    api.add_user("root", role="admin")
    root = api.session_token("root")
    bob = api.add_user("bob")
    services.fail = {"threads"}
    url = f"/v1/admin/users/{bob.user_id}"
    assert api.call("DELETE", url, cookie=root, json={"confirm_username": "bob"}).status_code == 502
    for path in (f"{url}/reset-password", f"{url}/unlock"):
        r = api.call("POST", path, cookie=root)
        assert r.status_code == 409 and r.json()["detail"] == "the account is being deleted; delete it again to finish"
    user = api.auth.store.get_user(bob.user_id)
    assert user.status == "deleting" and user.password_hash == bob.password_hash
    assert not [e for e in api.audit.events if e["action"] in ("admin.password_reset", "admin.user_unlocked")]


def test_two_admins_cannot_disable_each_other_at_the_same_moment(tmp_path):
    """The last-admin rule is checked in the transaction that writes, so one of two crossing actions loses."""
    store = AuthStore(tmp_path / "a.db")
    a = store.create_user("ann", "admin", "h", False, None)
    b = store.create_user("ben", "admin", "h", False, None)
    start, results = threading.Barrier(2), []

    def disable(target, actor):
        start.wait()
        try:
            store.apply(target.user_id, "disable", actor_id=actor.user_id)
            results.append("ok")
        except Refused as exc:
            results.append(exc.detail)

    threads = [threading.Thread(target=disable, args=(b, a)), threading.Thread(target=disable, args=(a, b))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["ok", "you cannot disable the last active admin"]
    assert len(store.active_admin_ids()) == 1


def test_the_check_constraints(tmp_path):
    store = AuthStore(tmp_path / "a.db")
    user = store.create_user("bob", "user", "h", False, None)
    with store._connect() as db:
        for bad in ("disabled = 2", "must_change_password = 5", "status = 'deleting', disabled = 0",
                    "credentials_version = -1", "status = 'gone'"):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(f"UPDATE users SET {bad} WHERE user_id = ?", (user.user_id,))


V3 = """
CREATE TABLE users (user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('admin','user')), password_hash TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','deleting')), reset_by_admin_at TEXT,
    password_changed_at TEXT NOT NULL, created_at TEXT NOT NULL, created_by TEXT, last_login_at TEXT,
    password_reset_notice_at TEXT, credentials_version INTEGER NOT NULL DEFAULT 0);
CREATE TABLE sessions (session_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE, kind TEXT NOT NULL CHECK (kind IN ('browser','api')),
    name TEXT, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, idle_expires_at TEXT, absolute_expires_at TEXT NOT NULL,
    ip TEXT, user_agent TEXT, credentials_version INTEGER NOT NULL DEFAULT 0);
CREATE INDEX sessions_user ON sessions (user_id, kind);
CREATE INDEX sessions_expiry ON sessions (absolute_expires_at);
CREATE TABLE login_throttle (key TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0, window_start TEXT NOT NULL,
    lock_count INTEGER NOT NULL DEFAULT 0, locked_until TEXT);
INSERT INTO users (user_id, username, role, password_hash, disabled, must_change_password, status,
    password_changed_at, created_at, credentials_version) VALUES
    ('u1', 'ann', 'admin', 'h', 0, 0, 'active', 't', 't', 3),
    ('u2', 'bob', 'user', 'h', 0, 7, 'deleting', 't', 't', 1),
    ('u3', 'cid', 'user', 'h', 2, 0, 'active', 't', 't', 0);
INSERT INTO sessions (session_id, token_hash, user_id, kind, created_at, last_seen_at, absolute_expires_at,
    credentials_version) VALUES ('s1', 'x', 'u1', 'browser', 't', 't', '9999', 3);
PRAGMA user_version = 3;
"""


def test_a_v3_database_is_rebuilt_with_the_checks_and_keeps_every_row(tmp_path):
    path = tmp_path / "auth.db"
    with sqlite3.connect(path) as db:
        db.executescript(V3)
    store = AuthStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    ann, bob, cid = (store.get_user(u) for u in ("u1", "u2", "u3"))
    assert ann.credentials_version == 3 and store.session_by_hash("x")[1].user_id == "u1"  # sessions still joined
    assert bob.disabled and bob.must_change_password is True and account_state(bob) == AccountState.DELETING
    assert cid.disabled and account_state(cid) == AccountState.DISABLED
    assert [u.user_id for u in store.deleting_users()] == ["u2"]
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("DELETE FROM users WHERE user_id = 'u1'")  # the foreign key still cascades
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    AuthStore(path)  # a second start changes nothing


# --- the Users panel's state and actions ----------------------------------------------------


def listed(api, cookie):
    return {u["username"]: u for u in api.call("GET", "/v1/admin/users", cookie=cookie).json()}


def test_the_list_carries_each_accounts_state_and_actions(api, services):
    root = api.add_user("root", role="admin")
    admin = api.session_token("root")
    api.add_user("amy")
    carl = api.add_user("carl")
    api.add_user("dora", must_change=True)
    eve = api.add_user("eve")
    api.call("PATCH", f"/v1/admin/users/{carl.user_id}", cookie=admin, json={"disabled": True})
    for _ in range(5):
        api.login("amy", "wrong password!")
    services.fail = {"audit"}
    api.call("DELETE", f"/v1/admin/users/{eve.user_id}", cookie=admin, json={"confirm_username": "eve"})
    users = listed(api, admin)
    assert {n: (u["state"], u["actions"]) for n, u in users.items()} == {
        "root": ("active", []),  # your own account: change your password from the user menu
        "amy": ("active", ["disable", "make_admin", "reset_password", "unlock", "delete"]),
        "carl": ("disabled", ["enable", "make_admin", "reset_password", "delete"]),
        "dora": ("must_change", ["disable", "make_admin", "reset_password", "delete"]),
        "eve": ("deleting", ["delete"]),
    }
    assert users["eve"]["deletion"] == {"running": False, "error": "audit: disk full",
                                        "attempted_at": users["eve"]["deletion"]["attempted_at"]}
    assert users["root"]["deletion"] is None and root.user_id == users["root"]["user_id"]


# --- deletion: answered in time, resumed --------------------------------------------------------


class SlowPurge:
    """DataServices.purge that waits until the test lets it finish."""

    def __init__(self, real):
        self.real, self.go = real, None

    async def __call__(self, target_id, actor_id):
        self.go = self.go or asyncio.Event()
        await self.go.wait()
        return await self.real(target_id, actor_id)


def test_a_long_deletion_answers_202_and_finishes_in_the_background(api, settings):
    api.add_user("root", role="admin")
    admin = api.session_token("root")
    bob = api.add_user("bob")
    api.auth.s = settings.model_copy(update={"admin_delete_wait_s": 0.05})
    slow = SlowPurge(api.auth.services.purge)
    api.auth.services.purge = slow
    r = api.call("DELETE", f"/v1/admin/users/{bob.user_id}", cookie=admin, json={"confirm_username": "bob"})
    assert r.status_code == 202 and r.json()["status"] == "deleting"
    row = listed(api, admin)["bob"]
    assert row["state"] == "deleting" and row["deletion"]["running"] is True and row["active_sessions"] == 0
    again = api.call("DELETE", f"/v1/admin/users/{bob.user_id}", cookie=admin, json={"confirm_username": "bob"})
    assert again.status_code == 202  # a retry joins the running purge instead of starting a second one
    api.client.portal.call(slow.go.set)
    api.client.portal.call(asyncio.wait, set(api.auth.deletions.values()))
    assert api.auth.store.get_user(bob.user_id) is None and "bob" not in listed(api, admin)
    assert api.audit.actions().count("admin.user_deleted") == 1


def test_a_stopped_deletion_is_resumed(api, services):
    root = api.add_user("root", role="admin")
    admin = api.session_token("root")
    bob = api.add_user("bob")
    services.fail = {"ingest"}
    assert api.call("DELETE", f"/v1/admin/users/{bob.user_id}", cookie=admin,
                    json={"confirm_username": "bob"}).status_code == 502
    assert api.client.portal.call(api.auth.resume_deletions) == 1  # still failing: tried, kept
    assert api.auth.store.get_user(bob.user_id).deletion_error == "ingest: disk full"
    services.fail = set()
    assert api.client.portal.call(api.auth.resume_deletions) == 1
    assert api.auth.store.get_user(bob.user_id) is None
    deleted = [e for e in api.audit.events if e["action"] == "admin.user_deleted"]
    assert deleted == [{"action": "admin.user_deleted", "subject": bob.user_id, "actor": root.user_id,
                        "user_id": root.user_id, "detail": {"username": "bob", "resumed": True}}]
    assert api.client.portal.call(api.auth.resume_deletions) == 0


def test_auth_resumes_deletions_at_start_up(settings, clock, services):
    store = AuthStore(settings.db_path, clock)
    bob = store.create_user("bob", "user", "h", False, None)
    store.apply(bob.user_id, "delete")  # auth stopped in the middle of this deletion
    app = create(settings.model_copy(update={"deletion_resume_delay_s": 0}), clock=clock,
                 transport=httpx.MockTransport(services))
    with TestClient(app, base_url="http://testserver") as client:
        api = Api(client, settings)
        for _ in range(100):
            if api.auth.store.get_user(bob.user_id) is None:
                break
            client.portal.call(asyncio.sleep, 0.02)
        assert api.auth.store.get_user(bob.user_id) is None
        assert [c[0] for c in services.calls if c[2] == f"/internal/users/{bob.user_id}"] == ["ingest", "threads", "audit"]


def test_the_cli_lists_the_same_states_and_refuses_what_the_panel_refuses(api, monkeypatch, capsys, settings):
    from test_cli import run

    for key, value in {"AUTH_DB_PATH": settings.db_path, "AUTH_SIGNING_KEY_FILE": settings.signing_key_file,
                       "AUTH_PUBLIC_KEYS_FILE": settings.public_keys_file, "AUTH_AUDIT_URL": "",
                       "AUTH_ARGON2_TIME_COST": 1, "AUTH_ARGON2_MEMORY_KIB": 1024, "AUTH_ARGON2_PARALLELISM": 1}.items():
        monkeypatch.setenv(key, str(value))
    api.add_user("amy", must_change=True)
    bob = api.add_user("bob")
    api.auth.store.apply(bob.user_id, "delete")
    api.auth.store.deletion_failed(bob.user_id, "threads: disk full")
    assert run(monkeypatch, "list-users") == 0
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}
    assert "must change password" in lines["amy"] and "deleting (threads: disk full)" in lines["bob"]
    assert run(monkeypatch, "reset-password", "--username", "bob", "--password-stdin", stdin=PASSWORD + " 2026\n") == 1
    assert "being deleted" in capsys.readouterr().err
    assert run(monkeypatch, "unlock", "--username", "bob") == 1


def test_a_purge_that_raises_is_recorded_and_the_resumer_goes_on(api, services):
    """Any failure inside a purge (not only a data service's answer) is kept on the account; the DELETE answers
    502, and the resumer still tries every other account."""
    api.add_user("root", role="admin")
    admin = api.session_token("root")
    amy, bob = api.add_user("amy"), api.add_user("bob")
    real = api.auth.services.purge

    async def purge(target_id, actor_id):
        if target_id == amy.user_id:
            raise RuntimeError("the database is locked")
        return await real(target_id, actor_id)

    api.auth.services.purge = purge
    r = api.call("DELETE", f"/v1/admin/users/{amy.user_id}", cookie=admin, json={"confirm_username": "amy"})
    assert r.status_code == 502 and r.json()["failed_step"] == "auth"
    api.auth.store.apply(bob.user_id, "delete")  # a second stopped deletion, after amy's in name order
    assert api.client.portal.call(api.auth.resume_deletions) == 2
    assert api.auth.store.get_user(bob.user_id) is None
    assert api.auth.store.get_user(amy.user_id).deletion_error == "auth: RuntimeError: the database is locked"
    deleted = [e for e in api.audit.events if e["action"] == "admin.user_deleted"]
    assert deleted[-1]["detail"] == {"username": "bob", "resumed": True, "requested_by": "unknown"}


def test_an_upgrade_that_fails_halfway_leaves_the_database_as_it_was(tmp_path, monkeypatch):
    """A crash after the old users table was dropped rolls the whole upgrade back; the next start upgrades."""
    from auth import store as store_module

    path = tmp_path / "auth.db"
    with sqlite3.connect(path) as db:
        db.executescript(V3)

    def snapshot():
        with sqlite3.connect(path) as db:
            return (db.execute("PRAGMA user_version").fetchone()[0],
                    db.execute("SELECT * FROM users ORDER BY rowid").fetchall(),
                    db.execute("SELECT * FROM sessions").fetchall(),
                    sorted(r[0] for r in db.execute("SELECT name FROM sqlite_master")))

    before = snapshot()
    broken = [*store_module._V3_TO_V4]
    broken.insert(broken.index("DROP TABLE users") + 1, "SELECT no_such_column FROM users_v4")
    monkeypatch.setattr(store_module, "_V3_TO_V4", broken)
    with pytest.raises(sqlite3.OperationalError):
        AuthStore(path)
    assert snapshot() == before  # version 3, every row, no users_v4 left behind
    monkeypatch.undo()
    AuthStore(path)
    assert snapshot()[0] == 4 and len(snapshot()[1]) == 3 and len(snapshot()[2]) == 1
