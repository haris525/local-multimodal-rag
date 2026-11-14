"""The credentials version (docs/STATE_MACHINES.md, M1 and M3): a session works only while the account's
credentials are the ones it was opened with, and a sign-in that overlaps a change gets no session at all.
Proofs A2 and A3 turned around."""

import sqlite3

import httpx
from fastapi.testclient import TestClient

from auth.api import create
from auth.sessions import hash_token, new_token
from auth.store import AuthStore
from conftest import PASSWORD, Api, cookie_value

NEW_PASSWORD = "a completely different passphrase"
GET = {"x-original-method": "GET"}


def verify_status(api, cookie=None, bearer=None):
    return api.call("GET", "/internal/verify", cookie=cookie, bearer=bearer, headers=GET).status_code


def during_the_password_check(api, change):
    """Run `change` while a sign-in's password check is in progress (between reading the account and
    creating the session)."""
    real_verify = api.auth.passwords.verify

    async def verify(password_hash, password):
        ok = await real_verify(password_hash, password)
        await change()
        return ok

    api.auth.passwords.verify = verify
    return real_verify


def test_A2_a_sign_in_that_overlaps_a_password_change_gets_no_session(api):
    alice = api.add_user("alice")

    async def change():  # what POST /v1/auth/password does
        api.auth.store.apply(alice.user_id, "change_password", expected_version=alice.credentials_version,
                             password_hash=await api.auth.passwords.hash(NEW_PASSWORD))

    real = during_the_password_check(api, change)
    r = api.login("alice", PASSWORD)
    api.auth.passwords.verify = real
    assert r.status_code == 401 and "set-cookie" not in r.headers
    assert api.auth.store.count_sessions(alice.user_id) == 0
    assert api.audit.events[-1]["detail"]["reason"] == "credentials_changed"
    assert api.login("alice", PASSWORD).status_code == 401 and api.login("alice", NEW_PASSWORD).status_code == 200


def test_A3_a_sign_in_that_overlaps_a_disable_does_not_come_back_on_enable(api):
    bob = api.add_user("bob")

    async def change():  # what disabling does
        api.auth.store.apply(bob.user_id, "disable")

    real = during_the_password_check(api, change)
    r = api.login("bob", PASSWORD)
    api.auth.passwords.verify = real
    assert r.status_code == 401 and api.auth.store.count_sessions(bob.user_id) == 0
    api.auth.store.apply(bob.user_id, "enable")
    assert api.auth.store.count_sessions(bob.user_id) == 0 and api.login("bob").status_code == 200


def test_a_session_from_before_a_disable_stays_dead_after_enable(api):
    """Even a session the disable did not delete (it raced the delete) is refused, also after enable."""
    bob = api.add_user("bob")
    cookie = api.session_token("bob")
    with api.auth.store._connect() as db:  # a disable whose session delete never happened
        db.execute("UPDATE users SET disabled = 1, credentials_version = credentials_version + 1")
    assert verify_status(api, cookie) == 401
    with api.auth.store._connect() as db:
        db.execute("UPDATE users SET disabled = 0")
    api.auth.store.apply(bob.user_id, "disable")
    api.auth.store.apply(bob.user_id, "enable")
    assert verify_status(api, cookie) == 401
    assert api.auth.store.count_sessions(bob.user_id) == 0  # deleted when it was refused


def test_every_credentials_change_increases_the_version(api, services):
    root = api.add_user("root", role="admin")
    admin = api.session_token("root")
    bob = api.add_user("bob")
    version = lambda: api.auth.store.get_user(bob.user_id).credentials_version  # noqa: E731
    url = f"/v1/admin/users/{bob.user_id}"
    seen = [version()]
    cookie = api.session_token("bob")
    r = api.call("POST", "/v1/auth/password", cookie=cookie,
                 json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})
    assert r.status_code == 200
    seen.append(version())
    fresh = cookie_value(r, api.s.cookie_name)  # the session the change opens has the new version
    assert verify_status(api, fresh) == 204 and verify_status(api, cookie) == 401
    assert api.call("POST", f"{url}/reset-password", cookie=admin).status_code == 200
    seen.append(version())
    assert api.call("PATCH", url, cookie=admin, json={"disabled": True}).status_code == 200
    seen.append(version())
    assert api.call("PATCH", url, cookie=admin, json={"disabled": False}).status_code == 200
    seen.append(version())
    services.fail = {"threads"}
    assert api.call("DELETE", url, cookie=admin, json={"confirm_username": "bob"}).status_code == 502
    seen.append(version())
    assert all(b > a for a, b in zip(seen, seen[1:])) and root.credentials_version == 0
    assert verify_status(api, fresh) == 401


def test_a_token_created_while_the_password_changes_is_refused(api):
    alice = api.add_user("alice")
    cookie = api.session_token("alice")

    async def change():
        api.auth.store.apply(alice.user_id, "change_password", expected_version=alice.credentials_version,
                             password_hash=await api.auth.passwords.hash(NEW_PASSWORD))

    real = during_the_password_check(api, change)
    r = api.call("POST", "/v1/auth/tokens", cookie=cookie, json={"name": "eval", "days": 30,
                                                                  "current_password": PASSWORD})
    api.auth.passwords.verify = real
    assert r.status_code == 401 and api.auth.store.list_tokens(alice.user_id) == []


def test_create_session_needs_the_version_it_was_checked_against(tmp_path):
    store = AuthStore(tmp_path / "a.db")
    user = store.create_user("alice", "user", "hash", False, None)
    _, token_hash = new_token("browser")
    assert store.create_session(user.user_id, "browser", token_hash, None, None, "9999", None, None,
                                credentials_version=user.credentials_version + 1) is None
    made = store.create_session(user.user_id, "browser", token_hash, None, None, "9999", None, None,
                                credentials_version=user.credentials_version)
    assert made.credentials_version == user.credentials_version
    store.apply(user.user_id, "disable")
    assert store.create_session(user.user_id, "api", "x" * 64, "t", None, "9999", None, None,
                                credentials_version=store.get_user(user.user_id).credentials_version) is None


V2 = """
CREATE TABLE users (user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('admin','user')), password_hash TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','deleting')), reset_by_admin_at TEXT,
    password_changed_at TEXT NOT NULL, created_at TEXT NOT NULL, created_by TEXT, last_login_at TEXT,
    password_reset_notice_at TEXT);
CREATE TABLE sessions (session_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE, kind TEXT NOT NULL CHECK (kind IN ('browser','api')),
    name TEXT, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, idle_expires_at TEXT, absolute_expires_at TEXT NOT NULL,
    ip TEXT, user_agent TEXT);
CREATE INDEX sessions_user ON sessions (user_id, kind);
CREATE INDEX sessions_expiry ON sessions (absolute_expires_at);
CREATE TABLE login_throttle (key TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0, window_start TEXT NOT NULL,
    lock_count INTEGER NOT NULL DEFAULT 0, locked_until TEXT);
PRAGMA user_version = 2;
"""


def v2_database(path, passwords, clock):
    """An auth.db as written before the credentials version: one account with a live browser session."""
    token, token_hash = new_token("browser")
    now = clock().isoformat(timespec="seconds")
    later = clock().replace(year=clock().year + 1).isoformat(timespec="seconds")
    with sqlite3.connect(path) as db:
        db.executescript(V2)
        db.execute("INSERT INTO users (user_id, username, role, password_hash, password_changed_at, created_at)"
                   " VALUES ('11111111-1111-4111-8111-111111111111', 'alice', 'user', ?, ?, ?)",
                   (passwords.hash_sync(PASSWORD), now, now))
        db.execute("INSERT INTO sessions (session_id, token_hash, user_id, kind, created_at, last_seen_at,"
                   " idle_expires_at, absolute_expires_at) VALUES ('s1', ?, '11111111-1111-4111-8111-111111111111',"
                   " 'browser', ?, ?, ?, ?)", (token_hash, now, now, later, later))
    return token


def test_a_v2_database_keeps_its_sessions_working(settings, clock, services):
    from auth.passwords import Passwords

    token = v2_database(settings.db_path, Passwords(settings), clock)
    app = create(settings, clock=clock, transport=httpx.MockTransport(services))
    with TestClient(app, base_url="http://testserver") as client:
        api = Api(client, settings)
        assert verify_status(api, token) == 204  # signed in before the upgrade, still signed in
        with sqlite3.connect(settings.db_path) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] >= 3
            assert "credentials_version" in {r[1] for r in db.execute("PRAGMA table_info(sessions)")}
        user = api.auth.store.get_user_by_name("alice")
        api.auth.store.apply(user.user_id, "change_password", password_hash="other", expected_version=0)
        assert verify_status(api, token) == 401
    AuthStore(settings.db_path)  # a second start changes nothing
    with sqlite3.connect(settings.db_path) as db:  # the refused session was deleted
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE token_hash = ?", (hash_token(token),)).fetchone()[0] == 0


def test_a_password_change_whose_check_overlaps_a_reset_does_not_undo_it(api):
    """An admin resets bob's password while bob's own change is checking his current password: the change is
    refused (401), the reset stays, and nobody gets a session with the password bob chose."""
    api.add_user("root", role="admin")
    bob = api.add_user("bob")
    cookie = api.session_token("bob")
    temporary_hash = api.auth.passwords.hash_sync("a temporary passphrase")

    async def reset():
        api.auth.store.apply(bob.user_id, "reset", password_hash=temporary_hash)

    real = during_the_password_check(api, reset)
    r = api.call("POST", "/v1/auth/password", cookie=cookie,
                 json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})
    api.auth.passwords.verify = real
    assert r.status_code == 401 and "set-cookie" not in r.headers
    after = api.auth.store.get_user(bob.user_id)
    assert after.password_hash == temporary_hash and after.must_change_password and after.password_reset_notice_at
    assert api.auth.store.count_sessions(bob.user_id) == 0
    assert api.login("bob", NEW_PASSWORD).status_code == 401


def test_a_rehash_at_sign_in_never_overwrites_a_reset(api, monkeypatch):
    bob = api.add_user("bob")
    temporary_hash = api.auth.passwords.hash_sync("a temporary passphrase")

    async def reset():
        api.auth.store.apply(bob.user_id, "reset", password_hash=temporary_hash)

    monkeypatch.setattr(api.auth.passwords, "needs_rehash", lambda password_hash: True)
    real = during_the_password_check(api, reset)
    r = api.login("bob", PASSWORD)
    api.auth.passwords.verify = real
    assert r.status_code == 401 and api.auth.store.get_user(bob.user_id).password_hash == temporary_hash


def test_a_password_change_must_say_which_version_it_acted_on(tmp_path):
    store = AuthStore(tmp_path / "a.db")
    bob = store.create_user("bob", "user", "h", False, None)
    import pytest

    with pytest.raises(ValueError):
        store.apply(bob.user_id, "change_password", password_hash="h2")
    store.apply(bob.user_id, "reset", password_hash="h3")
    from auth.store import Refused

    with pytest.raises(Refused):
        store.apply(bob.user_id, "change_password", password_hash="h2", expected_version=bob.credentials_version)
    assert store.get_user(bob.user_id).password_hash == "h3"
