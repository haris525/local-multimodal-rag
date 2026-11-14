from rag_common.identity import Identity, Verifier

from conftest import PASSWORD, cookie_value


def admin_cookie(api) -> tuple[str, object]:
    admin = api.add_user("root", role="admin")
    return api.session_token("root"), admin


def test_only_admins_with_a_browser_session(api):
    api.add_user("alice")
    cookie = api.session_token("alice")
    r = api.call("GET", "/v1/admin/users", cookie=cookie)
    assert r.status_code == 403 and r.json() == {"detail": "admin only"}
    assert api.call("GET", "/v1/admin/users").status_code == 401
    root, _ = admin_cookie(api)
    assert api.call("POST", "/v1/admin/users", cookie=root, csrf=False, json={"username": "bob"}).status_code == 403


def test_create_user_with_a_temporary_password(api, services, clock):
    root, admin = admin_cookie(api)
    r = api.call("POST", "/v1/admin/users", cookie=root, json={"username": "Bob", "role": "user"})
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["username"] == "bob" and body["user"]["must_change_password"] is True
    assert set(body["user"]) == {"user_id", "username", "role", "must_change_password", "disabled", "status",
                                 "locked_until", "created_at", "last_login_at", "active_sessions", "state",
                                 "actions", "deletion"}
    assert body["user"]["state"] == "must_change" and body["user"]["deletion"] is None
    assert body["user"]["actions"] == ["disable", "make_admin", "reset_password", "delete"]
    temporary = body["temporary_password"]
    assert len(temporary) == 16
    login = api.login("bob", temporary)
    assert login.status_code == 200 and login.json()["user"]["must_change_password"] is True
    # the capability check went to every data service, signed as system by auth
    checked = [c for c in services.calls if c[2] == "/internal/capabilities"]
    assert sorted(c[0] for c in checked) == ["audit", "ingest", "rag", "threads"]
    ident = Verifier.from_file(api.s.public_keys_file, {"auth"}, clock=lambda: clock().timestamp()).verify(checked[0][3])
    assert ident == Identity(admin.user_id, "system", "auth")
    created = [e for e in api.audit.events if e["action"] in ("auth.account_created", "admin.user_created")]
    assert created[0]["user_id"] == body["user"]["user_id"] and created[0]["actor"] == admin.user_id
    assert created[1]["user_id"] == admin.user_id and created[1]["subject"] == body["user"]["user_id"]
    assert api.call("POST", "/v1/admin/users", cookie=root, json={"username": "bob"}).status_code == 409
    assert api.call("POST", "/v1/admin/users", cookie=root, json={"username": "b!"}).status_code == 400


def test_capability_guard(api, services):
    root, _ = admin_cookie(api)
    services.multi_user["threads"] = False
    r = api.call("POST", "/v1/admin/users", cookie=root, json={"username": "bob"})
    assert r.status_code == 409 and r.json() == {"detail": "multi-user not ready: threads"}
    assert api.auth.store.get_user_by_name("bob") is None
    services.multi_user = dict.fromkeys(services.multi_user, False)
    assert api.call("POST", "/v1/admin/users", cookie=root,
                    json={"username": "eve", "role": "admin"}).json() == {"detail": "multi-user not ready: ingest"}


def test_self_and_last_admin_guards(api):
    root, admin = admin_cookie(api)
    me = f"/v1/admin/users/{admin.user_id}"
    assert api.call("PATCH", me, cookie=root, json={"disabled": True}).status_code == 409
    assert api.call("PATCH", me, cookie=root, json={"role": "user"}).status_code == 409
    assert api.call("DELETE", me, cookie=root, json={"confirm_username": "root"}).status_code == 409
    assert api.call("POST", f"{me}/reset-password", cookie=root).status_code == 409
    other = api.add_user("ops", role="admin")
    r = api.call("PATCH", f"/v1/admin/users/{other.user_id}", cookie=root, json={"role": "user"})
    assert r.status_code == 200 and r.json()["role"] == "user"
    # The caller is always an active admin, so through the API the last-admin rule only matters in races;
    # check it on the store: root is now the only active admin.
    from auth.store import Refused

    for action, values in (("disable", {}), ("delete", {}), ("set_role", {"role": "user"})):
        try:
            api.auth.store.apply(admin.user_id, action, actor_id=other.user_id, **values)
            raise AssertionError(f"{action} of the last active admin was allowed")
        except Refused as exc:
            assert exc.status == 409 and "last active admin" in exc.detail
    assert api.auth.store.get_user(admin.user_id).role == "admin" and not api.auth.store.get_user(admin.user_id).disabled


def test_last_admin_guard_counts_only_active_admins(api):
    root, admin = admin_cookie(api)
    other = api.add_user("ops", role="admin")
    api.auth.store.apply(other.user_id, "disable")
    ops_disabled = f"/v1/admin/users/{other.user_id}"
    assert api.call("PATCH", ops_disabled, cookie=root, json={"role": "user"}).status_code == 200


def test_disable_revokes_sessions_and_enable(api):
    root, _ = admin_cookie(api)
    bob = api.add_user("bob")
    bob_cookie = api.session_token("bob")
    r = api.call("PATCH", f"/v1/admin/users/{bob.user_id}", cookie=root, json={"disabled": True})
    assert r.status_code == 200 and r.json()["disabled"] is True and r.json()["active_sessions"] == 0
    assert api.call("GET", "/v1/auth/session", cookie=bob_cookie).status_code == 401
    assert api.call("PATCH", f"/v1/admin/users/{bob.user_id}", cookie=root, json={"disabled": False}).json()[
        "disabled"] is False
    assert api.login("bob").status_code == 200
    assert {"auth.disabled_by_admin", "admin.user_disabled", "auth.enabled_by_admin", "admin.user_enabled"} <= set(
        api.audit.actions())


def test_reset_password(api):
    root, _ = admin_cookie(api)
    bob = api.add_user("bob")
    bob_cookie = api.session_token("bob")
    r = api.call("POST", f"/v1/admin/users/{bob.user_id}/reset-password", cookie=root)
    assert r.status_code == 200
    temporary = r.json()["temporary_password"]
    assert api.call("GET", "/v1/auth/session", cookie=bob_cookie).status_code == 401  # sessions revoked
    user = api.auth.store.get_user(bob.user_id)
    assert user.must_change_password and user.reset_by_admin_at
    cookie = cookie_value(api.login("bob", temporary), "rag_session")
    session = api.call("GET", "/v1/auth/session", cookie=cookie).json()
    assert session["user"]["must_change_password"] and session["password_reset_by_admin_at"] == user.reset_by_admin_at
    assert api.call("GET", "/internal/verify", cookie=cookie, headers={"X-Original-Method": "GET"}).status_code == 403
    r = api.call("POST", "/v1/auth/password", cookie=cookie,
                 json={"current_password": temporary, "new_password": "a completely new passphrase"})
    assert r.status_code == 200
    assert api.auth.store.get_user(bob.user_id).reset_by_admin_at is None
    assert api.call("POST", f"/v1/admin/users/{'1' * 8}-1111-4111-8111-{'1' * 12}/reset-password",
                    cookie=root).status_code == 404
    assert api.call("POST", "/v1/admin/users/not-a-uuid/reset-password", cookie=root).status_code == 422


def test_the_reset_notice_stays_until_the_user_dismisses_it(api):
    """An admin who resets a password and signs in as the user must change the temporary password, but that
    change does not clear the notice: only an explicit dismiss does, and it is recorded in the user's trail."""
    root, _ = admin_cookie(api)
    bob = api.add_user("bob")
    temporary = api.call("POST", f"/v1/admin/users/{bob.user_id}/reset-password", cookie=root).json()[
        "temporary_password"]
    reset_at = api.auth.store.get_user(bob.user_id).reset_by_admin_at
    cookie = cookie_value(api.login("bob", temporary), "rag_session")
    r = api.call("POST", "/v1/auth/password", cookie=cookie,
                 json={"current_password": temporary, "new_password": "the admin chose this one"})
    cookie = cookie_value(r, "rag_session")
    session = api.call("GET", "/v1/auth/session", cookie=cookie).json()
    assert session["password_reset_by_admin_at"] is None  # the forced change is done
    assert session["security_notice"] == {"kind": "password_reset_by_admin", "at": reset_at}
    cookie = cookie_value(api.login("bob", "the admin chose this one"), "rag_session")  # a later sign-in
    assert api.call("GET", "/v1/auth/session", cookie=cookie).json()["security_notice"]["at"] == reset_at
    assert api.call("POST", "/v1/auth/security-notice/dismiss", cookie=cookie, csrf=False).status_code == 403
    assert api.call("POST", "/v1/auth/security-notice/dismiss", cookie=cookie).status_code == 204
    assert api.call("GET", "/v1/auth/session", cookie=cookie).json()["security_notice"] is None
    assert [e for e in api.audit.events if e["action"] == "auth.security_notice_dismissed"][0]["detail"][
        "reset_at"] == reset_at
    assert api.call("POST", "/v1/auth/security-notice/dismiss", cookie=cookie).status_code == 204  # repeatable
    assert api.audit.actions().count("auth.security_notice_dismissed") == 1
    token = api.api_token(cookie, password="the admin chose this one")
    assert api.call("POST", "/v1/auth/security-notice/dismiss", bearer=token).status_code == 403  # browser only


def test_a_v1_database_keeps_a_pending_reset_notice(tmp_path):
    import sqlite3

    from auth.store import SCHEMA_VERSION, AuthStore

    db = tmp_path / "v1.db"
    with sqlite3.connect(db) as c:  # the v1 users table, with one reset still waiting for its forced change
        c.executescript("""CREATE TABLE users (user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            role TEXT NOT NULL, password_hash TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0,
            must_change_password INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active',
            reset_by_admin_at TEXT, password_changed_at TEXT NOT NULL, created_at TEXT NOT NULL, created_by TEXT,
            last_login_at TEXT); PRAGMA user_version = 1;""")
        c.execute("INSERT INTO users (user_id, username, role, password_hash, must_change_password, reset_by_admin_at,"
                  " password_changed_at, created_at) VALUES ('u1', 'bob', 'user', 'h', 1, 't1', 't1', 't0'),"
                  " ('u2', 'amy', 'user', 'h', 0, NULL, 't1', 't0')")
    store = AuthStore(db)
    assert store.get_user("u1").password_reset_notice_at == "t1" and store.get_user("u2").password_reset_notice_at is None
    with sqlite3.connect(db) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    AuthStore(db)  # a second start changes nothing
    assert store.get_user("u1").password_reset_notice_at == "t1"


def test_unlock(api):
    root, _ = admin_cookie(api)
    bob = api.add_user("bob")
    for _ in range(5):
        api.login("bob", "wrong password!")
    listed = {u["username"]: u for u in api.call("GET", "/v1/admin/users", cookie=root).json()}
    assert listed["bob"]["locked_until"]
    assert api.call("POST", f"/v1/admin/users/{bob.user_id}/unlock", cookie=root).status_code == 204
    assert api.login("bob").status_code == 200


def test_delete_with_confirmation(api, services):
    root, admin = admin_cookie(api)
    bob = api.add_user("bob")
    url = f"/v1/admin/users/{bob.user_id}"
    r = api.call("DELETE", url, cookie=root, json={"confirm_username": "bobby"})
    assert r.status_code == 400 and api.auth.store.get_user(bob.user_id).status == "active"
    assert api.call("DELETE", url, cookie=root, json={"confirm_username": "BOB"}).status_code == 204
    assert api.auth.store.get_user(bob.user_id) is None
    purged = [c[0] for c in services.calls if c[2] == f"/internal/users/{bob.user_id}"]
    assert purged == ["ingest", "threads", "audit"]
    assert api.audit.events[-1] == {"action": "admin.user_deleted", "subject": bob.user_id, "actor": admin.user_id,
                                    "user_id": admin.user_id, "detail": {"username": "bob"}}
    assert api.call("DELETE", url, cookie=root, json={"confirm_username": "bob"}).status_code == 404
    api.add_user("bob")  # the username can be reused


def test_purge_resumes_after_a_failed_step(api, services):
    root, _ = admin_cookie(api)
    bob = api.add_user("bob")
    bob_cookie = api.session_token("bob")
    services.fail = {"threads"}
    url = f"/v1/admin/users/{bob.user_id}"
    r = api.call("DELETE", url, cookie=root, json={"confirm_username": "bob"})
    assert r.status_code == 502
    assert r.json() == {"detail": "threads: disk full", "status": "deleting", "failed_step": "threads"}
    user = api.auth.store.get_user(bob.user_id)
    assert user.status == "deleting" and user.disabled
    assert api.call("GET", "/v1/auth/session", cookie=bob_cookie).status_code == 401  # access stopped at once
    assert api.login("bob").status_code == 401
    assert api.call("PATCH", url, cookie=root, json={"disabled": False}).status_code == 409
    services.fail = set()
    services.calls.clear()
    assert api.call("DELETE", url, cookie=root, json={"confirm_username": "bob"}).status_code == 204
    assert [c[0] for c in services.calls if c[2].startswith("/internal/users/")] == ["ingest", "threads", "audit"]
    assert api.auth.store.get_user(bob.user_id) is None


def test_admin_audit_reads(api):
    root, _ = admin_cookie(api)
    assert api.call("GET", "/v1/admin/audit/verify", cookie=root).json()["ok"] is True
    events = api.call("GET", "/v1/admin/audit/security?limit=5", cookie=root).json()
    assert events == [{"seq": 2, "time": "t", "action": "auth.login_failed", "user_id": "anonymous",
                       "actor": "anonymous", "detail": {}, "username": None}]
    assert api.call("GET", "/v1/admin/audit/security?limit=501", cookie=root).status_code == 422


def test_admin_list(api):
    root, _ = admin_cookie(api)
    api.add_user("bob")
    users = api.call("GET", "/v1/admin/users", cookie=root).json()
    assert [u["username"] for u in users] == ["bob", "root"]
    assert next(u for u in users if u["username"] == "root")["active_sessions"] == 1
    assert PASSWORD not in str(users) and "password_hash" not in str(users)
