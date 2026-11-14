import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auth.api import create
from auth.config import Settings

from conftest import PASSWORD, cookie_value


def test_login_errors_are_generic(api):
    api.add_user("alice")
    wrong = api.login("alice", "wrong password!")
    unknown = api.login("bob", PASSWORD)
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "invalid username or password"}
    assert "set-cookie" not in wrong.headers


def test_login_requires_the_header(api):
    api.add_user("alice")
    r = api.login("alice", csrf=False)
    assert r.status_code == 403 and r.json() == {"detail": "missing request header"}
    assert api.call("POST", "/v1/auth/login", content=b"{not json").status_code == 422
    assert api.call("POST", "/v1/auth/login", json={"username": "", "password": "x"}).status_code == 422


def test_login_success(api):
    api.add_user("alice")
    r = api.login("ALICE")  # usernames are case-insensitive
    assert r.status_code == 200 and r.json()["user"]["username"] == "alice"
    assert r.json()["user"]["must_change_password"] is False and r.headers["cache-control"] == "no-store"
    assert api.auth.store.get_user_by_name("alice").last_login_at
    assert api.audit.actions() == ["auth.login_succeeded"]


def test_login_replaces_a_previous_session(api):
    api.add_user("alice")
    first = api.session_token("alice")
    second = cookie_value(api.login("alice", cookie=first), "rag_session")
    assert api.call("GET", "/v1/auth/session", cookie=first).status_code == 401
    assert api.call("GET", "/v1/auth/session", cookie=second).status_code == 200


def test_idle_and_absolute_expiry(api, clock):
    api.add_user("alice")
    cookie = api.session_token("alice")
    clock.advance(hours=11)
    assert api.call("GET", "/v1/auth/session", cookie=cookie).status_code == 200  # activity extends idle expiry
    clock.advance(hours=11, minutes=59)
    assert api.call("GET", "/v1/auth/session", cookie=cookie).status_code == 200
    clock.advance(hours=12, seconds=1)  # 12 h without activity
    assert api.call("GET", "/v1/auth/session", cookie=cookie).status_code == 401
    assert api.auth.store.count_sessions(api.auth.store.get_user_by_name("alice").user_id) == 0

    cookie = api.session_token("alice")
    for _ in range(14):  # active every 12 hours, still ends after 7 days
        clock.advance(hours=12)
        r = api.call("GET", "/v1/auth/session", cookie=cookie)
    assert r.status_code == 401


def test_api_tokens_have_no_idle_expiry(api, clock):
    api.add_user("alice")
    token = api.api_token(api.session_token("alice"))
    clock.advance(days=29)
    assert api.call("GET", "/v1/auth/session", bearer=token, csrf=False).status_code == 200
    clock.advance(days=1, seconds=1)
    assert api.call("GET", "/v1/auth/session", bearer=token, csrf=False).status_code == 401


def test_disabled_user_loses_sessions(api):
    alice = api.add_user("alice")
    cookie = api.session_token("alice")
    token = api.api_token(cookie)
    api.auth.store.apply(alice.user_id, "disable")
    assert api.call("GET", "/v1/auth/session", cookie=cookie).status_code == 401
    assert api.call("GET", "/internal/verify", bearer=token).status_code == 401
    assert api.login("alice").status_code == 401
    assert api.audit.events[-1]["detail"]["reason"] == "disabled"


def test_must_change_blocks_data_but_allows_password_change(api):
    api.add_user("alice", must_change=True)
    cookie = api.session_token("alice")
    assert api.call("GET", "/internal/verify", cookie=cookie).status_code == 403
    assert api.call("GET", "/internal/verify", cookie=cookie).headers["x-auth-detail"] == "password change required"
    session = api.call("GET", "/v1/auth/session", cookie=cookie)
    assert session.status_code == 200 and session.json()["user"]["must_change_password"] is True
    assert api.call("POST", "/v1/auth/tokens", cookie=cookie,
                    json={"name": "x", "current_password": PASSWORD}).status_code == 403
    r = api.call("POST", "/v1/auth/password", cookie=cookie,
                 json={"current_password": PASSWORD, "new_password": "a brand new passphrase"})
    assert r.status_code == 200 and r.json()["user"]["must_change_password"] is False
    assert api.call("GET", "/internal/verify", cookie=cookie_value(r, "rag_session")).status_code == 204


def test_password_change_revokes_everything_and_rotates_the_cookie(api):
    api.add_user("alice")
    cookie, other = api.session_token("alice"), api.session_token("alice")
    token = api.api_token(cookie)
    r = api.call("POST", "/v1/auth/password", cookie=cookie,
                 json={"current_password": PASSWORD, "new_password": "a brand new passphrase"})
    assert r.status_code == 200
    new = cookie_value(r, "rag_session")
    assert new not in (cookie, other)
    for old in (cookie, other):
        assert api.call("GET", "/v1/auth/session", cookie=old).status_code == 401
    assert api.call("GET", "/v1/auth/session", bearer=token).status_code == 401
    assert api.call("GET", "/v1/auth/session", cookie=new).status_code == 200
    assert api.login("alice", "a brand new passphrase").status_code == 200
    assert "auth.password_changed" in api.audit.actions()


def test_password_change_errors(api):
    api.add_user("alice")
    cookie = api.session_token("alice")
    change = lambda current, new, **kw: api.call("POST", "/v1/auth/password", cookie=cookie,  # noqa: E731
                                                 json={"current_password": current, "new_password": new}, **kw)
    assert change(PASSWORD, "too short").json() == {"detail": "Use at least 12 characters."}
    assert change(PASSWORD, PASSWORD).json() == {"detail": "Choose a different password."}
    assert change(PASSWORD, "alice in wonderland").json() == {"detail": "Don't include your username."}
    assert change("wrong password!", "a brand new passphrase").status_code == 401
    assert change(PASSWORD, "a brand new passphrase", csrf=False).status_code == 403
    for _ in range(4):  # wrong current passwords count toward the lockout
        change("wrong password!", "a brand new passphrase")
    assert change(PASSWORD, "a brand new passphrase").status_code == 429


def test_tokens_are_shown_once_and_cannot_manage_accounts(api, settings):
    api.add_user("admin", role="admin")
    cookie = api.session_token("admin")
    r = api.call("POST", "/v1/auth/tokens", cookie=cookie,
                 json={"name": "eval", "days": 10, "current_password": PASSWORD})
    assert r.status_code == 201 and r.json()["token"].startswith("rag_pat_")
    token, token_id = r.json()["token"], r.json()["token_id"]
    listed = api.call("GET", "/v1/auth/tokens", cookie=cookie).json()
    assert [t["token_id"] for t in listed] == [token_id] and "token" not in listed[0]
    assert set(listed[0]) == {"token_id", "name", "created_at", "last_used_at", "expires_at"}
    # a token is not a browser session: no account management, no token management
    assert api.call("GET", "/v1/admin/users", bearer=token).status_code == 403
    assert api.call("POST", "/v1/admin/users", bearer=token, json={"username": "bob"}).status_code == 403
    assert api.call("GET", "/v1/auth/tokens", bearer=token).status_code == 403
    assert api.call("POST", "/v1/auth/tokens", cookie=cookie,
                    json={"name": "x", "current_password": "wrong password!"}).status_code == 401
    assert api.call("POST", "/v1/auth/tokens", cookie=cookie,
                    json={"name": "x", "days": 366, "current_password": PASSWORD}).status_code == 422
    # revoke
    assert api.call("DELETE", f"/v1/auth/tokens/{'0' * 32}", cookie=cookie).status_code == 404
    assert api.call("DELETE", f"/v1/auth/tokens/{token_id}", cookie=cookie).status_code == 204
    assert api.call("GET", "/v1/auth/session", bearer=token).status_code == 401
    assert {"auth.token_created", "auth.token_revoked"} <= set(api.audit.actions())


def test_cannot_revoke_someone_elses_token(api):
    api.add_user("alice")
    api.add_user("bob")
    alice_token_id = api.call("GET", "/v1/auth/tokens", cookie=(c := api.session_token("alice"))).json()
    api.api_token(c)
    token_id = api.call("GET", "/v1/auth/tokens", cookie=c).json()[0]["token_id"]
    assert alice_token_id == []
    assert api.call("DELETE", f"/v1/auth/tokens/{token_id}", cookie=api.session_token("bob")).status_code == 404


def test_session_and_setup_required(api):
    r = api.call("GET", "/v1/auth/session")
    assert r.status_code == 401 and r.json() == {"detail": "sign in required", "setup_required": True}
    api.add_user("alice")
    assert api.call("GET", "/v1/auth/session").json()["setup_required"] is False
    body = api.call("GET", "/v1/auth/session", cookie=api.session_token("alice")).json()
    assert body["user"]["username"] == "alice" and body["password_reset_by_admin_at"] is None
    assert body["security_notice"] is None


def test_logout(api):
    api.add_user("alice")
    cookie = api.session_token("alice")
    r = api.call("POST", "/v1/auth/logout", cookie=cookie)
    assert r.status_code == 204 and r.headers["clear-site-data"] == '"cache"'
    assert r.headers["set-cookie"].startswith("rag_session=;") and "Max-Age=0" in r.headers["set-cookie"]
    assert api.call("GET", "/v1/auth/session", cookie=cookie).status_code == 401
    assert api.call("POST", "/v1/auth/logout").status_code == 204  # always
    assert "auth.logout" in api.audit.actions()


def test_purge_removes_expired_sessions(api, clock):
    api.add_user("alice")
    api.session_token("alice")
    clock.advance(days=8)
    api.auth.purge()
    assert api.auth.store.count_sessions(api.auth.store.get_user_by_name("alice").user_id) == 0


# --- startup ----------------------------------------------------------------------------------


def test_startup_refuses_a_mismatched_public_key(settings, secrets_dir, tmp_path):
    doc = json.loads((secrets_dir / "identity_public.json").read_text())
    doc["keys"]["auth"] = doc["keys"]["rag"]
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="does not match"):
        with TestClient(create(settings.model_copy(update={"public_keys_file": stale}))):
            pass


@pytest.mark.parametrize("origin", ["http://localhost:8000/", "http://host/path", "ftp://x", "http://rag.lan"])
def test_unsafe_origins_are_refused(origin):
    with pytest.raises(ValueError):
        Settings(public_origin=origin)
    if origin == "http://rag.lan":
        assert Settings(public_origin=origin, allow_insecure_http=True).cookie_name == "rag_session"


def test_bootstrap_admin(settings, tmp_path, clock):
    password_file = Path(tmp_path / "bootstrap")
    password_file.write_text("a bootstrap passphrase\n")
    s = settings.model_copy(update={"bootstrap_admin": "Root", "bootstrap_password_file": password_file})
    with TestClient(create(s, clock=clock)) as client:
        user = client.app.state.auth.store.get_user_by_name("root")
        assert user.role == "admin" and user.must_change_password
    with TestClient(create(s, clock=clock)) as client:  # only while no account exists
        assert client.app.state.auth.store.count_users() == 1


def test_audit_writes_are_signed_by_auth(settings, clock):
    """The audit service accepts auth events only with auth's signature (as the audit writer)."""
    app = create(settings.model_copy(update={"audit_url": "http://audit"}), clock=clock)
    with TestClient(app):
        auth = app.state.auth
        assert auth.audit.signer is auth.signer and auth.audit.service == "auth"
