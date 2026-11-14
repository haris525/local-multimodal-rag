"""GET /internal/verify: what nginx auth_request sees."""

from rag_common.identity import Identity, Verifier

from conftest import CSRF


def verifier(settings, clock):
    return Verifier.from_file(settings.public_keys_file, {"auth"}, clock=lambda: clock().timestamp())


def test_signed_identity_for_a_session(api, settings, clock):
    alice = api.add_user("alice")
    r = api.call("GET", "/internal/verify", cookie=api.session_token("alice"), csrf=False,
                 headers={"X-Original-Method": "GET", "X-Request-ID": "req-42"})
    assert r.status_code == 204
    assert verifier(settings, clock).verify(r.headers["x-identity"]) == Identity(alice.user_id, "user", "auth")


def test_admin_role_is_carried(api, settings, clock):
    admin = api.add_user("root", role="admin")
    r = api.call("GET", "/internal/verify", cookie=api.session_token("root"), headers={"X-Original-Method": "GET"})
    assert verifier(settings, clock).verify(r.headers["x-identity"]) == Identity(admin.user_id, "admin", "auth")


def test_csrf_header_rule_uses_the_original_method(api):
    api.add_user("alice")
    cookie = api.session_token("alice")
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        r = api.call("GET", "/internal/verify", cookie=cookie, csrf=False, headers={"X-Original-Method": method})
        assert r.status_code == 403 and r.headers["x-auth-detail"] == "missing request header"
        r = api.call("GET", "/internal/verify", cookie=cookie, headers={"X-Original-Method": method, **CSRF})
        assert r.status_code == 204
    for method in ("GET", "HEAD"):
        assert api.call("GET", "/internal/verify", cookie=cookie, csrf=False,
                        headers={"X-Original-Method": method}).status_code == 204
    # no original method: treated as unsafe
    assert api.call("GET", "/internal/verify", cookie=cookie, csrf=False).status_code == 403


def test_bearer_tokens_are_exempt_from_the_header(api, settings, clock):
    alice = api.add_user("alice")
    token = api.api_token(api.session_token("alice"))
    r = api.call("GET", "/internal/verify", bearer=token, csrf=False, headers={"X-Original-Method": "POST"})
    assert r.status_code == 204
    assert verifier(settings, clock).verify(r.headers["x-identity"]).user_id == alice.user_id


def test_rejections(api):
    api.add_user("alice")
    cookie = api.session_token("alice")
    token = api.api_token(cookie)
    get = {"X-Original-Method": "GET"}
    assert api.call("GET", "/internal/verify", headers=get).status_code == 401
    assert api.call("GET", "/internal/verify", cookie="forged", headers=get).status_code == 401
    assert api.call("GET", "/internal/verify", bearer="rag_pat_forged", headers=get).status_code == 401
    # a browser token offered as a bearer token, or an API token offered as a cookie, is refused
    assert api.call("GET", "/internal/verify", bearer="rag_pat_" + cookie, headers=get).status_code == 401
    assert api.call("GET", "/internal/verify", cookie=token, headers=get).status_code == 401
    r = api.call("GET", "/internal/verify", headers=get)
    assert r.headers["x-auth-detail"] == "sign in required" and "x-identity" not in r.headers


def test_session_touch_is_throttled(api, clock):
    api.add_user("alice")
    cookie = api.session_token("alice")

    def last_seen() -> str:
        with api.auth.store._connect() as db:
            return db.execute("SELECT last_seen_at FROM sessions").fetchone()[0]

    first = last_seen()
    clock.advance(seconds=30)
    api.call("GET", "/internal/verify", cookie=cookie, headers={"X-Original-Method": "GET"})
    assert last_seen() == first  # at most one write per minute per session
    clock.advance(seconds=31)
    api.call("GET", "/internal/verify", cookie=cookie, headers={"X-Original-Method": "GET"})
    assert last_seen() != first
