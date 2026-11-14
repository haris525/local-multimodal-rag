import sqlite3

from auth.config import Settings
from auth.sessions import API_TOKEN_PREFIX, clear_cookie_header, cookie_header, hash_token, new_token

from conftest import cookie_value


def test_tokens():
    browser, browser_hash = new_token("browser")
    api, api_hash = new_token("api")
    assert api.startswith(API_TOKEN_PREFIX) and not browser.startswith(API_TOKEN_PREFIX)
    assert browser_hash == hash_token(browser) and len(api_hash) == 64 and len(browser) >= 40


def test_database_holds_no_plaintext_token(api, settings):
    api.add_user("alice")
    cookie = api.session_token("alice")
    token = api.api_token(cookie)
    db = sqlite3.connect(settings.db_path)
    dump = "\n".join(db.iterdump())
    db.close()
    assert cookie not in dump and token not in dump
    assert hash_token(cookie) in dump and hash_token(token) in dump


def test_cookie_flags_for_http_origin():
    s = Settings(public_origin="http://localhost:8000")
    header = cookie_header(s, "tok", 3600)
    assert header == "rag_session=tok; Path=/; HttpOnly; SameSite=Strict; Max-Age=3600"
    assert "Domain" not in header and "Secure" not in header
    assert clear_cookie_header(s) == "rag_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"


def test_cookie_flags_for_https_origin():
    s = Settings(public_origin="https://rag.example.lan")
    assert cookie_header(s, "tok", 60) == ("__Host-rag_session=tok; Path=/; HttpOnly; SameSite=Strict; Max-Age=60;"
                                           " Secure")
    assert clear_cookie_header(s).endswith("Max-Age=0; Secure")


def test_login_cookie_lasts_until_absolute_expiry(api, settings):
    api.add_user("alice")
    r = api.login("alice")
    header = r.headers["set-cookie"]
    assert f"Max-Age={settings.absolute_timeout_days * 86400}" in header and "HttpOnly" in header
    assert cookie_value(r, "rag_session")


def test_a_malformed_bearer_token_is_a_401_not_a_fall_back_to_the_cookie(api):
    api.add_user("alice")
    cookie = api.session_token("alice")
    good = api.api_token(cookie)
    for bad in ("rag_pat_" + "x" * 300, "rag_pat_short", good + "!", "rag_pat_" + "a b" * 20, "rag_pat_" + "x" * 129,
                "not-a-token-at-all-but-long-enough-to-look-real-xxxxxxxx", ""):
        headers = {"Authorization": f"Bearer {bad}"}
        assert api.call("GET", "/v1/auth/session", cookie=cookie, headers=headers).status_code == 401, bad
        assert api.call("GET", "/internal/verify", cookie=cookie, headers=headers).status_code == 401, bad
    assert api.call("GET", "/v1/auth/session", cookie=cookie).status_code == 200  # the cookie alone still works
    assert api.call("GET", "/v1/auth/session", bearer=good).status_code == 200
    assert api.call("GET", "/v1/auth/session", cookie=cookie, headers={"Authorization": "Basic eDp5"}).status_code == 200
