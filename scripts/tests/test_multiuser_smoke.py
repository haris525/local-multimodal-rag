"""The pure parts of scripts/multiuser_smoke.py; the live run needs a deployed stack.

    uv run --with httpx --with pytest pytest -q scripts/tests
"""

import csv
import io
import json
import random
import struct
import sys
import zlib
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import multiuser_smoke as smoke  # noqa: E402

ADMIN_ID = "3f0c9a52-8d0e-4c1b-9a57-2b4e0f6d9c11"
HTTP_COOKIE = "rag_session=tok123; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800"
HTTPS_COOKIE = "__Host-rag_session=tok123; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800; Secure"


# --- URLs and cookies -----------------------------------------------------------------------


@pytest.mark.parametrize("url, origin", [
    ("http://localhost:8000", "http://localhost:8000"),
    ("http://localhost:8000/api/", "http://localhost:8000"),
    ("https://RAG.example.lan/", "https://rag.example.lan"),
    ("https://rag.example.lan:443", "https://rag.example.lan"),
    ("http://127.0.0.1:80/x", "http://127.0.0.1"),
    ("http://[::1]:8000", "http://[::1]:8000"),
])
def test_origin_is_what_a_browser_sends(url, origin):
    assert smoke.origin_of(url) == origin


@pytest.mark.parametrize("bad", ["ftp://host", "localhost:8000", "http://", ""])
def test_origin_refuses_other_urls(bad):
    with pytest.raises(ValueError):
        smoke.origin_of(bad)


def test_set_cookie_is_parsed_into_name_value_flags_and_params():
    c = smoke.parse_set_cookie(HTTPS_COOKIE)
    assert (c["name"], c["value"]) == ("__Host-rag_session", "tok123")
    assert {"path", "httponly", "samesite", "max-age", "secure"} == c["flags"]
    assert c["params"] == {"path": "/", "samesite": "Strict", "max-age": "604800"}


def test_the_auth_cookie_format_passes_k1_for_its_scheme():
    assert smoke.cookie_problems(HTTP_COOKIE, https=False) == []
    assert smoke.cookie_problems(HTTPS_COOKIE, https=True) == []


@pytest.mark.parametrize("header, https, problem", [
    ("rag_session=t; Path=/; SameSite=Strict", False, "missing httponly"),
    ("rag_session=t; Path=/; HttpOnly; SameSite=Lax", False, "SameSite is not Strict"),
    ("rag_session=t; Path=/; HttpOnly; SameSite=Strict; Domain=example.lan", False, "Domain is set"),
    ("rag_session=t; Path=/api; HttpOnly; SameSite=Strict", False, "Path is not /"),
    ("rag_session=t; Path=/; HttpOnly; SameSite=Strict; Secure", False, "plain http needs the rag_session name without Secure"),
    ("rag_session=t; Path=/; HttpOnly; SameSite=Strict; Secure", True, "https needs the __Host-rag_session name and Secure"),
    ("__Host-rag_session=t; Path=/; HttpOnly; SameSite=Strict", True, "https needs the __Host-rag_session name and Secure"),
])
def test_k1_reports_each_wrong_attribute(header, https, problem):
    assert problem in smoke.cookie_problems(header, https)


# --- server-sent events, generated files, facts -----------------------------------------------


def test_sse_events_are_split_on_blank_lines_and_joined_across_data_lines():
    lines = ["event: sources", 'data: {"route": "search",', 'data: "sources": []}', "", ": comment", "event: token",
             'data:{"text": "Hi"}', "", "event: done", 'data: {"citations": [1]}']  # no blank line at the end
    assert list(smoke.sse_events(lines)) == [("sources", {"route": "search", "sources": []}), ("token", {"text": "Hi"}),
                                             ("done", {"citations": [1]})]


def _png_chunks(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind, payload = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(kind + payload) & 0xFFFFFFFF
        chunks.append((kind, payload))
        pos += 12 + length
    return chunks


def test_png_is_a_valid_rgb_image_with_the_stripes():
    data = smoke.png(10, 6, [(255, 0, 0), (0, 0, 255)])
    chunks = _png_chunks(data)
    assert [k for k, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[0][1]) == (10, 6, 8, 2, 0, 0, 0)
    raw = zlib.decompress(chunks[1][1])
    assert len(raw) == 6 * (1 + 10 * 3)
    assert raw[1:4] == b"\xff\x00\x00" and raw[-3:] == b"\x00\x00\xff"


def test_facts_are_reproducible_and_the_second_account_gets_other_words():
    fa = smoke.Facts.make(random.Random("x"))
    assert fa == smoke.Facts.make(random.Random("x"))
    fb = smoke.Facts.make(random.Random("y"), taken={fa.project, fa.beacon, fa.fruit})
    assert {fb.project, fb.beacon}.isdisjoint({fa.project, fa.beacon}) and fb.fruit != fa.fruit
    assert fa.code in fa.needles() and f"project {fa.project}" in fa.needles()


def test_sample_files_hold_the_facts_and_are_valid():
    facts = smoke.Facts.make(random.Random(1))
    files = smoke.sample_files("abc123", "A", facts)
    assert set(files) == {f"{smoke.PREFIX}notes-abc123.txt", f"{smoke.PREFIX}table-abc123.csv", f"{smoke.PREFIX}figure-abc123.png"}
    note = files[f"{smoke.PREFIX}notes-abc123.txt"].decode()
    assert f"The vault code for project {facts.project} is {facts.code}." in note
    rows = list(csv.reader(io.StringIO(files[f"{smoke.PREFIX}table-abc123.csv"].decode())))
    assert rows[0] == ["item", "quantity"] and len(rows) == 3
    _png_chunks(files[f"{smoke.PREFIX}figure-abc123.png"])
    assert files != smoke.sample_files("abc123", "B", facts)  # B's image differs


def test_answers_match_facts_whatever_the_case_or_dash():
    assert smoke.says("The code is kestrel\u20114721.", "KESTREL-4721")
    assert smoke.says("KESTREL\u20144721", "kestrel-4721")
    assert not smoke.says("KESTREL 4721", "KESTREL-4721")
    assert smoke.leaks("my fruit: Feijoa, room B123", ["feijoa", "C999", "b123", ""]) == ["feijoa", "b123"]


# --- refusal, report -------------------------------------------------------------------------


def user(name, uid):
    return {"username": name, "user_id": uid}


def test_a_fresh_install_with_only_the_admin_is_accepted():
    assert smoke.refusal_reasons([user("admin", ADMIN_ID)], ADMIN_ID, {"documents": [], "chats": [], "memories": []}) == []


def test_other_accounts_leftovers_and_admin_data_are_refused():
    users = [user("admin", ADMIN_ID), user("alice", "u1"), user(f"{smoke.PREFIX}a-123456", "u2")]
    reasons = smoke.refusal_reasons(users, ADMIN_ID, {"documents": [{"doc_id": "d"}], "chats": [], "memories": [{}, {}]})
    assert any("1 other account(s) exist (alice)" in r for r in reasons)
    assert any("--cleanup" in r and f"{smoke.PREFIX}a-123456" in r for r in reasons)
    assert any("the admin has 1 documents" in r for r in reasons) and any("the admin has 2 memories" in r for r in reasons)
    assert not any("chats" in r for r in reasons)


def test_other_accounts_can_be_allowed_explicitly_but_leftovers_and_admin_data_still_refuse():
    users = [user("admin", ADMIN_ID), user("alice", "u1")]
    empty = {"documents": [], "chats": [], "memories": []}
    assert smoke.refusal_reasons(users, ADMIN_ID, empty, allow_other_accounts=True) == []
    users.append(user(f"{smoke.PREFIX}a-123456", "u2"))
    reasons = smoke.refusal_reasons(users, ADMIN_ID, {**empty, "memories": [{}]}, allow_other_accounts=True)
    assert any("--cleanup" in r for r in reasons) and any("the admin has 1 memories" in r for r in reasons)
    assert not any("other account(s) exist" in r for r in reasons)
    assert smoke.parse_args(["--admin-user", "a", "--allow-other-accounts"]).allow_other_accounts


def test_report_counts_failures_and_prints_a_table():
    lines = []
    r = smoke.Report(out=lines.append)
    assert r.add("D1", "first", True, "ignored when passing") is True
    assert r.add("C1", "second", False, "404 expected, got 200") is False
    assert [c.id for c in r.failed] == ["C1"] and r.checks[0].detail == ""
    table = r.table()
    assert "D1  pass    first" in table and "C1  FAIL    second" in table and "404 expected, got 200" in table
    assert table.endswith("1 of 2 checks passed")
    assert lines[1].startswith("[FAIL] C1")


# --- the gateway client ----------------------------------------------------------------------


def gateway(handler, url="http://localhost:8000"):
    return smoke.Gateway(url, smoke.origin_of(url), transport=httpx.MockTransport(handler))


def test_browser_style_headers_for_sessions_and_script_style_for_tokens():
    gw = gateway(lambda r: httpx.Response(200))
    acc = smoke.Account("b", cookie="rag_session=c", token="rag_pat_t")
    post = gw.headers("POST", acc, smoke._DEFAULT, True, False, None)
    assert post == {"Cookie": "rag_session=c", "Origin": "http://localhost:8000", "X-Requested-With": "local-rag"}
    assert "Origin" not in gw.headers("GET", acc, smoke._DEFAULT, True, False, None)
    assert gw.headers("POST", acc, smoke._DEFAULT, True, True, None) == {"Authorization": "Bearer rag_pat_t"}
    assert gw.headers("POST", acc, None, False, False, {"X-User-Id": "x"}) == {"Cookie": "rag_session=c", "X-User-Id": "x"}


def test_the_gateway_client_never_keeps_a_cookie_between_requests():
    seen = []

    def handler(request):
        seen.append(request.headers.get("cookie"))
        return httpx.Response(200, headers={"set-cookie": HTTP_COOKIE}, json={"user": {"user_id": "u"}})

    gw = gateway(handler)
    gw.login("a", "pw", paced=False)
    gw.call("GET", "/api/documents")
    assert seen == [None, None]


def test_login_waits_out_the_gateway_rate_limit_but_returns_a_lockout(monkeypatch):
    monkeypatch.setattr(smoke, "LOGIN_GAP_S", 0)
    monkeypatch.setattr(smoke.time, "sleep", lambda s: None)
    answers = [httpx.Response(429, text="<html>429</html>"), httpx.Response(200, json={"user": {}})]
    assert gateway(lambda r: answers.pop(0)).login("a", "pw").status_code == 200 and not answers
    locked = gateway(lambda r: httpx.Response(429, headers={"retry-after": "900"}, json={"detail": "too many"}))
    r = locked.login("a", "pw")
    assert r.status_code == 429 and r.headers["retry-after"] == "900"


def test_login_sends_the_csrf_header_and_origin():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(401)

    gateway(handler).login("Alice", "pw", paced=False)
    assert seen["x-requested-with"] == "local-rag" and seen["origin"] == "http://localhost:8000"
    assert seen["body"] == {"username": "Alice", "password": "pw"}


def test_chat_collects_the_stream_and_reports_a_refusal():
    stream = ('event: memory\ndata: {"action": "saved", "items": [{"memory_id": "m", "text": "Desk in B123"}]}\n\n'
              'event: sources\ndata: {"route": "search", "sources": [{"n": 1, "doc_id": "d1"}]}\n\n'
              'event: token\ndata: {"text": "It is "}\n\nevent: token\ndata: {"text": "X-1."}\n\n'
              'event: done\ndata: {"citations": [1]}\n\n')

    def handler(request):
        body = json.loads(request.content)
        if body["doc_ids"] == ["other"]:
            return httpx.Response(404, json={"detail": "unknown documents: other"})
        assert body["thread_id"] == "t" and body["think"] is False
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    gw = gateway(handler)
    acc = smoke.Account("a", cookie="rag_session=c")
    out = gw.chat(acc, "q", ["d1"], "t")
    assert (out["status"], out["route"], out["answer"], out["citations"]) == (200, "search", "It is X-1.", [1])
    assert smoke.saved_texts(out) == ["Desk in B123"]
    refused = gw.chat(acc, "q", ["other"])
    assert refused["status"] == 404 and "unknown documents" in refused["body"]


def test_chat_sends_a_search_task_only_when_given():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(422, json={"detail": "unknown search task: x"})

    gw = gateway(handler)
    acc = smoke.Account("b", cookie="rag_session=c")
    gw.chat(acc, "q", ["d1"])
    out = gw.chat(acc, "q", ["d1"], search_task="f" * 32)
    assert "search_task" not in sent[0] and sent[1]["search_task"] == "f" * 32
    assert out["status"] == 422 and "unknown search task" in out["body"]


def test_cookie_pair_and_detail():
    r = httpx.Response(200, headers={"set-cookie": HTTPS_COOKIE})
    assert smoke.cookie_pair(r) == "__Host-rag_session=tok123"
    with pytest.raises(RuntimeError):
        smoke.cookie_pair(httpx.Response(200))
    assert smoke.detail_of(httpx.Response(403, json={"detail": "admin only"})) == "admin only"
    assert smoke.detail_of(httpx.Response(502, text="bad gateway")) == "bad gateway"


# --- docker checks and the command line -------------------------------------------------------


def test_container_snippets_compile_and_each_image_gets_a_tool_it_has():
    for snippet in (smoke.NET_PY, smoke.DIRECT_PY, smoke.RAG_ONLY_PY, smoke.INGEST_LEFT_PY, smoke.THREADS_LEFT_PY, smoke.AUDIT_KEY_PY):
        compile(snippet, "snippet", "exec")
    assert smoke.reach_argv("llm", "auth", 8007)[0] == "sh" and "curl" in smoke.reach_argv("llm", "auth", 8007)[2]
    assert smoke.reach_argv("sql", "qdrant", 6333) == ["python", "-c", smoke.NET_PY, "qdrant", "6333"]
    assert set(smoke.COMPUTE_ONLY) == {"llm", "models", "sql"}


def test_last_json_takes_the_last_json_line():
    assert smoke.last_json('warning: x\n{"a": 1}\n{"b": 2}\n') == {"b": 2}
    with pytest.raises(ValueError):
        smoke.last_json("Traceback (most recent call last):")


def test_compose_exec_reports_a_missing_docker(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(smoke.subprocess, "run", missing)
    code, out = smoke.compose_exec("rag", ["true"], ".")
    assert code == 127 and "FileNotFoundError" in out


def test_command_line(monkeypatch, capsys):
    args = smoke.parse_args(["--admin-user", "admin"])
    assert (args.url, args.origin, args.docker, args.skip_lockout, args.cleanup) == ("http://localhost:8000", None, False, False, False)
    with pytest.raises(SystemExit) as exc:
        smoke.parse_args(["--admin-user", "admin", "--url", "localhost:8000"])
    assert exc.value.code == 2
    monkeypatch.delenv("RAG_ADMIN_PASSWORD", raising=False)
    assert smoke.main(["--admin-user", "admin"]) == 2
    assert "RAG_ADMIN_PASSWORD" in capsys.readouterr().err


# --- account deletion that goes on in the background (202) --------------------------------------


class Lists:
    """GET /api/admin/users answers, one per poll."""

    def __init__(self, *pages):
        self.pages = list(pages)

    def call(self, method, path, acc=None, **kwargs):
        assert (method, path) == ("GET", "/api/admin/users")
        return httpx.Response(200, json=self.pages.pop(0))


def test_a_deletion_answered_202_is_followed_until_the_account_is_gone():
    running = [{"user_id": "u1", "deletion": {"running": True, "error": None}}]
    r = smoke.settle_deletion(Lists(running, running, []), None, "u1", httpx.Response(202), poll_s=0)
    assert r.status_code == 204


def test_a_deletion_that_stops_in_the_background_reports_why():
    stopped = [{"user_id": "u1", "deletion": {"running": False, "error": "threads: disk full"}}]
    r = smoke.settle_deletion(Lists(stopped), None, "u1", httpx.Response(202), poll_s=0)
    assert r.status_code == 502 and smoke.detail_of(r) == "threads: disk full"


def test_a_finished_answer_is_returned_as_it_is():
    assert smoke.settle_deletion(Lists(), None, "u1", httpx.Response(204), poll_s=0).status_code == 204
