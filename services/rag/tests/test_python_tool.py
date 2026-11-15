"""run_python (rag.sandbox): offered only when the setting or the request asks; ships only the request's selected
documents, fetched as the signed caller, once per request; data too large is a tool error, never cut; every
failure is a tool error (status "error") the model and the step see, with a notice when the sandbox itself failed;
a run that worked is a numbered source; the sandbox client sends no identity, and the frame it sends."""

import asyncio
import json

import httpx
import pytest
from test_deepagents import (QUESTION, REPORT, TABLES, USER, AnswerLLM, Ingest, Retrieval, Scripted, Sql, call, done,
                             event, steps, todos)

from rag_common.http import ServiceError, make_client
from rag_common.identity import Identity, current_identity
from rag_common.sandbox import decode

import rag.agent_deepagents_stock as stock
import rag.sandbox as sandbox_module
from rag import arms, notices
from rag.agent import SourceBook, ToolContext, _tools
from rag.chat import ChatRequest, deep_python, request_problem
from rag.clients import RetryPolicy, SandboxClient, SqlClient, TooLarge
from rag.config import Settings
from rag.conversation import Conversation
from rag.sandbox import folder, tail_tokens

OTHER = "f" * 32  # a document of the same user that is not selected
DUCKDB = b"DUCK" + bytes(range(256)) * 4


class DataSql(Sql):
    """sql with the tables file: records which documents were fetched and as whom."""

    def __init__(self, size: int | None = None, delay: float = 0.0):
        super().__init__()
        self.files: list[tuple[str, str | None]] = []
        self.size, self.delay = size, delay

    async def tables_file(self, doc_id, max_bytes, reserve=None, unreserve=None):
        """As SqlClient.tables_file: the size is checked and reserved before the body arrives."""
        ident = current_identity.get()
        self.files.append((doc_id, ident.user_id if ident else None))
        data = DUCKDB if self.size is None else b"x" * self.size
        if len(data) > max_bytes:
            raise TooLarge(len(data))
        if reserve:
            reserve(len(data))
        await asyncio.sleep(self.delay)  # the download: other requests run meanwhile
        return bytearray(data)


class DataIngest(Ingest):
    def __init__(self, passages: int = 3):
        super().__init__()
        self.pages: list[tuple[str, int, str | None]] = []
        self.passages = passages

    async def chunks(self, doc_id, offset, limit):
        ident = current_identity.get()
        self.pages.append((doc_id, offset, ident.user_id if ident else None))
        records = [{"id": f"p{i}", "chunk_index": i, "kind": "text", "page": i + 1, "section": "", "text": f"passage {i}",
                    "user_id": USER, "doc_id": doc_id, "n_tokens": 3} for i in range(self.passages)]
        return records[offset:offset + limit]


def reply(stdout: str = "", **changes) -> dict:
    return {"exit_code": 0, "signal": None, "timed_out": False, "duration_ms": 1234, "timeout_s": 120.0,
            "stdout": stdout, "stdout_bytes": len(stdout), "stderr": "", "stderr_bytes": 0, "files": [],
            "files_total": 0, "error": None, **changes}


class Sandbox:
    """The sandbox client: records each run (code, time limit, files, identity in context) and plays replies."""

    def __init__(self, *replies: dict, error: Exception | None = None):
        self.runs: list[tuple[str, float, dict[str, bytes], str | None]] = []
        self.replies = list(replies)
        self.error = error

    async def execute(self, code, timeout_s, files, run_id=None):
        ident = current_identity.get()
        self.runs.append((code, timeout_s, dict(files), ident.user_id if ident else None))
        if self.error:
            raise self.error
        return self.replies.pop(0) if self.replies else reply("42\n")


def on(**changes) -> Settings:
    return Settings(deep_python="on", sandbox_url="http://sandbox:8008", **changes)


def deps(sandbox: Sandbox | None = None, s: Settings | None = None, sql: Sql | None = None, ingest=None, model=None):
    from types import SimpleNamespace

    return SimpleNamespace(s=s or on(), sql=sql or DataSql(), ingest=ingest or DataIngest(),
                           sandbox=sandbox or Sandbox(), retrieval=Retrieval(), llm=AnswerLLM(),
                           graph_llm=model or Scripted())


def context(d, docs=(TABLES, REPORT), python: bool | None = None) -> ToolContext:
    req = ChatRequest(question="q", doc_ids=[x.doc_id for x in docs], mode="deep", python=python)
    return ToolContext(d, req, list(docs), SourceBook())


def as_user(coro):
    async def go():
        queue = notices.begin()
        token = current_identity.set(Identity(USER, "user", "auth"))
        try:
            return await coro(), queue
        finally:
            current_identity.reset(token)
    return asyncio.run(go())


def run_python(ctx: ToolContext, code: str, cid: str = "c1"):
    """The tool message the model reads, and the notices: invoked as a tool call, as the arms invoke it."""
    tool = next(t for t in _tools(ctx) if t.name == "run_python")
    return as_user(lambda: tool.ainvoke({"type": "tool_call", "name": "run_python", "args": {"code": code}, "id": cid}))


@pytest.fixture(autouse=True)
def nothing_held():
    """Each test starts with no Python data held: a request dropped without close() gives its bytes back when its
    session is collected (the finalizer), which the cyclic collector does here."""
    import gc

    gc.collect()
    assert sandbox_module.held_bytes() == 0
    yield


# --- the switch ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("setting, field, offered", [
    ("off", None, False), ("off", True, False), ("off", False, False),
    ("request", None, False), ("request", True, True), ("request", False, False),
    ("on", None, True), ("on", True, True), ("on", False, False)])
def test_the_setting_and_the_request_decide_whether_the_tool_is_offered(setting, field, offered):
    ctx = context(deps(s=Settings(deep_python=setting, sandbox_url="http://sandbox:8008")), python=field)
    assert deep_python(ctx.s, ctx.req) is offered
    assert ("run_python" in {t.name for t in _tools(ctx)}) is offered
    assert ("run_python" in {t.name for t in stock.stock_tools(_tools(ctx))}) is offered  # deepagents_stock


def test_python_is_off_by_default_and_asking_for_it_then_is_a_422():
    s = Settings()
    assert s.deep_python == "off" and s.sandbox_url == "" and ChatRequest(question="q").python is None
    req = ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep", python=True)
    assert "RAG_DEEP_PYTHON=off" in request_problem(s, req)
    assert request_problem(Settings(deep_python="request", sandbox_url="http://sandbox:8008"), req) is None
    assert request_problem(s, ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep", python=False)) is None
    assert request_problem(s, ChatRequest(question="q", doc_ids=[TABLES.doc_id], python=True)) is None  # single mode
    with pytest.raises(ValueError, match="RAG_SANDBOX_URL"):
        Settings(deep_python="request")
    with pytest.raises(ValueError):
        Settings(deep_python="true", sandbox_url="http://sandbox:8008")  # off, request or on only
    with pytest.raises(ValueError, match="RAG_SANDBOX_REQUEST_TIMEOUT_S"):
        Settings(sandbox_timeout_s=700)
    with pytest.raises(ValueError, match="RAG_SANDBOX_SUBNET"):
        Settings(sandbox_subnet="not-a-net")


# --- what is shipped ----------------------------------------------------------------------------------------------


def test_only_the_selected_documents_are_shipped_fetched_as_the_caller():
    d = deps()
    message, _ = run_python(context(d), "print(42)")
    assert message.status == "success"
    code, timeout_s, files, who = d.sandbox.runs[0]
    assert (code, timeout_s, who) == ("print(42)", 120.0, USER)
    assert sorted(files) == ["data/1_orders.xlsx/tables.duckdb", "data/2_report.pdf/text.jsonl", "data/manifest.json"]
    assert d.sql.files == [(TABLES.doc_id, USER)]  # only the selected document with tables
    assert {p[0] for p in d.ingest.pages} == {REPORT.doc_id} and {p[2] for p in d.ingest.pages} == {USER}
    assert files["data/1_orders.xlsx/tables.duckdb"] == DUCKDB
    manifest = json.loads(files["data/manifest.json"])
    assert [m["doc_id"] for m in manifest["documents"]] == [TABLES.doc_id, REPORT.doc_id]
    assert OTHER not in files["data/manifest.json"].decode()
    orders = manifest["documents"][0]
    assert orders["tables_file"] == "data/1_orders.xlsx/tables.duckdb" and orders["text_file"] is None
    assert orders["tables"][0]["table"] == "orders" and orders["tables"][0]["columns"][1] == {"name": "revenue",
                                                                                              "type": "number"}
    lines = [json.loads(x) for x in files["data/2_report.pdf/text.jsonl"].decode().splitlines()]
    assert lines[0] == {"chunk_index": 0, "kind": "text", "page": 1, "text": "passage 0"}  # no owner, no ids, no blanks


def test_long_texts_are_read_page_by_page():
    d = deps(ingest=DataIngest(passages=2500))
    run_python(context(d, docs=(REPORT,)), "print(1)")
    assert [p[1] for p in d.ingest.pages] == [0, 1000, 2000]
    assert d.sandbox.runs[0][2]["data/1_report.pdf/text.jsonl"].count(b"\n") == 2500


def test_python_tool_advertises_the_exact_selected_input_paths_before_fetching():
    d = deps()
    ctx = context(d)
    tool = next(t for t in _tools(ctx) if t.name == "run_python")
    advertised = json.loads(tool.description.rsplit("\n", 1)[1])
    assert d.sql.files == [] and d.ingest.pages == [] and d.sandbox.runs == []
    run_python(ctx, "print(42)")
    files = d.sandbox.runs[0][2]
    manifest = json.loads(files["data/manifest.json"])
    assert advertised == [{key: entry[key] for key in ("doc_id", "tables_file", "text_file")}
                          for entry in manifest["documents"]]
    assert all(entry[key] in files for entry in advertised for key in ("tables_file", "text_file") if entry[key])
    assert OTHER not in tool.description


def test_missing_input_file_returns_the_manifest_and_pandas_recovery_guidance():
    trace = "FileNotFoundError: [Errno 2] No such file or directory: 'data/guessed-id/orders.xlsx'"
    ctx = context(deps(Sandbox(reply(exit_code=1, stderr=trace))))
    message, _ = run_python(ctx, "import pandas as pd; pd.read_excel('data/guessed-id/orders.xlsx')")
    assert message.status == "error" and trace in message.content
    assert "data/manifest.json" in message.content and "Original uploads are not present" in message.content
    assert "read_only=True" in message.content and ".df()" in message.content
    assert ctx.book.sources == []


def test_the_data_is_fetched_once_per_request_and_identical_code_is_not_run_again():
    d = deps(Sandbox(reply("1\n"), reply("2\n")))
    ctx = context(d)
    first, _ = run_python(ctx, "print(1)", "c1")
    second, _ = run_python(ctx, "print(2)", "c2")
    again, _ = run_python(ctx, "print(1)", "c3")
    assert len(d.sql.files) == 1 and len(d.sandbox.runs) == 2
    assert d.sandbox.runs[0][2] == d.sandbox.runs[1][2]
    assert "already made this exact call" in again.content and again.artifact == first.artifact
    other, _ = run_python(ctx, "print(1) ", "c4")  # whitespace differs: different code, run again
    assert len(d.sandbox.runs) == 3 and other.status == "success"


def test_data_over_the_limit_is_a_tool_error_naming_the_setting_never_cut():
    d = deps(sql=DataSql(size=3 * 1024 * 1024), s=on(sandbox_input_mb=2))
    ctx = context(d)
    message, queue = run_python(ctx, "print(1)")
    assert message.status == "error" and "RAG_SANDBOX_INPUT_MB" in message.content and "orders.xlsx" in message.content
    assert d.sandbox.runs == [] and [n["code"] for n in queue] == ["python_input_too_large"]
    message, _ = run_python(ctx, "print(2)", "c2")  # refused again without fetching again
    assert message.status == "error" and len(d.sql.files) == 1 and d.sandbox.runs == []


def test_a_failure_to_fetch_the_data_is_a_tool_error_and_a_notice():
    class Broken(DataSql):
        async def tables_file(self, doc_id, max_bytes, reserve=None, unreserve=None):
            raise ServiceError(502, "sql unreachable (ConnectError)")

    message, queue = run_python(context(deps(sql=Broken())), "print(1)")
    assert message.status == "error" and "could not be fetched" in message.content
    assert [n["code"] for n in queue] == ["python_unavailable"]


def test_document_folders_are_safe_names():
    assert folder(3, "Q3 report (final).pdf") == "data/3_Q3_report_final_.pdf"
    assert folder(1, "../../etc/passwd") == "data/1_etc_passwd" and folder(2, "...") == "data/2_document"


# --- results and errors -------------------------------------------------------------------------------------------


def test_a_run_that_worked_is_a_numbered_source_with_its_code_and_output():
    d = deps(Sandbox(reply("South 2735.62\n", stderr="FutureWarning: x", files=[
        {"path": "by_region.csv", "size": 20, "text": "region,total\nSouth,1\n"}], files_total=1)))
    ctx = context(d)
    message, _ = run_python(ctx, "import pandas as pd\nprint('South', 2735.62)")
    assert message.status == "success" and message.artifact == [1]
    assert message.content.startswith("[1] Python run (exit code 0, 1.2 s)\nSouth 2735.62")
    assert "FutureWarning" in message.content and "by_region.csv (20 bytes)" in message.content
    src = ctx.book.get(1)
    assert src.kind == "python" and src.doc_name == "Python" and "import pandas as pd" in src.text
    assert "South 2735.62" in src.text and src.location.startswith("Python run")


@pytest.mark.parametrize("result, words", [
    (reply(exit_code=1, stderr="Traceback (most recent call last):\n  File \"main.py\", line 2\nKeyError: 'regon'"),
     ["exit code 1", "Traceback (most recent call last)", "KeyError: 'regon'"]),
    (reply("partial\n", exit_code=None, timed_out=True, error="the run passed its time limit (120 s) and was stopped"),
     ["timed out after 120 s", "RAG_SANDBOX_TIMEOUT_S", "partial"]),
    (reply(exit_code=1, error="a file it wrote, or its output, passed 256 MB (SANDBOX_MAX_FILE_MB)",
           stderr="OSError: [Errno 27] File too large"), ["SANDBOX_MAX_FILE_MB", "File too large"]),
    (reply(exit_code=None, signal="SIGKILL", error="it was killed (SIGKILL), most likely for using more memory"),
     ["SIGKILL"]),
    (reply(""), ["printed nothing"]),
])
def test_every_failed_run_is_a_tool_error_the_model_reads(result, words):
    d = deps(Sandbox(result))
    ctx = context(d)
    message, queue = run_python(ctx, "print(x)")
    assert message.status == "error" and all(w in message.content for w in words), message.content
    assert ctx.book.sources == [] and queue == []  # the code's own failure: no source, no notice


@pytest.mark.parametrize("status, words", [(503, "busy"), (502, "could not be reached"), (504, "could not be reached"),
                                           (400, "refused")])
def test_a_sandbox_that_cannot_run_the_code_is_a_tool_error_and_a_notice(status, words):
    message, queue = run_python(context(deps(Sandbox(error=ServiceError(status, "detail")))), "print(1)")
    assert message.status == "error" and words in message.content
    assert [n["code"] for n in queue] == ["python_unavailable"]


def test_empty_and_overlong_code_are_refused_before_anything_is_sent():
    d = deps(s=on(sandbox_code_chars=100))
    for code in ("   ", "x = 1\n" * 50):
        message, _ = run_python(context(d), code)
        assert message.status == "error"
    assert d.sandbox.runs == [] and d.sql.files == []


def test_long_output_is_cut_for_the_model_with_a_note_and_a_traceback_keeps_its_end():
    s = on(sandbox_result_tokens=200)
    message, _ = run_python(context(deps(Sandbox(reply("line\n" * 3000)), s=s)), "print(1)")
    assert "RAG_SANDBOX_RESULT_TOKENS" in message.content and len(message.content) < 3000
    trace = "Traceback (most recent call last):\n" + "  frame\n" * 2000 + "ValueError: the real problem"
    assert tail_tokens(trace, 50).endswith("ValueError: the real problem")


# --- through the arms ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("agent", ["langgraph", "deepagents", "deepagents_stock"])
def test_every_core_arm_runs_python_and_streams_its_step_and_errors(agent):
    lead = [todos(("Q3 revenue", "in_progress"))] if agent == "deepagents" else []
    model = Scripted(steps=[*lead, call("run_python", {"code": "print(undefined)"}),
                            call("run_python", {"code": "print('South', 2735.62)"}), done(), done()])
    d = deps(Sandbox(reply(exit_code=1, stderr="Traceback (most recent call last):\nNameError: name 'undefined'"),
                     reply("South 2735.62\n")), model=model)
    req = ChatRequest(question=QUESTION, doc_ids=[TABLES.doc_id], mode="deep", agent=agent, python=True)

    async def go():
        return [e async for e in arms.get(agent).deep_search(d, req, [TABLES], Conversation(), QUESTION, {})]

    events, _ = as_user(go)
    ran = steps(events, {"run_python"})
    assert [x["args"] for x in ran] == [{"code": "print(undefined)"}, {"code": "print('South', 2735.62)"}]
    assert "NameError" in ran[0]["error"] and ran[1]["error"] is None and ran[1]["sources"] == [1]
    assert [s["kind"] for s in event(events, "sources")["sources"]] == ["python"]
    assert any("run_python" in kwargs.get("tool_names", []) for _, kwargs in model.calls)


# --- the client ---------------------------------------------------------------------------------------------------


def test_the_sandbox_client_sends_the_frame_and_no_identity_even_inside_a_users_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["job"] = decode(request.read(), max_header_bytes=1 << 20, max_files=10)
        return httpx.Response(200, json=reply("ok\n"))

    async def go():
        http = make_client("http://sandbox:8008")  # as rag.api makes it: no signer
        http._transport = httpx.MockTransport(handler)
        return await SandboxClient(http).execute("print('ok')", 60.0, [("data/manifest.json", b"{}")])

    result, _ = as_user(go)
    assert result["stdout"] == "ok\n" and "x-identity" not in seen["headers"]
    assert seen["headers"]["content-type"] == "application/octet-stream"
    assert seen["job"].code == "print('ok')" and seen["job"].timeout_s == 60.0
    assert [(f.path, bytes(f.data)) for f in seen["job"].files] == [("data/manifest.json", b"{}")]


def test_the_tables_file_download_stops_at_the_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000, headers={"content-length": "5000"})

    async def go(limit):
        http = httpx.AsyncClient(base_url="http://sql:8005", transport=httpx.MockTransport(handler))
        return await SqlClient(http).tables_file(TABLES.doc_id, limit)

    assert asyncio.run(go(5000)) == b"x" * 5000
    with pytest.raises(TooLarge) as caught:
        asyncio.run(go(4999))
    assert caught.value.size == 5000


# --- fair use: one run per user, run time per question, data held for all questions ------------------------------


class SlowSandbox(Sandbox):
    """Records how many runs are in flight at once, per user."""

    def __init__(self, delay: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.delay, self.inflight, self.most = delay, {}, {}

    async def execute(self, code, timeout_s, files, run_id=None):
        ident = current_identity.get()
        user = ident.user_id if ident else None
        self.inflight[user] = self.inflight.get(user, 0) + 1
        self.most[user] = max(self.most.get(user, 0), self.inflight[user])
        await asyncio.sleep(self.delay)
        self.inflight[user] -= 1
        self.runs.append((code, timeout_s, dict(files), user))
        return reply("ok\n")


def test_one_run_per_user_at_a_time_while_other_users_run_beside_it():
    from rag_common.testing import new_user_id

    sandbox = SlowSandbox()
    other = new_user_id()

    async def go():
        notices.begin()

        async def as_(user, code):
            token = current_identity.set(Identity(user, "user", "auth"))
            try:
                ctx = context(deps(sandbox))
                tool = next(t for t in _tools(ctx) if t.name == "run_python")
                return await tool.ainvoke({"type": "tool_call", "name": "run_python", "args": {"code": code}, "id": code})
            finally:
                current_identity.reset(token)

        return await asyncio.gather(as_(USER, "print(1)"), as_(USER, "print(2)"), as_(other, "print(3)"),
                                    as_(other, "print(4)"))

    messages = asyncio.run(go())
    assert all(m.status == "success" for m in messages)
    assert sandbox.most == {USER: 1, other: 1} and len(sandbox.runs) == 4
    assert sandbox_module.user_turns._users == {}  # nothing kept once idle


def test_a_users_run_that_cannot_get_a_turn_is_a_tool_error_with_a_notice(monkeypatch):
    async def go():
        notices.begin()
        token = current_identity.set(Identity(USER, "user", "auth"))
        try:
            async with sandbox_module.user_turns.turn(USER, 1, 5):  # the user's other run, still going
                ctx = context(deps(s=on(sandbox_request_timeout_s=200, sandbox_timeout_s=100)))
                monkeypatch.setattr(ctx.s, "sandbox_request_timeout_s", 0.05)
                tool = next(t for t in _tools(ctx) if t.name == "run_python")
                message = await tool.ainvoke({"type": "tool_call", "name": "run_python", "args": {"code": "print(1)"},
                                              "id": "c1"})
                return message, notices.collected()
        finally:
            current_identity.reset(token)

    message, queue = asyncio.run(go())
    assert message.status == "error" and "RAG_SANDBOX_USER_RUNS" in message.content
    assert [n["code"] for n in queue] == ["python_busy"]


def test_a_question_has_a_run_time_budget_and_each_run_gets_what_is_left():
    d = deps(Sandbox(reply("a\n", duration_ms=6000), reply("b\n", duration_ms=4000)),
             s=on(sandbox_request_seconds=10))
    ctx = context(d)
    first, _ = run_python(ctx, "print('a')", "c1")
    second, _ = run_python(ctx, "print('b')", "c2")
    third, queue = run_python(ctx, "print('c')", "c3")
    assert first.status == second.status == "success"
    assert [r[1] for r in d.sandbox.runs] == [10.0, 4.0]  # the second run's limit is what was left
    assert third.status == "error" and "RAG_SANDBOX_REQUEST_SECONDS" in third.content and len(d.sandbox.runs) == 2
    assert [n["code"] for n in queue] == ["python_time_used"]


def test_a_timed_out_run_counts_its_whole_limit():
    d = deps(Sandbox(reply(exit_code=None, timed_out=True, timeout_s=8.0, duration_ms=100)),
             s=on(sandbox_request_seconds=10))
    ctx = context(d)
    run_python(ctx, "while True: pass", "c1")
    run_python(ctx, "print(1)", "c2")
    assert [r[1] for r in d.sandbox.runs] == [10.0, 2.0]


def test_data_held_for_all_questions_is_capped_and_given_back_when_the_research_ends():
    assert sandbox_module.held_bytes() == 0
    d = deps(s=on(sandbox_total_input_mb=1))
    first = context(d)
    message, _ = run_python(first, "print(1)", "c1")
    held = sandbox_module.held_bytes()
    assert message.status == "success" and held > 0  # the first question's data is held
    big = DataSql(size=1024 * 1024 - held + 10)  # with it, a second question would pass the shared cap
    second = context(deps(sql=big, s=on(sandbox_total_input_mb=1)))
    message, queue = run_python(second, "print(2)", "c2")
    assert message.status == "error" and "RAG_SANDBOX_TOTAL_INPUT_MB" in message.content
    assert [n["code"] for n in queue] == ["python_busy"] and sandbox_module.held_bytes() == held
    sandbox_module.close(first)
    assert sandbox_module.held_bytes() == 0
    message, _ = run_python(second, "print(2)", "c3")  # not refused for good: it works once there is room
    assert message.status == "success"
    sandbox_module.close(second)
    assert sandbox_module.held_bytes() == 0


@pytest.mark.parametrize("agent", ["langgraph", "deepagents"])
def test_every_arm_gives_the_data_back_when_its_research_ends(agent):
    model = Scripted(steps=[*([todos(("Q3", "in_progress"))] if agent == "deepagents" else []),
                            call("run_python", {"code": "print(1)"}), done(), done()])
    d = deps(model=model)
    req = ChatRequest(question=QUESTION, doc_ids=[TABLES.doc_id], mode="deep", agent=agent, python=True)

    async def go():
        return [e async for e in arms.get(agent).deep_search(d, req, [TABLES], Conversation(), QUESTION, {})]

    events, _ = as_user(go)
    assert steps(events, {"run_python"})[0]["error"] is None and sandbox_module.held_bytes() == 0


# --- reservations before downloads, fetch slots, cancellation --------------------------------------------------------


class PeakSql(DataSql):
    """Records the shared total right after each reservation and how many downloads are in flight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.peak, self.inflight, self.most = 0, 0, 0

    async def tables_file(self, doc_id, max_bytes, reserve=None, unreserve=None):
        self.inflight += 1
        self.most = max(self.most, self.inflight)
        try:
            data = DUCKDB if self.size is None else b"x" * self.size
            if len(data) > max_bytes:
                raise TooLarge(len(data))
            if reserve:
                reserve(len(data))
            self.peak = max(self.peak, sandbox_module.held_bytes())
            await asyncio.sleep(self.delay)
            return bytearray(data)
        finally:
            self.inflight -= 1


def _many(users, sql, s, sandbox=None):
    """One run_python per user, all at once, each in a request of its own; (messages, notice codes)."""
    async def go():
        notices.begin()

        async def one(user, i):
            token = current_identity.set(Identity(user, "user", "auth"))
            try:
                ctx = context(deps(sandbox or Sandbox(), sql=sql, s=s), docs=(TABLES,))
                tool = next(t for t in _tools(ctx) if t.name == "run_python")
                message = await tool.ainvoke({"type": "tool_call", "name": "run_python",
                                              "args": {"code": f"print({i})"}, "id": f"c{i}"})
                return message, ctx
            finally:
                current_identity.reset(token)

        out = await asyncio.gather(*(one(u, i) for i, u in enumerate(users)))
        return out, [n["code"] for n in notices.collected()]
    return asyncio.run(go())


def test_concurrent_requests_never_reserve_more_than_the_shared_total():
    from rag_common.testing import new_user_id

    size = 300 * 1024
    sql = PeakSql(size=size, delay=0.05)
    s = on(sandbox_total_input_mb=1, sandbox_fetches=20)  # room for 3 tables files of 300 KiB, not 12
    results, codes = _many([new_user_id() for _ in range(12)], sql, s)
    ok = [ctx for m, ctx in results if m.status == "success"]
    assert sql.most > 3  # the downloads did overlap
    assert sql.peak <= 1024 * 1024 and 1 <= len(ok) <= 3
    assert sum(m.status == "error" and "RAG_SANDBOX_TOTAL_INPUT_MB" in m.content for m, _ in results) == 12 - len(ok)
    assert "python_busy" in codes
    for ctx in ok:
        sandbox_module.close(ctx)
    assert sandbox_module.held_bytes() == 0


def test_one_fetch_per_user_and_a_global_cap_on_fetches():
    from rag_common.testing import new_user_id

    same = PeakSql(delay=0.03)
    results, _ = _many([USER] * 5, same, on(sandbox_fetches=10))
    assert same.most == 1 and all(m.status == "success" for m, _ in results)
    for _, ctx in results:
        sandbox_module.close(ctx)
    spread = PeakSql(delay=0.03)
    results, _ = _many([new_user_id() for _ in range(6)], spread, on(sandbox_fetches=2))
    assert spread.most == 2 and all(m.status == "success" for m, _ in results)
    for _, ctx in results:
        sandbox_module.close(ctx)
    assert sandbox_module.held_bytes() == 0


def test_a_cancelled_download_gives_its_reservation_back_at_once():
    sql = PeakSql(size=200 * 1024, delay=5)

    async def go():
        notices.begin()
        token = current_identity.set(Identity(USER, "user", "auth"))
        try:
            ctx = context(deps(sql=sql), docs=(TABLES,))
            tool = next(t for t in _tools(ctx) if t.name == "run_python")
            task = asyncio.ensure_future(tool.ainvoke({"type": "tool_call", "name": "run_python",
                                                       "args": {"code": "print(1)"}, "id": "c1"}))
            while sql.peak == 0:
                await asyncio.sleep(0.01)
            held_during = sandbox_module.held_bytes()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return held_during, sandbox_module.held_bytes()
        finally:
            current_identity.reset(token)

    during, after = asyncio.run(go())
    assert during >= 200 * 1024 and after == 0  # no close() needed


def test_the_tables_file_is_reserved_before_its_body_and_kept_in_one_buffer():
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"y" * 3000, headers={"content-length": "3000"})

    async def go(transport, limit):
        http = httpx.AsyncClient(base_url="http://sql:8005", transport=transport)
        return await SqlClient(http).tables_file(TABLES.doc_id, limit, lambda n: events.append(("reserve", n)),
                                                 lambda n: events.append(("unreserve", n)))

    data = asyncio.run(go(httpx.MockTransport(handler), 5000))
    assert isinstance(data, bytearray) and bytes(data) == b"y" * 3000
    assert events == [("reserve", 3000), ("unreserve", 0)]

    class Refused(Exception):
        pass

    def refuse(n):
        raise Refused()

    async def refused():
        http = httpx.AsyncClient(base_url="http://sql:8005", transport=httpx.MockTransport(handler))
        return await SqlClient(http).tables_file(TABLES.doc_id, 5000, refuse)

    with pytest.raises(Refused):
        asyncio.run(refused())  # the reservation can refuse before the body is read


def test_a_failure_mid_body_gives_the_whole_reservation_back():
    events = []

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"z" * 100
            raise httpx.ReadError("connection lost")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Broken(), headers={"content-length": "1000"})

    async def go():
        http = httpx.AsyncClient(base_url="http://sql:8005", transport=httpx.MockTransport(handler))
        return await SqlClient(http, RetryPolicy(backoff_s=0)).tables_file(
            TABLES.doc_id, 5000, lambda n: events.append(("reserve", n)), lambda n: events.append(("unreserve", n)))

    with pytest.raises(Exception):
        asyncio.run(go())
    reserved = sum(n for k, n in events if k == "reserve")
    assert reserved == sum(n for k, n in events if k == "unreserve") and reserved >= 1000  # the retry too


class BlockingSandbox(Sandbox):
    """A run that only ends when cancelled; cancel() records the run id and whether the user's turn was held."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.ids: list[str] = []
        self.cancelled: list[tuple[str, bool]] = []

    async def execute(self, code, timeout_s, files, run_id=None):
        self.ids.append(run_id)
        self.started.set()
        await asyncio.sleep(3600)

    async def cancel(self, run_id):
        self.cancelled.append((run_id, USER in sandbox_module.user_turns._users))
        await asyncio.sleep(0.05)  # the sandbox kills the run and cleans up
        return {"ended": True}


def test_a_cancelled_request_kills_its_run_and_keeps_the_users_turn_until_it_ended():
    async def go():
        notices.begin()
        sandbox = BlockingSandbox()
        token = current_identity.set(Identity(USER, "user", "auth"))
        try:
            ctx = context(deps(sandbox), docs=(TABLES,))
            tool = next(t for t in _tools(ctx) if t.name == "run_python")
            task = asyncio.ensure_future(tool.ainvoke({"type": "tool_call", "name": "run_python",
                                                       "args": {"code": "print(1)"}, "id": "c1"}))
            await sandbox.started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            sandbox_module.close(ctx)
            return sandbox, USER in sandbox_module.user_turns._users
        finally:
            current_identity.reset(token)

    sandbox, still_held = asyncio.run(go())
    assert len(sandbox.ids[0]) == 32 and sandbox.cancelled == [(sandbox.ids[0], True)]
    assert still_held is False  # given back once the sandbox said the run ended


def test_the_sandbox_client_sends_the_run_id_and_streams_in_small_pieces():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["job"] = decode(request.read(), max_header_bytes=1 << 20, max_files=10)
        return httpx.Response(200, json=reply("ok\n"))

    async def go():
        http = httpx.AsyncClient(base_url="http://sandbox:8008", transport=httpx.MockTransport(handler))
        big = bytearray(b"q" * (3 * 1024 * 1024 + 5))
        return await SandboxClient(http).execute("print(1)", 5.0, [("data/big.bin", big)], "a" * 32)

    asyncio.run(go())
    assert seen["headers"]["x-run-id"] == "a" * 32
    assert len(seen["job"].files[0].data) == 3 * 1024 * 1024 + 5
