"""Search tasks in rag (rag.search_tasks): a preset or one of the caller's own custom tasks, never another user's;
every embedding of a turn's question with the turn's task (single-shot, each deep-research search, fan-out
researchers, the per-document candidates, which share the dense vector); exhaustive mode ignoring it; and records
and logs that name the task's kind, never a custom task's text."""

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_deepagents import QUESTION, REPORT, TABLES, USER, AnswerLLM, Ingest, Sql, call, calls, done
from test_deepagents import Retrieval as ScriptedRetrieval
from test_fanout import Routed, task

from conftest import KEYS
from rag_common.contracts import DocStatus, DocType, DocumentInfo
from rag_common.http import make_client
from rag_common.identity import Identity, Verifier, current_identity
from rag_common.index import IMAGE_VECTOR, TEXT_VECTOR
from rag_common.testing import new_user_id

import rag.api as api
from rag import arms, notices
from rag.agent import SourceBook, ToolContext, search_tool, text_searcher
from rag.chat import ChatRequest, request_problem, request_settings, search_options, single_shot, with_settings
from rag.clients import ModelsClient, RetryPolicy, ServiceError
from rag.config import Settings
from rag.conversation import Conversation, ThreadsClient
from rag.retrieval import Retrieval, SearchOptions, SearchResult
from rag.search_tasks import BadSearchTask, SearchTask, presets, resolve

A, B = new_user_id(), new_user_id()
DOC_A, DOC_B = "a" * 32, "b" * 32
TASK_A = "1" * 32  # A's custom task
CURRENT = "Given a question about the user's documents, retrieve passages with the facts needed to answer it"
MULTI = "Given a multi-hop question, retrieve documents that can help answer the question"
CONTRACTS = "Given a question about an agreement, retrieve the clauses that answer it"


def as_user(user_id: str, make_coro):
    async def run():
        token = current_identity.set(Identity(user_id, "user", "auth"))
        try:
            return await make_coro()
        finally:
            current_identity.reset(token)
    return asyncio.run(run())


class ThreadsServer:
    """A fake threads service: verifies rag's signature and answers with the signed user's own search tasks."""

    def __init__(self, tasks: dict | None = None, status: int | None = None):
        self.verifier = Verifier.from_file(KEYS.public_keys_file, {"rag"})
        self.tasks = tasks if tasks is not None else {A: {TASK_A: CONTRACTS}}
        self.status = status
        self.seen: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        ident = self.verifier.verify(request.headers["x-identity"])
        self.seen.append((request.url.path, ident.user_id))
        if self.status:
            return httpx.Response(self.status, json={"detail": "database is locked"})
        task_id = request.url.path.rsplit("/", 1)[-1]
        instruction = self.tasks.get(ident.user_id, {}).get(task_id)
        if not request.url.path.startswith("/v1/search-tasks/") or instruction is None:
            return httpx.Response(404, json={"detail": "search task not found"})
        return httpx.Response(200, json={"task_id": task_id, "name": "Contracts", "instruction": instruction,
                                         "created_at": "t", "updated_at": "t"})


def threads_client(server: ThreadsServer) -> ThreadsClient:
    http = make_client("http://threads:8006", signer=KEYS.signer("rag"))
    http._transport = httpx.MockTransport(server)  # keeps the request hooks: request id and signature
    return ThreadsClient(http)


# --- resolution -------------------------------------------------------------------------------------------------


def test_presets_and_no_task_resolve_from_the_settings_without_threads():
    s = Settings()

    def get(task_id):
        return asyncio.run(resolve(s, None, task_id))  # presets never call threads

    assert get(None) == get("general") == SearchTask("general", "preset:general", CURRENT)
    assert get("multi-step") == SearchTask("multi-step", "preset:multi-step", MULTI)
    assert get("exact") == SearchTask("exact", "preset:exact", "")  # the bare question
    other = Settings(query_instruction="Retrieve passages relevant to the question")
    assert asyncio.run(resolve(other, None, None)).instruction == "Retrieve passages relevant to the question"
    for unknown in ("nope", "general-2", "f" * 31):
        with pytest.raises(BadSearchTask, match="unknown search task"):
            get(unknown)
    assert [(p["id"], p["name"], p["instruction"]) for p in presets(s)] == [
        ("general", "General", CURRENT), ("multi-step", "Multi-step research", MULTI), ("exact", "Exact lookup", "")]
    assert all(p["description"] and "measured" in p for p in presets(s))


def test_the_general_instruction_setting_is_one_checked_line():
    assert Settings(query_instruction="  Retrieve passages  ").query_instruction == "Retrieve passages"
    assert Settings(query_instruction="").query_instruction == ""
    for bad in ("two\nlines", "x" * 301):
        with pytest.raises(ValidationError, match="RAG_QUERY_INSTRUCTION"):
            Settings(query_instruction=bad)


def test_rag_reads_a_custom_task_with_the_callers_own_identity_and_leaves_scoping_to_threads():
    """Even a threads that answered anyone (no owner scoping, unlike the fake below) is asked as the caller: rag
    signs each read with the identity of the chat it serves, so threads' owner check (tests in services/threads,
    with identities signed by rag) decides."""
    seen = []
    verifier = Verifier.from_file(KEYS.public_keys_file, {"rag"})

    def careless(request: httpx.Request) -> httpx.Response:
        ident = verifier.verify(request.headers["x-identity"])
        seen.append((ident.user_id, ident.issuer, ident.role))
        return httpx.Response(200, json={"task_id": TASK_A, "name": "Contracts", "instruction": CONTRACTS,
                                         "created_at": "t", "updated_at": "t"})

    http = make_client("http://threads:8006", signer=KEYS.signer("rag"))
    http._transport = httpx.MockTransport(careless)
    for user in (A, B):
        as_user(user, lambda: resolve(Settings(), ThreadsClient(http), TASK_A))
    assert seen == [(A, "rag", "user"), (B, "rag", "user")]


def test_a_custom_task_is_read_for_the_caller_only_and_another_users_is_unknown():
    server = ThreadsServer()
    threads = threads_client(server)
    assert as_user(A, lambda: resolve(Settings(), threads, TASK_A)) == SearchTask(TASK_A, "custom", CONTRACTS)
    with pytest.raises(BadSearchTask, match="not one of yours"):
        as_user(B, lambda: resolve(Settings(), threads, TASK_A))
    assert server.seen == [(f"/v1/search-tasks/{TASK_A}", A), (f"/v1/search-tasks/{TASK_A}", B)]


def test_threads_down_is_a_service_error_and_a_stored_task_that_fails_the_check_is_refused():
    with pytest.raises(ServiceError):
        as_user(A, lambda: resolve(Settings(), threads_client(ThreadsServer(status=503)), TASK_A))
    broken = threads_client(ThreadsServer({A: {TASK_A: "two\nlines"}}))
    with pytest.raises(BadSearchTask, match="cannot be used: the instruction must be a single line"):
        as_user(A, lambda: resolve(Settings(), broken, TASK_A))


def test_resolving_logs_the_kind_never_the_custom_text(caplog):
    with caplog.at_level(logging.INFO, logger="rag.search_tasks"):
        as_user(A, lambda: resolve(Settings(), threads_client(ThreadsServer()), TASK_A))
    record = next(r for r in caplog.records if r.name == "rag.search_tasks")
    assert record.fields == {"search_task_kind": "custom"}
    assert CONTRACTS not in caplog.text


# --- the question's embedding -------------------------------------------------------------------------------------


def models_client(reply):
    """A ModelsClient whose models answers `reply(body)` and records each body sent on the wire."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json=reply(body))

    http = httpx.AsyncClient(base_url="http://models:8003", transport=httpx.MockTransport(handler))
    return ModelsClient(http, RetryPolicy(backoff_s=0.0)), sent


def test_embed_query_sends_the_instruction_and_checks_models_applied_it():
    client, sent = models_client(lambda body: {"vectors": [[0.5, 0.5]], "dim": 2, "instruction": body["instruction"]})
    assert asyncio.run(client.embed_query("which clause?", CONTRACTS)) == [0.5, 0.5]
    assert asyncio.run(client.embed_query("which clause?", "")) == [0.5, 0.5]  # Exact lookup: applied as none
    assert sent == [{"texts": ["which clause?"], "role": "query", "instruction": CONTRACTS},
                    {"texts": ["which clause?"], "role": "query", "instruction": ""}]


@pytest.mark.parametrize("reply", [
    {"vectors": [[1.0, 0.0]], "dim": 2},  # an earlier models: ignores the field, says nothing
    {"vectors": [[1.0, 0.0]], "dim": 2, "instruction": CURRENT},  # applied its own default instead
    {"vectors": [[1.0, 0.0]], "dim": 2, "instruction": None},
])
def test_a_models_that_did_not_apply_the_instruction_is_an_error_not_a_silent_default(reply, caplog):
    client, _ = models_client(lambda body: reply)
    with caplog.at_level(logging.ERROR, logger="rag.clients"), pytest.raises(ServiceError) as err:
        asyncio.run(client.embed_query("which clause?", CONTRACTS))
    assert err.value.status == 502 and "models does not apply search tasks" in err.value.detail
    assert "rebuild and restart models" in err.value.detail
    assert CONTRACTS not in caplog.text  # the log says what went wrong, never the custom text


class RecordingModels:
    def __init__(self):
        self.queries: list[tuple[str, str]] = []
        self.images: list[str] = []

    @staticmethod
    def vector(instruction: str) -> list[float]:
        return [float(len(instruction)), 1.0]

    async def embed_query(self, text, instruction):
        self.queries.append((text, instruction))
        return self.vector(instruction)

    async def embed_image_query(self, text):
        self.images.append(text)
        return [0.0, 1.0]


class RecordingQdrant:
    def __init__(self):
        self.requests, self.groups = [], []

    async def query_batch_points(self, collection, requests):
        self.requests += requests
        return [SimpleNamespace(points=[]) for _ in requests]

    async def query_points_groups(self, collection, **kwargs):
        self.groups.append(kwargs)
        return SimpleNamespace(groups=[])


def test_search_embeds_the_question_with_the_task_for_dense_search_and_the_per_document_candidates():
    for instruction, sent in ((None, CURRENT), ("", ""), (CONTRACTS, CONTRACTS)):
        models, qdrant = RecordingModels(), RecordingQdrant()
        retrieval = Retrieval(qdrant, models, None, Settings())
        opts = SearchOptions(retrievers=("dense", "image"), rerank=False, instruction=instruction)
        as_user(A, lambda: retrieval.search("q", [DOC_A], opts))  # noqa: B023 (run at once)
        assert models.queries == [("q", sent)]
        assert models.images == ["q"]  # image queries keep their own instruction
        dense = next(r for r in qdrant.requests if r.using == TEXT_VECTOR)
        assert dense.query == qdrant.groups[0]["query"] == RecordingModels.vector(sent)  # one vector for both
        assert next(r for r in qdrant.requests if r.using == IMAGE_VECTOR).query == [0.0, 1.0]


class RecordingRetrieval:
    def __init__(self):
        self.options: list[SearchOptions] = []

    async def search(self, query, doc_ids, options=None, scope=None):
        self.options.append(options)
        return SearchResult(candidates=[], terms=[], hits={})

    async def neighbours(self, cands):
        return {}

    async def fetch(self, ids):
        return []


def doc(doc_id: str, owner: str = A) -> DocumentInfo:
    return DocumentInfo(doc_id=doc_id, name=f"{doc_id[0]}.pdf", doc_type=DocType.PDF, size=1, status=DocStatus.READY,
                        n_chunks=3, created_at="t", updated_at="t", owner_id=owner)


def test_every_search_of_a_turn_uses_its_task_single_shot_and_deep_research_alike():
    req = ChatRequest(question="q", doc_ids=[DOC_A], mode="deep", search_task=TASK_A)
    req.use_task(SearchTask(TASK_A, "custom", CONTRACTS))
    retrieval = RecordingRetrieval()
    deps = SimpleNamespace(s=Settings(), retrieval=retrieval, planner=None, ingest=None, sql=None)

    async def turn():
        _ = [e async for e in single_shot(deps, req, [], Conversation(), "q", {})]
        ctx = ToolContext(deps, req, [doc(DOC_A)], SourceBook())
        await search_tool(ctx).ainvoke({"query": "the lead's search"})  # LangGraph and both deepagents arms
        await text_searcher(ctx)("verify's search of the text documents")

    as_user(A, turn)
    assert [o.instruction for o in retrieval.options] == [CONTRACTS] * 3
    assert search_options(ChatRequest(question="q")).instruction is None  # no task resolved: General


def test_fanout_researchers_search_with_the_turns_task():
    text = "Report: what report.pdf says about Q3 revenue"

    class TaskRetrieval(ScriptedRetrieval):
        def __init__(self):
            super().__init__()
            self.instructions = []

        async def search(self, query, doc_ids, options=None, scope=None):
            self.instructions.append(options.instruction)
            return await super().search(query, doc_ids, options, scope)

    model = Routed(scripts={
        QUESTION: [calls(task(text)), call("search", {"query": "Q3 revenue by region"}), done()],
        text: [call("search", {"query": f"Q3 revenue {i}"}) for i in range(2)] + [done("The report says 340 [1].")]})
    deps = SimpleNamespace(s=Settings(), graph_llm=model, sql=Sql(), retrieval=TaskRetrieval(), ingest=Ingest(),
                           llm=AnswerLLM())
    req = ChatRequest(question=QUESTION, doc_ids=[TABLES.doc_id, REPORT.doc_id], mode="deep", agent="deepagents",
                      fanout=True, search_task="multi-step")
    req.use_task(SearchTask("multi-step", "preset:multi-step", MULTI))
    deps = with_settings(deps, request_settings(deps.s, req))

    async def go():
        notices.begin()
        return [e async for e in arms.get("deepagents").deep_search(deps, req, [TABLES, REPORT], Conversation(),
                                                                     QUESTION, {})]

    as_user(USER, go)
    assert len(model.of(text)) == 3  # the researcher ran its two searches, then reported
    assert len(deps.retrieval.instructions) == 3 and set(deps.retrieval.instructions) == {MULTI}


def test_exhaustive_mode_accepts_a_task_and_ignores_it():
    req = ChatRequest(question="list every clause", doc_ids=[DOC_A], mode="exhaustive", search_task="exact")
    assert request_problem(Settings(), req) is None  # not a 422, unlike the Deep search settings
    req.use_task(SearchTask("exact", "preset:exact", ""))
    assert api._task_kind(req) is None


# --- the API and the records ---------------------------------------------------------------------------------------


class FakeIngest:
    def __init__(self, docs):
        self.docs = docs

    async def documents(self, doc_ids):
        return [d for d in self.docs if d.doc_id in doc_ids]


@pytest.fixture
def client():
    retrieval = RecordingRetrieval()
    api.state.update(deps=SimpleNamespace(ingest=FakeIngest([doc(DOC_A, A), doc(DOC_B, B)]), planner=None,
                                          retrieval=retrieval, threads=threads_client(ThreadsServer())))
    yield TestClient(api.app), retrieval  # no lifespan: the fakes above stand in for the real clients


def test_search_uses_the_callers_own_task_and_refuses_another_users(client):
    http, retrieval = client
    r = http.post("/v1/search", json={"question": "q", "doc_ids": [DOC_A], "search_task": TASK_A},
                  headers=KEYS.headers(A))
    assert r.status_code == 200 and r.json()["search_task_kind"] == "custom"
    r = http.post("/v1/search", json={"question": "q", "doc_ids": [DOC_A]}, headers=KEYS.headers(A))
    assert r.json()["search_task_kind"] == "preset:general"
    assert [o.instruction for o in retrieval.options] == [CONTRACTS, CURRENT]
    for path in ("/v1/search", "/v1/chat"):  # B cannot use A's task, whatever B knows of its id
        r = http.post(path, json={"question": "q", "doc_ids": [DOC_B], "search_task": TASK_A}, headers=KEYS.headers(B))
        assert r.status_code == 422 and "not one of yours" in r.json()["detail"], path
        r = http.post(path, json={"question": "q", "doc_ids": [DOC_B], "search_task": "nope"}, headers=KEYS.headers(B))
        assert r.status_code == 422 and "unknown search task" in r.json()["detail"], path
        r = http.post(path, json={"question": "q", "doc_ids": [DOC_B], "search_task": "Bad Id"}, headers=KEYS.headers(B))
        assert r.status_code == 422, path
    assert len(retrieval.options) == 2  # nothing refused was searched


def test_threads_down_while_resolving_a_custom_task_is_a_502(client):
    http, _ = client
    api.state["deps"].threads = threads_client(ThreadsServer(status=503))
    r = http.post("/v1/search", json={"question": "q", "doc_ids": [DOC_A], "search_task": TASK_A},
                  headers=KEYS.headers(A))
    assert r.status_code == 502 and r.json()["detail"].startswith("threads service")


def test_read_all_neither_resolves_nor_uses_a_task(client, monkeypatch):
    """A stale id or threads being down cannot fail a Read all question, which embeds nothing."""
    http, _ = client
    api.state["deps"].threads = threads_client(ThreadsServer(status=503))
    api.state["audit"] = FakeAudit()
    asked = []

    async def answer(deps, req, docs, conv):
        asked.append(req)
        yield "sources", {"route": "exhaustive", "sources": []}
        yield "token", {"text": "Every clause."}
        yield "done", {"citations": []}

    monkeypatch.setattr(api, "respond", answer)
    body = {"question": "list every clause", "doc_ids": [DOC_A], "search_task": TASK_A}
    r = http.post("/v1/chat", json=body | {"mode": "exhaustive"}, headers=KEYS.headers(A))
    assert r.status_code == 200 and "Every clause." in r.text
    assert asked[0].task is None and api._task_kind(asked[0]) is None
    assert api.state["audit"].events[-1][1]["search_task_kind"] is None  # recorded as not used
    r = http.post("/v1/chat", json=body | {"mode": "single"}, headers=KEYS.headers(A))
    assert r.status_code == 502 and len(asked) == 1  # a mode that embeds needs its task


def test_the_presets_route_lists_the_built_in_tasks_for_a_signed_in_user(client):
    http, _ = client
    assert http.get("/v1/search-tasks/presets").status_code == 401
    listed = http.get("/v1/search-tasks/presets", headers=KEYS.headers(A)).json()
    assert [p["id"] for p in listed] == ["general", "multi-step", "exact"] and listed[0]["instruction"] == CURRENT


class FakeAudit:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def record(self, action, subject=None, actor="local", **detail):
        self.events.append((action, detail))


class SavingThreads:
    def __init__(self):
        self.appended: list[list] = []

    async def append(self, thread_id, messages, doc_ids):
        self.appended.append(messages)

    async def context(self, thread_id):
        raise AssertionError("not needed")

    async def memories(self):
        return []


class QuietLLM:
    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return {"content": '{"memories": []}'}


@pytest.mark.parametrize("mode, kind", [("single", "custom"), ("deep", "custom"), ("exhaustive", None)])
def test_the_turn_record_and_audit_name_the_kind_and_never_the_custom_text(mode, kind):
    async def scenario():
        audit, threads = FakeAudit(), SavingThreads()
        api.state.update(audit=audit, deps=SimpleNamespace(threads=threads, llm=QuietLLM(), s=api.settings))

        async def events():
            yield "sources", {"route": "search", "sources": []}
            yield "token", {"text": "Clause 4 [1]."}
            yield "done", {"citations": [1]}

        req = ChatRequest(question="which clause?", doc_ids=[DOC_A], thread_id="c" * 32, mode=mode, search_task=TASK_A)
        req.use_task(SearchTask(TASK_A, "custom", CONTRACTS))
        async for _ in api._sse(events(), req, []):
            pass
        await asyncio.gather(*api._background)
        return audit, threads

    audit, threads = as_user(A, scenario)
    meta = threads.appended[0][1].meta
    assert (meta["search_task"], meta["search_task_kind"]) == (TASK_A, kind)  # the id lets the UI ask again
    assert audit.events[0][1]["search_task_kind"] == kind and "search_task" not in audit.events[0][1]
    assert CONTRACTS not in json.dumps(meta) + json.dumps(audit.events, default=str)
