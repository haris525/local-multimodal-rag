"""The chat event protocol (rag_common.chat_events, docs/STATE_MACHINES.md M11 and M12): every branch of a turn,
single-shot and deep, emits events in the order NEXT allows and ends with done; the wire format reads back
the same; every intent kind is handled; the web UI labels every route."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import rag.agent as agent
import rag.api as api
import rag.turn as turn
from rag.chat import ChatRequest
from rag.config import get_settings
from rag.conversation import Conversation
from rag.fusion import Candidate
from rag.intent import KINDS, Intent
from rag.prompts import Source
from rag.retrieval import SearchResult
from rag_common.chat_events import NEXT, ChatEvent, Phase, ProtocolError, check_order, read_chat
from rag_common.contracts import DocStatus, DocType, DocumentInfo, MemoryItem, ThreadContext
from rag_common.index import ChunkPayload

T, DOC = "a" * 32, "d" * 32
NOTE = MemoryItem(memory_id="1" * 32, text="The user works in finance.", source="user", created_at="t")
DOCS = [DocumentInfo(doc_id=DOC, name="report.pdf", doc_type=DocType.PDF, size=1, status=DocStatus.READY,
                     n_chunks=3, created_at="t", updated_at="t")]


class LLM:
    """The router answers with `intent`; the assistant and history calls with `reply`; answers stream."""

    def __init__(self, intent: dict, reply: str = "A reply."):
        self.intent, self.reply = intent, reply

    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return {"content": json.dumps(self.intent) if json_mode else self.reply}

    async def stream(self, messages, max_tokens, think):
        if think:
            yield "reasoning", "Thinking it over."
        yield "token", "South had 340 k$ [1]."
        yield "usage", {"completion_tokens": 7}
        yield "finish", "stop"


class Threads:
    async def remember(self, text, thread_id, source="auto"):
        return MemoryItem(memory_id="2" * 32, text=text, source=source, created_at="t")

    async def replace(self, memory_id, text, source="auto"):
        return None

    async def forget(self, memory_id):
        return None


def candidate() -> Candidate:
    payload = ChunkPayload(doc_id=DOC, doc_name="report.pdf", doc_type="pdf", kind="text", chunk_index=0,
                           text="South had 340 k$.", search_text="South had 340 k$.", text_sha="x", n_tokens=5)
    return Candidate(id="p1", payload=payload, score=0.9)


class Retrieval:
    def __init__(self, found: bool):
        self.found = found
        self.models = SimpleNamespace(embed_documents=self.embed)

    @staticmethod
    async def embed(texts):  # the saved note is about finance; nothing the tests save is a duplicate of it
        return [[1.0, 0.0] if "finance" in t else [0.0, 1.0] for t in texts]

    async def search(self, question, doc_ids, options=None, scope=None):
        return SearchResult(candidates=[candidate()] if self.found else [], terms=["south"], hits={"dense": 1})

    async def neighbours(self, cands):
        return {}


def deps(intent: dict, found: bool = True, reply: str = "A reply."):
    return SimpleNamespace(s=get_settings(), llm=LLM(intent, reply), threads=Threads(), retrieval=Retrieval(found),
                           planner=None, ingest=None, sql=None, graph_llm=None)


def run(d, req: ChatRequest, conv: Conversation) -> list[tuple[str, dict]]:
    async def collect():
        return [e async for e in turn.respond(d, req, DOCS if req.doc_ids else [], conv)]
    return asyncio.run(collect())


def thread_req(question: str, docs: bool = True, mode: str = "single", think: bool = False) -> ChatRequest:
    return ChatRequest(question=question, doc_ids=[DOC] if docs else [], thread_id=T, mode=mode, think=think)


class FakeGraph:
    """Deep research's graph, replayed: a plan, one search tool call that finds a source, a verify round."""

    def __init__(self, book, find: bool):
        self.book, self.find = book, find

    async def astream(self, state, stream_mode, config):
        yield {"plan": {"sub_questions": ["Which region had the highest revenue?"]}}
        yield {"agent": {"messages": [AIMessage(content="", tool_calls=[{"id": "c1", "name": "search",
                                                                        "args": {"query": "revenue by region"}}])]}}
        if self.find:
            self.book.add(Source(0, "p1", DOC, "report.pdf", "text", "p. 1", "South had 340 k$.", 5))
        yield {"tools": {"messages": [ToolMessage(content="[1] report.pdf", tool_call_id="c1", name="search",
                                                  artifact=[1] if self.find else [])]}}
        yield {"verify": {"missing": ["the year"]}}
        yield {"agent": {"messages": [AIMessage(content="DONE")]}}
        yield {"verify": {"missing": []}}


@pytest.fixture
def deep_graph(monkeypatch):
    found = {"find": True}
    monkeypatch.setattr(agent, "build_graph", lambda llm, tools, book, steps, rounds, *rest, **kw: FakeGraph(book, found["find"]))
    return found


# Each branch: (name, intent the router returns, request, expected route, expected event names)
BRANCHES = [
    ("memory_save", {"intent": "memory_save", "save": ["The user prefers short answers."]},
     thread_req("remember I prefer short answers"), "memory_save", ["memory", "sources", "token", "done"]),
    ("memory_recall", {"intent": "memory_recall"}, thread_req("what do you know about me?"), "memory_recall",
     ["sources", "token", "done"]),
    ("memory_forget", {"intent": "memory_forget", "forget": [1], "forget_confirmed": True},
     thread_req("forget that I work in finance"), "memory_forget", ["memory", "sources", "token", "done"]),
    ("assistant", {"intent": "assistant"}, thread_req("what can you do?"), "assistant", ["sources", "token", "done"]),
    ("history", {"intent": "history"}, thread_req("what did I ask earlier?"), "history", ["sources", "token", "done"]),
    ("no documents", {"intent": "documents"}, thread_req("what does the report say?", docs=False), "no_documents",
     ["sources", "token", "done"]),
    ("single shot", {"intent": "documents"}, thread_req("Which region had the highest revenue?", think=True),
     "search", ["sources", "reasoning", "token", "done"]),
    ("mixed message", {"intent": "memory_save", "save": ["The user is in sales."], "question": "Revenue by region?"},
     thread_req("I am in sales; which region had the highest revenue?"), "search",
     ["memory", "token", "sources", "token", "done"]),
    ("deep", {"intent": "documents"}, thread_req("Which region had the highest revenue?", mode="deep"), "deep",
     ["step", "step", "step", "sources", "token", "done"]),
    ("deep, mixed message", {"intent": "memory_save", "save": ["The user is in sales."], "question": "Revenue?"},
     thread_req("I am in sales; revenue by region?", mode="deep"), "deep",
     ["memory", "token", "step", "step", "step", "sources", "token", "done"]),
]


@pytest.mark.parametrize("name, intent, req, route, expected", BRANCHES, ids=[b[0] for b in BRANCHES])
def test_every_branch_of_a_turn_follows_the_order(deep_graph, name, intent, req, route, expected):
    reply = "You asked about revenue." if name == "history" else "A reply."
    events = run(deps(intent, reply=reply), req, Conversation(T, memories=[NOTE]))
    names = [n for n, _ in events]
    assert names == expected
    assert all(isinstance(n, ChatEvent) for n in names)
    assert check_order(names) == Phase.DONE
    assert dict(events)[ChatEvent.SOURCES]["route"] == route


@pytest.mark.parametrize("mode", ["single", "deep"])
def test_nothing_found_still_ends_in_order(deep_graph, mode):
    deep_graph["find"] = False
    events = run(deps({"intent": "documents"}, found=False), thread_req("anything?", mode=mode), Conversation(T))
    names = [n for n, _ in events]
    assert check_order(names) == Phase.DONE and names[-3:] == ["sources", "token", "done"]
    assert dict(events)[ChatEvent.DONE]["finish_reason"] == "no_sources"


def test_history_without_the_answer_searches_the_documents(deep_graph):
    events = run(deps({"intent": "history"}, reply="NEED_SEARCH"), thread_req("what did the report say?"),
                 Conversation(T))
    assert check_order(n for n, _ in events) == Phase.DONE and dict(events)[ChatEvent.SOURCES]["route"] == "search"


@pytest.mark.parametrize("mode", ["single", "deep"])
def test_a_stateless_call_follows_the_order(deep_graph, mode):
    req = ChatRequest(question="Which region had the highest revenue?", doc_ids=[DOC], mode=mode)
    events = run(deps({"intent": "memory_save"}), req, Conversation())
    assert check_order(n for n, _ in events) == Phase.DONE and ChatEvent.MEMORY not in dict(events)


def test_every_intent_kind_is_handled_and_a_new_one_fails_loudly(deep_graph, monkeypatch):
    covered = {b[1]["intent"] for b in BRANCHES}
    assert covered == set(KINDS)  # the branches above reach every kind the router may return

    async def unknown_route(llm, message, conv, documents=0, max_tokens=0):
        return Intent("summarize_everything")  # a kind added to Kind but not to respond

    monkeypatch.setattr(turn, "route", unknown_route)
    with pytest.raises(AssertionError):
        run(deps({}), thread_req("anything"), Conversation(T))


# --- the wire: rag's SSE text, read back with the shared parser ------------------------------


class Audit:
    async def record(self, *args, **kwargs):
        pass


class SavingThreads:
    async def append(self, thread_id, messages, doc_ids):
        pass

    async def context(self, thread_id):
        return ThreadContext(summary="", summary_upto=0, messages=[])

    async def memories(self):
        return []


def wire(events) -> list[str]:
    async def produce():
        api.state.update(audit=Audit(), deps=SimpleNamespace(threads=SavingThreads(), llm=LLM({}), s=api.settings))

        async def source():
            for e in events:
                if isinstance(e, Exception):
                    raise e
                yield e

        text = "".join([chunk async for chunk in api._sse(source(), ChatRequest(question="q", doc_ids=[DOC]), [])])
        await asyncio.gather(*api._background)
        return text.split("\n")
    return asyncio.run(produce())


def test_the_wire_reads_back_complete(deep_graph):
    events = run(deps({"intent": "documents"}), thread_req("Revenue?", think=True), Conversation(T))
    reply = read_chat(wire(events))
    assert reply.complete and reply.failure is None and reply.route == "search"
    assert reply.answer == "South had 340 k$ [1]." and reply.citations == [1] and len(reply.sources) == 1


def test_a_failure_after_the_start_is_one_error_event():
    reply = read_chat(wire([(ChatEvent.SOURCES, {"route": "search", "sources": []}), RuntimeError("llm went away")]))
    assert [n for n, _ in reply.events] == ["sources", "error"] and reply.terminal == "error"
    assert reply.failure == "RuntimeError: llm went away" and not reply.complete


def test_an_event_the_protocol_does_not_know_becomes_an_error_event():
    reply = read_chat(wire([("summary", {"text": "x"})]))
    assert reply.terminal == "error" and "summary" in reply.error


# --- the order table itself -----------------------------------------------------------------


@pytest.mark.parametrize("docs, mode", [(True, "single"), (False, "single"), (True, "deep")])
def test_a_credential_to_remember_gets_the_fixed_refusal_even_when_the_router_says_assistant(deep_graph, docs, mode):
    """The router answered "assistant" and the model wrote its own refusal; the turn must take the
    memory path and its fixed text, whatever the router returns, with or without documents and in deep mode."""
    d = deps({"intent": "assistant"}, reply="I can't store passwords or other sensitive credentials.")
    events = run(d, thread_req("remember my password is hunter2", docs=docs, mode=mode), Conversation(T))
    names = [name for name, _ in events]
    text = "".join(data.get("text", "") for name, data in events if name == "token")
    sources = next(data for name, data in events if name == "sources")
    assert sources["route"] == "memory_save" and "search" not in names[:names.index("sources")]
    assert "I don't store passwords, keys, tokens or other credentials, so I did not save that." in text
    assert "hunter2" not in text and "sensitive credentials" not in text


def test_the_order_table():
    assert check_order(["memory", "memory", "token", "step", "sources", "reasoning", "token", "done"]) == Phase.DONE
    assert check_order(["sources", "error"]) == Phase.ERRORED and check_order(["error"]) == Phase.ERRORED
    for bad in (["token", "token"], ["sources", "sources"], ["sources", "step"], ["done"], ["sources", "done", "token"],
                ["sources", "memory"], ["step", "memory"], ["sources", "error", "done"], ["summary"]):
        with pytest.raises(ProtocolError):
            check_order(bad)
    assert all(ChatEvent.ERROR in NEXT[p] for p in Phase if p not in (Phase.DONE, Phase.ERRORED))


def test_a_cut_stream_is_reported():
    reply = read_chat(['event: sources', 'data: {"route": "search", "sources": []}', '', 'event: token',
                       'data: {"text": "Half an ans"}', ''])
    assert reply.terminal is None and reply.answer == "Half an ans"
    assert reply.failure == "the answer stream ended before its done event"


# --- the web UI labels every route -----------------------------------------------------------


def test_the_ui_has_a_label_for_every_route():
    app_js = Path(__file__).resolve().parents[2] / "gateway" / "static" / "app.js"
    block = re.search(r"const ROUTE_LABEL = \{(.*?)\};", app_js.read_text(encoding="utf-8"), re.S).group(1)
    labelled = set(re.findall(r"(\w+):", block))
    assert set(turn.ROUTES) == labelled
    no_search = re.search(r"const NO_SEARCH_ROUTES = new Set\(\[(.*?)\]\)", app_js.read_text(encoding="utf-8")).group(1)
    assert set(re.findall(r'"(\w+)"', no_search)) == set(turn.DIRECT_ROUTES)
