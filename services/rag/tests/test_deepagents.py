"""Deep research on the deepagents harness (rag.agent_deepagents) and on the harness out of the box
(rag.agent_deepagents_stock) next to the LangGraph one (rag.agent), with scripted models: the same tools with the
same scoping and errors, the same source numbers, sources event and answer prompt; a large SQL result kept in the
request's virtual files and read back with read_file; todos; budgets and a full slot ending the research with a
notice; sub-agents sharing the book and the budget; summarization; thinking for the plan and verify steps only;
no disk writes and no network; the prompts in real tokens; the stock harness's defaults kept with our guards
around them; the switch and a 422 for a wrong value."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, ValidationError

from conftest import KEYS
from rag_common.chat_events import STEP_TOOLS, check_order
from rag_common.contracts import ColumnSchema, DocStatus, DocType, DocumentInfo, SqlResult, TableSchema
from rag_common.identity import Identity, current_identity
from rag_common.index import ChunkPayload
from rag_common.testing import new_user_id

import rag.agent as agent
import rag.agent_deepagents as da
import rag.agent_deepagents_stock as stock
import rag.api as api
import rag.turn as turn
from rag import arms, notices, tokens
from rag.chat import ChatRequest
from rag.config import Settings
from rag.conversation import Conversation
from rag.fusion import Candidate

USER = new_user_id()
TABLES = DocumentInfo(doc_id="d" * 32, name="orders.xlsx", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
                      created_at="t", updated_at="t",
                      tables=[TableSchema(table="orders", source='sheet "Orders"', sheet="Orders", rows=80, row_filter=True,
                                          columns=[ColumnSchema(name="Region", sql_name="region", type="text", profile="4"),
                                                   ColumnSchema(name="Revenue", sql_name="revenue", type="number",
                                                                profile="0-9999")])])
REPORT = DocumentInfo(doc_id="e" * 32, name="report.pdf", doc_type=DocType.PDF, size=1, status=DocStatus.READY,
                      n_chunks=3, created_at="t", updated_at="t")
OTHER = "f" * 32  # a document of the user that is not selected
QUESTION = "Which region had the highest Q3 revenue?"
_HF = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots"
TOKENIZER = os.environ.get("RAG_TEST_TOKENIZER") or next((str(p) for p in _HF.glob("*/tokenizer.json")), "")


@pytest.fixture(autouse=True)
def estimate_mode():
    tokens.configure(None)
    yield
    tokens.configure(None)


# --- fakes ------------------------------------------------------------------------------------------------------


class Scripted(BaseChatModel):
    """Plays a script, one reply per research step (then DONE); answers rag.agent's plan and verify prompts and
    summary requests itself. Records every call's messages and settings."""

    steps: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)
    plan: str = '{"sub_questions": ["Q3 revenue per region"]}'

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tool_names=[t.name if hasattr(t, "name") else t["name"] for t in tools], **kwargs)

    def _generate(self, *args, **kwargs):
        raise NotImplementedError("research runs async")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append((list(messages), kwargs))
        first = str(messages[0].content)
        if first.startswith("Break this question"):
            reply = AIMessage(self.plan, response_metadata={"finish_reason": "stop"})
        elif "NOT yet answered" in first:
            reply = AIMessage('{"missing": []}', response_metadata={"finish_reason": "stop"})
        elif first.startswith("Summarize this research so far"):
            reply = AIMessage("Question: Q3 revenue. South 2735.62 [2].", response_metadata={"finish_reason": "stop"})
        else:
            reply = self.steps.pop(0) if self.steps else done()
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def research_calls(self) -> list:
        """The research steps' calls (not plan, verify or summary)."""
        return [(m, k) for m, k in self.calls if not str(m[0].content).startswith(("Break this", "Question:", "Summarize"))
                and "NOT yet answered" not in str(m[0].content)]


def call(name: str, args: dict, cid: str | None = None) -> AIMessage:
    return calls((name, args, cid))


_IDS = iter(range(1, 1_000_000))


def calls(*items) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": n, "args": a, "id": cid or f"c{next(_IDS)}-{n}"} for n, a, cid in items],
                     response_metadata={"finish_reason": "tool_calls"})


def done(text: str = "DONE") -> AIMessage:
    return AIMessage(text, response_metadata={"finish_reason": "stop"})


def todos(*items: tuple[str, str]) -> AIMessage:
    return call("write_todos", {"todos": [{"content": c, "status": s} for c, s in items]})


class Sql:
    """Records each query with the identity it ran as. `wide` rows make a result near the SQL budget."""

    def __init__(self, wide: bool = False):
        self.calls: list[tuple[str, str, str | None]] = []
        self.wide = wide

    async def query(self, doc_id, sql, max_rows=200):
        ident = current_identity.get()
        self.calls.append((doc_id, sql, ident.user_id if ident else None))
        if self.wide:
            rows = [[1949 + r, *(f"{(r + 1) * (c + 7) * 1234.567:.3f}" for c in range(6))] for r in range(80)]
            return SqlResult(columns=["year", *(f"source_{c}" for c in range(6))], rows=rows, truncated=False)
        return SqlResult(columns=["region", "q3_revenue"], rows=[["South", "2735.62"], ["North", "2101.10"]],
                         truncated=False)


class Retrieval:
    def __init__(self):
        self.calls: list[tuple[str, list[str], str | None]] = []

    async def search(self, query, doc_ids, options=None, scope=None):
        ident = current_identity.get()
        self.calls.append((query, list(doc_ids), ident.user_id if ident else None))
        if REPORT.doc_id not in doc_ids:
            return SimpleNamespace(candidates=[])
        text = "Third-quarter revenue by region: the South region led with 340 (thousand dollars), ahead of the North."
        payload = ChunkPayload(doc_id=REPORT.doc_id, doc_name=REPORT.name, user_id=USER, kind="text", text=text,
                               chunk_index=4, n_tokens=30, page=3, doc_type="pdf", search_text=text, text_sha="x")
        return SimpleNamespace(candidates=[Candidate(id="r4", payload=payload, score=0.9, ranks={})])

    async def fetch(self, ids):
        return []


class Ingest:
    def __init__(self):
        self.calls: list[tuple[list[str], list[str], str | None]] = []

    async def grep(self, doc_ids, terms, limit):
        ident = current_identity.get()
        self.calls.append((list(doc_ids), list(terms), ident.user_id if ident else None))
        return []


class AnswerLLM:
    """The answer step: records the prompt it was given."""

    def __init__(self):
        self.messages = None

    async def stream(self, messages, max_tokens, think):
        self.messages = messages
        yield "token", "South led with 2735.62 [2], while the report says 340 [1]."
        yield "finish", "stop"


def make_deps(model: Scripted, s: Settings | None = None, sql: Sql | None = None):
    return SimpleNamespace(s=s or Settings(), graph_llm=model, sql=sql or Sql(), retrieval=Retrieval(), ingest=Ingest(),
                           llm=AnswerLLM())


def research(deps, implementation: str, docs=(TABLES, REPORT), think: bool | None = None, question: str = QUESTION):
    """One deep research as the user USER: (events, notices)."""
    req = ChatRequest(question=question, doc_ids=[d.doc_id for d in docs], mode="deep", agent=implementation,
                      deep_think=think)
    module = arms.get(implementation)

    async def go():
        queue = notices.begin()
        token = current_identity.set(Identity(USER, "user", "auth"))
        try:
            events = [e async for e in module.deep_search(deps, req, list(docs), Conversation(), question, {})]
        finally:
            current_identity.reset(token)
        return events, queue
    return asyncio.run(go())


def steps(events, names=None) -> list[dict]:
    return [d for n, d in events if n == "step" and (names is None or d["tool"] in names)]


def event(events, name) -> dict:
    return next(d for n, d in events if n == name)


def tool_messages(model: Scripted) -> list[ToolMessage]:
    """Every tool result the model saw, by its call id (the last time it saw it)."""
    seen = {}
    for messages, _ in model.calls:
        for m in messages:
            if isinstance(m, ToolMessage):
                seen[m.tool_call_id] = m
    return list(seen.values())


RESEARCH_TOOLS = {"search", "grep", "read", "list_tables", "describe_table", "sql", "peek", "profile_column"}


def same_calls() -> list[AIMessage]:
    """The same research, one call a step: a search, a grep, SQL on a document that is not selected, real SQL."""
    return [call("search", {"query": "Q3 revenue by region"}), call("grep", {"terms": ["South"]}),
            call("sql", {"doc_id": OTHER, "query": "SELECT 1"}),
            call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT region, SUM(revenue) AS q3_revenue FROM orders GROUP BY 1"}),
            done()]


# --- the same tools, numbers and answer -------------------------------------------------------------------------


def test_both_implementations_scope_the_tools_alike_and_number_the_sources_and_build_the_answer_the_same():
    runs = {}
    for name in ("langgraph", "deepagents"):
        model = Scripted(steps=[*([todos(("Q3 revenue per region", "in_progress"))] if name == "deepagents" else []),
                                *same_calls(), done()])
        deps = make_deps(model)
        events, queue = research(deps, name)
        runs[name] = (events, deps, model)
    (lg, lg_deps, _), (da_events, da_deps, da_model) = runs["langgraph"], runs["deepagents"]
    # the same calls reached the same services, for the same user, with the same selection
    assert lg_deps.retrieval.calls == da_deps.retrieval.calls == [("Q3 revenue by region", [TABLES.doc_id, REPORT.doc_id], USER)]
    assert lg_deps.ingest.calls == da_deps.ingest.calls == [([TABLES.doc_id, REPORT.doc_id], ["South"], USER)]
    assert lg_deps.sql.calls == da_deps.sql.calls and [c[0] for c in da_deps.sql.calls] == [TABLES.doc_id]  # OTHER refused
    # the same steps, errors included (as ToolNode(handle_tool_errors=True) words them)
    assert steps(lg, RESEARCH_TOOLS) == steps(da_events, RESEARCH_TOOLS)
    refused = steps(da_events, {"sql"})[0]
    assert refused["error"].startswith("Error: ValueError('doc_id is not one of the selected documents")
    # the same numbered sources, the same answer prompt and the same citations
    assert event(lg, "sources") == event(da_events, "sources")
    assert [s["n"] for s in event(da_events, "sources")["sources"]] == [1, 2]
    assert lg_deps.llm.messages == da_deps.llm.messages
    lg_done, da_done = event(lg, "done"), event(da_events, "done")
    assert lg_done["citations"] == da_done["citations"] == [1, 2] and lg_done["invalid_citations"] == []
    assert set(lg_done["timings"]) <= set(da_done["timings"])  # the harness adds its own counters
    assert {"write_todos", "read_file"} <= set(da_model.calls[0][1]["tool_names"])
    assert RESEARCH_TOOLS <= set(da_model.calls[0][1]["tool_names"])


@pytest.mark.parametrize("arm", ["langgraph", "deepagents", "deepagents_stock"])
def test_every_core_arm_receives_the_rule_on_the_forms_a_name_takes_in_a_column(arm):
    model = Scripted(steps=[done()])
    research(make_deps(model), arm)
    messages, _ = model.research_calls()[0]
    prompt = " ".join(da.flat(messages[0]).content.split())
    for part in ("- Names (places, countries, organizations, products): before filtering, grouping or counting rows "
                 "by one, see how the column writes it (its distinct values or its profile)",
                 "abbreviations, codes, short or official names instead of the common one",
                 "match every form that means the same",
                 "merely contain the text: check what it caught before counting"):
        assert part in prompt


def test_the_research_rules_are_the_langgraph_agents_word_for_word():
    s = Settings()
    prompt = da.system_prompt(s, QUESTION, [TABLES, REPORT])
    assert agent._SYSTEM.format(sql_rows=s.sql_max_rows) in prompt
    for rule in ("FLOOR(year / 10) * 10", "MAX - MIN + 1", "ROUND(100.0 * part / whole, 2)", "profile_column",
                 "search the text documents too, even after SQL answered", "the answer must then report both"):
        assert rule in prompt
    assert "- orders.xlsx (spreadsheet; table orders" in prompt and "- report.pdf (pdf)" in prompt
    assert "at least" in da.TODO_RULES and "text documents (PDF, Word, text, images) and tables are both" in da.TODO_RULES


# --- the virtual files ------------------------------------------------------------------------------------------


def test_a_large_sql_result_is_kept_in_the_virtual_files_and_read_back_while_the_answer_gets_it_whole():
    path = "/large_tool_results/c-sql"
    model = Scripted(steps=[todos(("decades", "in_progress")),
                            call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT * FROM orders"}, "c-sql"),
                            call("read_file", {"file_path": path, "offset": 10, "limit": 5}, "c-read"),
                            todos(("decades", "completed")), done(), done()])
    deps = make_deps(model, sql=Sql(wide=True))
    events, _ = research(deps, "deepagents", docs=(TABLES,))
    whole = " ".join(p["text"] for p in deps.llm.messages[1]["content"] if p.get("type") == "text")  # the answer prompt
    seen = {m.tool_call_id: m for m in tool_messages(model)}
    kept = seen["c-sql"].content
    assert kept.startswith(f"Source(s) [1]: this result is ") and f"so it was kept in {path}" in kept
    assert tokens.count_tokens(kept) <= Settings().deep_preview_tokens + 120 < tokens.count_tokens(whole)
    assert seen["c-sql"].artifact == [1]
    read = seen["c-read"].content  # lines 11 to 15 of the kept result, under read_file's line-range header
    assert read.startswith("@@ lines 11-15 of ") and seen["c-read"].status != "error"
    assert "| 1954 |" in read and "| 1958 |" in read and "| 1959 |" not in read  # lines 1-5: title, query, count, header
    assert "@@ lines 1-" in kept  # the preview names its line ranges the same way
    assert "| 1955 |" in whole and "| 1975 |" in whole  # rows the preview did not show reach the answer
    sql_step = steps(events, {"sql"})[0]
    assert sql_step["sources"] == [1] and sql_step["error"] is None
    assert steps(events, {"read_file"})[0]["args"] == {"file_path": path, "offset": 10, "limit": 5}


def test_a_small_result_stays_inline():
    model = Scripted(steps=[call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}, "c-sql"), done(), done()])
    deps = make_deps(model)
    research(deps, "deepagents", docs=(TABLES,))
    seen = {m.tool_call_id: m for m in tool_messages(model)}
    assert seen["c-sql"].content.startswith("[1] orders.xlsx | SQL result") and "South | 2735.62" in seen["c-sql"].content


# --- planning and verify ----------------------------------------------------------------------------------------


def test_todos_are_streamed_and_verify_lists_the_open_ones_once_then_stops_when_nothing_new_was_found():
    model = Scripted(steps=[todos(("South revenue", "in_progress"), ("North revenue", "pending")),
                            call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}), done(), done(), done()])
    deps = make_deps(model)
    events, queue = research(deps, "deepagents", docs=(TABLES,))
    plan = steps(events, {"write_todos"})
    assert plan == [{"tool": "write_todos", "args": {"todos": ["South revenue (in_progress)", "North revenue (pending)"]},
                     "sources": None, "error": None}]
    verify = steps(events, {"verify"})
    assert len(verify) == 1 and verify[0]["args"]["missing"] == ["South revenue", "North revenue"]
    nudge = model.research_calls()[3][0][-1]  # plan, SQL, DONE, then the step answering verify
    assert isinstance(nudge, HumanMessage) and "These todos are still open:\n- South revenue\n- North revenue" in nudge.content
    # the second DONE found nothing new: no second round; the open todos become a notice
    assert len(model.research_calls()) == 4
    assert [n["code"] for n in queue] == ["research_incomplete"] and "North revenue" in queue[0]["message"]


def test_a_mixed_selection_searches_the_text_documents_when_only_sql_was_used():
    model = Scripted(steps=[call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}), done(), done()])
    deps = make_deps(model)
    events, _ = research(deps, "deepagents")
    assert (QUESTION, [REPORT.doc_id], USER) in deps.retrieval.calls  # the report, searched on its own
    verify = steps(events, {"verify"})[0]
    assert verify["args"]["missing"][0] == "what the text documents say (report.pdf)"
    nudge = model.research_calls()[2][0][-1].content
    assert "The text documents (report.pdf) had not been searched yet" in nudge and "both must be reported" in nudge
    assert {s["doc_name"] for s in event(events, "sources")["sources"]} == {"orders.xlsx", "report.pdf"}


def test_a_pdf_with_a_table_counts_as_a_text_document_in_the_check_of_a_mixed_selection():
    """The rule of rag.agent.text_documents in the deepagents check: a PDF that has a table is text."""
    pdf = REPORT.model_copy(update={"tables": [TableSchema(table="p3_table1", source="page 3, table 1", rows=4,
                                                           row_filter=False, columns=[])]})
    model = Scripted(steps=[call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}), done(), done()])
    deps = make_deps(model)
    events, _ = research(deps, "deepagents", docs=(TABLES, pdf))
    assert (QUESTION, [REPORT.doc_id], USER) in deps.retrieval.calls
    assert steps(events, {"verify"})[0]["args"]["missing"][0] == "what the text documents say (report.pdf)"
    assert agent.text_documents([TABLES, pdf]) == {REPORT.doc_id: "report.pdf"}


@pytest.mark.parametrize("implementation", ["deepagents"])
def test_malformed_todo_arguments_are_refused_by_the_tool_and_shown_without_failing_the_run(implementation):
    """The model's arguments are untyped (rag.agent types its JSON fields): a string, a list of strings or
    dicts without content never break the step events or the open-todo notice."""
    model = Scripted(steps=[call("write_todos", {"todos": "find the South figure"}),
                            call("write_todos", {"todos": ["South figure", {"status": "pending"}, 7]}),
                            todos(("South figure", "in_progress")),
                            call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}), done(), done()])
    deps = make_deps(model)
    events, queue = research(deps, implementation, docs=(TABLES,))
    plans = steps(events, {"write_todos"})
    assert plans[0]["args"] == {"todos": ["find the South figure (pending)"]} and plans[0]["error"]
    assert plans[1]["args"] == {"todos": ["South figure (pending)", "7 (pending)"]} and plans[1]["error"]
    assert plans[2]["error"] is None and event(events, "done")["finish_reason"] == "stop"
    assert [n["code"] for n in queue][-1:] == ["research_incomplete"] and "South figure" in queue[-1]["message"]
    assert da.open_todos("x") == ["x"] and da.open_todos(None) == [] and da.open_todos(42) == []


# --- budgets ----------------------------------------------------------------------------------------------------


def test_the_tool_budget_stops_the_research_with_a_notice_and_refuses_calls_beyond_it():
    s = Settings(agent_max_steps=2)
    model = Scripted(steps=[calls(*[("sql", {"doc_id": TABLES.doc_id, "query": f"SELECT {i}"}, f"q{i}") for i in range(3)])])
    deps = make_deps(model, s)
    events, queue = research(deps, "deepagents", docs=(TABLES,))
    sql = sorted(steps(events, {"sql"}), key=lambda x: x["args"]["query"])  # calls of one reply run at once
    assert [x["error"] is None for x in sql] == [True, True, False] and "budget is used up" in sql[2]["error"]
    assert len(deps.sql.calls) == 2 and len(model.research_calls()) == 1  # no model call after the budget
    assert [n["code"] for n in queue] == ["research_budget"] and "RAG_AGENT_MAX_STEPS" in queue[0]["message"]
    assert event(events, "done")["timings"]["tool_calls"] == 2 and event(events, "sources")["sources"]


def test_the_harness_budget_stops_endless_planning_with_a_notice():
    s = Settings(deep_harness_calls=2)
    model = Scripted(steps=[todos(("a", "pending")), todos(("a", "in_progress")), todos(("a", "completed"))])
    deps = make_deps(model, s)
    events, queue = research(deps, "deepagents", docs=(TABLES,))
    assert "research_harness_budget" in [n["code"] for n in queue]
    assert len(model.research_calls()) == 3 and steps(events, {"write_todos"})[-1]["error"]


def test_a_research_prompt_that_outgrows_the_slot_stops_with_a_notice_before_the_model_is_called():
    model = Scripted(steps=[call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT *"}),
                            call("sql", {"doc_id": TABLES.doc_id, "query": "x"})])
    # a 9,000-token slot: the first step fits; with the large result (kept inline here) the next one does not
    deps = make_deps(model, Settings(llm_ctx=2 * 9000, deep_evict_tokens=100000), Sql(wide=True))
    events, queue = research(deps, "deepagents", docs=(TABLES,))
    assert len(model.research_calls()) == 1 and len(deps.sql.calls) == 1
    assert [n["code"] for n in queue][:1] == ["research_context_full"]
    assert event(events, "sources")["sources"][0]["n"] == 1  # the answer still uses what was found


def test_a_failed_step_is_retried_once_and_a_second_failure_is_a_notice():
    import httpx
    import openai

    class Busy(Scripted):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls.append((list(messages), kwargs))
            raise openai.APIStatusError("busy", response=httpx.Response(503, request=httpx.Request("POST", "http://llm")),
                                        body=None)

    model = Busy()
    deps = make_deps(model, Settings(retry_backoff_s=0))
    events, queue = research(deps, "deepagents", docs=(TABLES,))
    assert len(model.calls) == 2 and [n["code"] for n in queue] == ["research_step_failed"]
    assert event(events, "done")["finish_reason"] == "no_sources"


# --- sub-agents and summarization -------------------------------------------------------------------------------


def test_a_sub_agent_researches_with_the_same_tools_book_and_budget_and_reports_with_source_numbers():
    s = Settings(deep_subagents=True)
    model = Scripted(steps=[
        todos(("Q3 revenue per region", "in_progress")),
        call("task", {"description": "Q3 revenue per region from orders.xlsx; return figures with [n]",
                      "subagent_type": "researcher"}, "c-task"),
        call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT region, SUM(revenue) FROM orders GROUP BY 1"}),  # sub-agent
        done("South 2735.62 [1], North 2101.10 [1]."),  # the sub-agent's report
        todos(("Q3 revenue per region", "completed")), done(), done()])
    deps = make_deps(model, s)
    events, _ = research(deps, "deepagents")
    sub_messages, _ = model.research_calls()[2]
    assert "You research one sub-question of the lead researcher's question" in sub_messages[0].content
    assert QUESTION in sub_messages[0].content and "- orders.xlsx (spreadsheet" in sub_messages[0].content
    assert sub_messages[1].content.startswith("Q3 revenue per region from orders.xlsx")
    assert "task" not in model.research_calls()[2][1]["tool_names"] and "sql" in model.research_calls()[2][1]["tool_names"]
    report = {m.tool_call_id: m for m in tool_messages(model)}["c-task"]
    assert report.content == "South 2735.62 [1], North 2101.10 [1]."
    names = [x["tool"] for x in steps(events)]
    assert names.index("sql") < names.index("task")  # the sub-agent's steps stream as they happen
    timings = event(events, "done")["timings"]
    assert timings["tool_calls"] == 1 and timings["harness_calls"] == 3  # its SQL counts in the shared budget
    found = [(x["n"], x["kind"]) for x in event(events, "sources")["sources"]]
    assert found == [(1, "sql"), (2, "text")]  # the sub-agent's SQL, then verify's search of the report: one book


def test_a_long_research_is_summarized_into_the_virtual_files_and_goes_on():
    s = Settings(deep_summary_tokens=1200, deep_summary_keep_tokens=300, deep_evict_tokens=100000)
    model = Scripted(steps=[call("sql", {"doc_id": TABLES.doc_id, "query": f"SELECT {i}"}) for i in range(4)]
                     + [call("read_file", {"file_path": "/conversation_history/" + "x", "limit": 3}), done(), done()])
    deps = make_deps(model, s, Sql(wide=True))
    events, queue = research(deps, "deepagents", docs=(TABLES,))
    summarized = steps(events, {"summarize"})
    assert summarized and summarized[0]["args"]["kept_in"].startswith("/conversation_history/")
    later = model.research_calls()[-1][0]
    assert any(isinstance(m, HumanMessage) and "has been summarized" in m.content and "South 2735.62 [2]" in m.content
               for m in later)
    assert any(str(m[0].content).startswith("Summarize this research so far") for m, _ in model.calls)
    assert [x["n"] for x in event(events, "sources")["sources"]] == [1, 2, 3, 4]  # the book keeps every result


# --- thinking -------------------------------------------------------------------------------------------------


def thinks(kwargs: dict) -> bool:
    return (kwargs.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking") is True


def test_deep_think_makes_the_plan_and_verify_steps_of_deepagents_think_and_no_other():
    s = Settings()
    model = Scripted(steps=[todos(("a", "in_progress")), call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}),
                            done(), done()])
    research(make_deps(model, s), "deepagents", docs=(TABLES,), think=True)
    flags = [thinks(k) for _, k in model.research_calls()]
    assert flags == [True, False, False, True]  # plan, sql step, DONE, the step answering verify
    rooms = [k["max_tokens"] for _, k in model.research_calls()]
    assert rooms == [s.deep_think_tokens, s.agent_tokens, s.agent_tokens, s.deep_think_tokens]
    assert model.research_calls()[0][1]["temperature"] == s.think_temperature
    model = Scripted(steps=[todos(("a", "in_progress")), done(), done()])
    research(make_deps(model, s), "deepagents", docs=(TABLES,))
    assert not any(thinks(k) for _, k in model.calls)


# --- no disk, no network ----------------------------------------------------------------------------------------

_WATCH: list | None = None
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _audit(name: str, args: tuple) -> None:
    if _WATCH is None:
        return
    if name == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & _WRITE_FLAGS):
            _WATCH.append((name, str(path), mode))
    elif name in ("os.mkdir", "os.rename", "os.replace", "os.remove", "os.truncate", "shutil.copyfile", "shutil.rmtree",
                  "socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto", "http.client.connect"):
        _WATCH.append((name, repr(args)[:200]))


sys.addaudithook(_audit)


def test_a_run_with_eviction_summarization_and_sub_agents_writes_nothing_to_disk_and_opens_no_connection(tmp_path):
    global _WATCH

    def script():
        return [todos(("a", "in_progress")),
                call("task", {"description": "a", "subagent_type": "researcher"}, "c-task"),
                call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT * FROM orders"}, "c-sql"),
                call("read_file", {"file_path": "/large_tool_results/c-sql", "limit": 5}), done("found [1]"),
                call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 2"}), done(), done()]

    s = Settings(deep_subagents=True, deep_summary_tokens=2500, deep_summary_keep_tokens=300)
    research(make_deps(Scripted(steps=script()), s, Sql(wide=True)), "deepagents", docs=(TABLES,))  # warm: lazy imports
    model = Scripted(steps=script())
    deps = make_deps(model, s, Sql(wide=True))
    req = ChatRequest(question=QUESTION, doc_ids=[TABLES.doc_id], mode="deep", agent="deepagents")

    async def go():
        global _WATCH
        notices.begin()
        _WATCH = []  # inside the running loop: its own socket pair is made before
        try:
            return [e async for e in da.deep_search(deps, req, [TABLES], Conversation(), QUESTION, {})]
        finally:
            watched, _WATCH = _WATCH, None
            go.watched = watched

    events = asyncio.run(go())
    assert {x["tool"] for x in steps(events)} >= {"task", "sql", "read_file", "summarize"}
    assert go.watched == []
    _WATCH = []  # the watcher itself: a file written and a host looked up are both caught
    try:
        (tmp_path / "control.txt").write_text("x")
        try:
            import socket
            socket.getaddrinfo("localhost", 80)
        except OSError:
            pass
        control = list(_WATCH)
    finally:
        _WATCH = None
    assert any(x[0] == "open" and x[1].endswith("control.txt") for x in control)
    assert any(x[0] == "socket.getaddrinfo" for x in control)


# --- the switch ---------------------------------------------------------------------------------------------------


def test_the_request_overrides_the_setting_and_a_wrong_value_is_refused(monkeypatch):
    ran = []

    def fake(name):
        async def deep_search(deps, req, docs, conv, question, timings):
            ran.append(name)
            for e in agent.nothing_found(timings):
                yield e
        return deep_search

    monkeypatch.setattr(agent, "deep_search", fake("langgraph"))
    monkeypatch.setattr(da, "deep_search", fake("deepagents"))
    monkeypatch.setattr(stock, "deep_search", fake("deepagents_stock"))

    def respond(s, **fields):
        deps = SimpleNamespace(s=s, llm=None, threads=None, retrieval=None)
        req = ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep", **fields)

        async def go():
            return [e async for e in turn.respond(deps, req, [TABLES], Conversation())]
        monkeypatch.setattr(turn, "standalone_question", lambda deps, q, conv: asyncio.sleep(0, q))
        return asyncio.run(go())

    respond(Settings())
    respond(Settings(deep_agent="deepagents"))
    respond(Settings(deep_agent="deepagents"), agent="langgraph")
    respond(Settings(), agent="deepagents")
    respond(Settings(deep_agent="deepagents_stock"))
    respond(Settings(deep_agent="deepagents_stock"), agent="deepagents")
    respond(Settings(), agent="deepagents_stock")
    assert ran == ["langgraph", "deepagents", "langgraph", "deepagents", "deepagents_stock", "deepagents",
                   "deepagents_stock"]
    with pytest.raises(ValidationError, match="unknown deep-search agent 'crewai'; available: langgraph, deepagents, "
                                              "deepagents_stock"):
        Settings(deep_agent="crewai")
    with pytest.raises(ValidationError, match="available: langgraph, deepagents, deepagents_stock"):
        ChatRequest(question="q", doc_ids=[TABLES.doc_id], agent="crewai")
    monkeypatch.setenv("RAG_DEEP_AGENT", "deepagents")
    monkeypatch.setenv("RAG_DEEP_THINK", "true")
    from_env = Settings()
    assert from_env.deep_agent == "deepagents" and from_env.deep_think is True


def test_the_api_answers_422_for_a_wrong_agent_or_deep_think_and_names_the_registered_arms():
    client = TestClient(api.app)
    for body in ({"agent": "crewai"}, {"agent": "LangGraph"}, {"agent": "deepagents-stock"}, {"deep_think": "sometimes"}):
        r = client.post("/v1/chat", json={"question": "q", "doc_ids": [TABLES.doc_id], "mode": "deep", **body},
                        headers=KEYS.headers(USER))
        assert r.status_code == 422, body
        if "agent" in body:
            assert f"unknown deep-search agent {body['agent']!r}; available: " + ", ".join(arms.names()) in r.text


def test_the_registry_holds_the_core_arms_first_and_refuses_a_module_without_deep_search():
    assert arms.names()[:3] == ["langgraph", "deepagents", "deepagents_stock"]
    assert arms.get("deepagents_stock") is stock and arms.get("langgraph") is agent
    with pytest.raises(ValueError, match="unknown deep-search agent 'nope'"):
        arms.get("nope")
    with pytest.raises(TypeError, match="has no deep_search"):
        arms.register("broken", pytest)
    assert "broken" not in arms.names()


@pytest.mark.parametrize("implementation", ["deepagents", "deepagents_stock"])
def test_a_deepagents_turn_follows_the_event_order_and_names_only_known_steps(implementation):
    model = Scripted(steps=[todos(("a", "in_progress")), call("read_file", {"file_path": "/nowhere"}),
                            call("ls", {"path": "/"}) if implementation == "deepagents_stock" else done(),
                            call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}), done(), done()])
    deps = make_deps(model)
    deps.threads, deps.planner = None, None
    req = ChatRequest(question=QUESTION, doc_ids=[TABLES.doc_id, REPORT.doc_id], mode="deep", agent=implementation)

    async def go():
        notices.begin()
        return [e async for e in turn.respond(deps, req, [TABLES, REPORT], Conversation())]

    import rag.turn as t
    original = t.standalone_question
    t.standalone_question = lambda deps, q, conv: asyncio.sleep(0, q)
    try:
        events = asyncio.run(go())
    finally:
        t.standalone_question = original
    assert check_order([n for n, _ in events]) == "done"
    assert {x["tool"] for x in steps(events)} <= set(STEP_TOOLS)
    assert RESEARCH_TOOLS | {"plan", "verify", "write_todos", "read_file", "task", "summarize"} <= set(STEP_TOOLS)
    assert {"doc_grep", "ls", "glob", "write_file", "edit_file", "delete"} <= set(STEP_TOOLS)


# --- prompt sizes in real tokens ------------------------------------------------------------------------------


@pytest.mark.skipif(not TOKENIZER, reason="no Qwen3 tokenizer.json on this machine")
def test_the_trimmed_harness_prompts_are_a_fraction_of_the_built_in_ones_and_leave_the_slot_to_the_research():
    """deepagents 0.7 puts its guidance in the tool descriptions; langchain's todo list brings its own prompts."""
    import deepagents.middleware.filesystem as fs
    from deepagents.middleware.subagents import TASK_TOOL_DESCRIPTION
    from langchain.agents.middleware.todo import WRITE_TODOS_SYSTEM_PROMPT, WRITE_TODOS_TOOL_DESCRIPTION

    assert tokens.configure(TOKENIZER, 1.05)
    s = Settings(deep_subagents=True)
    builtin = sum(map(tokens.count_tokens, (
        WRITE_TODOS_SYSTEM_PROMPT, WRITE_TODOS_TOOL_DESCRIPTION, fs.LIST_FILES_TOOL_DESCRIPTION,
        fs.READ_FILE_TOOL_DESCRIPTION, fs.WRITE_FILE_TOOL_DESCRIPTION, fs.EDIT_FILE_TOOL_DESCRIPTION,
        fs.DELETE_TOOL_DESCRIPTION, fs.GLOB_TOOL_DESCRIPTION, fs.GREP_TOOL_DESCRIPTION, TASK_TOOL_DESCRIPTION)))
    trimmed = sum(map(tokens.count_tokens, (da.BASE, da.TODO_RULES, da.TODO_TOOL, da.FILES_RULES, da.READ_FILE_TOOL,
                                            da.TASK_RULES, da.TASK_TOOL)))
    assert builtin > 1800 and trimmed < 800 and trimmed * 3 < builtin
    model = Scripted(steps=[done()])
    research(make_deps(model, s), "deepagents")
    messages, kwargs = model.research_calls()[0]
    first = da.request_tokens(None, messages, [])  # the system message (flattened) and the question
    assert isinstance(messages[0].content, str)
    # the tools of the first prompt (8 research tools, write_todos, read_file, task) are counted by the guard too;
    # with them the first prompt is a small share of the slot and leaves the rest to results, retries and output
    assert first < 1600
    assert first + 1800 + s.research_retry_tokens + s.prompt_margin_tokens < s.slot_tokens // 4


# --- the stock harness (agent="deepagents_stock") ---------------------------------------------------------------


def system_text(messages) -> str:
    return da.flat(messages[0]).content


def test_the_stock_harness_keeps_its_default_prompts_and_tools_and_answers_from_the_shared_book():
    """deepagents 0.7: no authored base prompt and no todo list by default; the file tools and task carry their
    guidance in their descriptions."""
    model = Scripted(steps=[call("doc_grep", {"terms": ["South"]}),
                            call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT region, SUM(revenue) FROM orders GROUP BY 1"}),
                            call("write_file", {"file_path": "/notes.md", "content": "South 2735.62 [1]"}),
                            call("read_file", {"file_path": "/notes.md"}),
                            done("South had the highest Q3 revenue, 2735.62 [1].")])
    deps = make_deps(model)
    events, queue = research(deps, "deepagents_stock")
    first, kwargs = model.calls[0]
    names = kwargs["tool_names"]
    assert {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "task"} <= set(names)
    assert RESEARCH_TOOLS - {"grep"} | {"doc_grep"} <= set(names)
    assert "execute" not in names and "write_todos" not in names  # no sandbox; no todo list by default in 0.7
    prompt = system_text(first)
    assert prompt == stock.stock_prompt(Settings(), QUESTION, [TABLES, REPORT])  # ours only: 0.7 adds none
    assert "doc_grep: exact codes" in prompt and "search or doc_grep" in prompt and "search or grep" not in prompt
    # our tools under the new name reach the same services, scoped as always; the answer comes from the book
    assert deps.ingest.calls == [([TABLES.doc_id, REPORT.doc_id], ["South"], USER)]
    assert [c[0] for c in deps.sql.calls] == [TABLES.doc_id]
    seen = {m.name: m for m in tool_messages(model)}
    assert seen["read_file"].status != "error" and "South 2735.62 [1]" in seen["read_file"].content  # in memory
    assert [x["tool"] for x in steps(events)] == ["doc_grep", "sql", "write_file", "read_file"]
    assert not steps(events, {"verify"})  # the harness has no verify step
    assert [x["n"] for x in event(events, "sources")["sources"]] == [1]
    assert "South | 2735.62" in str(deps.llm.messages)  # the answer prompt holds the SQL result, as source [1]
    timings = event(events, "done")["timings"]
    assert timings["tool_calls"] == 2 and timings["harness_calls"] == 2
    assert queue == []


def test_the_stock_general_purpose_sub_agent_keeps_its_default_stack_with_our_budget_and_steps():
    from deepagents.middleware.subagents import DEFAULT_SUBAGENT_PROMPT

    model = Scripted(steps=[call("task", {"description": "Find Q3 revenue per region in orders.xlsx with sql",
                                          "subagent_type": "general-purpose"}, "c-task"),
                            call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}),  # the sub-agent
                            done("South 2735.62 [1]."),  # its report, never asked for: the budget is used
                            done("South led [1].")])
    deps = make_deps(model, Settings(agent_max_steps=1))
    events, queue = research(deps, "deepagents_stock", docs=(TABLES,))
    sub_messages, sub_kwargs = model.calls[1]
    assert system_text(sub_messages).startswith(DEFAULT_SUBAGENT_PROMPT)
    assert {"ls", "read_file", "doc_grep", "sql"} <= set(sub_kwargs["tool_names"])  # its default stack: files
    assert "task" not in sub_kwargs["tool_names"] and "write_todos" not in sub_kwargs["tool_names"]
    names = [x["tool"] for x in steps(events)]
    assert names == ["sql", "task"]  # the sub-agent's step streams before its task ends
    # its SQL used the one research call: neither the sub-agent nor the lead takes another step (budget notice)
    assert len(model.calls) == 2 and [n["code"] for n in queue] == ["research_budget"]


def test_the_stock_harness_is_stopped_by_our_slot_guard_and_budgets_instead_of_failing():
    s = Settings(agent_max_steps=10)
    model = Scripted(steps=[call("sql", {"doc_id": TABLES.doc_id, "query": f"SELECT {i}"}) for i in range(10)])
    deps = make_deps(model, s, Sql(wide=True))
    events, queue = research(deps, "deepagents_stock", docs=(TABLES,))
    codes = [n["code"] for n in queue]
    assert "research_context_full" in codes  # stock eviction (20,000 tokens as 4 characters each) never fires
    assert len(deps.sql.calls) < 10 and event(events, "done")["finish_reason"] == "stop"
    largest = max(da.request_tokens(da.flat(m[0]), m[1:], []) for m, _ in model.calls)
    assert largest + s.research_retry_tokens + s.prompt_margin_tokens <= s.slot_tokens


def test_a_stock_run_writes_its_files_to_memory_only_and_opens_no_connection():
    global _WATCH

    def script():
        return [todos(("a", "in_progress")), call("write_file", {"file_path": "/plan.md", "content": "x"}),
                call("edit_file", {"file_path": "/plan.md", "old_string": "x", "new_string": "y"}),
                call("task", {"description": "count rows", "subagent_type": "general-purpose"}),
                call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1"}), done("1 [1]"), call("ls", {"path": "/"}),
                done("done")]

    research(make_deps(Scripted(steps=script())), "deepagents_stock", docs=(TABLES,))  # warm: lazy imports
    model = Scripted(steps=script())
    deps = make_deps(model)
    req = ChatRequest(question=QUESTION, doc_ids=[TABLES.doc_id], mode="deep", agent="deepagents_stock")

    async def go():
        global _WATCH
        notices.begin()
        _WATCH = []
        try:
            return [e async for e in stock.deep_search(deps, req, [TABLES], Conversation(), QUESTION, {})]
        finally:
            watched, _WATCH = _WATCH, None
            go.watched = watched

    events = asyncio.run(go())
    assert {"write_file", "edit_file", "task", "sql", "ls"} <= {x["tool"] for x in steps(events)}
    assert "/plan.md" in {m.name: m for m in tool_messages(model)}["ls"].content
    assert go.watched == []


@pytest.mark.skipif(not TOKENIZER, reason="no Qwen3 tokenizer.json on this machine")
def test_the_stock_harness_first_prompt_in_real_tokens_fits_the_slot_with_room_for_results():
    """On 0.7.28 the harness's guidance is in its tool descriptions: about 2,600 tokens of schemas for its file
    tools and task, about 4,300 for the whole first prompt."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    assert tokens.configure(TOKENIZER, 1.05)
    s = Settings()

    class Capturing(Scripted):
        bound: list = Field(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            self.bound = tools
            return super().bind_tools(tools, **kwargs)

    model = Capturing(steps=[done()])
    research(make_deps(model, s), "deepagents_stock")
    messages, kwargs = model.calls[0]
    harness = [t for t in model.bound if getattr(t, "name", None) in {"ls", "read_file", "write_file", "edit_file",
                                                                       "delete", "glob", "grep", "task"}]
    assert len(harness) == 8 and len(kwargs["tool_names"]) == 16
    harness_tokens = sum(tokens.count_tokens(json.dumps(convert_to_openai_tool(t))) for t in harness)
    assert harness_tokens > 2000  # the harness's own guidance, whole
    first = da.request_tokens(da.flat(messages[0]), messages[1:], model.bound)
    assert first + s.research_retry_tokens + s.prompt_margin_tokens + 15000 < s.slot_tokens
