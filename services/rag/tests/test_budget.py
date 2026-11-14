"""Prompt budgets in the model's tokens: every prompt plus its output room fits one llm slot (32,768 tokens),
counted with the tokenizer (or a conservative estimate), and every cut or fallback is a notice.

Regression cases: 9 SQL results made a 33,880-token research
prompt that a chars/3 estimate put at 13,500; a 200-column peek was 45,384 tokens; 11 SQL sources made a
36,906-token answer prompt; and a deep question over a spreadsheet and a report answered from SQL alone,
never reading the report that stated another figure."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_common.contracts import ColumnSchema, DocStatus, DocType, DocumentInfo, SheetLayout, SqlResult, TableSchema
from rag_common.index import ChunkPayload

from rag import notices, tokens
from rag.agent import (_PLAN, _SYSTEM, SourceBook, ToolContext, _strings, _tools, _within_budget, build_graph, deep_search,
                       fair_evidence, peek_text, tools_tokens)
from rag.chat import ChatRequest, source_budget, standalone_question, stream_answer
from rag.config import Settings
from rag.conversation import Conversation
from rag.fusion import Candidate
from rag.planner import Planner
from rag.prompts import SYSTEM, Source, build_messages, sql_source

_HF = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots"
TOKENIZER = os.environ.get("RAG_TEST_TOKENIZER") or next((str(p) for p in _HF.glob("*/tokenizer.json")), "")


@pytest.fixture(autouse=True)
def estimate_mode():
    tokens.configure(None)
    yield
    tokens.configure(None)


def numbers_table(rows: int, cols: int = 8) -> str:
    head = "| year | " + " | ".join(f"source_{c}" for c in range(cols)) + " |"
    return head + "\n" + "\n".join("| " + str(1949 + r) + " | " + " | ".join(f"{(r + 1) * (c + 7) * 1234.567:.3f}"
                                                                          for c in range(cols)) + " |" for r in range(rows))


# --- counting --------------------------------------------------------------------------------------------------


def test_the_estimate_counts_every_digit_and_is_at_least_two_and_a_half_times_chars_over_three_on_numbers():
    text = numbers_table(40)
    digits = sum(ch.isdigit() for ch in text)
    assert tokens.count_tokens(text) >= digits  # Qwen writes each digit as a token
    assert tokens.count_tokens(text) > 1.6 * (len(text) // 3)  # the old chars/3 estimate was 1.6-2.3x too low


@pytest.mark.skipif(not TOKENIZER, reason="no Qwen3 tokenizer.json on this machine")
def test_the_estimate_is_never_below_the_tokenizer_and_the_tokenizer_count_carries_the_safety_factor():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    samples = [numbers_table(60), "row 3: A='Geographic Area', B='April 1, 2020 Estimates Base'\n" * 30,
               _SYSTEM, json.dumps({"name": "sql", "arguments": {"query": "SELECT region, SUM(q3) FROM t GROUP BY 1"}})]
    for text in samples:
        real = len(tok.encode(text, add_special_tokens=False).ids)
        assert tokens.estimate(text) >= real
    assert tokens.configure(TOKENIZER, 1.05) and tokens.exact()
    real = len(tok.encode(samples[0], add_special_tokens=False).ids)
    assert tokens.count_tokens(samples[0]) == -(-real * 105 // 100)
    cut = tokens.truncate_tokens(samples[0], 500)
    assert samples[0].startswith(cut) and tokens.count_tokens(cut) <= 500


def test_the_estimate_is_at_or_above_llama_servers_own_count_on_numbers_codes_ids_and_prose():
    """Runs of capitals ("ABW", "SP.POP.TOTL") and random-case IDs took 1.7-2x the tokens the
    estimate gave them. The fixture holds llama-server's own counts (its /tokenize endpoint)."""
    fixture = json.loads((Path(__file__).parent / "fixtures" / "served_tokens.json").read_text(encoding="utf-8"))
    assert len(fixture["samples"]) >= 15
    for sample in fixture["samples"]:
        assert tokens.estimate(sample["text"]) >= 1.05 * sample["served"], sample["name"]
        assert tokens.estimate(sample["text"]) <= 2.5 * sample["served"], sample["name"]  # still usable as a budget


def test_a_missing_tokenizer_file_falls_back_to_the_estimate():
    assert tokens.configure("/no/such/tokenizer.json") is False and not tokens.exact()
    assert tokens.count_tokens("12345") == tokens.estimate("12345")


def test_truncate_keeps_a_prefix_within_the_limit():
    text = numbers_table(30)
    cut = tokens.truncate_tokens(text, 300)
    assert text.startswith(cut) and tokens.count_tokens(cut) <= 300 < tokens.count_tokens(cut + text[len(cut):][:20])


def test_message_tokens_counts_tool_calls_and_tool_results():
    call = AIMessage("", tool_calls=[{"name": "sql", "args": {"query": "SELECT * FROM t LIMIT 200"}, "id": "c1"}])
    assert tokens.message_tokens([call]) > tokens.MESSAGE_TOKENS + 10
    result = ToolMessage(numbers_table(20), tool_call_id="c1")
    assert tokens.message_tokens([result]) >= tokens.count_tokens(numbers_table(20))


# --- H1: the research prompt never outgrows the slot -----------------------------------------------------------

TABLES = DocumentInfo(doc_id="e" * 32, name="eia.xlsx", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
                      created_at="t", updated_at="t",
                      tables=[TableSchema(table="gen", source='sheet "Gen"', sheet="Gen", rows=900, row_filter=True,
                                          columns=[ColumnSchema(name="Year", sql_name="year", type="number", profile="")])])


class WideSql:
    """Every query returns 200 rows of 9 numeric columns, as SELECT * on the EIA sheet does."""

    def __init__(self):
        self.queries = []

    async def query(self, doc_id, sql, max_rows=200):
        self.queries.append(sql)
        rows = [[str(1949 + r)] + [f"{(r + 1) * (c + 7) * 1234.567:.3f}" for c in range(8)] for r in range(200)]
        return SqlResult(columns=["year"] + [f"source_{c}" for c in range(8)], rows=rows, truncated=True)


class GreedyModel:
    """Asks for one more SELECT * every step and records the size of every research prompt it is sent."""

    def __init__(self, s: Settings, tools):
        self.s, self.fixed, self.prompts, self.step = s, tools_tokens(tools), [], 0

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        first = str(messages[0].content)
        if first.startswith("Break this question"):
            return AIMessage('{"sub_questions": ["generation by source and year"]}', response_metadata={"finish_reason": "stop"})
        if "NOT yet answered" in first:
            return AIMessage('{"missing": []}', response_metadata={"finish_reason": "stop"})
        self.prompts.append(self.fixed + tokens.message_tokens(messages))
        self.step += 1
        return AIMessage("", tool_calls=[{"name": "sql", "args": {"doc_id": TABLES.doc_id,
                                                                  "query": f"SELECT * FROM gen LIMIT 200 OFFSET {self.step}"},
                                          "id": f"c{self.step}"}], response_metadata={"finish_reason": "tool_calls"})


def test_the_research_stops_before_its_prompt_and_a_retried_step_outgrow_the_slot():
    s = Settings(agent_max_steps=40, sql_budget=6000)  # large results, many steps: only the slot can stop it

    async def run():
        queue = notices.begin()
        ctx = ToolContext(SimpleNamespace(s=s, sql=WideSql()), ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep"),
                          [TABLES], SourceBook())
        tools = _tools(ctx)
        model = GreedyModel(s, tools)
        await build_graph(model, tools, ctx.book, s, [TABLES]).ainvoke(
            {"messages": [], "question": "q", "sub_questions": [], "missing": [], "rounds": 0, "tool_calls": 0},
            config={"recursion_limit": 200})
        return model, queue

    model, queue = asyncio.run(run())
    assert len(model.prompts) >= 3 and all(p + s.research_retry_tokens + s.prompt_margin_tokens <= s.slot_tokens
                                           for p in model.prompts)
    assert "research_context_full" in [n["code"] for n in queue]


def test_a_full_research_prompt_is_one_notice_and_verify_sends_no_one_back():
    """With a small slot the context-full notice came three times (its message carried the prompt
    size) and verify kept sending the agent back to a prompt with no room."""
    class Insisting(GreedyModel):
        async def ainvoke(self, messages):
            if "NOT yet answered" in str(messages[0].content):
                self.verified = getattr(self, "verified", 0) + 1
                return AIMessage('{"missing": ["something else"]}', response_metadata={"finish_reason": "stop"})
            return await super().ainvoke(messages)

    s = Settings(llm_ctx=2 * 9000, sql_budget=6000)
    report = DocumentInfo(doc_id="b" * 32, name="report.pdf", doc_type=DocType.PDF, size=1, status=DocStatus.READY,
                          created_at="t", updated_at="t")

    async def run():
        queue = notices.begin()
        ctx = ToolContext(SimpleNamespace(s=s, sql=WideSql()), ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep"),
                          [TABLES, report], SourceBook())
        tools = _tools(ctx)
        model = Insisting(s, tools)

        async def text_search(query):
            src = ctx.book.add(Source(0, "r1", report.doc_id, report.name, "text", "p. 1", "The report says 42.", 8))
            return ctx.found([src])

        await build_graph(model, tools, ctx.book, s, [TABLES, report], text_search).ainvoke(
            {"messages": [], "question": "q", "sub_questions": [], "missing": [], "rounds": 0, "tool_calls": 0},
            config={"recursion_limit": 80})
        return model, queue

    model, queue = asyncio.run(run())
    codes = [n["code"] for n in queue]
    assert codes.count("research_context_full") == 1 and "research_incomplete" in codes
    assert "prompt" not in next(n["message"] for n in queue if n["code"] == "research_context_full").split("hold")[0]
    assert model.verified == 1  # verify checked once and sent no one back to a full prompt
    assert all(p + s.research_retry_tokens + s.prompt_margin_tokens <= s.slot_tokens for p in model.prompts)


def test_a_step_still_cut_after_its_retry_is_a_notice():
    class CutModel(GreedyModel):
        async def ainvoke(self, messages):
            if str(messages[0].content).startswith("Break this question") or "NOT yet answered" in str(messages[0].content):
                return await super().ainvoke(messages)
            return AIMessage("I will now", response_metadata={"finish_reason": "length"})

    s = Settings()

    async def run():
        queue = notices.begin()
        ctx = ToolContext(SimpleNamespace(s=s, sql=WideSql()), ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep"),
                          [TABLES], SourceBook())
        tools = _tools(ctx)
        await build_graph(CutModel(s, tools), tools, ctx.book, s, [TABLES]).ainvoke(
            {"messages": [], "question": "q", "sub_questions": [], "missing": [], "rounds": 0, "tool_calls": 0},
            config={"recursion_limit": 40})
        return queue

    assert "research_step_cut" in [n["code"] for n in asyncio.run(run())]


# --- H2: peek is capped in tokens and columns ------------------------------------------------------------------


def wide_rows(ncols: int, nrows: int = 23) -> list[list]:
    return [[r + 1, f"item {r} with a rather long label that goes on", *[f"{(r + 7) * (c + 3) * 1.234:.3f}" for c in range(ncols - 1)]]
            for r in range(nrows)]


def test_peek_of_a_200_column_sheet_shows_30_columns_names_the_rest_and_fits_its_token_cap():
    s = Settings()
    layout = SheetLayout(header_rows=[1], data_rows=[2, 501], raw_table="w__raw", sheet_rows=501, raw_rows=501)
    text = peek_text("w", "w", layout, ["_row"] + [f"c{i}" for i in range(1, 201)], wide_rows(200), s)
    assert tokens.count_tokens(text) <= s.peek_tokens
    assert "AD=" in text and "AE=" not in text and "more cells" in text
    assert "Columns AE-GR (170 more) are not shown; query w__raw (c31 to c200)" in text
    assert "'item 0 with a rather long label that goe...'" in text  # cells cut at peek_cell_chars


def test_peek_says_when_the_raw_table_was_cut_and_where_the_next_table_starts():
    s = Settings()
    layout = SheetLayout(header_rows=[1], data_rows=[2, 9], raw_table="w__raw", sheet_rows=250000, raw_rows=100000,
                         next_table_row=12)
    text = peek_text("w", "w", layout, ["_row", "c1", "c2"], [[1, "Name", "Val"], [2, "a", "1"]], s)
    assert "only the first 100000 of the sheet's 250000 rows" in text and "Another table starts at row 12" in text


def test_peek_on_a_second_table_reads_its_raw_rows_from_where_that_table_starts():
    class RawSql:
        queries: list[str] = []

        async def query(self, doc_id, sql, max_rows=200):
            self.queries.append(sql)
            return SqlResult(columns=["_row", "c1", "c2"], rows=[[12, "Team", "People"]], truncated=False)

    layout = SheetLayout(header_rows=[12], data_rows=[13, 14], raw_table="t2__raw", raw_first_row=12)
    doc = TABLES.model_copy(update={"tables": [TABLES.tables[0].model_copy(update={"table": "t2", "layout": layout})]})
    sql = RawSql()
    ctx = ToolContext(SimpleNamespace(s=Settings(), sql=sql), ChatRequest(question="q", doc_ids=[doc.doc_id], mode="deep"),
                      [doc], SourceBook())
    peek = next(t for t in _tools(ctx) if t.name == "peek").coroutine
    asyncio.run(peek(doc.doc_id, "t2"))
    assert "_row < 27" in sql.queries[0]  # 12 + 15 rows


# --- H3 and M10: answer prompts fit the slot, and the answer's room is what the slot leaves --------------------


def big_conversation() -> Conversation:
    conv = Conversation(thread_id="t", summary="Earlier the user compared sources and years. " * 200)
    conv.turns = [{"role": r, "content": "How did coal and gas change from 2000 to 2024? " * 30} for r in ("user", "assistant") * 3]
    conv.memories = [SimpleNamespace(text="Prefers tables with yearly figures and short notes. " * 6) for _ in range(100)]
    return conv


def sql_sources(n: int, s: Settings) -> list[Source]:
    book = SourceBook()
    for i in range(n):
        rows = [[str(1949 + r)] + [f"{(r + 1) * (c + 7) * 1234.567:.3f}" for c in range(8)] for r in range(200)]
        src = sql_source(0, "d", "eia.xlsx", "query", f"SELECT * FROM gen LIMIT 200 OFFSET {i * 200}",
                         SqlResult(columns=["year"] + [f"s{c}" for c in range(8)], rows=rows, truncated=True), s.sql_budget)
        src.id = f"sql:{i}"
        book.add(src)
    return book.sources


@pytest.mark.parametrize("think", [False, True])
def test_the_deep_answer_keeps_sources_only_while_the_whole_prompt_and_its_answer_fit_the_slot(think):
    s = Settings()
    conv = big_conversation()
    sources = sql_sources(14, s)
    budget = source_budget(s, "q " * 50, conv, think, images=2)
    kept = _within_budget(sources, budget)
    messages = build_messages("q " * 50, kept, {}, conv.prompt_block(s.conversation_turns), conv.memory_block())
    total = tokens.message_tokens(messages) + 2 * s.image_tokens + (s.think_tokens if think else s.answer_tokens)
    assert 0 < len(kept) < len(sources) and total <= s.slot_tokens - s.prompt_margin_tokens
    assert budget < s.source_budget  # the conversation and memory took their share


class RecordingLLM:
    def __init__(self, finish: str = "stop", usage: dict | None = None):
        self.max_tokens, self.messages, self.finish, self.usage = None, None, finish, usage

    async def stream(self, messages, max_tokens, think):
        self.max_tokens, self.messages = max_tokens, messages
        yield "token", "Partial"
        if self.usage:
            yield "usage", self.usage
        yield "finish", self.finish


def run_stream(deps, sources, think=False, conv=None):
    async def go():
        queue = notices.begin()
        events = [e async for e in stream_answer(deps, "q", sources, think, {}, conv or Conversation())]
        return events, queue
    return asyncio.run(go())


def test_the_answer_room_is_clamped_to_what_the_prompt_leaves_and_the_cut_notice_says_so():
    s = Settings(source_budget=40000)
    llm = RecordingLLM("length", {"prompt_tokens": 30000, "completion_tokens": 1900})
    deps = SimpleNamespace(s=s, llm=llm, ingest=None)
    huge = [Source(1, "a", "d", "doc.pdf", "text", "p. 1", numbers_table(260), 0)]  # leaves less than 4,096
    _, queue = run_stream(deps, huge)
    prompt = tokens.message_tokens(llm.messages)
    assert llm.max_tokens < s.answer_tokens and prompt + llm.max_tokens <= s.slot_tokens - s.prompt_margin_tokens
    [cut] = [n for n in queue if n["code"] == "answer_cut"]
    assert "all 1900 tokens the prompt left in the model's context" in cut["message"]


def test_an_answer_cut_at_its_own_limit_names_the_setting_and_the_tokens_used():
    s = Settings()
    deps = SimpleNamespace(s=s, llm=RecordingLLM("length", {"prompt_tokens": 900, "completion_tokens": 4096}), ingest=None)
    _, queue = run_stream(deps, [Source(1, "a", "d", "doc.pdf", "text", "p. 1", "short", 2)])
    assert deps.llm.max_tokens == s.answer_tokens
    assert "reached its length limit (4096 of 4096 tokens, RAG_ANSWER_TOKENS)" in queue[0]["message"]


# --- L14: verify sees every source ---------------------------------------------------------------------------


def test_verify_evidence_is_shared_so_a_late_source_is_never_dropped_for_an_early_long_one():
    s = Settings()
    sources = sql_sources(3, Settings(sql_budget=20000))
    sources.append(Source(4, "late", "p", "report.pdf", "text", "p. 9", "South led third-quarter revenue with 340.", 12))
    evidence = fair_evidence(sources, 6000, s.agent_preview_chars)
    assert tokens.count_tokens(evidence) <= 6000 + 40
    assert "South led third-quarter revenue with 340." in evidence and evidence.count("[cut]") == 3


# --- the live run: a deep question over a spreadsheet and a report reads the report too ----------------------

ORDERS = DocumentInfo(doc_id="a" * 32, name="smoke-orders.xlsx", doc_type=DocType.SPREADSHEET, size=1,
                      status=DocStatus.READY, created_at="t", updated_at="t",
                      tables=[TableSchema(table="orders", source='sheet "Orders"', sheet="Orders", rows=40, row_filter=True,
                                          columns=[ColumnSchema(name="Region", sql_name="region", type="text", profile="4 distinct"),
                                                   ColumnSchema(name="Revenue", sql_name="revenue", type="number", profile="")])])
REPORT = DocumentInfo(doc_id="b" * 32, name="smoke-aurora_report.pdf", doc_type=DocType.PDF, size=1,
                      status=DocStatus.READY, created_at="t", updated_at="t")
QUESTION = "Which region had the highest third-quarter revenue, and how much?"


class OrdersSql:
    async def query(self, doc_id, sql, max_rows=200):
        return SqlResult(columns=["region", "q3_revenue"], rows=[["South", "2735.62"], ["North", "2101.10"]], truncated=False)


class Retrieval:
    def __init__(self):
        self.searched: list[list[str]] = []

    async def search(self, query, doc_ids, options=None, scope=None):
        self.searched.append(list(doc_ids))
        if doc_ids != [REPORT.doc_id]:
            return SimpleNamespace(candidates=[])
        text = "Third-quarter revenue by region: the South region led with 340 (thousand dollars), ahead of the North."
        payload = ChunkPayload(doc_id=REPORT.doc_id, doc_name=REPORT.name, user_id="u", kind="text", text=text,
                               chunk_index=4, n_tokens=30, page=3, doc_type="pdf", search_text=text, text_sha="x")
        return SimpleNamespace(candidates=[Candidate(id="r4", payload=payload, score=0.9, ranks={})])


class SqlOnlyModel:
    """The live failure: plans one SQL sub-question, runs SQL, says DONE, and verify finds nothing missing."""

    def __init__(self):
        self.agent_prompts = []

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        first = str(messages[0].content)
        if first.startswith("Break this question"):
            return AIMessage('{"sub_questions": ["Q3 revenue per region from the orders table"]}',
                             response_metadata={"finish_reason": "stop"})
        if "NOT yet answered" in first:
            return AIMessage('{"missing": []}', response_metadata={"finish_reason": "stop"})
        self.agent_prompts.append("\n".join(str(m.content) for m in messages))
        if not any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage("", tool_calls=[{"name": "sql", "args": {"doc_id": ORDERS.doc_id, "query":
                             "SELECT region, SUM(revenue) AS q3_revenue FROM orders WHERE quarter = 3 GROUP BY 1 ORDER BY 2 DESC"},
                                              "id": "c1"}], response_metadata={"finish_reason": "tool_calls"})
        return AIMessage("DONE", response_metadata={"finish_reason": "stop"})


def test_smoke_deep_case_a_question_over_a_spreadsheet_and_a_report_also_searches_the_report():
    s = Settings()
    model, retrieval, llm = SqlOnlyModel(), Retrieval(), RecordingLLM()
    deps = SimpleNamespace(s=s, graph_llm=model, retrieval=retrieval, sql=OrdersSql(), llm=llm, ingest=None)
    req = ChatRequest(question=QUESTION, doc_ids=[ORDERS.doc_id, REPORT.doc_id], mode="deep")

    async def run():
        queue = notices.begin()
        events = [e async for e in deep_search(deps, req, [ORDERS, REPORT], Conversation(), QUESTION, {})]
        return events, queue

    events, _ = asyncio.run(run())
    assert [REPORT.doc_id] in retrieval.searched  # the report was searched on its own
    sources = next(data for name, data in events if name == "sources")["sources"]
    assert {x["doc_name"] for x in sources} == {"smoke-orders.xlsx", "smoke-aurora_report.pdf"}
    prompt = json.dumps(llm.messages)
    assert "2735.62" in prompt and "South region led with 340" in prompt  # the answer sees both figures
    assert "When sources disagree, say so and cite each of them." in llm.messages[0]["content"]
    assert any("the text documents" in p and "both must be reported" in p for p in model.agent_prompts)
    steps = [data for name, data in events if name == "step" and data["tool"] == "verify"]
    assert steps and "what the text documents say" in steps[0]["args"]["missing"][0]


def test_the_plan_and_the_research_rules_ask_for_the_text_documents_when_tables_are_selected_too():
    assert "text documents (PDF, Word, text, images) and tables are both selected" in " ".join(_PLAN.split())
    assert "search the text documents too, even after SQL answered" in _SYSTEM


def test_a_tables_only_selection_runs_no_extra_text_search():
    s = Settings()
    retrieval = Retrieval()
    deps = SimpleNamespace(s=s, graph_llm=SqlOnlyModel(), retrieval=retrieval, sql=OrdersSql(), llm=RecordingLLM(), ingest=None)
    req = ChatRequest(question=QUESTION, doc_ids=[ORDERS.doc_id], mode="deep")

    async def run():
        notices.begin()
        return [e async for e in deep_search(deps, req, [ORDERS], Conversation(), QUESTION, {})]

    asyncio.run(run())
    assert retrieval.searched == []


def _deep(report, model):
    retrieval = Retrieval()
    deps = SimpleNamespace(s=Settings(), graph_llm=model, retrieval=retrieval, sql=OrdersSql(), llm=RecordingLLM(),
                           ingest=None)
    req = ChatRequest(question=QUESTION, doc_ids=[ORDERS.doc_id, report.doc_id], mode="deep")

    async def run():
        queue = notices.begin()
        return [e async for e in deep_search(deps, req, [ORDERS, report], Conversation(), QUESTION, {})], queue

    events, queue = asyncio.run(run())
    return retrieval, events, queue


def test_a_pdf_that_also_holds_a_table_is_still_searched_as_a_text_document():
    """The text guarantee took "has no tables" for "is a text document", so a report with one
    extracted table was never searched."""
    table = TableSchema(table="p7_t1", source="page 7, table 1", sheet=None, rows=4, row_filter=False,
                        columns=[ColumnSchema(name="Year", sql_name="year", type="number", profile="min 2020")])
    retrieval, events, _ = _deep(REPORT.model_copy(update={"tables": [table]}), SqlOnlyModel())
    assert [REPORT.doc_id] in retrieval.searched
    sources = next(data for name, data in events if name == "sources")["sources"]
    assert {x["doc_name"] for x in sources} == {"smoke-orders.xlsx", "smoke-aurora_report.pdf"}


class VerifySays(SqlOnlyModel):
    def __init__(self, reply: str):
        super().__init__()
        self.reply = reply

    async def ainvoke(self, messages):
        if "NOT yet answered" in str(messages[0].content):
            return AIMessage(self.reply, response_metadata={"finish_reason": "stop"})
        return await super().ainvoke(messages)


def test_verify_fields_are_typed_and_a_check_without_usable_json_is_a_notice():
    """{"missing": "none"} once became "Still missing: n; o; n"; a reply that was not JSON passed silently."""
    for reply in ('{"missing": "none"}', '{"missing": "Nothing."}', '{"missing": [null, {"a": 1}]}'):
        _, _, queue = _deep(REPORT, VerifySays(reply))
        assert [n["code"] for n in queue if n["code"].startswith("research_")] == [], reply
    _, _, queue = _deep(REPORT, VerifySays("the evidence covers the question"))
    assert "research_check_failed" in [n["code"] for n in queue]
    assert _strings("the 1970s count") == ["the 1970s count"] and _strings(["a", None, 3, " "]) == ["a", "3"]
    assert _strings(None) == [] and _strings({"q": 1}) == [] and _strings(["none"]) == [] and _strings(["N/A", "x"]) == ["x"]


def test_the_rules_round_percentages_after_scaling_and_count_partial_periods_by_their_range():
    rules = " ".join(_SYSTEM.split())
    assert "ROUND(100.0 * col, 2) for a column of fractions (0.0149 is 1.49%), never ROUND(fraction, 2)" in rules
    assert "the years covered are MAX - MIN + 1" in rules and "COUNT(DISTINCT year)" not in rules
    assert "Let SQL do every count, sum, average and grouping" in rules
    assert "0.0149 is 1.49%, 0.0051 is 0.51%" in SYSTEM


# --- L13: notices for a cut rewrite and the planner's choices -------------------------------------------------


class OneReply:
    def __init__(self, reply: dict):
        self.reply = reply

    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return self.reply


def test_a_cut_rewrite_searches_the_question_as_written_and_says_so():
    conv = Conversation(turns=[{"role": "user", "content": "Tell me about coal."}, {"role": "assistant", "content": "Coal..."}])
    deps = SimpleNamespace(s=Settings(), llm=OneReply({"content": "coal generation in", "finish_reason": "length"}))

    async def run():
        queue = notices.begin()
        return await standalone_question(deps, "and gas?", conv), queue

    question, queue = asyncio.run(run())
    assert question == "and gas?" and [n["code"] for n in queue] == ["rewrite_cut"]


@pytest.mark.parametrize("decision, code", [
    ({"route": "search"}, "planner_search"),
    ({"route": "sql", "doc_id": ORDERS.doc_id, "table": "nope", "sql": "SELECT 1"}, "plan_incomplete"),
    ({"route": "sql", "doc_id": ORDERS.doc_id, "table": "orders"}, "plan_incomplete"),
])
def test_the_planner_says_when_the_answer_comes_from_text_search(decision, code):
    async def run():
        queue = notices.begin()
        plan = await Planner(OneReply({"content": json.dumps(decision), "finish_reason": "stop"}), None, 2500).plan(
            QUESTION, [ORDERS])
        return plan, queue

    plan, queue = asyncio.run(run())
    assert plan.route == "search" and [n["code"] for n in queue] == [code]
