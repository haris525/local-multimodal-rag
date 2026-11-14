"""Deep research tools and graph: deduped calls, whole SQL results, read on a SQL source, peek, document names
as ids, a plan that sees the documents, a verify step that stops when nothing new was found, and thinking in
the plan and verify steps only (RAG_DEEP_THINK)."""

import asyncio
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage

from rag_common.contracts import ColumnSchema, DocStatus, DocType, DocumentInfo, SheetLayout, SqlResult, TableSchema

from rag import notices
from rag.agent import _PLAN, SourceBook, ToolContext, _documents, _tools, build_graph, json_object
from rag.chat import ChatRequest
from rag.config import Settings

LAYOUT = SheetLayout(header_rows=[3, 4], data_rows=[5, 62], footnote_rows=[63, 64], merged=["A1:H1"], blank_rows=[61],
                     raw_table="pop__raw", scores={"layout": 0.97, "simple": 0.52})
DOC = DocumentInfo(doc_id="d" * 32, name="census.xlsx", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
                   created_at="t", updated_at="t",
                   tables=[TableSchema(table="pop", source='sheet "POP"', sheet="POP", rows=57, row_filter=True,
                                       layout=LAYOUT, columns=[ColumnSchema(name="Area", sql_name="area", type="text",
                                                                            profile="57 distinct")]),
                           TableSchema(table="p1_table1", source="page 1, table 1", rows=3, row_filter=False,
                                       columns=[])])


class FakeSql:
    def __init__(self):
        self.queries: list[str] = []

    async def query(self, doc_id, sql, max_rows=200):
        self.queries.append(sql)
        if "__raw" in sql:
            return SqlResult(columns=["_row", "c1", "c2"], rows=[[1, "Title merged across", None], [3, "Area", "2020"],
                                                                 [68, "Release Date: January 2026", None]],
                             truncated=False)
        rows = [[f"group {i:03d} label", i * 1000] for i in range(80)]  # longer than a passage preview
        return SqlResult(columns=["year", "n"], rows=rows, truncated=False)


def context(sql: FakeSql | None = None) -> ToolContext:
    deps = SimpleNamespace(s=Settings(), sql=sql or FakeSql())
    return ToolContext(deps, ChatRequest(question="q", doc_ids=[DOC.doc_id], mode="deep"), [DOC], SourceBook())


def tool(ctx: ToolContext, name: str):
    return next(t for t in _tools(ctx) if t.name == name).coroutine


def test_identical_sql_is_not_run_again_and_adds_no_source_and_results_are_whole():
    ctx = context()
    sql = tool(ctx, "sql")
    text, first = asyncio.run(sql(DOC.doc_id, "SELECT year, COUNT(*) FROM t GROUP BY 1"))
    assert "group 079" in text and len(text) > ctx.s.agent_preview_chars  # whole, not a 1,000-character preview
    again, second = asyncio.run(sql(DOC.doc_id, "SELECT  year, COUNT(*)\nFROM t GROUP BY 1"))  # same query, other spacing
    assert second == first and "already made this exact call" in again
    assert len(ctx.deps.sql.queries) == 1 and len(ctx.book.sources) == 1


def test_read_on_a_sql_source_explains_instead_of_failing():
    ctx = context()
    asyncio.run(tool(ctx, "sql")(DOC.doc_id, "SELECT 1"))
    text, numbers = asyncio.run(tool(ctx, "read")(1))
    assert numbers == [] and "is a sql result, not a passage" in text
    assert "no source [9]" in asyncio.run(tool(ctx, "read")(9))[0]


def test_tools_accept_the_document_name_for_its_id():
    ctx = context()
    card, _ = asyncio.run(tool(ctx, "describe_table")("census.xlsx", "pop"))
    assert "table=pop" in card and "layout: header rows [3, 4]" in card
    asyncio.run(tool(ctx, "sql")("census.xlsx", "SELECT 2"))
    asyncio.run(tool(ctx, "sql")("pop", "SELECT 3"))  # a table name that only one selected document has
    card, _ = asyncio.run(tool(ctx, "describe_table")("0", "pop"))  # a guessed id, but only one document has pop
    assert "table=pop" in card
    assert ctx.deps.sql.queries == ["SELECT 2", "SELECT 3"]


def test_peek_shows_raw_rows_layout_and_the_raw_table_and_says_when_there_is_none():
    ctx = context()
    text, numbers = asyncio.run(tool(ctx, "peek")(DOC.doc_id, "pop"))
    assert "header rows [3, 4], data rows [5, 62], footnote rows [63, 64]" in text
    assert "row 1: A='Title merged across'" in text and "\n...\nrow 68:" in text and "Merged cells: A1:H1" in text
    assert "SQL table pop__raw" in text and numbers == [1] and ctx.book.sources[0].kind == "peek"
    assert "Both readings were scored (1 at best): layout 0.97, simple 0.52." in text
    assert "pop__raw" in ctx.deps.sql.queries[0]
    none, _ = asyncio.run(tool(ctx, "peek")(DOC.doc_id, "p1_table1"))
    assert "has no raw cells" in none


def test_peek_names_the_sheet_columns_of_a_table_to_the_right_and_its_blank_spacer_columns():
    from rag.agent import peek_text

    layout = SheetLayout(header_rows=[1], data_rows=[2, 3], first_col=4, blank_columns=["G"], raw_table="t__raw")
    text = peek_text("Sheet", "t", layout, ["_row", "c1", "c2"], [[1, "Team", "People"], [2, "Eng", "10"]], Settings())
    assert "row 1: E='Team', F='People'" in text and "c1 is column E" in text
    assert "Blank columns inside the table (left out): G." in text


def test_the_plan_prompt_lists_the_selected_documents_and_their_tables():
    image = DOC.model_copy(update={"doc_id": "e" * 32, "name": "chart.png", "doc_type": DocType.IMAGE, "tables": []})
    listing = _documents([image, DOC])
    assert "- chart.png (image)" in listing and "- census.xlsx (spreadsheet; table pop (57 rows: area)" in listing
    assert "{documents}" in _PLAN and "at least one sub-question for each" in _PLAN


class ScriptedModel:
    """Plays the planner, the agent and the verifier of the research graph."""

    def __init__(self):
        self.calls = {"plan": 0, "agent": 0, "verify": 0}

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        first = str(messages[0].content)
        if first.startswith("Break this question"):
            self.calls["plan"] += 1
            return AIMessage('{"sub_questions": ["How many per decade?"]}', response_metadata={"finish_reason": "stop"})
        if first.startswith("Question:") and "NOT yet answered" in first:
            self.calls["verify"] += 1
            return AIMessage('{"missing": ["the count for the 1970s"]}', response_metadata={"finish_reason": "stop"})
        self.calls["agent"] += 1
        if self.calls["agent"] == 1:
            return AIMessage("", tool_calls=[{"name": "sql", "args": {"doc_id": DOC.doc_id, "query": "SELECT 1"},
                                              "id": "c1"}], response_metadata={"finish_reason": "tool_calls"})
        return AIMessage("DONE", response_metadata={"finish_reason": "stop"})


def test_verify_stops_when_a_round_found_nothing_new_and_tells_the_user():
    async def run():
        queue = notices.begin()
        ctx = context()
        model = ScriptedModel()
        graph = build_graph(model, _tools(ctx), ctx.book, ctx.s, [DOC])
        await graph.ainvoke({"messages": [], "question": "q", "sub_questions": [], "missing": [], "rounds": 0,
                             "tool_calls": 0}, config={"recursion_limit": 40})
        return model, queue

    model, queue = asyncio.run(run())
    # verify once (missing), the agent once more (no new source), then the stale check ends it without a model call
    assert model.calls == {"plan": 1, "agent": 3, "verify": 1}
    assert [n["code"] for n in queue] == ["research_incomplete"] and "the count for the 1970s" in queue[0]["message"]


class Bound:
    """A model bound with settings, as ChatOpenAI.bind gives it: records each call's settings."""

    def __init__(self, model, settings: dict):
        self.model, self.settings = model, settings

    def bind(self, **settings):
        return Bound(self.model, {**self.settings, **settings})

    async def ainvoke(self, messages):
        return await self.model.reply(messages, self.settings)


class Recording(ScriptedModel):
    """The scripted graph's model, recording what each call was bound with. After thinking it writes prose
    around its JSON (no JSON mode then)."""

    def __init__(self):
        super().__init__()
        self.bound: list[tuple[str, dict]] = []

    def bind_tools(self, tools):
        return Bound(self, {})

    def bind(self, **settings):
        return Bound(self, settings)

    async def reply(self, messages, settings):
        msg = await self.ainvoke(messages)
        first = str(messages[0].content)
        kind = "plan" if first.startswith("Break") else "verify" if "NOT yet answered" in first else "agent"
        self.bound.append((kind, settings))
        thinking = (settings.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking") is True
        if thinking and msg.content.startswith("{"):
            return AIMessage("Here it is:\n" + msg.content, response_metadata=msg.response_metadata)
        return msg


def thinks(settings: dict) -> bool:
    return (settings.get("extra_body") or {}).get("chat_template_kwargs", {}).get("enable_thinking") is True


@pytest.mark.parametrize("think", [False, True])
def test_deep_think_makes_only_the_plan_and_verify_steps_think(think):
    async def run():
        queue = notices.begin()
        ctx = context()
        model = Recording()
        graph = build_graph(model, _tools(ctx), ctx.book, ctx.s, [DOC], think=think)
        state = await graph.ainvoke({"messages": [], "question": "q", "sub_questions": [], "missing": [], "rounds": 0,
                                     "tool_calls": 0}, config={"recursion_limit": 40})
        return model, state, queue

    model, state, queue = asyncio.run(run())
    s = Settings()
    kinds = [(kind, thinks(b), "response_format" in b) for kind, b in model.bound]
    plan_and_verify = [k for k in kinds if k[0] != "agent"]
    assert plan_and_verify == [("plan", think, not think), ("verify", think, not think)]
    assert all(k == ("agent", False, False) for k in kinds if k[0] == "agent")
    if think:
        plan = next(b for kind, b in model.bound if kind == "plan")
        assert plan["max_tokens"] == s.deep_think_tokens and plan["temperature"] == s.think_temperature
    assert state["sub_questions"] == ["How many per decade?"]  # parsed from the prose after the thought, too
    assert "plan_failed" not in [n["code"] for n in queue]


def test_a_json_reply_is_found_after_a_thought():
    assert json_object('<think>{"no": 1}</think>\n{"missing": ["x"]}') == {"missing": ["x"]}
    assert json_object('Sure.\n{"sub_questions": ["a"]}\nDone.') == {"sub_questions": ["a"]}
    with pytest.raises(ValueError):
        json_object("no json here")
