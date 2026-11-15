"""Deep research: a LangGraph state graph around a ReAct tool loop.

    plan -> agent <-> tools -> verify -> (agent, for missing facts | end)

`plan` splits the question into sub-questions, knowing which documents (and tables) are selected; `agent`
(Qwen3.8 with tool calling) reasons and calls tools until it has evidence; `verify` checks the evidence
against every sub-question and sends the agent back for what is missing (bounded rounds, and never again
when the last round found nothing new). When text documents and tables are both selected and no text passage
was found, verify searches the text documents itself, so a figure a report states differently is never
missed. The cited answer is then streamed from the sources the tools found, with the same prompt as
single-shot answers.

Tools are registered in TOOLS: each entry builds one StructuredTool from a ToolContext, so a new tool is one
function and one registry line (run_python, rag.sandbox, is offered only when RAG_DEEP_PYTHON or the request's
`python` asks). Every tool returns (text for the model, source numbers it produced). Identical calls are
answered from the first call's result without running again. SQL results reach the model whole (up to the SQL budget), peek results within peek_tokens; passages
are cut at agent_preview_chars. Every prompt is counted in the model's tokens (rag.tokens): the research
prompt (tool schemas, rules and every message so far) may grow until it and a retried step's room fill the
llm slot. Budgets (tool calls, the slot, verify rounds) end the research with a notice to the user, never
silently.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from rag_common.chat_events import ChatEvent
from rag_common.contracts import ColumnSchema, DocType, DocumentInfo, TableSchema
from rag_common.index import point_id
from rag_common.logging import log_fields
from rag_common.profiling import detect
from rag_common.retry import TRANSIENT_STATUS

from rag.chat import (ChatRequest, Deps, Event, deep_python, deep_think, figure_slots, nothing_found, per_document,
                      source_budget, stream_answer)
from rag.config import Settings
from rag.conversation import Conversation
from rag.fusion import Candidate
from rag.notices import notice
from rag.planner import schema_cards
from rag.prompts import Source, location, source_cost, sql_source
from rag.retrieval import SearchOptions
from rag.sandbox import close as close_sandbox
from rag.sandbox import run_python_tool
from rag.tokens import TOOLS_PREAMBLE_TOKENS, count_tokens, message_tokens, truncate_tokens
from rag.totals import profile_totals, table_totals
from rag.value_structure import profile_structure, separator_counts

log = logging.getLogger("rag.agent")

_PLAN = """Break this question about the user's selected documents into the smallest set of sub-questions
(1 to 4) that together answer it. A simple question is its own single sub-question.
Selected documents:
{documents}
When the question refers to several documents, write at least one sub-question for each of them. When text
documents (PDF, Word, text, images) and tables are both selected, also ask what the text documents say about
it, even when a table seems to hold the answer: a report may state the figure differently. Keep the
question's own terms, periods and units. Do not answer the question.
Question: {question}
Return JSON: {{"sub_questions": ["..."]}}"""

_SYSTEM = """You research questions in the user's selected documents with tools. Think step by step:
decide what you need, call a tool, read the result, repeat.
- search: passages by meaning and keywords. grep: exact codes, IDs, names, numbers or phrases.
- read: the text just before and after a numbered passage from search or grep (not SQL results or images).
- list_tables, describe_table, sql: structured data. Describe the relevant table to learn its columns, types
  and values, then compute the answer with one DuckDB SELECT. Let SQL do every count, sum, average and
  grouping; never add up rows yourself, and aggregate instead of listing many rows (at most {sql_rows} come back).
- profile_column: one column's counts, its most frequent values, range and quartiles, examples and the
  detected pattern (dates stored as numbers, codes, units, mixed values). Use it to understand a column
  before filtering or grouping on it; it is cheaper and safer than exploratory SQL.
- Names (places, countries, organizations, products): before filtering, grouping or counting rows by one, see
  how the column writes it (its distinct values or its profile). Data often uses abbreviations, codes, short or
  official names instead of the common one; match every form that means the same. A substring or
  case-insensitive match can also catch unrelated values that merely contain the text: check what it caught
  before counting.
- Totals: a table can mix detail rows with rows that total them (a label such as Total or All ..., a country
  above its states, regions above their states, a parent code above its parts) and can hold total columns.
  Before adding up, counting, averaging or ranking rows, look at the label column's values (table descriptions
  and profiles mark likely totals) and leave out every total that covers rows you already include; when the
  question asks for the total or a share of it, use the total row. If unsure, compare its value with the SUM
  of the rows it would cover.
- SQL uses the snake_case column names that list_tables and describe_table show, not the headers seen in
  passages. A DATE column compares with DATE '2024-01-01' or YEAR(column). Round a percentage after
  making it one: ROUND(100.0 * part / whole, 2), or ROUND(100.0 * col, 2) for a column of fractions (0.0149
  is 1.49%), never ROUND(fraction, 2).
- DuckDB: `/` always returns a decimal (7 / 2 = 3.5). For whole-number buckets use FLOOR(), for example
  FLOOR(year / 10) * 10 for decades, and YEAR(date_column) for the year of a date. For a count per period
  (year, decade), also return MIN and MAX of the year in it: the years covered are MAX - MIN + 1 (within the
  data's range), so a period that is only partly covered is visible even when some years have no rows.
- peek: the raw cells of a sheet before its header was read. Use it when column names look wrong (very long,
  generic or repeated) or numbers are stored as text; the raw table it names can be queried with sql.
- Never repeat a call with the same arguments; the result will not change. If a result is not what you
  need, change the query.
- When text documents and tables are both selected, search the text documents too, even after SQL answered:
  a report may give a different figure, and the answer must then report both.
Cover every sub-question. When you have evidence for all of them, stop calling tools and reply DONE."""

_VERIFY = """Question: {question}
Sub-questions: {subs}

Evidence found so far:
{evidence}

Which sub-questions are NOT yet answered by this evidence? A sub-question is answered when the evidence
contains the facts or numbers it needs; a comparison or ratio of numbers that are present is not missing.
Do not list something a tool has already returned.
Return JSON: {{"missing": ["sub-question or fact still needed"]}}"""

_WHOLE = ("sql", "peek", "profile", "python")  # shown whole (already within the SQL, peek, profile or Python budget)


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    sub_questions: list[str]
    missing: list[str]
    rounds: int
    tool_calls: int


class SourceBook:
    """Stable numbering of every source the agent has seen."""

    def __init__(self):
        self.sources: list[Source] = []
        self._by_id: dict[str, Source] = {}

    def add_candidates(self, cands: list[Candidate]) -> list[Source]:
        out = []
        for c in cands:
            if c.id not in self._by_id:
                p = c.payload
                src = Source(len(self.sources) + 1, c.id, p.doc_id, p.doc_name, p.kind, location(p), p.text,
                             p.n_tokens, p.asset, c.score, dict(c.ranks), p.chunk_index, p.generation)
                self.sources.append(src)
                self._by_id[c.id] = src
            out.append(self._by_id[c.id])
        return out

    def add(self, src: Source) -> Source:
        src.n = len(self.sources) + 1
        self.sources.append(src)
        self._by_id[src.id] = src
        return src

    def get(self, n: int) -> Source | None:
        return self.sources[n - 1] if 1 <= n <= len(self.sources) else None

    @staticmethod
    def render(found: list[Source], preview_chars: int = 1000) -> str:
        """Passages cut at preview_chars; SQL and peek results whole (the SQL budget already bounds them)."""
        if not found:
            return "No results."
        return "\n\n".join(f"[{s.n}] {s.doc_name} | {s.location}\n{s.text if s.kind in _WHOLE else s.text[:preview_chars]}"
                           for s in found)


@dataclass
class ToolContext:
    """What every tool may use. `seen` answers a repeated call from its first result."""

    deps: Deps
    req: ChatRequest
    docs: list[DocumentInfo]
    book: SourceBook
    seen: dict[str, tuple[str, list[int]]] = field(default_factory=dict)
    sandbox: Any = None  # run_python's state for the request (rag.sandbox.Session), made on its first call

    @property
    def s(self) -> Settings:
        return self.deps.s

    def found(self, sources: list[Source]) -> tuple[str, list[int]]:
        return SourceBook.render(sources, self.s.agent_preview_chars), [s.n for s in sources]

    def doc(self, doc_id: str) -> DocumentInfo:
        """A selected document by its id, its file name, or the name of a table only it has (the model sometimes
        passes the name or the table)."""
        by_id = {d.doc_id: d for d in self.docs}
        doc = by_id.get(doc_id) or next((d for d in self.docs if d.name == doc_id), None)
        if doc is None:
            owners = [d for d in self.docs if any(t.table == doc_id for t in d.tables)]
            doc = owners[0] if len(owners) == 1 else None
        if doc is None:
            raise ValueError("doc_id is not one of the selected documents; call list_tables for the ids")
        return doc

    def table(self, doc_id: str, table: str) -> tuple[DocumentInfo, TableSchema]:
        """The document and its table; an id that matches no selected document is forgiven when only one of
        them has the table (the model has passed "0" or a guess)."""
        try:
            doc = self.doc(doc_id)
        except ValueError:
            doc = self.doc(table)  # raises the same error when no single document has the table
        match = next((t for t in doc.tables if t.table == table), None)
        if match is None:
            raise ValueError(f"document {doc.name} has no table {table!r}; call list_tables first")
        return doc, match

    async def once(self, key: str, run: Callable[[], Any]) -> tuple[str, list[int]]:
        """Run a tool call, or answer an identical earlier call from its result."""
        if key in self.seen:
            text, numbers = self.seen[key]
            return (f"You already made this exact call; its result is source(s) {numbers or 'none'}. Use it, or "
                    "change the arguments if you need something else."), numbers
        result = await run()
        self.seen[key] = result
        return result


def _key(tool: str, **args) -> str:
    return tool + ":" + json.dumps({k: " ".join(str(v).split()) if isinstance(v, str) else v for k, v in args.items()},
                                   sort_keys=True)


# --- tools --------------------------------------------------------------------------------------------------


def search_tool(ctx: ToolContext) -> StructuredTool:
    async def search(query: str) -> tuple[str, list[int]]:
        async def run():
            result = await ctx.deps.retrieval.search(query, ctx.req.doc_ids, SearchOptions(top_k=5))
            return ctx.found(ctx.book.add_candidates(result.candidates))
        return await ctx.once(_key("search", query=query), run)
    return _tool(search, "Hybrid search (meaning, keywords, exact terms, figures) over the selected documents.")


def grep_tool(ctx: ToolContext) -> StructuredTool:
    async def grep(terms: list[str]) -> tuple[str, list[int]]:
        async def run():
            hits = await ctx.deps.ingest.grep(ctx.req.doc_ids, [str(t) for t in terms][:5], limit=ctx.s.grep_limit)
            return ctx.found(ctx.book.add_candidates(await ctx.deps.retrieval.fetch([h.id for h in hits])))
        return await ctx.once(_key("grep", terms=sorted(map(str, terms))), run)
    return _tool(grep, "Exact, case-insensitive match of literal terms such as codes, IDs, names, numbers, phrases.")


def read_tool(ctx: ToolContext) -> StructuredTool:
    async def read(source: int) -> tuple[str, list[int]]:
        src = ctx.book.get(int(source))
        if src is None:
            return f"There is no source [{source}] yet.", []
        if src.chunk_index is None:
            return (f"Source [{src.n}] is a {src.kind} result, not a passage of a document, so there is no text "
                    "before or after it. Use sql or search instead."), []

        async def run():
            ids = [point_id(src.doc_id, src.chunk_index + d, src.generation) for d in (-1, 1) if src.chunk_index + d >= 0]
            return ctx.found(ctx.book.add_candidates(await ctx.deps.retrieval.fetch(ids)))
        return await ctx.once(_key("read", source=src.n), run)
    return _tool(read, "Read the passages just before and after a numbered passage from search or grep (not for "
                       "SQL results or images).")


def list_tables_tool(ctx: ToolContext) -> StructuredTool:
    async def list_tables() -> tuple[str, list[int]]:
        return schema_cards(ctx.docs, budget_tokens=ctx.s.list_tables_tokens, full=False) or "No tables.", []
    return _tool(list_tables, "List every table in the selected documents with its columns.")


def describe_table_tool(ctx: ToolContext) -> StructuredTool:
    async def describe_table(doc_id: str, table: str) -> tuple[str, list[int]]:
        doc, match = ctx.table(doc_id, table)
        card = schema_cards([doc.model_copy(update={"tables": [match]})], budget_tokens=ctx.s.schema_card_tokens)
        layout = match.layout
        if layout and (len(layout.header_rows) > 1 or layout.footnote_rows or layout.section_rows):
            card += (f"\n  layout: header rows {layout.header_rows}, data rows {layout.data_rows}, "
                     f"footnote rows {layout.footnote_rows or 'none'} (peek shows the raw cells)")
        key = _key("table_totals", doc_id=doc.doc_id, table=match.table)  # a repeated call runs its queries once
        if key not in ctx.seen:
            ctx.seen[key] = (await table_totals(ctx.deps.sql, doc, match, ctx.s), [])
        totals = ctx.seen[key][0]
        return card + ("\n" + totals if totals else ""), []
    return _tool(describe_table, "Columns of one table with types, units, full headers and their values or ranges.")


def sql_tool(ctx: ToolContext) -> StructuredTool:
    async def sql(doc_id: str, query: str) -> tuple[str, list[int]]:
        doc = ctx.doc(doc_id)

        async def run():
            result = await ctx.deps.sql.query(doc.doc_id, query, ctx.s.sql_max_rows)
            src = sql_source(0, doc.doc_id, doc.name, "query", query, result, ctx.s.sql_budget)
            src.id = f"sql:{doc.doc_id}:{len(ctx.book.sources)}"
            return ctx.found([ctx.book.add(src)])
        return await ctx.once(_key("sql", doc_id=doc.doc_id, query=query), run)
    return _tool(sql, f"Run one read-only DuckDB SELECT on one document's tables. Aggregate in SQL; at most "
                      f"{ctx.s.sql_max_rows} rows come back.")


def _column(table: TableSchema, column: str) -> ColumnSchema:
    """A column of the table by its sql_name, display name or full header (the model passes any of them)."""
    key = column.strip().strip('"').lower()
    for c in table.columns if key else []:  # an empty name matches nothing (not a column without a header)
        if key == c.sql_name or key == c.name.lower() or (c.header and key == c.header.lower()):
            return c
    names = ", ".join(c.sql_name for c in table.columns)
    raise ValueError(f"table {table.table} has no column {column!r}; its columns are: {names[:800]}")


def profile_text(table: TableSchema, col: ColumnSchema, stats: list, top: list[list], s: Settings) -> str:
    """A column profile as the model sees it, within profile_tokens. The pattern found at ingest (from every
    original cell) wins over one detected from the sampled values."""
    n, filled, distinct, lo, hi, spread = (list(stats) + [None] * 6)[:6]
    n, filled, distinct = int(n or 0), int(filled or 0), int(distinct or 0)
    empty = n - filled
    about = ", ".join(x for x in (col.type, col.unit or "", f"header {col.header!r}" if col.header else "") if x)
    lines = [f"Column {col.sql_name} of table {table.table} ({about}), fresh from the data:",
             f"rows {n}; empty {empty} ({empty / n:.1%})" if n else "rows 0",
             f"distinct values {distinct}" + (" (unique: one row each)" if filled > 1 and distinct == filled else "")]
    if col.type in ("number", "date") and filled:
        parts = [p.strip(" '\"").removesuffix(".0") for p in str(spread or "").strip("[]").split(",")] if spread else ["-"]
        lines.append(f"min {str(lo).removesuffix('.0')}, max {str(hi).removesuffix('.0')}, quartiles {' | '.join(parts)}")
    elif spread:
        lines.append(f"text length {spread} characters")
    values = [str(v) for v, _ in top if v is not None]
    pattern, warning = detect(values, col.type, filled, distinct, col.header or col.name)
    lines.append(f"pattern: {col.pattern or pattern or 'none detected'}")
    if col.warning or warning:
        lines.append(f"warning: {col.warning or warning}")
    cut = s.peek_cell_chars
    shown = ", ".join(f"{'NULL' if v is None else repr(str(v)[:cut])} {c}" for v, c in top)
    if distinct <= len(top) or (top and top[0][1] >= 3):
        lines.append(f"most frequent values (with counts): {shown}")
    else:  # (almost) every value occurs once: a few of them say more than a list of counts of 1
        lines.append("examples (values rarely repeat): " + ", ".join(repr(v[:cut]) for v in values[:10]))
    return truncate_tokens("\n".join(lines), s.profile_tokens)


def profile_column_tool(ctx: ToolContext) -> StructuredTool:
    async def profile_column(table: str, column: str, doc_id: str = "") -> tuple[str, list[int]]:
        doc, match = ctx.table(doc_id or table, table)
        col = _column(match, column)

        async def run():
            q, t = f'"{col.sql_name}"', match.table  # both are names from the schema, never the model's text
            spread = {"number": f"CAST(quantile_cont({q}, [0.25, 0.5, 0.75]) AS VARCHAR)",
                      "date": f"CAST(quantile_disc({q}, [0.25, 0.5, 0.75]) AS VARCHAR)"}.get(
                col.type, f"CAST(MIN(LENGTH(CAST({q} AS VARCHAR))) AS VARCHAR) || '-' || "
                          f"CAST(MAX(LENGTH(CAST({q} AS VARCHAR))) AS VARCHAR)")
            seps = "".join(f", {x}" for x in separator_counts(q)) if col.type == "text" else ""  # rag.value_structure
            stats = await ctx.deps.sql.query(
                doc.doc_id, f"SELECT COUNT(*), COUNT({q}), COUNT(DISTINCT {q}), CAST(MIN({q}) AS VARCHAR), "
                            f"CAST(MAX({q}) AS VARCHAR), {spread}{seps} FROM {t}", 1)
            top = await ctx.deps.sql.query(
                doc.doc_id, f"SELECT CAST({q} AS VARCHAR) AS value, COUNT(*) AS n FROM {t} GROUP BY 1 "
                            "ORDER BY n DESC, hash(value) LIMIT 20", 20)
            row = stats.rows[0] if stats.rows else []
            text = profile_text(match, col, row, top.rows, ctx.s)
            structure = await profile_structure(ctx.deps.sql, doc, t, col, row, len(top.rows), ctx.s)
            text += "\n" + structure if structure else ""
            distinct = int(row[2] or 0) if len(row) > 2 else 0
            totals = await profile_totals(ctx.deps.sql, doc, match, col, top.rows, distinct, ctx.s)  # rag.totals
            text += "\n" + totals if totals else ""
            src = Source(0, f"profile:{doc.doc_id}:{t}:{col.sql_name}", doc.doc_id, doc.name, "profile",
                         f"profile of column {col.sql_name} in table {t}", text, count_tokens(text))
            return ctx.found([ctx.book.add(src)])
        return await ctx.once(_key("profile_column", doc_id=doc.doc_id, table=match.table, column=col.sql_name), run)
    return _tool(profile_column, "Statistics of one column, fresh from the data: row and empty counts, the 20 most "
                                 "frequent values with counts, min, max and quartiles, examples and the detected "
                                 "pattern (dates stored as numbers, codes, units, mixed values). doc_id is optional "
                                 "when only one selected document has the table.")


def _col(i: int) -> str:
    s = ""
    while i:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


def _cell(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return repr(text if len(text) <= limit else text[:limit] + "...")


def peek_text(sheet: str, table: str, layout, columns: list[str], rows: list[list], s: Settings) -> str:
    """The raw cells as the model sees them: at most peek_columns columns (cells cut at peek_cell_chars), and
    within peek_tokens (rows dropped from the middle, then the text cut)."""
    width = max(0, len(columns) - 1)
    shown = min(width, s.peek_columns)
    off = int(getattr(layout, "first_col", 0) or 0)  # a table to the right of another starts in a later column
    head = [f'Raw cells of sheet "{sheet}" before its header was read (first {s.peek_rows} and last 8 rows; column '
            "letters as in Excel).",
            f"Read as: header rows {layout.header_rows or 'none'}, data rows {layout.data_rows}, footnote rows "
            f"{layout.footnote_rows or 'none'}, section rows {layout.section_rows or 'none'}.",
            f"Merged cells: {', '.join(layout.merged[:20]) or 'none'}. Blank rows: "
            f"{', '.join(map(str, layout.blank_rows[:20])) or 'none'}."
            + (f" Blank columns inside the table (left out): {', '.join(layout.blank_columns[:20])}."
               if getattr(layout, "blank_columns", None) else "")]
    if layout.raw_rows is not None and layout.sheet_rows is not None and layout.raw_rows < layout.sheet_rows:
        head.append(f"The raw table holds only the first {layout.raw_rows} of the sheet's {layout.sheet_rows} rows "
                    "(INGEST_RAW_TABLE_MAX_ROWS): the last rows below are the last kept, not the sheet's last.")
    if layout.next_table_row:
        head.append(f"Another table starts at row {layout.next_table_row}; it was read as a table of its own.")
    scores = getattr(layout, "scores", None) or {}
    if getattr(layout, "reader", "layout") == "simple":
        head.append("The simple reading (the first full row is the header) fitted the data better than the layout "
                    "reading, so the table was read that way; check the column names against the raw cells.")
    if "layout" in scores and "simple" in scores:
        head.append(f"Both readings were scored (1 at best): layout {scores['layout']}, simple {scores['simple']}"
                    + (" - both low: check the column names against the raw cells." if max(scores.values()) < 0.6
                       else "."))
    body: list[tuple[int, str]] = []
    for row in rows:
        number, cells = row[0], row[1:]
        filled = [f"{_col(i + off)}={_cell(v, s.peek_cell_chars)}" for i, v in enumerate(cells[:shown], 1)
                  if v not in (None, "")]
        more = sum(1 for v in cells[shown:] if v not in (None, ""))
        line = f"row {number}: " + (", ".join(filled) if filled else "(blank)") + (f", ... {more} more cells" if more else "")
        body.append((number, line))
    tail = []
    if width > shown:
        tail.append(f"Columns {_col(shown + 1 + off)}-{_col(width + off)} ({width - shown} more) are not shown; query {raw_name(layout)} "
                    f"(c{shown + 1} to c{width}) for them.")
    tail.append(f"Every cell is also in the SQL table {raw_name(layout)} (columns _row, c1, c2, ... as text; c1 is "
                f"column {_col(1 + off)}; _row is the sheet row). The parsed table {table} is preferred; use {raw_name(layout)} with "
                "TRY_CAST(c3 AS DOUBLE) only when the parsed header or types are wrong.")

    def render(kept: list[tuple[int, str]]) -> str:
        lines, previous = list(head), 0
        for number, line in kept:
            if previous and number > previous + 1:
                lines.append("...")
            previous = number
            lines.append(line)
        return "\n".join(lines + tail)

    text = render(body)
    while count_tokens(text) > s.peek_tokens and len(body) > 3:
        del body[len(body) // 2]  # keep the top (title, header) and the bottom (footnotes) longest
        text = render(body)
    return truncate_tokens(text, s.peek_tokens)


def raw_name(layout) -> str:
    return layout.raw_table or "the raw table"


def peek_tool(ctx: ToolContext) -> StructuredTool:
    async def peek(doc_id: str, table: str) -> tuple[str, list[int]]:
        doc, match = ctx.table(doc_id, table)
        layout = match.layout
        if layout is None or not layout.raw_table:
            return (f"Table {table} came from a PDF or an older upload and has no raw cells; describe_table shows "
                    "its columns."), []

        async def run():
            raw, n, first = layout.raw_table, ctx.s.peek_rows, layout.raw_first_row
            query = (f"SELECT * FROM {raw} WHERE _row < {first + n} OR _row > (SELECT MAX(_row) FROM {raw}) - 8 "
                     "ORDER BY _row")
            result = await ctx.deps.sql.query(doc.doc_id, query, n + 8)
            text = peek_text(match.sheet or table, table, layout, result.columns, result.rows, ctx.s)
            src = Source(0, f"peek:{doc.doc_id}:{table}", doc.doc_id, doc.name, "peek", f"raw cells of table {table}",
                         text, count_tokens(text))
            return ctx.found([ctx.book.add(src)])
        return await ctx.once(_key("peek", doc_id=doc.doc_id, table=table), run)
    return _tool(peek, "Show the raw cells of a table's sheet or CSV before its header was read (first and last rows, "
                       "merged cells, blank rows), to check how the header, notes and footnotes were read.")


def _tool(fn, description: str) -> StructuredTool:
    return StructuredTool.from_function(coroutine=fn, name=fn.__name__, description=description,
                                        response_format="content_and_artifact")


@dataclass(frozen=True)
class ToolEntry:
    build: Callable[[ToolContext], StructuredTool]
    tables_only: bool = False  # offered only when a selected document has tables
    python: bool = False  # offered only when the request may run Python (rag.chat.deep_python)


TOOLS: list[ToolEntry] = [
    ToolEntry(search_tool), ToolEntry(grep_tool), ToolEntry(read_tool),
    ToolEntry(list_tables_tool, True), ToolEntry(describe_table_tool, True), ToolEntry(sql_tool, True),
    ToolEntry(peek_tool, True), ToolEntry(profile_column_tool, True), ToolEntry(run_python_tool, python=True),
]


def _tools(ctx: ToolContext) -> list[StructuredTool]:
    has_tables = any(d.tables for d in ctx.docs)
    python = deep_python(ctx.s, ctx.req)
    return [entry.build(ctx) for entry in TOOLS
            if (has_tables or not entry.tables_only) and (python or not entry.python)]


# --- model calls --------------------------------------------------------------------------------------------


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in TRANSIENT_STATUS
    return isinstance(exc, openai.APIConnectionError | openai.APITimeoutError)


async def invoke(runnable, messages: list[AnyMessage], s: Settings, purpose: str, max_tokens: int) -> AIMessage:
    """One research model call: a busy, failed or timed-out call is retried once after the backoff; a reply cut
    at its token limit once with more room. Both are logged; a second failure raises."""
    try:
        msg = await runnable.ainvoke(messages)
    except Exception as exc:
        if not _transient(exc):
            raise
        log_fields(log, f"{purpose} failed; retrying once", logging.WARNING, error=f"{type(exc).__name__}: {exc}"[:300])
        await asyncio.sleep(s.retry_backoff_s)
        msg = await runnable.ainvoke(messages)
    if (msg.response_metadata or {}).get("finish_reason") == "length" and max_tokens < s.retry_max_tokens:
        room = min(s.retry_max_tokens, max(max_tokens + 1, int(max_tokens * s.retry_tokens_factor)))
        log_fields(log, f"{purpose}: reply cut at its token limit; retrying once with more room", logging.WARNING,
                   max_tokens=max_tokens, retry_tokens=room)
        msg = await runnable.bind(max_tokens=room).ainvoke(messages)
    return msg


def thinking(s: Settings) -> dict:
    """Model settings of a step that thinks (RAG_DEEP_THINK): the thinking template, the model card's thinking
    sampling and RAG_DEEP_THINK_TOKENS of room for the thought and the reply."""
    return {"temperature": s.think_temperature, "top_p": s.think_top_p, "max_tokens": s.deep_think_tokens,
            "extra_body": {"top_k": s.top_k, "chat_template_kwargs": {"enable_thinking": True}}}


def json_object(text: str) -> object:
    """The JSON object in a reply: the whole reply, or (after thinking, without JSON mode) the outermost {...}."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


async def _llm_json(llm: ChatOpenAI, prompt: str, s: Settings, purpose: str, think: bool = False) -> dict:
    """A JSON answer at temperature 0; {} (and the reason in the log) when it cannot be had. With `think` the
    step thinks first, with the thinking sampling and without JSON mode (its grammar would bind the thought too)."""
    try:
        if think:
            msg = await invoke(llm.bind(**thinking(s)), [HumanMessage(prompt)], s, purpose, s.deep_think_tokens)
        else:
            msg = await invoke(llm.bind(response_format={"type": "json_object"}, temperature=0),
                               [HumanMessage(prompt)], s, purpose, s.agent_tokens)
        if (msg.response_metadata or {}).get("finish_reason") == "length":
            raise ValueError("reply cut at its token limit")
        value = json_object(msg.content) if think else json.loads(msg.content)
        return value if isinstance(value, dict) else {}
    except Exception as exc:  # noqa: BLE001 - every failure means "no usable JSON"; the caller says what it does
        log_fields(log, f"{purpose} gave no usable JSON", logging.WARNING, error=f"{type(exc).__name__}: {exc}"[:300])
        return {}


def _strings(value) -> list[str]:
    """A list field of the model's JSON as strings: a lone string is one item, unless it says there is none
    ("none", "nothing", "n/a"); anything else is empty. {"missing": "none"} once became "n; o; n"."""
    if isinstance(value, list):
        return [x for v in value if isinstance(v, str | int | float) for x in _strings(str(v))]
    if isinstance(value, str) and value.strip().lower().strip(".") not in _NONE:
        return [value.strip()]
    return []


def text_documents(docs: list[DocumentInfo]) -> dict[str, str]:
    """The selected documents that are text ({doc_id: name}) for the check of a mixed selection: everything but
    spreadsheets, so a PDF with a table counts as text."""
    return {d.doc_id: d.name for d in docs if d.doc_type != DocType.SPREADSHEET}


_NONE = ("", "none", "nothing", "n/a", "na", "no", "null", "nothing missing")


def _documents(docs: list[DocumentInfo]) -> str:
    lines = []
    for d in docs:
        if d.tables:
            tables = "; ".join(f"table {t.table} ({t.rows} rows: {', '.join(c.sql_name for c in t.columns)[:300]})"
                               for t in d.tables[:5])
            lines.append(f"- {d.name} ({d.doc_type.value}; {tables})")
        else:
            lines.append(f"- {d.name} ({d.doc_type.value})")
    return "\n".join(lines)


def cap_tool_calls(msg: AIMessage, remaining: int) -> AIMessage:
    """One reply may ask for several tools at once; keep only as many as the budget has left (every kept
    call gets its tool result, so the conversation stays valid for the next model call)."""
    if len(msg.tool_calls) <= remaining:
        return msg
    kwargs = {k: v for k, v in msg.additional_kwargs.items() if k != "tool_calls"}
    return msg.model_copy(update={"tool_calls": msg.tool_calls[:remaining], "additional_kwargs": kwargs})


def tools_tokens(tools: list[StructuredTool]) -> int:
    """The tool schemas as the chat template puts them in every research prompt."""
    return count_tokens(json.dumps([convert_to_openai_tool(t) for t in tools])) + TOOLS_PREAMBLE_TOKENS


TextSearch = Callable[[str], Any]  # async (query) -> (rendered results, source numbers), over the text documents


def build_graph(llm: ChatOpenAI, tools: list[StructuredTool], book: SourceBook, s: Settings,
                docs: list[DocumentInfo], text_search: TextSearch | None = None, think: bool = False):
    """The research graph; with `think` the plan and verify steps think (RAG_DEEP_THINK, a request's deep_think)."""
    agent_llm = llm.bind_tools(tools)
    state_of = {"verified_at": -1, "text_searched": False, "stopped": False}
    fixed = tools_tokens(tools)
    text_docs = text_documents(docs)
    mixed = bool(text_docs) and any(d.tables for d in docs)

    async def plan(state: ResearchState) -> dict:
        prompt = _PLAN.format(question=state["question"], documents=_documents(docs))
        subs = _strings((await _llm_json(llm, prompt, s, "research plan", think)).get("sub_questions"))[:4]
        if not subs:
            notice("plan_failed", "Deep search could not split the question into steps, so it researched the "
                   "question as a whole.")
            subs = [state["question"]]
        brief = f"Question: {state['question']}\nSub-questions to research:\n" + "\n".join(f"- {q}" for q in subs)
        system = _SYSTEM.format(sql_rows=s.sql_max_rows)
        return {"sub_questions": subs, "messages": [SystemMessage(system), HumanMessage(brief)]}

    async def agent(state: ResearchState) -> dict:
        remaining = s.agent_max_steps - state["tool_calls"]
        if remaining <= 0:
            state_of["stopped"] = True
            notice("research_budget", f"Deep search used all {s.agent_max_steps} of its tool calls; the answer uses "
                   f"what it found so far ({s.research_budget_from}).")
            return {"messages": [AIMessage("Tool budget reached. DONE")]}
        prompt = fixed + message_tokens(state["messages"])
        if prompt + s.research_retry_tokens + s.prompt_margin_tokens > s.slot_tokens:
            state_of["stopped"] = True  # verify sends no one back: there is no room for another step
            notice("research_context_full", "Deep search found more material than one model call can hold; it "
                   "stopped researching and answers from what it has.", prompt_tokens=prompt, slot_tokens=s.slot_tokens)
            return {"messages": [AIMessage("Research prompt full. DONE")]}
        try:
            msg = await invoke(agent_llm, state["messages"], s, "research step", s.agent_tokens)
        except Exception as exc:  # noqa: BLE001 - the answer step still runs on the sources found so far
            log.exception("research step failed")
            notice("research_step_failed", f"A deep-search step failed ({type(exc).__name__}); the answer uses what "
                   "was found before it.")
            return {"messages": [AIMessage("Research step failed. DONE")]}
        if (msg.response_metadata or {}).get("finish_reason") == "length":
            notice("research_step_cut", "A deep-search step was cut at its length limit even after a retry with more "
                   f"room ({s.research_retry_tokens} tokens; RAG_AGENT_TOKENS, RAG_RETRY_MAX_TOKENS); research goes on "
                   "with what that step managed to ask for.", prompt_tokens=prompt)
        msg = cap_tool_calls(msg, remaining)
        return {"messages": [msg], "tool_calls": state["tool_calls"] + len(msg.tool_calls)}

    def incomplete(missing: list[str]) -> dict:
        if missing:
            notice("research_incomplete", "Deep search could not find everything the question needs; the answer "
                   "says what is missing. Still missing: " + "; ".join(missing))
        return {"missing": []}

    async def cover_text(state: ResearchState) -> dict | None:
        """Mixed selection, no passage of a text document found yet: search the text documents once."""
        if not mixed or text_search is None or state_of["text_searched"]:
            return None
        if any(src.doc_id in text_docs for src in book.sources):
            return None
        state_of["text_searched"] = True
        try:
            rendered, numbers = await text_search(state["question"])
        except Exception as exc:  # noqa: BLE001 - the answer still runs on the table results
            log.exception("text-document search failed")
            notice("text_search_failed", f"Deep search could not search the text documents ({type(exc).__name__}); "
                   "the answer uses the table results only.")
            return None
        if not numbers:
            return None
        names = ", ".join(sorted(set(text_docs.values())))
        if state["rounds"] >= s.agent_verify_rounds or state_of["stopped"]:
            return None  # the passages are in the book and reach the answer; verify still says what is missing
        note = (f"The text documents ({names}) had not been searched yet. A search of them for the question found "
                f"these sources:\n{rendered}\nCompare them with the table results: when they give a different "
                "figure, both must be reported. Research anything still missing, then reply DONE.")
        return {"missing": [f"what the text documents say ({names})"], "rounds": state["rounds"] + 1,
                "messages": [HumanMessage(note)]}

    async def verify(state: ResearchState) -> dict:
        if not book.sources:
            return {"missing": []}
        covered = await cover_text(state)
        if covered is not None:
            return covered
        if len(book.sources) == state_of["verified_at"]:  # nothing new since the last check: its list still holds
            return incomplete(state["missing"])
        state_of["verified_at"] = len(book.sources)
        result = await _llm_json(llm, _verify_prompt(state, book, s, think), s, "research check", think)
        if "missing" not in result:
            notice("research_check_failed", "Deep search could not check its evidence against the question (the "
                   "check gave no usable answer), so the answer may miss something the question needs.")
            return {"missing": []}
        missing = _strings(result.get("missing"))[:3]
        if not missing:
            return {"missing": []}
        if state["rounds"] >= s.agent_verify_rounds or state_of["stopped"]:  # no rounds or no room left
            return incomplete(missing)
        note = "Still missing:\n" + "\n".join(f"- {m}" for m in missing) + "\nResearch these, then reply DONE."
        return {"missing": missing, "rounds": state["rounds"] + 1, "messages": [HumanMessage(note)]}

    def after_agent(state: ResearchState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("verify", verify)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "verify": "verify"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("verify", lambda st: "agent" if st["missing"] else END, {"agent": "agent", END: END})
    return graph.compile()


def _verify_prompt(state: ResearchState, book: SourceBook, s: Settings, think: bool = False) -> str:
    frame = _VERIFY.format(question=state["question"], subs="; ".join(state["sub_questions"]), evidence="")
    output = max(s.deep_think_tokens, s.research_retry_tokens) if think else s.research_retry_tokens
    room = s.slot_tokens - s.prompt_margin_tokens - output - count_tokens(frame) - 50
    evidence = fair_evidence(book.sources, min(s.verify_evidence_tokens, room), s.agent_preview_chars)
    return _VERIFY.format(question=state["question"], subs="; ".join(state["sub_questions"]), evidence=evidence)


def fair_evidence(sources: list[Source], budget: int, preview_chars: int) -> str:
    """Every source rendered as the research steps saw it, within `budget` tokens shared fairly: a source
    shorter than its share gives the rest to the others, a longer one is cut to its share. A late source is
    never dropped for an early long one."""
    if not sources:
        return "No results."
    blocks = [(f"[{x.n}] {x.doc_name} | {x.location}\n", x.text if x.kind in _WHOLE else x.text[:preview_chars])
              for x in sources]
    costs = [count_tokens(h + b) + 2 for h, b in blocks]
    shares = [0] * len(blocks)
    left = max(0, budget)
    for i, j in enumerate(sorted(range(len(blocks)), key=costs.__getitem__)):
        shares[j] = min(costs[j], left // (len(blocks) - i))
        left -= shares[j]
    out = []
    for (head, body), cost, share in zip(blocks, costs, shares, strict=True):
        if share >= cost:
            out.append(head + body)
        else:
            out.append(head + truncate_tokens(body, max(0, share - count_tokens(head) - 6)) + " [cut]")
    return "\n\n".join(out)


def _within_budget(sources: list[Source], budget: int) -> list[Source]:
    """Sources in the order found, while they fit (each with its "[n] name | location" line, counted in the
    model's tokens). Numbers are kept, so [n] in the answer is the same [n] the research steps showed; a source
    that did not fit leaves a gap and citing it is flagged invalid."""
    kept, used = [], 0
    for s in sources:
        cost = source_cost(s)
        if used + cost <= budget:
            kept.append(s)
            used += cost
    return kept


def text_searcher(ctx: ToolContext) -> TextSearch:
    """A search of the selected text documents only (verify's check of a mixed selection); results join the book."""
    text_ids = list(text_documents(ctx.docs))

    async def text_search(query: str) -> tuple[str, list[int]]:
        result = await ctx.deps.retrieval.search(query, text_ids, SearchOptions(top_k=5))
        return ctx.found(ctx.book.add_candidates(result.candidates))
    return text_search


async def answer_from_book(deps: Deps, req: ChatRequest, conv: Conversation, question: str, book: SourceBook,
                           tool_calls: int, timings: dict) -> AsyncIterator[Event]:
    """The end of every deep research, whichever implementation ran it: the sources event from the book (within
    the answer's source budget, with the numbers the research steps showed), then the cited answer."""
    images = figure_slots(deps.s, [(src.doc_id, src.asset) for src in book.sources])
    budget = source_budget(deps.s, question, conv, req.think, images)
    sources = _within_budget(book.sources, budget)
    if len(sources) < len(book.sources):
        notice("sources_dropped", f"{len(book.sources) - len(sources)} of the {len(book.sources)} sources deep search "
               f"found did not fit the answer's prompt ({budget} tokens for sources: RAG_SOURCE_BUDGET, or what the "
               "model's context leaves after the answer, figures and conversation) and were left out.",
               budget=budget)
    yield ChatEvent.SOURCES, {"query": question, "route": "deep", "route_detail": f"{tool_calls} tool calls",
                              "terms": [], "hits": {}, "per_doc": per_document(sources),
                              "sources": [s.card() for s in sources]}
    if not sources:
        for event in nothing_found(timings):
            yield event
        return
    async for event in stream_answer(deps, question, sources, req.think, timings, conv):
        yield event


async def deep_search(deps: Deps, req: ChatRequest, docs: list[DocumentInfo], conv: Conversation,
                      question: str, timings: dict) -> AsyncIterator[Event]:
    """Research tools only read documents; nothing in the graph can write memory, because document text
    is untrusted and could otherwise plant a memory by prompt injection."""
    started = time.perf_counter()
    book = SourceBook()
    ctx = ToolContext(deps, req, docs, book)
    graph = build_graph(deps.graph_llm, _tools(ctx), book, deps.s, docs, text_searcher(ctx),
                        think=deep_think(deps.s, req))
    state: ResearchState = {"messages": [], "question": question, "sub_questions": [], "missing": [], "rounds": 0,
                            "tool_calls": 0}
    pending: dict[str, dict] = {}
    config = {"recursion_limit": deps.s.agent_recursion_limit}
    try:
        async for update in graph.astream(state, stream_mode="updates", config=config):
            for node, change in update.items():
                change = change or {}
                if node == "plan":
                    yield ChatEvent.STEP, {"tool": "plan", "args": {"sub_questions": change.get("sub_questions", [])},
                                           "sources": None, "error": None}
                elif node == "agent":
                    for msg in change.get("messages", []):
                        for call in getattr(msg, "tool_calls", None) or []:
                            pending[call["id"]] = call["args"]
                elif node == "tools":
                    for msg in change.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            error = msg.content if msg.status == "error" else None
                            yield ChatEvent.STEP, {"tool": msg.name, "args": pending.get(msg.tool_call_id, {}),
                                                   "sources": msg.artifact or [], "error": error}
                elif node == "verify" and change.get("missing"):
                    yield ChatEvent.STEP, {"tool": "verify", "args": {"missing": change["missing"]}, "sources": None,
                                           "error": None}
    finally:
        close_sandbox(ctx)  # the request's Python data goes as soon as the research ends
    timings["research_ms"] = round((time.perf_counter() - started) * 1000)
    timings["tool_calls"] = len(pending)
    async for event in answer_from_book(deps, req, conv, question, book, len(pending), timings):
        yield event
