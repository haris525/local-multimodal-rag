"""Query planner for structured data: SQL, SQL-filtered hybrid search, or plain search.

A plan that cannot be used (the model's reply cut off even after its retry with more room, unreadable JSON,
a query that fails after one repair, an unreachable sandbox) falls back to plain search, and the user is
told so with a notice; the fallback is never silent.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from rag_common.contracts import DocumentInfo, TableSchema

from rag.clients import LLMClient, ServiceError, SqlClient
from rag.notices import notice
from rag.prompts import Source, sql_source
from rag.retrieval import RowScope
from rag.tokens import count_tokens
from rag.totals import card_note

log = logging.getLogger("rag.planner")
Route = Literal["sql", "filter", "search"]

_PROMPT = """You route a question over the user's tables. Available tables:

{cards}

Choose one route:
- "sql": the answer needs an exact computation or lookup over columns: totals, counts, averages, minimum or maximum, rankings, or listing every row that matches column values or a literal word or phrase in a text column (use ILIKE). Write one DuckDB SELECT over one table.
- "filter": the question restricts rows by column values AND asks about a concept in free-text columns that a literal phrase cannot capture (for example "complaints about delivery problems"). Give only the boolean condition for the column values (the body of a WHERE clause) over one table marked row_filter=true; the concept is matched by search afterwards.
- "search": anything else, including questions no table can answer.

Examples (illustrative names):
- "Which orders mention battery overheating?" -> "sql": SELECT _row, order_id, notes FROM orders WHERE notes ILIKE '%battery overheating%'
- "Total sales per region?" -> "sql": SELECT region, SUM(amount) FROM orders GROUP BY region
- "Which North orders had shipping problems?" -> "filter" with where: region = 'North' (shipping problems is a concept: late delivery, damaged packaging, and so on)

Rules: use only the listed tables and their sql_name columns. Compare text with the exact values listed in a column's profile when they are listed; otherwise use ILIKE with % wildcards. Select the _row column whenever you return individual rows. In DuckDB `/` always returns a decimal; use FLOOR() for whole-number buckets, and YEAR(date_column) for years.

Answer with JSON only: {{"route": "sql" | "filter" | "search", "doc_id": "...", "table": "...", "sql": "...", "where": "..."}}

Question: {question}"""

_REPAIR = """This DuckDB query failed.
Query: {sql}
Error: {error}
Tables:
{cards}
Return JSON only: {{"sql": "<corrected single SELECT>"}}"""


@dataclass
class Plan:
    route: Route = "search"
    sources: list[Source] = field(default_factory=list)
    scope: RowScope | None = None
    detail: str = ""


def _card(doc: DocumentInfo, t: TableSchema, full: bool) -> str:
    head = (f"doc_id={doc.doc_id} ({doc.name}) table={t.table} source={t.source} rows={t.rows} "
            f"row_filter={'true' if t.row_filter else 'false'}")
    lines = [head]
    if t.title:
        lines.append(f"  title: {t.title[:200]}")
    if t.notes and full:
        lines.append(f"  notes: {' / '.join(t.notes.splitlines())[:300]}")
    if not full:
        return "\n".join(lines + [f"  columns: {', '.join(c.sql_name for c in t.columns)}"])
    for c in t.columns:
        same = re.sub(r"[^0-9a-z]+", "_", c.name.lower()).strip("_") in (c.sql_name, c.sql_name.removeprefix("c_"))
        label = c.header or ("" if same else c.name)
        extra = f" [{label[:120]}]" if label else ""
        unit = f", {c.unit}" if c.unit else ""
        pattern = f"; pattern: {c.pattern}" if c.pattern else ""
        warning = f" WARNING: {c.warning}" if c.warning else ""
        total = f"; {note}" if (note := card_note(c)) else ""  # rag.totals
        lines.append(f"  - {c.sql_name} ({c.type}{unit}){extra}: {c.profile}{pattern}{total}{warning}")
    return "\n".join(lines)


def schema_cards(docs: list[DocumentInfo], question: str = "", budget_tokens: int = 6000, full: bool = True) -> str:
    """Schema cards within a budget in the model's tokens. Tables whose names, columns and values share words
    with the question come first with full profiles (statistics, pattern, warning); the rest fall back to a
    one-line summary."""
    words = set(re.findall(r"[a-z0-9]{3,}", question.lower()))

    def relevance(item: tuple[DocumentInfo, TableSchema]) -> int:
        t = item[1]
        text = (t.source + " " + (t.title or "") + " " +
                " ".join(f"{c.name} {c.header or ''} {c.sql_name} {c.profile}" for c in t.columns)).lower()
        return sum(w in text for w in words)

    tables = sorted(((d, t) for d in docs for t in d.tables), key=relevance, reverse=True)
    lines, used = [], 0
    for i, (doc, table) in enumerate(tables):
        card = _card(doc, table, full)
        if used + count_tokens(card) > budget_tokens:
            card = _card(doc, table, full=False)
        if used + count_tokens(card) > budget_tokens:
            lines.append(f"... and {len(tables) - i} more tables (ask for them by name)")
            break
        lines.append(card)
        used += count_tokens(card) + 1
    return "\n".join(lines)


def _parse(content: str) -> dict:
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(text)


class Planner:
    def __init__(self, llm: LLMClient, sql: SqlClient, sql_budget: int, max_tokens: int = 600,
                 card_tokens: int = 6000, sql_rows: int = 200, filter_rows: int = 5000):
        self.llm = llm
        self.sql = sql
        self.sql_budget = sql_budget
        self.max_tokens = max_tokens
        self.card_tokens = card_tokens
        self.sql_rows = sql_rows
        self.filter_rows = filter_rows

    async def _ask(self, prompt: str, purpose: str) -> dict:
        """The model's JSON; ValueError when its reply is cut (after the client's retry) or not JSON."""
        msg = await self.llm.complete([{"role": "user", "content": prompt}], max_tokens=self.max_tokens,
                                      json_mode=True, purpose=purpose)
        if msg.get("finish_reason") == "length":
            raise ValueError("the plan was cut off at its token limit")
        return _parse(msg.get("content") or "{}")

    async def plan(self, question: str, docs: list[DocumentInfo]) -> Plan:
        tables = {(d.doc_id, t.table): (d, t) for d in docs for t in d.tables}
        if not tables:
            return Plan()
        cards = schema_cards(docs, question, self.card_tokens)
        try:
            decision = await self._ask(_PROMPT.format(cards=cards, question=question), "SQL planner")
        except (ValueError, ServiceError) as exc:
            notice("planner_failed", "The SQL planner could not make a plan for the selected tables "
                   f"({_why(exc)}), so this answer comes from text search over the documents and may miss "
                   "exact figures. Deep search can compute them step by step.", error=str(exc))
            return Plan(detail="planner unavailable")

        route = decision.get("route", "search")
        key = (decision.get("doc_id"), decision.get("table"))
        if route not in ("sql", "filter"):
            notice("planner_search", "No selected table answers this question directly, so the answer comes from "
                   "text search over the documents (their passages and table rows).", level=logging.INFO)
            return Plan(detail="no table route")
        if key not in tables:
            notice("plan_incomplete", f"The SQL plan named a table that is not in the selected documents "
                   f"({key[1]!r}), so this answer comes from text search and may miss exact figures.", table=key[1])
            return Plan(detail="unknown table")
        doc, table = tables[key]
        if route == "filter" and table.row_filter and decision.get("where"):
            return await self._filter(doc, table, decision["where"], cards)
        if decision.get("sql"):
            return await self._sql(doc, table, decision["sql"], cards)
        notice("plan_incomplete", f'The SQL plan for table "{table.table}" came without its query, so this answer '
               "comes from text search and may miss exact figures.", table=table.table)
        return Plan(detail="incomplete plan")

    async def _run(self, doc: DocumentInfo, sql: str, cards: str, max_rows: int):
        """Run once; on a SQL error, ask for one repair and run again."""
        try:
            return sql, await self.sql.query(doc.doc_id, sql, max_rows)
        except ServiceError as exc:
            if exc.status != 400:
                raise
            fixed = (await self._ask(_REPAIR.format(sql=sql, error=exc.detail, cards=cards), "SQL repair")).get("sql") or ""
            return fixed, await self.sql.query(doc.doc_id, fixed, max_rows)

    async def _sql(self, doc: DocumentInfo, table: TableSchema, sql: str, cards: str) -> Plan:
        try:
            sql, result = await self._run(doc, sql, cards, max_rows=self.sql_rows)
        except (ServiceError, ValueError) as exc:
            notice("sql_failed", f'A SQL query on table "{table.table}" failed ({_why(exc)}), so this answer comes '
                   "from text search and may miss exact figures.", error=str(exc))
            return Plan(detail="sql failed")
        source = sql_source(1, doc.doc_id, doc.name, table.table, sql, result, self.sql_budget)
        if result.truncated:
            notice("sql_rows_cut", f"The SQL result was cut at {len(result.rows)} rows; the answer sees only those.")
        return Plan("sql", [source], detail=sql)

    async def _filter(self, doc: DocumentInfo, table: TableSchema, where: str, cards: str) -> Plan:
        sql = f"SELECT _row FROM {table.table} WHERE {where}"
        try:
            sql, result = await self._run(doc, sql, cards, max_rows=self.filter_rows)
            rows = tuple(int(r[0]) for r in result.rows if r and r[0] is not None)  # a repair may select other columns
        except (ServiceError, ValueError, TypeError) as exc:
            notice("filter_failed", f'Filtering the rows of table "{table.table}" failed ({_why(exc)}), so this '
                   "answer searched every row instead.", error=str(exc))
            return Plan(detail="filter failed")
        note = f"{len(rows)}{'+' if result.truncated else ''} rows of table {table.table} match: {where}"
        if not rows:
            source = sql_source(1, doc.doc_id, doc.name, table.table, sql, result, self.sql_budget)
            return Plan("sql", [source], detail=note)
        if result.truncated:
            notice("filter_rows_cut", f"More than {len(rows)} rows matched the filter; search looked at the first "
                   f"{len(rows)} only.")
        return Plan("filter", scope=RowScope(doc.doc_id, table.table, rows), detail=note)


def _why(exc: BaseException) -> str:
    if isinstance(exc, ServiceError):
        if exc.status in (408, 504):
            return "it timed out"
        if exc.status in (502, 503):
            return "a service was busy or unreachable"
        return "the query was not valid"
    text = str(exc)
    return "the plan was cut off at its length limit" if "cut off" in text else "the plan could not be read"
