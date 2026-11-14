"""Total and subtotal rows, and total columns, for the table tools.

A table often mixes detail rows with rows that total them: a label such as "Total" or "All causes", a country
among its states, regions above their states, a parent code above its parts. Adding such a row to the rows it
covers counts them twice, and leaving it out when the question asks for the total misses it. The schema cards
(describe_table, the planner) mark totals by name, and describe_table and profile_column add what the sums show:

    possible totals in column area: 'United States' (the other values add up to 3 times its population, so it
    may total them in 3 levels); 'South', 'West', 'Midwest', 'Northeast' (the 4 largest after it add up to it:
    possibly subtotals)
    codes in column naics_code nest: '441' begins '4411', '4412'; '722' begins '7225' (a code that begins longer
    codes may total them)

Evidence:

- Name (cards, no query): a value whose words say total ("Total", "Totals", "Subtotal", "Grand total",
  "Overall", "Aggregate", or a value that starts with "All" but not "All other", a remainder category), among
  the values a card lists; and a number column whose name or header says total.
- Sums: for a text column with 3 to RAG_TOTALS_LABELS_MAX distinct values, one GROUP BY query sums each measure
  (a number column that is not a year, a date part, a code or an id; at most MEASURES, counts before rates) per
  value. The value with the largest sum is a total when the sum of all other values is 1 to MAX_LEVELS times
  its own, within RAG_TOTALS_TOLERANCE (the others then hold that many complete sets of parts, such as regions
  and states), and it exceeds the second largest by more than 4 times the tolerance (values with equal sums are
  not totals). Then the largest other values that add up to it (2 or more, not all of them) are subtotals, one
  set per level above the last.
- Codes: in the same query's values, a code of 2 to 10 digits that begins at least two longer codes of the
  column is their parent, which may total them (the query runs for codes alone when the card shows one).

At most RAG_TOTALS_COLUMNS text columns of a table are summed (0 turns the sums off); one result fits
TOTALS_TOKENS. A failed query is a line in the result and a totals_check_failed notice; the rest stands.
"""

from __future__ import annotations

import ast
import re

from rag_common.contracts import ColumnSchema, DocumentInfo, TableSchema
from rag_common.http import ServiceError

from rag.config import Settings
from rag.notices import notice
from rag.tokens import truncate_tokens

MEASURES = 3  # number columns summed per query
MAX_LEVELS = 4  # the other values may add up to 1 to this many times the total
LISTED = 8  # values named per finding
TOTALS_TOKENS = 600  # one tool's totals lines
NAME = re.compile(r"\b(total|totals|subtotal|sub-total|grand total|overall|aggregate)\b|^\W*all\b(?!\s+other)", re.I)
COLUMN_NAME = re.compile(r"(^|_)(total|totals|subtotal|overall|aggregate)(_|$)|^all_(?!other)", re.I)
_NOT_MEASURE = re.compile(r"(^|_)(year|yr|month|day|date|week|quarter|hour|code|id|fips|zip|rank|lat|lon|latitude|"
                          r"longitude)(_|$)", re.I)
_RATE = re.compile(r"(^|_)(rate|pct|percent|per|share|ratio|average|avg|mean|median|index)(_|$)", re.I)
_CODE = re.compile(r"[0-9]{2,10}")  # digits only: NAICS, FIPS and similar nested codes
_LISTED_VALUE = re.compile(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\") \(\d+\)")  # a card's "'value' (count)"
HINT = "(leave a total out when adding up the rows or columns it covers; use it when the total is asked for)"


def names_total(value: str) -> bool:
    return bool(NAME.search(" ".join(str(value).split())))


def column_says_total(col: ColumnSchema) -> bool:
    return bool(COLUMN_NAME.search(col.sql_name) or (col.header and NAME.search(col.header)))


def listed_values(col: ColumnSchema) -> list[str]:
    """The values a card shows: every value its profile lists (ingest lists them up to 30 distinct), then the most
    frequent ones."""
    out = []
    for m in _LISTED_VALUE.finditer(col.profile or ""):
        try:
            out.append(str(ast.literal_eval(m.group(1))))
        except (ValueError, SyntaxError):
            continue
    return out + [str(v) for v, _ in (col.top or []) if v is not None and str(v) not in out]


def card_note(col: ColumnSchema) -> str:
    """What a schema card adds to a column's line: the listed values that say total, or that a number column's name
    does. Empty when neither."""
    if col.type == "number":
        return "possibly a total of other columns (its name says so)" if column_says_total(col) else ""
    if col.type != "text":
        return ""
    hits = [v for v in listed_values(col) if names_total(v)]
    return f"possible totals by name: {_names(hits)}" if hits else ""


def _year_like(col: ColumnSchema) -> bool:
    try:
        lo, hi = float(col.min or ""), float(col.max or "")
    except ValueError:
        return False
    return lo.is_integer() and hi.is_integer() and 1000 <= lo <= hi <= 2999


def measures(table: TableSchema) -> list[ColumnSchema]:
    """The number columns worth summing: not a year, date part, code or id; counts and amounts before rates."""
    out = [c for c in table.columns if c.type == "number" and not _NOT_MEASURE.search(c.sql_name)
           and not _year_like(c) and not any("date" in (x or "").lower() for x in (c.pattern, c.warning))]
    return sorted(out, key=lambda c: bool(_RATE.search(c.sql_name)))[:MEASURES]


def label_columns(table: TableSchema, s: Settings) -> list[ColumnSchema]:
    """Text columns whose distinct count (when ingest knew it) allows the sums query: at most RAG_TOTALS_COLUMNS."""
    cols = [c for c in table.columns
            if c.type == "text" and (c.distinct is None or 3 <= c.distinct <= s.totals_labels_max)]
    return cols[:s.totals_columns]


def sums_query(table: str, col: ColumnSchema, ms: list[ColumnSchema], s: Settings) -> str:
    q = f'"{col.sql_name}"'
    sums = "".join(f', SUM("{m.sql_name}")' for m in ms)
    return (f"SELECT CAST({q} AS VARCHAR) AS value, COUNT(*) AS n{sums} FROM {table} WHERE {q} IS NOT NULL "
            f"GROUP BY 1 ORDER BY n DESC, value LIMIT {s.totals_labels_max + 1}")


def _num(x: object) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _names(values: list[str]) -> str:
    shown = ", ".join(repr(v[:40]) for v in values[:LISTED])
    return shown + (f" and {len(values) - LISTED} more" if len(values) > LISTED else "")


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def by_sums(rows: list[list], ms: list[ColumnSchema], tol: float) -> str:
    """The total and its subtotals that the first measure with a match shows, as one finding; "" when none."""
    for j, m in enumerate(ms):
        sums = {str(r[0]): v for r in rows if len(r) > 2 + j and (v := _num(r[2 + j])) is not None}
        if len(sums) < 3:
            continue
        top = max(sums, key=lambda k: sums[k])
        sx = sums[top]
        rest = sum(sums.values()) - sx
        second = max(v for k, v in sums.items() if k != top)
        if sx <= 0 or sx <= second * (1 + 4 * tol):  # a total clearly exceeds each part (equal sums are not totals)
            continue
        levels = next((k for k in range(1, MAX_LEVELS + 1) if _close(rest, k * sx, tol)), 0)
        if not levels:
            continue
        what = (f"its {m.sql_name} equals the sum of all other values" if levels == 1 else
                f"the other values add up to {levels} times its {m.sql_name}, so it may total them in {levels} levels")
        found = [f"{top[:40]!r} ({what})"]
        left = sorted((k for k in sums if k != top), key=lambda k: -sums[k])
        for _ in range(levels - 1):  # one set of subtotals per level above the last
            cum, k = 0.0, 0
            while k < len(left) and cum < sx * (1 - tol):
                cum += sums[left[k]]
                k += 1
            if not (2 <= k < len(left) and _close(cum, sx, tol)):
                break
            found.append(f"{_names(left[:k])} (the {k} largest after it add up to it: possibly subtotals)")
            left = left[k:]
        return "; ".join(found)
    return ""


def by_codes(values: list[str]) -> str:
    """Codes that begin longer codes of the column, as one finding; "" when none."""
    codes = sorted({v.strip() for v in values if v and _CODE.fullmatch(v.strip())})
    if len(codes) < 3:
        return ""
    parents = {}
    for c in codes:
        kids = [d for d in codes if d != c and d.startswith(c)]
        if len(kids) >= 2:
            parents[c] = kids
    if not parents:
        return ""
    shown = "; ".join(f"{p!r} begins {_names(k)}" for p, k in list(parents.items())[:LISTED])
    more = f"; {len(parents) - LISTED} more such codes" if len(parents) > LISTED else ""
    return f"{shown}{more} (a code that begins longer codes may total them)"


def findings(col: ColumnSchema, rows: list[list], ms: list[ColumnSchema], s: Settings) -> list[str]:
    """One column's lines from its sums query rows (value, n, sums...)."""
    out = []
    if len(rows) > s.totals_labels_max:  # more distinct values than the check handles
        return out
    if sums := by_sums(rows, ms, s.totals_tolerance):
        out.append(f"possible totals in column {col.sql_name}: {sums}")
    if codes := by_codes([str(r[0]) for r in rows if r and r[0] is not None]):
        out.append(f"codes in column {col.sql_name} nest: {codes}")
    return out


def _failed(table: str, col: ColumnSchema, exc: ServiceError) -> str:
    notice("totals_check_failed", f"Total rows in column {col.sql_name} of table {table} could not be checked, so the "
           "result was given without that check.", table=table, status=exc.status, error=str(exc)[:300])
    return f"total rows in column {col.sql_name} not checked (the query failed: {str(exc.detail)[:200]})"


async def column_totals(sql, doc: DocumentInfo, table: TableSchema, col: ColumnSchema, s: Settings) -> list[str]:
    """The sums query for one column through the SQL sandbox, as lines; [] when the column has none to show."""
    ms = measures(table)
    if not ms and not any(_CODE.fullmatch(v.strip()) for v in listed_values(col)):
        return []  # nothing to sum, and no codes among the values a card shows: nothing to learn
    try:
        result = await sql.query(doc.doc_id, sums_query(table.table, col, ms, s), s.totals_labels_max + 1)
    except ServiceError as exc:
        return [_failed(table.table, col, exc)]
    return findings(col, result.rows, ms, s)


async def table_totals(sql, doc: DocumentInfo, table: TableSchema, s: Settings) -> str:
    """describe_table's addition for one table, or ""."""
    lines = []
    for col in label_columns(table, s):
        lines += await column_totals(sql, doc, table, col, s)
    if not lines:
        return ""
    return truncate_tokens("\n".join([*lines, HINT]), TOTALS_TOKENS)


async def profile_totals(sql, doc: DocumentInfo, table: TableSchema, col: ColumnSchema, top: list[list],
                         distinct: int, s: Settings) -> str:
    """profile_column's addition for one column, or "": the fresh most frequent values that say total, and for a text
    column with 3 to RAG_TOTALS_LABELS_MAX distinct values (when RAG_TOTALS_COLUMNS is not 0) what the sums show."""
    if col.type == "number":
        return "possibly a total of other columns (its name says so) " + HINT if column_says_total(col) else ""
    if col.type != "text":
        return ""
    named = [str(v) for v, _ in top if v is not None and names_total(v)]
    lines = [f"possible totals by name: {_names(named)}"] if named else []
    if s.totals_columns and 3 <= distinct <= s.totals_labels_max:
        lines += await column_totals(sql, doc, table, col.model_copy(update={"distinct": distinct}), s)
    if not lines:
        return ""
    return truncate_tokens("\n".join([*lines, HINT]), TOTALS_TOKENS)
