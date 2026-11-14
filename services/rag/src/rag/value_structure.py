"""Value structure for the profile tools: the forms in which a text column's values end.

A column whose values mostly hold a separator, such as "12 km NNE of Town, Region", often names a place, an
organization or a category in its last part, and may write it in several forms: the full name in most rows, an
abbreviation or a code in a few. profile_column (rag.agent) adds a summary of that last part, so the model sees
every form before it filters or counts:

    last part after ',' (trimmed; 98% of 4000 values contain it, a value without it counts whole): 57 distinct, most rows first:
    'Westmark' 1203, 'Eastvale' 812, 'Northport' 377, 'EV' 46, ..., 'W.M.' 1, 'NP' 1
    possible abbreviations or codes: 'EV' 46, 'W.M.' 1, 'NP' 1

- The separator is the first of SEPARATORS (comma first) that more than half of the column's non-empty values
  contain. A value without it is its own last part. A column whose every value the profile already lists
  (profile_column's most frequent values; a schema card lists up to LISTED) gets no summary.
- Every distinct last part is listed with its row count, most rows first, while there are at most
  RAG_PROFILE_PARTS_MAX (150) of them; otherwise the PARTS_TOP with the most rows and how many more.
- A part that looks like a short code (one to three letters, or two to six letters and digits with at least one
  digit; mostly upper case; dots ignored, so "W.M." counts) is flagged on the last line, also when it is rare or
  past the list (found among every part, at most CODES_MAX).
- Size: parts are cut at RAG_PEEK_CELL_CHARS (40) characters, and one column's summary fits
  RAG_PROFILE_PARTS_TOKENS (2,000): the list loses its rarest parts first, the codes line stays.
- Queries: the separator counts join profile_column's statistics query; then one query for the summarized
  column. A failed query is a line in the result and a profile_parts_failed notice; the rest of the profile
  stands.
"""

from __future__ import annotations

import re

from rag_common.contracts import ColumnSchema, DocumentInfo
from rag_common.http import ServiceError

from rag.config import Settings
from rag.notices import notice
from rag.tokens import count_tokens, truncate_tokens

SEPARATORS = (",", ";", "|", " - ")  # in this order; " - " with its spaces, so hyphenated words do not count
MOSTLY = 0.5  # the share of non-empty values that must contain the separator (more than this)
PARTS_TOP = 30  # parts listed when there are more than RAG_PROFILE_PARTS_MAX
CODES_MAX = 50  # codes flagged, at most
LISTED = 30  # a schema card lists every value of a text column with at most this many (ingest.tables)
PART_CHARS = 100  # a part as the query returns it (shown cut at RAG_PEEK_CELL_CHARS)
_CODE = re.compile(r"[A-Za-z]{1,3}|(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{2,6}")
_CANDIDATE = "length(part) <= 12 AND regexp_full_match(part, '[A-Za-z0-9.]+')"  # every code passes (a superset)


def looks_like_code(part: str) -> bool:
    """One to three letters, or two to six letters and digits with at least one digit, mostly upper case; dots
    are ignored ("W.M.", "E.V.")."""
    bare = part.replace(".", "")
    if len(part) > 12 or not _CODE.fullmatch(bare):
        return False
    letters = [c for c in bare if c.isalpha()]
    return 2 * sum(c.isupper() for c in letters) > len(letters)


def _lit(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def separator_counts(q: str) -> list[str]:
    """SQL aggregates: how many values of the column `q` (a quoted name from the schema) contain each separator."""
    return [f"COUNT(*) FILTER (WHERE contains(CAST({q} AS VARCHAR), {_lit(sep)}))" for sep in SEPARATORS]


def pick(counts: list, filled: int) -> tuple[str, int] | None:
    """The first separator that more than half of the `filled` values contain, with its count; None otherwise."""
    for sep, n in zip(SEPARATORS, counts):
        if filled and _int(n) > MOSTLY * filled:
            return sep, _int(n)
    return None


def parts_query(q: str, table: str, sep: str, s: Settings) -> str:
    """Each distinct last part with its row count and the number of distinct parts: the RAG_PROFILE_PARTS_MAX with
    the most rows, then the code candidates past them (CODES_MAX at most)."""
    part = f"trim(string_split(CAST({q} AS VARCHAR), {_lit(sep)})[-1])"
    return (f"WITH p AS (SELECT {part} AS part FROM {table} WHERE {q} IS NOT NULL), "
            "g AS (SELECT part, COUNT(*) AS n FROM p GROUP BY part) "
            f"SELECT left(part, {PART_CHARS}) AS value, n, COUNT(*) OVER () AS parts FROM g "
            f"QUALIFY row_number() OVER (ORDER BY n DESC, part) <= {s.profile_parts_max} OR ({_CANDIDATE}) "
            f"ORDER BY n DESC, part LIMIT {s.profile_parts_max + CODES_MAX}")


def parts_lines(sep: str, with_sep: int, filled: int, rows: list[list], s: Settings, budget: int) -> list[str]:
    """One column's summary within `budget` tokens, from parts_query's rows (value, n, distinct parts)."""
    found = [("" if r[0] is None else str(r[0]), _int(r[1]), _int(r[2])) for r in rows if len(r) >= 3]
    total = found[0][2] if found else 0
    full = total <= s.profile_parts_max
    ranked = found[:min(total, s.profile_parts_max)]
    listed = ranked if full else ranked[:PARTS_TOP]
    codes = [(v, n) for v, n, _ in found if looks_like_code(v)]
    more_codes = len(codes) > CODES_MAX or len(found) - len(ranked) >= CODES_MAX  # the query's limit was reached
    cut = s.peek_cell_chars
    entries = [f"{v[:cut]!r} {n}" for v, n, _ in listed]
    whole = ", a value without it counts whole" if with_sep < filled else ""
    share = f"{100 * with_sep // filled}%" if filled else "0%"  # rounded down: 100% only when every value has it
    head = (f"last part after {sep!r} (trimmed; {share} of {filled} values contain it{whole}): {total} distinct, "
            "most rows first:")
    flagged = ("possible abbreviations or codes: " + ", ".join(f"{v!r} {n}" for v, n in codes[:CODES_MAX])
               + (", and possibly more" if more_codes else "")) if codes else ""

    def render(k: int) -> list[str]:
        lines = [head] + ([", ".join(entries[:k])] if k else [])
        if total > k:
            why = (f"over RAG_PROFILE_PARTS_MAX ({s.profile_parts_max}); SQL can count them" if not full
                   else "cut to fit RAG_PROFILE_PARTS_TOKENS")
            lines.append(f"... {total - k} more distinct last parts ({why})")
        return lines + ([flagged] if flagged else [])

    k = len(entries)
    lines = render(k)
    while k and count_tokens("\n".join(lines)) > budget:
        over, dropped = count_tokens("\n".join(lines)) - budget, 0
        while k and dropped < over:  # the rarest parts go first
            k -= 1
            dropped += count_tokens(entries[k]) + 1
        lines = render(k)
    text = "\n".join(lines)
    if count_tokens(text) <= budget:
        return lines
    # over even without the list (a small budget): the codes line is kept
    # whole when it fits, and the lines before it are cut
    room = budget - count_tokens(flagged) - 1
    if flagged and room > 0:
        return [*truncate_tokens("\n".join(lines[:-1]), room).split("\n"), flagged]
    return truncate_tokens(text, budget).split("\n")


def _failed(table: str, columns: list[str], exc: ServiceError) -> str:
    named = ", ".join(columns)
    notice("profile_parts_failed", f"The value structure of column(s) {named} of table {table} could not be read, so "
           "the profile was given without it.", table=table, status=exc.status, error=str(exc)[:300])
    return f"value structure of {named} not available (the query failed: {str(exc.detail)[:200]})"


async def column_parts(sql, doc: DocumentInfo, table: str, col: ColumnSchema, found: tuple[str, int], filled: int,
                       s: Settings, budget: int) -> list[str]:
    """One column's summary: the parts query through the SQL sandbox, rendered within `budget` tokens."""
    sep, with_sep = found
    try:
        result = await sql.query(doc.doc_id, parts_query(f'"{col.sql_name}"', table, sep, s),
                                 s.profile_parts_max + CODES_MAX)
    except ServiceError as exc:
        return [_failed(table, [col.sql_name], exc)]
    return parts_lines(sep, with_sep, filled, result.rows, s, budget)


async def profile_structure(sql, doc: DocumentInfo, table: str, col: ColumnSchema, stats: list, shown: int,
                            s: Settings) -> str:
    """profile_column's addition, or "": `stats` is its statistics row (rows, filled, distinct, min, max, spread,
    then the separator counts), `shown` the number of most frequent values it lists."""
    stats = list(stats)
    filled, distinct = (_int(stats[1]), _int(stats[2])) if len(stats) > 2 else (0, 0)
    found = pick(stats[6:], filled) if col.type == "text" else None
    if found is None or distinct <= shown:  # no separator in most values, or every value is listed already
        return ""
    return "\n".join(await column_parts(sql, doc, table, col, found, filled, s, s.profile_parts_tokens))
