"""Where a table sits in a sheet: title and notes above it, a header of one to four rows, units, section
labels, footnotes below it.

Government and business workbooks rarely start with a clean header row. The census population sheet, for
example, has an accessibility note and a title in rows 1-2, each merged across all columns, a two-row
header (rows 3-4: labels over the years 2020-2025, given as integers), the data, a blank row, one more data
row and six merged footnotes. The old rule (fill every merged range, then take the first row with at least
60% of the widest row's cells) took the merged title as the header.

The rules here, in order:
1. Merged ranges are NOT filled before the layout is found: a merged title or footnote is one cell.
2. A banner row (one filled cell that is a wide merge, long text, or alone in column A) above the table is a
   title or a note, never a header.
3. The body's column types (number, date, text) come from the rows with two or more cells; the first data
   row is the first row that matches them, confirmed by the next row. A row of label numbers (two or more
   integers in a strictly rising or falling run, such as 2020..2025, 2024..2022 or the percentiles
   10..90) is a header row when it is the table's first row or its key text columns are blank (census
   row 4); with a filled key column (a label in column A) it is data. Placeholders ("TBD", "unknown",
   "pending", "n.d.", "<10") count as missing values, so they never make a data row look like a header;
   and rows just above the first data row that hold numbers or dates where the body does are data too
   (a first row with an odd cell, or a date written another way, is never joined into the column names).
4. One-cell text rows directly above the first data row are section labels (the groups of a grouped
   table); above them, after at most one blank row, the header band is the non-blank, non-banner rows (at
   most 4). A one-cell row that is neither a merged group label nor a units row is a note ("Units:
   millions"), never part of a column name. Without a header band, one-cell rows above the table are its
   title and notes, and the first row is the header when it is all text (an all-text table, even with
   long or repeated labels); otherwise the columns are named column_N, with a warning.
5. Header rows are joined per column, merges filled inside the band only; a last band row of "(...)"
   cells is a units row.
6. Long names are shortened: a word prefix or suffix shared by three or more columns is moved to the
   table notes (never leaving a bare number), then a 48-character cap. The full text stays per column.
   Names are unique (a generated suffix never collides with another column's own name).
7. In the data, a one-cell row is a section label when it clearly works as one (written in capitals, ending
   with ":", or part of the groups that start right above the first data row), filled down into a Section
   column; a merged or note-marked one ("Note:", "Source:", "(1)") is a note, and so is long text in a column
   of short labels; below the last full row, long text and rows after a blank row are footnotes. Any other
   one-cell row is a data row with empty cells (a key-only row) and is kept. Vertical merges (row labels)
   are filled down, blank rows are skipped. Section labels and notes inside the data are listed in warnings.
8. A header-like row (text over number columns) after a blank row starts a second table, read on its own
   (read_tables); the sheet's second table becomes "<sheet> (2)". A row that repeats the header (a page
   break) is skipped instead. Rows below the last row that fits the columns are notes.
9. In a workbook, cell styles tell a section label from a key-only data row: when the group labels above
   the data are bold or indented unlike the data rows' first column, only one-cell rows styled like them are
   sections (fills are ignored: banded rows fill every other row).
10. Two readings, the better one kept: every table is also read by the simple reader of the main branch
   (parsers/simple.py), both readings are scored (the share of data rows that fit the column types, how
   plausible the header is, the rows kept against the other reading, column_N names where the other reading
   has real ones), and the better one is kept; a tie keeps this one. Both scores are recorded in the layout
   (peek shows them), and a low score for both is a warning. The raw table and peek show the original cells.
11. A 2-cell key:value row at the left ("Source, Census Bureau", "Units, Thousands", "Release date,
   2024-05-01") is never part of the header band unless it is merged or a units row; a row of label numbers
   is a header when it is the first row filling 60% of the table's width; a one-column sheet takes a first
   text row as its header; tables side by side, separated by a blank column, are read on their own.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd

from rag_common import numbers

from ingest.parsers.simple import simple_read

SCAN_ROWS = 1000  # the first data row must be within this many rows
MAX_HEADER_ROWS = 4
NAME_MAX = 48  # characters in a column name; the full header text is kept separately
RAW_CELL_CHARS = 2000
_DASHES = (chr(0x2014), chr(0x2013))  # em and en dash
NA_TOKENS = frozenset({
    "not available", "na", "n/a", "n.a.", "(na)", "nan", "-", "--", "---", *_DASHES, "..", "...", "x", "(x)",
    "(s)", "(d)", "(z)", "w", "(w)", "*", "(*)", "null", "none", "missing", "not applicable", "withheld",
})
CURRENCY = "$" + "".join(map(chr, (0x20AC, 0xA3, 0xA5)))  # dollar, euro, pound, yen
_NUMBER = numbers.NUMBER  # the shared grammar (rag_common.numbers)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")
_UNIT = re.compile(r"^\((?P<u>[^()]{1,60})\)$")
_INTEGER = re.compile(r"-?\d+(\.0+)?")
_BARE_NUMBER = re.compile(r"[\d.,/ -]+")
_LEAD_STOP = frozenset({"from", "of", "in", "for", "by", "to", "and", "the"})
_NOT_TITLE = ("note", "source", "release", "next update", "[", "(", "*", "table with row headers", "suggested citation")
_PLACEHOLDERS = frozenset({
    "tbd", "tba", "unknown", "pending", "n.d.", "n/d", "nd", "n.r.", "nr", "n/r", "not reported", "suppressed",
    "redacted", "?", "??", "n.a", "n.e.", "ne",
})
_CENSORED = re.compile(r"^[<>]=?\s*-?[" + CURRENCY + r"]?\d[\d,]*(\.\d+)?\s*%?$")  # "<10", ">=100", "< 0.5%"
_UNIT_WORDS = frozenset({
    "c", "f", "k", "%", "mm", "cm", "m", "km", "in", "ft", "mi", "km/h", "mph", "m/s", "kg", "g", "mg", "t", "lb", "lbs",
    "oz", "l", "ml", "gal", "usd", "eur", "gbp", "$", "kwh", "mwh", "gwh", "twh", "kw", "mw", "gw", "w", "h", "hrs",
    "min", "s", "ms", "pct", "percent", "ppm", "db", "hz", "bar", "pa", "kpa", "mb", "gb",
})
_NOTE_MARK = re.compile(r"^(notes?|sources?|footnotes?)\s*\d*\s*[:.)-]|^\*|^\(\d{1,2}\)\s|^\([a-z]\)\s"
                        r"|^page \d+( of \d+)?$|^\(?continued\)?$", re.I)
_LONG_TEXT = 40  # a one-cell row this long reads as a title, note or footnote


def blank(v: object) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def placeholder(text: str) -> bool:
    """A cell that stands for a missing number: a "not available" marker, "TBD", "pending", "<10" ..."""
    s = text.strip().lower()
    return s in NA_TOKENS or s in _PLACEHOLDERS or bool(_CENSORED.match(s))


def kind(v: object) -> str:
    """blank | na | num | date | text"""
    if blank(v):
        return "blank"
    if isinstance(v, bool):
        return "text"
    if isinstance(v, int | float):
        return "na" if isinstance(v, float) and v != v else "num"
    if isinstance(v, datetime | date | time):
        return "date"
    s = str(v).strip()
    if placeholder(s):
        return "na"
    if _NUMBER.match(s):
        return "num"
    if _ISO_DATE.match(s):
        return "date"
    return "text"


def label(v: object) -> str:
    """A cell as display text: integers without '.0', dates as ISO, whitespace collapsed."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, datetime):
        return f"{v:%Y-%m-%d}" if v.time() == time() else v.isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    return " ".join(str(v).split())


Style = Callable[[int, int], object]


@dataclass(frozen=True)
class Merge:
    """A merged range, 0-based and inclusive."""

    r0: int
    r1: int
    c0: int
    c1: int


@dataclass
class TableLayout:
    frame: pd.DataFrame  # index = source row numbers (1-based)
    preamble: list[str] = field(default_factory=list)  # every non-blank line above the header band
    title: str = ""
    notes: list[str] = field(default_factory=list)  # preamble lines other than the title, a column note, footnotes
    headers: dict[str, str] = field(default_factory=dict)  # column name -> full header, when different
    units: dict[str, str] = field(default_factory=dict)
    layout: dict = field(default_factory=dict)  # SheetLayout fields (Excel row numbers)
    raw: list[list[str | None]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # rows not kept as data, columns without a header


class _Grid:
    def __init__(self, rows: list[list[object]], merges: list[Merge]):
        width = max((len(r) for r in rows), default=0)
        self.rows = [list(r) + [None] * (width - len(r)) for r in rows]
        self.width = width
        self.merges = merges
        self.anchor = {(m.r0, m.c0): m for m in merges}
        self.covered: dict[tuple[int, int], Merge] = {}
        for m in merges:
            for r in range(m.r0, m.r1 + 1):
                for c in range(m.c0, m.c1 + 1):
                    if (r, c) != (m.r0, m.c0):
                        self.covered[(r, c)] = m
        self._filled = [[c for c, v in enumerate(row) if not blank(v)] for row in self.rows]
        last = [max(f) + 1 for f in self._filled if len(f) >= 2]
        self.table_width = max(last, default=width)

    def filled(self, r: int) -> list[int]:
        return self._filled[r]

    def wide_merge(self, r: int, c: int) -> bool:
        m = self.anchor.get((r, c))
        return bool(m and m.c1 - m.c0 + 1 >= max(2, 0.5 * self.table_width))

    def long_or_wide(self, r: int) -> bool:
        """A one-cell row that reads as a title, note or footnote wherever it is: a wide merge, long text, or a
        note marker ("Note:", "Source:", "(1) ...")."""
        f = self._filled[r]
        if len(f) != 1:
            return False
        c, v = f[0], self.rows[r][f[0]]
        if self.wide_merge(r, c):
            return True
        text = label(v)
        return kind(v) == "text" and (len(text) >= _LONG_TEXT or bool(_NOTE_MARK.match(text)))

    def is_banner(self, r: int) -> bool:
        """A title, note or footnote outside the data: a long or wide one-cell row, or one alone in column A of
        a table with three or more columns."""
        f = self._filled[r]
        if len(f) != 1:
            return False
        if self.long_or_wide(r):
            return True
        return kind(self.rows[r][f[0]]) == "text" and f[0] == 0 and self.table_width >= 3


@dataclass
class _Body:
    sig: list[str | None]
    key: list[int]


def _signature(g: _Grid) -> _Body:
    """The majority kind per column over the body (rows with 2+ filled cells, skipping the first few), and the
    key columns (filled in at least 90% of those rows)."""
    tw = g.table_width
    cand = [r for r in range(len(g.rows)) if len(g.filled(r)) >= 2 and not g.is_banner(r)]
    body = cand[min(3, len(cand) // 4):] or cand  # a small table skips one row: notes below it must not outvote it
    sig: list[str | None] = []
    key: list[int] = []
    for c in range(tw):
        kinds = [kind(g.rows[r][c]) for r in body]
        solid = [k for k in kinds if k not in ("blank", "na")]
        if body and (len(solid) + kinds.count("na")) >= 0.9 * len(body):
            key.append(c)
        if not solid:
            sig.append(None)
            continue
        top = max(("num", "date", "text"), key=solid.count)
        sig.append(top if solid.count(top) >= 0.6 * len(solid) else "text")
    return _Body(sig, key)


def _cell_kind(g: _Grid, r: int, c: int) -> str:
    """A cell's kind, a cell under a merge from above taking its anchor's."""
    m = g.covered.get((r, c))
    return kind(g.rows[m.r0][m.c0]) if m and m.r0 < r and m.c0 == c else kind(g.rows[r][c])


def _label_numbers(g: _Grid, r: int) -> bool:
    """Two or more integers in a strictly rising or falling run, and no other numbers: 2020..2025, 2024..2022,
    the percentiles 10, 25, 50, 75, 90; or years repeated under groups (2024, 2025, 2024, 2025)."""
    nums = [g.rows[r][c] for c in g.filled(r) if kind(g.rows[r][c]) == "num"]
    if len(nums) < 2 or not all(_INTEGER.fullmatch(label(v).replace(",", "")) for v in nums):
        return False
    values = [int(float(label(v).replace(",", ""))) for v in nums]
    pairs = list(zip(values, values[1:]))
    if all(a < b for a, b in pairs) or all(a > b for a, b in pairs):
        return True
    return all(1800 <= v <= 2100 for v in values) and len(set(values)) < len(values)  # years per group


def _header_numbers(g: _Grid, r: int, body: _Body, first: int) -> bool:
    """A row of label numbers that heads the columns: the table's first row (with a text label or a blank
    corner), or a row whose key text columns are blank or under a merge from above (census row 4). A row of
    rising values with a label in its key column ("A", 1850, 1900, 2000) is data."""
    if not _label_numbers(g, r):
        return False
    text_keys = [c for c in body.key if body.sig[c] == "text"]
    nums = [label(g.rows[r][c]).replace(",", "") for c in g.filled(r) if kind(g.rows[r][c]) == "num"]
    years = all(1800 <= int(float(v)) <= 2100 for v in nums)
    if not text_keys and r != first and years:  # years under groups: their label column is blank in this row,
        text_keys = [c for c in range(g.table_width) if body.sig[c] == "text"]  # so it is not filled in 90%
    if r == first or r == _first_wide(g):  # the table's first row, or the first that fills 60% of its width
        cells = [kind(g.rows[r][c]) for c in range(g.table_width)]
        return "text" in cells or (bool(text_keys) and all(cells[c] == "blank" for c in text_keys))
    return bool(text_keys) and all(kind(g.rows[r][c]) == "blank" for c in text_keys)


_PREAMBLE_KEY = re.compile(
    r"^(data\s+)?(source|sources|units?|unit of measure|release(d)?(\s+date)?|last\s+updated(\s+date)?|updated|"
    r"table\s*\d+[a-z]?|note|notes|prepared(\s+by)?|as\s+of|title|series(\s+id)?|frequency|period|geography|"
    r"survey|contact|copyright|downloaded|retrieved|published|author|dataset|indicator|seasonal adjustment)\b",
    re.I)


_TABLE_NUMBER = re.compile(r"^table\s*[a-z]?\d+(\.\d+)*[a-z]?\s*[:.]?$", re.I)
_YEAR = re.compile(r"^(18|19|20)\d\d$")


def _key_value(g: _Grid, r: int) -> bool:
    """A preamble row at the left of the table, not merged. Two cells (a key and its value): a preamble key
    ("Source", "Units", "Release date", "Table 4", "As of"), a key ending with ":", or, in a table of three or
    more columns, a number or date as its value. Three cells: "Table 4" or a preamble key first, or a lone
    year after two text cells in a table wider than the row ("Table 4, Population by state, 2024"). Two header
    labels at the left ("NAICS Code", "Kind of Business") are not one."""
    f = g.filled(r)
    tw = g.table_width
    if (len(f) not in (2, 3) or f != list(range(len(f))) or tw < len(f) or kind(g.rows[r][0]) != "text"
            or any(g.anchor.get((r, c)) for c in f)):
        return False
    key = label(g.rows[r][0])
    known = bool(_PREAMBLE_KEY.match(key) or _TABLE_NUMBER.match(key)) or key.endswith(":")
    if len(f) == 2:
        return known or (tw >= 3 and kind(g.rows[r][1]) in ("num", "date"))
    if _label_numbers(g, r):
        return False  # "Source, 2023, 2024": a key column over years is a header ("Table 4, ..., 2024" has one year)
    lone_year = kind(g.rows[r][1]) == "text" and bool(_YEAR.match(label(g.rows[r][2])))
    return known or (lone_year and tw > 3)


def _first_wide(g: _Grid) -> int | None:
    """The first row filling at least 60% of the table's width (with two or more cells), preamble rows of a
    key and its value left out ("Units, Thousands" fills 2 of 3 columns)."""
    need = max(2, 0.6 * g.table_width)
    return next((r for r in range(len(g.rows)) if len(g.filled(r)) >= need and not _key_value(g, r)), None)


def _matches(g: _Grid, r: int, body: _Body) -> bool:
    """Row r's cells match the body's column types (merges filled downward count as filled)."""
    if g.is_banner(r):
        return False
    tw = g.table_width
    cells = {c: _cell_kind(g, r, c) for c in range(tw)}
    solid = {c: k for c, k in cells.items() if k not in ("blank", "na")}
    if len(solid) + sum(k == "na" for k in cells.values()) < 2:
        return False
    if body.key and sum(cells[c] != "blank" for c in body.key) < 0.8 * len(body.key):
        return False
    typed = [c for c in solid if body.sig[c] is not None]
    # a text column holds anything (a date or number among labels); number and date columns must match
    if not typed or sum(solid[c] == body.sig[c] or body.sig[c] == "text" for c in typed) < 0.8 * len(typed):
        return False
    if not any(s in ("num", "date") for s in body.sig) or any(solid.get(c) in ("num", "date") for c in typed):
        return True
    return _marker_row(g, r, body)  # its number columns hold only markers ("<5", "*", "(S)", "?")


def _agrees(g: _Grid, r: int, body: _Body, first: int) -> bool:
    """Row r is a data row. A row under a vertical merge that starts in a header row (census A3:A4 over the
    years row) is part of the header; a row label merged down through data rows is not."""
    if not _matches(g, r, body) or _header_numbers(g, r, body, first):
        return False
    anchors = {m.r0 for c in range(g.table_width) if (m := g.covered.get((r, c))) and m.r0 < r}
    return all(_matches(g, a, body) for a in anchors)


def _note_row(g: _Grid, r: int, body: _Body, after_blank: bool) -> bool:
    """A row of two or more cells below the last row that fits the columns, read as a note: it starts with a
    note marker ("Source:"), holds a number where the body has its labels ("1, Includes armed forces"), or
    follows a blank row. Anything else (a last row with an odd value: "n7, see note") stays data."""
    f = g.filled(r)
    head = g.rows[r][f[0]]
    if kind(head) == "text" and _NOTE_MARK.match(label(head)):
        return True
    text_keys = [c for c in body.key if body.sig[c] == "text"]
    return after_blank or any(kind(g.rows[r][c]) == "num" for c in text_keys)


def _marker_row(g: _Grid, r: int, body: _Body) -> bool:
    """A data row whose number or date cells are all placeholders ("<5", "*", "(S)", "Suppressed", "?"): its
    label columns hold text, as in every data row; a header row holds text, never a marker, over its number
    columns."""
    text_keys = [c for c in body.key if body.sig[c] == "text"]
    if not text_keys or not all(_cell_kind(g, r, c) == "text" for c in text_keys):
        return False
    typed = [_cell_kind(g, r, c) for c in range(g.table_width) if body.sig[c] in ("num", "date")]
    filled = [k for k in typed if k != "blank"]
    return bool(filled) and all(k == "na" for k in filled)


def _data_like(g: _Grid, r: int, body: _Body) -> bool:
    """A row of data rather than labels: in the columns the body holds numbers or dates in, at least one cell is
    a number or date of that kind (or all are placeholders under filled labels), and such cells (placeholders
    included) are no fewer than the others."""
    hits = misses = 0
    numbered = _marker_row(g, r, body)
    for c in range(g.table_width):
        if body.sig[c] not in ("num", "date"):
            continue
        k = _cell_kind(g, r, c)
        if k == "blank":
            continue
        if k == body.sig[c]:
            hits, numbered = hits + 1, True
        elif k == "na":
            hits += 1
        else:
            misses += 1
    return numbered and hits >= misses


def _one_cell_label(g: _Grid, r: int, body: _Body) -> str | None:
    """The text of a one-cell row that could be a section label (a short text cell, not a wide merge)."""
    f = g.filled(r)
    if len(f) != 1 or g.long_or_wide(r):
        return None
    c, v = f[0], g.rows[r][f[0]]
    if kind(v) != "text" or (c < len(body.sig) and body.sig[c] not in ("text", None)):
        return None
    return label(v)


def _clearly_section(text: str) -> bool:
    """A label that reads as a group heading: ending with ":", or in capitals without digits ("ASSETS"; a
    product code like "ABC-123" is not one)."""
    bare = re.sub(r"\(\d{1,2}\)|\[\d{1,2}\]", "", text)  # footnote marks: "ADJUSTED(2)"
    letters = [ch for ch in bare if ch.isalpha()]
    caps = len(letters) >= 3 and all(ch.isupper() for ch in letters) and not any(ch.isdigit() for ch in bare)
    return text.endswith(":") or caps


def _is_note(g: _Grid, r: int, body: _Body, multi: list[int], last_multi: int) -> bool:
    """A one-cell row in or below the data that is a note: a wide merge or a note marker anywhere; below the
    last row with two or more cells, long text or a row after a blank row; inside the data, long text unless
    its column holds long text (a comment column)."""
    c, v = g.filled(r)[0], g.rows[r][g.filled(r)[0]]
    if g.wide_merge(r, c) or (kind(v) == "text" and _NOTE_MARK.match(label(v))):
        return True
    if r > last_multi:
        return g.long_or_wide(r) or any(not g.filled(x) for x in range(last_multi + 1, r))
    if not g.long_or_wide(r):
        return False
    if c >= len(body.sig) or body.sig[c] != "text":
        return True
    lengths = sorted(len(label(g.rows[x][c])) for x in multi if kind(g.rows[x][c]) == "text")
    return c == 0 and (not lengths or lengths[len(lengths) // 2] < 30)


def _looks_like_header(g: _Grid, r: int) -> bool:
    """A first row of text cells only heads its columns, even with long, repeated or reappearing labels (an
    all-text table: a survey export, a list of names and comments)."""
    cells = [kind(g.rows[r][c]) for c in g.filled(r)]
    return len(cells) >= 2 and all(k == "text" for k in cells)


def _header_like(g: _Grid, r: int, body: _Body) -> bool:
    """A row of text labels over columns the body holds numbers or dates in: the header of another table."""
    cells = {c: kind(g.rows[r][c]) for c in g.filled(r)}
    return (len(cells) >= 2 and all(k == "text" for k in cells.values())
            and any(c < len(body.sig) and body.sig[c] in ("num", "date") for c in cells))


_CONTINUED = re.compile(r"\s*\((cont\.?|continued)\)\s*$|\s+(cont\.?|continued)$", re.I)


def _texts(g: _Grid, r: int) -> list[str]:
    """A row's cells as comparable text: lower case, without a trailing "(cont.)" or "(continued)"."""
    return [_CONTINUED.sub("", label(g.rows[r][c])).lower() if not blank(g.rows[r][c]) else ""
            for c in range(g.table_width)]


def _next_table(g: _Grid, body: _Body, body_rows: list[int], header: int | None) -> tuple[int | None, list[int]]:
    """(the row where a second table starts below the first, rows that repeat the header). A second table
    starts after a blank row with a header-like row (an optional one-cell title row before it) followed by a
    row with two or more cells; a row that repeats the table's own header is a page-break repeat, skipped."""
    repeats: list[int] = []
    own = _texts(g, header) if header is not None else None
    for pos in range(1, len(body_rows) - 1):
        r, prev = body_rows[pos], body_rows[pos - 1]
        if own is not None and len(g.filled(r)) >= 2 and _texts(g, r) == own:
            repeats.append(r)
            continue
        if not _header_like(g, r, body) or len(g.filled(body_rows[pos + 1])) < 2:
            continue
        if r - prev > 1:
            return r, repeats
        if len(g.filled(prev)) == 1 and pos >= 2 and prev - body_rows[pos - 2] > 1:
            return prev, repeats
    return None, repeats


def _find(g: _Grid, body: _Body) -> tuple[list[int], int, dict[int, str]]:
    """(header band rows, first data row, section labels directly above the data), all 0-based."""
    n = len(g.rows)
    if g.table_width == 1:  # one column: a first text row heads it
        filled = [r for r in range(n) if g.filled(r)]
        if len(filled) >= 2 and kind(g.rows[filled[0]][0]) == "text":
            return [filled[0]], filled[1], {}
    first = next((r for r in range(n) if g.filled(r) and not g.is_banner(r)), 0)
    data = None
    for r in range(first, min(n, SCAN_ROWS)):
        if _agrees(g, r, body, first):
            nxt = next((x for x in range(r + 1, n) if g.filled(x) and not g.is_banner(x)), None)
            if nxt is None or _agrees(g, nxt, body, first) or len(g.filled(nxt)) == 1:
                data = r
                break
    if data is None:  # nothing matches: the first row with two or more cells
        data = next((r for r in range(n) if len(g.filled(r)) >= 2), 0)
    while data - 1 >= first:  # rows just above that hold data (an odd cell: "TBD", a date written otherwise)
        up = data - 1
        if (len(g.filled(up)) < 2 or g.is_banner(up) or _header_numbers(g, up, body, first)
                or not _data_like(g, up, body)):
            break
        data = up
    sections: dict[int, str] = {}
    r = data - 1
    while r >= 0 and (text := _one_cell_label(g, r, body)) is not None:  # groups start right above the data
        if g.anchor.get((r, g.filled(r)[0])):
            break
        sections[r] = text
        r -= 1
    if r >= 1 and not g.filled(r) and len(g.filled(r - 1)) >= 2 and not g.is_banner(r - 1):
        r -= 1  # one blank row between a header and its data
    band: list[int] = []
    while r >= 0 and len(band) < MAX_HEADER_ROWS and g.filled(r) and not g.is_banner(r):
        f = g.filled(r)
        unit_row = not band and _UNIT.match(label(g.rows[r][f[0]]))
        if len(f) == 1 and not g.anchor.get((r, f[0])) and not unit_row:
            break  # a short note above the header ("Units: millions"): preamble, never part of a column name
        if _key_value(g, r) and band:
            break  # "Source, Census Bureau" above the header: preamble, never glued onto the column names
        if (band and f == [0, 1] and g.table_width >= 3 and 0 in g.filled(band[0])
                and len(g.filled(band[0])) >= 0.6 * g.table_width):
            break  # two cells over a full header row ("Agency, Dept of Labor"): a note, not a header row
        band.insert(0, r)
        r -= 1
    if band and all(_key_value(g, x) for x in band) and data < n and (
            _looks_like_header(g, data) or (_label_numbers(g, data) and kind(g.rows[data][0]) == "text" and all(
                _YEAR.match(label(g.rows[data][c]).replace(",", "")) for c in g.filled(data)
                if kind(g.rows[data][c]) == "num"))):  # years, never any two numbers ("Coal, 100, 90" is data)
        nxt = next((x for x in range(data + 1, n) if g.filled(x)), None)  # a key: value row is never the only
        if nxt is not None:                                                # header row: the row under it heads
            band, data, sections = [data], nxt, {}
    if not band:
        sections = {}  # one-cell rows over a table without a header are its title and notes
        if data + 1 < n and _looks_like_header(g, data):
            band, data = [data], data + 1
    return band, data, sections


def _pick_title(lines: list[str]) -> str:
    """The preamble line closest to the table that reads like a title (not a note, source or bracketed remark)."""
    for line in reversed(lines):
        if line.lower().startswith(_NOT_TITLE) or len(line) > 250 or len(line) < 8:
            continue
        return line
    return lines[-1][:250] if lines else ""


def _band_names(g: _Grid, band: list[int]) -> tuple[list[str], dict[int, str]]:
    """Join the header band column by column; merges are filled inside the band only. A last band row whose
    cells are all "(...)" is a units row."""
    tw = g.table_width
    cells = {(r, c): g.rows[r][c] for r in band for c in range(tw)}
    for m in g.merges:
        if m.r0 in band:
            for r in range(m.r0, m.r1 + 1):
                for c in range(m.c0, min(m.c1, tw - 1) + 1):
                    if r in band:
                        cells[(r, c)] = g.rows[m.r0][m.c0]
    rows, units = list(band), {}
    if len(rows) >= 2:
        last = [cells[(rows[-1], c)] for c in range(tw) if not blank(cells[(rows[-1], c)])]
        if last and all(_UNIT.match(label(v)) for v in last):
            units = {c: _UNIT.match(label(cells[(rows[-1], c)])).group("u") for c in range(tw)
                     if not blank(cells[(rows[-1], c)])}
            rows = rows[:-1]
        elif len(last) >= 2 and all(label(v).lower() in _UNIT_WORDS for v in last):  # "C", "mm", "km/h"
            units = {c: label(cells[(rows[-1], c)]) for c in range(tw) if not blank(cells[(rows[-1], c)])}
            rows = rows[:-1]
        else:  # "(USD)", "(n)" with one stray cell in column A ("2, (USD), (n)"): still a units row
            rest = {c: cells[(rows[-1], c)] for c in range(1, tw) if not blank(cells[(rows[-1], c)])}
            if len(rest) >= 2 and all(_UNIT.match(label(v)) for v in rest.values()):
                units = {c: _UNIT.match(label(v)).group("u") for c, v in rest.items()}
                rows = rows[:-1]
    names = []
    for c in range(tw):
        parts: list[str] = []
        for r in rows:
            v = cells[(r, c)]
            if blank(v):
                continue
            text = label(v)
            if parts and text == parts[-1]:
                continue
            if parts and text.lower().startswith(parts[-1].lower()):  # "Order" over "Order ID" -> "Order ID"
                parts[-1] = text
                continue
            parts.append(text)
        names.append(" ".join(parts))
    return names, units


def _common_affix(names: list[str]) -> tuple[list[str], str]:
    """Strip a word prefix/suffix shared by 3+ sibling names. Returns the short names and a note describing the
    pattern, or the names unchanged when that would leave a bare number (census "... 2020" stays whole)."""
    idx = [i for i, n in enumerate(names) if n]
    if len(idx) < 3:
        return names, ""
    words = {i: names[i].split() for i in idx}
    best: tuple[int, int, list[int]] = (0, 0, [])
    for i in idx:
        members = [j for j in idx if words[j][:2] == words[i][:2]]
        if len(members) < 3:
            continue
        ws = [words[j] for j in members]
        p = 0
        while all(len(w) > p + 1 for w in ws) and len({w[p] for w in ws}) == 1:
            p += 1
        s = 0
        while all(len(w) > p + s + 1 for w in ws) and len({w[-1 - s] for w in ws}) == 1:
            s += 1
        if p + s > best[0] + best[1]:
            best = (p, s, members)
    p, s, members = best
    if p + s < 2:
        return names, ""
    out = list(names)
    sample = words[members[0]]
    prefix = " ".join(sample[:p])
    for j in members:
        core = words[j][p:len(words[j]) - s]
        if j == members[0]:
            while len(core) > 1 and core[0].lower() in _LEAD_STOP:
                prefix += " " + core[0]
                core = core[1:]
        else:
            while len(core) > 1 and core[0].lower() in _LEAD_STOP:
                core = core[1:]
        out[j] = " ".join(core).strip(" ,;:") or names[j]
        if _BARE_NUMBER.fullmatch(out[j]):
            return names, ""
    suffix = " ".join(sample[len(sample) - s:]) if s else ""
    comma = "," if s and sample[len(sample) - s - 1].endswith(",") else ""
    pattern = " ".join(x for x in (prefix, "<name>" + comma, suffix) if x)
    span = (f"{members[0] + 1}-{members[-1] + 1}" if members == list(range(members[0], members[-1] + 1))
            else ", ".join(str(j + 1) for j in members))
    return out, f'Columns {span} are "{pattern}"'


def _cap(name: str) -> str:
    if len(name) <= NAME_MAX:
        return name
    cut = name[:NAME_MAX].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:(-") or name[:NAME_MAX]


def _unique(names: list[str]) -> list[str]:
    """Unique names: a repeat gets _2, _3 ..., skipping any suffix another column already has as its own name
    ("A", "A", "A_2" gives "A", "A_3", "A_2")."""
    own = set(names)
    used: set[str] = set()
    out = []
    for n in names:
        candidate, k = n, 1
        while candidate in used or (candidate != n and candidate in own):
            k += 1
            candidate = f"{n}_{k}"
        used.add(candidate)
        out.append(candidate)
    return out


def _letter(c: int) -> str:
    """The sheet column letter of a 0-based column (0 is A, 26 is AA)."""
    s = ""
    c += 1
    while c:
        c, rem = divmod(c - 1, 26)
        s = chr(65 + rem) + s
    return s


def _a1(m: Merge) -> str:
    return f"{_letter(m.c0)}{m.r0 + 1}:{_letter(m.c1)}{m.r1 + 1}"


def _rows_text(rows: list[int]) -> str:
    """'5, 9-12, 40' from sorted row numbers."""
    spans, start = [], None
    for i, r in enumerate(rows):
        if start is None:
            start = r
        if i + 1 == len(rows) or rows[i + 1] != r + 1:
            spans.append(f"{start}" if start == r else f"{start}-{r}")
            start = None
    return ", ".join(spans[:12]) + (", ..." if len(spans) > 12 else "")


def max_tables() -> int:
    """Tables read from one sheet (INGEST_MAX_TABLES_PER_SHEET); the rest stay part of the last, with a warning."""
    try:
        return max(1, int(os.environ.get("INGEST_MAX_TABLES_PER_SHEET", "50")))
    except ValueError:
        return 50


_SAMPLE_ROWS = 2000  # data rows scored per reading (spread over the table)


def _quality(g: _Grid, header: list[int], data: list[int], names: list[str], unit_rows: set[int]) -> dict:
    """How well a reading fits its table: the share of its data rows (label-number rows left out) whose cells
    fit the column types those rows give, where a lone label in a table with number or date columns (a section
    or note row read as data) does not fit; the number of fitting rows; how plausible its header is (1 at
    best); and its share of column_N names."""
    def label_row(r: int) -> bool:  # a row of years with no text label: a header row read as data
        cells = [g.rows[r][c] for c in g.filled(r)]
        return (_label_numbers(g, r) and not any(kind(v) == "text" for v in cells)
                and all(kind(v) != "num" or 1800 <= float(label(v).replace(",", "")) <= 2100 for v in cells))

    rows = [r for r in data if 0 <= r < len(g.rows) and g.filled(r)]
    rows = [r for r in rows[:5] if not label_row(r)] + rows[5:]  # a header row read as data sits at the top
    sample = rows[::max(1, len(rows) // _SAMPLE_ROWS)][:_SAMPLE_ROWS]
    types = []
    for c in range(g.table_width):
        kinds = [k for r in sample if (k := _cell_kind(g, r, c)) not in ("blank", "na")]
        top = max(("num", "date", "text"), key=kinds.count) if kinds else None
        types.append(top if top and kinds.count(top) >= 0.6 * len(kinds) else ("text" if kinds else None))
    fitting = 0
    for r in sample:
        filled = g.filled(r)
        if g.table_width >= 2 and len(filled) == 1 and any(
                t in ("num", "date") for c, t in enumerate(types) if c not in filled):
            continue  # "North" alone in a table of sales: a section label, not a data row
        typed = [(k, types[c]) for c in range(g.table_width) if types[c] and (k := _cell_kind(g, r, c)) not in ("blank", "na")]
        if not typed or sum(k == t or t == "text" for k, t in typed) >= 0.8 * len(typed):
            fitting += 1
    fit = fitting / len(sample) if sample else 0.0
    q = 1.0
    under = rows[0] if rows and header and rows[0] == header[-1] + 1 else None
    if under is not None and blank(g.rows[under][0]) and any(
            types[c] == "num" and _cell_kind(g, under, c) == "text" for c in g.filled(under)):
        q -= 0.3  # the first "data" row has no label and words over numbers (", n, per 100k, 2023"): a header row
        fitting = max(0, fitting - (under in sample))
    for i, r in enumerate(header):
        cells = [kind(g.rows[r][c]) for c in g.filled(r)]
        if r in unit_rows:
            continue
        nums = [label(g.rows[r][c]) for c in g.filled(r) if kind(g.rows[r][c]) == "num"]
        if len(nums) == 1 and _YEAR.match(nums[0].replace(",", "")) and "text" in cells:
            continue  # one year among the labels (", n, per 100k, 2023"): a label, not a number
        if _label_numbers(g, r) and (i > 0 or ("text" in cells and r == _first_wide(g)) or all(
                1800 <= int(float(label(g.rows[r][c]).replace(",", ""))) <= 2100 for c in g.filled(r) if kind(g.rows[r][c]) == "num")):
            continue  # years, or label numbers in the table's first wide row ("Measure, 10, 25, 50"): column labels
        if cells and "text" not in cells:
            q -= 0.8  # a "header" of numbers alone ("1, 2, 3" over more numbers) is a data row
            break
        if any(k in ("num", "na") for k in cells):
            q -= 0.5  # numbers or placeholders in the column names
            break
    if any(len(g.filled(r)) == 1 and g.wide_merge(r, g.filled(r)[0]) for r in header):
        q -= 0.6  # a merged title filled across: the title is not a header
    if any(_key_value(g, r) for r in header):
        q -= 0.4  # "Source, Census Bureau" read as column names
    real = [n for n in names if not n.startswith("column_")]
    words = [n.split() for n in real]
    if len(words) >= 2 and all(len(w) > 2 for w in words) and len({tuple(w[:2]) for w in words}) == 1:
        q -= 0.4  # every name starts with the same title ("Sales by region, 2024 Region", "... Sales")
    base = [re.sub(r"_\d+$", "", n) for n in real]
    if len(set(base)) < len(base):
        q -= 0.1  # repeated labels: a second header row is probably missing
    generated = (len(names) - len(real)) / len(names) if names else 1.0
    return {"fit": fit, "kept": round(fitting / len(sample) * len(rows)) if sample else 0, "header": max(0.0, q),
            "generated": generated, "none": not header}


def _scores(mine: dict, theirs: dict | None) -> dict:
    """Each reading's score: 0.35 x fit + 0.45 x header plausibility + 0.2 x rows kept against the other reading;
    column_N names cost more when the other reading has real names there."""
    readings = {"layout": mine} if theirs is None else {"layout": mine, "simple": theirs}
    most = max(x["kept"] for x in readings.values()) or 1
    out = {}
    for name, x in readings.items():
        other = theirs if name == "layout" else mine
        header = 0.4 if x["none"] else x["header"] - 0.5 * x["generated"]  # no header at all: column_N names
        if other is not None and other["generated"] < x["generated"] and other["header"] >= 0.8:
            header -= 0.3 * (x["generated"] - other["generated"])  # real, plausible names in the other reading
        out[name] = round(0.35 * x["fit"] + 0.45 * max(0.0, header) + 0.2 * x["kept"] / most, 3)
    return out


def _percent_columns(g: _Grid, names: list[str], index: list[int], first_row: int, number_format) -> list[str]:
    """Columns whose numbers are formatted as percentages in the workbook (fractions: 0.052 shows as 5.2%)."""
    if number_format is None or not index:
        return []
    out = []
    sample = index[:200]
    for c, name in enumerate(names):
        cells = [r - first_row for r in sample if kind(g.rows[r - first_row][c]) == "num"]
        fmts = [str(number_format(first_row - 1 + r, c) or "") for r in cells]
        if cells and sum("%" in f for f in fmts) >= 0.8 * len(cells):
            out.append(name)
    return out


def _simple_layout(g: _Grid, frame: pd.DataFrame, header: list[int], preamble: list[str], first_row: int,
                   scores: dict, max_raw: int) -> TableLayout | None:
    """The table as the simple reader of the main branch reads it (it scored better than the layout reading),
    or None when it finds no rows."""
    if frame.empty or not header:
        return None
    frame = frame.set_axis(_unique([str(c) for c in frame.columns]), axis=1)
    title = _pick_title(preamble)
    kept_raw = min(len(g.rows), max_raw)
    layout = {
        "header_rows": header,
        "data_rows": [int(frame.index[0]), int(frame.index[-1])],
        "footnote_rows": [], "section_rows": [],
        "blank_rows": [first_row + r for r in range(header[-1] - first_row + 1, len(g.rows)) if not g.filled(r)][:50],
        "merged": [_a1(m) for m in g.merges][:50],
        "sheet_rows": len(g.rows), "raw_rows": kept_raw, "raw_first_row": first_row, "reader": "simple",
        "scores": scores,
    }
    raw = [[None if blank(v) else label(v)[:RAW_CELL_CHARS] for v in row] for row in g.rows[:kept_raw]]
    return TableLayout(frame=frame, preamble=preamble, title=title, notes=[x for x in preamble if x != title],
                       layout=layout, raw=raw,
                       warnings=[f"the two readings of this table disagree: it was read with the simple rule (the first "
                                 f"full row is the header), which keeps more rows that fit the columns (score "
                                 f"{scores['simple']}, against {scores['layout']} for the layout reading); check the "
                                 f"column names and the first rows against the raw cells (peek)"])


def _column_blocks(rows: list[list[object]]) -> list[tuple[int, int]]:
    """Column ranges [c0, c1) of tables side by side: split at a column blank in every row when the block to its
    right starts with a column of text labels under a text header (its own table), the block on each side has
    two or more columns, and the two blocks do not share their first and last rows. A blank spacer column inside
    one table (numbers on its right, or the same rows on both sides) is no split."""
    width = max((len(r) for r in rows), default=0)
    cell = lambda r, c: r[c] if c < len(r) else None  # noqa: E731
    blank_cols = [c for c in range(width) if all(blank(cell(r, c)) for r in rows)]
    cuts, start = [], 0
    for c in blank_cols:
        right = next((x for x in range(c + 1, width) if x not in blank_cols), None)
        if right is None or c - start < 2 or not any(not blank(cell(r, x)) for r in rows for x in range(right + 1, width)):
            continue
        def span(c0: int, c1: int) -> tuple[int, int] | None:  # first and last row with a cell in columns c0..c1-1
            used = [i for i, r in enumerate(rows) if any(not blank(cell(r, x)) for x in range(c0, c1))]
            return (used[0], used[-1]) if used else None

        left, other = span(start, c), span(right, width)
        if left == other:
            continue  # same header row, same rows: one table with a blank spacer column (Employee .. | Manager ..)
        if (left and other and left[0] == other[0] and other[1] <= left[1] and not any(
                kind(cell(r, x)) in ("num", "date") for r in rows[other[0] + 1:] for x in range(right, width))):
            continue  # text columns on the left block's rows, blank for some of them: attributes, the same table
        labels = [cell(r, right) for r in rows if not blank(cell(r, right))]
        head = next((r for r in rows if any(not blank(cell(r, x)) for x in range(right, width))), None)
        if (len(labels) >= 3 and sum(kind(v) == "text" for v in labels) >= 0.8 * len(labels) and head is not None
                and all(kind(cell(head, x)) == "text" for x in range(right, width) if not blank(cell(head, x)))):
            cuts.append((start, c))
            start = right
    return cuts + [(start, width)] if cuts else [(0, width)]


def read_tables(rows: list[list[object]], merges: list[Merge] | None = None, first_row: int = 1,
                limit: int | None = None, style: Style | None = None, number_format: Style | None = None,
                ) -> list[TableLayout]:
    """Every table in a sheet's cell grid, top to bottom: a second table below the first (after a blank row,
    with its own header) is read on its own rather than as rows of the first, and tables side by side
    (separated by a blank column) are read on their own. `style(row, col)` and `number_format(row, col)`
    (0-based grid positions) give a workbook cell's style key and number format, or None."""
    merges = merges or []
    blocks = _column_blocks(rows)
    width = max((len(r) for r in rows), default=0)
    used = [c for c in range(width) if any(c < len(r) and not blank(r[c]) for r in rows)]
    spacers = [c for c in range(used[0], used[-1]) if c not in used] if used else []
    if len(blocks) == 1:
        tables = _read_stacked(rows, merges, first_row, limit, style, number_format)
        for t in tables:
            t.layout["blank_columns"] = [_letter(c) for c in spacers]
        return tables
    if len(blocks) > 1:
        out: list[TableLayout] = []
        for c0, c1 in blocks:
            part = [list(r[c0:c1]) for r in rows]
            moved = [Merge(m.r0, m.r1, m.c0 - c0, m.c1 - c0) for m in merges if m.c0 >= c0 and m.c1 < c1]
            shift = (lambda f, o: (lambda r, c: f(r, c + o)) if f else None)
            for t in _read_stacked(part, moved, first_row, limit, shift(style, c0), shift(number_format, c0)):
                t.layout["first_col"] = c0
                t.layout["blank_columns"] = [_letter(c) for c in spacers if c0 <= c < c1]
                if not t.frame.empty:
                    out.append(t)
        return out or [TableLayout(pd.DataFrame())]
    return _read_stacked(rows, merges, first_row, limit, style, number_format)


def _read_stacked(rows: list[list[object]], merges: list[Merge], first_row: int, limit: int | None,
                  style: Style | None, number_format: Style | None) -> list[TableLayout]:
    """The tables of one column block, top to bottom."""
    limit = limit or max_tables()
    out: list[TableLayout] = []
    while rows and len(out) < limit:
        nxt = _next_start(rows, merges, first_row, style, number_format)
        if nxt is None or len(out) + 1 == limit:
            table = read_table(rows, merges, first_row, style, number_format)
            if nxt is not None:
                table.warnings.append(f"more than {limit} tables on this sheet: the rows from {nxt} on stay part of "
                                      "this table (INGEST_MAX_TABLES_PER_SHEET)")
            out.append(table)
            break
        cut = nxt - first_row
        first = read_table(rows[:cut], [m for m in merges if m.r1 < cut], first_row, style, number_format)
        first.layout["next_table_row"] = nxt
        out.append(first)
        merges = [Merge(m.r0 - cut, m.r1 - cut, m.c0, m.c1) for m in merges if m.r0 >= cut]
        rows, first_row = rows[cut:], nxt
    return out


def _next_start(rows: list[list[object]], merges: list[Merge], first_row: int, style: Style | None,
                number_format: Style | None) -> int | None:
    """Where the next table of a column block starts (a source row number), or None. A table starts after a
    blank row, so the rows are read up to the start of the block after next (then further, doubling, when no
    table starts there) instead of to the end: reading every remaining row for each table of a sheet with 50
    stacked tables took 45 s. These reads are not scored."""
    starts, gap = [], False
    for i, row in enumerate(rows):
        empty = all(blank(v) for v in row)
        if gap and not empty:
            starts.append(i)
        gap = empty
    ends = starts[1:] + [len(rows)]
    k = 0
    while True:
        end = ends[min(k, len(ends) - 1)]
        window = rows if end >= len(rows) else rows[:end]
        probe = read_table(window, [m for m in merges if m.r1 < end], first_row, style, number_format, score=False)
        nxt = probe.layout.get("next_table_row")
        if nxt is not None or end >= len(rows):
            return nxt
        k = 2 * k + 1


def raw_max_rows() -> int:
    """Rows kept for the <table>__raw SQL table (INGEST_RAW_TABLE_MAX_ROWS). Read from the environment because
    parsing runs in the isolated child process, which inherits it."""
    try:
        return max(0, int(os.environ.get("INGEST_RAW_TABLE_MAX_ROWS", "100000")))
    except ValueError:
        return 100_000


def read_table(rows: list[list[object]], merges: list[Merge] | None = None, first_row: int = 1,
               style: Style | None = None, number_format: Style | None = None, score: bool = True) -> TableLayout:
    """The table in a sheet's cell grid. `first_row` is the source row number of rows[0]; `style(row, col)`
    (0-based positions in the whole grid) gives a workbook cell's style key."""
    g = _Grid(rows, merges or [])
    if not g.rows or not g.width:
        return TableLayout(pd.DataFrame())
    body = _signature(g)
    band, data_start, sections = _find(g, body)
    tw = g.table_width
    full, units = _band_names(g, band) if band else ([""] * tw, {})
    long_names = sum(len(n) > 30 for n in full) >= 3 or max(map(len, full), default=0) > NAME_MAX
    short, column_note = _common_affix(full) if long_names else (full, "")
    if column_note and units and len(set(units.values())) == 1:
        column_note += f" in {next(iter(units.values()))}"
    names = _unique([_cap(n) or f"column_{i + 1}" for i, n in enumerate(short)])

    footnotes: list[str] = []
    footnote_rows: list[int] = []
    for m in g.merges:  # row labels merged down (and narrow merges) inside the data are filled
        if m.r0 >= data_start and not g.long_or_wide(m.r0):
            for r in range(m.r0, m.r1 + 1):
                for c in range(m.c0, m.c1 + 1):
                    if c < len(g.rows[r]):
                        g.rows[r][c] = g.rows[m.r0][m.c0]
    body_rows = [r for r in range(data_start, len(g.rows)) if g.filled(r)]
    next_table, repeats = _next_table(g, body, body_rows, band[-1] if band else None)
    if next_table is not None:
        body_rows = [r for r in body_rows if r < next_table]
    body_rows = [r for r in body_rows if r not in set(repeats)]  # a header repeated after a page break
    multi = [r for r in body_rows if len(g.filled(r)) >= 2]
    fits = [r for r in multi if _data_like(g, r, body) or _matches(g, r, body)]
    last_fit = fits[-1] if fits else (multi[-1] if multi else -1)
    trailing = {r for r in multi if r > last_fit and _note_row(g, r, body, any(not g.filled(x) for x in range(last_fit + 1, r)))}
    grouped = bool(sections)  # the data starts under a group label: one-cell rows like it are groups too
    strict = grouped and all(_clearly_section(t) for t in sections.values())  # "ASSETS": later groups alike

    def style_of(r: int, c: int) -> object:
        return style(first_row - 1 + r, c) if style else None

    label_style = data_style = None
    if grouped:
        top = min(sections)
        col = g.filled(top)[0]
        label_style = style_of(top, col)
        keys = [style_of(r, col) for r in multi[:20]]  # the data rows' usual style in the label's column
        data_style = max(set(keys), key=keys.count) if keys else None
    styled = label_style is not None and label_style != data_style  # bold or indented group labels only
    data_rows, index, section_col = [], [], []
    current = next(iter(sections.values()), None)
    section_rows = [first_row + r for r in sections]
    trailing_notes: list[int] = []
    for pos, r in enumerate(body_rows):
        f = g.filled(r)
        if r in trailing:
            footnotes.append(" ".join(label(g.rows[r][c]) for c in f))
            footnote_rows.append(first_row + r)
            trailing_notes.append(first_row + r)
            continue
        if len(f) == 1:
            if _is_note(g, r, body, multi, last_fit):  # a note inside or below the table
                footnotes.append(label(g.rows[r][f[0]]))
                footnote_rows.append(first_row + r)
                continue
            text = _one_cell_label(g, r, body)
            later = [x for x in body_rows[pos + 1:pos + 4] if len(g.filled(x)) >= 2]
            section = text is not None and bool(later) and (grouped or _clearly_section(text))
            if section and grouped and styled:
                section = style_of(r, f[0]) == label_style
            elif section and strict:
                section = _clearly_section(text)
            if section:
                current = text
                section_rows.append(first_row + r)
                continue
        data_rows.append(g.rows[r][:tw])  # any other row is data, even with one cell filled (a key-only row)
        index.append(first_row + r)
        section_col.append(current)
    frame = pd.DataFrame(data_rows, columns=names, index=pd.Index(index, name="row"))
    if section_rows:
        frame.insert(0, _unique([*names, "Section"])[-1], section_col)
    keep = [c for c in frame.columns if not frame[c].map(blank).all() or not str(c).startswith("column_")]
    frame = frame[keep]

    first_band = band[0] if band else data_start
    preamble = [" ".join(label(v) for v in g.rows[x] if not blank(v)) for x in range(first_band) if g.filled(x)]
    title = _pick_title(preamble)
    notes = [line for line in preamble if line != title] + ([column_note] if column_note else []) + footnotes
    blank_rows = [first_row + r for r in range(data_start, len(g.rows)) if not g.filled(r)]
    kept_raw = min(len(g.rows), raw_max_rows())
    warnings = []
    section_rows = sorted(section_rows)
    if section_rows:
        labels = sorted({v for v in section_col if v} | set(sections.values()))
        warnings.append(f"rows {_rows_text(section_rows)} read as section labels ({', '.join(labels[:6])}"
                        f"{', ...' if len(labels) > 6 else ''}), kept in a Section column")
    inner = [r for r in footnote_rows if r < first_row + last_fit]  # notes below the table are ordinary footnotes
    if inner:
        warnings.append(f"rows {_rows_text(inner)} inside the table read as notes, not data")
    if trailing_notes:
        warnings.append(f"rows {_rows_text(trailing_notes)} below the table do not fit its columns and were read "
                        "as notes")
    if repeats:
        warnings.append(f"rows {_rows_text([first_row + r for r in repeats])} repeat the header (a page break) and "
                        "were skipped")
    odd = [first_row + r for r in band[1:] if not _label_numbers(g, r) and not (units and r == band[-1])
           and any(kind(g.rows[r][c]) in ("num", "na") for c in g.filled(r))]
    if odd:
        warnings.append(f"header rows {_rows_text(odd)} hold numbers or placeholders: check the column names "
                        "(peek shows the raw cells)")
    scores: dict | None = None
    if score:  # skipped by a read that only looks for where the next table starts
        part = rows[:next_table] if next_table is not None else rows  # the same rows as this reading's table
        simple_frame, simple_header, simple_pre = simple_read(part, [m for m in merges or [] if m.r1 < len(part)],
                                                              first_row)
        unit_rows = {band[-1]} if band and units else set()
        kept_names = [str(c) for c in frame.columns if c in set(names)]  # without the Section column, blank column_N
        mine = _quality(g, band, [i - first_row for i in index], kept_names, unit_rows)
        theirs = _quality(g, [h - first_row for h in simple_header], [i - first_row for i in simple_frame.index],
                          [str(c) for c in simple_frame.columns], set()) if not simple_frame.empty else None
        scores = _scores(mine, theirs)
        if theirs is not None and scores["simple"] > scores["layout"] + 0.05 and theirs["kept"] > mine["kept"]:
            simple = _simple_layout(g, simple_frame, simple_header, simple_pre, first_row, scores, raw_max_rows())
            if simple is not None:
                return simple
    if scores and max(scores.values()) < 0.6:
        warnings.append(f"the table's layout is unclear (score {scores['layout']} for this reading, "
                        f"{scores.get('simple', 0)} for the simple one): check the column names (peek shows the raw cells)")
    generated = [c for c in frame.columns if str(c).startswith("column_")]
    if generated:
        warnings.append(f"{len(generated)} columns have no header and are named {generated[0]}...")
    layout = {
        "header_rows": [first_row + r for r in band],
        "data_rows": [index[0], index[-1]] if index else [],
        "footnote_rows": footnote_rows,
        "section_rows": section_rows,
        "blank_rows": blank_rows[:50],
        "merged": [_a1(m) for m in g.merges][:50],
        "sheet_rows": len(g.rows),
        "raw_rows": kept_raw,
        "raw_first_row": first_row,
        "reader": "layout",
        "scores": scores,
        "percent_columns": _percent_columns(g, names, index, first_row, number_format),
    }
    if next_table is not None:
        layout["next_table_row"] = first_row + next_table
    raw = [[None if blank(v) else label(v)[:RAW_CELL_CHARS] for v in row] for row in g.rows[:kept_raw]]
    return TableLayout(
        frame=frame,
        preamble=preamble,
        title=title,
        notes=notes,
        headers={n: f for n, f in zip(names, full, strict=True) if f and f != n and n in frame.columns},
        units={names[c]: u for c, u in units.items() if c < len(names) and names[c] in frame.columns},
        layout=layout,
        raw=raw,
        warnings=warnings,
    )
