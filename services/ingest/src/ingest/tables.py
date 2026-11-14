"""Tables as SQL: DuckDB files and column profiles. Queries run in the separate sql sandbox.

Types: a column is a number when at least 90% of its filled cells parse as numbers (currency, real thousands
separators, %, parenthesised negatives); "not available" markers ("Not Available", "(S)", "--", ...) are NULL
and do not count against it. Whole numbers are BIGINT (so // and % behave; exact beyond 2^53), 8-digit
YYYYMMDD integers and Excel serial day numbers (20000-73050, 1954 to 2099) in a column named like a date are
DATE, cells that are all dates (date objects or ISO "YYYY-MM-DD" text) are DATE (TIMESTAMP when they carry a
time; ISO timestamps with a time zone, such as USGS "2025-01-01T04:39:18.208Z", in UTC). Identifier columns stay
text: a name that ends in an identifier word ("Product ID", "Zip code", "FIPS"), but not a count of them
("Codes issued", "Number of IDs"), and digit strings with a leading zero (postal codes "02134"). A sheet also
gets <table>__raw: every cell of the sheet as text, so a bad header can be worked around in SQL (the
deep-search peek tool points to it).

Profiles: every column gets its empty and distinct counts, the most frequent values with counts (text), min,
max and quartiles (numbers and dates), examples, and the pattern and warning rag_common.profiling detects in
the original cells (dates written as YYYYMMDD numbers, codes with leading zeros, numbers with a unit, mixed
values, a unique key ...). Cells of a number or date column that did not convert are counted in the warning
with examples, never dropped silently. Column warnings also become document warnings (column_warnings).
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path

import duckdb
import pandas as pd

from rag_common import numbers
from rag_common import numbers as rag_numbers  # coerce_column names its parsed values "numbers"
from rag_common.contracts import ColumnSchema, SheetLayout, TableSchema
from rag_common.profiling import date_name, detect, identifier_name, leading_zero_codes

from ingest.chunking import format_value
from ingest.parsers.layout import placeholder

_CURRENCY = numbers.CURRENCY
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?")
_ISO_ZONED = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})")
_PROFILE_SAMPLE = 5000  # original cells the pattern is detected from (spread over the column)
_MAX_DISTINCT_LISTED = 30
_reserved: set[str] | None = None


@dataclass
class TableSource:
    name_hint: str  # sheet name, or "p3_t2" for a PDF table
    source: str  # human label: 'sheet "Sales"' or "page 3, table 2"
    sheet: str | None
    frame: pd.DataFrame  # index = source row numbers
    row_filter: bool  # rows are indexed one per chunk, so the planner may filter by rows
    title: str = ""
    notes: str = ""
    headers: dict[str, str] = field(default_factory=dict)  # column name -> full header text
    units: dict[str, str] = field(default_factory=dict)
    layout: dict | None = None  # SheetLayout fields
    raw: list[list[str | None]] | None = None  # every cell as text -> <table>__raw


def _reserved_words() -> set[str]:
    global _reserved
    if _reserved is None:
        rows = duckdb.sql("SELECT keyword_name FROM duckdb_keywords() WHERE keyword_category = 'reserved'")
        _reserved = {r[0].lower() for r in rows.fetchall()}
    return _reserved


def sql_identifier(name: str, taken: set[str]) -> str:
    """Lowercase snake_case that never needs quoting: letters, digits, underscores."""
    ident = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or "col"
    if ident[0].isdigit():
        ident = f"c_{ident}"
    if ident in _reserved_words():
        ident = f"{ident}_"
    base, n = ident, 2
    while ident in taken:
        ident, n = f"{base}_{n}", n + 1
    taken.add(ident)
    return ident


def is_missing_marker(value: object) -> bool:
    """A cell that stands for a missing value: "not available" markers and placeholders ("TBD", "pending",
    "n.d.", "<10"), stored as NULL in number and date columns and counted."""
    return isinstance(value, str) and placeholder(value)


def _to_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return None if isinstance(value, float) and value != value else float(value)
    return numbers.parse(str(value))  # the shared grammar: signs, U+2212, exponents, currency, %, (negatives)


def _yyyymmdd(n: float) -> date | None:
    try:
        n = int(n)
        y, m, d = n // 10000, n // 100 % 100, n % 100
        return date(y, m, d) if 1800 <= y <= 2100 else None
    except (ValueError, TypeError):
        return None


def is_identifier(name: str, filled: pd.Series | None = None) -> bool:
    """A column of codes rather than quantities: its name ends in an identifier word and does not count them,
    or its cells are digit strings with a leading zero (postal codes typed as text in a CSV), unless they are
    zero-padded small numbers (months "01"-"12", hours "00"-"23")."""
    if identifier_name(name):
        return True
    if filled is None:
        return False
    texts = [str(v).strip() for v in filled if isinstance(v, str)]
    return bool(texts) and leading_zero_codes(texts, name)


def _iso(value: object) -> date | datetime | None:
    text = str(value).strip()
    try:
        if _ISO_DAY.fullmatch(text):
            return date.fromisoformat(text)
        if _ISO_TIME.fullmatch(text):
            return datetime.fromisoformat(text.replace("T", " "))
        if _ISO_ZONED.fullmatch(text):  # stored as UTC without the zone
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None
    return None


def _exact_int(value: object) -> int | None:
    """A whole number without going through float (2**60 + 1 stays exact)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = re.sub(r"[" + _CURRENCY + r",\s]", "", str(value).strip()).replace(numbers.MINUS, "-").lstrip("+")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if not re.fullmatch(r"-?\d+", text):
        return None
    return -int(text) if negative else int(text)


def coerce_column(series: pd.Series, name: str = "") -> pd.Series:
    """Give a column one SQL type (see the module docstring). `name` decides identifier and date columns."""
    present = series.map(lambda v: format_value(v) != "" and not is_missing_marker(v))
    filled = series[present]
    if filled.empty:  # nothing but blanks and placeholders: keep the placeholders as text ("pending", "TBD")
        return series.map(lambda v: format_value(v) or None).astype("string")
    values = series.where(present, None)
    if not all(isinstance(v, datetime | date) for v in filled) and all(isinstance(v, str) for v in filled):
        parsed = filled.map(_iso)
        if parsed.notna().all():  # ISO date text (a CSV): the same as date cells
            values = values.map(lambda v: _iso(v) if v is not None and v == v else None)
            filled = parsed
    if all(isinstance(v, datetime | date) for v in filled):
        if all(not isinstance(v, datetime) or v.time() == time() for v in filled):  # dates without a time
            return values.map(lambda v: (v.date() if isinstance(v, datetime) else v) if v is not None and v == v else None)
        return pd.to_datetime(values, errors="coerce")
    numbers = filled.map(_to_number)
    texts = [v for v in filled if isinstance(v, str)]
    if (numbers.notna().mean() >= 0.9 and not is_identifier(name, filled)
            and not (len(texts) == len(filled) and rag_numbers.exponent_codes(texts))):  # "1E5": codes, not 100000
        values = series.map(lambda v: _to_number(v) if present_value(v) else None)
        whole = values.dropna()
        if len(whole) and (whole % 1 == 0).all():
            if date_name(name) and whole.between(18000101, 21001231).all() and all(_yyyymmdd(x) for x in whole):
                return values.map(lambda x: _yyyymmdd(x) if x is not None and x == x else None)
            # Excel serial day numbers are never converted: a cell formatted as a date arrives as a date, and a
            # number under a date-like header ("Visitors per day", "Day 1 revenue") is a number. The profile
            # names numbers in the day-number range under a date header instead.
            if whole.abs().max() < 2**53:
                return values.astype("Int64")
            exact = series.map(lambda v: _exact_int(v) if present_value(v) else None)
            if exact.dropna().map(lambda x: abs(x) < 2**63).all():
                return exact.astype("Int64")
        return values.astype("Float64")
    return series.map(lambda v: format_value(v) or None).astype("string")


def present_value(value: object) -> bool:
    return format_value(value) != "" and not is_missing_marker(value)


def _kind(series: pd.Series) -> str:
    values = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series) or (len(values) and all(isinstance(v, date) for v in values)):
        return "date"
    return "number" if pd.api.types.is_numeric_dtype(series) else "text"


def _show(value: object) -> str:
    if isinstance(value, float) or (hasattr(value, "dtype") and pd.api.types.is_float_dtype(value.dtype)):
        return format_value(float(value))
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    return format_value(value)[:60]


def _spread(items: list, limit: int) -> list:
    """At most `limit` items taken evenly over the list (head, middle and tail alike)."""
    step = max(1, len(items) // limit)
    return items[::step][:limit]


def column_profile(original: pd.Series, typed: pd.Series, missing: int = 0, name: str = "") -> dict:
    """The ColumnSchema profile fields of one column (type, profile and the optional statistics): counts and
    ranges from the typed values; pattern and warning from the original cells (rag_common.profiling)."""
    kind = _kind(typed)
    values = typed.dropna()
    filled = len(values)
    distinct = int(values.astype(str).nunique()) if filled else 0
    out: dict = {"type": kind, "empty": len(typed) - filled, "distinct": distinct}
    texts = [format_value(v) for v in original if present_value(v)]
    pattern, warning = detect(_spread(texts, _PROFILE_SAMPLE), kind, filled, distinct, name)
    lost = [format_value(v) for v, t in zip(original, typed, strict=True) if present_value(v) and pd.isna(t)]
    if lost and kind != "text":
        shown = ", ".join(repr(v[:30]) for v in lost[:3])
        noun = "numbers" if kind == "number" else "dates"
        warning = "; ".join(x for x in (warning, f"{len(lost)} cells that are not {noun} were stored as NULL "
                                                 f"(e.g. {shown})") if x)
    if not filled:
        profile = "empty"
    elif kind in ("number", "date"):
        ordered = values.sort_values().reset_index(drop=True)
        quarter = [_show(ordered.iloc[round(p * (filled - 1))]) for p in (0.25, 0.5, 0.75)]
        out |= {"min": _show(ordered.iloc[0]), "max": _show(ordered.iloc[-1]), "quantiles": quarter}
        profile = (f"min {out['min']}, max {out['max']}, quartiles {' | '.join(quarter)}" if kind == "number"
                   else f"{out['min']} to {out['max']}, median {quarter[1]}")
    else:
        counts = values.astype(str).value_counts()
        if distinct <= 50 or counts.iloc[0] > 1:
            out["top"] = [(str(v)[:60], int(c)) for v, c in counts.head(10).items()]
        if distinct <= _MAX_DISTINCT_LISTED:  # every value, so the planner can compare with them exactly
            profile = f"{distinct} distinct: " + ", ".join(f"{str(v)[:60]!r} ({c})" for v, c in counts.items())
        else:
            unique = " (unique)" if distinct == filled and filled > 1 else ""
            profile = f"{distinct} distinct{unique}, e.g. " + ", ".join(repr(str(v)[:40]) for v in values.unique()[:3])
    out["examples"] = [str(_show(v))[:60] for v in pd.unique(values.astype(object))[:3]] if filled else []
    empty = out["empty"]
    if empty and filled:
        profile += f"; {empty} empty" + (f" ({missing} marked not available)" if missing else "")
    return out | {"profile": profile, "pattern": pattern or None, "warning": warning or None}


def column_warnings(schemas: list[TableSchema]) -> list[str]:
    """Document warnings for columns whose stored type will mislead a query; one line for a column with the
    same warning in several tables (a workbook of yearly sheets laid out alike)."""
    by: dict[tuple[str, str], list[str]] = {}
    for schema in schemas:
        for c in schema.columns:
            if c.warning:
                by.setdefault((c.name, c.warning), []).append(schema.table)
    out = []
    for (name, warning), tables in by.items():
        where = f'Table "{tables[0]}"' if len(tables) == 1 else f'Tables "{tables[0]}" to "{tables[-1]}" ({len(tables)} tables)'
        out.append(f'{where}, column "{name}": {warning}')
    return out


def _write_raw(con: duckdb.DuckDBPyConnection, name: str, raw: list[list[str | None]], first_row: int = 1) -> None:
    """Every cell as text; _row is the source row number (a second table on a sheet starts below row 1)."""
    width = max((len(r) for r in raw), default=0)
    if not width:
        return
    frame = pd.DataFrame(
        [[first_row + i, *(list(r) + [None] * (width - len(r)))] for i, r in enumerate(raw)],
        columns=["_row", *[f"c{i + 1}" for i in range(width)]],
    ).astype({f"c{i + 1}": "string" for i in range(width)})
    con.register("incoming_raw", frame)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming_raw")
    con.unregister("incoming_raw")


def write_tables(db_path: Path, sources: list[TableSource]) -> list[TableSchema]:
    """Write every table to one DuckDB file (replacing it) and return the schema cards."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    schemas: list[TableSchema] = []
    table_names: set[str] = set()
    with duckdb.connect(str(tmp)) as con:
        for src in sources:
            table = sql_identifier(src.name_hint, table_names)
            taken = {"_row"}
            columns, typed = [], {}
            for position, original in enumerate(src.frame.columns):
                name = str(original)
                ident = sql_identifier(name, taken)
                column = src.frame.iloc[:, position].reset_index(drop=True)  # by position: names may repeat
                typed[ident] = coerce_column(column, src.headers.get(name, name))
                text = _kind(typed[ident]) == "text"  # a text column keeps "pending", "<10", "?" as they are
                missing = 0 if text else int(column.map(is_missing_marker).sum())
                fields = column_profile(column, typed[ident], missing, src.headers.get(name, name))
                if name in (src.layout or {}).get("percent_columns", []) and fields["type"] == "number":
                    fields["pattern"] = ("fractions shown as % in Excel (0.052 is 5.2%): multiply by 100 for a "
                                         "percentage, ROUND(100.0 * col, 2)")
                columns.append(ColumnSchema(name=name, sql_name=ident, header=src.headers.get(name),
                                            unit=src.units.get(name), missing=missing, **fields))
            frame = pd.DataFrame({"_row": list(src.frame.index), **typed})
            con.register("incoming", frame)
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM incoming")
            con.unregister("incoming")
            layout = None
            if src.layout is not None:
                raw_name = None
                if src.raw:
                    raw_name = f"{table}__raw"  # sql_identifier never yields "__", so this cannot collide
                    _write_raw(con, raw_name, src.raw, src.layout.get("raw_first_row", 1))
                layout = SheetLayout(**{**src.layout, "raw_table": raw_name})
            schemas.append(
                TableSchema(
                    table=table,
                    source=src.source,
                    sheet=src.sheet,
                    rows=len(frame),
                    row_filter=src.row_filter,
                    columns=columns,
                    title=src.title or None,
                    notes=src.notes or None,
                    layout=layout,
                )
            )
    tmp.replace(db_path)
    return schemas
