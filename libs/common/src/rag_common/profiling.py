"""What a column holds: a detected pattern and the warning it deserves.

Shared by ingest, which profiles every column from its original cells when it writes the table, and rag,
whose profile_column research tool profiles one column afresh through the sql sandbox. A pattern names how
the values are written (dates as YYYYMMDD numbers, codes with leading zeros, numbers with a unit, a unique
key, ...); a warning says when the stored type will mislead a query (dates or numbers stored as text,
dates stored as numbers, mixed values) and how to work with it in DuckDB.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date

from rag_common import numbers

_YYYYMMDD = re.compile(r"((18|19|20|21)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\.0+)?")
_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_ZONED = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})")
_ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?")
_TEXT_DATES = (  # (pattern, how it is written, DuckDB format)
    (re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"), "D.M.YYYY", "%d.%m.%Y"),
    (re.compile(r"\d{4}-(0[1-9]|1[0-2])"), "YYYY-MM", "%Y-%m"),
    (re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{4}", re.I), "Mon YYYY", "%b %Y"),
    (re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}, \d{4}", re.I), "Mon D, YYYY",
     "%b %d, %Y"),
)
_LEADING_ZERO = re.compile(r"0\d+")
_DIGITS = re.compile(r"\d+")
_SLASH_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/\d{4}( \d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?)?", re.I)
_YEAR_NAME = re.compile(r"\b(year|years|yr|yrs|fy|fiscal|season|vintage|cohort)\b", re.I)
_WITH_UNIT = re.compile(r"-?\d[\d,]*(\.\d+)?\s*([A-Za-z][A-Za-z/.]{0,7})")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{1,23}")
_SERIAL = (20000, 73050)  # Excel day numbers from 1954-10-03 to 2099-12-31
_ID_WORDS = (r"code|codes|id|ids|zip|zipcode|postcode|postal|fips|naics|sic|isbn|issn|phone|ein|ssn|sku|upc|ean|gtin"
             r"|duns|cusip|isin")
_IDENTIFIER_NAME = re.compile(rf"\b({_ID_WORDS})\s*(\([^)]*\))?\s*$", re.I)  # the name ends in the word
_COUNT_NAME = re.compile(r"\b(number|count|counts|total|totals|num|no\.|qty|quantity|issued|assigned|used|sold|per)\b|#",
                         re.I)


_TIME_PART = re.compile(r"\b(month|months|mon|day|days|hour|hours|hr|hh|minute|minutes|min|mm|second|seconds|sec|"
                        r"week|weeks|wk|quarter|qtr)\b", re.I)


def _words(name: str) -> str:
    return re.sub(r"[_\-.]+", " ", name)


_DATE_WORD = re.compile(r"\b(date|day|dt)\b", re.I)
_PER_DATE = re.compile(r"\bper\s+(date|day)\b", re.I)


def date_name(name: str) -> bool:
    """A column named for a date: the whole word date, day or dt ("Begin Date", "BEGIN_DATE", "Day"), never a
    word that only contains it ("Candidates", "Holiday sales", "Days on market") or a rate ("Visitors per day")."""
    words = _words(name)
    return bool(_DATE_WORD.search(words)) and not _PER_DATE.search(words)


def leading_zero_codes(values: list[str], name: str = "") -> bool:
    """Digit strings with a leading zero that are codes (postal codes "02134", bins "0012"), not zero-padded small
    numbers: months "01" to "12" and hours "00" to "23" (two digits at most), or a column named month, day,
    hour, minute, week or quarter."""
    texts = [str(v).strip() for v in values]
    if not texts or not all(_DIGITS.fullmatch(t) for t in texts) or not any(_LEADING_ZERO.fullmatch(t) for t in texts):
        return False
    return any(len(t) > 2 for t in texts) and not _TIME_PART.search(_words(name))


def identifier_name(name: str) -> bool:
    """A column of codes by its name: it ends in an identifier word ("Product ID", "Zip code", "NAICS Code"),
    and does not count them ("Codes issued", "Number of IDs")."""
    return bool(_IDENTIFIER_NAME.search(name)) and not _COUNT_NAME.search(name)


def _share(values: list[str], pattern: re.Pattern) -> float:
    return sum(1 for v in values if pattern.fullmatch(v)) / len(values)


def _valid_yyyymmdd(v: str) -> bool:
    m = _YYYYMMDD.fullmatch(v)
    if not m:
        return False
    try:
        date(int(m.group(1)), int(m.group(3)), int(m.group(4)))
    except ValueError:
        return False
    return True


def _slash_dates(vals: list[str]) -> tuple[str, str, str] | None:
    """(pattern, warning) for dates written with slashes, or None. Day first when any first part is above 12,
    month first when any second part is, ambiguous otherwise."""
    parts = [m for v in vals if (m := _SLASH_DATE.fullmatch(v))]
    if len(parts) < 0.9 * len(vals):
        return None
    first = max(int(m.group(1)) for m in parts)
    second = max(int(m.group(2)) for m in parts)
    if first > 12 and second <= 12:
        return "dates as text (D/M/YYYY)", "dates stored as text: use STRPTIME(col, '%d/%m/%Y')", ""
    if second > 12 and first <= 12:
        return "dates as text (M/D/YYYY)", "dates stored as text: use STRPTIME(col, '%m/%d/%Y')", ""
    return ("dates as text (D/M/YYYY or M/D/YYYY: ambiguous, no part is above 12)",
            "dates stored as text, day and month order unknown: STRPTIME(col, '%d/%m/%Y') or STRPTIME(col, '%m/%d/%Y'); "
            "check with the document", "")


def detect(values: list[str], sql_type: str, filled: int | None = None, distinct: int | None = None,
           name: str = "") -> tuple[str, str]:
    """(pattern, warning) from a column's cell texts (its original cells at ingest, or a sample of its values),
    its SQL type (number | date | text) and its name. `filled` and `distinct` (non-empty and distinct counts of
    the whole column) tell a unique text key; without them no key is claimed. Both strings may be empty."""
    vals = [" ".join(str(v).split()) for v in values if v is not None and str(v).strip()]
    if not vals:
        return "", ""
    if sum(map(_valid_yyyymmdd, vals)) >= 0.95 * len(vals):
        if sql_type == "date":
            return "a DATE (written as YYYYMMDD numbers in the file): compare with DATE '2024-01-01' or YEAR(col)", ""
        return ("dates written as YYYYMMDD numbers",
                "dates stored as numbers: use STRPTIME(CAST(col AS VARCHAR), '%Y%m%d') to compare or group them")
    if _share(vals, _ISO_ZONED) >= 0.95:
        if sql_type == "date":
            return "a TIMESTAMP in UTC (ISO 8601 with a time zone in the file): compare with TIMESTAMP '2025-01-01 00:00:00'", ""
        return "ISO 8601 timestamps as text", "timestamps stored as text: use CAST(col AS TIMESTAMPTZ)"
    if sql_type == "text":
        if _share(vals, _ISO_TIME) >= 0.95 or _share(vals, _ISO_DAY) >= 0.95:
            return "ISO dates as text", "dates stored as text, some not valid dates: use TRY_CAST(col AS DATE)"
        slashed = _slash_dates(vals)
        if slashed:
            return slashed[0], slashed[1]
        for pattern, written, fmt in _TEXT_DATES:
            if _share(vals, pattern) >= 0.9:
                return f"dates as text ({written})", f"dates stored as text: use STRPTIME(col, '{fmt}')"
        formats = [("YYYY-MM-DD", "%Y-%m-%d", _ISO_DAY), ("M/D/YYYY", "%m/%d/%Y", _SLASH_DATE)] + [
            (w, f, p) for p, w, f in _TEXT_DATES]
        used = [(w, f) for w, f, p in formats if any(p.fullmatch(v) for v in vals)]
        if len(used) >= 2 and sum(any(p.fullmatch(v) for _, _, p in formats) for v in vals) >= 0.9 * len(vals):
            listed = ", ".join(repr(f) for _, f in used)
            return (f"dates as text in {len(used)} formats ({', '.join(w for w, _ in used)})",
                    f"dates stored as text in several formats: use TRY_STRPTIME(col, [{listed}])")
    if sql_type == "date":
        nums = [v for v in vals if _DIGITS.fullmatch(v.removesuffix(".0"))]
        if nums and len(nums) == len(vals) and all(_SERIAL[0] <= int(v.removesuffix(".0")) <= _SERIAL[1] for v in nums):
            return "a DATE (Excel day numbers in the file): compare with DATE '2024-01-01'", ""
        return "", ""
    if sql_type == "text" and leading_zero_codes(vals, name):
        return "codes with leading zeros (kept as text)", ""
    if sql_type == "text" and identifier_name(name):
        return "codes or identifiers (text: compare them as text, in quotes)", ""
    if _share(vals, numbers.PERCENT) >= 0.5:
        if sql_type == "number":
            return "percentages (stored without the % sign: 5.2 means 5.2%)", ""
        return "percentages as text", f"percentages stored as text: use {numbers.SQL_CLEAN}"
    if _share(vals, numbers.MONEY) >= 0.5:
        if sql_type == "number":
            return "currency amounts (stored without the currency sign)", ""
        return "currency amounts as text", f"amounts stored as text: use {numbers.SQL_CLEAN}"
    if sql_type == "text":
        units = [m.group(2) for v in vals if (m := _WITH_UNIT.fullmatch(v))]
        common = Counter(units).most_common(1)
        if len(units) >= 0.8 * len(vals) and common and common[0][1] >= 0.8 * len(units):
            return (f"numbers with the unit {common[0][0]!r}",
                    "numbers with a unit stored as text: use TRY_CAST(regexp_extract(col, '-?[0-9.]+') AS DOUBLE)")
        numeric = _share(vals, numbers.NUMBER)
        if 0.5 <= numeric < 1:
            return (f"mixed: {numeric:.0%} numbers, {1 - numeric:.0%} text",
                    f"mostly numbers stored as text: {numbers.SQL_CLEAN} reads the numbers and skips the rest")
    if sql_type == "number":  # distinct numbers are not a key (populations rarely repeat)
        ints = [v for v in vals if _DIGITS.fullmatch(v.removesuffix(".0"))]
        if (ints and len(ints) == len(vals) and _YEAR_NAME.search(_words(name))
                and all(1800 <= int(v.removesuffix(".0")) <= 2100 for v in ints)):
            return "years", ""  # under a year-like name only: elevations, rents and calories fall in 1800-2100 too
        last = _words(name).split()[-1:] if name.strip() else []
        if (ints and len(ints) == len(vals) and last and last[0].lower() in ("date", "day", "dt") and date_name(name)
                and all(_SERIAL[0] <= int(v.removesuffix(".0")) <= _SERIAL[1] for v in ints)):
            return ("numbers in the Excel day-number range under a date header",
                    "these may be Excel dates stored as numbers: DATE '1899-12-30' + CAST(col AS INTEGER) reads them")
        return "", ""
    if filled and distinct and distinct == filled and filled >= 20:
        return "unique values (a key: one row each)", ""
    if (any(numbers.parse(v) is None for v in vals) or numbers.exponent_codes(vals)) and all(
            _CODE.fullmatch(v) for v in vals) and sum(
            any(c.isdigit() for c in v) for v in vals) >= 0.5 * len(vals):
        return "codes or identifiers", ""  # never values that all read as numbers ("1.2e-05")
    return "", ""

