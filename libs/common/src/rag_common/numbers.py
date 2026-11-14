"""One grammar for numbers written as text, shared by the sheet layout reader, the column typing and the column
profiles (ingest) and the profile_column tool (rag), so they never disagree about what a number is.

A number: an optional sign ("+", "-" or the Unicode minus U+2212), an optional currency sign (dollar, euro,
pound, yen), digits with real thousands groups ("1,234,567", but not the code list "4411,4412") or plain,
an optional decimal part, an optional exponent ("1.2e-05", "3.4E+08"), an optional "%" and optional
parentheses for a negative ("(5.2%)"). `parse` gives its value (a percentage keeps its number: "5.2%" is
5.2), `normalize` the text float() and DuckDB read.
"""

from __future__ import annotations

import re

MINUS = chr(0x2212)  # the Unicode minus sign, as typed in reports and spreadsheets
CURRENCY = "$" + "".join(map(chr, (0x20AC, 0xA3, 0xA5)))  # dollar, euro, pound, yen
_SIGN = r"[+\-" + MINUS + r"]"
NUMBER = re.compile(  # one sign, before or after a currency sign ("-$35", "$-35"), never two ("--5")
    r"^\(?\s*(" + _SIGN + r"\s*[" + CURRENCY + r"]?|[" + CURRENCY + r"]\s*" + _SIGN + r"?)?\s*"
    r"(\d{1,3}(,\d{3})+|\d+)(\.\d+)?([eE][+\-]?\d{1,3})?\s*%?\s*\)?$"
)
_EXPONENT_CODE = re.compile(r"^\d+[eE]\d+$")  # "1E5", "2E10": part numbers, not 100000
PERCENT = re.compile(r"^\(?\s*[+\-" + MINUS + r"]?\s*\d[\d,]*(\.\d+)?([eE][+\-]?\d{1,3})?\s*%\s*\)?$")
MONEY = re.compile(r"^\(?\s*[+\-" + MINUS + r"]?\s*[" + CURRENCY + r"]\s*\d[\d,]*(\.\d+)?\s*\)?$")
# The SQL expression that reads such text: the Unicode minus, currency, "%", "+" and thousands commas removed.
SQL_CLEAN = ("TRY_CAST(regexp_replace(replace(col, chr(8722), '-'), '[^0-9eE.+-]', '', 'g') AS DOUBLE)")


def is_number(text: str) -> bool:
    return bool(NUMBER.match(text.strip()))


def normalize(text: str) -> str:
    """The number as plain text float() reads: sign, digits, decimal point, exponent ("-5.2" for "(5.2%)")."""
    s = text.strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.replace(MINUS, "-")
    s = re.sub(r"[()" + CURRENCY + r",%\s]", "", s)
    if s.startswith("+"):
        s = s[1:]
    s = s.replace("-+", "-").replace("+-", "-").replace("--", "")
    return f"-{s.lstrip('-')}" if negative and not s.startswith("-") else s


def parse(text: str) -> float | None:
    """The value of a number written as text, or None when it is not one."""
    if not is_number(text):
        return None
    try:
        return float(normalize(text))
    except ValueError:
        return None


def exponent_codes(values: list[str]) -> bool:
    """True when a column's values that read as numbers are all written like "1E5" (digits, E, digits) and
    none is a plain number: codes such as part numbers. Exponents count only in columns that hold other
    numbers or decimal mantissas ("1.2e-05", "0.03")."""
    vals = [str(v).strip() for v in values if str(v).strip()]
    numeric = [v for v in vals if is_number(v)]
    return bool(numeric) and all(_EXPONENT_CODE.match(v) for v in numeric)


def decimal_comma(text: str) -> str:
    """A number written with a decimal comma ("1,5", "1.234,56", "-0,75 %") as one with a decimal point; other
    text unchanged. For files whose fields are separated by ";", where the comma is the decimal mark."""
    s = text.strip()
    m = re.fullmatch(r"([+\-" + MINUS + r"]?)(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(\s*%?)", s)
    if not m or ("," not in s and "." not in s):
        return text
    sign, whole, frac, pct = m.groups()
    whole = whole.replace(".", "")
    return f"{sign}{whole}{'.' + frac if frac else ''}{pct.strip()}"
