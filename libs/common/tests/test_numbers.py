"""The one number grammar shared by the layout reader, the column typing and the column profiles."""

import pytest

from rag_common import numbers

MINUS = chr(0x2212)


@pytest.mark.parametrize("text, value", [
    ("+9.5%", 9.5), ("-0.4%", -0.4), (MINUS + "3.4", -3.4), ("1.2e-05", 1.2e-05), ("3.4E+08", 3.4e8), ("2e-3", 0.002),
    ("(5.2%)", -5.2), ("$1,234.50", 1234.5), (chr(0x20AC) + "7", 7.0), ("1,234,567", 1234567.0), ("42", 42.0),
    ("-$35", -35.0), (MINUS + "1,200", -1200.0),
])
def test_numbers_with_signs_exponents_percent_currency_and_groups_parse(text, value):
    assert numbers.is_number(text) and numbers.parse(text) == pytest.approx(value)
    assert float(numbers.normalize(text)) == pytest.approx(value)


@pytest.mark.parametrize("text", ["4411,4412", "1,23", "abc", "", "1.2.3", "e5", "12 km", "ABW", "2024-06-28", "--5",
                                  "+-5", "-" + MINUS + "5", "$--5"])
def test_code_lists_words_units_and_dates_are_not_numbers(text):
    assert numbers.parse(text) is None and not numbers.is_number(text)


def test_one_sign_goes_before_or_after_the_currency_and_exponent_codes_are_codes():
    assert numbers.parse("$-35") == -35.0 and numbers.parse("-$35") == -35.0
    assert numbers.exponent_codes(["1E5", "2E10", "3E7"]) and not numbers.exponent_codes(["1E5", "42", "7"])
    assert not numbers.exponent_codes(["1.2e-05", "0.03", "3.4E-08"]) and not numbers.exponent_codes(["Paris"])


def test_a_decimal_comma_becomes_a_decimal_point_and_other_text_is_kept():
    assert [numbers.decimal_comma(t) for t in ("1,5", "1.234,56", "2.000,00", "-0,75 %", "300,10", "10,750")] == [
        "1.5", "1234.56", "2000.00", "-0.75%", "300.10", "10.750"]
    assert [numbers.decimal_comma(t) for t in ("Paris", "1.5", "A1,2", "")] == ["Paris", "1.5", "A1,2", ""]


def test_the_sql_expression_reads_the_unicode_minus_plus_signs_and_exponents():
    duckdb = pytest.importorskip("duckdb")
    sql = numbers.SQL_CLEAN.replace("col", "v")
    rows = duckdb.sql(f"SELECT {sql} FROM (VALUES ('{MINUS}3.4'), ('+9.5%'), ('1.2e-05'), ('$1,234.50')) t(v)").fetchall()
    assert [r[0] for r in rows] == pytest.approx([-3.4, 9.5, 1.2e-05, 1234.5])
