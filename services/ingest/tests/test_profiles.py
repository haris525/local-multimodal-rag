"""Column profiles written at ingest: counts, top values, ranges and quartiles, examples, the detected pattern
and a warning (also a document warning) when the stored type would mislead a query. Real samples first, then
adversarial sheets."""

from pathlib import Path

import pytest

from ingest.parsers.sheets import read_csv, read_xlsx
from ingest.tables import TableSource, column_warnings, write_tables

DATA = Path(__file__).resolve().parents[3] / "examples" / "documents" / "data"


def sample(name: str) -> Path:
    path = next(DATA.rglob(name), None)
    if path is None:
        pytest.skip(f"sample file {name} not in this checkout")
    return path


def schema_of(tmp_path, doc):
    sheet = doc.sheets[0]
    [schema] = write_tables(tmp_path / "t.duckdb", [TableSource(sheet.name, "x", sheet.name, sheet.frame, True,
                                                                headers=sheet.headers, units=sheet.units)])
    return schema, {c.sql_name: c for c in schema.columns}


def test_noaa_begin_date_was_written_as_yyyymmdd(tmp_path):
    schema, cols = schema_of(tmp_path, read_csv(sample("noaa-us-billion-dollar-disasters-1980-2024.csv").read_bytes(), "n.csv"))
    begin = cols["begin_date"]
    assert begin.type == "date" and begin.pattern == "a DATE (written as YYYYMMDD numbers in the file): compare with DATE '2024-01-01' or YEAR(col)"
    assert (begin.min, begin.max, begin.empty) == ("1980-04-10", "2024-10-09", 0) and len(begin.quantiles) == 3
    assert cols["disaster"].top[0] == ("Severe Storm", 203) and cols["disaster"].distinct == 7
    assert "'Severe Storm' (203)" in cols["disaster"].profile and not column_warnings([schema])


def test_census_population_numbers_have_ranges_and_no_pattern(tmp_path):
    _, cols = schema_of(tmp_path, read_xlsx(sample("census-state-population-estimates-2020-2025.xlsx")))
    pop = cols["population_estimate_as_of_july_1_2025"]
    assert pop.type == "number" and (pop.min, pop.max) == ("588753", "341784857")
    assert pop.quantiles == ["2029733", "5193088", "10127884"] and pop.pattern is None and pop.warning is None
    assert cols["geographic_area"].pattern == "unique values (a key: one row each)"


def test_retail_naics_codes_are_codes_not_mixed_numbers(tmp_path):
    schema, cols = schema_of(tmp_path, read_xlsx(sample("census-monthly-retail-sales-1992-2026.xlsx")))
    naics = cols["naics_code"]
    assert naics.type == "text" and naics.pattern.startswith("codes or identifiers") and naics.warning is None
    assert naics.empty == 14 and "4411,4412" in [v for v, _ in naics.top] + naics.examples
    assert "9 empty (9 marked not available)" in cols["jan_2026"].profile


def test_usgs_time_strings_become_utc_timestamps_and_the_id_is_a_code(tmp_path):
    _, cols = schema_of(tmp_path, read_csv(sample("usgs-earthquakes-m5-plus-2025.csv").read_bytes(), "u.csv"))
    time = cols["time"]
    assert time.type == "date" and time.pattern == "a TIMESTAMP in UTC (ISO 8601 with a time zone in the file): compare with TIMESTAMP '2025-01-01 00:00:00'"
    assert time.min == "2025-01-01 04:39:18"  # 2025-01-01T04:39:18.208Z, stored in UTC
    assert cols["id"].pattern.startswith("codes or identifiers") and cols["id"].distinct == 2130
    assert cols["magtype"].top[0] == ("mww", 1161) and "6 empty" in cols["nst"].profile


@pytest.mark.parametrize("text, column, pattern, warning", [
    ("Town,Postal\nBoston,02134\nCambridge,02139\nNYC,10001\n", "postal", "codes with leading zeros", None),
    ("Item,Change\nA,(5.2%)\nB,3.1%\nC,-\nD,4.0%\n", "change", "percentages (stored without the % sign", None),
    ("Event,Start\nx,20240105\ny,20240317\nz,20231231\n", "start", "dates written as YYYYMMDD numbers", "dates stored as numbers"),
    ("Trail,Length\nA,12 km\nB,5.5 km\nC,100 km\nD,3 km\n", "length", "numbers with the unit 'km'", "numbers with a unit"),
    ("Name,Score\n" + "".join(f"n{i},{i}\n" for i in range(7)) + "n7,see note\nn8,absent\n", "score", "mixed: 78% numbers",
     "mostly numbers stored as text"),
])
def test_adversarial_columns_get_their_pattern_and_warning(tmp_path, text, column, pattern, warning):
    schema, cols = schema_of(tmp_path, read_csv(text.encode(), "a.csv"))
    col = cols[column]
    assert col.pattern.startswith(pattern)
    if warning:
        assert col.warning.startswith(warning)
        assert column_warnings([schema]) == [f'Table "{schema.table}", column "{col.name}": {col.warning}']
    else:
        assert col.warning is None


def test_cells_that_did_not_convert_are_counted_in_a_warning(tmp_path):
    text = "Name,Amount\n" + "".join(f"n{i},{i * 10}\n" for i in range(20)) + "n20,see invoice\n"
    schema, cols = schema_of(tmp_path, read_csv(text.encode(), "a.csv"))
    amount = cols["amount"]
    assert amount.type == "number" and amount.empty == 1
    assert amount.warning == "1 cells that are not numbers were stored as NULL (e.g. 'see invoice')"
    assert column_warnings([schema])[0].endswith(amount.warning)


def test_placeholders_are_missing_values_not_lost_cells(tmp_path):
    text = "Name,Amount\n" + "".join(f"n{i},{i * 10}\n" for i in range(20)) + "n20,TBD\nn21,<10\nn22,pending\n"
    _, cols = schema_of(tmp_path, read_csv(text.encode(), "a.csv"))
    amount = cols["amount"]
    assert amount.type == "number" and amount.empty == 3 and amount.missing == 3 and amount.warning is None
    assert "3 empty (3 marked not available)" in amount.profile
