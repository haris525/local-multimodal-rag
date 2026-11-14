"""Sheet layout: title and preamble rows, multi-row headers, merged cells, units rows, section labels and
footnotes. The sample workbooks in examples/documents/data are the regressions (government downloads, read
unmodified)."""

from pathlib import Path

import duckdb
import pytest
from openpyxl import Workbook

from ingest.parsers.layout import Merge, read_table
from ingest.parsers.sheets import read_csv, read_xlsx
from ingest.tables import TableSource, write_tables

DATA = Path(__file__).resolve().parents[3] / "examples" / "documents" / "data"


def sample(name: str) -> Path:
    path = next(DATA.rglob(name), None)
    if path is None:
        pytest.skip(f"sample file {name} not in this checkout")
    return path


def table_source(sheet) -> TableSource:
    return TableSource(sheet.name, f'sheet "{sheet.name}"', sheet.name, sheet.frame, True, title=sheet.title,
                       notes=sheet.notes, headers=sheet.headers, units=sheet.units, layout=sheet.layout, raw=sheet.raw)


def test_census_population_workbook_regression(tmp_path):
    """The reported failure: a merged accessibility note and title over a two-row header (labels over years
    given as integers), footnotes merged across the table. The title was read as the header."""
    [sheet] = read_xlsx(sample("census-state-population-estimates-2020-2025.xlsx")).sheets
    years = [f"Population Estimate (as of July 1) {y}" for y in range(2020, 2026)]
    assert list(sheet.frame.columns) == ["Geographic Area", "April 1, 2020 Estimates Base", *years]
    assert max(len(c) for c in sheet.frame.columns) <= 48
    assert len(sheet.frame) == 57 and sheet.frame.index[0] == 5 and sheet.frame.index[-1] == 62
    assert 61 not in sheet.frame.index  # the blank row
    assert sheet.title.startswith("Annual Estimates of the Resident Population")
    assert "leading dots indicate sub-parts" in sheet.notes and "Release Date: January 2026" in sheet.notes
    assert sheet.layout["header_rows"] == [3, 4] and sheet.layout["footnote_rows"] == [63, 64, 65, 66, 67, 68]

    [schema] = write_tables(tmp_path / "t.duckdb", [table_source(sheet)])
    assert [c.type for c in schema.columns] == ["text"] + ["number"] * 7
    assert schema.title == sheet.title and schema.layout.raw_table == "nst_est2025_pop__raw"
    with duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True) as con:
        assert con.execute("SELECT data_type FROM information_schema.columns WHERE table_name = 'nst_est2025_pop' "
                           "AND column_name = 'population_estimate_as_of_july_1_2025'").fetchone() == ("BIGINT",)
        top = con.execute(
            "SELECT geographic_area, (population_estimate_as_of_july_1_2025 - population_estimate_as_of_july_1_2024)"
            " * 100.0 / population_estimate_as_of_july_1_2024 AS g FROM nst_est2025_pop"
            " WHERE geographic_area LIKE '.%' ORDER BY g DESC LIMIT 2").fetchall()
        assert [r[0] for r in top] == [".South Carolina", ".Idaho"] and round(top[0][1], 2) == 1.46
        assert con.execute("SELECT c1 FROM nst_est2025_pop__raw WHERE _row = 2").fetchone()[0].startswith("Annual Estimates")


def test_retail_two_row_header_without_merges_and_section_rows(tmp_path):
    sheets = {s.name: s for s in read_xlsx(sample("census-monthly-retail-sales-1992-2026.xlsx")).sheets}
    assert len(sheets) == 35
    sheet = sheets["2025"]
    assert list(sheet.frame.columns)[:4] == ["Section", "NAICS Code", "Kind of Business", "Jan. 2025"]
    assert list(sheet.frame.columns)[-1] == "TOTAL"
    assert set(sheet.frame["Section"]) == {"NOT ADJUSTED", "ADJUSTED(2)"}
    assert sheet.title == "Estimates of Monthly Retail and Food Services Sales by Kind of Business: 2025"
    assert all(list(s.frame.columns)[:3] == ["Section", "NAICS Code", "Kind of Business"] for s in sheets.values())

    [schema] = write_tables(tmp_path / "r.duckdb", [table_source(sheet)])
    by = {c.sql_name: c for c in schema.columns}
    assert by["naics_code"].type == "text"  # identifiers stay text: "4411,4412" is a list of codes, not 44114412
    assert by["jan_2025"].type == "number" and by["jan_2025"].missing > 0  # "(S)" suppressed cells are NULL
    with duckdb.connect(str(tmp_path / "r.duckdb"), read_only=True) as con:
        assert con.execute("SELECT dec_2025 FROM c_2025 WHERE section = 'NOT ADJUSTED' AND "
                           "kind_of_business = 'Retail and food services sales, total'").fetchone() == (780707,)


def test_units_row_and_not_available_markers(tmp_path):
    sheets = {s.name: s for s in read_xlsx(sample("eia-us-electricity-generation-by-source-1949-2026.xlsx")).sheets}
    sheet = sheets["Annual Data"]
    assert list(sheet.frame.columns)[1:4] == ["Coal", "Petroleum", "Natural Gas"]
    assert sheet.headers["Solar"] == "Electricity Net Generation From Solar, All Sectors"
    assert sheet.units["Solar"] == "Million Kilowatthours"
    assert 'Columns 2-14 are "Electricity Net Generation From <name>, All Sectors" in Million Kilowatthours' in sheet.notes
    assert sheet.frame.index[0] == 13  # the units row (12) is not data

    [schema] = write_tables(tmp_path / "e.duckdb", [table_source(sheet)])
    solar = next(c for c in schema.columns if c.sql_name == "solar")
    assert (solar.type, solar.unit, solar.header) == ("number", "Million Kilowatthours", sheet.headers["Solar"])
    assert solar.missing == 35
    with duckdb.connect(str(tmp_path / "e.duckdb"), read_only=True) as con:
        assert con.execute("SELECT solar FROM annual_data WHERE annual_total = 2025").fetchone()[0] == pytest.approx(295670.6, abs=1)


def test_noaa_dates_become_dates_and_clean_csvs_keep_their_names(tmp_path):
    noaa = read_csv(sample("noaa-us-billion-dollar-disasters-1980-2024.csv").read_bytes(), "noaa.csv").sheets[0]
    [schema] = write_tables(tmp_path / "n.duckdb", [table_source(noaa)])
    types = {c.sql_name: c.type for c in schema.columns}
    assert types["begin_date"] == "date" and types["deaths"] == "number"
    with duckdb.connect(str(tmp_path / "n.duckdb"), read_only=True) as con:
        rows = con.execute("SELECT FLOOR(YEAR(begin_date) / 10) * 10 AS d, COUNT(*) FROM noaa GROUP BY d ORDER BY d").fetchall()
        assert [(int(d), n) for d, n in rows] == [(1980, 33), (1990, 57), (2000, 67), (2010, 131), (2020, 115)]
        assert con.execute("SELECT typeof(deaths) FROM noaa LIMIT 1").fetchone() == ("BIGINT",)
    for name, first in [("cdc-leading-causes-of-death-us-1999-2017.csv", ["Year", "113 Cause Name"]),
                        ("world-bank-gdp-per-capita-internet-use-2000-2024.csv", ["country", "iso3_code"]),
                        ("usgs-earthquakes-m5-plus-2025.csv", ["time", "latitude"])]:
        sheet = read_csv(sample(name).read_bytes(), name).sheets[0]
        assert list(sheet.frame.columns)[:2] == first and sheet.frame.index[0] == 2 and not sheet.title


def test_csv_with_title_lines_like_the_original_noaa_download():
    text = ("Billion-Dollar Weather and Climate Disasters\nCPI-Adjusted Costs in Millions of Dollars\n"
            "Name,Disaster,Begin Date,End Date,Cost,Deaths\nA,Flooding,19800410,19800417,2756.4,30\n"
            "B,Drought,19800601,19801130,40480.8,1260\nC,Storm,19810101,19810102,1200,0\n")
    sheet = read_csv(text.encode(), "noaa.csv").sheets[0]
    assert list(sheet.frame.columns) == ["Name", "Disaster", "Begin Date", "End Date", "Cost", "Deaths"]
    assert list(sheet.frame.index) == [4, 5, 6]
    assert "Billion-Dollar Weather and Climate Disasters" in sheet.preamble


def test_merged_title_over_one_header_row_header_after_blank_row_and_footnotes(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Table 1. Widgets by plant", None, None])
    ws.merge_cells("A1:C1")
    ws.append(["Plant", "Widgets", "Share"])
    for row in [["P1", 100, 0.5], ["P2", 60, 0.3], ["P3", 40, 0.2]]:
        ws.append(row)
    ws2 = wb.create_sheet("Report")
    for row in [["Report 2025"], [], ["Region", "Sales", "Units"], [], ["North", 10, 2], ["South", 20, 3],
                ["East", 5, 1], [], ["Note: preliminary figures, subject to revision by the agency"], ["Source: internal"]]:
        ws2.append(row)
    wb.save(tmp_path / "w.xlsx")
    first, second = read_xlsx(tmp_path / "w.xlsx").sheets
    assert (list(first.frame.columns), first.title) == (["Plant", "Widgets", "Share"], "Table 1. Widgets by plant")
    assert list(second.frame.columns) == ["Region", "Sales", "Units"] and list(second.frame.index) == [5, 6, 7]
    assert second.layout["footnote_rows"] == [9, 10] and "Source: internal" in second.notes


def test_all_text_table_and_numbers_only_table():
    text = read_csv(b"name,city\nAnn,Paris\nBo,Rome\nCy,Oslo\n", "t.csv").sheets[0]
    assert list(text.frame.columns) == ["name", "city"] and list(text.frame.index) == [2, 3, 4]
    numbers = read_table([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert list(numbers.frame.columns) == ["column_1", "column_2", "column_3"] and len(numbers.frame) == 3


def test_vertical_row_labels_are_filled_down_and_long_shared_names_are_shortened():
    rows = [["Region", "Quarterly revenue from domestic retail customers, all segments",
             "Quarterly revenue from online retail customers, all segments",
             "Quarterly revenue from wholesale customers, all segments"],
            ["North", 1, 2, 3], [None, 4, 5, 6], ["South", 7, 8, 9]]
    table = read_table(rows, [Merge(1, 2, 0, 0)])
    assert list(table.frame["Region"]) == ["North", "North", "South"]
    assert list(table.frame.columns)[1:] == ["domestic retail", "online retail", "wholesale"]
    assert table.headers["online retail"] == "Quarterly revenue from online retail customers, all segments"
    assert 'Columns 2-4 are "Quarterly revenue from <name> customers, all segments"' in table.notes
