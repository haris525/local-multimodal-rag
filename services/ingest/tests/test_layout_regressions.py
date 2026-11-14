"""Adversarial sheets for the sheet-layout code, as regressions. Each case failed in an earlier version:
the header was lost (descending years, all-text tables, numeric labels, a first data row of rising values),
rows were dropped (key-only rows, labels in column A of grouped tables), notes were joined into column names,
two stacked tables were read as one, and some types were wrong (counts as text, postal codes as numbers, ISO
and Excel serial dates, integers past 2^53, duplicate names that crashed the write)."""

import datetime as dt
from pathlib import Path

import duckdb
import pytest
from openpyxl import Workbook

from ingest.parsers.sheets import read_csv, read_xlsx
from ingest.tables import TableSource, write_tables


def csv(text: str, name: str = "t.csv"):
    return read_csv(text.encode(), name)


def frame(text: str):
    doc = csv(text)
    [sheet] = doc.sheets
    return sheet, doc.warnings


def typed(tmp_path: Path, sheet) -> dict[str, str]:
    src = TableSource(sheet.name, "x", sheet.name, sheet.frame, True, title=sheet.title, notes=sheet.notes,
                      headers=sheet.headers, units=sheet.units, layout=sheet.layout, raw=sheet.raw)
    db = tmp_path / f"{sheet.name}.duckdb"
    [schema] = write_tables(db, [src])
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
                           [schema.table]).fetchall()
    return dict(rows)


def column(sheet, name):
    return [None if v is None or v != v else v for v in sheet.frame[name]]


# --- H4: labels alone in column A are the groups of the table, not footnotes -------------------------------


def test_grouped_table_keeps_its_header_and_groups():
    sheet, warnings = frame("Region,Sales,Units\nNorth,,\nBoston,100,5\nNYC,200,6\nSouth,,\nMiami,50,2\nAtlanta,70,3\n")
    assert list(sheet.frame.columns) == ["Section", "Region", "Sales", "Units"]
    assert list(sheet.frame.index) == [3, 4, 6, 7] and column(sheet, "Section") == ["North", "North", "South", "South"]
    assert sheet.layout["header_rows"] == [1] and sheet.layout["section_rows"] == [2, 5]
    assert warnings == ['Sheet "t": rows 2, 5 read as section labels (North, South), kept in a Section column']


def test_balance_sheet_sections_in_capitals():
    sheet, _ = frame("Item,FY2024,FY2023\nASSETS,,\nCash,100,90\nReceivables,50,40\nLIABILITIES,,\nPayables,30,20\nDebt,70,60\n")
    assert list(sheet.frame.columns) == ["Section", "Item", "FY2024", "FY2023"]
    assert column(sheet, "Section") == ["ASSETS", "ASSETS", "LIABILITIES", "LIABILITIES"]


def test_two_column_grouped_table():
    sheet, _ = frame("Region,Sales\nNorth,\nBoston,100\nNYC,200\nSouth,\nMiami,50\n")
    assert list(sheet.frame.columns) == ["Section", "Region", "Sales"] and list(sheet.frame.index) == [3, 4, 6]


# --- H5: a key-only row is data unless it clearly works as a label ------------------------------------------


@pytest.mark.parametrize("text, index, kept_row", [
    ("Product,Q1,Q2\nWidget A,10,20\nWidget B,11,21\nWidget C,,\nWidget D,12,22\nWidget E,13,23\n", [2, 3, 4, 5, 6], 4),
    ("Product,Q1\nA,10\nB,\nC,12\nD,13\n", [2, 3, 4, 5], 3),
    ("Id,Comment,Score\n1,good,5\n2,,\n,This customer wrote a very long comment that exceeds forty characters,\n3,bad,1\n"
     "4,ok,3\n", [2, 3, 4, 5, 6], 5),
])
def test_key_only_rows_are_kept_as_data(text, index, kept_row):
    sheet, warnings = frame(text)
    assert list(sheet.frame.index) == index and kept_row in sheet.frame.index
    assert "Section" not in sheet.frame.columns and warnings == []


def test_a_group_label_that_starts_like_a_note_word_is_still_a_group():
    sheet, _ = frame("Item,Units,Price\nNotebooks,,\nA5 ruled,10,2.5\nA4 plain,5,3.0\nSources of light,,\nLamp,2,20\n")
    assert column(sheet, "Section") == ["Notebooks", "Notebooks", "Sources of light"]


def test_a_note_inside_the_table_is_listed_in_a_warning():
    sheet, warnings = frame("Name,Q1,Q2\nA,1,2\nB,3,4\nNote: Q2 figures were revised in March,,\nC,5,6\nD,7,8\n")
    assert list(sheet.frame.index) == [2, 3, 5, 6] and "Note: Q2 figures were revised in March" in sheet.notes
    assert warnings == ['Sheet "t": rows 4 inside the table read as notes, not data']


# --- M6: the header survives descending years, all-text tables and numeric labels -------------------------


@pytest.mark.parametrize("text, columns, first", [
    ("In millions,2024,2023,2022\nRevenue,100,90,80\nCost,60,55,50\nProfit,40,35,30\nTax,8,7,6\n",
     ["In millions", "2024", "2023", "2022"], 2),
    ("Timestamp,What is your favourite programming language and why do you like it so much?,Which editor do you use?\n"
     "1/5/2024 10:00:00,Python because it is readable,VS Code\n1/5/2024 10:05:00,Rust for safety,Neovim\n"
     "1/6/2024 09:00:00,Go,VS Code\n", ["Timestamp", "What is your favourite programming language and", "Which editor do you use?"], 2),
    ("Name,Comment,Comment\nBob,ok,fine\nAnn,meh,good\nJoe,yes,no\n", ["Name", "Comment", "Comment_2"], 2),
    ("Status,Owner\nOpen,Ann\nStatus,Bob\nClosed,Joe\n", ["Status", "Owner"], 2),
    ("Measure,10,25,50,75,90\nIncome,20,35,50,80,120\nHeight,150,160,170,180,190\nWeight,50,60,70,80,95\n",
     ["Measure", "10", "25", "50", "75", "90"], 2),
    ("Item,Q1,Q2,Q3\nA,1850,1900,2000\nB,5,6,7\nC,8,9,10\nD,1,2,3\n", ["Item", "Q1", "Q2", "Q3"], 2),
    ("Name,Start,End\nA,1990,1995\nB,1991,1999\nC,1985,2001\nD,2003,\nE,2004,2010\n", ["Name", "Start", "End"], 2),
])
def test_the_header_is_kept(text, columns, first):
    sheet, _ = frame(text)
    assert list(sheet.frame.columns) == columns and sheet.frame.index[0] == first


def test_descending_years_with_a_blank_corner_and_columns_without_a_header_are_warned():
    sheet, warnings = frame(",2024,2023,2022\nRevenue,100,90,80\nCost,60,55,50\nProfit,40,35,30\n")
    assert list(sheet.frame.columns) == ["column_1", "2024", "2023", "2022"] and len(sheet.frame) == 3
    assert warnings == ['Sheet "t": 1 columns have no header and are named column_1...']


def test_a_header_after_more_than_fifty_note_lines_is_found():
    text = "\n".join([f"Note line {i} with some text" for i in range(55)] + ["Name,Val", "a,1", "b,2", "c,3"]) + "\n"
    sheet, _ = frame(text)
    assert list(sheet.frame.columns) == ["Name", "Val"] and list(sheet.frame.index) == [57, 58, 59]
    assert "Note line 0 with some text" in sheet.notes


# --- M7: a short note above the header is a note, not part of the column names ------------------------------


@pytest.mark.parametrize("text, columns", [
    ("Units: millions\nName,Val\na,1\nb,2\nc,3\n", ["Name", "Val"]),
    (",Units: millions,\nName,Val,Other\na,1,x\nb,2,y\nc,3,z\n", ["Name", "Val", "Other"]),
    ("Source: survey\nUnits: millions\nName,Val\na,1\nb,2\nc,3\n", ["Name", "Val"]),
])
def test_a_note_above_the_header_stays_out_of_the_column_names(text, columns):
    sheet, _ = frame(text)
    assert list(sheet.frame.columns) == columns and "Units: millions" in sheet.preamble


def test_a_units_row_under_the_header_is_still_read_as_units():
    sheet, _ = frame("Station,Temp,Rain\n(id),(C),(mm)\nA,1.5,3\nB,2.5,4\nC,3.5,5\n")
    assert list(sheet.frame.columns) == ["Station", "Temp", "Rain"] and sheet.units == {"Station": "id", "Temp": "C", "Rain": "mm"}


# --- stacked tables ---------------------------------------------------------------------------------------


def test_two_stacked_tables_are_read_as_two_tables(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Sales by region"])
    ws.append(["Region", "Q1", "Q2"])
    for r in [["North", 1, 2], ["South", 3, 4], ["East", 5, 6]]:
        ws.append(r)
    ws.append([])
    ws.append(["Headcount by team"])
    ws.append(["Team", "People", "Open roles"])
    for r in [["Eng", 10, 2], ["Ops", 4, 1]]:
        ws.append(r)
    wb.save(tmp_path / "two.xlsx")
    doc = read_xlsx(tmp_path / "two.xlsx")
    first, second = doc.sheets
    assert (first.name, list(first.frame.columns), list(first.frame.index)) == ("Sheet", ["Region", "Q1", "Q2"], [3, 4, 5])
    assert (second.name, list(second.frame.columns), list(second.frame.index)) == ("Sheet (2)", ["Team", "People", "Open roles"], [9, 10])
    assert first.title == "Sales by region" and second.title == "Headcount by team"
    assert first.layout["next_table_row"] == 7 and second.layout["raw_first_row"] == 7
    assert typed(tmp_path, first)["q1"] == "BIGINT"  # the second header no longer turns Q1 into text
    [schema] = write_tables(tmp_path / "s2.duckdb", [TableSource(second.name, "x", second.name, second.frame, True,
                                                                 layout=second.layout, raw=second.raw)])
    with duckdb.connect(str(tmp_path / "s2.duckdb"), read_only=True) as con:
        assert con.execute(f"SELECT MIN(_row), MAX(_row) FROM {schema.layout.raw_table}").fetchone() == (7, 10)
    assert doc.warnings == ['Sheet "Sheet": 2 tables found (data from rows 3, 9), read separately']


# --- L15: types ------------------------------------------------------------------------------------------


def test_types_counts_postal_codes_iso_dates(tmp_path):
    sheet, _ = frame("Team,Codes issued,Bugs\nA,10,3\nB,20,4\nC,30,5\n")
    assert typed(tmp_path, sheet)["codes_issued"] == "BIGINT"
    sheet, _ = frame("Town,Postal\nBoston,02134\nCambridge,02139\nNYC,10001\n")
    assert typed(tmp_path, sheet)["postal"] == "VARCHAR" and column(sheet, "Postal")[0] == "02134"
    rows = ["Name,Shipped date,Qty"] + [f"n{i},{'' if i < 2000 else '2024-01-%02d' % (1 + i % 28)},{i}" for i in range(3000)]
    sheet, _ = frame("\n".join(rows) + "\n")
    assert typed(tmp_path, sheet)["shipped_date"] == "DATE"
    sheet, _ = frame("Event,Product ID,Number of IDs\nx,0017,3\ny,0042,4\n")
    types = typed(tmp_path, sheet)
    assert types["product_id"] == "VARCHAR" and types["number_of_ids"] == "BIGINT"


def test_excel_serial_numbers_stay_numbers_and_integers_past_2_53_stay_exact(tmp_path):
    """Serial day numbers are never converted (a date-formatted cell already arrives as a date); the
    profile names them instead."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Day", "Value", "Big"])
    for i in range(10):
        ws.append([45292 + i, i, 2**60 + i])
    ws.cell(row=2, column=1).number_format = "General"
    wb.save(tmp_path / "serial.xlsx")
    [sheet] = read_xlsx(tmp_path / "serial.xlsx").sheets
    types = typed(tmp_path, sheet)
    assert types["day"] == "BIGINT" and types["big"] == "BIGINT"
    with duckdb.connect(str(tmp_path / f"{sheet.name}.duckdb"), read_only=True) as con:
        day, big = con.execute("SELECT day, big FROM sheet ORDER BY _row LIMIT 1").fetchone()
    assert day == 45292 and big == 2**60
    wb = Workbook()
    ws = wb.active
    ws.append(["Shipped", "Qty"])
    ws.append([dt.datetime(2024, 1, 5), 1])
    ws.append([dt.datetime(2024, 1, 6), 2])
    wb.save(tmp_path / "dated.xlsx")
    [dated] = read_xlsx(tmp_path / "dated.xlsx").sheets
    assert typed(tmp_path, dated)["shipped"] == "DATE"  # a date-formatted cell is a date


def test_day_of_month_numbers_stay_numbers(tmp_path):
    sheet, _ = frame("Day,Value\n1,5\n2,6\n31,7\n")
    assert typed(tmp_path, sheet)["day"] == "BIGINT"


# --- L16, L17 -------------------------------------------------------------------------------------------


def test_duplicate_generated_names_do_not_crash_the_write(tmp_path):
    sheet, _ = frame("Name,A,A,A_2\nx,1,2,3\ny,4,5,6\nz,7,8,9\n")
    assert list(sheet.frame.columns) == ["Name", "A", "A_3", "A_2"]
    assert set(typed(tmp_path, sheet)) == {"_row", "name", "a", "a_3", "a_2"}


def test_the_layout_says_when_the_raw_table_was_cut(monkeypatch):
    monkeypatch.setenv("INGEST_RAW_TABLE_MAX_ROWS", "5")
    sheet, _ = frame("Name,Val\n" + "".join(f"n{i},{i}\n" for i in range(20)))
    assert (sheet.layout["sheet_rows"], sheet.layout["raw_rows"], len(sheet.raw)) == (21, 5, 5)
