"""More sheet-layout regressions: a first data row with
an odd cell was joined into the header, column names that only contain "date" or "day" turned numbers into
Excel dates, footnotes that look like data, a header repeated after a page break, a blank row that split a
table, key-only rows read as groups, zero-padded months read as codes, and more tables than the cap."""

import random
from pathlib import Path

import duckdb
import pytest
from openpyxl import Workbook
from openpyxl.styles import Font

from ingest.parsers.layout import read_tables
from ingest.parsers.sheets import read_csv, read_xlsx
from ingest.tables import TableSource, write_tables


def only(text: str, name: str = "t.csv"):
    doc = read_csv(text.encode(), name)
    [sheet] = doc.sheets
    return sheet, doc.warnings


def typed(tmp_path: Path, sheet) -> dict[str, tuple[str, str | None]]:
    db = tmp_path / f"{abs(hash(sheet.name))}.duckdb"
    [schema] = write_tables(db, [TableSource(sheet.name, "x", sheet.name, sheet.frame, True, headers=sheet.headers)])
    with duckdb.connect(str(db), read_only=True) as con:
        types = dict(con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
                                 [schema.table]).fetchall())
    return {c.sql_name: (types[c.sql_name], c.pattern) for c in schema.columns}


random.seed(5)


def body(n: int) -> str:
    return "".join(f"item{i},{random.randint(1, 999)},{random.randint(1, 99)}\n" for i in range(n))


# --- must fix 2: an odd first data row is a data row, never part of the header ------------------------------


@pytest.mark.parametrize("text, columns, rows, first", [
    ("County,Cases,Rate\nAlpha,<10,1.2\nBeta,25,3.4\nGamma,40,5.1\nDelta,12,2.2\n", ["County", "Cases", "Rate"], 4, 2),
    ("Name,Value,Count\nitem_first,TBD,5\n" + body(50), ["Name", "Value", "Count"], 51, 2),
    ("Name,Shipped,Qty\nfirst,unknown,5\n" + "".join(f"n{i},2024-01-{1 + i % 28:02d},{i}\n" for i in range(50)),
     ["Name", "Shipped", "Qty"], 51, 2),
    ("Event,When,Qty\nfirst,2024-01-05,1\n" + "".join(f"e{i},01/{1 + i % 28:02d}/2024,{i}\n" for i in range(50)),
     ["Event", "When", "Qty"], 51, 2),
    ("Name,Value,Count\nA,n.d.,5\nB,pending,6\n" + body(50), ["Name", "Value", "Count"], 52, 2),
    ("Event,When\na,2024-01-05\nb,01/06/2024\nc,\"Jan 7, 2024\"\nd,2024-01-08\n", ["Event", "When"], 4, 2),
    ("Name,Alice\nAge,30\nCity,Paris\nRole,Engineer\n", ["Name", "Alice"], 3, 2),
])
def test_an_odd_first_data_row_stays_data(text, columns, rows, first):
    sheet, warnings = only(text)
    assert list(sheet.frame.columns) == columns and len(sheet.frame) == rows and sheet.frame.index[0] == first
    assert sheet.layout["header_rows"] == [1] and not any("header rows" in w for w in warnings)


def test_the_county_row_with_a_suppressed_count_keeps_its_value_as_a_missing_marker(tmp_path):
    sheet, _ = only("County,Cases,Rate\nAlpha,<10,1.2\nBeta,25,3.4\nGamma,40,5.1\n")
    assert sheet.frame.loc[2, "County"] == "Alpha"
    db = tmp_path / "c.duckdb"
    [schema] = write_tables(db, [TableSource(sheet.name, "x", sheet.name, sheet.frame, True)])
    cases = {c.sql_name: c for c in schema.columns}["cases"]
    assert cases.type == "number" and cases.missing == 1 and "1 empty (1 marked not available)" in cases.profile


def test_a_units_row_with_a_stray_cell_in_column_a_is_a_units_row():
    """"2, (USD), (n)" under the header is a units row (the 2 is stray), not data that turns Sales into text,
    which is how the simple reader and an earlier fallback read it."""
    sheet, warnings = only("Region,Sales,Units\n2,(USD),(n)\nNorth,10,1\nSouth,20,2\nEast,30,3\nWest,40,4\n")
    assert list(sheet.frame.columns) == ["Region", "Sales", "Units"] and list(sheet.frame.index) == [3, 4, 5, 6]
    assert sheet.layout["reader"] == "layout" and sheet.units == {"Sales": "USD", "Units": "n"} and warnings == []


# --- must fix 1: numbers become Excel dates only in a workbook, under a real date header --------------------


def test_names_that_only_contain_date_or_day_never_make_dates(tmp_path):
    sheet, _ = only("Year,Consolidated revenue,Candidates,Holiday sales,Days on market\n"
                    "2021,25000,41000,30500,20100\n2022,31000,52000,33100,22000\n2023,40000,61000,35500,25000\n")
    types = typed(tmp_path, sheet)
    assert {k: v[0] for k, v in types.items()} == {"year": "BIGINT", "consolidated_revenue": "BIGINT", "candidates": "BIGINT",
                                                    "holiday_sales": "BIGINT", "days_on_market": "BIGINT"}


def test_csv_integers_under_a_date_header_are_never_serial_dates(tmp_path):
    sheet, _ = only("Name,Ship Date\na,45292\nb,45293\nc,45300\n")
    assert typed(tmp_path, sheet)["ship_date"][0] == "BIGINT"


def test_workbook_numbers_in_the_serial_range_stay_numbers_and_only_a_date_header_gets_a_hint(tmp_path):
    """"Visitors per day" and "Day 1 revenue" became dates. Serial numbers are never converted now; a
    header whose last word is date, day or dt (not a rate) gets a pattern that says how to read them."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Ship Date", "Candidates", "BEGIN_DAY", "Visitors per day", "Day 1 revenue"])
    for i in range(5):
        ws.append([45292 + i, 41000 + i, 45300 + i, 30000 + i, 25000 + i])
    wb.save(tmp_path / "s.xlsx")
    [sheet] = read_xlsx(tmp_path / "s.xlsx").sheets
    types = typed(tmp_path, sheet)
    assert {k: v[0] for k, v in types.items()} == {"ship_date": "BIGINT", "candidates": "BIGINT", "begin_day": "BIGINT",
                                                    "visitors_per_day": "BIGINT", "day_1_revenue": "BIGINT"}
    hint = "numbers in the Excel day-number range under a date header"
    assert types["ship_date"][1] == hint and types["begin_day"][1] == hint
    assert types["visitors_per_day"][1] is None and types["day_1_revenue"][1] is None and types["candidates"][1] is None


# --- notes, repeated headers, blank rows, stacked tables ----------------------------------------------------


@pytest.mark.parametrize("text, rows, note", [
    ("State,Population,Area\nAlabama,5000000,52420\nAlaska,730000,665384\nArizona,7300000,113990\n\nSource:,U.S. Census Bureau,\n",
     [2, 3, 4], "Source: U.S. Census Bureau"),
    ("State,Population\nAlabama,5000000\nAlaska,730000\nArizona,7300000\n1,Includes armed forces\n2,Preliminary\n",
     [2, 3, 4], "1 Includes armed forces"),
    ("State,Population\n" + "".join(f"S{i},{1000 + i}\n" for i in range(50)) + "1,Includes armed forces\n2,Preliminary\n",
     list(range(2, 52)), "2 Preliminary"),
])
def test_footnotes_that_look_like_data_are_notes(tmp_path, text, rows, note):
    sheet, warnings = only(text)
    assert list(sheet.frame.index) == rows and note in sheet.notes
    assert any("below the table do not fit its columns and were read as notes" in w for w in warnings)
    assert typed(tmp_path, sheet)["population"][0] == "BIGINT"


def test_a_last_row_with_an_odd_value_stays_data():
    sheet, _ = only("Name,Score\n" + "".join(f"n{i},{i}\n" for i in range(7)) + "n7,see note\n")
    assert list(sheet.frame.index)[-1] == 9


def test_a_header_repeated_after_a_page_break_is_merged_not_split():
    doc = read_csv(b"Name,Score\na,1\nb,2\nc,3\n\nName,Score\nd,4\ne,5\nf,6\n", "p.csv")
    [sheet] = doc.sheets
    assert list(sheet.frame.index) == [2, 3, 4, 7, 8, 9]
    assert doc.warnings == ['Sheet "p": rows 6 repeat the header (a page break) and were skipped']


def test_a_blank_row_then_a_row_with_a_placeholder_does_not_split_the_table():
    doc = read_csv(b"Item,Amount,Status\nA,10,ok\nB,20,ok\n\nC,pending,review\nD,30,ok\nE,40,ok\n", "b.csv")
    [sheet] = doc.sheets
    assert list(sheet.frame.columns) == ["Item", "Amount", "Status"] and list(sheet.frame.index) == [2, 3, 5, 6, 7]


def test_three_stacked_tables_of_different_widths(tmp_path):
    wb = Workbook()
    ws = wb.active
    for row in [["Table 1"], ["Region", "Q1", "Q2"], ["N", 1, 2], ["S", 3, 4], ["E", 5, 6], [], ["Team", "People"],
                ["Eng", 10], ["Ops", 4], ["HR", 2], [], ["Table 3"], ["Year", "Revenue", "Cost", "Profit"],
                [2022, 10, 5, 5], [2023, 12, 6, 6]]:
        ws.append(row)
    wb.save(tmp_path / "three.xlsx")
    sheets = read_xlsx(tmp_path / "three.xlsx").sheets
    assert [list(s.frame.columns) for s in sheets] == [["Region", "Q1", "Q2"], ["Team", "People"],
                                                       ["Year", "Revenue", "Cost", "Profit"]]


def test_both_readings_are_scored_and_the_simple_one_wins_when_the_layout_reading_is_worse(monkeypatch):
    """The safety net: every table is read both ways and scored. A layout reading that takes a data row as the
    header (forced here) loses to the simple reading, and both scores are recorded."""
    from ingest.parsers import layout as L

    text = "Region,Sales,Units\nNorth,10,1\nSouth,20,2\nEast,30,3\nWest,40,4\nMid,50,5\n"
    plain, _ = only(text)
    assert plain.layout["reader"] == "layout" and plain.layout["scores"] == {"layout": 1.0, "simple": 1.0}
    real_find = L._find
    monkeypatch.setattr(L, "_find", lambda g, body: ([2],) + tuple(real_find(g, body)[1:]))  # "South,20,2" as header
    sheet, warnings = only(text)
    assert sheet.layout["reader"] == "simple" and list(sheet.frame.columns) == ["Region", "Sales", "Units"]
    assert len(sheet.frame) == 5 and sheet.layout["scores"]["simple"] > sheet.layout["scores"]["layout"] + 0.05
    assert any("read with the simple rule" in w for w in warnings)


def test_a_section_label_read_as_data_scores_lower_and_a_single_column_is_not_unclear():
    sheet, warnings = only("Region,Sales,Units\nNorth,,\nBoston,100,5\nNYC,200,6\nSouth,,\nMiami,50,2\nAtlanta,70,3\n")
    assert sheet.layout["reader"] == "layout" and list(sheet.frame.columns) == ["Section", "Region", "Sales", "Units"]
    assert sheet.layout["scores"]["layout"] > sheet.layout["scores"]["simple"] + 0.05
    for text in ("Name\nAnn\nBo\nCy\n", "Visitors\n120\n135\n150\n"):
        sheet, warnings = only(text)
        assert len(sheet.frame.columns) == 1 and len(sheet.frame) == 3 and not any("unclear" in w for w in warnings)


def test_two_cell_preamble_rows_stay_out_of_the_header_and_the_sql_sum_is_right(tmp_path):
    """The critical case: "Source, Census Bureau" above a year header made the header
    ["Source", "Census Bureau", column_3, ...] and SUM of the last column 3268 instead of 1245."""
    body = "".join(f"State {i},{100 + i},{110 + i},{120 + i}\n" for i in range(10))
    for preamble in ("Source,Census Bureau\n", "Units,Thousands\n", "Release date,2024-05-01\n",
                     "Source,Census Bureau\n\n", "Population by state\nRelease date,2024-05-01\n"):
        sheet, warnings = only(preamble + "State,2021,2022,2023\n" + body)
        assert list(sheet.frame.columns) == ["State", "2021", "2022", "2023"] and len(sheet.frame) == 10, preamble
        assert sheet.layout["reader"] == "layout" and set(sheet.layout["scores"]) == {"layout", "simple"}
    sheet, _ = only("Source,Census Bureau\nState,Population,Area,Density\n" + body)
    assert list(sheet.frame.columns) == ["State", "Population", "Area", "Density"]  # not "Source State"
    [schema] = write_tables(tmp_path / "t.duckdb", [TableSource("t", "x", "t", sheet.frame, True, title=sheet.title,
                                                                  notes=sheet.notes, headers=sheet.headers,
                                                                  units=sheet.units, layout=sheet.layout)])
    with duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True) as con:
        assert con.execute(f'SELECT SUM(density) FROM "{schema.table}"').fetchone()[0] == 1245


def test_a_world_bank_export_and_a_two_key_header_are_read_whole():
    wdi = ('"Data Source","World Development Indicators",\n\n"Last Updated Date","2024-06-28",\n\n'
           '"Country Name","Country Code","Indicator Name","Indicator Code","1960","1961",\n'
           '"Aruba","ABW","Population, total","SP.POP.TOTL","54608","55811",\n'
           '"Angola","AGO","Population, total","SP.POP.TOTL","5357195","5441333",\n')
    sheet, warnings = only(wdi)
    assert list(sheet.frame.columns)[:6] == ["Country Name", "Country Code", "Indicator Name", "Indicator Code",
                                             "1960", "1961"] and len(sheet.frame) == 2
    sheet, _ = only("NAICS Code,Kind of Business,Jan,Feb\n441,Motor vehicles,10,11\n722,Food services,20,21\n"
                    "445,Grocery,30,31\n")
    assert list(sheet.frame.columns) == ["NAICS Code", "Kind of Business", "Jan", "Feb"] and len(sheet.frame) == 3


def test_signed_unicode_minus_and_scientific_numbers_are_numbers_and_sort_as_numbers(tmp_path):
    minus = chr(0x2212)
    cases = {"Change": (["+9.5%", "-0.4%", "+10.2%", minus + "1.1%"], "+10.2%"),
             "Delta": ([minus + "3.4", "4.5", minus + "10.2", "2.0"], "4.5"),
             "p_value": (["1.2e-05", "0.03", "3.4E-08", "2e-3"], "0.03")}
    for name, (values, top) in cases.items():
        text = f"Key,{name}\n" + "".join(f"k{i},{v}\n" for i, v in enumerate(values))
        sheet, _ = only(text)
        assert all(chr(0xE2) not in str(v) for v in sheet.frame[name]), name  # read as UTF-8, not cp1252
        db = tmp_path / f"{name}.duckdb"
        [schema] = write_tables(db, [TableSource("t", "x", "t", sheet.frame, True, headers=sheet.headers,
                                                 layout=sheet.layout)])
        column = schema.columns[1]
        assert column.type == "number" and column.pattern != "codes or identifiers", name
        with duckdb.connect(str(db), read_only=True) as con:
            best = con.execute(f'SELECT key FROM "{schema.table}" ORDER BY {column.sql_name} DESC LIMIT 1').fetchone()[0]
        assert best == f"k{values.index(top)}", name


def test_percent_formatted_cells_get_a_pattern_that_says_they_are_fractions(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["State", "Growth rate (%)", "People"])
    for state, rate, people in [("A", 0.052, 10), ("B", 0.013, 20), ("C", 0.101, 30)]:
        ws.append([state, rate, people])
        ws.cell(row=ws.max_row, column=2).number_format = "0.0%"
    wb.save(tmp_path / "pct.xlsx")
    [sheet] = read_xlsx(tmp_path / "pct.xlsx").sheets
    assert sheet.layout["percent_columns"] == ["Growth rate (%)"]
    [schema] = write_tables(tmp_path / "p.duckdb", [TableSource("t", "x", "t", sheet.frame, True, headers=sheet.headers,
                                                               layout=sheet.layout)])
    rate, people = schema.columns[1], schema.columns[2]
    assert rate.pattern.startswith("fractions shown as % in Excel") and "ROUND(100.0 * col, 2)" in rate.pattern
    assert not (people.pattern or "").startswith("fractions")


def test_chunks_of_each_table_point_at_its_own_sql_table_when_a_sheet_is_named_like_a_copy(tmp_path):
    """Chunks were paired with schemas by sheet name, so a copied sheet "Sales (2)" and the
    second table of "Sales" shared one SQL table."""
    from types import SimpleNamespace

    from ingest.config import Settings
    from ingest.pipeline import Pipeline

    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    for r in [["Region", "Q1", "Q2"], ["North", 1, 2], ["South", 3, 4], ["East", 5, 6], [], ["Team", "People", "Open"],
              ["Eng", 10, 2], ["Ops", 4, 1]]:
        ws.append(r)
    copy = wb.create_sheet("Sales (2)")
    for r in [["Region", "Q1", "Q2"], ["North", 100, 200], ["South", 300, 400], ["East", 500, 600]]:
        copy.append(r)
    wb.save(tmp_path / "book.xlsx")
    parsed = read_xlsx(tmp_path / "book.xlsx")
    counter = SimpleNamespace(count=lambda t: len(t.split()), count_many=lambda ts: [len(t.split()) for t in ts])
    pipe = Pipeline(Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None), None, None, None, counter, None)
    schemas = write_tables(tmp_path / "book.duckdb", pipe.table_sources(parsed))
    doc = SimpleNamespace(doc_id="d" * 32, name="book.xlsx")
    chunks = pipe.chunk(doc, parsed, schemas)
    tables = {(c.sheet, c.table) for c in chunks if c.table}
    by_sheet = {s.sheet: s.table for s in schemas}
    assert tables == {(name, by_sheet[name]) for name in ("Sales", "Sales (table 2)", "Sales (2)")}
    assert len(set(by_sheet.values())) == 3


def _sum(tmp_path, sheet, column: int) -> float:
    db = tmp_path / f"{abs(hash(tuple(sheet.frame.columns)))}.duckdb"
    [schema] = write_tables(db, [TableSource("t", "x", "t", sheet.frame, True, headers=sheet.headers, units=sheet.units,
                                             layout=sheet.layout, raw=sheet.raw)])
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute(f'SELECT SUM({schema.columns[column].sql_name}) FROM "{schema.table}"').fetchone()[0]


def test_semicolon_files_keep_decimal_points_and_the_raw_cells_as_written(tmp_path):
    """Every cell of a ";" file went through the decimal-comma rewrite, so 0.125 became 125
    and 1.250 became 1250 next to 2.5 and 10.75, also in the raw table."""
    sheet, warnings = only("Item;Weight_kg;Price\nA;0.125;3.99\nB;2.5;12.50\nC;1.250;7.25\nD;10.75;1.20\n")
    assert sheet.frame["Weight_kg"].tolist() == ["0.125", "2.5", "1.250", "10.75"] and warnings == []
    assert _sum(tmp_path, sheet, 1) == pytest.approx(14.625)
    assert [row[1] for row in sheet.raw] == ["Weight_kg", "0.125", "2.5", "1.250", "10.75"]
    sheet, warnings = only("Item;Amount\nA;1.234,56\nB;2.000,00\nC;300,10\n")
    assert sheet.frame["Amount"].tolist() == ["1234.56", "2000.00", "300.10"] and warnings == []
    assert [row[1] for row in sheet.raw] == ["Amount", "1.234,56", "2.000,00", "300,10"]  # as written
    sheet, warnings = only("Item;Weight;Price\nA;1.250;1,5\nB;2.500;2,25\n")  # the file's commas settle 1.250
    assert sheet.frame.values.tolist() == [["A", "1250", "1.5"], ["B", "2500", "2.25"]]
    sheet, warnings = only("Item;Weight\nA;1.250\nB;2.500\nC;3.750\n")  # nothing settles it: kept, named
    assert sheet.frame["Weight"].tolist() == ["1.250", "2.500", "3.750"]
    assert len(warnings) == 1 and "dot may group thousands or mark the decimals" in warnings[0]
    sheet, warnings = only("Item;Weight\nA;1,5\nB;2.75\n")
    assert sheet.frame["Weight"].tolist() == ["1,5", "2.75"] and "mixes decimal commas and decimal points" in warnings[0]


@pytest.mark.parametrize("preamble, years", [
    ("Units,Thousands\n", [2023, 2024]), ("Source,Census Bureau\n", [2023, 2024]),
    ("Release date,2024-05-01\n", [2023, 2024]), ("Table 4,Population by state,2024\n", [2022, 2023]),
    ("Table 4,Population by state,2024\n", [2021, 2022, 2023]),
    ("Table 4,Population by state,2024\n", [2020, 2021, 2022, 2023]),
])
def test_preamble_rows_over_narrow_tables_stay_out_of_the_header(tmp_path, preamble, years):
    """Over three columns a two-cell preamble fills 60% of the width and was taken for the
    header ("Units | Thousands | column_3", SUM 4069 against 2045); a three-cell one gave column_N."""
    head = "State," + ",".join(map(str, years)) + "\n"
    body = "".join(f"State {i}," + ",".join(str(100 * (k + 1) + i) for k in range(len(years))) + "\n" for i in range(10))
    sheet, warnings = only(preamble + head + body)
    assert list(sheet.frame.columns) == ["State"] + [str(y) for y in years] and len(sheet.frame) == 10
    assert sheet.layout["reader"] == "layout" and warnings == []
    assert _sum(tmp_path, sheet, len(years)) == sum(100 * len(years) + i for i in range(10))


def test_a_two_cell_note_over_a_full_header_and_a_source_row_over_two_columns_are_not_glued_on():
    sheet, _ = only("Agency,Dept of Labor\nState,Workers,Openings,Hires\n"
                    + "".join(f"State {i},{1000 + i},{50 + i},{40 + i}\n" for i in range(10)))
    assert list(sheet.frame.columns) == ["State", "Workers", "Openings", "Hires"] and len(sheet.frame) == 10
    sheet, _ = only("Source,FRED\nYear,Value\n" + "".join(f"{2000 + i},{1.5 + i}\n" for i in range(10)))
    assert list(sheet.frame.columns) == ["Year", "Value"] and len(sheet.frame) == 10
    sheet, _ = only("Source,Count\nWeb,10\nMail,4\nPhone,7\n")  # a header that only starts like a preamble key
    assert list(sheet.frame.columns) == ["Source", "Count"] and len(sheet.frame) == 3


def test_a_year_among_the_unit_labels_of_a_second_header_row_keeps_the_layout_reading(tmp_path):
    """", n, per 100k, 2023" under the header cost the layout reading half its header score, so the simple
    reading took over with the units row as data: SUM(Beds) 2211 against 188."""
    text = ("County,Admissions,Rate,Beds\n,n,per 100k,2023\n"
            + "".join(f"County {i},{100 + i},{1.5 + i},{20 + i}\n" for i in range(8)))
    sheet, warnings = only(text)
    assert list(sheet.frame.columns) == ["County", "Admissions n", "Rate per 100k", "Beds 2023"] and len(sheet.frame) == 8
    assert sheet.layout["reader"] == "layout" and _sum(tmp_path, sheet, 3) == 188
    assert sheet.layout["scores"]["layout"] > sheet.layout["scores"]["simple"]
    sheet, _ = only("Region,Population,Area,Density\n,2020,sq km,per sq km\n"
                    + "".join(f"R{i},{5000 + i},{300 + i},{16 + i}\n" for i in range(10)))
    assert list(sheet.frame.columns)[:2] == ["Region", "Population 2020"] and len(sheet.frame) == 10


def test_the_simple_reading_wins_only_with_more_fitting_rows_and_its_warning_says_the_readings_disagree(monkeypatch):
    from ingest.parsers import layout as L

    text = "Region,Sales,Units\nNorth,10,1\nSouth,20,2\nEast,30,3\nWest,40,4\nMid,50,5\n"
    real_find = L._find
    monkeypatch.setattr(L, "_find", lambda g, body: ([2],) + tuple(real_find(g, body)[1:]))
    sheet, warnings = only(text)
    assert sheet.layout["reader"] == "simple"
    assert any("the two readings of this table disagree" in w and "check the column names" in w for w in warnings)


@pytest.mark.parametrize("title", ["", "Energy by source\n"])
def test_a_key_label_over_years_is_a_header_not_a_preamble_row(tmp_path, title):
    """"Source, 2023, 2024" was taken for a key: value row and the first data row
    became the header (Coal | 100 | 90, SUM 383 instead of 473)."""
    energy = "Coal,100,90\nGas,200,210\nWind,50,70\nSolar,30,45\nHydro,60,58\n"
    sheet, warnings = only(title + "Source,2023,2024\n" + energy)
    assert list(sheet.frame.columns) == ["Source", "2023", "2024"] and len(sheet.frame) == 5 and warnings == []
    assert _sum(tmp_path, sheet, 2) == 473
    sheet, _ = only(title + "Geography,2022,2023\n" + "".join(f"County {i},{1000 + 7 * i},{1010 + 5 * i}\n" for i in range(6)))
    assert list(sheet.frame.columns) == ["Geography", "2022", "2023"] and len(sheet.frame) == 6
    assert _sum(tmp_path, sheet, 2) == 6135


def test_thousands_commas_in_a_file_that_writes_decimals_with_a_point(tmp_path):
    """N2: "1,234" and "12,500" became 1.234 and 12.5 next to 3.99 and 12.50."""
    sheet, warnings = only("Item;Price;Units\nA;3.99;1,234\nB;12.50;12,500\nC;7.25;980\n")
    assert _sum(tmp_path, sheet, 2) == 14714 and [row[2] for row in sheet.raw][1:] == ["1,234", "12,500", "980"]
    assert len(warnings) == 1 and "read as thousands separators" in warnings[0]
    sheet, warnings = only("Item;Price_eur;Price_usd;Units\nA;1,5;3.99;1,234\nB;2,25;12.50;12,500\nC;10,75;7.25;980\n")
    assert sheet.frame["Price_eur"].tolist() == ["1.5", "2.25", "10.75"] and _sum(tmp_path, sheet, 3) == 14714
    sheet, warnings = only("Item;Price;Units\nA;3.99;1,234\nB;12.50;2,5\n")
    assert sheet.frame["Units"].tolist() == ["1,234", "2,5"] and "kept as written" in warnings[0]


def test_a_blank_spacer_column_before_text_columns_blank_in_the_last_rows_is_one_table():
    """G: Manager and Status blank for the last two employees split the table; a right block with its own
    numbers and fewer rows (Region, Q1 | Team, People) is still a table of its own."""
    rows = "".join(f"Emp {i},{50 + i},{3 + i},," + (f"{['Ann', 'Bob', 'Cy'][i % 3]},{['on track', 'late'][i % 2]}"
                                                     if i < 6 else ",") + "\n" for i in range(8))
    sheet, warnings = only("Employee,Score,Tickets,,Manager,Status\n" + rows)
    assert list(sheet.frame.columns) == ["Employee", "Score", "Tickets", "Manager", "Status"] and len(sheet.frame) == 8
    assert sheet.layout["blank_columns"] == ["D"]
    doc = read_csv(b"Region,Q1,,Team,People\nNorth,1,,Eng,10\nSouth,3,,Ops,4\nEast,5,,,\n", "t.csv")
    assert [list(s.frame.columns) for s in doc.sheets] == [["Region", "Q1"], ["Team", "People"]]


def test_a_blank_spacer_column_inside_one_table_does_not_split_it():
    """Employee, Score, Tickets, <blank>, Manager, Status was read as two tables."""
    rows = "".join(f"Emp {i},{50 + i},{3 + i},,{['Ann', 'Bob', 'Cy'][i % 3]},{['on track', 'late'][i % 2]}\n" for i in range(8))
    sheet, warnings = only("Employee,Score,Tickets,,Manager,Status\n" + rows)
    assert list(sheet.frame.columns) == ["Employee", "Score", "Tickets", "Manager", "Status"] and len(sheet.frame) == 8
    assert sheet.layout["blank_columns"] == ["D"] and warnings == []


def test_stacked_tables_are_found_without_reading_every_remaining_row_for_each(monkeypatch):
    """Each table re-read the rest of the sheet to find where the next one starts (50 stacked
    tables of 2,000 rows: 45 s). The next table is now looked for in the next block of rows first."""
    from ingest.parsers import layout as L

    lines = []
    for t in range(10):
        lines += [f"Item{t},Amount,Count"] + [f"i{r},{r},{r % 7}" for r in range(50)] + [""]
    seen = []
    real = L.read_table
    monkeypatch.setattr(L, "read_table", lambda rows, *a, **k: seen.append(len(rows)) or real(rows, *a, **k))
    doc = read_csv(("\n".join(lines) + "\n").encode(), "s.csv")
    assert [len(s.frame) for s in doc.sheets] == [50] * 10
    assert sum(seen) < 4 * len(lines)  # every remaining row for each table was about 6.5 times the sheet


def test_a_second_table_never_takes_the_name_of_a_copied_sheet(tmp_path):
    """Excel names a copied sheet "Sales (2)", the name a second table of "Sales" got."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    for r in [["Region", "Q1", "Q2"], ["North", 1, 2], ["South", 3, 4], ["East", 5, 6], [], ["Team", "People", "Open"],
              ["Eng", 10, 2], ["Ops", 4, 1]]:
        ws.append(r)
    copy = wb.create_sheet("Sales (2)")
    for r in [["Region", "Q1", "Q2"], ["North", 100, 200], ["South", 300, 400], ["East", 500, 600]]:
        copy.append(r)
    wb.save(tmp_path / "book.xlsx")
    sheets = read_xlsx(tmp_path / "book.xlsx").sheets
    assert [(s.name, list(s.frame.columns)) for s in sheets] == [
        ("Sales", ["Region", "Q1", "Q2"]), ("Sales (table 2)", ["Team", "People", "Open"]), ("Sales (2)", ["Region", "Q1", "Q2"])]


def test_more_tables_than_the_cap_are_warned_and_the_cap_is_a_setting(monkeypatch):
    blocks = []
    for t in range(6):
        blocks += [[f"Item{t}", "Amount", "Count"], *[[f"i{r}", r, r * 2] for r in range(5)], []]
    tables = read_tables(blocks, [], 1, limit=4)
    assert len(tables) == 4 and any("more than 4 tables on this sheet" in w for w in tables[-1].warnings)
    monkeypatch.setenv("INGEST_MAX_TABLES_PER_SHEET", "3")
    assert len(read_tables(blocks, [], 1)) == 3


# --- key-only rows and section labels ---------------------------------------------------------------------


def test_a_key_only_row_in_a_table_grouped_by_capitals_is_data():
    sheet, warnings = only("Item,FY2024,FY2023\nASSETS,,\nCash,100,90\nGoodwill,,\nReceivables,50,40\nLIABILITIES,,\n"
                           "Payables,30,20\n")
    assert list(sheet.frame.index) == [3, 4, 5, 7] and sheet.frame.loc[4, "Item"] == "Goodwill"
    assert warnings == ['Sheet "t": rows 2, 6 read as section labels (ASSETS, LIABILITIES), kept in a Section column']


def test_a_product_code_in_capitals_is_a_data_row_not_a_section():
    sheet, _ = only("Product,Q1,Q2\nWidget A,10,20\nABC-123,,\nWidget B,11,21\nWidget C,12,22\n")
    assert "Section" not in sheet.frame.columns and list(sheet.frame.index) == [2, 3, 4, 5]


def test_bold_group_labels_in_a_workbook_tell_sections_from_key_only_rows(tmp_path):
    wb = Workbook()
    ws = wb.active
    for row in [["Region", "Sales", "Units"], ["North"], ["Boston", 100, 5], ["Cambridge"], ["NYC", 200, 6],
                ["South"], ["Miami", 50, 2], ["Atlanta", 70, 3]]:
        ws.append(row)
    for r in (2, 6):
        ws.cell(row=r, column=1).font = Font(bold=True)
    wb.save(tmp_path / "g.xlsx")
    [sheet] = read_xlsx(tmp_path / "g.xlsx").sheets
    assert list(sheet.frame.index) == [3, 4, 5, 7, 8] and sheet.frame.loc[4, "Region"] == "Cambridge"
    assert list(sheet.frame["Section"]) == ["North", "North", "North", "South", "South"]


# --- markers as the only content of a number column, placeholders as text, styles, page breaks -------------


@pytest.mark.parametrize("marker", ["<5", "*", "(S)", "Suppressed", "?", "TBD", "n.d."])
def test_a_number_column_with_markers_keeps_its_header_and_rows(tmp_path, marker):
    """Marker-only rows failed the type match, so the header walk absorbed up to four data rows
    ("County C0 C1 C2", "Cases 874 <5")."""
    sheet, warnings = only(f"Item,Qty\na,{marker}\nb,5\nc,{marker}\nd,7\n")
    assert list(sheet.frame.columns) == ["Item", "Qty"] and list(sheet.frame.index) == [2, 3, 4, 5]
    assert sheet.layout["header_rows"] == [1] and sheet.layout.get("reader", "layout") == "layout" and warnings == []
    assert typed(tmp_path, sheet)["qty"][0] == "BIGINT"


def test_two_suppressed_columns():
    rows = "".join(f"C{i},{'<5' if i % 2 else i * 10},{'(S)' if i % 3 else i}\n" for i in range(12))
    sheet, _ = only("County,Cases,Deaths\n" + rows)
    assert list(sheet.frame.columns) == ["County", "Cases", "Deaths"] and len(sheet.frame) == 12


def test_a_text_column_of_placeholders_keeps_them_as_text(tmp_path):
    sheet, _ = only("Order,Status,Amount\nO1,pending,10\nO2,unknown,20\nO3,TBD,30\n")
    db = tmp_path / "s.duckdb"
    [schema] = write_tables(db, [TableSource(sheet.name, "x", sheet.name, sheet.frame, True)])
    status = {c.sql_name: c for c in schema.columns}["status"]
    assert status.type == "text" and status.missing == 0 and status.empty == 0
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(f"SELECT status FROM {schema.table} ORDER BY _row").fetchall()
    assert [r[0] for r in rows] == ["pending", "unknown", "TBD"]


def grouped_workbook(tmp_path, style) -> list:
    wb = Workbook()
    ws = wb.active
    for row in [["Region", "Sales", "Units"], ["North"], ["Boston", 100, 5], ["Cambridge"], ["NYC", 200, 6], ["South"],
                ["Miami", 50, 2]]:
        ws.append(row)
    style(ws)
    wb.save(tmp_path / "g.xlsx")
    [sheet] = read_xlsx(tmp_path / "g.xlsx").sheets
    return list(sheet.frame.index)


def test_banded_fills_do_not_decide_sections_but_indents_do(tmp_path):
    def banded(ws):
        from openpyxl.styles import PatternFill
        for r in (2, 4, 6):
            ws.cell(row=r, column=1).fill = PatternFill("solid", fgColor="FFDDEEFF")

    def indented_items(ws):
        from openpyxl.styles import Alignment
        for r in (3, 4, 5, 7):
            ws.cell(row=r, column=1).alignment = Alignment(indent=1)

    assert grouped_workbook(tmp_path, banded) == [3, 5, 7]  # no style signal: groups as in a CSV, with a warning
    assert grouped_workbook(tmp_path, indented_items) == [3, 4, 5, 7]  # Cambridge is indented like the data: data


def test_a_bold_first_column_gives_no_style_signal(tmp_path):
    def all_bold(ws):
        for r in range(1, 8):
            ws.cell(row=r, column=1).font = Font(bold=True)
    assert grouped_workbook(tmp_path, all_bold) == [3, 5, 7]


@pytest.mark.parametrize("text, rows", [
    ("Name,Score\na,1\nb,2\nName,Score (cont.)\nc,3\nd,4\n", [2, 3, 5, 6]),
    ("Name,Score\na,1\nb,2\n\nPage 2\nName,Score\nc,3\nd,4\n", [2, 3, 7, 8]),
])
def test_continued_headers_and_page_numbers_are_not_data(text, rows):
    sheet, _ = only(text)
    assert list(sheet.frame.index) == rows and list(sheet.frame.columns) == ["Name", "Score"]


def test_years_repeated_under_groups_are_a_header_row():
    sheet, _ = only("Region,Revenue,Revenue,Cost,Cost\n,2024,2025,2024,2025\nNorth,10,12,5,6\nSouth,20,22,9,10\n")
    assert list(sheet.frame.columns) == ["Region", "Revenue 2024", "Revenue 2025", "Cost 2024", "Cost 2025"]
    assert list(sheet.frame.index) == [3, 4]


# --- units, zero-padded numbers, mixed dates ----------------------------------------------------------------


def test_a_units_row_without_parentheses():
    sheet, _ = only("Station,Temp,Rain,Wind\n,C,mm,km/h\nA,1.5,3,10\nB,2.5,4,12\nC,3.5,5,14\n")
    assert list(sheet.frame.columns) == ["Station", "Temp", "Rain", "Wind"]
    assert sheet.units == {"Temp": "C", "Rain": "mm", "Wind": "km/h"}


def test_zero_padded_months_hours_and_short_numbers_stay_numbers_but_long_codes_do_not(tmp_path):
    sheet, _ = only("Year,Month,Hour,Bin,Postal,Sales\n2024,01,00,01,02134,5\n2024,02,07,02,02139,6\n2024,12,23,10,10001,7\n")
    types = typed(tmp_path, sheet)
    assert [types[c][0] for c in ("month", "hour", "bin", "postal")] == ["BIGINT", "BIGINT", "BIGINT", "VARCHAR"]
    assert types["postal"][1] == "codes with leading zeros (kept as text)"


def test_dates_in_several_formats_are_named_with_the_formats(tmp_path):
    sheet, _ = only("Event,When\na,2024-01-05\nb,01/06/2024\nc,\"Jan 7, 2024\"\nd,2024-01-08\n")
    db = tmp_path / "m.duckdb"
    [schema] = write_tables(db, [TableSource(sheet.name, "x", sheet.name, sheet.frame, True)])
    when = {c.sql_name: c for c in schema.columns}["when_"]
    assert when.pattern == "dates as text in 3 formats (YYYY-MM-DD, M/D/YYYY, Mon D, YYYY)"
    assert "TRY_STRPTIME(col, ['%Y-%m-%d', '%m/%d/%Y', '%b %d, %Y'])" in when.warning
