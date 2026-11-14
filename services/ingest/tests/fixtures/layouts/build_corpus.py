"""Writes the sheet-layout regression corpus: adversarial sheets in groups r1- to r8-, first- and supp-, as
.csv and .xlsx files next to this script. expected.json holds each file's expected tables (names, columns,
types, row counts) and the real samples' expectations; tests/test_corpus.py checks them, and that the
layout reader is never worse than the simple reader on a sheet that one reads right.

    uv run python tests/fixtures/layouts/build_corpus.py      # from services/ingest; rewrites the files

Two cases are kept small, with the shape of the larger original: the 200-column sheet has 30 data rows
instead of 300. The files are committed; rebuilding them must give the same bytes for CSVs (xlsx files carry
a timestamp).
"""

import random
from datetime import date, datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).parent
CASES: dict[str, object] = {}  # file name -> bytes (CSV) or a builder (xlsx)


def csv(name: str, text: str | bytes, enc: str = "utf-8") -> None:
    CASES[f"{name}.csv"] = text if isinstance(text, bytes) else text.encode(enc)


def xlsx(name: str, build) -> None:
    CASES[f"{name}.xlsx"] = build


# --- group r1 ---------------------------------------------------------------------------------------------
xlsx("r1-empty_sheet", lambda ws: None)
xlsx("r1-title_only", lambda ws: ws.append(["Just a title"]))
csv("r1-single_row", "a,b,c\n")
csv("r1-header_and_one_row", "Name,City\nBob,Paris\n")
csv("r1-empty_csv", "")
csv("r1-only_blank_lines", "\n\n\n")
csv("r1-all_text_survey_long_headers",
    "Timestamp,What is your favourite programming language and why do you like it so much?,Which editor do you use most "
    "often for your daily work?\n1/5/2024 10:00:00,Python because it is readable,VS Code\n1/5/2024 10:05:00,Rust for "
    "safety,Neovim\n1/6/2024 09:00:00,Go,VS Code\n")
csv("r1-all_text_dup_headers", "Name,Comment,Comment\nBob,ok,fine\nAnn,meh,good\nJoe,yes,no\n")
csv("r1-all_text_header_value_reappears", "Status,Owner\nOpen,Ann\nStatus,Bob\nClosed,Joe\n")
csv("r1-numeric_headers_percentiles", "Measure,10,25,50,75,90\nIncome,20,35,50,80,120\nHeight,150,160,170,180,190\n"
    "Weight,50,60,70,80,95\n")
csv("r1-descending_year_headers_fin", "In millions,2024,2023,2022\nRevenue,100,90,80\nCost,60,55,50\nProfit,40,35,30\nTax,8,7,6\n")
csv("r1-descending_year_headers_blank_a1", ",2024,2023,2022\nRevenue,100,90,80\nCost,60,55,50\nProfit,40,35,30\n")
csv("r1-first_data_row_looks_like_years", "Item,Q1,Q2,Q3\nA,1850,1900,2000\nB,5,6,7\nC,8,9,10\nD,1,2,3\n")
csv("r1-start_end_years_then_open", "Name,Start,End\nA,1990,1995\nB,1991,1999\nC,1985,2001\nD,2003,\nE,2004,2010\n")
csv("r1-section_labels_col_a", "Region,Sales,Units\nNorth,,\nBoston,100,5\nNYC,200,6\nSouth,,\nMiami,50,2\nAtlanta,70,3\n")
csv("r1-label_only_data_row", "Product,Q1,Q2\nWidget A,10,20\nWidget B,11,21\nWidget C,,\nWidget D,12,22\nWidget E,13,23\n")
csv("r1-long_single_comment_row", "Id,Comment,Score\n1,good,5\n2,,\n,This customer wrote a very long comment that exceeds "
    "forty characters,\n3,bad,1\n4,ok,3\n")


def two_tables(ws):
    ws.append(["Sales by region"])
    ws.append(["Region", "Q1", "Q2"])
    for r in [["North", 1, 2], ["South", 3, 4], ["East", 5, 6]]:
        ws.append(r)
    ws.append([])
    ws.append(["Headcount by team"])
    ws.append(["Team", "People", "Open roles"])
    for r in [["Eng", 10, 2], ["Ops", 4, 1]]:
        ws.append(r)


xlsx("r1-two_tables_stacked", two_tables)


def side_by_side(ws):
    ws.append(["Region", "Q1", None, "Team", "People"])
    for r in [["North", 1, None, "Eng", 10], ["South", 3, None, "Ops", 4], ["East", 5, None, None, None]]:
        ws.append(r)


xlsx("r1-two_tables_side_by_side", side_by_side)


def merged(ws):
    ws.append(["Quarterly report"])
    ws.merge_cells("A1:E1")
    ws.append(["Region", "City", "Sales", None, "Note"])
    ws.append([None, None, "Q1", "Q2", None])
    for rng in ("A2:A3", "B2:B3", "C2:D2", "E2:E3"):
        ws.merge_cells(rng)
    ws.append(["North", "Boston", 1, 2, "x"])
    ws.append([None, "NYC", 3, 4, "y"])
    ws.merge_cells("A4:A5")
    ws.append(["South", "Miami", 5, 6, "merged note"])
    ws.append([None, "Atlanta", 7, 8, None])
    ws.merge_cells("A6:A7")
    ws.merge_cells("E6:E7")
    ws.append(["Total", None, 16, 20, None])
    ws.merge_cells("A8:B8")


xlsx("r1-merged_everywhere", merged)


def hidden(ws):
    for r in [["Name", "Value"], ["a", 1], ["helper", 999], ["b", 2]]:
        ws.append(r)
    ws.row_dimensions[3].hidden = True


xlsx("r1-hidden_rows", hidden)


def formulas(ws):
    for r in [["Name", "A", "B", "Sum"], ["x", 1, 2, "=B2+C2"], ["y", 3, 4, "=B3+C3"], ["z", 5, 6, "=B4+C4"]]:
        ws.append(r)


xlsx("r1-formulas_no_cache", formulas)


def serial_dates(ws):
    ws.append(["Day", "Value"])
    for i in range(10):
        ws.append([45292 + i, i])


xlsx("r1-excel_serial_dates_general", serial_dates)


def late_dates(ws):
    ws.append(["Name", "Shipped date", "Qty"])
    for i in range(3000):
        ws.append([f"n{i}", None if i < 2000 else datetime(2024, 1, 1 + i % 28), i])


xlsx("r1-date_col_first_2000_blank", late_dates)
csv("r1-csv_iso_dates_first_2000_blank", "\n".join(
    ["Name,Shipped date,Qty"] + [f"n{i},{'' if i < 2000 else '2024-01-%02d' % (1 + i % 28)},{i}" for i in range(3000)]) + "\n")


def mixed_dt(ws):
    for r in [["Name", "When"], ["a", datetime(2024, 1, 1, 10, 30)], ["b", date(2024, 1, 2)], ["c", "n/a"],
              ["d", datetime(2024, 1, 3)]]:
        ws.append(r)


xlsx("r1-mixed_datetime_date_na", mixed_dt)
csv("r1-yyyymmdd_in_date_col", "Event,Begin Date,Count\na,19800410,1\nb,20240101,2\nc,20001231,3\n")
csv("r1-yyyymmdd_invalid_one", "Event,Begin Date,Count\na,19800410,1\nb,20241301,2\nc,20001231,3\n")
csv("r1-csv_bom", "\ufeffName,Value\nA,1\nB,2\nC,3\n")
csv("r1-csv_semicolon_decimal_comma", "Name;Price;Qty\nA;1,5;3\nB;2,25;4\nC;10,75;5\n")
csv("r1-csv_semicolon_thousands", "Name;Amount\nA;1.234,56\nB;2.000,00\nC;300,10\n")
csv("r1-csv_tab", "Name\tValue\nA\t1\nB\t2\nC\t3\n")
csv("r1-csv_latin1", "Name,Ville\nA,Montr\xe9al\nB,Qu\xe9bec\nC,Z\xfcrich\n".encode("latin-1"))
csv("r1-csv_utf16", "Name,Value\nA,1\nB,2\n".encode("utf-16"))
csv("r1-csv_quoted_newlines", 'Name,Note\nA,"line1\nline2"\nB,"x"\nC,"y"\n')
csv("r1-csv_ragged", "a,b,c\n1,2\n3,4,5,6,7\n8\n")
csv("r1-csv_leading_zero_zip_no_id_name", "Town,Postal\nBoston,02134\nCambridge,02139\nNYC,10001\n")
csv("r1-csv_numbers_codes_count", "Team,Codes issued,Bugs\nA,10,3\nB,20,4\nC,30,5\n")
csv("r1-csv_percent_and_parens", "Item,Change\nA,(5.2%)\nB,3.1%\nC,-\nD,1,234\n")
csv("r1-csv_header_after_51_notes", "\n".join([f"Note line {i} with some text" for i in range(55)]
                                              + ["Name,Val", "a,1", "b,2", "c,3"]) + "\n")
csv("r1-csv_two_header_rows_units", "Station,Temp,Rain\n(id),(C),(mm)\nA,1.5,3\nB,2.5,4\nC,3.5,5\n")
csv("r1-csv_formula_like_cells", "Name,Value\n=1+2,1\n+SUM(1),2\n@x,3\n")

# --- group r2 ---------------------------------------------------------------------------------------------
csv("r2-note_above_header_2col", "Units: millions\nName,Val\na,1\nb,2\nc,3\n")
csv("r2-note_above_header_3col_colB", ",Units: millions,\nName,Val,Other\na,1,x\nb,2,y\nc,3,z\n")
csv("r2-two_notes_above_header_2col", "Source: survey\nUnits: millions\nName,Val\na,1\nb,2\nc,3\n")
csv("r2-balance_sheet", "Item,FY2024,FY2023\nASSETS,,\nCash,100,90\nReceivables,50,40\nLIABILITIES,,\nPayables,30,20\nDebt,70,60\n")
csv("r2-label_rows_col_a_2col", "Region,Sales\nNorth,\nBoston,100\nNYC,200\nSouth,\nMiami,50\n")


def late2(ws):
    ws.append(["Name", "Shipped date", "Qty"])
    for i in range(3000):
        ws.append([f"n{i}", None if i < 2500 else date(2024, 1, 1 + i % 28), i])


xlsx("r2-xlsx_date_first_2500_blank", late2)


def bigint(ws):
    ws.append(["Name", "Big"])
    for i in range(5):
        ws.append([f"n{i}", 2**60 + i])


xlsx("r2-xlsx_bigint_over_2_53", bigint)

# --- group r3 ---------------------------------------------------------------------------------------------
csv("r3-sections_and_key_only", "Region,Sales,Units\nNorth,,\nBoston,100,5\nCambridge,,\nNYC,200,6\nSouth,,\nMiami,50,2\n"
    "Atlanta,70,3\n")
csv("r3-balance_sheet_key_only", "Item,FY2024,FY2023\nASSETS,,\nCash,100,90\nGoodwill,,\nReceivables,50,40\nLIABILITIES,,\n"
    "Payables,30,20\n")
csv("r3-key_only_caps_code", "Product,Q1,Q2\nWidget A,10,20\nABC-123,,\nWidget B,11,21\nWidget C,12,22\n")
csv("r3-key_only_colon", "Measure,2023,2024\nRevenue,10,12\nNote:,,\nCost,5,6\nMargin,5,6\n")
csv("r3-footnote_two_cells", "State,Population,Area\nAlabama,5000000,52420\nAlaska,730000,665384\nArizona,7300000,113990\n\n"
    "Source:,U.S. Census Bureau,\n")
csv("r3-footnote_numbered", "State,Population\nAlabama,5000000\nAlaska,730000\nArizona,7300000\n1,Includes armed forces\n"
    "2,Preliminary\n")
csv("r3-same_header_repeated_after_blank", "Name,Score\na,1\nb,2\nc,3\n\nName,Score\nd,4\ne,5\nf,6\n")
csv("r3-blank_then_text_row", "Item,Amount,Status\nA,10,ok\nB,20,ok\n\nC,pending,review\nD,30,ok\nE,40,ok\n")


def three_tables(ws):
    rows = [["Table 1"], ["Region", "Q1", "Q2"], ["N", 1, 2], ["S", 3, 4], ["E", 5, 6], [], ["Team", "People"], ["Eng", 10],
            ["Ops", 4], ["HR", 2], [], ["Table 3"], ["Year", "Revenue", "Cost", "Profit"], [2022, 10, 5, 5], [2023, 12, 6, 6]]
    for r in rows:
        ws.append(r)


xlsx("r3-three_stacked_tables", three_tables)
csv("r3-units_row_plain", "Station,Temp,Rain,Wind\n,C,mm,km/h\nA,1.5,3,10\nB,2.5,4,12\nC,3.5,5,14\n")
csv("r3-mixed_date_formats", 'Event,When\na,2024-01-05\nb,01/06/2024\nc,"Jan 7, 2024"\nd,2024-01-08\n')
csv("r3-iso_dates_one_bad", "Event,When\na,2024-01-05\nb,2024-02-30\nc,2024-03-01\n")
csv("r3-iso_mixed_zone", "Event,When\na,2024-01-05T10:00:00Z\nb,2024-01-05T10:00:00+02:00\nc,2024-01-05\n")
csv("r3-serial_false_positive", "Year,Consolidated revenue,Candidates,Holiday sales,Days on market\n"
    "2021,25000,41000,30500,20100\n2022,31000,52000,33100,22000\n2023,40000,61000,35500,25000\n")


def wide(ws):
    random.seed(3)
    ws.append(["Wide report 2025"])
    ws.merge_cells("A1:F1")
    ws.append(["Name"] + [f"Group {i // 10}" if i % 10 == 0 else None for i in range(199)])
    for i in range(0, 199, 10):
        ws.merge_cells(start_row=2, start_column=2 + i, end_row=2, end_column=min(200, 11 + i))
    ws.append([None] + [f"M{i}" for i in range(199)])
    for r in range(30):
        ws.append([f"item {r}"] + [round(random.random() * 100, 2) for _ in range(199)])
    ws.append(["Note: values in thousands"])


xlsx("r3-wide_200_cols_two_row_header", wide)
csv("r3-years_header_and_year_values", "Country,2019,2020,2021\nA,1950,1960,1970\nB,1990,2000,2010\nC,5,6,7\n")
csv("r3-total_then_notes", "Item,Value\na,1\nb,2\nc,3\nTotal,6\nNote: values are preliminary and subject to revision\n")
csv("r3-transposed_kv", "Name,Alice\nAge,30\nCity,Paris\nRole,Engineer\n")
csv("r3-retail_like_key_only", "Code,Business,Jan,Feb\n,NOT ADJUSTED,,\n441,Motor vehicles,10,11\n442,Furniture,,\n"
    "443,Electronics,12,13\n,ADJUSTED,,\n441,Motor vehicles,9,10\n")
csv("r3-units_and_zero_codes", "Part,Weight,Bin\nA,1.5 kg,0012\nB,2 kg,0340\nC,3.25 kg,1000\n")
csv("r3-dup_suffix_collision", "Name,A,A,A_2\nx,1,2,3\ny,4,5,6\nz,7,8,9\n")

# --- group r4 ---------------------------------------------------------------------------------------------
csv("r4-status_all_placeholders", "Order,Status,Amount\nO1,pending,10\nO2,unknown,20\nO3,pending,30\nO4,TBD,40\nO5,unknown,50\n")
csv("r4-status_mixed", "Order,Status,Amount\nO1,pending,10\nO2,shipped,20\nO3,unknown,30\nO4,delivered,40\nO5,pending,50\n")
csv("r4-age_band_with_lt10", "Age band,People\n<10,120\n10-19,340\n20-29,410\n>=65,90\n")
csv("r4-answer_col_question_mark", "Question,Answer\nQ1,?\nQ2,yes\nQ3,no\nQ4,?\n")
csv("r4-first_row_pending_status", "Ticket,State,Hours\nT1,pending,3\nT2,closed,5\nT3,closed,2\nT4,open,7\nT5,closed,1\n")
csv("r4-cases_lt10_first_row", "County,Cases,Rate\nAlpha,<10,1.2\nBeta,120,3.4\nGamma,<10,0.8\nDelta,450,5.1\nEps,33,2.2\n")
csv("r4-dept_codes_two_digits", "Department,Head,Budget\n01,Ann,100\n02,Bo,200\n10,Cy,300\n07,Di,400\n")
csv("r4-shift_hhmm", "Employee,Start hour,End hour\nA,0800,1600\nB,0930,1730\nC,1200,2000\n")
csv("r4-day_of_month_and_month", "Year,Month,Day,Sales\n2024,01,05,10\n2024,02,17,20\n2024,11,30,30\n")
_GROUPED = [["Region", "Sales", "Units"], ["North", None, None], ["Boston", 100, 5], ["Cambridge", None, None],
            ["NYC", 200, 6], ["South", None, None], ["Miami", 50, 2], ["Atlanta", 70, 3]]


def banded(ws):
    fill = PatternFill("solid", fgColor="FFDDEEFF")
    for i, r in enumerate(_GROUPED, 1):
        ws.append(r)
        if i % 2 == 0:
            for c in range(1, 4):
                ws.cell(row=i, column=c).fill = fill


xlsx("r4-banded_fill_sections", banded)


def bold_labels_and_bold_keyonly(ws):
    for r in _GROUPED[:7]:
        ws.append(r)
    for rr in (2, 6, 4):
        ws.cell(row=rr, column=1).font = Font(bold=True)


xlsx("r4-bold_labels_and_bold_keyonly_row", bold_labels_and_bold_keyonly)


def bold_key_column(ws):
    for i, r in enumerate(_GROUPED[:7], 1):
        ws.append(r)
        ws.cell(row=i, column=1).font = Font(bold=True)


xlsx("r4-bold_whole_key_column", bold_key_column)


def indented(ws):
    for r in [["Item", "FY2024", "FY2023"], ["Assets", None, None], ["Cash", 100, 90], ["Goodwill", None, None],
              ["Receivables", 50, 40], ["Liabilities", None, None], ["Payables", 30, 20]]:
        ws.append(r)
    for rr in (3, 4, 5, 7):
        ws.cell(row=rr, column=1).alignment = Alignment(indent=1)


xlsx("r4-indented_line_items_keyonly", indented)
csv("r4-page_break_repeat_exact", "Name,Score\na,1\nb,2\nName,Score\nc,3\nd,4\n")
csv("r4-page_break_repeat_cont", "Name,Score\na,1\nb,2\nName,Score (cont.)\nc,3\nd,4\n")
csv("r4-page_break_repeat_blank_and_case", "Name,Score\na,1\nb,2\n\nNAME,SCORE\nc,3\nd,4\n")
csv("r4-page_break_repeat_blank_page_title", "Name,Score\na,1\nb,2\n\nPage 2\nName,Score\nc,3\nd,4\n")


def date_names(ws):
    ws.append(["Year", "Update", "Mandate", "Today count", "Visitors per day", "Due date", "DT", "Day 1 revenue"])
    random.seed(4)
    for y in range(2020, 2025):
        ws.append([y] + [random.randint(20001, 73000) for _ in range(7)])


xlsx("r4-date_name_rule_xlsx", date_names)
csv("r4-date_name_rule_csv", "Year,Visitors per day,Due date\n2021,25000,45300\n2022,31000,45400\n2023,40000,45500\n")


def grouped_years(ws):
    ws.append(["Region", "Revenue", None, "Cost", None])
    ws.merge_cells("B1:C1")
    ws.merge_cells("D1:E1")
    ws.append([None, 2024, 2025, 2024, 2025])
    for r in [["North", 10, 12, 5, 6], ["South", 20, 22, 9, 10], ["East", 30, 33, 14, 15]]:
        ws.append(r)


xlsx("r4-grouped_nonmonotonic_years", grouped_years)
csv("r4-grouped_years_csv_unmerged", "Region,Revenue,Revenue,Cost,Cost\n,2024,2025,2024,2025\nNorth,10,12,5,6\n"
    "South,20,22,9,10\nEast,30,33,14,15\n")
csv("r4-heavy_suppression", "County,Cases\n" + "".join(f"C{i},{'<5' if i % 3 else i * 7}\n" for i in range(30)))
csv("r4-question_marks_numeric", "Item,Qty\na,?\nb,5\nc,?\nd,7\n")

# --- group r5: the safety net (the simple reader takes over when the layout reader doubts its header) ------
csv("r5-header_row_with_a_number", "Region,Sales,Units\n2,(USD),(n)\nNorth,10,1\nSouth,20,2\nEast,30,3\nWest,40,4\n")
csv("r5-question_marks_minimal", "Item,Qty\na,?\nb,5\nc,?\nd,7\n")

# --- group r6: two-cell preamble rows, merged titles, number grammar, single columns ------------------------
_years = list(range(1960, 1966))
_wdi = ('"Data Source","World Development Indicators",\n\n"Last Updated Date","2024-06-28",\n\n'
        '"Country Name","Country Code","Indicator Name","Indicator Code",' + ",".join(f'"{y}"' for y in _years) + ',\n')
for _i, (_name, _code) in enumerate([("Aruba", "ABW"), ("Africa Eastern and Southern", "AFE"), ("Afghanistan", "AFG"),
                                     ("Africa Western and Central", "AFW"), ("Angola", "AGO"), ("Albania", "ALB")]):
    _wdi += (f'"{_name}","{_code}","Population, total","SP.POP.TOTL",'
             + ",".join(f'"{50000 + _i * 1000 + k}"' for k in range(6)) + ",\n")
csv("r6-worldbank_wdi", _wdi)
csv("r6-fred_style", "Title:,Gross Domestic Product\nSeries ID:,GDP\nSource:,U.S. Bureau of Economic Analysis\n"
    "Units:,Billions of Dollars\nFrequency:,Quarterly\n\nobservation_date,GDP\n"
    + "".join(f"20{10 + i // 4}-{1 + 3 * (i % 4):02d}-01,{15000 + 50 * i}\n" for i in range(12)))
_states = "".join(f"State {i},{100 + i},{110 + i},{120 + i}\n" for i in range(10))
csv("r6-source_then_text_header", "Source,Census Bureau\nState,Population,Area,Density\n" + _states)
csv("r6-source_then_year_header", "Source,Census Bureau\nState,2021,2022,2023\n" + _states)
csv("r6-prepared_by_and_as_of", "Prepared by,Finance team\nAs of,2024-06-30\n\nRegion,Revenue,Cost,Margin\n" + _states)
csv("r6-release_date_then_years", "Release date,2024-05-01\nState,2021,2022,2023\n" + _states)
csv("r6-title_release_date_then_years", "Population by state\nRelease date,2024-05-01\nState,2021,2022,2023\n" + _states)
csv("r6-units_then_years", "Units,Thousands\nState,2021,2022,2023\n" + _states)
csv("r6-source_blank_then_years", "Source,Census Bureau\n\nState,2021,2022,2023\n" + _states)
csv("r6-title_then_years", "Population by state\nState,2021,2022,2023\n" + _states)
csv("r6-single_column_text", "Name\nAnn\nBo\nCy\nDee\n")
csv("r6-single_column_numbers", "Visitors\n120\n135\n150\n90\n")
csv("r6-numbers_only", "1,2,3\n4,5,6\n7,8,9\n")
csv("r6-signed_percent", "Region,Change\nNorth,+9.5%\nSouth,-0.4%\nEast,+10.2%\nWest,+2.0%\nMid,-1.1%\n")
csv("r6-unicode_minus", "Region,Change\nNorth,4.5\nSouth,\u22123.4\nEast,\u221210.2\nWest,2.0\n")
csv("r6-scientific", "Gene,p_value\nA,1.2e-05\nB,0.03\nC,3.4E-08\nD,0.5\nE,2e-3\n")
csv("r6-semicolon_decimal_comma_3dp", "Name;Price;Qty\nA;1,250;3\nB;2,500;4\nC;10,750;5\n")
csv("r6-dmy_dates", "Event,When\na,13/04/2024\nb,05/06/2024\nc,21/12/2024\n")
csv("r6-xy_coordinates", "Point,X,Y\nA,1.5,2.5\nB,3.1,4.2\nC,5.0,6.0\nD,7.5,8.5\n")
csv("r6-standings_w_l", "Team,W,L,PCT\nBoston,50,32,0.610\nNew York,48,34,0.585\nToronto,40,42,0.488\nChicago,30,52,0.366\n")
csv("r6-regions_na", "Product,NA,EU,APAC\nA,10,20,30\nB,11,21,31\nC,12,22,32\n")
csv("r6-title_then_standings", "2024 standings\nTeam,W,L\nBoston,50,32\nNew York,48,34\nToronto,40,42\n")


def merged_title_release_date(ws):
    ws.append(["Population estimates by region, 2020-2024"])
    ws.merge_cells("A1:E1")
    ws.append(["Release date", date(2024, 5, 1)])
    ws.append(["Region", "2020", "2021", "2022", "2023"])
    for i in range(12):
        ws.append([f"Region {i}", 100 + i, 110 + i, 120 + i, 130 + i])


def merged_title_table_number(ws):
    ws.append(["Population estimates by region"])
    ws.merge_cells("A1:E1")
    ws.append(["Table 4", 2024])
    ws.append(["Region", "Pop", "Area", "Density", "Growth"])
    for i in range(12):
        ws.append([f"Region {i}", 100 + i, 110 + i, 120 + i, 130 + i])


def merged_title_sparse(ws):
    ws.append(["Incidents reported by site"])
    ws.merge_cells("A1:E1")
    ws.append(["Site", "Jan", "Feb", "Mar", "Apr"])
    for i in range(12):
        ws.append([f"Site {i}"] + [None if (i + c) % 3 else i * c for c in range(4)])


def merged_title_units_with_stray_cell(ws):
    ws.append(["Sales by region, 2024"])
    ws.merge_cells("A1:C1")
    for r in [["Region", "Sales", "Units"], [2, "(USD)", "(n)"], ["North", 10, 1], ["South", 20, 2], ["East", 30, 3],
              ["West", 40, 4]]:
        ws.append(r)


def merged_title_second_header_row(ws):
    ws.append(["Hospital admissions by county"])
    ws.merge_cells("A1:D1")
    ws.append(["County", "Admissions", "Rate", "Beds"])
    ws.append([None, "n", "per 100k", 2023])
    for i in range(8):
        ws.append([f"County {i}", 100 + i, 1.5 + i, 20 + i])


def percent_format(ws):
    ws.append(["State", "Growth rate (%)"])
    for state, value in [("A", 0.052), ("B", 0.013), ("C", 0.101)]:
        ws.append([state, value])
        ws.cell(row=ws.max_row, column=2).number_format = "0.0%"


def copied_sheet_name(ws):
    ws.title = "Sales"
    for r in [["Region", "Q1", "Q2"], ["North", 1, 2], ["South", 3, 4], ["East", 5, 6], [], ["Team", "People", "Open"],
              ["Eng", 10, 2], ["Ops", 4, 1]]:
        ws.append(r)
    copy = ws.parent.create_sheet("Sales (2)")
    for r in [["Region", "Q1", "Q2"], ["North", 100, 200], ["South", 300, 400], ["East", 500, 600]]:
        copy.append(r)


xlsx("r6-merged_title_release_date", merged_title_release_date)
xlsx("r6-merged_title_table_number", merged_title_table_number)
xlsx("r6-merged_title_sparse", merged_title_sparse)
xlsx("r6-merged_title_units_with_stray_cell", merged_title_units_with_stray_cell)
xlsx("r6-merged_title_second_header_row", merged_title_second_header_row)
xlsx("r6-percent_format", percent_format)
xlsx("r6-copied_sheet_name", copied_sheet_name)

# --- group r7: narrow tables, one year in a units row, ";" decimal points, a blank spacer column -----------
def _narrow(years: list[int]) -> str:
    return ("State," + ",".join(map(str, years)) + "\n"
            + "".join(f"State {i}," + ",".join(str(100 * (k + 1) + i) for k in range(len(years))) + "\n" for i in range(10)))


csv("r7-units_over_three_columns", "Units,Thousands\n" + _narrow([2023, 2024]))
csv("r7-source_over_three_columns", "Source,Census Bureau\n" + _narrow([2023, 2024]))
csv("r7-release_date_over_three_columns", "Release date,2024-05-01\n" + _narrow([2023, 2024]))
csv("r7-table_number_row_over_three_columns", "Table 4,Population by state,2024\n" + _narrow([2022, 2023]))
csv("r7-table_number_row_over_four_columns", "Table 4,Population by state,2024\n" + _narrow([2021, 2022, 2023]))
csv("r7-table_number_row_over_five_columns", "Table 4,Population by state,2024\n" + _narrow([2020, 2021, 2022, 2023]))
csv("r7-unknown_key_over_text_header", "Agency,Dept of Labor\nState,Workers,Openings,Hires\n"
    + "".join(f"State {i},{1000 + i},{50 + i},{40 + i}\n" for i in range(10)))
csv("r7-source_over_two_columns", "Source,FRED\nYear,Value\n" + "".join(f"{2000 + i},{1.5 + i}\n" for i in range(10)))
csv("r7-units_row_with_a_year", "County,Admissions,Rate,Beds\n,n,per 100k,2023\n"
    + "".join(f"County {i},{100 + i},{1.5 + i},{20 + i}\n" for i in range(8)))
csv("r7-units_row_with_a_year_titled", "Hospital admissions by county\nCounty,Admissions,Rate,Beds\n,n,per 100k,2023\n"
    + "".join(f"County {i},{100 + i},{1.5 + i},{20 + i}\n" for i in range(8)))
csv("r7-population_over_2020", "Region,Population,Area,Density\n,2020,sq km,per sq km\n"
    + "".join(f"R{i},{5000 + i},{300 + i},{16 + i}\n" for i in range(10)))
csv("r7-semicolon_decimal_points", "Item;Weight_kg;Price\nA;0.125;3.99\nB;2.5;12.50\nC;1.250;7.25\nD;10.75;1.20\n")
csv("r7-semicolon_decimal_points_three_places", "Item;Weight_kg\nA;0.125\nB;2.500\nC;1.250\n")
csv("r7-semicolon_dot_either_way", "Item;Weight\nA;1.250\nB;2.500\nC;3.750\n")
csv("r7-semicolon_mixed_marks", "Item;Weight\nA;1,5\nB;2.75\nC;3,25\n")
csv("r7-spacer_column_inside_one_table", "Employee,Score,Tickets,,Manager,Status\n" + "".join(
    f"Emp {i},{50 + i},{3 + i},,{['Ann', 'Bob', 'Cy'][i % 3]},{['on track', 'late', 'done'][i % 3]}\n" for i in range(8)))
csv("r7-spacer_before_a_notes_column", "Account,2024,2023,,Comment\n"
    + "".join(f"Account {i},{1000 + i},{900 + i},,Note {i}\n" for i in range(6)))
csv("r7-side_by_side_same_rows", "Item,Qty,,Supplier,City\n" + "".join(f"Item {i},{i},,Supplier {i % 2},Berlin\n" for i in range(5)))

# --- group r8: a key label over years, thousands commas, a sparse right block -------------------------------
_energy = "Coal,100,90\nGas,200,210\nWind,50,70\nSolar,30,45\nHydro,60,58\n"
_counties = "".join(f"County {i},{1000 + 7 * i},{1010 + 5 * i}\n" for i in range(6))
csv("r8-source_over_two_years", "Source,2023,2024\n" + _energy)
csv("r8-source_over_two_years_titled", "Energy by source\nSource,2020,2021\n" + _energy)
csv("r8-geography_over_two_years", "Geography,2022,2023\n" + _counties)
csv("r8-geography_over_two_years_titled", "Population by county\nGeography,2022,2023\n" + _counties)
csv("r8-semicolon_thousands_commas_point_file", "Item;Price;Units\nA;3.99;1,234\nB;12.50;12,500\nC;7.25;980\n")
csv("r8-semicolon_mixed_locales", "Item;Price_eur;Price_usd;Units\nA;1,5;3.99;1,234\nB;2,25;12.50;12,500\nC;10,75;7.25;980\n")
csv("r8-spacer_sparse_right_block", "Employee,Score,Tickets,,Manager,Status\n" + "".join(
    f"Emp {i},{50 + i},{3 + i},," + (f"{['Ann', 'Bob', 'Cy'][i % 3]},{['on track', 'late', 'done'][i % 3]}" if i < 6 else ",")
    + "\n" for i in range(8)))

# --- group first (the order of the random draws matters) ------------------------------------------------
random.seed(5)


def body(n: int) -> str:
    return "".join(f"item{i},{random.randint(1, 999)},{random.randint(1, 99)}\n" for i in range(n))


csv("first-first_row_TBD", "Name,Value,Count\nitem_first,TBD,5\n" + body(50))
csv("first-first_row_unknown_and_date", "Name,Shipped,Qty\nfirst,unknown,5\n"
    + "".join(f"n{i},2024-01-{1 + i % 28:02d},{i}\n" for i in range(50)))
csv("first-first_row_iso_rest_mdy", "Event,When,Qty\nfirst,2024-01-05,1\n"
    + "".join(f"e{i},01/{1 + i % 28:02d}/2024,{i}\n" for i in range(50)))
csv("first-numbered_footnotes_50", "State,Population\n" + "".join(f"S{i},{random.randint(10**5, 10**7)}\n" for i in range(50))
    + "1,Includes armed forces\n2,Preliminary\n")
csv("first-first_two_rows_text_in_num", "Name,Value,Count\nA,n.d.,5\nB,pending,6\n" + body(50))
csv("first-county_suppressed_first_row", "County,Cases,Rate\nAlpha,<10,1.2\nBeta,25,3.4\nGamma,40,5.1\nDelta,12,2.2\n")

# --- group supp: one number column of markers, ten tables per marker; and two suppressed columns ------
for marker, tag in (("<5", "lt5"), ("*", "star"), ("(S)", "S"), ("Suppressed", "word"), ("?", "q")):
    for seed in range(10):
        random.seed(seed)
        rows = "".join(f"C{i},{marker if random.random() < 0.45 else random.randint(5, 900)}\n" for i in range(40))
        csv(f"supp-{tag}-{seed}", "County,Cases\n" + rows)
random.seed(11)
csv("supp-two_columns", "County,Cases,Deaths\n" + "".join(
    f"C{i},{'<5' if random.random() < 0.4 else random.randint(5, 900)},{'(S)' if random.random() < 0.6 else random.randint(0, 40)}\n"
    for i in range(40)))


def write() -> None:
    for name, case in CASES.items():
        path = HERE / name
        if isinstance(case, bytes):
            path.write_bytes(case)
        else:
            wb = Workbook()
            case(wb.active)
            wb.save(path)


if __name__ == "__main__":
    write()
    print(f"{len(CASES)} files in {HERE}")
