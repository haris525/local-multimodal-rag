from openpyxl import Workbook

from ingest.parsers.sheets import read_csv, read_xlsx


def test_xlsx_skips_title_rows_and_joins_two_row_headers(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Quarterly export"
    ws.append([])
    ws.append(["Order", "Customer", None, "Sale", None])
    ws.merge_cells("B3:C3")
    ws.merge_cells("D3:E3")
    ws.append(["Order ID", "Name", "Region", "Amount", "Date"])
    ws.append(["A-1", "Ann", "North", 10.5, None])
    ws.append([None, None, None, None, None])
    ws.append(["A-2", "Bo", "South", 7, None])
    wb.save(tmp_path / "s.xlsx")

    sheet = read_xlsx(tmp_path / "s.xlsx").sheets[0]
    assert list(sheet.frame.columns) == ["Order ID", "Customer Name", "Customer Region", "Sale Amount", "Sale Date"]
    assert list(sheet.frame.index) == [5, 7]  # Excel row numbers, blank row dropped
    assert sheet.preamble == "Quarterly export"


def test_xlsx_reports_formulas_without_saved_values(tmp_path):
    wb = Workbook()
    wb.active.append(["a", "b", "sum"])
    wb.active.append([1, 2, "=A2+B2"])
    wb.save(tmp_path / "f.xlsx")
    doc = read_xlsx(tmp_path / "f.xlsx")
    assert doc.sheets[0].blank_formulas == 1
    assert "formula cells have no saved value" in doc.warnings[0]


def test_csv_sniffs_semicolon_dialect():
    doc = read_csv("sku;qty\nINV-1;4\nINV-2;9\n".encode(), "stock.csv")
    sheet = doc.sheets[0]
    assert (sheet.name, list(sheet.frame.columns), list(sheet.frame["qty"])) == ("stock", ["sku", "qty"], ["4", "9"])
