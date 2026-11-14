# Growth from 2020 to 2025 of: annual not-adjusted retail and food services sales (Census retail), total U.S. net
# generation (EIA annual), and U.S. resident population on July 1 (Census).
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/excel'
wb = openpyxl.load_workbook(B + "/census-monthly-retail-sales-1992-2026.xlsx", data_only=True)


def retail_total(year):
    rows = list(wb[str(year)].iter_rows(values_only=True))
    assert str(rows[4][14]).strip() == "TOTAL"
    start = next(i for i, r in enumerate(rows) if str(r[1]).strip() == "NOT ADJUSTED")
    return next(r[14] for r in rows[start:] if str(r[1]).strip() == "Retail and food services sales, total")


ws = openpyxl.load_workbook(B + "/eia-us-electricity-generation-by-source-1949-2026.xlsx", data_only=True)["Annual Data"]
gen = {r[0]: r[13] for r in ws.iter_rows(min_row=13, values_only=True) if isinstance(r[0], int)}
us = next(openpyxl.load_workbook(B + "/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
          .iter_rows(min_row=5, max_row=5, values_only=True))
series = {
    "retail sales (USD millions, NSA)": (retail_total(2020), retail_total(2025)),
    "net generation (million kWh)": (gen[2020], gen[2025]),
    "population July 1": (us[2], us[7]),  # columns: area, April 2020 base, 2020, 2021, ..., 2025
}
print("check: April 1, 2020 base", us[1], "| July 1, 2020", us[2], "| July 1, 2025", us[7])
for k, (a, b) in series.items():
    print(k, a, "->", b, "growth %", round((b / a - 1) * 100, 3))
