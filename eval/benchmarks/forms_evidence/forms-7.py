# Grocery store sales in 2024 (Census retail workbook, not-adjusted annual total) and per U.S. resident on
# July 1, 2024 (Census population). The trap: a case-insensitive match on "grocery" also takes "Supermarkets and
# other grocery (except convenience) stores", a part of grocery stores, and counts it twice.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
ws = openpyxl.load_workbook(B + "/excel/census-monthly-retail-sales-1992-2026.xlsx", data_only=True)["2024"]
rows = list(ws.iter_rows(values_only=True))
assert str(rows[4][14]).strip() == "TOTAL"
labels = [str(r[1]).strip() if r[1] is not None else "" for r in rows]
start, end = labels.index("NOT ADJUSTED"), next(i for i, x in enumerate(labels) if x.startswith("ADJUSTED"))
nsa = rows[start + 1:end]
hits = [(r[0], str(r[1]).strip(), r[14]) for r in nsa if "grocery" in str(r[1]).lower()]
for h in hits:
    print("NAICS", h[0], "|", h[1], "|", h[2])
grocery = next(t for code, name, t in hits if name == "Grocery stores")

wp = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
us = next(r for r in wp.iter_rows(values_only=True) if r[0] == "United States")
pop24 = us[6]
print("U.S. resident population July 1, 2024:", pop24)
print("\ngold: grocery stores", grocery, "million; per resident", round(grocery * 1e6 / pop24, 2))
naive = sum(t for _, _, t in hits)
print("naive (every row with 'grocery'):", naive, "million; per resident", round(naive * 1e6 / pop24, 2))
