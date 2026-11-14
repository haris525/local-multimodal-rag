# Total U.S. retail and food services sales in 2025 (Census retail workbook, not-adjusted annual total) and the
# amount per U.S. resident on July 1, 2025 (Census population). The trap is the other way round: both answers
# are total rows, "Retail and food services sales, total" and "United States". Dropping them and summing the
# rest counts the NAICS sub-lines (and the regions) more than once.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

ws = openpyxl.load_workbook(DATA + "/excel/census-monthly-retail-sales-1992-2026.xlsx", data_only=True)["2025"]
rows = list(ws.iter_rows(values_only=True))
assert str(rows[4][14]).strip() == "TOTAL"
labels = [str(r[1]).strip() if r[1] is not None else "" for r in rows]
start, end = labels.index("NOT ADJUSTED"), next(i for i, x in enumerate(labels) if x.startswith("ADJUSTED"))
nsa = [(str(r[0]).strip() if r[0] is not None else "", str(r[1]).strip(), r[14]) for r in rows[start + 1:end]
       if isinstance(r[14], (int, float))]
total = next(t for c, n, t in nsa if n == "Retail and food services sales, total")
three = sum(t for c, n, t in nsa if c.isdigit() and len(c) == 3)
print("Retail and food services sales, total:", total, "| sum of the 13 three-digit NAICS rows:", three)

wp = openpyxl.load_workbook(DATA + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
geo = {r[0]: r[7] for r in wp.iter_rows(min_row=5, values_only=True) if r[0] and isinstance(r[7], int)}
us = geo["United States"]
print("United States, July 1, 2025:", us)
print("\ngold:", total, "million; per resident", round(total * 1e6 / us, 2))
coded = sum(t for c, n, t in nsa if c)
print("naive (total rows dropped, every NAICS row summed):", coded, "million; per resident", round(coded * 1e6 / us, 2))
nokw = sum(t for c, n, t in nsa if "total" not in n.lower())
print("naive (rows with 'total' dropped, the rest summed):", nokw, "million; per resident",
      round(nokw * 1e6 / us, 2))
allgeo = sum(v for k, v in geo.items() if k != "United States")
print("naive (population as the sum of every row but the United States):", allgeo, "| per resident",
      round(total * 1e6 / allgeo, 2))
