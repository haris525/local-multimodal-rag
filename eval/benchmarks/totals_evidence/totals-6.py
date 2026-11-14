# The kind of business with the highest 2025 sales (Census retail workbook, not-adjusted annual totals) and its
# share of retail and food services sales. The trap: the sheet's first rows are aggregates. A naive max picks
# "Retail and food services sales, total"; dropping only rows with "total" in the label still leaves the
# "excl" aggregates (no keyword) and GAFO(1), so it picks "Retail sales and food services excl gasoline stations".
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
print("rows without a NAICS code:", [(n, t) for c, n, t in nsa if not c])
total = next(t for c, n, t in nsa if n == "Retail and food services sales, total")
kinds = sorted(((t, n, c) for c, n, t in nsa if c), reverse=True)
for k in kinds[:5]:
    print(k)
best = kinds[0]
print("\ngold:", best[1], "(NAICS", best[2], ")", best[0], "million; share of", total, "=",
      round(best[0] / total * 100, 3), "%")
naive = max(nsa, key=lambda x: x[2])
print("naive (max over every row):", naive[1], naive[2], "=", round(naive[2] / total * 100, 3), "%")
kw = max((x for x in nsa if "total" not in x[1].lower()), key=lambda x: x[2])
print("naive (rows with 'total' dropped):", kw[1], kw[2], "=", round(kw[2] / total * 100, 3), "%")
