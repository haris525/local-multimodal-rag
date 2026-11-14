# 2024 sales of the Census category for general merchandise, apparel and accessories, furniture and other
# department-store-type goods, and its share of retail sales excluding food services (Census retail workbook,
# not-adjusted annual totals). The trap: the workbook writes the category only as the abbreviation "GAFO(1)",
# explained in footnote (1); "General merchandise stores" is one part of it.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
ws = openpyxl.load_workbook(B + "/excel/census-monthly-retail-sales-1992-2026.xlsx", data_only=True)["2024"]
rows = list(ws.iter_rows(values_only=True))
assert str(rows[4][14]).strip() == "TOTAL"
labels = [str(r[1]).strip() if r[1] is not None else "" for r in rows]
start, end = labels.index("NOT ADJUSTED"), next(i for i, x in enumerate(labels) if x.startswith("ADJUSTED"))
nsa = {str(r[1]).strip(): r[14] for r in rows[start + 1:end]}
by_code = {str(r[0]).strip(): r[14] for r in rows[start + 1:end] if r[0] not in (None, "")}
print("footnote:", next(str(r[0]) for r in rows if str(r[0]).startswith("(1)")))
print("labels containing 'GAFO' or 'general merchandise':",
      [k for k in nsa if "GAFO" in k or "general merchandise" in k.lower()])

gafo, retail = nsa["GAFO(1)"], nsa["Retail sales, total"]
parts = ["442", "443", "448", "451", "452", "4532"]
print("sum of the footnote's NAICS codes:", sum(by_code[c] for c in parts), "| GAFO(1) row:", gafo)
print("retail sales, total (excl. food services):", retail)
print("\ngold: GAFO", gafo, "million; share", round(gafo / retail * 100, 3), "%")
gm = nsa["General merchandise stores"]
print("naive (General merchandise stores only):", gm, "million; share", round(gm / retail * 100, 3), "%")
dept = [r[14] for r in rows[start + 1:end] if str(r[1]).strip() == "Department stores"]
print("naive (rows named 'Department stores', NAICS 4522 and 45221):", dept, "sum", sum(dept))
