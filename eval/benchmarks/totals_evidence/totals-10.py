# The energy source whose U.S. net generation grew the most from 2024 to 2025 in absolute terms (EIA annual
# data). The trap: the sheet's last column is "Electricity Net Generation Total (including from sources not
# shown)", so a naive max over the columns picks the total.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

ws = openpyxl.load_workbook(DATA + "/excel/eia-us-electricity-generation-by-source-1949-2026.xlsx",
                            data_only=True)["Annual Data"]
rows = list(ws.iter_rows(values_only=True))
head = next(r for r in rows if r[0] == "Annual Total")
names = [str(h).replace("Electricity Net Generation ", "").replace(", All Sectors", "") for h in head]
year = {r[0]: r for r in rows if isinstance(r[0], int)}
a, b = year[2024], year[2025]
change = sorted(((b[i] - a[i], names[i], a[i], b[i]) for i in range(1, len(head))), reverse=True)
for c in change:
    print(f"{c[1]}: {c[2]} -> {c[3]}, change {c[0]:.3f} ({(c[3] / c[2] - 1) * 100:.2f} %)")
sources = [c for c in change if not c[1].startswith("Total")]
best = sources[0]
print("\ngold:", best[1], "+", round(best[0], 3), "million kWh,", best[2], "->", best[3],
      "(", round((best[3] / best[2] - 1) * 100, 3), "% )")
pct = max(sources, key=lambda c: c[3] / c[2] if c[2] > 0 else 0)
print("largest percentage rise:", pct[1], round((pct[3] / pct[2] - 1) * 100, 3), "%")
print("naive (max over every column):", change[0][1], "+", round(change[0][0], 3))
