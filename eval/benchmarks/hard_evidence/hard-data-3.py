# Year(s) with the most billion-dollar disasters by begin date (NOAA) -> wind + solar share of net generation (EIA annual).
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
n = pd.read_csv(B + "/csv/noaa-us-billion-dollar-disasters-1980-2024.csv")
n["year"] = n["Begin Date"] // 10000
c = n.groupby("year").size().sort_values(ascending=False)
print("top years by count:")
print(c.head(5).to_string())
top = sorted(c[c == c.max()].index.tolist())
print("tied top years:", top, "count", c.max())
n["end_year"] = n["End Date"] // 10000
print("events that start and end in different years:")
print(n[n.year != n.end_year][["Name", "Begin Date", "End Date"]].to_string(index=False))
ws = openpyxl.load_workbook(B + "/excel/eia-us-electricity-generation-by-source-1949-2026.xlsx", data_only=True)["Annual Data"]
hdr = [str(h) for h in next(ws.iter_rows(min_row=11, max_row=11, values_only=True))]
rows = {r[0]: dict(zip(hdr, r)) for r in ws.iter_rows(min_row=13, values_only=True) if isinstance(r[0], int)}
W = "Electricity Net Generation From Wind, All Sectors"
S = "Electricity Net Generation From Solar, All Sectors"
T = "Electricity Net Generation Total (including from sources not shown), All Sectors"
for y in top:
    r = rows[y]
    print(y, "wind", r[W], "solar", r[S], "total", r[T], "share %", round((r[W] + r[S]) / r[T] * 100, 3))
