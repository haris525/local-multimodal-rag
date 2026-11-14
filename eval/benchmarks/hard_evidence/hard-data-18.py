# EIA monthly data: solar and coal net generation, January-July 2026 vs January-July 2025 (the annual sheet has
# no 2026 row, so the months must be summed).
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

P = DATA + '/excel/eia-us-electricity-generation-by-source-1949-2026.xlsx'
wbk = openpyxl.load_workbook(P, data_only=True)
ws = wbk["Monthly Data"]
cols = ["month", "coal", "petroleum", "gas", "otherfossil", "nuclear", "pumped", "hydro", "wood", "waste", "geo",
        "solar", "wind", "total"]
m = pd.DataFrame(list(ws.iter_rows(min_row=13, values_only=True)), columns=cols)
for c in cols[1:]:
    m[c] = pd.to_numeric(m[c], errors="coerce")
m["month"] = pd.to_datetime(m.month)
print("last month in the sheet:", m.month.max().date())
annual_years = [r[0] for r in wbk["Annual Data"].iter_rows(min_row=13, values_only=True) if isinstance(r[0], int)]
print("last year in Annual Data:", max(annual_years))
a = m[(m.month >= "2025-01-01") & (m.month <= "2025-07-01")]
b = m[(m.month >= "2026-01-01") & (m.month <= "2026-07-01")]
assert len(a) == 7 and len(b) == 7
for c in ("solar", "coal", "wind", "total"):
    x, y = a[c].sum(), b[c].sum()
    print(c, "Jan-Jul 2025:", round(x, 3), "Jan-Jul 2026:", round(y, 3), "change:", round(y - x, 3), "pct:", round((y / x - 1) * 100, 3))
