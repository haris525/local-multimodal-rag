# EIA monthly data: first month in which wind generated more than coal, and the first month in which solar
# generated more than conventional hydroelectric power.
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

P = DATA + '/excel/eia-us-electricity-generation-by-source-1949-2026.xlsx'
ws = openpyxl.load_workbook(P, data_only=True)["Monthly Data"]
cols = ["month", "coal", "petroleum", "gas", "otherfossil", "nuclear", "pumped", "hydro", "wood", "waste", "geo",
        "solar", "wind", "total"]
hdr = next(ws.iter_rows(min_row=11, max_row=11, values_only=True))
print("columns used:", hdr[1], "|", hdr[7], "|", hdr[11], "|", hdr[12])
m = pd.DataFrame(list(ws.iter_rows(min_row=13, values_only=True)), columns=cols)
for c in cols[1:]:
    m[c] = pd.to_numeric(m[c], errors="coerce")  # "Not Available" -> NaN
m["month"] = pd.to_datetime(m.month)
w = m[m.wind > m.coal]
print("first wind > coal:", w.iloc[0][["month", "wind", "coal"]].to_dict())
print("months with wind > coal:", [d.strftime("%Y-%m") for d in w.month])
s = m[m.solar > m.hydro]
print("first solar > conventional hydro:", s.iloc[0][["month", "solar", "hydro"]].to_dict())
