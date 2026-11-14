# Cause with the largest % rise in the U.S. age-adjusted death rate 1999-2017 (CDC) -> state with the highest
# 2017 rate for it -> that state's July 1, 2025 population (Census).
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
cdc = pd.read_csv(B + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
us = cdc[(cdc.State == "United States") & (cdc["Cause Name"] != "All causes")]
p = us.pivot(index="Cause Name", columns="Year", values="Age-adjusted Death Rate")[[1999, 2017]]
p["pct"] = (p[2017] / p[1999] - 1) * 100
print(p.sort_values("pct", ascending=False).to_string())
cause = p.pct.idxmax()
s = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == cause) & (cdc.State != "United States")]
s = s.sort_values("Age-adjusted Death Rate", ascending=False)
print(s.head(3)[["State", "Deaths", "Age-adjusted Death Rate"]].to_string(index=False))
state = s.iloc[0].State
ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
for r in ws.iter_rows(min_row=5, max_row=62, values_only=True):
    if r[0] and r[0].lstrip(".") == state:
        print(state, "base 2020:", r[1], "July 2024:", r[6], "July 2025:", r[7])
