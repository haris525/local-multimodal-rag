# Highest 2017 influenza and pneumonia age-adjusted death rate (CDC) -> population change from the April 2020 base
# to July 1, 2025 (Census), and the state with the next-largest percentage decline.
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
cdc = pd.read_csv(B + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
f = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "Influenza and pneumonia") & (cdc.State != "United States")]
print(f.sort_values("Age-adjusted Death Rate", ascending=False).head(3)[["State", "Deaths", "Age-adjusted Death Rate"]].to_string(index=False))
ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
rows = [r for r in ws.iter_rows(min_row=5, max_row=62, values_only=True) if r[0]]
pop = pd.DataFrame(rows, columns=["area", "base2020", 2020, 2021, 2022, 2023, 2024, 2025])
pop["area"] = pop.area.str.lstrip(".")
st = pop[~pop.area.isin(["United States", "Northeast", "Midwest", "South", "West", "Puerto Rico"])].copy()
st["change"] = st[2025] - st.base2020
st["pct"] = (st[2025] / st.base2020 - 1) * 100
print(len(st), "states incl. DC")
print(st.sort_values("pct").head(4)[["area", "base2020", 2025, "change", "pct"]].to_string(index=False))
