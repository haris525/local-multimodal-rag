# Five most populous states on July 1, 2025 (Census) -> highest and lowest 2017 all-cause age-adjusted death rate (CDC).
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
rows = [r for r in ws.iter_rows(min_row=5, max_row=62, values_only=True) if r[0]]
pop = pd.DataFrame(rows, columns=["area", "base2020", 2020, 2021, 2022, 2023, 2024, 2025])
pop["area"] = pop.area.str.lstrip(".")
st = pop[~pop.area.isin(["United States", "Northeast", "Midwest", "South", "West", "Puerto Rico"])]
top = st.sort_values(2025, ascending=False).head(6)
print(top[["area", "base2020", 2025]].to_string(index=False))
names = top.area.head(5).tolist()
print("top 5:", names)
cdc = pd.read_csv(B + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
r = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "All causes") & cdc.State.isin(names)]
print(r[["State", "Deaths", "Age-adjusted Death Rate"]].sort_values("Age-adjusted Death Rate", ascending=False).to_string(index=False))
