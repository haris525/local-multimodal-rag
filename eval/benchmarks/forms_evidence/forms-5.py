# Washington, D.C.: 2017 age-adjusted heart disease death rate (CDC) and the population change from the April 1,
# 2020 estimates base to July 1, 2025 (Census). The trap: both files write the city as "District of Columbia";
# "Washington" is the state.
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
cdc = pd.read_csv(B + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
print("CDC states containing 'Washington' or 'Columbia':",
      [s for s in cdc.State.unique() if "Washington" in s or "Columbia" in s])
hd = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "Heart disease")].set_index("State")

ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
pop = {r[0].lstrip("."): (r[1], r[7]) for r in ws.iter_rows(min_row=10, max_row=62, values_only=True) if r[0]}

for name, label in [("District of Columbia", "gold"), ("Washington", "naive (Washington state)")]:
    base, p25 = pop[name]
    print(f"\n{label}: {name}")
    print("  heart disease 2017: rate", hd.loc[name, "Age-adjusted Death Rate"], "deaths", hd.loc[name, "Deaths"])
    print("  population:", base, "->", p25, "change", p25 - base, "pct", round((p25 / base - 1) * 100, 4))
