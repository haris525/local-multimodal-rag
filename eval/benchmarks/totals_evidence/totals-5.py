# The state with the largest population gain from the April 1, 2020 estimates base to July 1, 2025 (Census),
# and its 2017 age-adjusted death rate from all causes (CDC). The trap: the United States and the four regions
# are rows of the same table, so a naive max picks the United States (and then the CDC's national row).
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

ws = openpyxl.load_workbook(DATA + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
rows = [r for r in ws.iter_rows(min_row=5, values_only=True) if r[0] and isinstance(r[7], int)]
gain = sorted(((r[7] - r[1], r[0], r[1], r[7]) for r in rows), reverse=True)
for g in gain[:8]:
    print(g)
states = [g for g in gain if g[1].startswith(".") and g[1] != ".Puerto Rico"]
best = states[0]
name = best[1].lstrip(".")
cdc = pd.read_csv(DATA + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
rate = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "All causes")].set_index("State")["Age-adjusted Death Rate"]
deaths = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "All causes")].set_index("State").Deaths
print("\ngold:", name, "+", best[0], "(", best[2], "to", best[3], ", +", round(best[0] / best[2] * 100, 3), "% )",
      "| 2017 all-cause rate", rate[name], "per 100,000 (", deaths[name], "deaths )")
top = gain[0]
print("naive (max over every row):", top[1], "+", top[0], "| CDC all-cause rate", rate[top[1]],
      "| next row:", gain[1][1], "+", gain[1][0])
