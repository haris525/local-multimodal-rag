# Virginia and Maryland together: July 1, 2025 population (Census) and 2017 deaths from all causes (CDC).
# The trap: "Virginia" is a substring of "West Virginia" (and the Census names carry a leading dot, so an exact
# match on "Virginia" finds nothing there).
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
raw = {r[0]: r for r in ws.iter_rows(min_row=10, max_row=62, values_only=True) if r[0]}
print("Census names as written:", [k for k in raw if "Virginia" in k or "Maryland" in k])
print("exact match on 'Virginia':", "Virginia" in raw)
pop25 = {k.lstrip("."): r[7] for k, r in raw.items()}

cdc = pd.read_csv(B + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
d17 = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "All causes")].set_index("State").Deaths

want = ["Virginia", "Maryland"]
for s in want + ["West Virginia"]:
    print(s, "population July 1, 2025:", pop25[s], "| deaths 2017:", d17[s])
print("\ngold: population", sum(pop25[s] for s in want), "| deaths", sum(d17[s] for s in want))

naive = [s for s in pop25 if "Virginia" in s or "Maryland" in s]
print("naive (substring 'Virginia' or 'Maryland'):", naive)
print("naive: population", sum(pop25[s] for s in naive), "| deaths", sum(d17[s] for s in naive))
