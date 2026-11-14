# The three most populous states on July 1, 2025 (Census) and their combined deaths from all causes in 2017
# (CDC). The trap: the Census table lists the United States and the four regions as rows, so a naive top three
# is United States, South and West, none of which is a state.
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

ws = openpyxl.load_workbook(DATA + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
rows = [(r[7], r[0]) for r in ws.iter_rows(min_row=5, values_only=True) if r[0] and isinstance(r[7], int)]
ranked = sorted(rows, reverse=True)
print("largest rows:", ranked[:8])
states = [(v, k.lstrip(".")) for v, k in ranked if k.startswith(".") and k != ".Puerto Rico"][:3]
cdc = pd.read_csv(DATA + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
d17 = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "All causes")].set_index("State").Deaths
for v, k in states:
    print(k, v, "| deaths 2017:", d17[k])
print("\ngold:", [k for _, k in states], "| deaths 2017 combined", sum(int(d17[k]) for _, k in states))
naive = [k for _, k in ranked[:3]]
print("naive (top three over every row):", naive, "| in the CDC file:", {k: int(d17[k]) for k in naive if k in d17})
