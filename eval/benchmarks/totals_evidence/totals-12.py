# The Census region with the largest population gain from the April 1, 2020 estimates base to July 1, 2025,
# and its share of the national gain. The trap is the other way round: the answer needs the region rows and the
# United States row, which are aggregates of the states (the table does not say which state is in which
# region), while a naive max over the aggregate rows picks the United States.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

ws = openpyxl.load_workbook(DATA + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
rows = {r[0]: r for r in ws.iter_rows(min_row=5, values_only=True) if r[0] and isinstance(r[7], int)}
regions = ["Northeast", "Midwest", "South", "West"]
gain = {k: rows[k][7] - rows[k][1] for k in regions + ["United States"]}
for k, g in gain.items():
    print(k, rows[k][1], "->", rows[k][7], "gain", g)
assert sum(gain[k] for k in regions) == gain["United States"]
best = max(regions, key=gain.get)
print("\ngold:", best, "+", gain[best], "| share of the national gain", gain["United States"],
      "=", round(gain[best] / gain["United States"] * 100, 3), "%")
naive = max(gain, key=gain.get)
print("naive (max over every aggregate row):", naive, "+", gain[naive])
states = [k for k in rows if k.startswith(".") and k != ".Puerto Rico"]
print("naive (aggregate rows dropped): only", len(states), "state rows remain, with no region column")
