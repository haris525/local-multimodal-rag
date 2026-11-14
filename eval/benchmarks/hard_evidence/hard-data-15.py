# Tropical cyclone costs by full decade of begin date (NOAA): the costliest decade, its share of all 1980-2024
# tropical cyclone cost, and its average cost per event. CPI-adjusted, USD millions (2024 dollars).
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/csv'
n = pd.read_csv(B + "/noaa-us-billion-dollar-disasters-1980-2024.csv")
CPI = "CPI-Adjusted Cost (USD millions; 2024 dollars)"
n["decade"] = (n["Begin Date"] // 10000) // 10 * 10
tc = n[n.Disaster == "Tropical Cyclone"]
d = tc.groupby("decade")[CPI].agg(["sum", "count"])
d["mean"] = d["sum"] / d["count"]
print(d.round(1).to_string())
full = d.loc[[1980, 1990, 2000, 2010]]
best = full["sum"].idxmax()
total = tc[CPI].sum()
print("all tropical cyclones 1980-2024:", round(total, 1), "events", len(tc))
print("costliest full decade:", best, round(full.loc[best, "sum"], 1), "share %", round(full.loc[best, "sum"] / total * 100, 3),
      "events", int(full.loc[best, "count"]), "mean", round(full.loc[best, "mean"], 1))
print(tc[tc.decade == best][["Name", CPI]].sort_values(CPI, ascending=False).to_string(index=False))
