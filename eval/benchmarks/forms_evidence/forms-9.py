# Hurricane Fran and Hurricane Ida (NOAA billion-dollar disasters): combined CPI-adjusted cost and deaths, and how
# many times costlier the costlier one was. The trap: "Hurricane Fran" is a substring of "Hurricane Frances" and
# "Hurricane Francine", and "Hurricane Ida" of "Hurricane Idalia".
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + "/csv"
n = pd.read_csv(B + "/noaa-us-billion-dollar-disasters-1980-2024.csv")
COST = "CPI-Adjusted Cost (USD millions; 2024 dollars)"
cols = ["Name", "Begin Date", COST, "Deaths"]

gold = n[n.Name.str.match(r"Hurricane (Fran|Ida) \(")]
print(gold[cols].to_string(index=False))
c = gold.set_index("Name")[COST]
print("gold: cost", round(gold[COST].sum(), 1), "| deaths", gold.Deaths.sum(), "| Ida / Fran",
      round(c.iloc[1] / c.iloc[0], 3))

for label, mask in [("'Hurricane Fran' or 'Hurricane Ida' as substrings",
                     n.Name.str.contains("Hurricane Fran") | n.Name.str.contains("Hurricane Ida")),
                    ("'Fran' or 'Ida' as substrings", n.Name.str.contains("Fran") | n.Name.str.contains("Ida")),
                    ("'fran' or 'ida', case-insensitive",
                     n.Name.str.contains("fran", case=False) | n.Name.str.contains("ida", case=False))]:
    m = n[mask]
    print(f"\nnaive ({label}): {len(m)} rows, cost {round(m[COST].sum(), 1)}, deaths {m.Deaths.sum()}")
    print(m[cols].to_string(index=False))
