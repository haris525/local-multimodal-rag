# The share of all U.S. deaths in 2017 that the ten causes in the CDC file account for, and the deaths from
# other causes. The trap runs both ways: the "All causes" row is the denominator and must not be added to the
# causes; dropping it leaves no denominator (a share of 100%), adding it gives 174%. Summing every State row
# doubles the counts ("United States" is the sum of the states and DC) but not the share.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

cdc = pd.read_csv(DATA + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
us = cdc[(cdc.Year == 2017) & (cdc.State == "United States")].set_index("Cause Name").Deaths
print(us.sort_values(ascending=False).to_string())
total = int(us["All causes"])
ten = int(us.drop("All causes").sum())
print("\ngold: ten causes", ten, "of", total, "=", round(ten / total * 100, 3), "% | other causes", total - ten)
naive = int(us.sum())
print("naive (every cause row, All causes included):", naive, "=", round(naive / total * 100, 3), "%")
print("naive (All causes dropped, the ten causes taken as all deaths): 100 % | other causes 0")
rows = cdc[cdc.Year == 2017]
both = int(rows[rows["Cause Name"] == "All causes"].Deaths.sum())
part = int(rows[rows["Cause Name"] != "All causes"].Deaths.sum())
print("naive (every State row, United States included):", part, "of", both, "=", round(part / both * 100, 3),
      "% | other causes", both - part)
