# The five states with the most Alzheimer's disease deaths in 2017 (CDC) and their combined deaths.
# The trap: the State column holds a "United States" row (the national total) next to the 50 states and DC,
# so a naive top five puts the nation first and drops the fifth state.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

cdc = pd.read_csv(DATA + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
alz = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "Alzheimer's disease")].sort_values("Deaths", ascending=False)
print("rows:", len(alz), "| United States row:", int(alz[alz.State == "United States"].Deaths.iloc[0]),
      "| sum of the other 51 rows:", int(alz[alz.State != "United States"].Deaths.sum()))
states = alz[alz.State != "United States"].head(5)
print(states[["State", "Deaths"]].to_string(index=False))
print("\ngold: top five", list(states.State), "combined", int(states.Deaths.sum()))
naive = alz.head(5)
print("naive (top five over every row):", list(naive.State), "combined", int(naive.Deaths.sum()))
