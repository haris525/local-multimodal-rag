# The cause with the most deaths in Florida in 2017 (CDC) and its margin over the second-ranked cause.
# The trap: the file has an "All causes" row next to the ten causes, so a naive max over the rows picks the
# total, and its margin is measured against heart disease.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

cdc = pd.read_csv(DATA + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
fl = cdc[(cdc.Year == 2017) & (cdc.State == "Florida")].sort_values("Deaths", ascending=False)
print(fl[["Cause Name", "Deaths", "Age-adjusted Death Rate"]].to_string(index=False))
causes = fl[fl["Cause Name"] != "All causes"]
print("ten causes:", len(causes), "| their sum:", causes.Deaths.sum())
first, second = causes.iloc[0], causes.iloc[1]
print("\ngold:", first["Cause Name"], first.Deaths, "| second:", second["Cause Name"], second.Deaths,
      "| margin", first.Deaths - second.Deaths)
top, runner = fl.iloc[0], fl.iloc[1]
print("naive (max over every row):", top["Cause Name"], top.Deaths, "| margin over", runner["Cause Name"],
      top.Deaths - runner.Deaths)
