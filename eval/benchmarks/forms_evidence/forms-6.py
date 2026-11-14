# U.S. deaths from chronic lower respiratory diseases against cerebrovascular diseases, 1999 to 2017 (CDC): the
# years the first exceeded the second, the first such year, and both counts for 2017. The trap: the short cause
# names are "CLRD" and "Stroke"; the long names carry ICD-10 codes ("Chronic lower respiratory diseases (J40-J47)").
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + "/csv"
c = pd.read_csv(B + "/cdc-leading-causes-of-death-us-1999-2017.csv")
print(c[["113 Cause Name", "Cause Name"]].drop_duplicates().to_string(index=False))
for col in ["Cause Name", "113 Cause Name"]:
    for name in ["Chronic lower respiratory diseases", "Cerebrovascular diseases"]:
        print(f"exact '{name}' in {col}:", (c[col] == name).sum(), "rows")

us = c[c.State == "United States"].pivot(index="Year", columns="Cause Name", values="Deaths")
states = c[c.State != "United States"].groupby(["Year", "Cause Name"]).Deaths.sum().unstack()
assert (states[["CLRD", "Stroke"]] == us[["CLRD", "Stroke"]]).all().all()  # the states add up to the U.S. row
t = us[["CLRD", "Stroke"]].copy()
t["CLRD > Stroke"] = t.CLRD > t.Stroke
print("\n" + t.to_string())
years = t.index[t["CLRD > Stroke"]].tolist()
print("\ngold: years", len(years), "| first", years[0], "| 2017 CLRD", t.loc[2017, "CLRD"], "Stroke",
      t.loc[2017, "Stroke"])
flu = us["Influenza and pneumonia"] > us["Stroke"]
print("naive (influenza and pneumonia taken as the respiratory cause): years above stroke:", int(flu.sum()))
