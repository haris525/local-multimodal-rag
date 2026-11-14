# False premise: Katrina is not the deadliest event in the NOAA file (Maria is). Katrina's deaths, and CDC U.S.
# influenza and pneumonia deaths in 2005, the year Katrina began.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/csv'
n = pd.read_csv(B + "/noaa-us-billion-dollar-disasters-1980-2024.csv")
print(n.sort_values("Deaths", ascending=False).head(3)[["Name", "Begin Date", "Deaths"]].to_string(index=False))
k = n[n.Name.str.contains("Katrina")]
print(k[["Name", "Begin Date", "Deaths"]].to_string(index=False))
cdc = pd.read_csv(B + "/cdc-leading-causes-of-death-us-1999-2017.csv")
r = cdc[(cdc.Year == 2005) & (cdc.State == "United States") & (cdc["Cause Name"] == "Influenza and pneumonia")]
print(r[["Year", "Cause Name", "Deaths", "Age-adjusted Death Rate"]].to_string(index=False))
