# South Asia (World Bank region): largest gain in internet use, in percentage points, from 2014 to 2024, and that
# economy's 2024 GDP per capita.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/csv'
w = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
sa = w[w.region == "South Asia"]
print("South Asia economies:", sorted(sa.country.unique()))
p = sa.pivot(index="country", columns="year", values="internet_users_pct_of_population")[[2014, 2024]]
p["gain_pp"] = p[2024] - p[2014]
print(p.sort_values("gain_pp", ascending=False).to_string())
best = p.gain_pp.idxmax()
g = sa[(sa.country == best) & (sa.year == 2024)].gdp_per_capita_current_usd.iloc[0]
print("largest gain:", best, round(p.loc[best, "gain_pp"], 4), "GDP per capita 2024:", g)
