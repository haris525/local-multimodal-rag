# Turkey vs Iran: M5+ event counts (USGS 2025) and 2024 GDP per capita (World Bank, where the names differ).
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/csv'
u = pd.read_csv(B + "/usgs-earthquakes-m5-plus-2025.csv")
wb = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
pl = u.place.fillna("")
for name in ["Turkey", "Iran"]:
    sel = u[pl.str.endswith(", " + name)]
    print(name, "ends with:", len(sel), "contains:", pl.str.contains(name).sum(), "max mag:", sel.mag.max())
print("World Bank rows named 'Turkey' or 'Iran' literally:", wb[wb.country.isin(["Turkey", "Iran"])].shape[0])
r = wb[(wb.year == 2024) & wb.country.isin(["Turkiye", "Iran, Islamic Rep."])]
print(r[["country", "gdp_per_capita_current_usd", "internet_users_pct_of_population"]].to_string(index=False))
g = r.set_index("country")["gdp_per_capita_current_usd"]
print("ratio Turkiye / Iran:", round(g["Turkiye"] / g["Iran, Islamic Rep."], 3))
