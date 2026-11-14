# Countries named in the place text of M7.0+ events (USGS 2025) -> lowest 2024 GDP per capita (World Bank).
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/csv'
u = pd.read_csv(B + "/usgs-earthquakes-m5-plus-2025.csv")
wb = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
m7 = u[u.mag >= 7.0][["time", "mag", "place"]]
print("M7.0+ events:", len(m7))
print(m7.to_string())
# Place text -> World Bank economy name. Strict: a country named in the place text.
strict = {"Cayman Islands": "Cayman Islands", "Burma (Myanmar)": "Myanmar", "Tonga": "Tonga",
          "Russia": "Russian Federation", "Philippines": "Philippines", "Japan": "Japan"}
# Loose readings some analysts may add (a state, or a region with no country in the text).
loose = {"Alaska": "United States", "Tibetan Plateau": "China", "Macquarie Island": "Australia",
         "Hubbard Glacier": "Canada"}
g24 = wb[wb.year == 2024].set_index("country")["gdp_per_capita_current_usd"]
rows = []
for key, econ in {**strict, **loose}.items():
    hits = m7[m7.place.str.contains(key, regex=False)]
    if len(hits):
        rows.append((econ, key in strict, len(hits), hits.mag.max(), g24.get(econ)))
t = pd.DataFrame(rows, columns=["economy", "strict", "events", "max_mag", "gdp_pc_2024"]).sort_values("gdp_pc_2024")
print(t.to_string(index=False))
print("lowest (strict):", t[t.strict].iloc[0].to_dict())
print("lowest (strict + loose):", t.iloc[0].to_dict())
print(m7[m7.place.str.contains("Myanmar")].to_string())
