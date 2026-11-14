# M5+ events located in Dominica (USGS 2025), and Dominica's 2024 GDP per capita and internet use (World Bank).
# The trap: "Dominica" is a substring of "Dominican Republic", in both files.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + "/csv"
u = pd.read_csv(B + "/usgs-earthquakes-m5-plus-2025.csv")
wb = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
pl = u.place.fillna("")

print("place names Dominica as a whole word:", pl.str.contains(r"\bDominica\b").sum())
naive = u[pl.str.contains("Dominica")]
print("naive: place contains 'Dominica':", len(naive))
print(naive[["time", "mag", "place"]].to_string(index=False))

w24 = wb[wb.year == 2024]
print("\nWorld Bank rows whose name contains 'Dominica' (2024):")
hits = w24[w24.country.str.contains("Dominica")]
print(hits[["country", "iso3_code", "gdp_per_capita_current_usd", "internet_users_pct_of_population"]]
      .to_string(index=False))
d = w24[w24.country == "Dominica"].iloc[0]
print("\ngold: 0 events; Dominica", d.gdp_per_capita_current_usd, round(d.internet_users_pct_of_population, 4))
print("naive: 3 events; averaging both rows gives GDP", round(hits.gdp_per_capita_current_usd.mean(), 2),
      "and internet", round(hits.internet_users_pct_of_population.mean(), 4))
