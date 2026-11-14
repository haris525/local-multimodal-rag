# M5+ events whose place description names China (USGS 2025), and the first year China's GDP per capita exceeded
# $10,000 with that year's GDP per capita and internet use (World Bank). The trap: a case-insensitive "china"
# also matches "Nichinan, Japan", and the World Bank names "Hong Kong SAR, China" and "Macao SAR, China" contain
# "China".
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + "/csv"
u = pd.read_csv(B + "/usgs-earthquakes-m5-plus-2025.csv")
wb = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
pl = u.place.fillna("")

china = u[pl.str.contains(r"\bChina\b")]
print(china[["time", "mag", "place"]].to_string(index=False))
print("gold: events naming China:", len(china))
naive = u[pl.str.contains("china", case=False)]
print("naive (case-insensitive 'china'):", len(naive), "| extra:", sorted(set(naive.place) - set(china.place)))
print("(events written as Xizang or the Tibetan Plateau do not name China:",
      pl.str.contains("Xizang|Tibetan").sum(), "events)")

c = wb[wb.country == "China"].set_index("year")
first = c.index[c.gdp_per_capita_current_usd > 10000].min()
print("\ngold: first year above $10,000:", first, "| GDP", c.loc[first, "gdp_per_capita_current_usd"],
      "| internet", c.loc[first, "internet_users_pct_of_population"])
print(c.loc[first - 1:first, ["gdp_per_capita_current_usd", "internet_users_pct_of_population"]].to_string())
like = wb[wb.country.str.contains("China")]
print("\nWorld Bank names containing 'China':", sorted(like.country.unique()))
over = like[like.gdp_per_capita_current_usd > 10000].sort_values("year")
print("naive: first year above $10,000:", over.year.min(), "|",
      over[over.year == over.year.min()][["country", "gdp_per_capita_current_usd"]].to_string(index=False))
