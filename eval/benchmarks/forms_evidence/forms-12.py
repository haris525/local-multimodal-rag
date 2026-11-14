# Seven Pacific island countries ranked by 2024 GDP per capita (World Bank); the two highest. The trap: the file
# writes Nauru as "Naoero" (ISO code NRU), so a lookup by the English name finds nothing and the ranking silently
# loses it.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + "/csv"
wb = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
print("rows named 'Nauru':", (wb.country == "Nauru").sum(), "| rows with ISO code NRU:",
      sorted(wb[wb.iso3_code == "NRU"].country.unique()))

asked = ["Fiji", "Marshall Islands", "Naoero", "Palau", "Samoa", "Tonga", "Tuvalu"]
r = wb[(wb.year == 2024) & wb.country.isin(asked)].sort_values("gdp_per_capita_current_usd", ascending=False)
print(r[["country", "iso3_code", "gdp_per_capita_current_usd"]].to_string(index=False))
print("gold: top two", r.country.iloc[0], r.gdp_per_capita_current_usd.iloc[0], "|", r.country.iloc[1],
      r.gdp_per_capita_current_usd.iloc[1])

naive = r[r.country.isin(["Fiji", "Marshall Islands", "Nauru", "Palau", "Samoa", "Tonga", "Tuvalu"])]
print("naive (names as asked, Nauru missing): top two", naive.country.iloc[0],
      naive.gdp_per_capita_current_usd.iloc[0], "|", naive.country.iloc[1], naive.gdp_per_capita_current_usd.iloc[1])
