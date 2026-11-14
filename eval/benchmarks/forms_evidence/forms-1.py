# India or Peru: more M5+ events in the USGS 2025 file? Then the 2024 GDP per capita (World Bank) of the one with
# more. The trap: "India" is also a substring of the Indian Ocean ridges and the South Indian Ocean.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + "/csv"
u = pd.read_csv(B + "/usgs-earthquakes-m5-plus-2025.csv")
wb = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
pl = u.place.fillna("")

india_country = pl.str.endswith(", India")
india_region = pl.str.endswith(", India region")
peru = pl.str.endswith(", Peru")
print("India (place ends with ', India'):", india_country.sum())
print("India region (Nicobar Islands, India region):", india_region.sum())
print("India total with the region:", (india_country | india_region).sum())
print("Peru (place ends with ', Peru'):", peru.sum(), "| contains 'Peru':", pl.str.contains("Peru").sum())

naive = pl.str.contains("India")
print("\nnaive: place contains 'India':", naive.sum())
print(u[naive & ~(india_country | india_region)].place.str.replace(r"^\d+ km \w+ of ", "", regex=True)
      .value_counts().to_string())

g = wb[wb.year == 2024].set_index("country").gdp_per_capita_current_usd
print("\nGDP per capita 2024: Peru", g["Peru"], "| India", g["India"])
print("gold: Peru has more (24 vs 20, or 22 with India region); Peru's GDP per capita", g["Peru"])
print("naive: India has more (", naive.sum(), "vs", peru.sum(), "); India's GDP per capita", g["India"])
