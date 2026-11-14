# Warmest year on the NASA chart (2024, a coarse fact read from the image) -> costliest NOAA disaster that began in 2024.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/csv'
n = pd.read_csv(B + "/noaa-us-billion-dollar-disasters-1980-2024.csv")
CPI = "CPI-Adjusted Cost (USD millions; 2024 dollars)"
n["year"] = n["Begin Date"] // 10000
print("chart fact used: warmest year = 2024 (about +1.28 C); 2025 about +1.18 C")
y = n[n.year == 2024].sort_values(CPI, ascending=False)
print(y.head(3)[["Name", "Disaster", "Begin Date", CPI, "Deaths"]].to_string(index=False))
top = y.iloc[0]
total = y[CPI].sum()
print("2024 events:", len(y), "total CPI-adjusted (USD millions):", round(total, 1))
print("top:", top["Name"], top[CPI], "deaths", top["Deaths"], "share %", round(top[CPI] / total * 100, 3))
