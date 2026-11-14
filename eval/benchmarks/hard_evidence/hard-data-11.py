# 2024 retail and food services sales per resident (Census retail, not adjusted, annual TOTAL / Census July 1, 2024
# population) as a share of 2024 U.S. GDP per capita (World Bank).
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
wb = openpyxl.load_workbook(B + "/excel/census-monthly-retail-sales-1992-2026.xlsx", data_only=True)
rows = list(wb["2024"].iter_rows(values_only=True))
assert str(rows[4][14]).strip() == "TOTAL"
nsa_start = next(i for i, r in enumerate(rows) if str(r[1]).strip() == "NOT ADJUSTED")
sales = next(r[14] for r in rows[nsa_start:] if str(r[1]).strip() == "Retail and food services sales, total")
print("2024 retail and food services sales, not adjusted, USD millions:", sales)
ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
pop = next(r[6] for r in ws.iter_rows(min_row=5, max_row=5, values_only=True))
print("U.S. population July 1, 2024:", pop)
w = pd.read_csv(B + "/csv/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
gdp = w[(w.country == "United States") & (w.year == 2024)].gdp_per_capita_current_usd.iloc[0]
print("U.S. GDP per capita 2024 (current US$):", gdp)
per = sales * 1e6 / pop
print("sales per resident:", round(per, 2), "share of GDP per capita %:", round(per / gdp * 100, 3))
