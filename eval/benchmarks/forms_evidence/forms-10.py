# NOAA billion-dollar disasters with Kansas in their name (combined CPI-adjusted cost and deaths), and Kansas's
# population change from the April 1, 2020 estimates base to July 1, 2025 (Census). The trap: a case-insensitive
# match on "kansas" also takes Arkansas, in both files.
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
n = pd.read_csv(B + "/csv/noaa-us-billion-dollar-disasters-1980-2024.csv")
COST = "CPI-Adjusted Cost (USD millions; 2024 dollars)"
cols = ["Name", "Begin Date", COST, "Deaths"]

gold = n[n.Name.str.contains(r"\bKansas\b")]
print(gold[cols].to_string(index=False))
print("gold: events", len(gold), "| cost", round(gold[COST].sum(), 1), "| deaths", gold.Deaths.sum())
naive = n[n.Name.str.contains("kansas", case=False)]
print("\nnaive (case-insensitive 'kansas'):")
print(naive[cols].to_string(index=False))
print("naive: events", len(naive), "| cost", round(naive[COST].sum(), 1), "| deaths", naive.Deaths.sum())

ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
rows = [r for r in ws.iter_rows(min_row=10, max_row=62, values_only=True) if r[0]]
ks = next(r for r in rows if r[0] == ".Kansas")
print("\ngold: Kansas", ks[1], "->", ks[7], "change", ks[7] - ks[1], "pct", round((ks[7] / ks[1] - 1) * 100, 3))
both = [r for r in rows if "kansas" in r[0].lower()]
print("naive (case-insensitive 'kansas'):", [r[0] for r in both], "change", sum(r[7] - r[1] for r in both))
