# False premise: coal generation did not fall from 2024 to 2025, it rose. Coal kWh per resident in 2024 and 2025
# (EIA annual net generation / Census July 1 population).
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + '/excel'
ws = openpyxl.load_workbook(B + "/eia-us-electricity-generation-by-source-1949-2026.xlsx", data_only=True)["Annual Data"]
hdr = next(ws.iter_rows(min_row=11, max_row=11, values_only=True))
assert hdr[1] == "Electricity Net Generation From Coal, All Sectors"
coal = {r[0]: r[1] for r in ws.iter_rows(min_row=13, values_only=True) if isinstance(r[0], int)}
ws2 = openpyxl.load_workbook(B + "/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
us = next(ws2.iter_rows(min_row=5, max_row=5, values_only=True))
pop = {2024: us[6], 2025: us[7]}
for y in (2024, 2025):
    print(y, "coal million kWh:", coal[y], "population:", pop[y], "kWh per resident:", round(coal[y] * 1e6 / pop[y], 2))
print("coal change:", round(coal[2025] - coal[2024], 3), "pct", round((coal[2025] / coal[2024] - 1) * 100, 3))
d = coal[2025] * 1e6 / pop[2025] - coal[2024] * 1e6 / pop[2024]
print("per-resident change kWh:", round(d, 2))
