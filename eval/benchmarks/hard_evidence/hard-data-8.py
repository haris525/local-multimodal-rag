# M5+ events (USGS 2025) whose place text refers to one of the 48 contiguous states (by name or postal code)
# -> the states, counts, and their combined July 1, 2025 population (Census).
import re
from pathlib import Path

import openpyxl
import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA
u = pd.read_csv(B + "/csv/usgs-earthquakes-m5-plus-2025.csv")
codes = {"AL": "Alabama", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut",
         "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
         "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
         "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
         "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
         "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
         "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
         "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}


def state_of(place):
    place = place or ""
    m = re.search(r", ([A-Z]{2})$", place)
    if m and m.group(1) in codes:
        return codes[m.group(1)]
    for name in sorted(codes.values(), key=len, reverse=True):  # longest first: West Virginia before Virginia
        if re.search(r"\b" + name + r"\b", place) and "Georgia Island" not in place:
            return name
    return None


u["state"] = u.place.map(state_of)
hit = u[u.state.notna()]
print(hit[["time", "mag", "type", "place", "state"]].to_string(index=False))
print("events:", len(hit))
print(hit.state.value_counts().to_string())
print("place text containing 'California':", u.place.fillna("").str.contains("California").sum())
print("places mentioning 'Georgia' (South Georgia Island, not the state):",
      u[u.place.fillna("").str.contains("Georgia")].place.tolist())
ws = openpyxl.load_workbook(B + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
pop = {r[0].lstrip("."): r[7] for r in ws.iter_rows(min_row=10, max_row=62, values_only=True) if r[0]}
total = 0
for s in sorted(hit.state.unique()):
    print(s, pop[s])
    total += pop[s]
print("combined July 1, 2025 population:", total)
