# M5+ events in the Federated States of Micronesia (USGS 2025), the largest one, and the country's 2024 GDP per
# capita and internet use (World Bank). The trap: USGS writes the country as "Micronesia" for 7 events and as
# "Federated States of Micronesia" for 1; the World Bank writes "Micronesia, Fed. Sts.".
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

B = DATA + "/csv"
u = pd.read_csv(B + "/usgs-earthquakes-m5-plus-2025.csv")
wb = pd.read_csv(B + "/world-bank-gdp-per-capita-internet-use-2000-2024.csv")
pl = u.place.fillna("")

fsm = u[pl.str.endswith(", Micronesia") | pl.str.endswith("Federated States of Micronesia")]
print(fsm[["time", "mag", "place"]].to_string(index=False))
print("events:", len(fsm), "| largest:", fsm.mag.max(), fsm.loc[fsm.mag.idxmax(), "time"])
print("(Fais and Colonia are in Yap State, one of the four states of the Federated States of Micronesia)")

naive = u[pl.str.contains("Federated States of Micronesia")]
print("\nnaive: place contains 'Federated States of Micronesia':", len(naive))
print("World Bank rows named 'Federated States of Micronesia':",
      (wb.country == "Federated States of Micronesia").sum())

r = wb[(wb.year == 2024) & (wb.country == "Micronesia, Fed. Sts.")]
print("\n" + r[["country", "iso3_code", "gdp_per_capita_current_usd", "internet_users_pct_of_population"]]
      .to_string(index=False))
