# Deaths caused by the NOAA billion-dollar disasters that began in 2017, U.S. deaths from unintentional
# injuries in 2017 (CDC), and their ratio. The trap: the CDC State column has a "United States" row that is the
# sum of the 50 states and DC, so summing every State row for the cause doubles the national count.
from pathlib import Path

import pandas as pd

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

noaa = pd.read_csv(DATA + "/csv/noaa-us-billion-dollar-disasters-1980-2024.csv")
began = noaa[noaa["Begin Date"] // 10000 == 2017]
print(began[["Name", "Begin Date", "End Date", "Deaths"]].to_string(index=False))
span = noaa[(noaa["Begin Date"] // 10000 != noaa["End Date"] // 10000)
            & ((noaa["Begin Date"] // 10000 == 2017) | (noaa["End Date"] // 10000 == 2017))]
print("events crossing into or out of 2017:", list(span.Name))
nd = int(began.Deaths.sum())

cdc = pd.read_csv(DATA + "/csv/cdc-leading-causes-of-death-us-1999-2017.csv")
ui = cdc[(cdc.Year == 2017) & (cdc["Cause Name"] == "Unintentional injuries")]
us = int(ui[ui.State == "United States"].Deaths.iloc[0])
assert us == int(ui[ui.State != "United States"].Deaths.sum())
print("\ngold:", len(began), "disasters,", nd, "deaths | unintentional injuries", us, "| ratio", round(us / nd, 3))
naive = int(ui.Deaths.sum())
print("naive (every State row summed, United States included):", naive, "| ratio", round(naive / nd, 3))
