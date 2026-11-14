# How many states had more than 10 million residents on July 1, 2025 (Census) and their share of the U.S.
# population. The trap: the table lists the United States and the four regions (Northeast, Midwest, South,
# West) as rows above the states, all over 10 million, so a naive count gives 15.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

ws = openpyxl.load_workbook(DATA + "/excel/census-state-population-estimates-2020-2025.xlsx", data_only=True).active
rows = [r for r in ws.iter_rows(min_row=5, values_only=True) if r[0] and isinstance(r[7], int)]
print("rows with a 2025 estimate:", len(rows), "| without a leading dot:", [r[0] for r in rows if r[0][0] != "."])
pop25 = {r[0]: r[7] for r in rows}
us = pop25["United States"]
states = {k.lstrip("."): v for k, v in pop25.items() if k.startswith(".") and k != ".Puerto Rico"}
assert len(states) == 51 and sum(states.values()) == us  # 50 states and DC add up to the U.S. row
big = sorted(((v, k) for k, v in states.items() if v > 10_000_000), reverse=True)
for v, k in big:
    print(k, v)
together = sum(v for v, _ in big)
print("\ngold:", len(big), "states,", together, "residents, share of the U.S.", round(together / us * 100, 3), "%")
naive = [(k, v) for k, v in pop25.items() if v > 10_000_000]
nsum = sum(v for _, v in naive)
print("naive (every row over 10 million):", len(naive), [k for k, _ in naive])
print("naive: residents", nsum, "share", round(nsum / us * 100, 3), "%")
