# First year in which annual not-adjusted sales of electronic shopping and mail-order houses (NAICS 4541)
# exceeded those of general merchandise stores (NAICS 452). Census retail workbook, one sheet per year.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

P = DATA + '/excel/census-monthly-retail-sales-1992-2026.xlsx'
wb = openpyxl.load_workbook(P, data_only=True)


def nsa_total(ws, code):
    """TOTAL column (13th value column) of the NOT ADJUSTED row with this NAICS code."""
    rows = list(ws.iter_rows(values_only=True))
    labels = [str(r[1]).strip() if r[1] is not None else "" for r in rows]
    start = labels.index("NOT ADJUSTED")
    end = next(i for i, x in enumerate(labels) if x.startswith("ADJUSTED"))
    for r in rows[start + 1:end]:
        if str(r[0]).strip() == code:
            assert str(rows[4][14]).strip() == "TOTAL", ws.title
            return r[14], str(r[1]).strip()


first = None
for year in range(1992, 2026):
    ws = wb[str(year)]
    e, en = nsa_total(ws, "4541")
    g, gn = nsa_total(ws, "452")
    flag = e > g
    print(year, en, e, "|", gn, g, "| ecom > gm:", flag)
    if flag and first is None:
        first = (year, e, g)
print("first year:", first)
