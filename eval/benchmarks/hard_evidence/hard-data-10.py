# Seasonally adjusted total retail and food services sales in 2020: the lowest month, its drop from February 2020,
# and the first month whose value was above February 2020. Census retail workbook, sheet "2020", ADJUSTED block.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")

P = DATA + '/excel/census-monthly-retail-sales-1992-2026.xlsx'
wb = openpyxl.load_workbook(P, data_only=True)


def sa_row(ws, label):
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(x).strip() for x in rows[4][2:14]]
    labels = [str(r[1]).strip() if r[1] is not None else "" for r in rows]
    start = next(i for i, x in enumerate(labels) if x.startswith("ADJUSTED"))
    for r in rows[start + 1:]:
        if str(r[1]).strip() == label:
            return dict(zip(hdr, r[2:14]))


months = {}
for y in ("2020", "2021"):
    months.update(sa_row(wb[y], "Retail and food services sales, total"))
for k, v in months.items():
    print(k, v)
y2020 = {k: v for k, v in months.items() if k.endswith("2020")}
low = min(y2020, key=y2020.get)
feb = months["Feb. 2020"]
print("lowest 2020 month:", low, y2020[low])
print("drop from Feb. 2020:", feb - y2020[low], "pct", round((feb - y2020[low]) / feb * 100, 3))
keys = list(months)
after = keys[keys.index("Feb. 2020") + 1:]
first_above = next(k for k in after if months[k] > feb)
print("first month above Feb. 2020:", first_above, months[first_above])
