# Combined 2025 sales of furniture and home furnishings stores (NAICS 442) and electronics and appliance stores
# (NAICS 443) in the Census retail workbook (not-adjusted annual totals), and the change from 2024. The trap:
# the sheet also has the subtotal row "442,443" (Furniture, home furn, electronics, and appliance stores) with
# no keyword in its label, and sub-lines under each code; summing the matching rows counts them twice or more.
from pathlib import Path

import openpyxl

DATA = str(Path(__file__).resolve().parents[3] / "examples" / "documents" / "data")
wb = openpyxl.load_workbook(DATA + "/excel/census-monthly-retail-sales-1992-2026.xlsx", data_only=True)


def nsa(year: str) -> list[tuple[str, str, object]]:
    rows = list(wb[year].iter_rows(values_only=True))
    assert str(rows[4][14]).strip() == "TOTAL"
    labels = [str(r[1]).strip() if r[1] is not None else "" for r in rows]
    start, end = labels.index("NOT ADJUSTED"), next(i for i, x in enumerate(labels) if x.startswith("ADJUSTED"))
    return [(str(r[0]).strip() if r[0] is not None else "", str(r[1]).strip(), r[14]) for r in rows[start + 1:end]]


out = {}
for year in ("2024", "2025"):
    rows = nsa(year)
    match = [(c, n, t) for c, n, t in rows if c.startswith(("442", "443"))]
    for m in match:
        print(year, m)
    code = {c: t for c, n, t in rows}
    both = code["442"] + code["443"]
    assert both == code["442,443"]
    words = sum(t for c, n, t in rows
                if isinstance(t, int) and ("furniture" in n.lower() or "electronics" in n.lower()))
    coded = sum(t for c, n, t in match if isinstance(t, int))
    out[year] = (both, code["442,443"] + both, words, coded)
    print(year, "442 + 443 =", both, "| 442,443 row =", code["442,443"])
g24, g25 = out["2024"][0], out["2025"][0]
print("\ngold: 2025", g25, "million; 2024", g24, "; change", g25 - g24, "or", round((g25 / g24 - 1) * 100, 3), "%")
for i, what in enumerate(["the 442,443 subtotal added to 442 and 443", "every row naming furniture or electronics",
                          "every row whose code starts with 442 or 443"], start=1):
    a, b = out["2025"][i], out["2024"][i]
    print(f"naive ({what}): 2025 {a}, 2024 {b}, change {round((a / b - 1) * 100, 3)} %")
