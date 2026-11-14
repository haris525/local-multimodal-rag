"""Generate one sample of every supported file type, each holding facts the smoke test asks about.
Every name starts with "smoke-", so a test run never replaces (and then deletes) a user's own file.

    uv run --with fpdf2 --with openpyxl --with pillow scripts/make_samples.py samples
"""

import csv
import io
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from fpdf import FPDF
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def bar_chart() -> Image.Image:
    img = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(img)
    d.text((220, 20), "Customer satisfaction by year", fill="black", font=_font(32))
    for i, (year, pct) in enumerate([(2023, 72), (2024, 78), (2025, 85)]):
        x = 150 + i * 230
        top = 540 - pct * 5
        d.rectangle([x, top, x + 140, 540], fill=(70, 110, 180))
        d.text((x + 35, top - 40), f"{pct}%", fill="black", font=_font(30))
        d.text((x + 35, 550), str(year), fill="black", font=_font(28))
    return img


def scanned_page() -> Image.Image:
    img = Image.new("RGB", (1240, 1754), (250, 250, 247))
    d = ImageDraw.Draw(img)
    lines = [
        "Appendix B - Warranty terms",
        "",
        "Warranty claims must be filed within 45 days of delivery.",
        "Claims require the original invoice and photos of the damage.",
        "Refunds are issued to the original payment method.",
    ]
    for i, line in enumerate(lines):
        d.text((110, 160 + i * 70), line, fill=(20, 20, 20), font=_font(40 if i == 0 else 34))
    return img.rotate(0.6, fillcolor=(250, 250, 247))


def write_pdf(path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Aurora Program Annual Report 2026", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Budget", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "The Aurora project budget for 2026 is 4.2 million dollars. Most of the budget funds the "
        "new data platform, and the remainder covers staff training. The steering committee reviews "
        "spending every quarter and can move up to ten percent between categories without board approval.",
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Regional revenue", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Table 1 lists third-quarter revenue by region in thousands of dollars.")
    pdf.ln(2)
    rows = [("Region", "Q3 revenue (k$)", "Growth"), ("North", "120", "4%"), ("South", "340", "11%"),
            ("East", "95", "-2%"), ("West", "210", "7%")]
    with pdf.table(col_widths=(50, 50, 40), text_align="LEFT") as table:
        for row in rows:
            table.row(row)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Table 1: Q3 revenue by region", new_x="LMARGIN", new_y="NEXT")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Customer satisfaction", new_x="LMARGIN", new_y="NEXT")
    buf = io.BytesIO()
    bar_chart().save(buf, format="PNG")
    pdf.image(buf, w=150)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Figure 1: Customer satisfaction survey results", new_x="LMARGIN", new_y="NEXT")
    pdf.add_page()
    buf = io.BytesIO()
    scanned_page().save(buf, format="PNG")
    pdf.image(buf, x=0, y=0, w=210, h=297)
    pdf.output(str(path))


def write_xlsx(path: Path) -> None:
    rng = random.Random(7)
    wb = Workbook()
    ws = wb.active
    ws.title = "Orders"
    ws["A1"] = "Orders export - confidential"
    ws.append([])
    ws.append(["Order", "Customer", None, "Sale", None, "Notes"])
    ws.merge_cells("B3:C3")
    ws.merge_cells("D3:E3")
    ws.append(["Order ID", "Name", "Region", "Amount", "Date", "Notes"])
    issues = ["", "", "", "late delivery", "battery overheating reported", "packaging damaged"]
    for i in range(1, 301):
        region = ["North", "South", "East", "West"][i % 4]
        note = issues[rng.randrange(len(issues))]
        if i in (8, 44, 128):
            region, note = "North", "battery overheating reported by customer"
        ws.append([f"ORD-{i:04d}", f"Customer {i}", region, round(rng.uniform(50, 900), 2),
                   date(2026, 1, 1) + timedelta(days=i % 200), note])
    targets = wb.create_sheet("Targets")
    targets.append(["Region", "Annual target"])
    for region, target in [("North", 40000), ("South", 52000), ("East", 31000), ("West", 45000)]:
        targets.append([region, target])
    wb.save(path)


def write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SKU", "Item", "Quantity", "Warehouse"])
        for i in range(1, 61):
            qty = 137 if i == 42 else (i * 7) % 90
            w.writerow([f"INV-2025-{i:04d}", f"Part {i}", qty, ["Austin", "Denver"][i % 2]])


def write_png(path: Path) -> None:
    img = Image.new("RGB", (1000, 640), (236, 240, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 940, 580], outline=(180, 30, 30), width=10)
    d.text((120, 150), "SERVER ROOM", fill=(180, 30, 30), font=_font(72))
    d.text((120, 300), "Temperature limit: 27 \u00b0C", fill=(20, 20, 20), font=_font(52))
    d.text((120, 400), "Call facilities at ext. 4410 if exceeded", fill=(20, 20, 20), font=_font(38))
    img.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke-handbook.txt").write_text(
        "Vacation policy\n\nEmployees may carry over up to 5 unused vacation days into the next year.\n"
        "Unused days beyond that limit expire on January 31.\n\nRemote work\n\nStaff may work remotely "
        "two days per week with manager approval.\n",
        encoding="utf-8",
    )
    (OUT / "smoke-runbook.md").write_text(
        "# Service runbook\n\n## Deployment\n\nDeploys run every Tuesday after the change review.\n\n"
        "### Rollback\n\nRollback requires approval from the on-call lead and takes about 12 minutes.\n",
        encoding="utf-8",
    )
    write_pdf(OUT / "smoke-aurora_report.pdf")
    write_xlsx(OUT / "smoke-orders.xlsx")
    write_csv(OUT / "smoke-inventory.csv")
    write_png(OUT / "smoke-server_room_sign.png")
    print(f"samples written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
