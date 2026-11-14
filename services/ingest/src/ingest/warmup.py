"""Build-time check that Docling parses a scanned page with OCR from local weights only.

Run with HF_HUB_OFFLINE=1 after `docling-tools models download`; fails the image build if any
model would still be fetched at runtime.
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from ingest.parsers.pdf import parse_pdf


def main() -> None:
    artifacts = sys.argv[1]
    page = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(page)
    for i in range(6):  # a paragraph, not a lone line (a lone line is laid out as a heading)
        draw.text((120, 200 + i * 90), f"Line {i}: offline OCR check for the ingest image", fill="black", font_size=44)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "scan.pdf"
        page.save(pdf, "PDF", resolution=150)
        doc = parse_pdf(str(pdf), artifacts_path=artifacts, num_threads=2)
    text = " ".join(e.text for e in doc.elements)
    if "OCR" not in text:
        raise SystemExit(f"warm-up OCR failed, got: {text!r}")
    print("docling offline warm-up ok:", text)


if __name__ == "__main__":
    main()
