"""PDF text in the top and bottom margins: Docling marks it as page headers and footers. Running headers,
footers and page numbers are dropped; text that appears once (a last section running into the bottom margin,
like a resume's education line) is content and is kept in its place."""

from types import SimpleNamespace

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)
from ingest.parsers import pdf


def test_running_headers_footers_and_page_numbers_are_furniture():
    assert pdf.margin_text_kept([(1, "Page 1 of 2"), (2, "Page 2 of 2")]) == [False, False]
    assert pdf.margin_text_kept([(1, "Annual Report 2024"), (2, "Annual Report 2024")]) == [False, False]
    assert pdf.margin_text_kept([(1, "Acme quarterly report, page 1"), (2, "Acme quarterly report, page 2")]) == [False, False]
    for number in ("3", "- 3 -", "p. 4", "Page 7", "12 / 40"):
        assert pdf.margin_text_kept([(1, number)]) == [False], number
    assert pdf.margin_text_kept([(1, "Confidential")]) == [False]  # too short to be content


def test_a_margin_line_that_appears_once_is_content():
    line = "Example State University, B.S., Physics, 2017"
    assert pdf.margin_text_kept([(1, "Page 1 of 2"), (2, line), (2, "Page 2 of 2")]) == [False, True, False]


def _prov(page: int, text: str) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, charspan=(0, len(text)),
                          bbox=BoundingBox(l=50, t=700, r=500, b=690, coord_origin=CoordOrigin.BOTTOMLEFT))


def test_the_parser_keeps_a_last_section_that_runs_into_the_bottom_margin(monkeypatch):
    doc = DoclingDocument(name="resume")

    def margin(label, page, text):
        doc.add_text(label=label, text=text, prov=_prov(page, text), content_layer=ContentLayer.FURNITURE)

    doc.add_heading("Jane Doe", level=1, prov=_prov(1, "Jane Doe"))
    margin(DocItemLabel.PAGE_HEADER, 1, "Acme hiring packet, confidential")
    doc.add_text(label=DocItemLabel.TEXT, text="Engineer with ten years of experience.",
                 prov=_prov(1, "Engineer with ten years of experience."))
    margin(DocItemLabel.PAGE_FOOTER, 1, "Page 1 of 2")
    margin(DocItemLabel.PAGE_HEADER, 2, "Acme hiring packet, confidential")
    margin(DocItemLabel.PAGE_HEADER, 2, "Prepared for the hiring committee")
    doc.add_heading("Education", level=1, prov=_prov(2, "Education"))
    margin(DocItemLabel.PAGE_FOOTER, 2, "Example State University, B.S., Physics, 2017")
    margin(DocItemLabel.PAGE_FOOTER, 2, "Page 2 of 2")
    monkeypatch.setitem(pdf._converter_cache, (None, 4), SimpleNamespace(convert=lambda path: SimpleNamespace(document=doc)))

    parsed = pdf.parse_pdf("resume.pdf", None, 4)
    assert [(e.text, e.section, e.page) for e in parsed.elements] == [
        ("Engineer with ten years of experience.", "Jane Doe", 1),
        ("Prepared for the hiring committee", "Jane Doe", 2),  # a header goes before its page's text
        ("Example State University, B.S., Physics, 2017", "Jane Doe > Education", 2),  # a footer after it
    ]


def test_margin_text_does_not_take_the_title_from_the_first_heading(monkeypatch):
    doc = DoclingDocument(name="report")
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Draft prepared for the review board", content_layer=ContentLayer.FURNITURE,
                 prov=_prov(1, "Draft prepared for the review board"))
    doc.add_heading("Quarterly Report", level=2, prov=_prov(1, "Quarterly Report"))
    doc.add_text(label=DocItemLabel.TEXT, text="Revenue grew.", prov=_prov(1, "Revenue grew."))
    monkeypatch.setitem(pdf._converter_cache, (None, 4), SimpleNamespace(convert=lambda path: SimpleNamespace(document=doc)))

    parsed = pdf.parse_pdf("report.pdf", None, 4)
    assert [(e.text, e.section) for e in parsed.elements] == [
        ("Draft prepared for the review board", ""), ("Revenue grew.", "Quarterly Report")]
