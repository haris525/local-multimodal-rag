"""PDF -> elements with Docling: layout, reading order, TableFormer tables, OCR, figure crops.

Runs inside a child process (see isolation.py), so it builds its own converter per call.

Docling marks text in the top and bottom margins as page headers and footers (page furniture). A real header
or footer repeats on other pages (a running title, a date) or is a page number; those are dropped. Text that
appears once, such as a last section running into the bottom margin, is content and is kept, in its place:
a header before the page's text, a footer after it.
"""

import re

import pandas as pd

from ingest.elements import Element, ParsedDocument
from ingest.parsers.images import to_png

MIN_FIGURE_SIDE = 120  # pixels at images_scale=2, i.e. about 60 pt: drops icons and bullets
# "3", "- 3 -", "Page 3", "page 3 of 12", "3 / 12", "p. 3" once digits are replaced by #
_PAGE_NUMBER = re.compile(r"^\W*(?:page|p\.?)?\W*#+(?:\W*(?:of|/)\W*#+)?\W*$")
MIN_MARGIN_WORDS = 3  # a margin line kept as content has at least this many words of two or more letters


def margin_text_kept(lines: list[tuple[int | None, str]]) -> list[bool]:
    """For each (page, text) of a page header or footer: is it content? Not when its text (digits ignored)
    is also a header or footer on another page, when it is a page number, or when it has fewer than
    MIN_MARGIN_WORDS words."""
    def norm(text: str) -> str:
        return " ".join(re.sub(r"\d+", "#", text.lower()).split())

    pages: dict[str, set[int | None]] = {}
    for page, text in lines:
        pages.setdefault(norm(text), set()).add(page)
    return [len(pages[norm(text)]) < 2 and not _PAGE_NUMBER.match(norm(text))
            and len(re.findall(r"[^\W\d_]{2,}", text)) >= MIN_MARGIN_WORDS for page, text in lines]


def _converter(artifacts_path: str | None, num_threads: int):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
        images_scale=2.0,
        queue_max_size=8,  # backpressure: the default buffers up to 100 rendered pages per stage
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=num_threads),
    )
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.heading_hierarchy_options.enabled = True  # levels from numbering, bookmarks, font style
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def _table_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() or f"column_{i + 1}" for i, c in enumerate(df.columns)]
    df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.Index(range(1, len(df) + 1), name="row")
    return df


_converter_cache: dict[tuple, object] = {}


def parse_pdf(path: str, artifacts_path: str | None = None, num_threads: int = 4) -> ParsedDocument:
    """Convert one PDF. The converter (and its models) is cached for the life of the process."""
    from docling_core.types.doc import (
        ContentLayer,
        DocItemLabel,
        ListItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    key = (artifacts_path, num_threads)
    if key not in _converter_cache:
        _converter_cache[key] = _converter(artifacts_path, num_threads)
    doc = _converter_cache[key].convert(path).document
    out = ParsedDocument()
    headings: dict[int, str] = {}
    section = ""
    tables = 0
    body = 0  # body elements so far: the title rule looks only at those, not at margin text
    margin_labels = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
    skipped_labels = {DocItemLabel.CAPTION, *margin_labels}

    # Page headers and footers (furniture, and any the body holds): the ones that are content wait for their page.
    margins = [(item.prov[0].page_no if getattr(item, "prov", None) else None, item.label, item.text.strip())
               for item, _ in doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE})
               if isinstance(item, TextItem) and item.label in margin_labels and item.text.strip()]
    waiting: dict[int | None, dict[str, list[str]]] = {}
    for (page, label, text), keep in zip(margins, margin_text_kept([(p, t) for p, _, t in margins]), strict=True):
        if keep:
            part = "header" if label == DocItemLabel.PAGE_HEADER else "footer"
            waiting.setdefault(page, {"header": [], "footer": []})[part].append(text)

    def margin(page: int | None, part: str) -> None:
        for text in waiting.get(page, {}).pop(part, []):
            out.elements.append(Element(kind="text", text=text, section=section, page=page))

    current = None
    for item, _level in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if page is not None and page != current:  # a new page: the last page's footers, then this one's headers
            if current is not None:
                margin(current, "footer")
            margin(page, "header")
            current = page
        if isinstance(item, TitleItem | SectionHeaderItem):
            # The first heading on page 1, before any body text, is the document title.
            is_title = isinstance(item, TitleItem) or (not headings and not body and page == 1)
            level = 0 if is_title else max(int(item.level), 1)
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = item.text.strip()
            section = " > ".join(headings[k] for k in sorted(headings))
        elif isinstance(item, TableItem):
            frame = item.export_to_dataframe(doc=doc)
            if frame.empty:
                continue
            tables += 1
            body += 1
            out.elements.append(
                Element(
                    kind="table",
                    text=item.caption_text(doc).strip(),
                    section=section,
                    page=page,
                    frame=_table_frame(frame),
                    table_index=tables,
                )
            )
        elif isinstance(item, PictureItem):
            image = item.get_image(doc)
            if image is None or min(image.size) < MIN_FIGURE_SIDE:
                continue
            body += 1
            out.elements.append(
                Element(kind="image", text=item.caption_text(doc).strip(), section=section, page=page, image=to_png(image))
            )
        elif isinstance(item, TextItem) and item.label not in skipped_labels and item.text.strip():
            text = item.text.strip()
            body += 1
            if isinstance(item, ListItem):
                text = f"{item.marker or '-'} {text}"
                prev = out.elements[-1] if out.elements else None
                if prev and prev.list_item and prev.section == section:
                    prev.text += "\n" + text
                    continue
                out.elements.append(Element(kind="text", text=text, section=section, page=page, list_item=True))
                continue
            out.elements.append(Element(kind="text", text=text, section=section, page=page))

    if current is not None:
        margin(current, "footer")
    for page in sorted(waiting, key=lambda p: (p is None, p or 0)):  # pages with margin text and no body text
        margin(page, "header")
        margin(page, "footer")
    if not out.elements:
        out.warnings.append("No content extracted from PDF")
    return out
