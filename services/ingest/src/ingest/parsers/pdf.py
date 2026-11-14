"""PDF -> elements with Docling: layout, reading order, TableFormer tables, OCR, figure crops.

Runs inside a child process (see isolation.py), so it builds its own converter per call.
"""

import pandas as pd

from ingest.elements import Element, ParsedDocument
from ingest.parsers.images import to_png

MIN_FIGURE_SIDE = 120  # pixels at images_scale=2, i.e. about 60 pt: drops icons and bullets


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
    skipped_labels = {DocItemLabel.CAPTION, DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

    for item, _level in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if isinstance(item, TitleItem | SectionHeaderItem):
            # The first heading on page 1, before any body text, is the document title.
            is_title = isinstance(item, TitleItem) or (not headings and not out.elements and page == 1)
            level = 0 if is_title else max(int(item.level), 1)
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = item.text.strip()
            section = " > ".join(headings[k] for k in sorted(headings))
        elif isinstance(item, TableItem):
            frame = item.export_to_dataframe(doc=doc)
            if frame.empty:
                continue
            tables += 1
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
            out.elements.append(
                Element(kind="image", text=item.caption_text(doc).strip(), section=section, page=page, image=to_png(image))
            )
        elif isinstance(item, TextItem) and item.label not in skipped_labels and item.text.strip():
            text = item.text.strip()
            if isinstance(item, ListItem):
                text = f"{item.marker or '-'} {text}"
                prev = out.elements[-1] if out.elements else None
                if prev and prev.list_item and prev.section == section:
                    prev.text += "\n" + text
                    continue
                out.elements.append(Element(kind="text", text=text, section=section, page=page, list_item=True))
                continue
            out.elements.append(Element(kind="text", text=text, section=section, page=page))

    if not out.elements:
        out.warnings.append("No content extracted from PDF")
    return out
