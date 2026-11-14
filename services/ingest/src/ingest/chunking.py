"""Structure-aware chunking.

- Prose: section -> paragraph -> sentence -> token windows, packed to a target size with no
  overlap. Neighbouring chunks are re-attached at answer time, so boundaries cost no context.
- Tables: Markdown with the header row repeated in every window.
- Sheets: a profile summary, then the whole table when it fits one chunk, otherwise one record
  per row ("Column: value" lines) so rows can be filtered by SQL and found by BM25 or vectors.
- Images: the vision model's description, with the image vector on the first chunk.
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from tokenizers import Tokenizer

from rag_common.contracts import TableSchema

from ingest.elements import Chunk, Element, Sheet


@dataclass(frozen=True)
class ChunkingConfig:
    text_target: int = 256
    text_max: int = 384
    min_tail: int = 64  # a shorter last chunk of a section is merged into the previous one
    table_max: int = 512
    cell_max_chars: int = 300
    long_cell_tokens: int = 64  # sheet cells longer than this are split instead of repeated


class TokenCounter:
    """Exact token counts with the embedding model's own tokenizer."""

    def __init__(self, tokenizer_file: str):
        self._tok = Tokenizer.from_file(tokenizer_file)
        self._tok.no_truncation()
        self._tok.no_padding()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(e.ids) for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def windows(self, text: str, size: int) -> list[str]:
        """Split text into consecutive pieces of at most `size` tokens, cut at token offsets."""
        offsets = self._tok.encode(text, add_special_tokens=False).offsets
        pieces, start = [], 0
        for i in range(size, len(offsets) + size, size):
            end = offsets[min(i, len(offsets)) - 1][1]
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            start = end
        return pieces


# --- sentences ----------------------------------------------------------------------------------

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "fig", "figs", "eq", "dr", "mr", "mrs", "ms", "prof", "no",
    "approx", "cf", "al", "inc", "ltd", "jr", "sr", "st", "u.s", "u.k", "dept", "est", "vol",
}
_BOUNDARY = re.compile(r"([.!?])([\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based splitter: ends at . ! ? followed by space and a capital or digit, except
    after common abbreviations and single-letter initials."""
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(text):
        before = text[start : match.start()].strip()
        word = before.rsplit(None, 1)[-1].lower() if before else ""
        if match.group(1) == "." and (word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())):
            continue
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# --- formatting ---------------------------------------------------------------------------------


def format_value(value: object) -> str:
    """Render a cell for display: ISO dates, integers without '.0', blanks for missing."""
    if value is None or value is pd.NaT:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.12g}"
    if isinstance(value, datetime):
        midnight = (value.hour, value.minute, value.second) == (0, 0, 0)
        return value.date().isoformat() if midnight else value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def context_header(doc_name: str, *parts: str) -> str:
    return " | ".join([f"Document: {doc_name}", *[p for p in parts if p]])


# --- prose --------------------------------------------------------------------------------------


@dataclass
class _Unit:
    text: str
    tokens: int
    page: int | None
    paragraph: int


def _units(elements: list[Element], counter: TokenCounter, cfg: ChunkingConfig) -> list[_Unit]:
    """Paragraphs that fit stay whole; longer ones become sentences; longer sentences become
    token windows."""
    units: list[_Unit] = []
    for p_idx, el in enumerate(elements):
        n = counter.count(el.text)
        if n <= cfg.text_max:
            units.append(_Unit(el.text, n, el.page, p_idx))
            continue
        for sentence in split_sentences(el.text):
            m = counter.count(sentence)
            pieces = [sentence] if m <= cfg.text_max else counter.windows(sentence, cfg.text_max)
            units.extend(_Unit(p, counter.count(p), el.page, p_idx) for p in pieces)
    return units


def _pack(units: list[_Unit], cfg: ChunkingConfig) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    size = 0
    for unit in units:
        over_target = size + unit.tokens > cfg.text_target
        if current and over_target and (size >= cfg.text_target // 2 or size + unit.tokens > cfg.text_max):
            groups.append(current)
            current, size = [], 0
        current.append(unit)
        size += unit.tokens
    if current:
        if groups and size < cfg.min_tail and sum(u.tokens for u in groups[-1]) + size <= cfg.text_max:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _join(units: list[_Unit]) -> str:
    out = units[0].text
    for prev, unit in zip(units, units[1:], strict=False):
        out += (" " if unit.paragraph == prev.paragraph else "\n\n") + unit.text
    return out


def prose_chunks(
    elements: list[Element], doc_name: str, counter: TokenCounter, cfg: ChunkingConfig, label: str = ""
) -> list[Chunk]:
    """Chunk consecutive text elements that share one section."""
    if not elements:
        return []
    section = elements[0].section
    header = context_header(doc_name, label, f"Section: {section}" if section else "")
    chunks = []
    for group in _pack(_units(elements, counter, cfg), cfg):
        text = _join(group)
        pages = [u.page for u in group if u.page is not None]
        chunks.append(
            Chunk(
                kind="text",
                text=text,
                search_text=f"{header}\n{text}",
                n_tokens=counter.count(text),
                section=section,
                page=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
            )
        )
    return chunks


# --- tables -------------------------------------------------------------------------------------


def _md_cell(value: object, max_chars: int) -> str:
    text = format_value(value).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def table_chunks(
    frame: pd.DataFrame,
    header_text: str,
    caption: str,
    counter: TokenCounter,
    cfg: ChunkingConfig,
    **location,
) -> list[Chunk]:
    """Markdown windows of whole rows; every window repeats the caption and header row.
    `frame.index` holds source row numbers (Excel rows, or 1-based table rows)."""
    columns = [_md_cell(c, cfg.cell_max_chars) for c in frame.columns]
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)
    prefix = f"{caption}\n{head}" if caption else head
    budget = max(cfg.table_max - counter.count(prefix), 32)
    lines = [
        "| " + " | ".join(_md_cell(v, cfg.cell_max_chars) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    sizes = counter.count_many(lines) if lines else []
    rows = list(frame.index)

    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end, used = start, 0
        while end < len(lines) and (end == start or used + sizes[end] <= budget):
            used += sizes[end]
            end += 1
        body = "\n".join(lines[start:end])
        if sizes[start] > budget:  # one row larger than a chunk: keep its first tokens
            body = counter.windows(body, budget)[0] + " ..."
        text = f"{prefix}\n{body}"
        chunks.append(
            Chunk(
                kind="table",
                text=text,
                search_text=f"{header_text}\n{text}",
                n_tokens=counter.count(text),
                row_start=int(rows[start]),
                row_end=int(rows[end - 1]),
                **location,
            )
        )
        start = end
    return chunks


# --- sheets -------------------------------------------------------------------------------------


def row_chunks(
    sheet: Sheet, table: str, doc_name: str, counter: TokenCounter, cfg: ChunkingConfig
) -> list[Chunk]:
    """One record per row. A row with long text cells (narratives, notes) is split: each piece
    repeats the row's short fields, so every chunk still says which record it belongs to. A row that is too
    large without any long cell (many columns, long column names) is split into groups of fields, each
    repeating the first field. A non-empty row never yields zero chunks."""
    columns = [str(c) for c in sheet.frame.columns]
    records = []
    for excel_row, values in zip(sheet.frame.index, sheet.frame.itertuples(index=False, name=None), strict=True):
        fields = [(c, format_value(v)) for c, v in zip(columns, values, strict=True)]
        records.append((int(excel_row), [(c, v) for c, v in fields if v]))
    texts = ["\n".join(f"{c}: {v}" for c, v in fields) for _, fields in records]
    sizes = counter.count_many(texts) if texts else []

    chunks: list[Chunk] = []
    for (excel_row, fields), text, size in zip(records, texts, sizes, strict=True):
        if not fields:
            continue
        header = context_header(doc_name, f"Sheet: {sheet.name}", f"Row {excel_row}")
        pieces = [text]
        if size > cfg.text_max:
            long = {c for c, v in fields if counter.count(v) > cfg.long_cell_tokens}
            if long:
                prefix = "\n".join(f"{c}: {v}" for c, v in fields if c not in long)
                budget = max(cfg.text_max - counter.count(prefix) - 8, 64)
                pieces = []
                for c, v in fields:
                    if c in long:
                        for part in _split_long(v, budget, counter, cfg):
                            pieces.append(f"{prefix}\n{c}: {part}" if prefix else f"{c}: {part}")
            else:
                pieces = _field_groups(fields, counter, cfg.text_max)
        for piece in pieces:
            chunks.append(
                Chunk(
                    kind="row",
                    text=piece,
                    search_text=f"{header}\n{piece}",
                    n_tokens=counter.count(piece),
                    sheet=sheet.name,
                    table=table,
                    row_start=excel_row,
                    row_end=excel_row,
                )
            )
    return chunks


def _field_groups(fields: list[tuple[str, str]], counter: TokenCounter, budget: int) -> list[str]:
    """`Column: value` lines packed into pieces of at most `budget` tokens; every piece after the first starts
    with the first field again (the row's key, such as its name), so each piece says which record it is."""
    lines = [f"{c}: {v}" for c, v in fields]
    key = lines[0]
    pieces, current, used = [], [], 0
    for line in lines:
        size = counter.count(line) + 1
        if current and used + size > budget:
            pieces.append("\n".join(current))
            current, used = [key], counter.count(key) + 1
        current.append(line)
        used += size
    if current and (not pieces or current != [key]):
        pieces.append("\n".join(current))
    return pieces


def _split_long(text: str, budget: int, counter: TokenCounter, cfg: ChunkingConfig) -> list[str]:
    local = ChunkingConfig(text_target=budget, text_max=budget, min_tail=cfg.min_tail)
    units = _units([Element(kind="text", text=text)], counter, local)
    return [_join(group) for group in _pack(units, local)]


def sheet_summary(sheet: Sheet, schema: TableSchema) -> str:
    rows = sheet.frame.index
    lines = [
        f'Sheet "{sheet.name}": {schema.rows} rows (rows {rows.min()}-{rows.max()}), '
        f"{len(schema.columns)} columns. SQL table: {schema.table}."
    ]
    if sheet.title:
        lines.append(f"Title: {sheet.title}")
    if sheet.notes:
        lines.append(f"Notes: {sheet.notes[:1500]}")
    layout = schema.layout
    if layout and len(layout.header_rows) > 1:
        lines.append(f"Header rows {layout.header_rows[0]}-{layout.header_rows[-1]}.")
    lines.append("Columns:")
    for c in schema.columns:
        full = f" [{c.header}]" if c.header else ""
        unit = f", {c.unit}" if c.unit else ""
        pattern = f"; {c.pattern}" if getattr(c, "pattern", None) else ""
        lines.append(f"- {c.name}{full} ({c.type}{unit}): {c.profile}{pattern}")
    return "\n".join(lines)


def summary_chunk(sheet: Sheet, summary: str, doc_name: str, counter: TokenCounter) -> Chunk:
    header = context_header(doc_name, f"Sheet: {sheet.name}", "Summary")
    return Chunk(
        kind="sheet_summary",
        text=summary,
        search_text=f"{header}\n{summary}",
        n_tokens=counter.count(summary),
        sheet=sheet.name,
    )


# --- images -------------------------------------------------------------------------------------


def image_chunks(
    element: Element, asset: str, doc_name: str, counter: TokenCounter, cfg: ChunkingConfig
) -> list[Chunk]:
    label = "Image" if element.standalone else f"Figure on page {element.page}" if element.page else "Figure"
    paragraphs = [Element(kind="text", text=p.strip(), page=element.page) for p in element.text.split("\n\n") if p.strip()]
    chunks = prose_chunks(paragraphs, doc_name, counter, cfg, label=label)
    for chunk in chunks:
        chunk.kind = "image"
        chunk.section = element.section
        chunk.asset = asset
    if chunks:
        chunks[0].image = element.image
    return chunks


def group_by_section(elements: Iterable[Element]) -> Iterable[list[Element]]:
    """Yield runs of consecutive text elements that share a section; other kinds singly. Text after a figure
    or table starts a new run: the chunker reads a run by its first element's kind, so text joined to a
    figure's run would be dropped."""
    run: list[Element] = []
    for el in elements:
        if el.kind == "text" and run and run[-1].kind == "text" and run[-1].section == el.section:
            run.append(el)
            continue
        if run:
            yield run
        run = [el]
    if run:
        yield run
