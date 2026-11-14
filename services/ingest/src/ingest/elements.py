"""Intermediate representations: parsers emit Elements, the chunker turns them into Chunks."""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

ElementKind = Literal["text", "table", "image"]


@dataclass
class Element:
    """One structural unit of a document, in reading order."""

    kind: ElementKind
    text: str = ""  # paragraph text, table caption, or image description
    section: str = ""  # heading path, e.g. "Results > Revenue"
    page: int | None = None
    frame: pd.DataFrame | None = None  # tables: header = columns, index = source row numbers
    table_index: int | None = None
    image: bytes | None = None  # PNG
    standalone: bool = False  # image files (vs figures inside a document)
    list_item: bool = False  # consecutive list items are merged into one paragraph


@dataclass
class Sheet:
    """One spreadsheet sheet as a frame; index = Excel row numbers. parsers/layout.py finds where the table sits
    in the sheet: its header rows, title, notes, footnotes, units and section labels."""

    name: str
    frame: pd.DataFrame
    preamble: str = ""  # text above the header rows (titles, notes), one line per row
    blank_formulas: int = 0  # formula cells with no cached value
    title: str = ""
    notes: str = ""  # notes above the header, a shared column-name pattern, footnotes below the data
    headers: dict[str, str] = field(default_factory=dict)  # column name -> full header text, when they differ
    units: dict[str, str] = field(default_factory=dict)  # column name -> unit, from a units row
    layout: dict = field(default_factory=dict)  # rag_common.contracts.SheetLayout fields
    raw: list[list[str | None]] = field(default_factory=list)  # every cell as text, for the <table>__raw table


@dataclass
class ParsedDocument:
    elements: list[Element] = field(default_factory=list)
    sheets: list[Sheet] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    kind: Literal["text", "table", "image", "sheet_summary", "row", "summary"]
    text: str  # what the user and the LLM see
    search_text: str  # contextual header + text; embedded and BM25-indexed
    n_tokens: int
    section: str = ""
    page: int | None = None
    page_end: int | None = None
    table_index: int | None = None
    sheet: str | None = None
    table: str | None = None  # SQL table name for sheet rows
    row_start: int | None = None
    row_end: int | None = None
    image: bytes | None = None  # set on the chunk that carries the image vector
    asset: str | None = None
