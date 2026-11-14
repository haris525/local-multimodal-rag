"""Request and response models that cross service boundaries."""

from enum import StrEnum
from typing import Annotated, Any, Literal

from fastapi import Path as PathParam
from pydantic import BaseModel, Field, model_validator

from rag_common.identity import USER_ID

MAX_SELECTED_DOCS = 10
# One document id format everywhere: ids end up in file paths (chunks, tables, assets), so every request
# field and path parameter that carries one is validated against it.
DOC_ID = r"^[0-9a-f]{32}$"
DocId = Annotated[str, Field(pattern=DOC_ID)]
DocIdPath = Annotated[str, PathParam(pattern=DOC_ID)]  # FastAPI path parameters need Path(), not Field()
RESERVED_AUDIT_USERS = ("legacy", "anonymous", "unattributed")


class DocStatus(StrEnum):
    """A document's lifecycle; ingest's registry holds the allowed moves (NEXT). A deleting document is hidden
    from its owner until its points and files are gone and its row is removed."""

    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class DocType(StrEnum):
    TEXT = "text"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"


# --- models service -------------------------------------------------------------------------


class EmbedTextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
    role: Literal["query", "document"] = "document"


class EmbedImageRequest(BaseModel):
    """Document side sends images; query side sends texts. Exactly one of the two."""

    images_b64: list[str] | None = Field(default=None, min_length=1, max_length=16)
    texts: list[str] | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def exactly_one(self) -> "EmbedImageRequest":
        if (self.images_b64 is None) == (self.texts is None):
            raise ValueError("send either images_b64 or texts")
        return self


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=128)


class RerankResponse(BaseModel):
    scores: list[float]


class ModelsInfo(BaseModel):
    text_model: str
    text_dim: int
    image_model: str
    image_dim: int
    rerank_model: str


# --- ingest service -------------------------------------------------------------------------


class ColumnSchema(BaseModel):
    name: str  # the column's display name (short; the frame column and the row chunks use it)
    sql_name: str
    type: Literal["text", "number", "date"]
    profile: str  # distinct values or examples and ranges, for writing correct WHERE clauses
    header: str | None = None  # the full original header text, when it differs from name (long, multi-row)
    unit: str | None = None  # from a units row under the header, e.g. "Million Kilowatthours"
    missing: int = 0  # cells that held a "not available" marker ("Not Available", "(S)", "--"), stored as NULL
    # the column profile (rag_common.profiling); all optional, so tables written before keep loading
    empty: int | None = None  # NULL cells after typing (blank, "not available", or not convertible)
    distinct: int | None = None
    top: list[tuple[str, int]] | None = None  # the most frequent values with their counts (text columns)
    min: str | None = None
    max: str | None = None
    quantiles: list[str] | None = None  # 25th, 50th and 75th percentile (number and date columns)
    examples: list[str] | None = None
    pattern: str | None = None  # how the values are written: "dates written as YYYYMMDD numbers", ...
    warning: str | None = None  # when the stored type will mislead a query, and what to do instead


class SheetLayout(BaseModel):
    """Where a table sat in its sheet (Excel row numbers), so a reader can check how it was read."""

    header_rows: list[int] = []
    data_rows: list[int] = []  # the first and the last
    footnote_rows: list[int] = []
    section_rows: list[int] = []  # one-cell label rows inside the data, filled down into a Section column
    blank_rows: list[int] = []
    blank_columns: list[str] = []  # blank spacer columns inside the table (sheet column letters), left out
    merged: list[str] = []  # merged ranges in A1 notation (at most 50)
    raw_table: str | None = None  # the SQL table holding every cell as text: _row, c1, c2, ...
    sheet_rows: int | None = None  # rows in the part of the sheet this table was read from
    raw_rows: int | None = None  # rows kept in raw_table; fewer than sheet_rows when it was cut
    raw_first_row: int = 1  # the source row number of raw_table's first row (its _row counts from here)
    next_table_row: int | None = None  # where the next table on the same sheet starts (read as its own table)
    reader: Literal["layout", "simple"] = "layout"  # "simple": the simple reading scored better (a warning says so)
    scores: dict[str, float] | None = None  # each reading's score (layout, simple): fit, header, rows kept
    percent_columns: list[str] = []  # columns formatted as percentages in the workbook: fractions (0.052 is 5.2%)
    first_col: int = 0  # the sheet column of this table's first column (tables side by side)


class TableSchema(BaseModel):
    table: str
    source: str  # 'sheet "Sales"' or "page 3, table 2"
    sheet: str | None = None
    rows: int
    row_filter: bool  # rows are indexed one per chunk, so retrieval can be restricted to rows
    columns: list[ColumnSchema]
    title: str | None = None  # the table's title, from the rows above its header
    notes: str | None = None  # notes above the header, a shared column-name pattern, footnotes below the data
    layout: SheetLayout | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    doc_type: DocType
    size: int
    status: DocStatus
    progress: str = ""
    error: str | None = None
    warnings: list[str] = []
    n_chunks: int = 0
    n_images: int = 0
    tables: list[TableSchema] = []
    created_at: str
    updated_at: str
    owner_id: str | None = None
    # The worker's claim on the row (a processing run, or a clean-up of a deleting row). Internal to ingest:
    # never serialized, so no API answer carries it.
    run_id: str | None = Field(default=None, exclude=True)


class DocIdsRequest(BaseModel):
    doc_ids: list[DocId] = Field(min_length=1, max_length=MAX_SELECTED_DOCS)


class SqlQuery(BaseModel):
    doc_id: DocId  # also guards the file path in the sandbox
    sql: str = Field(min_length=1, max_length=4000)
    max_rows: int = Field(default=200, ge=1, le=5000)


class SqlResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


class GrepRequest(BaseModel):
    doc_ids: list[DocId] = Field(min_length=1, max_length=MAX_SELECTED_DOCS)
    terms: list[str] = Field(min_length=1, max_length=10)
    limit: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def bounded_terms(self) -> "GrepRequest":
        self.terms = [t.strip() for t in self.terms if t.strip()]
        if not self.terms or any(len(t) > 100 for t in self.terms):
            raise ValueError("terms must be 1-100 characters")
        return self


class GrepHit(BaseModel):
    id: str
    doc_id: str
    chunk_index: int
    matched: int  # distinct terms found
    count: int  # total occurrences


# --- audit service --------------------------------------------------------------------------


class AuditEvent(BaseModel):
    service: str = Field(max_length=40)
    action: str = Field(max_length=60)  # e.g. document.uploaded, chat.answered
    actor: str = Field(default="local", max_length=120)
    subject: str | None = Field(default=None, max_length=2000)  # doc_id(s)
    detail: dict[str, Any] = Field(default_factory=dict)
    # The account the event belongs to. 'legacy' is reserved for rows written before accounts existed.
    user_id: str | None = Field(default=None, pattern=r"^(?:%s|anonymous|unattributed)$" % USER_ID[1:-1])


class MigrateLegacy(BaseModel):
    """Adopt data written before accounts existed (no owner) for one user."""

    owner_id: str = Field(pattern=USER_ID)
    dry_run: bool = False
    force: bool = False


# --- threads service ------------------------------------------------------------------------

THREAD_ID = r"^[0-9a-f]{32}$"


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    doc_ids: list[str] = []
    message_count: int = 0
    summarized: int = 0  # messages folded into the running summary
    created_at: str
    updated_at: str


class ThreadMessage(BaseModel):
    seq: int
    role: Literal["user", "assistant"]
    content: str
    meta: dict[str, Any] = {}  # assistant: sources, route, chunks per document, citations, timings
    created_at: str


class ThreadDetail(BaseModel):
    thread: ThreadInfo
    messages: list[ThreadMessage]


class ThreadCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=200)
    doc_ids: list[DocId] = Field(default_factory=list, max_length=MAX_SELECTED_DOCS)


class ThreadUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    doc_ids: list[DocId] | None = Field(default=None, max_length=MAX_SELECTED_DOCS)


class NewMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=100_000)
    meta: dict[str, Any] = Field(default_factory=dict)


class NewMessages(BaseModel):
    messages: list[NewMessage] = Field(min_length=1, max_length=4)
    doc_ids: list[DocId] | None = Field(default=None, max_length=MAX_SELECTED_DOCS)


class ThreadContext(BaseModel):
    """What the model sees of a conversation: the running summary plus the turns after it."""

    summary: str
    summary_upto: int  # seq of the last message folded into the summary
    messages: list[ThreadMessage]


class SummaryUpdate(BaseModel):
    summary: str = Field(max_length=20_000)
    upto: int


class MemoryItem(BaseModel):
    memory_id: str
    text: str
    source: Literal["user", "auto"]
    thread_id: str | None = None
    created_at: str


class MemoryCreate(BaseModel):
    text: str = Field(min_length=3, max_length=500)
    source: Literal["user", "auto"] = "user"
    thread_id: str | None = None
