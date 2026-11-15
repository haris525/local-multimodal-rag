"""Qdrant collection schema shared by the writer (ingest) and the reader (rag)."""

import uuid
import warnings
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_common.contracts import Generation, ModelsInfo
from rag_common.identity import USER_ID_RE
from rag_common.secrets import read_secret
from rag_common.snapshots import checked_generation, selected_generations

COLLECTION = "chunks"
TEXT_VECTOR = "text"
IMAGE_VECTOR = "image"
BM25_FIELD = "search_text"
USER_FIELD = "user_id"  # owner of the chunk; every read filters on it (scope_filter)
SCHEMA_VERSION = 1
_NAMESPACE = uuid.UUID("6f1c3a52-8d0e-4c1b-9a57-2b4e0f6d9c11")

ChunkKind = Literal["text", "table", "image", "sheet_summary", "row", "summary"]


class ChunkPayload(BaseModel):
    doc_id: str
    doc_name: str
    doc_type: str
    kind: ChunkKind
    chunk_index: int
    text: str
    search_text: str
    text_sha: str
    n_tokens: int
    section: str = ""
    page: int | None = None
    page_end: int | None = None
    table_index: int | None = None
    sheet: str | None = None
    table: str | None = None  # SQL table holding this sheet's rows
    row_start: int | None = None
    row_end: int | None = None
    asset: str | None = None
    user_id: str | None = None
    generation: Generation | None = None


def qdrant_client(url: str, api_key_file: Path | None, timeout: int) -> AsyncQdrantClient:
    """A Qdrant client that sends the API key read from a secret file (None: no key, for tests and local runs).
    The key crosses only the internal Docker network over plain http, so qdrant-client's warning about an
    insecure connection is silenced here; a missing or empty key file fails at startup."""
    api_key = read_secret(api_key_file)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Api key is used with an insecure connection", category=UserWarning)
        return AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout)


def point_id(doc_id: str, chunk_index: int, generation: str | None = None) -> str:
    key = f"{doc_id}:{chunk_index}" if generation is None else f"{doc_id}:{checked_generation(generation)}:{chunk_index}"
    return str(uuid.uuid5(_NAMESPACE, key))


def generation_condition(generation: str | None):
    return (qm.IsEmptyCondition(is_empty=qm.PayloadField(key="generation")) if generation is None else
            qm.FieldCondition(key="generation", match=qm.MatchValue(value=checked_generation(generation))))


def user_condition(user_id: str) -> qm.FieldCondition:
    if not USER_ID_RE.match(user_id or ""):
        raise ValueError("user_id is not a user id")  # never build a filter that could match unowned points
    return qm.FieldCondition(key=USER_FIELD, match=qm.MatchValue(value=user_id))


def scope_filter(user_id: str, doc_ids: list[str]) -> qm.Filter:
    """The owner's published snapshot of each selected document, including pre-generation documents."""
    pinned = selected_generations(doc_ids)
    branches = [qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                                generation_condition(pinned[doc_id] if pinned is not None else None)])
                for doc_id in doc_ids]
    # An empty selection must match nothing (an empty should-list is otherwise unfiltered).
    return qm.Filter(must=[user_condition(user_id), qm.Filter(should=branches)] if branches else
                     [user_condition(user_id), qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=[]))])


_USER_INDEX = qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True)
_PAYLOAD_INDEXES = {
    "generation": qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD),
    "doc_id": qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True),
    USER_FIELD: _USER_INDEX,
    BM25_FIELD: qm.TextIndexParams(
        type=qm.TextIndexType.TEXT,
        tokenizer=qm.TokenizerType.WORD,
        lowercase=True,
        stemmer=qm.SnowballParams(type=qm.Snowball.SNOWBALL, language=qm.SnowballLanguage.ENGLISH),
        stopwords=qm.Language.ENGLISH,
        scoring=True,
    ),
}


def _metadata(info: ModelsInfo) -> dict:
    return {"schema_version": SCHEMA_VERSION, **info.model_dump(exclude={"rerank_model"})}


async def ensure_collection(client: AsyncQdrantClient, info: ModelsInfo) -> None:
    """Create the collection and payload indexes if missing; refuse a mismatched existing one.

    Indexes are created before any upsert so HNSW builds the per-doc_id links from the start.
    """
    expected = _metadata(info)
    if not await client.collection_exists(COLLECTION):
        try:
            await client.create_collection(
                COLLECTION,
                vectors_config={
                    TEXT_VECTOR: qm.VectorParams(size=info.text_dim, distance=qm.Distance.COSINE),
                    IMAGE_VECTOR: qm.VectorParams(size=info.image_dim, distance=qm.Distance.COSINE),
                },
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128, payload_m=16),
                metadata=expected,
            )
        except UnexpectedResponse as exc:  # another service created it concurrently
            if exc.status_code != 409:
                raise
    collection = await client.get_collection(COLLECTION)
    for field, schema in _PAYLOAD_INDEXES.items():
        if field not in (collection.payload_schema or {}):
            await client.create_payload_index(COLLECTION, field, schema, wait=True)
    actual = collection.config.metadata or {}
    if {k: actual.get(k) for k in expected} != expected:
        raise RuntimeError(
            f"collection '{COLLECTION}' was built with {actual}, services expect {expected}; "
            "delete the collection and reindex"
        )


async def ensure_user_index(client: AsyncQdrantClient) -> None:
    """Create the owner index on an existing collection if it is missing (used by the legacy migration)."""
    collection = await client.get_collection(COLLECTION)
    if USER_FIELD not in (collection.payload_schema or {}):
        await client.create_payload_index(COLLECTION, USER_FIELD, _USER_INDEX, wait=True)
