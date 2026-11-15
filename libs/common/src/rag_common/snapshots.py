"""A request's immutable selection of published document generations.

None is the pre-generation layout. An absent context is for legacy callers; production RAG requests
bind their registry response once, before any research/SQL/asset read, and never refresh it mid-answer.
Published generations are retained until document deletion, so a reader needs no expiring lease.
"""

import re
from contextlib import contextmanager
from contextvars import ContextVar

GENERATION_RE = re.compile(r"^[0-9a-f]{32}$")
current_generations: ContextVar[dict[str, str | None] | None] = ContextVar("document_generations", default=None)


def checked_generation(value: str) -> str:
    if not isinstance(value, str) or not GENERATION_RE.fullmatch(value):
        raise ValueError("generation must be 32 lowercase hexadecimal characters")
    return value


def selected_generations(doc_ids: list[str]) -> dict[str, str | None] | None:
    pinned = current_generations.get()
    if pinned is None:
        return None
    if any(doc_id not in pinned for doc_id in doc_ids):
        raise ValueError("document is outside this request's published snapshot")
    return {doc_id: pinned[doc_id] for doc_id in doc_ids}


def generation_parameter(doc_id: str) -> dict[str, str]:
    pinned = selected_generations([doc_id])
    return {} if pinned is None else {"generation": pinned[doc_id] or "legacy"}


@contextmanager
def published_snapshots(documents):
    token = current_generations.set({doc.doc_id: doc.generation for doc in documents})
    try:
        yield
    finally:
        current_generations.reset(token)
