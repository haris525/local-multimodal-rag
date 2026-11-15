"""Four filtered retrievers -> RRF -> rerank -> threshold, duplicates, per-document coverage.

Every Qdrant read is scoped to the caller (the user in the request context): searches filter on the owner,
and reads by point ID (grep hits, deep-search reads, neighbours) use a filtered scroll instead of a plain
retrieve, because point IDs are deterministic and can be guessed; exhaustive mode reads whole documents with
the same owner and document filter (document_chunks). A point of another owner that still comes back is dropped
and logged. Without an identity in context, reads raise IdentityError (fail closed).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from rag_common.identity import require_user_id
from rag_common.index import (
    BM25_FIELD,
    COLLECTION,
    IMAGE_VECTOR,
    TEXT_VECTOR,
    USER_FIELD,
    ChunkPayload,
    generation_condition,
    point_id,
    scope_filter,
    user_condition,
)
from rag_common.logging import log_fields
from rag_common.snapshots import current_generations

from rag.clients import IngestClient, ModelsClient
from rag.config import Settings
from rag.fusion import Candidate, rrf, select, source_count
from rag.terms import exact_terms

log = logging.getLogger("rag.retrieval")
Retriever = Literal["dense", "bm25", "image", "grep"]
ALL_RETRIEVERS: tuple[Retriever, ...] = ("dense", "bm25", "image", "grep")


@dataclass(frozen=True)
class RowScope:
    """Planner 'filter' route: restrict one table of one document to these rows."""

    doc_id: str
    table: str
    rows: tuple[int, ...]


@dataclass
class SearchOptions:
    retrievers: tuple[Retriever, ...] = ALL_RETRIEVERS
    rerank: bool = True
    threshold: bool = True
    coverage: bool = True
    mmr: bool = True
    top_k: int | None = None  # None: grow with the selection size


@dataclass
class SearchResult:
    candidates: list[Candidate]
    terms: list[str]
    hits: dict[str, int]
    timings: dict[str, float] = field(default_factory=dict)


def _documents(doc_ids: list[str]) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))])


def selection_filter(user_id: str, doc_ids: list[str], scope: RowScope | None) -> qm.Filter:
    """The owner's chunks of the selected documents, with one table limited to the planner's rows. The owner
    condition sits outside the document part, so no branch of it (the RowScope `should`) can drop it."""
    if scope is None:
        return scope_filter(user_id, doc_ids)
    scoped = qm.Filter(
        must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=scope.doc_id)),
            qm.FieldCondition(key="table", match=qm.MatchValue(value=scope.table)),
            qm.FieldCondition(key="row_start", match=qm.MatchAny(any=list(scope.rows))),
        ]
    )
    others = [d for d in doc_ids if d != scope.doc_id]
    inner = qm.Filter(should=[_documents(others), scoped]) if others else scoped
    published = scope_filter(user_id, doc_ids)
    return qm.Filter(must=[user_condition(user_id), qm.Filter(must=[*published.must[1:], inner])])


def by_ids(user_id: str, ids: list[str]) -> qm.Filter:
    pinned = current_generations.get()
    conditions = scope_filter(user_id, list(pinned)).must[1:] if pinned is not None else [generation_condition(None)]
    return qm.Filter(must=[qm.HasIdCondition(has_id=ids), user_condition(user_id), *conditions])


def owned(points: list, user_id: str) -> list:
    """Drop points outside the owner or pinned generation, even if a backend violates the filter."""
    pinned = current_generations.get()
    keep = [p for p in points if (p.payload or {}).get(USER_FIELD) == user_id and
            (pinned is None and not (p.payload or {}).get("generation") or
             pinned is not None and (p.payload or {}).get("doc_id") in pinned and
             (p.payload or {}).get("generation") == pinned[(p.payload or {}).get("doc_id")])]
    if len(keep) != len(points):
        log_fields(log, "dropped points outside the owner or published snapshot", logging.ERROR, count=len(points) - len(keep))
    return keep


def _in_scope(payload: ChunkPayload, scope: RowScope | None) -> bool:
    if scope is None or payload.doc_id != scope.doc_id:
        return True
    return payload.table == scope.table and payload.row_start in scope.rows


def _vector(point) -> np.ndarray | None:
    vec = (point.vector or {}).get(TEXT_VECTOR) if isinstance(point.vector, dict) else None
    return np.asarray(vec, dtype=np.float32) if vec is not None else None


class Retrieval:
    def __init__(self, qdrant: AsyncQdrantClient, models: ModelsClient, ingest: IngestClient, s: Settings):
        self.qdrant = qdrant
        self.models = models
        self.ingest = ingest
        self.s = s
        self.weights = {"dense": s.weight_dense, "bm25": s.weight_bm25, "grep": s.weight_grep, "image": s.weight_image}

    async def search(
        self, question: str, doc_ids: list[str], opts: SearchOptions, scope: RowScope | None = None
    ) -> SearchResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        uid = require_user_id()
        flt = selection_filter(uid, doc_ids, scope)
        use = set(opts.retrievers)
        terms = exact_terms(question) if "grep" in use else []

        async def none():
            return None

        dense_vec, image_vec = await asyncio.gather(
            self.models.embed_query(question) if "dense" in use else none(),
            self.models.embed_image_query(question) if "image" in use else none(),
        )
        timings["embed_ms"] = (time.perf_counter() - start) * 1000

        names: list[str] = []
        requests: list[qm.QueryRequest] = []
        common = {"filter": flt, "with_payload": True, "with_vector": [TEXT_VECTOR]}
        if dense_vec is not None:
            names.append("dense")
            requests.append(qm.QueryRequest(query=dense_vec, using=TEXT_VECTOR, limit=self.s.dense_k, **common))
        if "bm25" in use:
            names.append("bm25")
            requests.append(qm.QueryRequest(query=qm.TextQuery(text=question), using=BM25_FIELD, limit=self.s.bm25_k, **common))
        if image_vec is not None:
            names.append("image")
            requests.append(
                qm.QueryRequest(query=image_vec, using=IMAGE_VECTOR, limit=self.s.image_k,
                                score_threshold=self.s.image_min_cosine, **common)
            )

        # Per-document candidates: the best dense matches inside each selected document, so one large
        # file (thousands of rows) cannot crowd a small one out of every ranked list.
        per_doc = (
            self.qdrant.query_points_groups(
                COLLECTION, group_by="doc_id", query=dense_vec, using=TEXT_VECTOR, query_filter=flt,
                limit=len(doc_ids), group_size=self.s.per_doc_k, with_payload=True, with_vectors=[TEXT_VECTOR],
            )
            if opts.coverage and dense_vec is not None
            else none()
        )
        t = time.perf_counter()
        batch, grep_hits, groups = await asyncio.gather(
            self.qdrant.query_batch_points(COLLECTION, requests=requests) if requests else none(),
            self.ingest.grep(doc_ids, terms, self.s.grep_k) if terms else none(),
            per_doc,
        )
        timings["search_ms"] = (time.perf_counter() - t) * 1000

        pool: dict[str, Candidate] = {}
        rankings: dict[str, list[str]] = {}
        for name, response in zip(names, batch or [], strict=True):
            rankings[name] = []
            for point in owned(response.points, uid):
                pid = str(point.id)
                if pid not in pool:
                    pool[pid] = Candidate(pid, ChunkPayload.model_validate(point.payload), _vector(point))
                rankings[name].append(pid)
        if grep_hits:
            for cand in await self.fetch([h.id for h in grep_hits if h.id not in pool]):
                pool[cand.id] = cand
            rankings["grep"] = [h.id for h in grep_hits if h.id in pool and _in_scope(pool[h.id].payload, scope)]
        for name, ids in rankings.items():
            for rank, pid in enumerate(ids, 1):
                pool[pid].ranks.setdefault(name, rank)

        fused = rrf(rankings, self.weights, self.s.rrf_k)[: self.s.rerank_candidates]
        ranked = [pool[pid] for pid, _ in fused]
        for cand, (_, score) in zip(ranked, fused, strict=True):
            cand.rrf = score
        for group in groups.groups if groups else []:  # guarantee each document a place in reranking
            for rank, point in enumerate(owned(group.hits, uid), 1):
                pid = str(point.id)
                cand = pool.setdefault(pid, Candidate(pid, ChunkPayload.model_validate(point.payload), _vector(point)))
                cand.ranks.setdefault("per_doc", rank)
                if cand not in ranked:
                    ranked.append(cand)

        if opts.rerank and ranked:
            t = time.perf_counter()
            scores = await self.models.rerank(question, [c.payload.search_text for c in ranked])
            for cand, score in zip(ranked, scores, strict=True):
                cand.score = score
            ranked.sort(key=lambda c: -c.score)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        k = opts.top_k or source_count(len(doc_ids), self.s.min_sources, self.s.max_sources)
        gated = opts.threshold and opts.rerank
        final = select(
            ranked,
            k=k,
            min_relevance=self.s.min_relevance if gated else None,
            duplicate_cosine=self.s.duplicate_cosine,
            coverage=opts.coverage,
            coverage_min_relevance=self.s.coverage_min_relevance if gated else None,
            mmr_lambda=self.s.mmr_lambda if opts.mmr else None,
            max_per_doc=self.s.max_per_doc,
            min_keep=self.s.min_keep,
            relative_relevance=self.s.relative_relevance,
        )
        timings["total_ms"] = (time.perf_counter() - start) * 1000
        return SearchResult(final, terms, {n: len(ids) for n, ids in rankings.items()}, timings)

    async def _scroll(self, ids: list[str], with_vectors) -> list:
        """The caller's points among these IDs; IDs of other owners' chunks match nothing."""
        uid = require_user_id()
        unique = list(dict.fromkeys(ids))
        points, _ = await self.qdrant.scroll(COLLECTION, scroll_filter=by_ids(uid, unique), limit=len(unique),
                                             with_payload=True, with_vectors=with_vectors)
        return owned(points, uid)

    async def document_chunks(self, doc_ids: list[str], skip_kinds: tuple[str, ...] = (),
                              page: int = 256) -> list[tuple[str, ChunkPayload]]:
        """Every chunk of these documents of the caller, as (point ID, payload) in reading order: document by
        document in the order given, each by chunk index. Chunks of `skip_kinds` are not read (exhaustive mode
        skips sheet rows and the generated document summary). A filtered scroll, page by page, like every read."""
        uid = require_user_id()
        flt = scope_filter(uid, doc_ids)
        if skip_kinds:
            flt = qm.Filter(must=flt.must, must_not=[qm.FieldCondition(key="kind",
                                                                       match=qm.MatchAny(any=list(skip_kinds)))])
        found: list = []
        offset = None
        while True:
            points, offset = await self.qdrant.scroll(COLLECTION, scroll_filter=flt, limit=page, offset=offset,
                                                      with_payload=True, with_vectors=False)
            found += owned(points, uid)
            if offset is None:
                break
        order = {d: i for i, d in enumerate(doc_ids)}
        chunks = [(str(p.id), ChunkPayload.model_validate(p.payload)) for p in found]
        chunks = [c for c in chunks if c[1].doc_id in order and c[1].kind not in skip_kinds]
        return sorted(chunks, key=lambda c: (order[c[1].doc_id], c[1].chunk_index))

    async def fetch(self, ids: list[str]) -> list[Candidate]:
        """Candidates for known point IDs, in the given order (missing IDs are skipped)."""
        if not ids:
            return []
        points = await self._scroll(ids, [TEXT_VECTOR])
        found = {str(p.id): Candidate(str(p.id), ChunkPayload.model_validate(p.payload), _vector(p)) for p in points}
        return [found[i] for i in ids if i in found]

    async def neighbours(self, cands: list[Candidate]) -> dict[str, tuple[ChunkPayload | None, ChunkPayload | None]]:
        """Previous and next prose chunk of the same section for each prose candidate."""
        wanted: dict[str, tuple[str, str]] = {}
        for c in cands:
            if c.payload.kind == "text":
                p = c.payload
                wanted[c.id] = (point_id(p.doc_id, p.chunk_index - 1, p.generation),
                                point_id(p.doc_id, p.chunk_index + 1, p.generation))
        if not wanted:
            return {}
        ids = sorted({i for pair in wanted.values() for i in pair})
        found = {str(pt.id): ChunkPayload.model_validate(pt.payload) for pt in await self._scroll(ids, False)}
        by_id = {c.id: c for c in cands}

        def same_section(pid: str, cid: str) -> ChunkPayload | None:
            n = found.get(pid)
            ref = by_id[cid].payload
            return n if n and n.kind == "text" and n.section == ref.section else None

        return {cid: (same_section(prev, cid), same_section(nxt, cid)) for cid, (prev, nxt) in wanted.items()}
