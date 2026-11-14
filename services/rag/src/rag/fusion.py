"""Rank fusion and final selection. Pure functions over candidates."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from rag_common.index import ChunkPayload


@dataclass
class Candidate:
    id: str
    payload: ChunkPayload
    vector: np.ndarray | None = None  # normalized text embedding, for duplicate detection
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank
    rrf: float = 0.0
    score: float | None = None  # reranker probability


def rrf(rankings: dict[str, list[str]], weights: dict[str, float], k: int = 60) -> list[tuple[str, float]]:
    """Weighted reciprocal rank fusion: score(d) = sum_r w_r / (k + rank_r(d))."""
    scores: dict[str, float] = defaultdict(float)
    for name, ids in rankings.items():
        weight = weights.get(name, 1.0)
        seen: set[str] = set()
        for rank, doc in enumerate(ids, 1):
            if doc not in seen:
                seen.add(doc)
                scores[doc] += weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def source_count(n_docs: int, min_sources: int, max_sources: int) -> int:
    """K grows with the selection so multi-document questions get room: min(max, max(min, 2|D|))."""
    return min(max_sources, max(min_sources, 2 * n_docs))


def _passes(cand: Candidate, floor: float | None) -> bool:
    return floor is None or cand.score is None or cand.score >= floor


def _similarity(a: Candidate, b: Candidate) -> float:
    return float(a.vector @ b.vector) if a.vector is not None and b.vector is not None else 0.0


def select(
    ranked: list[Candidate],
    k: int,
    min_relevance: float | None,
    duplicate_cosine: float,
    coverage: bool,
    coverage_min_relevance: float | None = None,
    mmr_lambda: float | None = None,
    max_per_doc: int = 0,
    min_keep: int = 2,
    relative_relevance: float | None = None,
) -> list[Candidate]:
    """Choose the final K sources from candidates ordered best first:

    1. drop near-duplicates (cosine above `duplicate_cosine`);
    2. coverage: the best chunk of every document, above a low floor. Broad questions ("what are
       these documents about?") score low with a reranker trained to find answers, so a single
       threshold would leave documents out;
    3. fill the remaining slots above the main threshold, by maximal marginal relevance when
       `mmr_lambda` is set: argmax  lambda * rel(d) - (1 - lambda) * max_s cos(d, s), which balances
       relevance against redundancy and so spreads chunks across documents and topics;
    4. at most `max_per_doc` chunks per document when set (0 = no cap).
    With `relative_relevance`, the main threshold becomes min(min_relevance, ratio x best score),
    never below the coverage floor: when nothing literally answers the question (overviews,
    "key findings"), the bar drops so the context can hold more than one or two chunks.
    The result keeps the relevance order."""
    unique: list[Candidate] = []
    for cand in ranked:
        if not any(_similarity(cand, u) > duplicate_cosine for u in unique):
            unique.append(cand)
    top_rrf = max((c.rrf for c in unique), default=0.0) or 1.0
    scores = [c.score for c in unique if c.score is not None]
    if min_relevance is not None and relative_relevance is not None and scores:
        min_relevance = max(coverage_min_relevance or 0.0, min(min_relevance, relative_relevance * max(scores)))

    def relevance(c: Candidate) -> float:
        return c.score if c.score is not None else c.rrf / top_rrf

    chosen: list[Candidate] = []
    per_doc: dict[str, int] = {}

    def take(c: Candidate) -> None:
        chosen.append(c)
        per_doc[c.payload.doc_id] = per_doc.get(c.payload.doc_id, 0) + 1

    if coverage:
        for cand in unique:
            if len(chosen) < k and cand.payload.doc_id not in per_doc and _passes(cand, coverage_min_relevance):
                take(cand)
    pool = [c for c in unique if c not in chosen and _passes(c, min_relevance)]
    while len(chosen) < k and pool:
        if mmr_lambda is None:
            pick = pool[0]
        else:
            pick = max(pool, key=lambda c: mmr_lambda * relevance(c)
                       - (1 - mmr_lambda) * max((_similarity(c, s) for s in chosen), default=0.0))
        pool.remove(pick)
        if not max_per_doc or per_doc.get(pick.payload.doc_id, 0) < max_per_doc:
            take(pick)
    if len(chosen) < min_keep:  # keep a little evidence so the model can say what is missing
        for cand in [c for c in unique if c not in chosen][: min_keep - len(chosen)]:
            take(cand)
    order = {c.id: i for i, c in enumerate(unique)}
    return sorted(chosen, key=lambda c: order[c.id])
