"""Ranking metrics with label matching, plus percentile and paired bootstrap intervals.

Each query has labels L (relevant chunks, or relevant locations) and a ranked result list. A result
earns gain 1 the first time it matches a not-yet-matched label, so a label is counted once and
nDCG stays within [0, 1].
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Matcher = Callable[[dict, dict], bool]  # (result, label) -> relevant?


@dataclass(frozen=True)
class QueryScores:
    rr: float  # reciprocal rank of the first relevant result within the cutoff
    recall: dict[int, float]
    ndcg: float


def gains(results: Sequence[dict], labels: Sequence[dict], match: Matcher) -> list[int]:
    matched: set[int] = set()
    out = []
    for result in results:
        hit = next((i for i, label in enumerate(labels) if i not in matched and match(result, label)), None)
        if hit is None:
            out.append(0)
        else:
            matched.add(hit)
            out.append(1)
    return out


def score_query(results: Sequence[dict], labels: Sequence[dict], match: Matcher, cutoff: int = 10,
                recall_at: Sequence[int] = (1, 5, 10)) -> QueryScores:
    if not labels:
        raise ValueError("queries without labels (unanswerable) are excluded from retrieval metrics")
    g = gains(results[:cutoff], labels, match)
    first = next((j for j, x in enumerate(g, 1) if x), None)
    dcg = sum(x / math.log2(j + 1) for j, x in enumerate(g, 1))
    ideal = sum(1 / math.log2(j + 1) for j in range(1, min(cutoff, len(labels)) + 1))
    return QueryScores(
        rr=1.0 / first if first else 0.0,
        recall={k: sum(g[:k]) / len(labels) for k in recall_at},
        ndcg=dcg / ideal,
    )


def bootstrap_ci(values: Sequence[float], resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile interval over query resamples."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], resamples: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over the same queries, with a percentile interval."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), resamples, alpha, seed)


# --- label matchers ---------------------------------------------------------------------------


def strict(result: dict, label: dict) -> bool:
    return result["id"] == label["chunk_id"]


def _overlap(a0, a1, b0, b1) -> bool:
    if a0 is None or b0 is None:
        return False
    return a0 <= (b1 if b1 is not None else b0) and b0 <= (a1 if a1 is not None else a0)


def location(result: dict, label: dict) -> bool:
    """Same chunk, or the same document and an overlapping page range, sheet rows or figure."""
    if strict(result, label):
        return True
    if result["doc_id"] != label["doc_id"]:
        return False
    if label.get("sheet") is not None:
        return result.get("sheet") == label["sheet"] and _overlap(
            result.get("row_start"), result.get("row_end"), label.get("row_start"), label.get("row_end"))
    if label.get("asset"):
        return result.get("asset") == label["asset"]
    return _overlap(result.get("page"), result.get("page_end"), label.get("page"), label.get("page_end"))
