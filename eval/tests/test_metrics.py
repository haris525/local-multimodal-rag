import math

import pytest

from rag_eval.metrics import bootstrap_ci, location, paired_bootstrap, score_query, strict


def r(cid: str, doc: str = "d", page: int | None = None, **kw) -> dict:
    return {"id": cid, "doc_id": doc, "page": page, "page_end": page, **kw}


def test_score_query_mrr_recall_ndcg():
    labels = [{"chunk_id": "x", "doc_id": "d"}, {"chunk_id": "y", "doc_id": "d"}]
    s = score_query([r("a"), r("x"), r("b"), r("y")], labels, strict, cutoff=10, recall_at=(1, 3))
    assert s.rr == 0.5
    assert s.recall == {1: 0.0, 3: 0.5}
    ideal = 1 + 1 / math.log2(3)
    assert s.ndcg == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)


def test_location_matches_overlapping_pages_and_sheet_rows_and_counts_a_label_once():
    page_label = {"chunk_id": "x", "doc_id": "d", "page": 3, "page_end": 3}
    assert location(r("other", page=3), page_label) and not location(r("other", page=4), page_label)
    row_label = {"chunk_id": "x", "doc_id": "d", "sheet": "S", "row_start": 10, "row_end": 12}
    assert location(r("z", sheet="S", row_start=12, row_end=20), row_label)
    s = score_query([r("p", page=3), r("q", page=3)], [page_label], location)
    assert (s.rr, s.recall[1], s.ndcg) == (1.0, 1.0, 1.0)


def test_bootstrap_intervals_bracket_the_mean_and_are_reproducible():
    values = [0.0, 0.5, 1.0, 1.0, 0.25]
    mean, lo, hi = bootstrap_ci(values)
    assert lo <= mean <= hi and bootstrap_ci(values) == (mean, lo, hi)
    assert paired_bootstrap([1, 1, 1], [0, 0, 0])[:3] == (1.0, 1.0, 1.0)
