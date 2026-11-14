import numpy as np
import pytest

from rag_common.index import ChunkPayload

from rag.fusion import Candidate, rrf, select, source_count


def cand(cid: str, doc: str, score: float, vec: list[float]) -> Candidate:
    payload = ChunkPayload(doc_id=doc, doc_name=doc, doc_type="pdf", kind="text", chunk_index=0, text="t",
                           search_text="t", text_sha="x", n_tokens=1)
    v = np.asarray(vec, dtype=np.float32)
    return Candidate(cid, payload, v / np.linalg.norm(v), score=score)


def test_rrf_sums_weighted_reciprocal_ranks_and_ignores_repeats():
    fused = dict(rrf({"dense": ["a", "b", "a"], "bm25": ["b", "c"]}, {"dense": 1.0, "bm25": 0.5}, k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 0.5 / 61)
    assert fused["c"] == pytest.approx(0.5 / 62)
    assert max(fused, key=fused.get) == "b"


def test_select_covers_every_document_before_thresholding_the_rest():
    ranked = [
        cand("a1", "A", 0.9, [1, 0, 0]),
        cand("a2", "A", 0.8, [1, 0.01, 0]),  # near-duplicate of a1
        cand("a3", "A", 0.7, [0, 1, 0]),
        cand("b1", "B", 0.01, [0, 0, 1]),    # below the main threshold, but document B's best chunk
        cand("c1", "C", 0.001, [1, 1, 1]),   # below even the coverage floor
    ]
    args = {"min_relevance": 0.05, "duplicate_cosine": 0.95, "coverage_min_relevance": 0.005}
    assert [c.id for c in select(ranked, k=4, coverage=True, **args)] == ["a1", "a3", "b1"]
    assert [c.id for c in select(ranked, k=4, coverage=False, **args)] == ["a1", "a3"]


def test_mmr_prefers_a_different_topic_over_a_near_copy_and_cap_limits_each_document():
    ranked = [
        cand("a1", "A", 0.95, [1, 0, 0]),
        cand("a2", "A", 0.90, [0.9, 0.3, 0]),   # close to a1
        cand("b1", "B", 0.80, [0, 0, 1]),       # different topic
    ]
    args = {"min_relevance": 0.05, "duplicate_cosine": 0.99, "coverage": False}
    assert [c.id for c in select(ranked, k=2, mmr_lambda=None, **args)] == ["a1", "a2"]
    assert [c.id for c in select(ranked, k=2, mmr_lambda=0.5, **args)] == ["a1", "b1"]
    assert [c.id for c in select(ranked, k=3, max_per_doc=1, **args)] == ["a1", "b1"]


def test_threshold_adapts_when_nothing_scores_high():
    weak = [cand(f"c{i}", "A", score, [i == j for j in range(4)]) for i, score in enumerate([0.05, 0.03, 0.02, 0.004])]
    args = {"k": 8, "min_relevance": 0.05, "duplicate_cosine": 0.95, "coverage": True, "coverage_min_relevance": 0.005}
    assert [c.id for c in select(weak, **args)] == ["c0", "c1"]  # fixed bar: only min_keep survives
    assert [c.id for c in select(weak, relative_relevance=0.25, **args)] == ["c0", "c1", "c2"]  # bar 0.0125


def test_source_count_grows_with_selection_within_bounds():
    assert [source_count(n, 8, 16) for n in (1, 5, 10)] == [8, 10, 16]
