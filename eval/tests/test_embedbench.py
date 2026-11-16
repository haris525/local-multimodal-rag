import json
import math

import numpy as np
import pytest
from rag_eval.embedbench import core


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_hotpot_pools_paragraphs_and_keeps_both_supporting_ones(tmp_path):
    rows = [
        {"_id": "a", "question": "q a", "type": "bridge", "level": "hard",
         "context": [["T1", ["One.", " Two."]], ["T2", ["Three."]], ["T3", ["Four."]]],
         "supporting_facts": [["T1", 0], ["T2", 0]]},
        {"_id": "b", "question": "q b", "type": "comparison", "level": "hard",
         "context": [["T1", ["One.", " Two."]], ["T4", ["Five."]]],
         "supporting_facts": [["T1", 1], ["T4", 0]]},
        {"_id": "c", "question": "q c", "type": "bridge", "level": "hard",
         "context": [["T5", ["Six."]]], "supporting_facts": [["T9", 0]]},  # gold not in context: skipped
    ]
    path = tmp_path / "h.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    pool = core.hotpot(path)
    assert len(pool.passages) == 5  # T1 is shared by a and b
    assert pool.skipped == 1
    a, b = pool.queries
    assert (a.stratum, b.stratum) == ("bridge", "comparison")
    texts = {p.pid: p.text for p in pool.passages}
    assert sorted(texts[next(iter(g))] for g in a.gold) == ["T1\nOne. Two.", "T2\nThree."]
    assert all(len(g) == 1 for g in a.gold)


def test_musique_strata_by_hops_and_supporting_paragraphs(tmp_path):
    rows = [{"id": "3hop1__1_2_3", "question": "q", "paragraphs": [
        {"title": "A", "paragraph_text": "a", "is_supporting": True},
        {"title": "B", "paragraph_text": "b", "is_supporting": False},
        {"title": "C", "paragraph_text": "c", "is_supporting": True},
        {"title": "D", "paragraph_text": "d", "is_supporting": True}]}]
    pool = core.musique(_jsonl(tmp_path / "m.jsonl", rows))
    (q,) = pool.queries
    assert q.stratum == "3hop" and len(q.gold) == 3 and len(pool.passages) == 4


def test_local_labels_groups_scopes_and_reindex_detection(tmp_path):
    chunks = _jsonl(tmp_path / "c.jsonl", [
        {"doc_name": "a.pdf", "chunk_index": 0, "text_sha": "s0", "search_text": "Document: a.pdf\nalpha"},
        {"doc_name": "a.pdf", "chunk_index": 1, "text_sha": "s1", "search_text": "Document: a.pdf\nbeta"},
        {"doc_name": "b.pdf", "chunk_index": 0, "text_sha": "s2", "search_text": "Document: b.pdf\ngamma"},
    ])
    samples = _jsonl(tmp_path / "s.jsonl", [
        {"id": "q1", "question": "alpha?", "docs": ["a.pdf"], "tags": ["single-document"]},
        {"id": "q2", "question": "table?", "docs": ["a.pdf"], "tags": ["table-sql"]},  # not a text question
        {"id": "q3", "question": "none?", "docs": ["b.pdf"], "tags": []},  # no labels: skipped
    ])
    labels = _jsonl(tmp_path / "l.jsonl", [
        {"id": "q1", "doc": "a.pdf", "text_sha": "s0", "group": 1},
        {"id": "q1", "doc": "a.pdf", "text_sha": "s1", "group": 1},  # an alternative for the same fact
        {"id": "q1", "doc": "b.pdf", "text_sha": "s2", "group": 2},
    ])
    pool, problems = core.local(chunks, samples, labels)
    assert problems == [] and pool.skipped == 1
    sel, everywhere = pool.queries
    assert (sel.tier, everywhere.tier) == ("easy-selected", "easy-all")
    assert sel.scope == frozenset({"a.pdf"}) and everywhere.scope is None
    assert sorted(len(g) for g in sel.gold) == [1, 2]

    stale = _jsonl(tmp_path / "l2.jsonl", [{"id": "q1", "doc": "a.pdf", "text_sha": "gone", "group": 1}])
    _, problems = core.local(chunks, samples, stale)
    assert problems and "gone"[:4] in problems[0]


def test_bm25_ranks_the_matching_passage_first_and_drops_non_matches():
    index = core.BM25(["the cat sat on the mat", "dogs bark loudly", "a cat and a dog"])
    s = index.scores("Which cat?")
    assert core.top(s, 3, positive_only=True) == [0, 2] or core.top(s, 3, positive_only=True) == [2, 0]
    assert 1 not in core.top(s, 3, positive_only=True)
    assert core.tokenize("What is the BM25 score?") == ["bm25", "score"]


def test_top_respects_the_mask_and_breaks_ties_by_index():
    s = np.array([0.5, 0.9, 0.9, 0.1])
    assert core.top(s, 3) == [1, 2, 0]
    assert core.top(s, 3, allowed=np.array([True, False, True, True])) == [2, 0, 3]
    assert core.top(s, 10, allowed=np.array([False, False, False, True])) == [3]


def test_rrf_matches_the_app_formula_and_tie_order():
    fused = core.rrf({"dense": ["a", "b", "c"], "bm25": ["c", "a"]}, k=60)
    score = {"a": 1 / 61 + 1 / 62, "b": 1 / 62, "c": 1 / 63 + 1 / 61}
    assert fused == sorted(score, key=lambda d: (-score[d], d))
    assert core.rrf({"x": ["b"], "y": ["a"]}) == ["a", "b"]  # equal scores: by id, as fusion.rrf


def test_rerank_order_reorders_the_candidates_and_keeps_the_tail():
    cand = list(range(35))
    scores = [0.0] * core.RERANK_CANDIDATES
    scores[29] = 0.9
    out = core.rerank_order(cand, scores)
    assert out[0] == 29 and out[1:30] == list(range(29)) and out[30:] == list(range(30, 35))


def test_score_counts_a_group_once_and_alternatives_satisfy_it():
    gold = [{"x", "x2"}, {"y"}]
    s = core.score(["a", "x", "x2", "y"], gold, ks=(1, 2, 4))
    assert s["recall@1"] == 0 and s["recall@2"] == 0.5 and s["recall@4"] == 1.0
    assert s["complete@2"] == 0 and s["complete@4"] == 1.0
    assert s["mrr@10"] == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert s["ndcg@10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)


def test_summarize_and_decide_follow_the_decision_rule():
    queries = [core.Query(str(i), "q", (frozenset({"g"}),), tier, "s") for tier in core.DECISION_TIERS for i in range(40)]
    hit, miss = {"recall@5": 1.0, "recall@10": 1.0, "complete@10": 1.0, "mrr@10": 1.0}, \
        {"recall@5": 0.0, "recall@10": 0.0, "complete@10": 0.0, "mrr@10": 0.0}
    base = {(q.tier, q.qid): (hit if int(q.qid) < 20 else miss) for q in queries}
    big = {(q.tier, q.qid): (hit if int(q.qid) < 32 else miss) for q in queries}
    same = dict(base)
    scores = {("base", "rerank"): base, ("big", "rerank"): big, ("same", "rerank"): same}
    summary = core.summarize(scores, queries)
    row = summary["hotpotqa"]["big / rerank"]
    assert row["delta recall@10"][0] == pytest.approx(0.3)
    verdicts = core.decide(summary, ["base", "big", "same"])
    assert verdicts["big"]["better"] and len(verdicts["big"]["tiers_passed"]) == 3
    assert not verdicts["same"]["better"]


def test_half_is_fixed_and_balanced():
    ids = [f"q{i}" for i in range(2000)]
    halves = [core.half(i) for i in ids]
    assert halves == [core.half(i) for i in ids]
    assert 900 < halves.count("tune") < 1100


def test_compare_filters_a_half_and_chooses_by_the_public_tiers():
    from rag_eval.embedbench import compare

    def per(r10):
        out = {}
        for tier in ("hotpotqa", "musique"):
            for i in range(200):
                v = r10(tier, i)
                out[f"{tier}/q{i}"] = {"recall@5": v, "recall@10": v, "complete@10": v, "mrr@10": v}
        for i in range(10):
            out[f"easy-selected/e{i}"] = {"recall@5": 1.0, "recall@10": 1.0, "complete@10": 1.0, "mrr@10": 1.0}
        return {"base/rerank": out}

    runs = {"C0": per(lambda t, i: 0.0), "C1": per(lambda t, i: 1.0 if t == "hotpotqa" else 0.0),
            "C2": per(lambda t, i: 1.0)}
    summary = compare.table(runs, "C0", "tune")
    n_tune = sum(core.half(f"q{i}") == "tune" for i in range(200))
    assert summary["hotpotqa"]["C0"]["n"] == n_tune
    assert summary["easy-selected"]["C0"]["n"] == 10  # small tiers are always whole
    assert summary["musique"]["C2"]["delta recall@10"][0] == pytest.approx(1.0)
    best, score = compare.choose(summary, ["C1", "C2"])
    assert best == "C2" and score["C1"] == pytest.approx(0.5)


def test_the_app_instructions_match_the_models_defaults():
    """embed-bench's "app" mode and its reranker use the defaults of services/models, whose query instruction is also
    General's (rag); gpu.py needs torch, so the files are read as source."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]

    def strings(path, names):
        found = {}
        for node in ast.walk(ast.parse((root / path).read_text(encoding="utf-8"))):
            target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
            if isinstance(target, ast.Name) and target.id in names and isinstance(node.value, ast.Constant):
                found[target.id] = node.value.value
        return found

    bench = strings("eval/src/rag_eval/embedbench/gpu.py", {"QUERY_INSTRUCTION", "RERANK_INSTRUCTION"})
    models = strings("services/models/src/models_svc/config.py", {"query_instruction", "rerank_instruction"})
    rag = strings("services/rag/src/rag/config.py", {"query_instruction"})
    assert bench["QUERY_INSTRUCTION"] == models["query_instruction"] == rag["query_instruction"]
    assert bench["RERANK_INSTRUCTION"] == models["rerank_instruction"]
