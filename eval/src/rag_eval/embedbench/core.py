"""Pure parts of the embedding benchmark: question pools, BM25, fusion, scoring and summaries (no torch).

Passages and questions come from three tiers:
- easy: the app's own chunks of the benchmark documents with hand-checked gold chunks (eval/benchmarks/
  local_passages.jsonl), searched in the question's selected documents (easy-selected, as the app does) and
  across every document (easy-all);
- hotpotqa: HotpotQA dev distractor, 2-hop, every question's paragraphs pooled into one shared corpus;
- musique: MuSiQue-Ans dev, 2 to 4 hops, pooled the same way.
"""

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_eval.metrics import bootstrap_ci, paired_bootstrap

KS = (1, 5, 10, 20, 40)
DEPTH = 40  # the app's dense_k and bm25_k
RRF_K = 60  # the app's rrf_k (equal weights for dense and bm25)
RERANK_CANDIDATES = 30  # the app's rerank_candidates
PIPELINES = ("dense", "hybrid", "rerank")
DECISION_METRIC = "recall@10"
DECISION_TIERS = ("easy-selected", "hotpotqa", "musique")
MIN_GAIN = 0.03


@dataclass(frozen=True)
class Passage:
    pid: str
    text: str
    doc: str = ""  # app documents: the file name; public sets: empty


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    gold: tuple[frozenset[str], ...]  # groups: each group is satisfied by retrieving any one of its passages
    tier: str  # easy-selected, easy-all, hotpotqa, musique
    stratum: str  # single-document, cross-document, bridge, comparison, 2hop, 3hop, 4hop
    scope: frozenset[str] | None = None  # document names searched; None: the whole pool


@dataclass
class Pool:
    name: str
    passages: list[Passage]
    queries: list[Query]
    skipped: int = 0  # questions left out (gold passage missing from the pool)


def pid_of(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def half(qid: str) -> str:
    """A fixed split of the public questions: "tune" (to choose by) or "test" (to report on), by a hash of the id."""
    return "tune" if int(hashlib.sha1(qid.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "test"


def _sample(rows: list, limit: int | None, seed: int) -> list:
    if not limit or limit >= len(rows):
        return rows
    keep = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[i] for i in keep]


def hotpot(path: Path, limit: int | None = None, seed: int = 7) -> Pool:
    """HotpotQA dev distractor: each question's 10 paragraphs join one shared pool; gold = its 2 supporting ones."""
    rows = _sample(json.loads(Path(path).read_text(encoding="utf-8")), limit, seed)
    passages: dict[str, Passage] = {}
    queries, skipped = [], 0
    for row in rows:
        ids = {}
        for title, sentences in row["context"]:
            text = f"{title}\n{''.join(sentences).strip()}"  # sentences keep their leading spaces
            pid = pid_of(title, text)
            passages.setdefault(pid, Passage(pid, text))
            ids[title] = pid
        titles = {t for t, _ in row["supporting_facts"]}
        if not titles or not titles <= ids.keys():
            skipped += 1
            continue
        gold = tuple(frozenset([ids[t]]) for t in sorted(titles))
        queries.append(Query(row["_id"], row["question"], gold, "hotpotqa", row["type"]))
    return Pool("hotpotqa", list(passages.values()), queries, skipped)


def musique(path: Path, limit: int | None = None, seed: int = 7) -> Pool:
    """MuSiQue-Ans dev: 20 paragraphs per question into one shared pool; gold = its 2 to 4 supporting ones."""
    rows = _sample([json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()],
                   limit, seed)
    passages: dict[str, Passage] = {}
    queries, skipped = [], 0
    for row in rows:
        gold = set()
        for p in row["paragraphs"]:
            text = f"{p['title']}\n{p['paragraph_text'].strip()}"
            pid = pid_of(p["title"], text)
            passages.setdefault(pid, Passage(pid, text))
            if p["is_supporting"]:
                gold.add(pid)
        hops = re.match(r"(\d)hop", row["id"])
        if not gold or not hops:
            skipped += 1
            continue
        groups = tuple(frozenset([pid]) for pid in sorted(gold))
        queries.append(Query(row["id"], row["question"], groups, "musique", f"{hops.group(1)}hop"))
    return Pool("musique", list(passages.values()), queries, skipped)


def text_question(sample: dict) -> bool:
    """Questions answered from passages: table questions go to SQL and image questions to the image index."""
    return not {"table-sql", "image"} & set(sample.get("tags", []))


def local(chunks: Path, samples: Path, labels: Path) -> tuple[Pool, list[str]]:
    """The app's chunks (export_chunks.py) with gold chunks from the labels file, matched by document name and
    text sha256 so a reindex shows up as a missing label instead of a silent miss. Labels of one question that
    share a `group` are alternatives (the same fact stated in two chunks); each group is one gold item. Each
    question is asked twice: in its selected documents (easy-selected) and across every document (easy-all)."""
    rows = [json.loads(line) for line in Path(chunks).read_text(encoding="utf-8").splitlines() if line.strip()]
    passages = [Passage(f"{r['doc_name']}#{r['chunk_index']}", r["search_text"], r["doc_name"]) for r in rows]
    by_sha = {(r["doc_name"], r["text_sha"]): p.pid for r, p in zip(rows, passages, strict=True)}
    gold: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    problems = []
    for line in Path(labels).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        pid = by_sha.get((label["doc"], label["text_sha"]))
        if pid is None:
            problems.append(f"{label['id']}: no chunk of {label['doc']} has text_sha {label['text_sha'][:12]}")
        else:
            gold[label["id"]][str(label.get("group", label["text_sha"]))].add(pid)
    queries, skipped = [], 0
    for line in Path(samples).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        if not text_question(s):
            continue
        if not gold.get(s["id"]):
            skipped += 1
            continue
        stratum = "cross-document" if "cross-document" in s.get("tags", []) else "single-document"
        g = tuple(frozenset(v) for _, v in sorted(gold[s["id"]].items()))
        queries.append(Query(s["id"], s["question"], g, "easy-selected", stratum, frozenset(s["docs"])))
        queries.append(Query(s["id"], s["question"], g, "easy-all", stratum, None))
    return Pool("easy", passages, queries, skipped), problems


# --- BM25 ---------------------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when", "where", "which", "who", "whom", "why", "with"])


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 (k1 1.2, b 0.75) over an inverted index; the same for every embedder arm."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        docs = [Counter(tokenize(t)) for t in texts]
        self.n = len(docs)
        lengths = np.array([sum(d.values()) for d in docs], dtype=float)
        avg = float(lengths.mean()) if self.n and lengths.mean() > 0 else 1.0
        postings: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
        for i, d in enumerate(docs):
            norm = k1 * (1 - b + b * lengths[i] / avg)
            for term, f in d.items():
                ids, weights = postings[term]
                ids.append(i)
                weights.append(f * (k1 + 1) / (f + norm))
        self.index = {}
        for term, (ids, weights) in postings.items():
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            self.index[term] = (np.asarray(ids, dtype=np.int64), np.asarray(weights, dtype=np.float32) * idf)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term in self.index:
                ids, weights = self.index[term]
                s[ids] += weights
        return s


def top(scores: np.ndarray, k: int, allowed: np.ndarray | None = None, positive_only: bool = False) -> list[int]:
    """Indexes of the k best scores, best first (ties: lower index first), within `allowed` if given."""
    s = scores.astype(np.float64, copy=True)
    if allowed is not None:
        s[~allowed] = -np.inf
    if positive_only:
        s[s <= 0] = -np.inf
    k = min(k, s.size)
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.lexsort((idx, -s[idx]))]
    return [int(i) for i in idx if np.isfinite(s[i])]


def rrf(rankings: dict[str, Sequence], k: int = RRF_K, weights: dict[str, float] | None = None) -> list:
    """Weighted reciprocal rank fusion, as services/rag/src/rag/fusion.py: score(d) = sum_r w_r / (k + rank_r(d))."""
    scores: dict = defaultdict(float)
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        seen = set()
        for rank, d in enumerate(ids, 1):
            if d not in seen:
                seen.add(d)
                scores[d] += w / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def rerank_order(candidates: Sequence, scores: Sequence[float], depth: int = DEPTH) -> list:
    """The reranked candidates (best score first, fused order on ties), then the rest of the fused list."""
    head = list(candidates[:RERANK_CANDIDATES])
    order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
    return ([head[i] for i in order] + list(candidates[RERANK_CANDIDATES:]))[:depth]


# --- scoring --------------------------------------------------------------------------------------


def score(ranking: Sequence, gold: Sequence[Iterable], ks: Sequence[int] = KS) -> dict[str, float]:
    """recall@k (share of gold groups found), complete@k (all found), MRR@10 and binary nDCG@10. A result counts
    once, for the first gold group it satisfies that is not yet satisfied, so nDCG stays within [0, 1]."""
    groups = [set(g) for g in gold]
    seen, done, hits = set(), set(), []
    for d in ranking:
        if d in seen:
            continue
        seen.add(d)
        hit = next((i for i, g in enumerate(groups) if i not in done and d in g), None)
        if hit is not None:
            done.add(hit)
        hits.append(0 if hit is None else 1)
    gold = groups
    out = {}
    for k in ks:
        found = sum(hits[:k])
        out[f"recall@{k}"] = found / len(gold)
        out[f"complete@{k}"] = float(found == len(gold))
    out["mrr@10"] = next((1 / (i + 1) for i, h in enumerate(hits[:10]) if h), 0.0)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:10]))
    out["ndcg@10"] = dcg / sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return out


def summarize(scores: dict[tuple[str, str], dict[tuple[str, str], dict[str, float]]], queries: Sequence[Query],
              base: str = "base") -> dict:
    """Group per-query scores by tier and by tier/stratum; means with 95% bootstrap intervals, and each arm minus
    the base arm on the same queries (paired bootstrap). `scores[(arm, pipeline)][(tier, qid)] = metrics`."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for q in queries:
        groups[q.tier].append((q.tier, q.qid))
        groups[f"{q.tier} / {q.stratum}"].append((q.tier, q.qid))
    out = {}
    for group, keys in sorted(groups.items()):
        rows = {}
        for (arm, pipeline), per_query in sorted(scores.items()):
            metrics = list(next(iter(per_query.values())).keys())
            row = {"n": len(keys)}
            for m in metrics:
                row[m] = bootstrap_ci([per_query[k][m] for k in keys])
            if arm != base and (base, pipeline) in scores:
                ref = scores[(base, pipeline)]
                for m in ("recall@5", "recall@10", "complete@10", "mrr@10"):
                    row[f"delta {m}"] = paired_bootstrap([per_query[k][m] for k in keys], [ref[k][m] for k in keys])
            rows[f"{arm} / {pipeline}"] = row
        out[group] = rows
    return out


def decide(summary: dict, arms: Sequence[str], base: str = "base") -> dict[str, dict]:
    """The decision rule, fixed before the run: in the rerank pipeline, recall@10 minus base has a 95% interval above 0 and a mean of at
    least MIN_GAIN on at least two of the three decision tiers."""
    verdicts = {}
    for arm in arms:
        if arm == base:
            continue
        passed = []
        for tier in DECISION_TIERS:
            row = summary.get(tier, {}).get(f"{arm} / rerank", {})
            mean, lo, _ = row.get(f"delta {DECISION_METRIC}", (float("nan"),) * 3)
            if lo > 0 and mean >= MIN_GAIN:
                passed.append(tier)
        verdicts[arm] = {"tiers_passed": passed, "better": len(passed) >= 2}
    return verdicts
