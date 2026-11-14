"""Retrieval ablations and answer-quality runs over a golden set."""

import json
from pathlib import Path

from rag_eval.client import Stack
from rag_eval.metrics import bootstrap_ci, location, paired_bootstrap, score_query, strict

ALL = ["dense", "bm25", "image", "grep"]
_PLAIN = {"rerank": False, "coverage": False, "mmr": False}
CONFIGS: dict[str, dict] = {
    "dense": {"retrievers": ["dense"], **_PLAIN},
    "bm25": {"retrievers": ["bm25"], **_PLAIN},
    "grep": {"retrievers": ["grep"], **_PLAIN},
    "dense+bm25": {"retrievers": ["dense", "bm25"], **_PLAIN},
    "dense+bm25+grep": {"retrievers": ["dense", "bm25", "grep"], **_PLAIN},
    "all four (RRF)": {"retrievers": ALL, **_PLAIN},
    "+ rerank": {"retrievers": ALL, "rerank": True, "coverage": False, "mmr": False},
    "+ rerank + coverage": {"retrievers": ALL, "rerank": True, "coverage": True, "mmr": False},
    "+ rerank + coverage + MMR (production)": {"retrievers": ALL, "rerank": True, "coverage": True, "mmr": True},
}
BASELINE = "+ rerank + coverage + MMR (production)"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_labels(stack: Stack, items: list[dict]) -> list[str]:
    """Labels whose chunk text changed since the golden set was written (same ID, different text)."""
    current: dict[str, str] = {}
    for doc_id in {r["doc_id"] for it in items for r in it["relevant"]}:
        current.update({c["id"]: c["text_sha"] for c in stack.chunks(doc_id)})
    return [f'{it["id"]}: {r["chunk_id"]}' for it in items for r in it["relevant"]
            if current.get(r["chunk_id"]) != r["text_sha"]]


def retrieval(stack: Stack, items: list[dict], configs: list[str], top_k: int = 10) -> dict:
    answerable = [it for it in items if it["answerable"] and it["relevant"]]
    per_config: dict[str, dict[str, list[float]]] = {}
    for name in configs:
        rows: dict[str, list[float]] = {}
        for it in answerable:
            res = stack.search(it["question"], it["doc_ids"], top_k=top_k, threshold=False, **CONFIGS[name])["results"]
            for level, matcher in (("strict", strict), ("location", location)):
                s = score_query(res, it["relevant"], matcher, cutoff=top_k)
                for metric, value in (("MRR@10", s.rr), ("nDCG@10", s.ndcg), *((f"Recall@{k}", v) for k, v in s.recall.items())):
                    rows.setdefault(f"{level} {metric}", []).append(value)
        per_config[name] = rows

    summary = {name: {m: bootstrap_ci(v) for m, v in rows.items()} for name, rows in per_config.items()}
    deltas = {}
    if BASELINE in per_config:
        base = per_config[BASELINE]
        for name, rows in per_config.items():
            if name != BASELINE:
                deltas[name] = {m: paired_bootstrap(rows[m], base[m]) for m in ("strict MRR@10", "location MRR@10")}
    return {"queries": len(answerable), "excluded_unanswerable": len(items) - len(answerable),
            "summary": summary, "delta_vs_production": deltas}


_FAITHFUL = """Sources:
{sources}

Answer to check:
\"\"\"{answer}\"\"\"

List every factual claim in the answer and whether the sources above support it.
Return JSON: {{"claims": [{{"claim": "...", "supported": true}}]}}"""

_CORRECT = """Question: {question}
Reference answer: {reference}
Candidate answer: {answer}
Score the candidate against the reference: 1 = correct and complete, 0.5 = partly correct, 0 = wrong or missing.
Return JSON: {{"score": 1, "reason": "..."}}"""

_ABSTAIN = """Question: {question}
Answer: {answer}
Does the answer say that the documents do not contain the information (rather than inventing an answer)?
Return JSON: {{"abstained": true}}"""


def _judge(stack: Stack, prompt: str, rows: dict[str, list[float]], max_tokens: int = 800) -> dict | None:
    """One LLM-judge call. Unusable JSON costs that one metric for that item (counted under "judge
    failures"), never the whole run; an unreachable model still stops the run."""
    try:
        verdict = stack.llm_json(prompt, max_tokens=max_tokens)
    except ValueError:  # malformed JSON
        verdict = None
    ok = isinstance(verdict, dict)
    rows.setdefault("judge failures", []).append(0.0 if ok else 1.0)
    return verdict if ok else None


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def answers(stack: Stack, items: list[dict], modes: list[str]) -> dict:
    results: dict[str, dict[str, list[float]]] = {}
    samples: dict[str, list[dict]] = {}
    for mode in modes:
        rows: dict[str, list[float]] = {}
        samples[mode] = []
        for it in items:
            out = stack.chat(it["question"], it["doc_ids"], mode=mode)
            if out.error:
                rows.setdefault("errors", []).append(1.0)
                continue
            rows.setdefault("errors", []).append(0.0)
            rows.setdefault("invalid citation rate", []).append(1.0 if out.invalid_citations else 0.0)
            if it["answerable"]:
                rows.setdefault("cited at least once", []).append(1.0 if out.citations else 0.0)
                srcs = "\n\n".join(f"[{s['n']}] {s['text']}" for s in out.sources)[:24000]
                faithful = _judge(stack, _FAITHFUL.format(sources=srcs, answer=out.answer), rows, max_tokens=1500)
                claims = [c for c in (faithful or {}).get("claims") or [] if isinstance(c, dict)]
                if claims:
                    rows.setdefault("faithfulness", []).append(sum(bool(c.get("supported")) for c in claims) / len(claims))
                verdict = _judge(stack, _CORRECT.format(question=it["question"], reference=it["reference_answer"],
                                                        answer=out.answer), rows)
                if verdict is not None and (score := _number(verdict.get("score", 0))) is not None:
                    rows.setdefault("correctness", []).append(score)
            else:
                verdict = _judge(stack, _ABSTAIN.format(question=it["question"], answer=out.answer), rows)
                if verdict is not None:
                    rows.setdefault("abstention", []).append(1.0 if verdict.get("abstained") else 0.0)
            samples[mode].append({"id": it["id"], "question": it["question"], "answer": out.answer, "route": out.route})
        results[mode] = rows
    return {"summary": {m: {k: bootstrap_ci(v) for k, v in rows.items()} for m, rows in results.items()},
            "samples": samples}
