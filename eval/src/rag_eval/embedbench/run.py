"""rag-eval embed-bench: compare text embedders on recall@k, retrieval only (no LLM).

    python -m rag_eval.embedbench.run --local-chunks chunks.jsonl [--arms base 4b 8b vl8b] [--limit 50]

Runs in the local-rag/models image with the GPU (see docs/benchmarks/embedding-models-2026-11-15.md, "Rerun it").
Order: BM25 once per pool; then one embedder at a time (passage vectors cached per model and pool, dense top 40
per question); then the reranker once over every arm's fused top 30 (scores cached per question and passage);
then scores, intervals, the decision rule, and a JSON and Markdown report in the reports folder.
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_eval.embedbench import core
from rag_eval.paths import eval_dir, public_dir, reports_dir

ARMS = {  # name -> (model folder, kind)
    "base": ("Qwen3-Embedding-0.6B", "text"),
    "4b": ("Qwen3-Embedding-4B", "text"),
    "8b": ("Qwen3-Embedding-8B", "text"),
    "vl8b": ("Qwen3-VL-Embedding-8B", "vl"),
}
RERANKER = "Qwen3-Reranker-0.6B"
LATENCY_QUESTIONS = 50


def say(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def model_path(name: str, dirs: list[Path]) -> Path:
    for d in dirs:
        if (d / name / "config.json").is_file():
            return d / name
    raise SystemExit(f"model {name} not found in {', '.join(map(str, dirs))}")


def load_pools(args) -> tuple[list[core.Pool], list[str]]:
    pools, problems = [], []
    if "easy" in args.tiers:
        pool, problems = core.local(args.local_chunks, eval_dir("benchmarks") / "local_samples.jsonl", args.local_labels)
        if problems:
            raise SystemExit("labels do not match the exported chunks (reindexed?):\n  " + "\n  ".join(problems))
        pools.append(pool)
    if "hotpotqa" in args.tiers:
        pools.append(core.hotpot(args.hotpot, args.limit, args.seed))
    if "musique" in args.tiers:
        pools.append(core.musique(args.musique, args.limit, args.seed))
    return pools, problems


def corpus_key(pool: core.Pool) -> str:
    h = hashlib.sha1()
    for p in pool.passages:
        h.update(p.pid.encode())
        h.update(hashlib.sha1(p.text.encode("utf-8")).digest())
    return h.hexdigest()[:12]


def masks(pool: core.Pool) -> list[np.ndarray | None]:
    docs = np.array([p.doc for p in pool.passages])
    cache: dict[frozenset, np.ndarray] = {}
    out = []
    for q in pool.queries:
        if q.scope is None:
            out.append(None)
        else:
            if q.scope not in cache:
                cache[q.scope] = np.isin(docs, list(q.scope))
            out.append(cache[q.scope])
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag-eval embed-bench", description=__doc__.split("\n\n")[0])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--tiers", nargs="+", default=["easy", "hotpotqa", "musique"], choices=["easy", "hotpotqa", "musique"])
    p.add_argument("--limit", type=int, help="public sets: a random subset of N questions (default: all)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-chunks", type=Path, help="export_chunks.py output for the benchmark account")
    p.add_argument("--local-labels", type=Path, default=eval_dir("benchmarks") / "local_passages.jsonl")
    p.add_argument("--hotpot", type=Path, default=public_dir() / "hotpot_dev_distractor_v1.json")
    p.add_argument("--musique", type=Path, default=public_dir() / "musique_ans_v1.0_dev.jsonl")
    p.add_argument("--models", type=Path, nargs="+", default=[Path("/models/hf"), Path("/bench-models")],
                   help="folders that hold the model folders")
    p.add_argument("--cache", type=Path, default=Path("/cache"), help="passage vectors and rerank scores")
    p.add_argument("--query-instruction", default="app", choices=["app", "model-default", "none"],
                   help="app: the app's default instruction (General's, the default); model-default: each model's own; none")
    p.add_argument("--query-instruction-text", default=None,
                   help="the query instruction itself (an empty string: none); overrides --query-instruction")
    p.add_argument("--rerank-instruction", default=None, help="the reranker instruction (default: the app's)")
    p.add_argument("--label", default=None, help="a name for this run in the report (default: the instruction mode)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--out", type=Path, default=reports_dir())
    args = p.parse_args(argv)
    args.label = args.label or (args.query_instruction if args.query_instruction_text is None else "custom")
    if "easy" in args.tiers and not args.local_chunks:
        raise SystemExit("--local-chunks is required for the easy tier")
    from rag_eval.embedbench import gpu  # torch only from here on

    started = time.time()
    args.cache.mkdir(parents=True, exist_ok=True)
    pools, _ = load_pools(args)
    for pool in pools:
        say(f"pool {pool.name}: {len(pool.queries)} questions, {len(pool.passages)} passages, {pool.skipped} skipped")

    # BM25: the same for every arm
    bm25: dict[str, list[list[int]]] = {}
    allowed = {pool.name: masks(pool) for pool in pools}
    for pool in pools:
        t = time.time()
        index = core.BM25([x.text for x in pool.passages])
        bm25[pool.name] = [core.top(index.scores(q.text), core.DEPTH, m, positive_only=True)
                           for q, m in zip(pool.queries, allowed[pool.name], strict=True)]
        say(f"bm25 {pool.name}: {time.time() - t:.0f} s")

    dense: dict[str, dict[str, list[list[int]]]] = {}
    costs: dict[str, dict] = {}
    for name in args.arms:
        folder, kind = ARMS[name]
        arm = gpu.Arm(name, model_path(folder, args.models), kind)
        emb = gpu.Embedder(arm, query_mode=args.query_instruction, query_text=args.query_instruction_text)
        cost = {"model": folder, "dim": emb.dim, "weights_gib": round(emb.weights_gib, 2), "load_s": round(emb.load_s, 1),
                "passages_per_s": {}, "vectors_mb": {}}
        say(f"{name}: {folder} loaded, {emb.weights_gib:.1f} GiB, dim {emb.dim}")
        dense[name] = {}
        for pool in pools:
            path = args.cache / f"{name}-{pool.name}-{corpus_key(pool)}.npy"
            if path.is_file():
                vectors = np.load(path)
            else:
                t = time.time()
                vectors = emb.passages([x.text for x in pool.passages])
                cost["passages_per_s"][pool.name] = round(len(pool.passages) / (time.time() - t), 1)
                np.save(path, vectors)
            cost["vectors_mb"][pool.name] = round(vectors.shape[0] * vectors.shape[1] * 4 / 1e6, 1)
            texts = list(dict.fromkeys(q.text for q in pool.queries))
            qv = emb.queries(texts)
            row = {text: i for i, text in enumerate(texts)}
            qmat = qv[[row[q.text] for q in pool.queries]]
            dense[name][pool.name] = gpu.dense_top(vectors, qmat, core.DEPTH, allowed[pool.name])
            say(f"{name} {pool.name}: dense done ({cost['passages_per_s'].get(pool.name, 'cached')} passages/s)")
        sample = [q.text for q in pools[-1].queries[:LATENCY_QUESTIONS]]
        cost["query_ms"] = emb.query_latency_ms(sample)
        cost["peak_gib"] = round(emb.peak_gib(), 2)
        cost["card_gib_loaded"] = gpu.gpu_used_gib()
        costs[name] = cost
        emb.close()
        (args.cache / f"dense-{name}-{args.label}.json").write_text(json.dumps(dense[name]), encoding="utf-8")

    fused = {name: {pool.name: [core.rrf({"dense": d, "bm25": b})[:core.DEPTH]
                                for d, b in zip(dense[name][pool.name], bm25[pool.name], strict=True)]
                    for pool in pools} for name in args.arms}

    reranked: dict[str, dict[str, list[list[int]]]] = {}
    rerank_cost = {}
    if not args.no_rerank:
        # one score cache per reranker instruction (the app's keeps the original file name)
        rerank_text = args.rerank_instruction or gpu.RERANK_INSTRUCTION
        suffix = "" if rerank_text == gpu.RERANK_INSTRUCTION else "-" + hashlib.sha1(rerank_text.encode()).hexdigest()[:10]
        score_file = args.cache / f"rerank-scores{suffix}.json"
        cache: dict[str, float] = json.loads(score_file.read_text()) if score_file.is_file() else {}
        todo: dict[str, tuple[str, str]] = {}
        for pool in pools:
            for name in args.arms:
                for q, cand in zip(pool.queries, fused[name][pool.name], strict=True):
                    for i in cand[:core.RERANK_CANDIDATES]:
                        key = f"{pool.name}|{q.qid}|{pool.passages[i].pid}"
                        if key not in cache:
                            todo[key] = (q.text, pool.passages[i].text)
        say(f"rerank: {len(todo)} new pairs ({len(cache)} cached)")
        t = time.time()
        reranker = gpu.Reranker(model_path(RERANKER, args.models), instruction=rerank_text)
        # Similar lengths share a batch (less padding, same scores); saved every 20,000 pairs so a stop loses little.
        keys = sorted(todo, key=lambda k: len(todo[k][0]) + len(todo[k][1]))
        for start in range(0, len(keys), 20000):
            part = keys[start:start + 20000]
            for k, s in zip(part, reranker.scores([todo[k] for k in part]), strict=True):
                cache[k] = s
            score_file.write_text(json.dumps(cache))
            say(f"rerank: {min(start + 20000, len(keys))}/{len(keys)} pairs")
        rerank_cost = {"pairs": len(keys), "pairs_per_s": round(len(keys) / max(time.time() - t, 1e-9), 1)}
        for name in args.arms:
            reranked[name] = {}
            for pool in pools:
                reranked[name][pool.name] = [
                    core.rerank_order(cand, [cache[f"{pool.name}|{q.qid}|{pool.passages[i].pid}"]
                                             for i in cand[:core.RERANK_CANDIDATES]])
                    for q, cand in zip(pool.queries, fused[name][pool.name], strict=True)]

    scores: dict[tuple[str, str], dict[tuple[str, str], dict[str, float]]] = {}
    queries = [q for pool in pools for q in pool.queries]
    for name in args.arms:
        for pipeline, ranks in (("dense", dense[name]), ("hybrid", fused[name]), ("rerank", reranked.get(name))):
            if ranks is None:
                continue
            per = {}
            for pool in pools:
                for q, ranking in zip(pool.queries, ranks[pool.name], strict=True):
                    per[(q.tier, q.qid)] = core.score([pool.passages[i].pid for i in ranking], q.gold)
            scores[(name, pipeline)] = per
    summary = core.summarize(scores, queries)
    verdicts = core.decide(summary, args.arms) if reranked else {}
    write_report(args, pools, costs, rerank_cost, summary, verdicts, scores, time.time() - started)


def _ci(v) -> str:
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def write_report(args, pools, costs, rerank_cost, summary, verdicts, scores, seconds) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    revisions = {}
    for d in args.models:
        f = d / "REVISIONS.json"
        if f.is_file():
            revisions.update(json.loads(f.read_text()))
    meta = {"created": stamp, "seconds": round(seconds), "arms": args.arms, "limit": args.limit, "seed": args.seed,
            "query_instruction": args.query_instruction, "label": args.label,
            "query_instruction_text": args.query_instruction_text, "rerank_instruction": args.rerank_instruction,
            "revisions": revisions, "pools": {p.name: {"questions": len(p.queries), "passages": len(p.passages),
                                                        "skipped": p.skipped} for p in pools}}
    payload = {"meta": meta, "costs": costs, "rerank": rerank_cost, "verdicts": verdicts, "summary": summary,
               "per_query": {f"{a}/{pl}": {f"{t}/{q}": m for (t, q), m in per.items()} for (a, pl), per in scores.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"embed-bench-{stamp}-{args.label}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    qi = args.query_instruction if args.query_instruction_text is None else repr(args.query_instruction_text)
    lines = [f"# Embedding benchmark, retrieval only ({stamp} UTC), run {args.label}", "",
             f"Query instruction: {qi}. Reranker instruction: {args.rerank_instruction or 'the app default'}.", "",
             "Pools: " + "; ".join(f"{n}: {v['questions']} questions over {v['passages']} passages" for n, v in meta["pools"].items()),
             "", "Mean [95% bootstrap interval]. Delta = arm minus base on the same questions (paired bootstrap).", ""]
    if verdicts:
        lines += ["## Decision (rule: rerank recall@10, interval above 0 and gain >= 0.03 on 2 of 3 tiers)", "",
                  "| arm | tiers passed | better than base |", "|---|---|---|"]
        lines += [f"| {a} | {', '.join(v['tiers_passed']) or 'none'} | {'yes' if v['better'] else 'no'} |" for a, v in verdicts.items()]
        lines.append("")
    for group, rows in summary.items():
        if " / " in group:
            continue
        n = next(iter(rows.values()))["n"]
        lines += [f"## {group} ({n} questions)", "",
                  "| arm / pipeline | recall@1 | recall@5 | recall@10 | recall@20 | complete@10 | MRR@10 | nDCG@10 | delta recall@10 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, r in rows.items():
            delta = _ci(r["delta recall@10"]) if "delta recall@10" in r else ""
            lines.append(f"| {key} | {r['recall@1'][0]:.3f} | {r['recall@5'][0]:.3f} | {_ci(r['recall@10'])} | "
                         f"{r['recall@20'][0]:.3f} | {r['complete@10'][0]:.3f} | {r['mrr@10'][0]:.3f} | {r['ndcg@10'][0]:.3f} | {delta} |")
        lines.append("")
    strata = [g for g in summary if " / " in g]
    if strata:
        arms_pipes = list(next(iter(summary.values())).keys())
        lines += ["## recall@10 by stratum", "", "| stratum | n | " + " | ".join(arms_pipes) + " |",
                  "|---|---|" + "---|" * len(arms_pipes)]
        for g in strata:
            rows = summary[g]
            lines.append(f"| {g} | {next(iter(rows.values()))['n']} | " +
                         " | ".join(f"{rows[k]['recall@10'][0]:.3f}" for k in arms_pipes) + " |")
        lines.append("")
    lines += ["## Cost", "", "| arm | model | dim | weights GiB | peak GiB | card GiB loaded | passages/s | query ms p50 / p95 | vectors MB |",
              "|---|---|---|---|---|---|---|---|---|"]
    for a, c in costs.items():
        pps = ", ".join(f"{k} {v}" for k, v in c["passages_per_s"].items()) or "cached"
        mb = ", ".join(f"{k} {v}" for k, v in c["vectors_mb"].items())
        lines.append(f"| {a} | {c['model']} | {c['dim']} | {c['weights_gib']} | {c['peak_gib']} | {c['card_gib_loaded']} | {pps} | "
                     f"{c['query_ms']['p50']:.1f} / {c['query_ms']['p95']:.1f} | {mb} |")
    if rerank_cost:
        lines += ["", f"Reranker: {rerank_cost['pairs']} new pairs at {rerank_cost['pairs_per_s']} pairs/s."]
    lines += ["", f"Run time {seconds / 60:.0f} min. Model revisions: " +
              (", ".join(f"{k} {v[:10]}" for k, v in revisions.items()) or "not recorded")]
    (args.out / f"embed-bench-{stamp}-{args.label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("report:", args.out / f"embed-bench-{stamp}-{args.label}.md")


if __name__ == "__main__":
    main()
