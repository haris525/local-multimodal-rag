"""Compare embed-bench runs on one half of the public questions, and choose among candidates without overfitting.

    python -m rag_eval.embedbench.compare --half tune|test --base C0.json C1.json C2.json ... \\
        [--arm base] [--pipeline rerank] [--choose C1 C2 C4 C5 C6]

Each report's per-question scores are filtered to one half (core.half: tune to choose by, test to report on); the
example-document tiers are small and always used whole. For every run: the mean of recall@10 and the other metrics
per tier with a 95% bootstrap interval, and the difference from --base on the same questions (paired bootstrap).
--choose picks, among the named runs, the one with the best recall@10 averaged over HotpotQA and MuSiQue with equal
weight (the rule for the General instruction, docs/design/search-tasks.md section 2); run it on the tune half.
"""

import argparse
import json
from pathlib import Path

from rag_eval.embedbench.core import half
from rag_eval.metrics import bootstrap_ci, paired_bootstrap

PUBLIC_TIERS = ("hotpotqa", "musique")
TIERS = ("easy-selected", "easy-all", "hotpotqa", "musique")
METRICS = ("recall@5", "recall@10", "complete@10", "mrr@10")


def load(path: Path) -> tuple[str, dict]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return report["meta"].get("label") or Path(path).stem, report["per_query"]


def keys(per_query: dict, tier: str, which: str) -> list[str]:
    """The question keys of one tier in one half ("all": every question)."""
    out = sorted(k for k in per_query if k.split("/", 1)[0] == tier)
    if which == "all" or tier not in PUBLIC_TIERS:
        return out
    return [k for k in out if half(k.split("/", 1)[1]) == which]


def table(runs: dict[str, dict], base: str, which: str, arm: str = "base", pipeline: str = "rerank") -> dict:
    """{tier: {label: {metric: (mean, lo, hi), "delta <metric>": (mean, lo, hi), "n": n}}}."""
    col = f"{arm}/{pipeline}"
    ref = runs[base][col]
    out = {}
    for tier in TIERS:
        ks = keys(ref, tier, which)
        if not ks:
            continue
        rows = {}
        for label, per_query in runs.items():
            per = per_query[col]
            row = {"n": len(ks)}
            for m in METRICS:
                row[m] = bootstrap_ci([per[k][m] for k in ks])
                if label != base:
                    row[f"delta {m}"] = paired_bootstrap([per[k][m] for k in ks], [ref[k][m] for k in ks])
            rows[label] = row
        out[tier] = rows
    return out


def choose(summary: dict, candidates: list[str]) -> tuple[str, dict[str, float]]:
    """The candidate with the best recall@10 averaged over HotpotQA and MuSiQue (equal weight)."""
    score = {c: sum(summary[t][c]["recall@10"][0] for t in PUBLIC_TIERS) / len(PUBLIC_TIERS) for c in candidates}
    return max(score, key=lambda c: (score[c], c)), score


def markdown(summary: dict, base: str, which: str) -> list[str]:
    lines = [f"Half: {which}. Mean [95% interval]; delta = run minus {base} on the same questions.", ""]
    for tier, rows in summary.items():
        n = next(iter(rows.values()))["n"]
        lines += [f"## {tier} ({n} questions)", "", "| run | recall@5 | recall@10 | complete@10 | MRR@10 | delta recall@10 |",
                  "|---|---|---|---|---|---|"]
        for label, r in rows.items():
            d = r.get("delta recall@10")
            delta = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""
            lines.append(f"| {label} | {r['recall@5'][0]:.3f} | {r['recall@10'][0]:.3f} [{r['recall@10'][1]:.3f}, "
                         f"{r['recall@10'][2]:.3f}] | {r['complete@10'][0]:.3f} | {r['mrr@10'][0]:.3f} | {delta} |")
        lines.append("")
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m rag_eval.embedbench.compare")
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--half", choices=["tune", "test", "all"], required=True)
    p.add_argument("--base", required=True, help="the label of the reference run")
    p.add_argument("--arm", default="base")
    p.add_argument("--pipeline", default="rerank", choices=["dense", "hybrid", "rerank"])
    p.add_argument("--choose", nargs="*", default=[], help="labels to choose the General instruction from")
    p.add_argument("--out", type=Path, help="write the Markdown here too")
    args = p.parse_args(argv)
    runs = dict(load(r) for r in args.reports)
    if args.base not in runs:
        raise SystemExit(f"--base {args.base} is not one of {sorted(runs)}")
    summary = table(runs, args.base, args.half, args.arm, args.pipeline)
    lines = [f"# embed-bench comparison: {args.arm} / {args.pipeline}", ""] + markdown(summary, args.base, args.half)
    if args.choose:
        best, score = choose(summary, args.choose)
        lines += ["## Choice (mean recall@10 over HotpotQA and MuSiQue)", ""]
        lines += [f"- {c}: {s:.4f}{'  <- chosen' if c == best else ''}" for c, s in sorted(score.items(), key=lambda x: -x[1])]
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
