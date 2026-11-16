"""Several deep-search variants in one fair run.

A variant is an agent name plus optional suffixes, in any order: +think (deep search's thinking, deep_think: true),
+fanout (researchers on reserved budgets, fanout: true), +calls<N> (a research budget of N calls,
research_calls: N) and +python (the run_python tool, python: true), for example "deepagents+think" or
"deepagents+fanout+calls20". rag checks each against its settings and answers a 422 for an arm without fan-out,
N outside its range, or Python where it is off (+python needs RAG_DEEP_PYTHON=request or on, RAG_SANDBOX_URL and the
compose profile sandbox). Fairness:

- Interleaved, randomized order: each repeat shuffles the questions, and each question runs every variant in its
  own shuffled order (seeded, so a run can be repeated exactly). GPU contention, cache warmth and drift over the
  hours of a run therefore reach every variant alike, instead of variant A getting the morning and B the evening.
- Identical requests: the same question text and doc_ids, answer thinking off, the same mode. Temperature and seed
  are rag's settings (the chat API takes neither), so every variant in one run uses the same ones; the settings
  the rag container sets are recorded with its image id, the git commit, the model and LLM_CTX.
- Model-free scores only (numbers, facts, citations, exact match, F1). No LLM judge takes part, so none can break
  a tie.

The run writes one report per variant (the same format as a single run, so `compare` works on any two) and a
combined report: every variant side by side, per tag, and a verdict for all questions, the local set and the
held-out public sets: the variant with the highest correctness against each other one, with a paired permutation
test, Holm-adjusted for the number of comparisons.
"""

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_eval import bench
from rag_eval.bench import Item
from rag_eval.client import Gateway, Turn
from rag_eval.dispatch import dispatch

PUBLIC_TAGS = ("hotpotqa", "musique")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SUFFIX = re.compile(r"^(?:think|fanout|python|calls([1-9][0-9]*))$")
_USAGE = ("write AGENT or AGENT+think, with +fanout and +calls<N> as further options in any order (+python too), for "
          "example deepagents+think or deepagents+fanout+calls20")


@dataclass(frozen=True)
class Variant:
    label: str  # as given: "deepagents+think"
    agent: str  # sent as the request's agent field
    deep_think: bool  # sent as deep_think: true
    fanout: bool = False  # sent as fanout: true
    research_calls: int | None = None  # sent as research_calls: N
    python: bool = False  # sent as python: true

    def request(self) -> tuple:
        """What this variant changes in the request, whatever the order of its suffixes."""
        return self.agent, self.deep_think, self.fanout, self.research_calls, self.python


def parse_variant(raw: str) -> Variant:
    """'deepagents+fanout+calls20' -> a variant; ValueError for anything else."""
    agent, *flags = raw.split("+")
    if not _NAME.match(agent):
        raise ValueError(f"variant {raw!r}: {_USAGE}")
    seen: dict[str, str] = {}
    for flag in flags:
        match = _SUFFIX.match(flag)
        if not match:
            raise ValueError(f"variant {raw!r}: {_USAGE}")
        kind = "calls" if match.group(1) else flag
        if kind in seen:
            raise ValueError(f"variant {raw!r}: +{kind} is given twice")
        seen[kind] = match.group(1) or ""
    return Variant(raw, agent, "think" in seen, "fanout" in seen, int(seen["calls"]) if "calls" in seen else None,
                   "python" in seen)


def parse_variants(spec: str) -> list[Variant]:
    """'langgraph,langgraph+think,deepagents+fanout' -> variants; ValueError for anything else."""
    out: list[Variant] = []
    for raw in (part.strip() for part in spec.split(",")):
        if not raw:
            continue
        variant = parse_variant(raw)
        if any(v.label == raw for v in out):
            raise ValueError(f"variant {raw!r} is listed twice")
        same = next((v for v in out if v.request() == variant.request()), None)
        if same:
            raise ValueError(f"variants {same.label!r} and {raw!r} send the same request")
        out.append(variant)
    if len(out) < 2:
        raise ValueError("--variants needs at least two variants, separated by commas")
    files = [bench.safe_label(v.label) for v in out]
    if len(set(files)) != len(files):
        raise ValueError(f"variants {', '.join(v.label for v in out)} would share report file names")
    return out


def schedule(items: list[Item], variants: list[Variant], repeats: int, seed: int) -> list[tuple[int, int, Item, Variant]]:
    """(repeat, position of the question in this repeat, item, variant) in run order."""
    rng = random.Random(seed)
    plan = []
    for r in range(1, repeats + 1):
        order = list(items)
        rng.shuffle(order)
        for i, item in enumerate(order, 1):
            mixed = list(variants)
            rng.shuffle(mixed)
            plan += [(r, i, item, v) for v in mixed]
    return plan


def run(gateway: Gateway, items: list[Item], doc_ids: dict[str, str], mode: str, repeats: int, variants: list[Variant],
        seed: int, sink: Callable[[dict], None] | None = None, progress: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic, concurrency: int = 1,
        search_task: str | None = None) -> dict[str, list[dict]]:
    """Run the interleaved schedule; returns {variant label: run records}. Answer thinking is always off, and every
    variant asks with the same `search_task` (when given).

    Requests are sent in the schedule's order; with concurrency above 1 up to that many are in flight
    (rag_eval.dispatch), so records arrive in the order they finish: `order` is each record's place in the
    schedule and `in_flight` the number of other requests running when it was sent."""
    plan = schedule(items, variants, repeats, seed)
    runs: dict[str, list[dict]] = {v.label: [] for v in variants}
    passed = dict.fromkeys(runs, 0)
    started, finished = clock(), 0

    def ask(job: tuple[int, int, Item, Variant]) -> Turn:
        _, _, item, v = job
        return gateway.ask(item.question, [doc_ids[name] for name in item.docs], mode=mode, think=False,
                           agent=v.agent, deep_think=v.deep_think, fanout=v.fanout, research_calls=v.research_calls,
                           python=v.python, search_task=search_task)

    def done(k: int, job: tuple[int, int, Item, Variant], turn: Turn, others: int) -> None:
        nonlocal finished
        r, i, item, v = job
        rec = bench.record(item, r, turn, doc_ids) | {"variant": v.label, "order": k}
        if concurrency > 1:
            rec["in_flight"] = others
        runs[v.label].append(rec)
        passed[v.label] += rec["scores"]["correct"] == 1.0
        finished += 1
        if sink:
            sink(rec)
        progress(progress_line(rec, finished, len(plan), r, repeats, i, len(items), runs, passed, clock() - started))

    dispatch(plan, ask, done, concurrency)
    return runs


def progress_line(rec: dict, done: int, total: int, repeat: int, repeats: int, index: int, n: int,
                  runs: dict[str, list[dict]], passed: dict[str, int], elapsed: float) -> str:
    verdict = "PASS" if rec["scores"]["correct"] == 1.0 else "ERROR" if rec["error"] else "FAIL"
    rates = ", ".join(f"{label} {passed[label]}/{len(rs)}" for label, rs in runs.items() if rs)
    left = elapsed / done * (total - done)
    return (f"[{done}/{total}] repeat {repeat}/{repeats}, question {index}/{n} {rec['id']}, {rec['variant']}: {verdict} "
            f"{rec['latency_s']:.0f}s, {rec['tool_errors']} tool errors | pass so far: {rates} | "
            f"{bench._duration(elapsed)} elapsed, ~{bench._duration(left)} left")


# --- the combined report ----------------------------------------------------------------------------------


def groups(items: list[Item]) -> dict[str, list[str]]:
    """The question groups a verdict is given for: all, and the local and held-out sets when both ran."""
    out = {"all questions": [it.id for it in items]}
    local = [it.id for it in items if not set(PUBLIC_TAGS) & set(it.tags)]
    held = [it.id for it in items if set(PUBLIC_TAGS) & set(it.tags)]
    if local and held:
        out["local set"] = local
        out["held-out public sets"] = held
    return out


def group_verdict(pq: dict[str, dict[str, dict]], ids: list[str], metric: str = "correct") -> dict:
    """The variant with the highest mean against every other, paired, with Holm-adjusted permutation p-values,
    and one plain sentence."""
    labels = list(pq)
    means = {label: float(np.mean([pq[label][q][metric] for q in ids if q in pq[label]])) for label in labels}
    ranked = sorted(labels, key=lambda label: -means[label])
    best, others = ranked[0], ranked[1:]
    against = {o: bench.paired(pq[best], pq[o], ids, metric) for o in others}
    for o, p in zip(others, bench.holm([against[o]["p_value"] for o in others])):
        against[o]["p_holm"] = p
    lines = [bench.verdict(best, o, against[o], metric, p_key="p_holm") for o in others]
    tied = [o for o in others if abs(means[o] - means[best]) < 1e-12]
    beaten = [o for o in others if o not in tied and against[o]["p_holm"] < bench.ALPHA]
    if tied:
        text = f"No single winner on {metric}: {', '.join([best, *tied])} score the same ({means[best]:.3f})."
    elif len(beaten) == len(others):
        text = (f"{best} wins on {metric} ({means[best]:.3f}), significantly ahead of every other variant "
                f"(Holm-adjusted p < {bench.ALPHA}).")
    elif beaten:
        rest = [o for o in others if o not in beaten]
        text = (f"{best} has the highest {metric} ({means[best]:.3f}); it is significantly ahead of "
                f"{', '.join(beaten)}, but not of {', '.join(f'{o} ({means[o]:.3f}, p = {against[o]['p_holm']:.3f})' for o in rest)}"
                f", so this run cannot separate those.")
    else:
        runner_up = others[0]
        text = (f"{best} has the highest {metric} ({means[best]:.3f}, next {runner_up} at {means[runner_up]:.3f}), but no "
                f"difference is significant (smallest Holm-adjusted p = {min(a['p_holm'] for a in against.values()):.3f}): "
                "this run cannot pick a winner.")
    return {"questions": len(ids), "metric": metric, "means": means, "ranking": ranked, "best": best,
            "against": against, "lines": lines, "text": text}


def combined(meta: dict, items: list[Item], runs: dict[str, list[dict]], variants: list[Variant],
             reports: dict[str, str]) -> dict:
    pq = {label: bench.per_question(rs) for label, rs in runs.items()}
    asked = [it for it in items if all(it.id in pq[label] for label in pq)]
    tags = sorted({t for it in asked for t in it.tags})
    return {
        "kind": "bench-combined", "meta": meta,
        "variants": [{"label": v.label, "agent": v.agent, "deep_think": v.deep_think, "fanout": v.fanout,
                      "research_calls": v.research_calls, "python": v.python, "report": reports.get(v.label)}
                     for v in variants],
        "verdicts": {name: group_verdict(pq, ids) for name, ids in groups(asked).items()},
        "summaries": {label: bench.summarize(asked, rs) for label, rs in runs.items()},
        "run_stats": {label: bench._run_stats(rs) for label, rs in runs.items()},
        "correct_by_tag": {tag: {"questions": sum(tag in it.tags for it in asked),
                                 **{label: float(np.mean([pq[label][it.id]["correct"] for it in asked if tag in it.tags]))
                                    for label in pq}} for tag in tags},
    } | ({"in_flight": in_flight(runs)} if (meta.get("concurrency") or 1) > 1 else {})


def in_flight(runs: dict[str, list[dict]]) -> dict[str, float]:
    """{variant: mean number of other requests in flight when its requests were sent}: the load each variant ran
    under, the same for every variant in a fair concurrent run."""
    return {label: float(np.mean([r.get("in_flight", 0) for r in rs])) for label, rs in runs.items() if rs}


def _load_sentence(c: dict) -> str:
    n = c["meta"].get("concurrency") or 1
    if n <= 1:
        return ""
    means = ", ".join(f"{label} {m:.2f}" for label, m in (c.get("in_flight") or {}).items())
    return (f" Requests were sent {n} at a time in that same order (--concurrency {n}), so each ran alongside up to "
            f"{n - 1} other{'s' if n > 2 else ''} and every latency was measured under load; other requests in flight "
            f"when a request was sent, mean per variant: {means}. A variant never runs alongside itself on the same "
            "question, so a heavier variant (fan-out, which uses several llama slots and queues behind the others) "
            "loads the others slightly more than they load it.")


def combined_markdown(c: dict) -> str:
    meta, labels = c["meta"], [v["label"] for v in c["variants"]]
    task = f", search task {meta['search_task']}" if meta.get("search_task") else ""
    lines = [f"# Benchmark of {len(labels)} variants ({meta['set']}, mode {meta['mode']}{task})", ""]
    for name, v in c["verdicts"].items():
        lines.append(f"**Verdict, {name} ({v['questions']} questions):** {v['text']}")
        lines.append("")
    lines += [f"Fairness: one run, interleaved: each repeat shuffles the questions and each question runs every variant "
              f"in a shuffled order (seed {meta['schedule_seed']}), so GPU load, cache warmth and drift reach every "
              "variant alike. Every variant gets the same question text and doc_ids, answer thinking off, and the same "
              "rag container, so the same temperature and seed settings (the chat API takes neither; recorded: "
              f"{meta['environment'].get('rag_settings') or 'rag defaults'}).{_load_sentence(c)}",
              f"Environment: {bench.environment_line(meta['environment'])}{bench.load_note(meta)}.",
              f"{meta['questions']} questions x {meta['repeats']} repeats x {len(labels)} variants = {meta['runs']} runs; "
              f"{meta['started']} to {meta['finished']}{'; INCOMPLETE (interrupted)' if meta.get('incomplete') else ''}.",
              "Scores are model-free (numbers, facts, citations, exact match, F1); no LLM judge takes part. Significance: "
              f"paired permutation test on per-question means, Holm-adjusted, at {bench.ALPHA}.", "",
              "## Variants", "",
              "| variant | agent | deep think | fan-out | research calls | python | correct | correct + cited | f1 | "
              "latency median / p90 (s) | tool errors per run | notices per run | limit hits per run | failed runs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in c["variants"]:
        o, s = c["summaries"][v["label"]]["overall"], c["run_stats"][v["label"]]
        lines.append(f"| {v['label']} | {v['agent']} | {'on' if v['deep_think'] else 'off'} | "
                     f"{'on' if v.get('fanout') else 'off'} | {v.get('research_calls') or 'rag setting'} | "
                     f"{'on' if v.get('python') else 'off'} | "
                     f"{bench._fmt(o.get('correct'))} | "
                     f"{bench._fmt(o.get('correct_cited'))} | {bench._fmt(o.get('f1'))} | {s['latency_median']:.1f} / "
                     f"{s['latency_p90']:.1f} | {s['tool_errors_per_run']:.2f} | {s['notices_per_run']:.2f} | "
                     f"{s['limit_hits_per_run']:.2f} | {s['failed_runs']} |")
    for name, v in c["verdicts"].items():
        lines += ["", f"## Against the best, {name}", "",
                  f"| other variant | {v['best']} | other | {v['best']} - other (95%) | p | Holm p | McNemar p | "
                  f"{v['best']} wins / loses / ties |",
                  "|---|---|---|---|---|---|---|---|"]
        for o, a in v["against"].items():
            lines.append(f"| {o} | {a['a']:.3f} | {a['b']:.3f} | {bench._fmt(a['difference'])} | {a['p_value']:.3f} | "
                         f"{a['p_holm']:.3f} | {a['mcnemar']['p_value']:.3f} | {a['wins']} / {a['losses']} / {a['ties']} |")
        lines += [""] + [f"- {line}" for line in v["lines"]]
    lines += ["", "## Correct per tag", "", "| tag | questions | " + " | ".join(labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    for tag, row in c["correct_by_tag"].items():
        lines.append(f"| {tag} | {row['questions']} | " + " | ".join(f"{row[label]:.3f}" for label in labels) + " |")
    lines += ["", "Per-variant reports: " + ", ".join(f"{v['label']}: {Path(v['report']).name}" for v in c["variants"]
                                                       if v.get("report"))]
    return "\n".join(lines) + "\n"


def write_combined(c: dict, reports: Path) -> tuple[Path, Path]:
    base = reports / f"bench-combined-{c['meta']['stamp']}"
    js, md = base.with_suffix(".json"), base.with_suffix(".md")
    js.write_text(json.dumps(c, indent=2, ensure_ascii=True), encoding="utf-8")
    md.write_text(combined_markdown(c), encoding="utf-8")
    return js, md
