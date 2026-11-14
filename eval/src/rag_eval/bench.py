"""Benchmark runs: a fixed question set asked through the live gateway, scored without a judge model,
repeated, summarized per tag with bootstrap intervals; and a paired comparison of two runs.

A question (one JSONL line) names the files to select, the documents its answer should cite, its tags,
and the checks that decide correctness: numbers (with tolerance), required facts, forbidden patterns, and
gold answers for exact match and F1 (docs/ARCHITECTURE.md, section 9, has the format).

Repeats are averaged per question first, and intervals resample questions, so repeats measure how stable
an answer is without pretending to be independent questions.
"""

import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_common.contracts import MAX_SELECTED_DOCS
from rag_eval.client import Gateway, Turn
from rag_eval.dispatch import dispatch
from rag_eval.metrics import bootstrap_ci, paired_bootstrap
from rag_eval.scoring import (citation_check, citations_in, contains_answer, exact_match, facts_match, numbers_match,
                              token_f1)

TAGS = ("single-document", "cross-document", "table-sql", "image", "numeric", "needs-deep", "hard-case",
        "hard", "multi-step", "distractors", "false-premise",  # the hard set
        "global", "enumeration", "coverage", "counting",  # the global set (whole-document questions)
        "name-forms",  # the forms set (names written in more than one form)
        "totals")  # the totals set (a total or subtotal row that changes the answer if included or left out)
# Per-question quality metrics, in report order. correct: the item's own checks (numbers and facts) for the
# local set, `contains` for public sets; correct_cited: correct and the citations check passed.
QUALITY = ("correct", "correct_cited", "citations_ok", "cites_all_expected", "numbers", "facts", "em", "f1",
           "contains")
OPS = ("latency_s", "tool_calls", "tool_errors", "duplicate_calls", "errors", "notices", "limit_hits",
       "completion_tokens")
_NOT_TOOLS = ("plan", "verify")  # deep-search steps that are not tool calls
# Notices that say a limit was reached (rag/notices: a cut reply or result, a full budget or prompt); new codes
# that end in _cut or _full, or name a budget, count too. An answer that stopped at its length limit is one more.
LIMIT_CODES = frozenset({"answer_cut", "rewrite_cut", "research_step_cut", "sql_rows_cut", "filter_rows_cut",
                         "research_budget", "research_context_full", "sources_dropped"})


def limit_hits(rec: dict) -> int:
    codes = [str(c) for c in rec.get("notices") or []]
    hits = sum(1 for c in codes if c in LIMIT_CODES or c.endswith(("_cut", "_full")) or "budget" in c)
    return hits + (rec.get("finish_reason") == "length")


@dataclass
class Item:
    id: str
    question: str
    docs: list[str]  # file names selected for the question (at most 10)
    tags: list[str] = field(default_factory=list)
    expected_docs: list[str] = field(default_factory=list)  # citations should point here; default: docs
    reference: str = ""  # the expected answer in words (for people; not scored)
    answers: list[str] = field(default_factory=list)  # gold answers and aliases for EM, F1 and contains
    numbers: list[dict] = field(default_factory=list)
    facts: list = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    correct_by: str = "checks"  # "checks": numbers and facts; "contains": a gold answer in the reply
    origin: str = ""
    note: str = ""

    def __post_init__(self):
        self.expected_docs = self.expected_docs or list(self.docs)


def parse_items(lines: Iterable[str], where: str = "set") -> list[Item]:
    """Items from JSONL lines; ValueError naming the line for anything malformed."""
    items, seen = [], set()
    known = set(Item.__dataclass_fields__)
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            extra = set(raw) - known
            if extra:
                raise ValueError(f"unknown fields {sorted(extra)}")
            item = Item(**raw)
            _check(item)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{where}:{n}: {exc}") from None
        if item.id in seen:
            raise ValueError(f"{where}:{n}: duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    return items


def _check(item: Item) -> None:
    if not item.id or not item.question.strip():
        raise ValueError("id and question are required")
    if not 1 <= len(item.docs) <= MAX_SELECTED_DOCS:
        raise ValueError(f"{item.id}: select 1 to {MAX_SELECTED_DOCS} documents")
    if not set(item.expected_docs) <= set(item.docs):
        raise ValueError(f"{item.id}: expected_docs must be among docs")
    if item.correct_by not in ("checks", "contains"):
        raise ValueError(f"{item.id}: correct_by is checks or contains")
    if item.correct_by == "checks" and not (item.numbers or item.facts):
        raise ValueError(f"{item.id}: give numbers or facts to check")
    if item.correct_by == "contains" and not item.answers:
        raise ValueError(f"{item.id}: give answers to match")
    for spec in item.numbers:  # fail at load time, not mid-run
        numbers_match("", [spec])
    facts_match("", item.facts, item.forbidden)


def load_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as f:
        return parse_items(f, str(path))


def select(items: list[Item], tags: Sequence[str] = (), skip_tags: Sequence[str] = (), ids: Sequence[str] = (),
           limit: int | None = None) -> list[Item]:
    """Items with any of `tags` (all when empty), none of `skip_tags`, and one of `ids` (all when empty)."""
    want, skip = {t.lower() for t in tags}, {t.lower() for t in skip_tags}
    out = [it for it in items
           if (not want or want & {t.lower() for t in it.tags}) and not skip & {t.lower() for t in it.tags}
           and (not ids or it.id in ids)]
    return out[:limit] if limit else out


# --- one run ----------------------------------------------------------------------------------------


def score(item: Item, answer: str, sources: list[dict], expected_ids: Iterable[str]) -> tuple[dict, list[str]]:
    """({metric: value}, why it is not correct). Metrics an item does not define are left out."""
    s: dict[str, float] = {}
    why: list[str] = []
    if item.answers:
        s["em"] = exact_match(answer, item.answers)
        s["f1"] = token_f1(answer, item.answers)
        s["contains"] = contains_answer(answer, item.answers)
    if item.numbers:
        ok, missing = numbers_match(answer, item.numbers)
        s["numbers"] = float(ok)
        why += [f"number missing: {m}" for m in missing]
    if item.facts or item.forbidden:
        ok, failed = facts_match(answer, item.facts, item.forbidden)
        s["facts"] = float(ok)
        why += failed
    cites = citation_check(answer, sources, expected_ids)
    s["citations_ok"] = float(cites["ok"])
    s["cites_all_expected"] = float(cites["all_expected"])
    if item.correct_by == "contains":
        correct = bool(s["contains"])
        if not correct:
            why.append("no gold answer in the reply")
    else:
        correct = all(s[k] for k in ("numbers", "facts") if k in s)
    s["correct"] = float(correct)
    s["correct_cited"] = float(correct and cites["ok"])
    if not cites["ok"]:
        why.append("citations: " + ("none" if not cites["cited"] else
                                    f"unresolved {cites['unresolved']}" if cites["unresolved"] else
                                    "outside the expected documents" if cites["outside"] else
                                    "no expected document cited"))
    return s, why


def _steps(reply) -> tuple[list[str], int, int]:
    """(tool names in order, tool errors, repeated identical calls) from the deep-search step events."""
    tools, errors, dupes, seen = [], 0, 0, set()
    for step in reply.steps:
        if step.get("tool") in _NOT_TOOLS:
            continue
        tools.append(str(step.get("tool")))
        errors += bool(step.get("error"))
        key = (step.get("tool"), json.dumps(step.get("args"), sort_keys=True, default=str))
        dupes += key in seen
        seen.add(key)
    return tools, errors, dupes


def record(item: Item, repeat: int, turn: Turn, doc_ids: dict[str, str]) -> dict:
    """The run record: the answer, its scores, and what the run cost."""
    expected = [doc_ids[name] for name in item.expected_docs]
    base = {"id": item.id, "repeat": repeat, "status": turn.status, "latency_s": round(turn.latency_s, 3),
            "error": turn.error}
    reply = turn.reply
    if reply is None or turn.error:  # a failed or cut answer is never scored as a complete one
        scores = {"correct": 0.0, "correct_cited": 0.0, "citations_ok": 0.0, "cites_all_expected": 0.0}
        if item.answers:
            scores.update(em=0.0, f1=0.0, contains=0.0)
        if item.numbers:
            scores["numbers"] = 0.0
        if item.facts or item.forbidden:
            scores["facts"] = 0.0
        tools, errors, dupes = _steps(reply) if reply else ([], 0, 0)
        return base | {"answer": reply.answer if reply else "", "scores": scores, "why": [f"error: {turn.error}"],
                       "route": reply.route if reply else None, "tools": tools, "tool_calls": len(tools),
                       "tool_errors": errors, "duplicate_calls": dupes,
                       "notices": [n.get("code") for n in reply.notices] if reply else [],
                       "finish_reason": None, "usage": None, "timings": reply.timings if reply else {},
                       "citations": [], "invalid_citations": [], "cited_docs": []}
    scores, why = score(item, reply.answer, reply.sources, expected)
    done = next((data for name, data in reversed(reply.events) if name == "done"), {})
    tools, errors, dupes = _steps(reply)
    name_of = {s.get("n"): s.get("doc_name") for s in reply.sources}
    cited = citations_in(reply.answer)  # what the scores checked; done's list holds only the valid ones
    return base | {
        "answer": reply.answer, "scores": scores, "why": why, "route": reply.route,
        "tools": tools, "tool_calls": int(reply.timings.get("tool_calls", len(tools))), "tool_errors": errors,
        "duplicate_calls": dupes, "notices": [n.get("code") for n in reply.notices],
        "finish_reason": done.get("finish_reason"), "usage": done.get("usage"), "timings": reply.timings,
        "citations": cited, "invalid_citations": reply.invalid_citations,
        "cited_docs": sorted({name_of[n] for n in cited if n in name_of}),
    }


def run(gateway: Gateway, items: list[Item], doc_ids: dict[str, str], mode: str, think: bool, repeats: int,
        agent: str | None, sink: Callable[[dict], None] | None = None, progress: Callable[[str], None] = print,
        deep_think: bool = False, clock: Callable[[], float] = time.monotonic, concurrency: int = 1) -> list[dict]:
    """Ask every item `repeats` times (all items, then again), stateless; returns the run records in the order
    they finished. After each run, `progress` gets one line: where the run is, this answer, the running pass rate
    and the time left. With concurrency above 1, up to that many requests are in flight (rag_eval.dispatch) and
    each record also carries `order`, its dispatch position, and `in_flight`, the other requests running when it
    was sent."""
    runs: list[dict] = []
    total, passed, started = len(items) * repeats, 0, clock()
    jobs = [(r, i, it) for r in range(1, repeats + 1) for i, it in enumerate(items, 1)]

    def ask(job: tuple[int, int, Item]) -> Turn:
        it = job[2]
        return gateway.ask(it.question, [doc_ids[name] for name in it.docs], mode=mode, think=think, agent=agent,
                           deep_think=deep_think)

    def done(k: int, job: tuple[int, int, Item], turn: Turn, others: int) -> None:
        nonlocal passed
        r, i, it = job
        rec = record(it, r, turn, doc_ids)
        if concurrency > 1:
            rec |= {"order": k, "in_flight": others}
        runs.append(rec)
        if sink:
            sink(rec)
        passed += rec["scores"]["correct"] == 1.0
        progress(progress_line(rec, len(runs), total, r, repeats, i, len(items), passed, clock() - started))

    dispatch(jobs, ask, done, concurrency)
    return runs


def _duration(seconds: float) -> str:
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def progress_line(rec: dict, done: int, total: int, repeat: int, repeats: int, index: int, n: int, passed: int,
                  elapsed: float) -> str:
    """One short line per run, for watching a run that takes hours. The time left is the time per finished run
    so far times the runs to go: finished runs count every slot, so with several requests in flight it already
    reflects the concurrency."""
    verdict = "PASS" if rec["scores"]["correct"] == 1.0 else "ERROR" if rec["error"] else "FAIL"
    left = elapsed / done * (total - done)
    line = (f"[{done}/{total}] repeat {repeat}/{repeats}, question {index}/{n} {rec['id']}: {verdict} "
            f"{rec['latency_s']:.0f}s, {rec['tool_calls']} tools, {rec['tool_errors']} tool errors | "
            f"pass rate {passed}/{done} ({passed / done:.0%}) | {_duration(elapsed)} elapsed, ~{_duration(left)} left")
    return line + (f" | {str(rec['error'])[:80]}" if rec["error"] else "")


# --- summaries ----------------------------------------------------------------------------------------


def _ops(rec: dict) -> dict[str, float]:
    usage = rec.get("usage") or {}
    out = {"latency_s": rec["latency_s"], "tool_calls": rec["tool_calls"], "tool_errors": rec["tool_errors"],
           "duplicate_calls": rec["duplicate_calls"], "errors": float(bool(rec["error"])),
           "notices": float(len(rec.get("notices") or [])), "limit_hits": float(limit_hits(rec))}
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        out["completion_tokens"] = float(usage["completion_tokens"])
    return out


def per_question(runs: list[dict]) -> dict[str, dict[str, float]]:
    """{id: {metric: mean over that question's repeats}}, quality and cost metrics alike, plus
    all_correct (every repeat correct) and majority_correct (more than half)."""
    grouped: dict[str, list[dict]] = {}
    for rec in runs:
        grouped.setdefault(rec["id"], []).append(rec)
    out = {}
    for qid, recs in grouped.items():
        values: dict[str, list[float]] = {}
        for rec in recs:
            for k, v in (rec["scores"] | _ops(rec)).items():
                values.setdefault(k, []).append(float(v))
        means = {k: sum(v) / len(v) for k, v in values.items()}
        correct = values["correct"]
        means["all_correct"] = float(all(correct))
        means["majority_correct"] = float(sum(correct) > len(correct) / 2)
        means["repeats"] = float(len(recs))
        out[qid] = means
    return out


def _intervals(rows: list[dict[str, float]], metrics: Sequence[str]) -> dict[str, list[float]]:
    return {m: list(bootstrap_ci([r[m] for r in rows if m in r])) for m in metrics if any(m in r for r in rows)}


def summarize(items: list[Item], runs: list[dict]) -> dict:
    """Overall and per-tag means with 95% bootstrap intervals over questions, plus latency percentiles."""
    pq = per_question(runs)
    metrics = (*QUALITY, "all_correct", "majority_correct", *OPS)
    rows = [pq[it.id] for it in items if it.id in pq]
    by_tag = {}
    for tag in sorted({t for it in items for t in it.tags}):
        tagged = [pq[it.id] for it in items if tag in it.tags and it.id in pq]
        if tagged:
            by_tag[tag] = {"questions": len(tagged), **_intervals(tagged, metrics)}
    latencies = sorted(r["latency_s"] for r in runs)
    return {
        "questions": len(rows), "runs": len(runs),
        "overall": _intervals(rows, metrics),
        "by_tag": by_tag,
        "latency": {"median": statistics.median(latencies), "p90": float(np.quantile(latencies, 0.9)),
                    "max": latencies[-1]} if latencies else {},
        "tool_names": dict(Counter(t for r in runs for t in r.get("tools") or []).most_common()),
        "notice_codes": dict(Counter(c for r in runs for c in r.get("notices") or []).most_common()),
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in runs).most_common()),
    }


def set_digest(items: list[Item]) -> str:
    """A fingerprint of the questions and their checks, so compare can tell two different sets apart."""
    blob = json.dumps([asdict(it) for it in items], sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-.")[:60] or "default"


def _fmt(ci: Sequence[float] | None, digits: int = 3) -> str:
    if not ci or any(math.isnan(x) for x in ci):
        return "-"
    mean, lo, hi = ci
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def markdown(report: dict, items: list[Item]) -> str:
    s, meta = report["summary"], report["meta"]
    lines = [f"# Benchmark: {meta['variant']} ({meta['set']}, mode {meta['mode']}{', think' if meta['think'] else ''}"
             f"{', deep think' if meta.get('deep_think') else ''})", "",
             f"{s['questions']} questions x {meta['repeats']} repeats = {s['runs']} runs; agent field: "
             f"{meta['agent'] or '(not sent)'}; deep_think: {'sent' if meta.get('deep_think') else 'not sent'}; "
             f"{'fanout: sent; ' if meta.get('fanout') else ''}"
             f"{'research_calls: ' + str(meta['research_calls']) + '; ' if meta.get('research_calls') else ''}"
             f"{'python: sent; ' if meta.get('python') else ''}"
             f"set {meta['set_digest']}; {meta['started']} to {meta['finished']}"
             f"{'; INCOMPLETE (interrupted)' if meta.get('incomplete') else ''}.",
             f"Environment: {environment_line(meta.get('environment') or {})}{load_note(meta)}.",
             "Means over questions (repeats averaged first) with 95% bootstrap intervals. Scores are model-free "
             "(numbers, facts, citations, exact match, F1); no LLM judge takes part.", "",
             "| metric | value |", "|---|---|"]
    for m in (*QUALITY, "all_correct", "majority_correct"):
        if m in s["overall"]:
            lines.append(f"| {m} | {_fmt(s['overall'][m])} |")
    lines += ["", "| cost per run | value |", "|---|---|"]
    for m in OPS:
        if m in s["overall"]:
            lines.append(f"| {m} | {_fmt(s['overall'][m], 2)} |")
    if s["latency"]:
        lines.append(f"| latency median / p90 / max (s) | {s['latency']['median']:.1f} / {s['latency']['p90']:.1f} / "
                     f"{s['latency']['max']:.1f} |")
    lines += ["", "## Per tag", "", "| tag | questions | correct | correct + cited | all repeats correct | f1 | latency (s) | tool errors |",
              "|---|---|---|---|---|---|---|---|"]
    for tag, t in s["by_tag"].items():
        lines.append(f"| {tag} | {t['questions']} | {_fmt(t.get('correct'))} | {_fmt(t.get('correct_cited'))} | "
                     f"{_fmt(t.get('all_correct'))} | {_fmt(t.get('f1'))} | {_fmt(t.get('latency_s'), 1)} | "
                     f"{_fmt(t.get('tool_errors'), 2)} |")
    lines += ["", f"Notices: {s['notice_codes'] or 'none'}. Finish reasons: {s['finish_reasons']}.", "",
              "## Per question", "", "| id | tags | correct | cited | f1 | latency (s) | first problem |", "|---|---|---|---|---|---|---|"]
    pq = per_question(report["runs"])
    for it in items:
        if it.id not in pq:
            continue
        q = pq[it.id]
        reps = int(q["repeats"])
        problem = next((w for rec in report["runs"] if rec["id"] == it.id for w in rec["why"]), "")
        f1 = f"{q['f1']:.2f}" if "f1" in q else "-"
        lines.append(f"| {it.id} | {', '.join(it.tags)} | {round(q['correct'] * reps)}/{reps} | "
                     f"{round(q['citations_ok'] * reps)}/{reps} | {f1} | {q['latency_s']:.1f} | "
                     f"{problem.replace('|', '/')[:90]} |")
    return "\n".join(lines) + "\n"


def build_report(meta: dict, items: list[Item], runs: list[dict]) -> dict:
    return {"kind": "bench", "meta": meta,
            "items": [{"id": it.id, "tags": it.tags, "question": it.question, "docs": it.docs} for it in items],
            "runs": runs, "summary": summarize(items, runs)}


def write_report(report: dict, items: list[Item], reports: Path) -> tuple[Path, Path]:
    reports.mkdir(parents=True, exist_ok=True)
    stamp = report["meta"]["stamp"]
    base = reports / f"bench-{safe_label(report['meta']['variant'])}-{stamp}"
    js, md = base.with_suffix(".json"), base.with_suffix(".md")
    js.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    md.write_text(markdown(report, items), encoding="utf-8")
    return js, md


def now() -> datetime:
    return datetime.now(UTC)


# --- paired comparison -------------------------------------------------------------------------------

ALPHA = 0.05


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def permutation_p(differences: Sequence[float], resamples: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired permutation (sign-flip) test of a zero mean difference, on per-question differences
    (each question's mean over repeats, so repeats count without posing as more questions)."""
    d = np.asarray(differences, dtype=float)
    if d.size == 0 or not d.any():
        return 1.0
    observed = abs(d.mean())
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(resamples, d.size))
    hits = int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).sum())
    return (hits + 1) / (resamples + 1)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order (several comparisons from one run)."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def paired(pa: dict[str, dict], pb: dict[str, dict], ids: Sequence[str], metric: str = "correct") -> dict:
    """Paired statistics of one metric on the given questions: wins, losses, ties (on the mean over repeats),
    the two means, a paired bootstrap interval of A - B, the permutation p-value, and the exact McNemar test on
    the questions only one side gets right in most repeats."""
    ids = [q for q in ids if metric in pa.get(q, {}) and metric in pb.get(q, {})]
    if not ids:
        return {"questions": 0}
    xs, ys = [pa[q][metric] for q in ids], [pb[q][metric] for q in ids]
    wins = [q for q, x, y in zip(ids, xs, ys) if x > y + 1e-12]
    losses = [q for q, x, y in zip(ids, xs, ys) if y > x + 1e-12]
    a_only = sum(1 for q in ids if pa[q]["majority_correct"] and not pb[q]["majority_correct"])
    b_only = sum(1 for q in ids if pb[q]["majority_correct"] and not pa[q]["majority_correct"])
    return {"questions": len(ids), "wins": len(wins), "losses": len(losses), "ties": len(ids) - len(wins) - len(losses),
            "win_ids": wins, "loss_ids": losses, "a": float(np.mean(xs)), "b": float(np.mean(ys)),
            "difference": list(paired_bootstrap(xs, ys)), "p_value": permutation_p(np.subtract(xs, ys)),
            "mcnemar": {"a_only_majority_correct": a_only, "b_only_majority_correct": b_only,
                        "p_value": mcnemar_exact(a_only, b_only)}}


def verdict(name_a: str, name_b: str, stats: dict, metric: str = "correct", p_key: str = "p_value") -> str:
    """One plain sentence: who wins on the metric, by how much, and whether that is significant."""
    if not stats.get("questions"):
        return f"No shared questions: {name_a} and {name_b} cannot be compared."
    a, b, (_, lo, hi), p = stats["a"], stats["b"], stats["difference"], stats[p_key]
    if abs(a - b) < 1e-12:
        return f"{name_a} and {name_b} score the same on {metric} ({a:.3f}): no winner."
    win, lose, wv, lv, cl, ch = (name_a, name_b, a, b, lo, hi) if a > b else (name_b, name_a, b, a, -hi, -lo)
    gap = f"{wv:.3f} vs {lv:.3f}, difference {wv - lv:.3f}, 95% CI [{cl:.3f}, {ch:.3f}]"
    if p < ALPHA:
        return f"{win} wins on {metric} over {lose} ({gap}); significant (p = {p:.3f})."
    return (f"{win} scores higher on {metric} than {lose} ({gap}), but the difference is not significant "
            f"(p = {p:.3f}): this run cannot tell them apart.")


def _run_stats(runs: list[dict]) -> dict:
    lat = sorted(r["latency_s"] for r in runs)
    return {"runs": len(runs), "latency_median": float(np.median(lat)) if lat else float("nan"),
            "latency_p90": float(np.quantile(lat, 0.9)) if lat else float("nan"),
            "tool_errors_per_run": float(np.mean([r["tool_errors"] for r in runs])) if runs else float("nan"),
            "notices_per_run": float(np.mean([len(r.get("notices") or []) for r in runs])) if runs else float("nan"),
            "limit_hits_per_run": float(np.mean([limit_hits(r) for r in runs])) if runs else float("nan"),
            "failed_runs": int(sum(bool(r["error"]) for r in runs)),
            "notice_codes": dict(Counter(c for r in runs for c in r.get("notices") or []).most_common())}


def compare(a: dict, b: dict, metric: str = "correct") -> dict:
    """Paired comparison of two benchmark reports on the questions they share, overall and per tag, with the
    cost of each side (latency median and p90, tool errors, notices, limit hits, failed runs) and a verdict."""
    pa, pb = per_question(a["runs"]), per_question(b["runs"])
    shared = [q for q in pa if q in pb]
    if not shared:
        raise SystemExit("the two reports share no question ids")
    if any(metric not in pa[q] or metric not in pb[q] for q in shared):
        raise SystemExit(f"metric {metric!r} is not in both reports for every shared question")
    overall = paired(pa, pb, shared, metric)
    tags: dict[str, list[str]] = {}
    for item in a.get("items", []):
        if item["id"] in pb:
            for tag in item.get("tags", []):
                tags.setdefault(tag, []).append(item["id"])
    notes = []
    for key in ("set_digest", "mode", "think", "deep_think", "fanout", "research_calls", "python", "repeats"):
        if a["meta"].get(key) != b["meta"].get(key):
            notes.append(f"{key} differs: {a['meta'].get(key)} vs {b['meta'].get(key)}")
    load_a, load_b = a["meta"].get("concurrency") or 1, b["meta"].get("concurrency") or 1
    if load_a != load_b:
        notes.append(f"concurrency differs: {load_a} vs {load_b}, so the latencies were measured under different load")
    env_a, env_b = a["meta"].get("environment") or {}, b["meta"].get("environment") or {}
    for key in ("rag_image", "model", "llm_ctx"):
        if env_a.get(key) and env_b.get(key) and env_a[key] != env_b[key]:
            notes.append(f"{key} differs: {env_a[key]} vs {env_b[key]}")
    only = sorted(set(pa) ^ set(pb))
    if only:
        notes.append(f"{len(only)} questions are in only one report and are left out")

    def cost(m: str) -> dict:
        xs = [pa[q][m] for q in shared if m in pa[q] and m in pb[q]]
        ys = [pb[q][m] for q in shared if m in pa[q] and m in pb[q]]
        return {"a": float(np.mean(xs)), "b": float(np.mean(ys)), "difference": list(paired_bootstrap(xs, ys))} if xs else {}

    keep = set(shared)
    return {
        "kind": "compare", "metric": metric, "a": a["meta"]["variant"], "b": b["meta"]["variant"],
        "settings": {side: {k: rep["meta"].get(k) for k in ("agent", "mode", "think", "deep_think", "fanout",
                                                              "research_calls", "python", "repeats")}
                     for side, rep in (("a", a), ("b", b))},
        "environment": {"a": env_a, "b": env_b},
        **{k: overall[k] for k in ("questions", "wins", "losses", "ties", "win_ids", "loss_ids")},
        "quality": {k: overall[k] for k in ("a", "b", "difference", "p_value")},
        "mcnemar": overall["mcnemar"], "sign_test_p": mcnemar_exact(overall["wins"], overall["losses"]),
        "by_tag": {tag: paired(pa, pb, ids, metric) for tag, ids in sorted(tags.items())},
        "cost": {m: cost(m) for m in ("latency_s", "tool_calls", "tool_errors", "limit_hits", "notices", "errors")},
        "runs": {"a": _run_stats([r for r in a["runs"] if r["id"] in keep]),
                 "b": _run_stats([r for r in b["runs"] if r["id"] in keep])},
        "verdict": verdict(f"A ({a['meta']['variant']})", f"B ({b['meta']['variant']})", overall, metric),
        "notes": notes,
    }


def _settings(s: dict) -> str:
    return (f"agent {s.get('agent') or '(not sent)'}, mode {s.get('mode')}, think {'on' if s.get('think') else 'off'}, "
            f"deep think {'on' if s.get('deep_think') else 'off'}, fan-out {'on' if s.get('fanout') else 'off'}, "
            f"research calls {s.get('research_calls') or 'rag setting'}, python {'on' if s.get('python') else 'off'}, "
            f"{s.get('repeats')} repeats")


def environment_line(env: dict) -> str:
    if not env:
        return "environment not recorded"
    image = env.get("rag_image") or "unknown"
    return (f"rag image {image[:19] if image.startswith('sha256:') else image}, git commit {env.get('git_commit') or 'unknown'}"
            f"{', app commit ' + env['app_commit'] if env.get('app_commit') else ''}, model {env.get('model') or 'unknown'}, "
            f"LLM_CTX {env.get('llm_ctx') or 'unknown'}")


def load_note(meta: dict) -> str:
    """The reports' note on concurrent requests (--concurrency); empty for a run that sent one at a time."""
    n = meta.get("concurrency") or 1
    if n <= 1:
        return ""
    return (f"; {n} requests in flight at a time (--concurrency {n}), so every latency was measured under load, and "
            "fan-out variants, which use several llama slots themselves, queue behind the other requests in flight")


def run_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = ["| side | runs | latency median / p90 (s) | tool errors per run | notices per run | limit hits per run | "
             "failed runs |", "|---|---|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s['runs']} | {s['latency_median']:.1f} / {s['latency_p90']:.1f} | "
                     f"{s['tool_errors_per_run']:.2f} | {s['notices_per_run']:.2f} | {s['limit_hits_per_run']:.2f} | "
                     f"{s['failed_runs']} |")
    return lines


def compare_markdown(c: dict) -> str:
    q, mc = c["quality"], c["mcnemar"]
    lines = [f"# {c['a']} (A) vs {c['b']} (B), metric {c['metric']}", "",
             f"**Verdict:** {c['verdict']}", "",
             f"{c['questions']} shared questions. A wins {c['wins']}, loses {c['losses']}, ties {c['ties']} "
             f"(per-question mean over repeats; sign test p = {c['sign_test_p']:.3f}).",
             f"A: {_settings(c['settings']['a'])}. B: {_settings(c['settings']['b'])}.",
             f"A: {environment_line(c['environment']['a'])}. B: {environment_line(c['environment']['b'])}.", "",
             f"- {c['metric']}: A {q['a']:.3f}, B {q['b']:.3f}; A - B = {_fmt(q['difference'])} (paired bootstrap, 95%); "
             f"paired permutation p = {q['p_value']:.3f}.",
             f"- Majority-correct questions only A gets: {mc['a_only_majority_correct']}; only B: "
             f"{mc['b_only_majority_correct']}; exact McNemar p = {mc['p_value']:.3f}.", "",
             "## Per tag (exploratory: not corrected for the number of tags)", "",
             "| tag | questions | A wins | B wins | ties | A | B | A - B (95%) | p |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, t in c["by_tag"].items():
        if t.get("questions"):
            lines.append(f"| {tag} | {t['questions']} | {t['wins']} | {t['losses']} | {t['ties']} | {t['a']:.3f} | "
                         f"{t['b']:.3f} | {_fmt(t['difference'])} | {t['p_value']:.3f} |")
    lines += ["", "## Cost", ""] + run_table([("A", c["runs"]["a"]), ("B", c["runs"]["b"])])
    lines += ["", "| per run, paired over questions | A mean | B mean | A - B (95%) |", "|---|---|---|---|"]
    for m, d in c["cost"].items():
        if d:
            lines.append(f"| {m} | {d['a']:.2f} | {d['b']:.2f} | {_fmt(d['difference'], 2)} |")
    codes = {side: c["runs"][side]["notice_codes"] for side in ("a", "b")}
    lines += ["", f"Notices: A {codes['a'] or 'none'}; B {codes['b'] or 'none'}."]
    if c["win_ids"] or c["loss_ids"]:
        lines += ["", f"A better on: {', '.join(c['win_ids']) or 'none'}", f"B better on: {', '.join(c['loss_ids']) or 'none'}"]
    if c["notes"]:
        lines += ["", "Notes: " + "; ".join(c["notes"]) + "."]
    lines += ["", "Scores are model-free (numbers, facts, citations, exact match, F1); no LLM judge takes part. "
              f"Significant means a paired permutation p below {ALPHA} on per-question means."]
    return "\n".join(lines) + "\n"
