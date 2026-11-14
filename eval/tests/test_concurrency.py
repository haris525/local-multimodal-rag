"""--concurrency: requests in flight at a time, with fakes only. At 1 the run loops are exactly the sequential
loops they replaced (same records, progress lines and stream order); at 2 every run happens once, at most two at a
time, in the schedule's dispatch order, with complete records and correct counts; a failed request is recorded as
an ERROR run, and an exception in a worker reaches the caller after the requests in flight finished."""

import json
import threading
import time
from collections import Counter

import pytest
from test_bench_cli import _local_set, gateway  # noqa: F401  (the stub gateway fixture)

from rag_common.chat_events import ChatReply
from rag_eval import bench, cli, variants
from rag_eval.bench import Item
from rag_eval.client import Turn
from rag_eval.dispatch import dispatch

ITEMS = [Item(id=f"q{i}", question=f"Q{i}?", docs=["d"], facts=["right"]) for i in range(5)]
DOC_IDS = {"d": "doc-1"}
VS = variants.parse_variants("a,b+think,c+fanout")


class FakeGateway:
    """Answers right for every agent but "b" on odd questions; `delay` seconds per call (by agent), `fail` the
    (question, agent) pairs that come back as a failed request. Tracks the requests in flight."""

    def __init__(self, delay: dict | None = None, fail: set | None = None, raise_on: set | None = None):
        self.delay, self.fail, self.raise_on = delay or {}, fail or set(), raise_on or set()
        self.lock = threading.Lock()
        self.now = self.peak = 0
        self.calls: list[tuple] = []

    def ask(self, question, doc_ids, mode="deep", think=False, agent=None, deep_think=False, fanout=False,
            research_calls=None, python=None) -> Turn:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            self.calls.append((question, agent, deep_think, fanout))
        try:
            time.sleep(self.delay.get(agent, 0.0))
            if (question, agent) in self.raise_on:
                raise RuntimeError("worker bug")
            if (question, agent) in self.fail:
                return Turn(None, 0, "ReadTimeout: timed out", 900.0)
            wrong = agent == "b" and int(question[1:-1]) % 2 == 1
            reply = ChatReply(events=[("done", {"finish_reason": "stop"})], terminal="done", route=mode,
                              answer="wrong" if wrong else "right [1]",
                              sources=[{"n": 1, "doc_id": doc_ids[0], "doc_name": "d", "text": "..."}])
            return Turn(reply, 200, None, 10.0 + len(question))
        finally:
            with self.lock:
                self.now -= 1


def ticking(step: float = 10.0):
    t = [0.0]

    def clock() -> float:
        t[0] += step
        return t[0]
    return clock


# --- the sequential loops as they were before --concurrency, kept here as the reference -----------------------


def old_variants_run(gateway, items, doc_ids, mode, repeats, vs, seed, sink, progress, clock):
    plan = variants.schedule(items, vs, repeats, seed)
    runs = {v.label: [] for v in vs}
    passed = dict.fromkeys(runs, 0)
    started = clock()
    for k, (r, i, item, v) in enumerate(plan, 1):
        turn = gateway.ask(item.question, [doc_ids[name] for name in item.docs], mode=mode, think=False,
                           agent=v.agent, deep_think=v.deep_think, fanout=v.fanout, research_calls=v.research_calls)
        rec = bench.record(item, r, turn, doc_ids) | {"variant": v.label, "order": k}
        runs[v.label].append(rec)
        passed[v.label] += rec["scores"]["correct"] == 1.0
        if sink:
            sink(rec)
        progress(variants.progress_line(rec, k, len(plan), r, repeats, i, len(items), runs, passed, clock() - started))
    return runs


def old_bench_run(gateway, items, doc_ids, mode, think, repeats, agent, sink, progress, deep_think, clock):
    runs = []
    total, passed, started = len(items) * repeats, 0, clock()
    for r in range(1, repeats + 1):
        for i, it in enumerate(items, 1):
            turn = gateway.ask(it.question, [doc_ids[name] for name in it.docs], mode=mode, think=think, agent=agent,
                               deep_think=deep_think)
            rec = bench.record(it, r, turn, doc_ids)
            runs.append(rec)
            if sink:
                sink(rec)
            passed += rec["scores"]["correct"] == 1.0
            progress(bench.progress_line(rec, len(runs), total, r, repeats, i, len(items), passed, clock() - started))
    return runs


def test_concurrency_one_is_exactly_the_sequential_loop():
    fail = {("Q3?", "b")}
    for kwargs in ({}, {"concurrency": 1}):
        want_sink, want_lines, got_sink, got_lines = [], [], [], []
        want = old_variants_run(FakeGateway(fail=fail), ITEMS, DOC_IDS, "deep", 2, VS, 7, want_sink.append,
                                want_lines.append, ticking())
        gw = FakeGateway(fail=fail)
        got = variants.run(gw, ITEMS, DOC_IDS, "deep", 2, VS, 7, sink=got_sink.append, progress=got_lines.append,
                           clock=ticking(), **kwargs)
        assert got == want and got_sink == want_sink and got_lines == want_lines and gw.peak == 1
        assert all("in_flight" not in rec for rec in got_sink)

        want_sink, want_lines, got_sink, got_lines = [], [], [], []
        want = old_bench_run(FakeGateway(fail=fail), ITEMS, DOC_IDS, "deep", False, 3, "b", want_sink.append,
                             want_lines.append, True, ticking())
        got = bench.run(FakeGateway(fail=fail), ITEMS, DOC_IDS, "deep", False, 3, "b", sink=got_sink.append,
                        progress=got_lines.append, deep_think=True, clock=ticking(), **kwargs)
        assert got == want and got_sink == want_sink and got_lines == want_lines
        assert all("order" not in rec and "in_flight" not in rec for rec in got)


def test_concurrency_two_runs_everything_once_in_dispatch_order_with_correct_records_and_counts():
    gw = FakeGateway(delay={"a": 0.03, "b": 0.0, "c": 0.01})  # a is slow, so records finish out of order
    streamed, lines = [], []
    runs = variants.run(gw, ITEMS, DOC_IDS, "deep", 2, VS, 7, sink=streamed.append, progress=lines.append,
                        concurrency=2)
    plan = variants.schedule(ITEMS, VS, 2, 7)
    assert gw.peak == 2 and len(streamed) == len(plan) == 5 * 2 * 3
    assert sorted(rec["order"] for rec in streamed) == list(range(1, len(plan) + 1))
    assert [rec["order"] for rec in streamed] != sorted(rec["order"] for rec in streamed)
    for rec in streamed:  # each record belongs to the job it was dispatched for
        r, _, item, v = plan[rec["order"] - 1]
        assert (rec["repeat"], rec["id"], rec["variant"]) == (r, item.id, v.label)
        assert rec["in_flight"] in (0, 1) and rec["error"] is None
    assert Counter(c[:2] for c in gw.calls) == Counter((item.question, v.agent) for _, _, item, v in plan)
    assert Counter(rec["variant"] for rec in streamed) == {"a": 10, "b+think": 10, "c+fanout": 10}
    assert {label: len(rs) for label, rs in runs.items()} == {"a": 10, "b+think": 10, "c+fanout": 10}
    assert [line.split("]")[0] for line in lines] == [f"[{k}/30" for k in range(1, 31)]
    assert lines[-1].split("pass so far: ")[1].startswith("a 10/10, b+think 6/10, c+fanout 10/10")
    load = variants.in_flight(runs)
    assert set(load) == {"a", "b+think", "c+fanout"} and all(0.5 < m <= 1.0 for m in load.values())

    single = bench.run(FakeGateway(delay={"x": 0.01}), ITEMS, DOC_IDS, "deep", False, 2, "x", progress=lambda _: None,
                       concurrency=2)
    assert sorted(rec["order"] for rec in single) == list(range(1, 11))
    assert [(rec["repeat"], rec["id"]) for rec in sorted(single, key=lambda rec: rec["order"])] == [
        (r, it.id) for r in (1, 2) for it in ITEMS]


def test_a_failure_under_concurrency_is_recorded_and_shown_not_lost():
    gw = FakeGateway(delay={"a": 0.01}, fail={("Q2?", "c"), ("Q4?", "a")})
    streamed, lines = [], []
    variants.run(gw, ITEMS, DOC_IDS, "deep", 1, VS, 7, sink=streamed.append, progress=lines.append, concurrency=2)
    assert len(streamed) == 15
    failed = [rec for rec in streamed if rec["error"]]
    assert sorted((rec["id"], rec["variant"]) for rec in failed) == [("q2", "c+fanout"), ("q4", "a")]
    assert all(rec["error"] == "ReadTimeout: timed out" and rec["scores"]["correct"] == 0.0 for rec in failed)
    assert sum(": ERROR " in line for line in lines) == 2
    assert bench._run_stats(failed)["failed_runs"] == 2


def test_a_worker_exception_reaches_the_caller_after_the_requests_in_flight():
    done: list[int] = []

    def work(job: int) -> int:
        time.sleep(0.02 if job == 2 else 0.0)
        if job == 1:
            raise RuntimeError("worker bug")
        return job

    with pytest.raises(RuntimeError, match="worker bug"):
        dispatch([1, 2, 3, 4, 5], work, lambda k, job, result, others: done.append(result), concurrency=2)
    assert done == [2]  # 2 was in flight with 1 and finished; nothing new was sent after the failure
    with pytest.raises(ValueError):
        dispatch([1], work, lambda *a: None, concurrency=0)


def test_reports_state_the_concurrency_only_when_above_one():
    assert bench.load_note({}) == "" and bench.load_note({"concurrency": 1}) == ""
    note = bench.load_note({"concurrency": 2})
    assert "--concurrency 2" in note and "measured under load" in note and "fan-out" in note and "queue" in note
    rec = bench.record(ITEMS[0], 1, FakeGateway().ask("Q0?", ["doc-1"], agent="a"), DOC_IDS)
    a = {"meta": {"variant": "a"}, "runs": [rec]}
    b = {**a, "meta": {"variant": "b", "concurrency": 2}}
    assert any("concurrency differs: 1 vs 2" in n for n in bench.compare(a, b)["notes"])
    assert not any("concurrency" in n for n in bench.compare(a, {**a, "meta": {"variant": "b"}})["notes"])


def test_cli_concurrency_reaches_the_reports_and_one_leaves_them_as_they_were(gateway, tmp_path, capsys):  # noqa: F811
    bench_set = _local_set(tmp_path, gateway)
    seq, par = tmp_path / "seq", tmp_path / "par"
    common = ["bench", "--set", bench_set, "--variants", "deepagents,langgraph", "--repeats", "2", "--url", gateway.url]
    cli.main([*common, "--reports", str(seq)])
    cli.main([*common, "--reports", str(par), "--concurrency", "2"])
    err = capsys.readouterr().err
    assert "2 requests in flight at a time" in err and "1800 s" in err
    assert len(gateway.chats) == 2 * 2 * 2 * 2

    meta = json.loads(next(seq.glob("bench-combined-*.json")).read_text(encoding="utf-8"))["meta"]
    assert "concurrency" not in meta and "request_timeout_s" not in meta
    md = next(seq.glob("bench-combined-*.md")).read_text(encoding="ascii")
    assert "under load" not in md and "--concurrency" not in md and "at a time" not in md

    combined = json.loads(next(par.glob("bench-combined-*.json")).read_text(encoding="utf-8"))
    assert combined["meta"]["concurrency"] == 2 and combined["meta"]["request_timeout_s"] == 1800
    assert set(combined["in_flight"]) == {"deepagents", "langgraph"}
    assert combined["verdicts"]["all questions"]["means"] == {"deepagents": 1.0, "langgraph": 0.5}
    md = next(par.glob("bench-combined-*.md")).read_text(encoding="ascii")
    assert "Requests were sent 2 at a time in that same order (--concurrency 2)" in md
    assert "Environment:" in md and "measured under load" in md
    per_variant = json.loads(next(par.glob("bench-langgraph-*.json")).read_text(encoding="utf-8"))
    orders = [rec["order"] for rec in per_variant["runs"]]
    assert orders == sorted(orders) and len(orders) == 4  # reports keep the run order
    assert "measured under load" in next(par.glob("bench-langgraph-*.md")).read_text(encoding="ascii")
    stream = [json.loads(line) for line in next(par.glob("bench-combined-*.runs.jsonl")).read_text().splitlines()]
    assert sorted(rec["order"] for rec in stream) == list(range(1, 9))

    with pytest.raises(SystemExit, match="--concurrency must be at least 1"):
        cli.main([*common, "--concurrency", "0"])
    with pytest.raises(SystemExit, match="--request-timeout"):
        cli.main([*common, "--request-timeout", "0"])
