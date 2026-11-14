import json

import pytest

from rag_common.chat_events import read_chat
from rag_eval import bench
from rag_eval.bench import Item, compare, mcnemar_exact, parse_items, per_question, record, select, summarize
from rag_eval.client import Turn

ITEM = Item(id="pop", question="Which state grew fastest?", docs=["pop.xlsx", "other.pdf"], expected_docs=["pop.xlsx"],
            tags=["single-document", "numeric"], answers=["South Carolina"],
            numbers=[{"value": 1.46, "tol": 0.01}], facts=["South Carolina"])
IDS = {"pop.xlsx": "p" * 32, "other.pdf": "o" * 32}


def sse(answer: str, sources: list[dict], steps: list[dict] = (), notices: list[dict] = (), cut: bool = False) -> list[str]:
    events = [("step", s) for s in steps] + [("notice", x) for x in notices]
    events += [("sources", {"route": "deep", "sources": sources}), ("token", {"text": answer})]
    if not cut:
        events.append(("done", {"citations": [], "invalid_citations": [], "finish_reason": "stop",
                                "usage": {"prompt_tokens": 900, "completion_tokens": 120}, "timings": {"tool_calls": len(steps)}}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).splitlines()


SRC = [{"n": 1, "doc_id": "p" * 32, "doc_name": "pop.xlsx"}, {"n": 2, "doc_id": "o" * 32, "doc_name": "other.pdf"}]
STEPS = [{"tool": "plan", "args": {}, "error": None},
         {"tool": "sql", "args": {"q": "SELECT 1"}, "error": None},
         {"tool": "sql", "args": {"q": "SELECT 1"}, "error": None},
         {"tool": "describe_table", "args": {"doc_id": "pop.xlsx"}, "error": "unknown document"},
         {"tool": "verify", "args": {"missing": ["x"]}, "error": None}]


def turn(answer: str, **kw) -> Turn:
    reply = read_chat(sse(answer, SRC, **kw))
    return Turn(reply, 200, reply.failure, 12.5)


def test_a_run_records_scores_tools_errors_notices_usage_and_cited_documents():
    rec = record(ITEM, 1, turn("South Carolina grew 1.456% [1].", steps=STEPS,
                               notices=[{"code": "planner_search", "message": "fell back"}]), IDS)
    assert rec["scores"]["correct"] == 1.0 and rec["scores"]["correct_cited"] == 1.0
    assert rec["scores"]["em"] == 0.0 and rec["scores"]["contains"] == 1.0 and 0 < rec["scores"]["f1"] < 1
    assert rec["tools"] == ["sql", "sql", "describe_table"]  # plan and verify are not tool calls
    assert (rec["tool_errors"], rec["duplicate_calls"], rec["tool_calls"]) == (1, 1, 5)  # tool_calls from timings
    assert rec["notices"] == ["planner_search"] and rec["finish_reason"] == "stop"
    assert rec["usage"]["completion_tokens"] == 120 and rec["cited_docs"] == ["pop.xlsx"]
    assert rec["why"] == [] and rec["latency_s"] == 12.5


def test_wrong_numbers_and_stray_citations_are_explained():
    rec = record(ITEM, 1, turn("Idaho grew 1.44% [2]."), IDS)
    assert rec["scores"]["correct"] == 0.0 and rec["scores"]["citations_ok"] == 0.0
    assert "number missing: 1.46" in rec["why"] and "missing: South Carolina" in rec["why"]
    assert "citations: outside the expected documents" in rec["why"]


def test_a_cut_stream_or_http_error_is_a_failed_run_never_a_scored_answer():
    rec = record(ITEM, 1, turn("South Carolina grew 1.46% [1].", cut=True), IDS)
    assert rec["scores"]["correct"] == 0.0 and rec["error"] == "the answer stream ended before its done event"
    assert rec["answer"].startswith("South Carolina")  # kept for reading, not scored
    failed = record(ITEM, 2, Turn(None, 409, "HTTP 409: not indexed", 0.1), IDS)
    assert failed["scores"]["correct"] == 0.0 and failed["why"] == ["error: HTTP 409: not indexed"]


def runs_for(item_id: str, correct: list[int], latency: float = 10.0, errors: int = 0) -> list[dict]:
    return [{"id": item_id, "repeat": i + 1, "latency_s": latency, "error": None, "tool_calls": 4, "tool_errors": errors,
             "duplicate_calls": 0, "notices": [], "tools": ["sql"], "finish_reason": "stop", "why": [],
             "usage": {"completion_tokens": 100}, "scores": {"correct": float(c), "correct_cited": float(c),
                                                             "citations_ok": 1.0, "f1": 0.5 * c}}
            for i, c in enumerate(correct)]


def test_repeats_are_averaged_per_question_before_the_intervals():
    runs = runs_for("a", [1, 1, 0]) + runs_for("b", [0, 0, 0])
    pq = per_question(runs)
    assert pq["a"]["correct"] == pytest.approx(2 / 3) and pq["a"]["majority_correct"] == 1.0 and pq["a"]["all_correct"] == 0.0
    items = [Item(id="a", question="q", docs=["x"], tags=["t1"], facts=["f"]),
             Item(id="b", question="q", docs=["x"], tags=["t1", "t2"], facts=["f"])]
    s = summarize(items, runs)
    assert s["questions"] == 2 and s["runs"] == 6
    assert s["overall"]["correct"][0] == pytest.approx(1 / 3)
    assert s["by_tag"]["t2"]["questions"] == 1 and s["by_tag"]["t2"]["correct"][0] == 0.0
    assert s["latency"]["median"] == 10.0 and s["tool_names"] == {"sql": 6}


def test_mcnemar_exact_p_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(10, 1) == pytest.approx(2 * (1 + 11) / 2**11)


def report(variant: str, per_q: dict[str, list[int]], latency: float, errors: int) -> dict:
    runs = [r for q, c in per_q.items() for r in runs_for(q, c, latency, errors)]
    return {"kind": "bench", "meta": {"variant": variant, "set_digest": "s", "mode": "deep", "think": False, "repeats": 3},
            "runs": runs}


def test_compare_counts_wins_losses_ties_and_cost_differences():
    a = report("deepagents", {"q1": [1, 1, 1], "q2": [1, 1, 0], "q3": [0, 0, 0], "q4": [1, 1, 1]}, 20.0, 0)
    b = report("langgraph", {"q1": [0, 0, 0], "q2": [1, 1, 0], "q3": [0, 0, 1], "q4": [1, 1, 1], "q5": [1, 1, 1]}, 30.0, 1)
    c = compare(a, b)
    assert (c["questions"], c["wins"], c["losses"], c["ties"]) == (4, 1, 1, 2)
    assert c["win_ids"] == ["q1"] and c["loss_ids"] == ["q3"]
    assert c["mcnemar"]["a_only_majority_correct"] == 1 and c["mcnemar"]["b_only_majority_correct"] == 0
    assert c["cost"]["latency_s"]["difference"][0] == pytest.approx(-10.0)
    assert c["cost"]["tool_errors"]["a"] == 0.0 and c["cost"]["tool_errors"]["b"] == 1.0
    assert any("only one report" in note for note in c["notes"])
    text = bench.compare_markdown(c)
    assert "A wins 1, loses 1, ties 2" in text and "McNemar" in text
    with pytest.raises(SystemExit, match="share no question"):
        compare(a, report("x", {"zz": [1]}, 1, 0))


def test_items_are_validated_with_line_numbers():
    good = json.dumps({"id": "a", "question": "q?", "docs": ["x.pdf"], "facts": ["f"]})
    assert parse_items([good, ""])[0].expected_docs == ["x.pdf"]
    cases = {
        json.dumps({"id": "a", "question": "q?", "docs": ["x.pdf"]}): "numbers or facts",
        json.dumps({"id": "a", "question": "q?", "docs": [f"{i}.pdf" for i in range(11)], "facts": ["f"]}): "1 to 10",
        json.dumps({"id": "a", "question": "q?", "docs": ["x.pdf"], "facts": ["f"], "expected_docs": ["y.pdf"]}): "among docs",
        json.dumps({"id": "a", "question": "q?", "docs": ["x.pdf"], "facts": [{"oops": 1}]}): "unknown fact",
        json.dumps({"id": "a", "question": "q?", "docs": ["x.pdf"], "facts": ["f"], "colour": 1}): "unknown fields",
        json.dumps({"id": "a", "question": "q?", "docs": ["x.pdf"], "correct_by": "contains"}): "answers",
    }
    for line, message in cases.items():
        with pytest.raises(ValueError, match=message):
            parse_items([good.replace('"a"', '"z"'), line], "set.jsonl")
    with pytest.raises(ValueError, match="set.jsonl:2: duplicate id a"):
        parse_items([good, good], "set.jsonl")


def test_select_by_tags_skip_tags_ids_and_limit():
    items = [Item(id=str(i), question="q", docs=["x"], tags=tags, facts=["f"])
             for i, tags in enumerate([["image"], ["table-sql", "numeric"], ["image", "hard-case"], []])]
    assert [it.id for it in select(items, ["IMAGE"])] == ["0", "2"]
    assert [it.id for it in select(items, ["image"], ["hard-case"])] == ["0"]
    assert [it.id for it in select(items, ids=["1", "3"])] == ["1", "3"]
    assert [it.id for it in select(items, limit=2)] == ["0", "1"]


def test_report_files_and_markdown(tmp_path):
    items = [Item(id="a", question="q", docs=["x"], tags=["image"], facts=["f"], answers=["f"])]
    runs = runs_for("a", [1, 0, 1])
    meta = {"variant": "tuned agent/v2", "set": "local", "mode": "deep", "think": False, "repeats": 3, "agent": "deepagents",
            "set_digest": "abc", "started": "s", "finished": "f", "stamp": "20250101-120000"}
    js, md = bench.write_report(bench.build_report(meta, items, runs), items, tmp_path)
    assert js.name == "bench-tuned-agent-v2-20250101-120000.json" and md.suffix == ".md"
    text = md.read_text(encoding="utf-8")
    assert "| correct | 0.667 [" in text and "| image | 1 |" in text and "| a | image | 2/3 |" in text
    assert json.loads(js.read_text(encoding="utf-8"))["summary"]["questions"] == 1
    text.encode("ascii")  # plain ASCII reports
