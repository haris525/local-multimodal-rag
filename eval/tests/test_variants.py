import subprocess
from collections import Counter

import pytest

from rag_eval import bench, environment, variants
from rag_eval.bench import Item
from rag_eval.variants import group_verdict, parse_variants, schedule


def test_variants_are_agent_names_with_an_optional_think_suffix():
    vs = parse_variants("langgraph, langgraph+think,deepagents,deepagents+think,deepagents_stock")
    assert [(v.label, v.agent, v.deep_think) for v in vs] == [
        ("langgraph", "langgraph", False), ("langgraph+think", "langgraph", True), ("deepagents", "deepagents", False),
        ("deepagents+think", "deepagents", True), ("deepagents_stock", "deepagents_stock", False)]
    for bad, message in (("langgraph", "at least two"), ("a,a", "twice"), ("a,b+fast", r"AGENT\+think"),
                         ("a,b c", r"AGENT\+think"), ("a+think,a-think", "file names")):
        with pytest.raises(ValueError, match=message):
            parse_variants(bad)



def test_fanout_and_calls_suffixes_combine_with_think_in_any_order():
    vs = parse_variants("custom_arm+fanout+calls20,custom_arm,deepagents+think+fanout,langgraph+calls5+think,"
                        "deepagents+calls30,deepagents_stock+fanout")
    assert [(v.label, v.agent, v.deep_think, v.fanout, v.research_calls) for v in vs] == [
        ("custom_arm+fanout+calls20", "custom_arm", False, True, 20),
        ("custom_arm", "custom_arm", False, False, None),
        ("deepagents+think+fanout", "deepagents", True, True, None),
        ("langgraph+calls5+think", "langgraph", True, False, 5),
        ("deepagents+calls30", "deepagents", False, False, 30),
        ("deepagents_stock+fanout", "deepagents_stock", False, True, None)]  # rag answers this one with a 422
    for bad, message in (("a,b+calls", r"AGENT\+think, with \+fanout and \+calls<N>"), ("a,b+calls0", r"AGENT\+think"),
                         ("a,b+calls07", r"AGENT\+think"), ("a,b+calls-3", r"AGENT\+think"),
                         ("a,b+Fanout", r"AGENT\+think"), ("a,b+fan-out", r"AGENT\+think"),
                         ("a,b+fanout+fanout", r"\+fanout is given twice"), ("a,b+calls5+calls6", r"\+calls is given twice"),
                         ("a,b+think+think", r"\+think is given twice"),
                         ("a+think+fanout,a+fanout+think", "'a\\+think\\+fanout' and 'a\\+fanout\\+think' send the same "
                                                           "request"),
                         ("a+calls20+fanout,a+fanout+calls20", "send the same request")):
        with pytest.raises(ValueError, match=message):
            parse_variants(bad)


def test_any_arm_the_rag_registers_is_a_variant_name_rag_validates_the_names():
    """eval knows no list of arms: an optional one (registered by rag) is passed through as given."""
    vs = parse_variants("langgraph,an_optional_arm,an_optional_arm+think")
    assert [(v.agent, v.deep_think) for v in vs] == [("langgraph", False), ("an_optional_arm", False),
                                                     ("an_optional_arm", True)]

ITEMS = [Item(id=f"q{i}", question=f"Q{i}?", docs=["d"], facts=["f"]) for i in range(6)]
VS = parse_variants("a,b,c+think")


def test_the_schedule_is_seeded_interleaved_per_question_and_randomized():
    plan = schedule(ITEMS, VS, repeats=3, seed=7)
    assert plan == schedule(ITEMS, VS, repeats=3, seed=7) and plan != schedule(ITEMS, VS, repeats=3, seed=8)
    assert len(plan) == 6 * 3 * 3
    blocks = [plan[k:k + 3] for k in range(0, len(plan), 3)]
    for block in blocks:  # every question and repeat runs all variants back to back, each once
        assert len({(r, it.id) for r, _, it, _ in block}) == 1 and {v.label for *_, v in block} == {"a", "b", "c+think"}
    orders = [[it.id for r, i, it, v in plan if r == rep and v.label == "a"] for rep in (1, 2, 3)]
    assert len({tuple(o) for o in orders}) > 1  # the question order changes between repeats
    first = Counter(block[0][3].label for block in blocks)
    assert len(first) == 3  # no variant always goes first


def pq_for(correct: dict[str, list[float]]) -> dict[str, dict]:
    return {f"q{i}": {"correct": c, "majority_correct": float(c > 0.5)} for i, c in enumerate(correct)}


def test_group_verdict_names_a_significant_winner_or_says_it_cannot():
    n = 40
    clear = {"good": pq_for([1.0] * n), "bad": pq_for([0.0] * 30 + [1.0] * 10), "worse": pq_for([0.0] * n)}
    v = group_verdict(clear, [f"q{i}" for i in range(n)])
    assert v["best"] == "good" and v["ranking"] == ["good", "bad", "worse"]
    assert v["text"].startswith("good wins on correct (1.000), significantly ahead of every other variant")
    assert all(a["p_holm"] < 0.05 for a in v["against"].values())
    close = {"x": pq_for([1.0] * 21 + [0.0] * 19), "y": pq_for([1.0] * 20 + [0.0] * 20)}
    v = group_verdict(close, [f"q{i}" for i in range(n)])
    assert "cannot pick a winner" in v["text"] and "not significant" in v["lines"][0]
    same = {"x": pq_for([1.0] * n), "y": pq_for([1.0] * n)}
    assert group_verdict(same, [f"q{i}" for i in range(n)])["text"].startswith("No single winner")


def test_permutation_holm_and_limit_hits():
    assert bench.permutation_p([0, 0, 0]) == 1.0
    assert bench.permutation_p([1.0] * 20) < 0.001
    assert bench.permutation_p([1, -1] * 10) == pytest.approx(1.0, abs=0.01)
    assert bench.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    rec = {"notices": ["research_budget", "answer_cut", "planner_search", "new_thing_cut", "context_full"],
           "finish_reason": "length"}
    assert bench.limit_hits(rec) == 5  # four limit notices and the length stop; planner_search is not a limit


def test_progress_lines_show_position_pass_rate_and_time_left():
    rec = {"id": "q2", "scores": {"correct": 1.0}, "error": None, "latency_s": 24.2, "tool_calls": 4, "tool_errors": 1}
    line = bench.progress_line(rec, done=2, total=9, repeat=1, repeats=3, index=2, n=3, passed=1, elapsed=50.0)
    assert line == ("[2/9] repeat 1/3, question 2/3 q2: PASS 24s, 4 tools, 1 tool errors | pass rate 1/2 (50%) | "
                    "50s elapsed, ~2m55s left")
    failed = bench.progress_line({**rec, "scores": {"correct": 0.0}, "error": "HTTP 502: bad gateway"}, 3, 9, 1, 3, 3,
                                 3, 1, 4000.0)
    assert ": ERROR " in failed and "1h06m elapsed" in failed and failed.endswith("| HTTP 502: bad gateway")
    multi = variants.progress_line({**rec, "variant": "b"}, 4, 54, 1, 3, 2, 6, {"a": [{}, {}], "b": [{}, {}]},
                                   {"a": 1, "b": 2}, 100.0)
    assert "question 2/6 q2, b: PASS" in multi and "pass so far: a 1/2, b 2/2" in multi


class FakeDocker:
    def __init__(self, docker: bool = True):
        self.docker = docker

    def __call__(self, args, **kwargs):
        def ok(out: str):
            return subprocess.CompletedProcess(args, 0, out, "")

        if args[0] == "git":
            return ok("abc123\n") if "rev-parse" in args else ok(" M eval/x.py\n")
        if not self.docker:
            raise FileNotFoundError("docker")
        if args[1] == "ps":
            return ok("rag01\n") if "label=com.docker.compose.service=rag" in args else ok("llm01\n")
        env = ('["RAG_LLM_MODEL=qwen3.8-27b", "RAG_LLM_CTX=65536", "RAG_TEMPERATURE=0.7", "RAG_AGENT_MAX_STEPS=8", '
               '"RAG_SERVICE_TOKEN=secret", "PATH=/usr/bin", "HF_TOKEN=hf_x"]')
        return ok(f"sha256:{'r' * 8 if args[-1] == 'rag01' else 'l' * 8}\t{env}\n")


def test_the_environment_records_images_model_context_and_only_safe_settings():
    env = environment.detect("local-rag", "deadbeef", runner=FakeDocker())
    assert env["git_commit"] == "abc123-dirty" and env["app_commit"] == "deadbeef"
    assert env["rag_image"] == "sha256:rrrrrrrr" and env["llm_image"] == "sha256:llllllll"
    assert env["model"] == "qwen3.8-27b" and env["llm_ctx"] == "65536"
    assert env["rag_settings"] == {"RAG_AGENT_MAX_STEPS": "8", "RAG_LLM_CTX": "65536", "RAG_LLM_MODEL": "qwen3.8-27b",
                                   "RAG_TEMPERATURE": "0.7"}  # no token, PATH or HF_TOKEN
    offline = environment.detect(runner=FakeDocker(docker=False))
    assert offline["rag_image"] is None and offline["model"] is None and offline["git_commit"] == "abc123-dirty"
    assert "unknown" in bench.environment_line(offline)


def test_compare_reports_per_tag_cost_and_a_plain_verdict():
    def report(variant, correct, latency, notices):
        runs = [{"id": f"q{i}", "repeat": 1, "latency_s": latency, "error": None, "tool_calls": 3, "tool_errors": 0,
                 "duplicate_calls": 0, "notices": notices, "tools": [], "finish_reason": "stop", "why": [],
                 "usage": None, "scores": {"correct": float(c), "citations_ok": 1.0}} for i, c in enumerate(correct)]
        items = [{"id": f"q{i}", "tags": ["numeric"] if i < 10 else ["image"]} for i in range(len(correct))]
        return {"kind": "bench", "items": items, "runs": runs,
                "meta": {"variant": variant, "repeats": 1, "mode": "deep", "environment": {"rag_image": "sha256:x"}}}

    a = report("deepagents", [1] * 30, 20.0, ["research_budget"])
    b = report("langgraph", [0] * 15 + [1] * 15, 30.0, [])
    c = bench.compare(a, b)
    assert c["verdict"].startswith("A (deepagents) wins on correct over B (langgraph)") and "significant" in c["verdict"]
    assert c["by_tag"]["numeric"]["wins"] == 10 and c["by_tag"]["image"]["wins"] == 5
    assert c["runs"]["a"]["limit_hits_per_run"] == 1.0 and c["runs"]["b"]["latency_p90"] == 30.0
    text = bench.compare_markdown(c)
    assert "**Verdict:** A (deepagents) wins" in text and "| numeric | 10 | 10 | 0 | 0 |" in text
    assert "limit hits per run" in text and "no LLM judge" in text


def test_the_python_suffix_sends_python_and_combines_with_the_others():
    vs = parse_variants("deepagents+python,deepagents,langgraph+calls20+python+think")
    assert [(v.label, v.agent, v.python, v.research_calls, v.deep_think) for v in vs] == [
        ("deepagents+python", "deepagents", True, None, False), ("deepagents", "deepagents", False, None, False),
        ("langgraph+calls20+python+think", "langgraph", True, 20, True)]
    for bad, message in (("a,b+python+python", r"\+python is given twice"), ("a,b+Python", r"AGENT\+think"),
                         ("a+python+think,a+think+python", "send the same request")):
        with pytest.raises(ValueError, match=message):
            parse_variants(bad)
