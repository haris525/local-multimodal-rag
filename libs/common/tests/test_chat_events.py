"""The shared chat stream reader (rag_common.chat_events): SSE parsing, the order table, terminal events."""

import ast
import sys
from pathlib import Path

import pytest

from rag_common import chat_events
from rag_common.chat_events import ChatEvent, Order, Phase, ProtocolError, check_order, parse_sse, read_chat


def sse(*events: tuple[str, str]) -> list[str]:
    lines = []
    for name, data in events:
        lines += [f"event: {name}", f"data: {data}", ""]
    return lines


def test_parse_sse_splits_on_blank_lines_and_joins_data_lines():
    lines = ["event: sources", 'data: {"route": "search",', 'data: "sources": []}', "", "data:{\"text\": \"Hi\"}"]
    assert list(parse_sse(lines)) == [("sources", {"route": "search", "sources": []}), ("message", {"text": "Hi"})]


def test_a_complete_answer():
    reply = read_chat(sse(("memory", '{"action": "saved", "items": [{"memory_id": "x", "text": "t"}]}'),
                          ("sources", '{"route": "search", "sources": [{"n": 1}]}'),
                          ("reasoning", '{"text": "hmm"}'), ("token", '{"text": "Hello "}'), ("token", '{"text": "[1]"}'),
                          ("done", '{"citations": [1], "invalid_citations": [], "timings": {"total_ms": 5}}')))
    assert reply.complete and reply.failure is None and reply.terminal == "done"
    assert reply.answer == "Hello [1]" and reply.route == "search" and reply.citations == [1]
    assert reply.memory[0]["action"] == "saved" and reply.timings == {"total_ms": 5} and len(reply.events) == 6


def test_a_cut_stream_and_an_error_are_told_apart():
    cut = read_chat(sse(("sources", '{"route": "search", "sources": []}'), ("token", '{"text": "Half"}')))
    assert cut.terminal is None and not cut.complete and "before its done event" in cut.failure
    failed = read_chat(sse(("sources", '{"route": "search", "sources": []}'), ("error", '{"message": "boom"}')))
    assert failed.terminal == "error" and failed.error == "boom" and failed.failure == "boom"


def test_an_unknown_or_misplaced_event_is_a_protocol_problem():
    unknown = read_chat(sse(("sources", '{"route": "search", "sources": []}'), ("summary", '{"text": "x"}'),
                            ("done", '{"citations": []}')))
    assert unknown.terminal == "done" and not unknown.complete and "unknown event 'summary'" in unknown.problem
    early = read_chat(sse(("token", '{"text": "a"}'), ("done", '{"citations": []}')))
    assert early.failure.startswith("protocol: 'done' may not follow 'notes'")


def test_the_order_follows_next():
    order = Order()
    assert [order.step(n) for n in ("memory", "token", "step", "sources", "token", "done")][-1] == Phase.DONE
    assert order.terminal == "done"
    with pytest.raises(ProtocolError):
        order.step("token")  # nothing after done
    assert check_order([]) == Phase.START
    assert {e.value for e in ChatEvent} == {"memory", "step", "sources", "reasoning", "token", "done", "error", "notice"}


def test_the_module_needs_only_the_standard_library():
    """The scripts load it from libs/common/src without installing the package or its dependencies."""
    tree = ast.parse(Path(chat_events.__file__).read_text(encoding="utf-8"))
    modules = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    modules |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules <= set(sys.stdlib_module_names)


def test_a_notice_may_come_anywhere_before_the_end_and_keeps_the_phase():
    reply = read_chat(sse(("notice", '{"code": "planner_fallback", "message": "SQL failed"}'),
                          ("sources", '{"route": "search", "sources": []}'), ("token", '{"text": "A"}'),
                          ("notice", '{"code": "answer_cut", "message": "cut"}'), ("done", '{"citations": []}')))
    assert reply.complete and [n["code"] for n in reply.notices] == ["planner_fallback", "answer_cut"]
    order = Order()
    for name in ("step", "notice", "step", "notice", "sources", "notice", "token", "notice", "done"):
        order.step(name)
    assert order.terminal == "done"
    with pytest.raises(ProtocolError):
        order.step("notice")  # nothing after done


def test_the_step_table_names_the_steps_of_every_deep_research_implementation():
    """Every `tool` a step event may carry is written down once, with what its args hold; the readers keep
    step events whatever their tool."""
    research = {"search", "grep", "read", "list_tables", "describe_table", "sql", "peek", "profile_column"}
    assert research | {"plan", "verify"} <= set(chat_events.STEP_TOOLS)  # langgraph
    assert research | {"verify", "write_todos", "read_file", "task", "summarize"} <= set(chat_events.STEP_TOOLS)  # deepagents
    stock = {"doc_grep", "write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "task"}
    assert research - {"grep"} | stock <= set(chat_events.STEP_TOOLS)  # deepagents_stock
    assert {"sections", "map", "reduce"} <= set(chat_events.STEP_TOOLS)  # exhaustive mode
    assert all(isinstance(v, str) and v.startswith("{") for v in chat_events.STEP_TOOLS.values())
    reply = read_chat(sse(("step", '{"tool": "write_todos", "args": {"todos": ["a (pending)"]}, "sources": null, "error": null}'),
                          ("step", '{"tool": "sql", "args": {"doc_id": "d", "query": "SELECT 1"}, "sources": [1], "error": null}'),
                          ("sources", '{"route": "deep", "sources": [{"n": 1}]}'), ("done", '{"citations": []}')))
    assert reply.complete and [s["tool"] for s in reply.steps] == ["write_todos", "sql"]
