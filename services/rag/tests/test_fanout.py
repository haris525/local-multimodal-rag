"""Fan-out (rag.fanout) on the core arms: leases (reserve, refuse, unused calls returned, the researcher limit, the
total never above the budget), the 422s and the start check, the concurrency cap, no nesting, the cap on a
researcher's report, the time limit and a failure with their notices, the research_calls override applied to every
arm alike, what is recorded, and a scripted run of the tuned deepagents arm with fan-out on, where both researchers
get their calls and the lead cannot spend them first. Nothing here names an optional arm."""

import asyncio
import random
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command
from pydantic import Field, ValidationError
from test_deepagents import (QUESTION, REPORT, TABLES, TOKENIZER, USER, Ingest, Retrieval, Sql, AnswerLLM, call,
                             calls, done, event, steps, todos)

from conftest import KEYS
from rag_common.identity import Identity, current_identity

import rag.agent as agent
import rag.agent_deepagents as da
import rag.agent_deepagents_stock as stock
import rag.api as api
import rag.turn as turn
from rag import arms, fanout, notices, tokens
from rag.agent import SourceBook
from rag.chat import (ChatRequest, deep_fanout, request_problem, request_settings, research_calls, with_settings)
from rag.config import Settings
from rag.conversation import Conversation
from rag.fanout import FanOut, Lease


@pytest.fixture(autouse=True)
def estimate_mode():
    tokens.configure(None)
    yield
    tokens.configure(None)


# --- fakes ------------------------------------------------------------------------------------------------------


class Routed(BaseChatModel):
    """One script per conversation, chosen by the start of its first human message: the lead's is the question, a
    researcher's is its task description (researchers run at the same time, so one shared script would interleave).
    Records (key, messages, settings) per call; `delay` sleeps before a key's replies."""

    scripts: dict = Field(default_factory=dict)
    calls: list = Field(default_factory=list)
    delay: dict = Field(default_factory=dict)
    in_flight: list = Field(default_factory=list)  # researcher keys between their first and last call

    @property
    def _llm_type(self) -> str:
        return "routed"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tool_names=[t.name if hasattr(t, "name") else t["name"] for t in tools], **kwargs)

    def _generate(self, *args, **kwargs):
        raise NotImplementedError("research runs async")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        first = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        key = next((k for k in self.scripts if first.startswith(k)), first[:40])
        self.calls.append((key, list(messages), kwargs))
        if self.delay.get(key):
            await asyncio.sleep(self.delay[key])
        script = self.scripts.get(key) or []
        reply = script.pop(0) if script else done("DONE" if key == QUESTION else f"{key} findings [1].")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def of(self, key: str) -> list:
        return [(m, k) for kk, m, k in self.calls if kk == key]


def make_deps(model, s: Settings | None = None, sql: Sql | None = None):
    return SimpleNamespace(s=s or Settings(), graph_llm=model, sql=sql or Sql(), retrieval=Retrieval(), ingest=Ingest(),
                           llm=AnswerLLM())


def run_deep(deps, agent_name: str = "deepagents", docs=(TABLES, REPORT), **fields):
    """One deep search as USER, with the request's settings applied as turn.respond applies them."""
    req = ChatRequest(question=QUESTION, doc_ids=[d.doc_id for d in docs], mode="deep", agent=agent_name, **fields)
    deps = with_settings(deps, request_settings(deps.s, req))

    async def go():
        queue = notices.begin()
        token = current_identity.set(Identity(USER, "user", "auth"))
        try:
            events = [e async for e in arms.get(agent_name).deep_search(deps, req, list(docs), Conversation(),
                                                                         QUESTION, {})]
        finally:
            current_identity.reset(token)
        return events, queue
    return asyncio.run(go())


def sql(i: int | str) -> tuple:
    return ("sql", {"doc_id": TABLES.doc_id, "query": f"SELECT {i}"}, None)


def task(description: str) -> tuple:
    return ("task", {"description": description, "subagent_type": "researcher"}, None)


def codes(queue) -> list[str]:
    return [n["code"] for n in queue]


# --- leases ------------------------------------------------------------------------------------------------------


def test_a_lease_is_reserved_refused_when_too_few_calls_are_free_and_unused_calls_come_back():
    fan = FanOut(budget=10, lease_size=4, max_tasks=3, return_tokens=800, timeout_s=300)
    assert fan.task_refusal(used=2) is None
    a = fan.reserve()
    assert fan.held() == 4 and fan.free(2) == 4
    assert fan.task_refusal(used=2) is None
    b = fan.reserve()
    assert fan.free(2) == 0 and fan.lead_refusal(2) == fanout.RESERVED
    why = fan.task_refusal(used=2)  # 0 free: no third lease
    assert why == fanout.REFUSED_FULL.format(lease=4, free=0, budget=10, used=2, held=8) and fan.refused == 1
    assert "only 0 are left of the 10" in why
    assert a.take() is None and a.used == 1
    fan.release(a, ran=True)  # a used 1 of its 4: 3 come back
    assert fan.held() == 4 and fan.free(3) == 3 and fan.delegated == 1
    for _ in range(4):
        assert b.take() is None
    assert b.take() == fanout.LEASE_USED.format(lease=4) and b.used == 4
    fan.release(b, ran=True)
    assert fan.held() == 0 and fan.free(7) == 3 and fan.started == 2


def test_the_researcher_limit_refuses_a_task_and_a_task_that_started_no_researcher_is_not_counted():
    fan = FanOut(budget=30, lease_size=4, max_tasks=2, return_tokens=800, timeout_s=300)
    first, second = fan.reserve(), fan.reserve()
    fan.release(first, ran=True)
    fan.release(second, ran=False)  # e.g. an unknown subagent type: no researcher ran
    assert fan.started == 1 and fan.task_refusal(0) is None
    fan.release(fan.reserve(), ran=True)
    why = fan.task_refusal(0)
    assert why == fanout.REFUSED_TASKS.format(tasks=2, free=30) and "RAG_DEEP_MAX_TASKS" in why


def test_the_total_of_used_and_reserved_calls_never_passes_the_budget_whatever_the_order():
    """A random mix of lead calls, task calls, researcher calls and researchers ending, through Research's own
    admission (the counting the arms use)."""
    rng = random.Random(11)
    for budget, lease_size, max_tasks in ((10, 4, 2), (10, 3, 3), (20, 8, 2), (5, 2, 4), (1, 1, 1), (30, 12, 2)):
        s = Settings(agent_max_steps=budget, deep_task_calls=lease_size, deep_max_tasks=max_tasks)
        for _ in range(40):
            run = da.Run(s, SourceBook(), False, frozenset({"sql"}), fanout=FanOut.of(s))
            gate = da.Research(run)
            running: list[Lease] = []
            for _ in range(60):
                op = rng.choice(["lead", "task", "researcher", "researcher", "end"])
                if op == "lead":
                    gate._refusal("sql")
                elif op == "task" and gate._refusal("task") is None:
                    running.append(run.fanout.reserve())
                elif op == "researcher" and running:
                    token = fanout.enter(rng.choice(running))
                    try:
                        gate._refusal("sql")
                    finally:
                        fanout.leave(token)
                elif op == "end" and running:
                    run.fanout.release(running.pop(rng.randrange(len(running))), ran=True)
                assert run.tool_calls + run.fanout.held() <= budget
                assert all(lease.used <= lease.size for lease in running)
                assert run.fanout.started <= max_tasks


def test_a_researcher_cannot_start_researchers_of_its_own():
    s = Settings(deep_fanout=True, deep_agent="deepagents")
    run = da.Run(s, SourceBook(), False, frozenset({"sql"}), fanout=FanOut.of(s))
    token = fanout.enter(run.fanout.reserve())
    try:
        assert da.Research(run)._refusal("task") == fanout.NESTED
    finally:
        fanout.leave(token)
    assert fanout.current() is None


def test_a_researcher_whose_prompt_fills_the_slot_ends_alone_and_the_lead_goes_on():
    s = Settings()
    run = da.Run(s, SourceBook(), False, frozenset({"sql"}), fanout=FanOut.of(s))
    lease = run.fanout.reserve()
    token = fanout.enter(lease)
    try:
        da.halt(run)  # what the slot guard does when the researcher's prompt is full
    finally:
        fanout.leave(token)
    assert lease.stopped and lease.done() and not run.stopped
    da.halt(run)  # in the lead: the research ends, as without fan-out
    assert run.stopped


# --- requests and settings ----------------------------------------------------------------------------------------


def test_fanout_is_a_422_for_an_arm_without_it_and_research_calls_outside_its_range_is_a_422():
    client = TestClient(api.app)
    supported = ", ".join(arms.fanout_names())
    for body, message in (
            ({"agent": "langgraph", "fanout": True}, f"deep-search agent 'langgraph' does not support fan-out; arms that do: {supported}"),
            ({"agent": "deepagents_stock", "fanout": True}, "'deepagents_stock' does not support fan-out"),
            ({"fanout": True}, "'langgraph' does not support fan-out"),  # RAG_DEEP_AGENT's default
            ({"fanout": "sometimes"}, "fanout"),
            ({"research_calls": 0}, "research_calls"),
            ({"research_calls": 31}, "research_calls must be between 1 and 30 (RAG_AGENT_MAX_STEPS_LIMIT), not 31"),
            ({"research_calls": "many"}, "research_calls")):
        r = client.post("/v1/chat", json={"question": "q", "doc_ids": [TABLES.doc_id], "mode": "deep", **body},
                        headers=KEYS.headers(USER))
        assert r.status_code == 422, body
        assert message in r.text, (body, r.text)
    s = Settings()
    ok = ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep", agent="deepagents", fanout=True, research_calls=30)
    assert request_problem(s, ok) is None
    assert request_problem(s, ok.model_copy(update={"fanout": False, "agent": "langgraph"})) is None
    single = ChatRequest(question="q", doc_ids=[TABLES.doc_id], fanout=True)  # single answers run no arm
    assert request_problem(s, single) is None


def test_the_fanout_setting_fails_at_start_for_an_arm_without_it_and_a_request_overrides_it(monkeypatch):
    for name in ("langgraph", "deepagents_stock"):
        with pytest.raises(ValidationError, match=f"RAG_DEEP_FANOUT is on, but deep-search agent '{name}' does not "
                                                  "support fan-out; arms that do: deepagents"):
            Settings(deep_fanout=True, deep_agent=name)
    req = ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep")
    assert deep_fanout(Settings(deep_agent="deepagents"), req) is False  # off by default
    monkeypatch.setenv("RAG_DEEP_AGENT", "deepagents")
    monkeypatch.setenv("RAG_DEEP_FANOUT", "true")
    s = Settings()
    assert s.deep_fanout is True
    assert deep_fanout(s, req) is True
    assert deep_fanout(s, req.model_copy(update={"fanout": False})) is False
    assert deep_fanout(s, req.model_copy(update={"agent": "langgraph"})) is False  # the setting reaches arms with it
    for field, bad in (("deep_task_calls", 0), ("deep_max_tasks", 0), ("deep_task_concurrency", 0),
                       ("deep_task_return_tokens", 0), ("deep_task_timeout_s", 0), ("agent_max_steps_limit", 0)):
        with pytest.raises(ValidationError):
            Settings(**{field: bad})
    assert arms.supports_fanout("deepagents") and not arms.supports_fanout("langgraph")
    assert not arms.supports_fanout("deepagents_stock") and not arms.supports_fanout("nope")


def test_research_calls_replaces_the_budget_for_every_arm_alike_and_scales_the_lease(monkeypatch):
    s = Settings()
    req = ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep", research_calls=20)
    r = request_settings(s, req)
    assert (r.agent_max_steps, r.deep_task_calls, r.agent_recursion_limit) == (20, 8, s.agent_recursion_limit + 20)
    assert r.research_budget_from == "the request's research_calls" and s.research_budget_from == "RAG_AGENT_MAX_STEPS"
    assert s.agent_max_steps == 10 and research_calls(s, req) == 20 and research_calls(s, ChatRequest(question="q")) == 10
    for n, lease, limit in ((5, 2, 80), (1, 1, 80), (3, 1, 80), (30, 12, 120)):
        r = request_settings(s, req.model_copy(update={"research_calls": n}))
        assert (r.agent_max_steps, r.deep_task_calls, r.agent_recursion_limit) == (n, lease, limit)
    assert request_settings(s, ChatRequest(question="q")) is s  # absent: the settings themselves
    assert request_settings(s, req.model_copy(update={"research_calls": 10})) is s

    seen = {}

    def fake(name):
        async def deep_search(deps, req, docs, conv, question, timings):
            seen[name] = (deps.s.agent_max_steps, deps.s.research_budget_from)
            for e in agent.nothing_found(timings):
                yield e
        return deep_search

    for name, module in (("langgraph", agent), ("deepagents", da), ("deepagents_stock", stock)):
        monkeypatch.setattr(module, "deep_search", fake(name))
    monkeypatch.setattr(turn, "standalone_question", lambda deps, q, conv: asyncio.sleep(0, q))
    for name in ("langgraph", "deepagents", "deepagents_stock"):
        deps = SimpleNamespace(s=s, llm=None, threads=None, retrieval=None)
        request = ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep", agent=name, research_calls=20)

        async def go():
            return [e async for e in turn.respond(deps, request, [TABLES], Conversation())]
        asyncio.run(go())
        assert deps.s is s  # the caller's deps keep their settings
    assert seen == dict.fromkeys(("langgraph", "deepagents", "deepagents_stock"), (20, "the request's research_calls"))


def test_a_smaller_research_budget_ends_the_langgraph_and_deepagents_research_with_its_own_name():
    class Graph(Routed):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            first = str(messages[0].content)
            if first.startswith("Break this question"):
                return ChatResult(generations=[ChatGeneration(message=done('{"sub_questions": ["Q3 revenue"]}'))])
            if "NOT yet answered" in first:
                return ChatResult(generations=[ChatGeneration(message=done('{"missing": []}'))])
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    for name in ("langgraph", "deepagents"):
        model = Graph(scripts={"Question:" if name == "langgraph" else QUESTION: [calls(*[sql(i) for i in range(5)])]})
        deps = make_deps(model)
        events, queue = run_deep(deps, name, docs=(TABLES,), research_calls=3)
        assert len(deps.sql.calls) == 3, name
        budget = [n for n in queue if n["code"] == "research_budget"]
        assert budget and budget[0]["message"] == ("Deep search used all 3 of its tool calls; the answer uses what it "
                                                   "found so far (the request's research_calls)."), name
        assert event(events, "done")["timings"]["tool_calls"] == 3


def test_fanout_and_the_research_budget_are_recorded_with_the_agent_and_deep_think(monkeypatch):
    monkeypatch.setattr(api, "settings", Settings())
    deep = ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep", agent="deepagents", fanout=True,
                       research_calls=20)
    assert api._deep(deep) == {"agent": "deepagents", "deep_think": False, "fanout": True, "research_calls": 20,
                               "python": False}
    assert api._deep(ChatRequest(question="q", doc_ids=[TABLES.doc_id], mode="deep")) == {
        "agent": "langgraph", "deep_think": False, "fanout": False, "research_calls": 10, "python": False}
    assert api._deep(ChatRequest(question="q", doc_ids=[TABLES.doc_id])) == {}


# --- delegation: concurrency, report cap, time limit, failure -----------------------------------------------------


def _research(s: Settings) -> tuple[da.Run, da.Research]:
    run = da.Run(s, SourceBook(), False, frozenset({"sql"}), fanout=FanOut.of(s))
    return run, da.Research(run)


def _task_call(i: int) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": "task", "args": {"description": f"part {i}", "subagent_type": "researcher"},
                                      "id": f"t{i}"})


def report(request, text: str = "South 2735.62 [1].") -> Command:
    return Command(update={"messages": [ToolMessage(text, tool_call_id=request.tool_call["id"])]})


@pytest.mark.parametrize("concurrency", [1, 2, 3])
def test_at_most_the_concurrency_cap_of_researchers_run_at_once_and_the_others_wait_holding_their_lease(concurrency):
    s = Settings(agent_max_steps=12, deep_task_calls=4, deep_max_tasks=3, deep_task_concurrency=concurrency)
    run, gate = _research(s)
    live, peak, held = [0], [0], []

    async def handler(request):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0)  # the other task calls of the reply reserve as soon as this one yields
        held.append(run.fanout.held())
        await asyncio.sleep(0.02)
        live[0] -= 1
        return report(request)

    async def go():
        notices.begin()
        return await asyncio.gather(*(gate.awrap_tool_call(_task_call(i), handler) for i in range(3)))

    results = asyncio.run(go())
    assert peak[0] == concurrency and all(isinstance(r, Command) for r in results)
    assert held[0] == 12  # all three leases were reserved before the first researcher's first await returned
    assert run.fanout.started == 3 and run.fanout.active == [] and run.harness_calls == 3


def test_a_long_report_reaches_the_lead_cut_with_a_note_and_a_notice_and_an_empty_one_says_so():
    s = Settings(deep_task_return_tokens=40)
    run, gate = _research(s)
    long = " ".join(f"Region {i} revenue {1000 + i} [{i % 3 + 1}]." for i in range(80))

    async def handler(request):
        fanout.current().found([1, 2])
        return report(request, long if request.tool_call["id"] == "t0" else "  ")

    async def go():
        queue = notices.begin()
        return [await gate.awrap_tool_call(_task_call(i), handler) for i in range(2)], queue

    (cut, empty), queue = asyncio.run(go())
    text = cut.update["messages"][-1].content
    body, note = text.split("\n\n[Report cut at ")
    assert tokens.count_tokens(body) <= 40 and long.startswith(body)
    assert note.startswith(f"40 of its {tokens.count_tokens(long)} tokens (RAG_DEEP_TASK_RETURN_TOKENS). The sources "
                           "it found stay in the source book: [1], [2].]")
    assert empty.update["messages"][-1].content == "The researcher returned no findings text. The sources it found " \
                                                    "stay in the source book: [1], [2]."
    assert codes(queue) == ["research_task_cut"] and "RAG_DEEP_TASK_RETURN_TOKENS" in queue[0]["message"]


def test_a_researcher_past_its_time_limit_or_failing_gives_the_lead_an_error_and_the_user_a_notice(caplog):
    s = Settings(deep_task_timeout_s=0.05)
    run, gate = _research(s)

    async def slow(request):
        fanout.current().found([3])
        await asyncio.sleep(5)

    async def broken(request):
        raise RuntimeError("graph broke")

    async def go():
        queue = notices.begin()
        return await gate.awrap_tool_call(_task_call(0), slow), await gate.awrap_tool_call(_task_call(1), broken), queue

    timed_out, failed, queue = asyncio.run(go())
    assert timed_out.status == "error" and timed_out.content == (
        "Error: the researcher did not finish within 0.05 s (RAG_DEEP_TASK_TIMEOUT_S), so it returned no findings. "
        "The sources it found stay in the source book: [3]. Research what is still missing yourself with the calls "
        "left.")
    assert failed.status == "error" and failed.content.startswith("Error: the researcher failed (RuntimeError) and "
                                                                  "returned no findings. It found no sources.")
    assert codes(queue) == ["research_task_failed", "research_task_failed"]
    assert queue[0]["message"] == ("A researcher ran past its time limit (0.05 s, RAG_DEEP_TASK_TIMEOUT_S) and was "
                                   "stopped; the sources it found are kept and the answer can use them.")
    assert "failed (RuntimeError)" in queue[1]["message"]
    assert run.fanout.active == [] and run.fanout.started == 2  # both ran; their leases are back
    assert any("notice research_task_failed" in r.getMessage() for r in caplog.records)  # logged
    assert any(r.getMessage() == "a researcher failed" for r in caplog.records)
    reported = [run.events.get_nowait()[1] for _ in range(run.events.qsize())]  # the task steps show the errors
    assert [x["tool"] for x in reported] == ["task", "task"] and all(x["error"] for x in reported)


# --- the tuned deepagents arm, scripted, with fan-out on ----------------------------------------------------------


ORDERS, TEXT = "Orders: Q3 revenue per region", "Report: what report.pdf says about Q3 revenue"


def test_both_researchers_get_their_calls_and_the_lead_the_rest():
    model = Routed(scripts={
        QUESTION: [todos(("Q3 revenue in orders.xlsx", "in_progress"), ("Q3 revenue in report.pdf", "in_progress")),
                   calls(task(ORDERS), task(TEXT)),
                   calls(("write_todos", {"todos": [{"content": "Q3 revenue in orders.xlsx", "status": "completed"},
                                                    {"content": "Q3 revenue in report.pdf", "status": "completed"}]},
                          None), sql("lead 1"), sql("lead 2"))],
        ORDERS: [call("sql", {"doc_id": TABLES.doc_id, "query": f"SELECT {i} FROM orders"}) for i in range(4)]
        + [done("South 2735.62 [1].")],
        TEXT: [call("search", {"query": f"Q3 revenue {i}"}) for i in range(4)] + [done("The report says 340 [2].")]})
    deps = make_deps(model, Settings())
    events, queue = run_deep(deps, fanout=True)
    research_steps = steps(events, {"sql", "search"})
    assert len(research_steps) == 10 and all(x["error"] is None for x in research_steps)  # none refused
    assert len(model.of(ORDERS)) == 5 and len(model.of(TEXT)) == 5  # 4 calls each, then the report
    timings = event(events, "done")["timings"]
    assert (timings["tool_calls"], timings["tasks"], timings["task_calls"], timings["tasks_refused"]) == (10, 2, 8, 0)
    reports = {m.tool_call_id: m.content for msgs, _ in model.of(QUESTION) for m in msgs if isinstance(m, ToolMessage)
               and m.name == "task"}
    assert sorted(reports.values()) == ["South 2735.62 [1].", "The report says 340 [2]."]
    assert codes(queue) == ["research_budget"]  # 10 of 10 used: the research ends, the answer uses the book
    assert {s["doc_name"] for s in event(events, "sources")["sources"]} == {"orders.xlsx", "report.pdf"}
    # the lead was told its budget and the cost of a researcher; the researchers their lease and the cap
    system = da.flat(model.of(QUESTION)[0][0][0]).content
    assert "You have 10 research calls for this question." in system and "reserves 4 of your calls" in system
    assert "Delegate first" in system and da.TASK_RULES not in system
    researcher = model.of(ORDERS)[0][0][0].content
    assert "You have 4 research calls. Your reply reaches the lead cut at 800 tokens" in researcher
    for key in (ORDERS, TEXT):  # no nesting: researchers never get task
        assert all("task" not in kwargs["tool_names"] and "sql" in kwargs["tool_names"] for _, kwargs in model.of(key))


def test_a_lead_that_spent_its_calls_first_cannot_start_researchers_that_would_have_none():
    """The live failure: the lead used the budget, then called task twice, and both researchers had every call
    refused. With fan-out the task calls are refused at once and no researcher starts."""
    model = Routed(scripts={QUESTION: [calls(*[sql(i) for i in range(7)]), calls(task(ORDERS), task(TEXT)), done()]})
    deps = make_deps(model)
    events, queue = run_deep(deps, fanout=True)
    refused = steps(events, {"task"})
    assert len(refused) == 2 and all(x["error"].startswith("Not run: a researcher reserves 4 research calls and only "
                                                           "3 are left of the 10") for x in refused)
    assert model.of(ORDERS) == [] and model.of(TEXT) == []  # no researcher started
    timings = event(events, "done")["timings"]
    assert (timings["tool_calls"], timings["tasks"], timings["tasks_refused"]) == (7, 0, 2)


def test_the_lead_cannot_spend_calls_reserved_in_the_same_reply_and_unused_calls_come_back():
    model = Routed(scripts={
        QUESTION: [calls(task(ORDERS), task(TEXT), sql("a"), sql("b"), sql("c")),
                   calls(*[sql(f"after {i}") for i in range(7)])],
        ORDERS: [call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT region FROM orders"}), done("South [1].")],
        TEXT: [done("Nothing on Q3 revenue.")]})
    deps = make_deps(model)
    events, queue = run_deep(deps, fanout=True)
    lead = [x for x in steps(events, {"sql"}) if x["args"]["query"] in ("SELECT a", "SELECT b", "SELECT c")]
    errors = [x["error"] for x in lead if x["error"]]
    assert len(errors) == 1 and errors[0] == fanout.RESERVED  # 8 reserved, 2 free: the third lead call waits
    # the researchers used 1 of their 8: 7 came back, and the lead's next reply used them all
    after = [x for x in steps(events, {"sql"}) if x["args"]["query"].startswith("SELECT after")]
    assert len(after) == 7 and all(x["error"] is None for x in after)
    timings = event(events, "done")["timings"]
    assert (timings["tool_calls"], timings["task_calls"]) == (10, 1)


def test_a_researcher_whose_lease_ran_out_reports_once_and_the_researcher_limit_holds():
    model = Routed(scripts={
        QUESTION: [calls(task(ORDERS)), calls(task(TEXT)), calls(task("Third")), done()],
        ORDERS: [calls(*[sql(f"o{i}") for i in range(6)]), call("sql", {"doc_id": TABLES.doc_id, "query": "x"}),
                 done("never asked")],
        TEXT: [done("Nothing.")]})
    deps = make_deps(model, Settings(deep_task_calls=3))
    events, queue = run_deep(deps, fanout=True)
    orders = [x for x in steps(events, {"sql"}) if x["args"]["query"].startswith(("SELECT o", "x"))]
    assert sum(x["error"] is None for x in orders) == 3  # its lease
    assert all(x["error"] == fanout.LEASE_USED.format(lease=3) for x in orders if x["error"])
    assert len(model.of(ORDERS)) == 2  # its calls ran out: one more call to report (it called a tool), then it ended
    third = steps(events, {"task"})[-1]
    assert third["error"].startswith("Not run: 2 researchers already ran for this question (RAG_DEEP_MAX_TASKS)")
    assert model.of("Third") == []
    first_report = next(m.content for msgs, _ in model.of(QUESTION) for m in msgs
                        if isinstance(m, ToolMessage) and m.name == "task")
    assert first_report == ("The researcher returned no findings text. The sources it found stay in the source book: "
                            "[1], [2], [3].")


def test_a_researcher_past_its_time_limit_ends_alone_and_the_run_answers_with_its_sources():
    model = Routed(scripts={
        QUESTION: [calls(task(ORDERS), task(TEXT))],
        ORDERS: [call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT region FROM orders"}), done("late")],
        TEXT: [call("search", {"query": "Q3 revenue"}), done("The report says 340 [2].")]},
        delay={ORDERS: 0.15})
    deps = make_deps(model, Settings(deep_task_timeout_s=0.25))
    events, queue = run_deep(deps, fanout=True)
    assert codes(queue).count("research_task_failed") == 1
    reports = {m.content for msgs, _ in model.of(QUESTION) for m in msgs if isinstance(m, ToolMessage) and m.name == "task"}
    assert "The report says 340 [2]." in reports
    late = next(r for r in reports if r.startswith("Error: the researcher did not finish within 0.25 s"))
    sql_source = next(x["sources"] for x in steps(events, {"sql"}))
    assert f"The sources it found stay in the source book: [{sql_source[0]}]." in late
    assert {s["doc_name"] for s in event(events, "sources")["sources"]} == {"orders.xlsx", "report.pdf"}
    assert event(events, "done")["finish_reason"] == "stop"


def test_without_fanout_nothing_changes_the_lead_spends_the_shared_pool_first():
    """The default (RAG_DEEP_SUBAGENTS on, fan-out off): the pool is first come, first served, as before."""
    model = Routed(scripts={QUESTION: [calls(*[sql(i) for i in range(10)]), calls(task(ORDERS))],
                            ORDERS: [call("sql", {"doc_id": TABLES.doc_id, "query": "SELECT 1 FROM orders"})]})
    deps = make_deps(model, Settings(deep_subagents=True))
    events, queue = run_deep(deps)
    assert model.of(ORDERS) == []  # the budget was used: the run ended before the lead could call task
    timings = event(events, "done")["timings"]
    assert "tasks" not in timings and timings["tool_calls"] == 10
    system = da.flat(model.of(QUESTION)[0][0][0]).content
    assert da.TASK_RULES in system and "You have 10 research calls" not in system


# --- prompt size in real tokens -----------------------------------------------------------------------------------


@pytest.mark.skipif(not TOKENIZER, reason="no Qwen3 tokenizer.json on this machine")
def test_the_first_prompt_with_fanout_in_real_tokens():
    """The tuned arm's first prompt, tool schemas included, with fan-out on against sub-agents on and both off
    (measured: 2,247 off, 2,535 with sub-agents, 2,613 with fan-out, two documents selected)."""
    assert tokens.configure(TOKENIZER, 1.05)

    class Capturing(Routed):
        bound: list = Field(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            self.bound = tools
            return super().bind_tools(tools, **kwargs)

    sizes = {}
    for label, settings, fields in (("off", Settings(), {}), ("subagents", Settings(deep_subagents=True), {}),
                                    ("fanout", Settings(), {"fanout": True})):
        model = Capturing()
        run_deep(make_deps(model, settings), **fields)
        messages, _ = model.of(QUESTION)[0]
        sizes[label] = da.request_tokens(da.flat(messages[0]), messages[1:], model.bound)
    s = Settings()
    assert sizes["off"] < sizes["subagents"] < sizes["fanout"] < sizes["subagents"] + 250
    assert sizes["fanout"] + s.research_retry_tokens + s.prompt_margin_tokens + 20000 < s.slot_tokens
