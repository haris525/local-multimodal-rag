"""Exhaustive mode (rag.exhaustive): reading order, packing, caps and their notices, hierarchical reduce, citation
mapping and checking, failures, cancellation, owner and document scoping, and the request field."""

import asyncio
import contextlib
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from conftest import KEYS
from rag_common.chat_events import check_order
from rag_common.identity import Identity, current_identity
from rag_common.index import COLLECTION, TEXT_VECTOR, ChunkPayload, point_id
from rag_common.testing import new_user_id

import rag.api as api
from rag import exhaustive, notices
from rag.chat import ChatRequest, request_problem
from rag.config import Settings
from rag.conversation import Conversation
from rag.exhaustive import Note, exhaustive_answer, fair_cut, parse_notes, passages_of, plan
from rag.retrieval import Retrieval
from rag.turn import respond

DOC_A, DOC_B, DOC_C = "a" * 32, "b" * 32, "c" * 32
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 28  # filler: one passage is over 500 tokens


def chunk(doc: str, i: int, text: str, section: str = "", page: int | None = None, kind: str = "text",
          sheet: str | None = None, owner: str | None = None) -> tuple[str, ChunkPayload]:
    return point_id(doc, i), ChunkPayload(doc_id=doc, doc_name=f"{doc[0]}.pdf", doc_type="pdf", kind=kind,
                                          chunk_index=i, text=text, search_text=text, text_sha="0" * 16,
                                          n_tokens=len(text.split()), section=section, page=page, sheet=sheet,
                                          user_id=owner)


class Chunks:
    """Retrieval with only document_chunks, over a fixed list (already in reading order)."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.asked: list = []

    async def document_chunks(self, doc_ids, skip_kinds=()):
        self.asked.append((list(doc_ids), tuple(skip_kinds)))
        return [c for c in self.chunks if c[1].doc_id in doc_ids and c[1].kind not in skip_kinds]


def numbers_in(messages) -> list[int]:
    user = messages[-1]["content"]
    return [int(n) for n in re.findall(r"^\[(\d+)\] ", user, re.M)]


class Model:
    """A fake llm. Map: one note per passage whose text has `keyword`, citing it (plus `extra` lines). Merge: keeps
    every other note (a real merge combines duplicates). The answer cites `answer`."""

    def __init__(self, keyword="limitation", answer="Two limits [1] and [99].", extra="", fail=(), cut=(),
                 merge=None):
        self.keyword, self.answer, self.extra, self.fail, self.cut, self.merge = keyword, answer, extra, fail, cut, merge
        self.calls: list[tuple[str, list[dict]]] = []
        self.streamed: list[dict] | None = None

    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        self.calls.append((purpose, messages))
        user = messages[-1]["content"]
        if purpose == "exhaustive map":
            index = sum(1 for p, _ in self.calls if p == purpose)
            if index in self.fail:
                raise RuntimeError("llm: busy")
            blocks = re.findall(r"^\[(\d+)\] [^\n]*\n([^\n]*)", user, re.M)
            lines = [f"- The text of passage {n} names a limitation [{n}]" for n, text in blocks if self.keyword in text]
            content = "\n".join(lines + ([self.extra] if self.extra else [])) or "NONE"
            return {"content": content, "finish_reason": "length" if index in self.cut else "stop"}
        if purpose == "exhaustive merge":
            if self.merge:
                return self.merge(user)
            notes = [line for line in user.split("Notes:\n", 1)[1].split("\n\nQuestion:")[0].splitlines()]
            return {"content": "\n".join(notes[::2]), "finish_reason": "stop"}
        raise AssertionError(purpose)

    async def stream(self, messages, max_tokens, think):
        self.streamed = messages
        yield "token", self.answer
        yield "usage", {"prompt_tokens": 100, "completion_tokens": 10}
        yield "finish", "stop"


def run(deps, req: ChatRequest, conv: Conversation | None = None):
    async def go():
        queue = notices.begin()
        events = [e async for e in exhaustive_answer(deps, req, [], conv or Conversation(), req.question, {})]
        return events, queue
    return asyncio.run(go())


def deps_with(chunks, model=None, **settings):
    return SimpleNamespace(s=Settings(**settings), llm=model or Model(), retrieval=Chunks(chunks))


def request(*docs) -> ChatRequest:
    return ChatRequest(question="List every limitation.", doc_ids=list(docs or [DOC_A]), mode="exhaustive")


def of(events, name):
    return [data for kind, data in events if kind == name]


# --- reading order, sections and packing -------------------------------------------------------------------------


def test_sections_follow_the_reading_order_and_small_sections_are_packed_up_to_the_budget():
    chunks = [chunk(DOC_A, i, f"{WORDS} part {i}", section=f"S{i // 2}", page=i + 1) for i in range(6)]
    passages = passages_of(chunks)
    one = passages[0].tokens
    p = plan(passages, budget=4 * one + 1, max_sections=100, max_tokens=10**6)
    assert [s.title for s in p.sections] == ["S0 (pp. 1-2)", "S1 (pp. 3-4)", "S2 (pp. 5-6)"]
    assert [[psg.n for psg in g.passages] for g in p.groups] == [[1, 2, 3, 4], [5, 6]]
    assert [g.progress(3) for g in p.groups] == ["1-2 of 3", "3 of 3"]
    assert p.groups[0].where() == "a.pdf: S0 (pp. 1-2), S1 (pp. 3-4)"
    assert p.cap is None and p.unread == [] and p.read_sections() == 3


def test_a_section_larger_than_one_map_call_is_split_between_passages():
    passages = passages_of([chunk(DOC_A, i, f"{WORDS} {i}", section="Results") for i in range(5)])
    p = plan(passages, budget=2 * passages[0].tokens + 1, max_sections=100, max_tokens=10**6)
    assert [[psg.n for psg in g.passages] for g in p.groups] == [[1, 2], [3, 4], [5]]
    assert [g.progress(1) for g in p.groups] == ["1 of 1, part 1 of 3", "1 of 1, part 2 of 3", "1 of 1, part 3 of 3"]


def test_a_passage_longer_than_one_map_call_is_read_up_to_that_size():
    passages = passages_of([chunk(DOC_A, 0, WORDS * 20, section="Big"), chunk(DOC_A, 1, "small", section="Big")])
    p = plan(passages, budget=200, max_sections=100, max_tokens=10**6)
    assert p.cut_passages == [passages[0]] and passages[0].tokens <= 200 and passages[0].block.endswith("[cut]")


def test_the_whole_turn_reads_documents_in_the_selected_order_and_numbers_passages_across_them():
    chunks = [chunk(DOC_B, 0, "b limitation one", section="Intro", page=1),
              chunk(DOC_A, 0, "a limitation one", section="Intro", page=1),
              chunk(DOC_A, 1, "a nothing here", section="Method", page=2)]
    deps = deps_with([chunks[1], chunks[2], chunks[0]], model=Model(answer="See [1] and [3]."))
    events, queue = run(deps, request(DOC_A, DOC_B))
    assert deps.retrieval.asked == [([DOC_A, DOC_B], ("row", "summary"))]  # rows and the summary are not read
    [sources] = of(events, "sources")
    assert [(s["n"], s["doc_id"]) for s in sources["sources"]] == [(1, DOC_A), (3, DOC_B)]
    assert sources["route"] == "exhaustive" and sources["per_doc"] == {"a.pdf": 1, "b.pdf": 1}
    prompt = deps.llm.streamed[-1]["content"][0]["text"]
    assert "Passage numbers by document: [1]-[2] a.pdf; [3] b.pdf." in prompt
    assert "Every one of the 3 sections of the selected documents was read." in prompt
    check_order([name for name, _ in events])
    assert queue == []


# --- map: notes, citations and steps ------------------------------------------------------------------------------


def test_notes_keep_only_citations_of_their_own_passages_and_a_note_without_one_is_dropped():
    text = ("- First limit [3][9]\n- Second limit [4, 5]\n* Range [3-5]\n1. Outside only [42]\n- No citation at all\n"
            "From a.pdf:\nNONE")
    notes, uncited = parse_notes(text, {3, 4, 5}, group=0)
    assert [(n.text, n.numbers) for n in notes] == [("First limit [3]", [3]), ("Second limit [4][5]", [4, 5]),
                                                     ("Range [3][4][5]", [3, 4, 5])]
    assert uncited == 2  # the bullets citing nothing of theirs; the heading line is not a note
    assert parse_notes("NONE", {1}, 0) == ([], 0)
    cut, _ = parse_notes("- whole [1]\n- stopped mid-wa", {1}, 0, cut=True)
    assert [n.text for n in cut] == ["whole [1]"]


def test_map_steps_report_progress_and_the_passages_cited_and_the_answer_cites_only_the_notes():
    chunks = [chunk(DOC_A, i, f"{'limitation' if i % 2 else 'plain'} {WORDS}", section=f"S{i}", page=i + 1)
              for i in range(4)]
    model = Model(answer="Limits: [2], [4] and [3].", extra="- Invented [77]")
    deps = deps_with(chunks, model, exhaustive_group_tokens=2 * passages_of(chunks)[0].tokens + 1)
    events, queue = run(deps, request())
    steps = of(events, "step")
    assert steps[0]["tool"] == "sections" and steps[0]["args"]["plan"] == "4 of 4 sections of 1 documents in 2 map calls"
    assert [(s["tool"], s["args"]["sections"], s["sources"]) for s in steps[1:]] == [("map", "1-2 of 4", [2]),
                                                                                    ("map", "3-4 of 4", [4])]
    [done] = of(events, "done")
    assert done["citations"] == [2, 4] and done["invalid_citations"] == [3]  # [3] was read, but no note cites it
    assert [s["n"] for s in of(events, "sources")[0]["sources"]] == [2, 4]
    assert done["timings"]["map_calls"] == 2 and done["timings"]["tool_calls"] == 2
    assert [n["code"] for n in queue] == ["exhaustive_notes_uncited"]  # the two "Invented [77]" lines


def test_map_concurrency_is_bounded_and_results_stay_in_reading_order():
    running, peak = 0, 0

    class Slow(Model):
        async def complete(self, messages, max_tokens, **kw):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01 if numbers_in(messages)[0] == 1 else 0.001)  # the first call finishes last
            running -= 1
            return await super().complete(messages, max_tokens, **kw)

    chunks = [chunk(DOC_A, i, f"limitation {WORDS}", section=f"S{i}") for i in range(4)]
    budget = passages_of(chunks)[0].tokens + 1
    for concurrency, want in ((1, 1), (2, 2)):
        peak = 0
        deps = deps_with(chunks, Slow(answer="[1][2][3][4]"), exhaustive_group_tokens=budget,
                         exhaustive_concurrency=concurrency)
        events, _ = run(deps, request())
        assert peak == want
        notes = deps.llm.streamed[-1]["content"][0]["text"]
        where = [notes.index(f"names a limitation [{n}]") for n in range(1, 5)]
        assert where == sorted(where)  # the first call finished last, and its notes still come first
    with pytest.raises(ValidationError):
        Settings(exhaustive_concurrency=3)  # llama-server has two slots


def test_nothing_relevant_in_any_section_is_said_plainly():
    deps = deps_with([chunk(DOC_A, 0, "nothing of note", section="S")])
    events, _ = run(deps, request())
    assert "found nothing" in of(events, "token")[0]["text"] and of(events, "done")[0]["finish_reason"] == "no_sources"
    check_order([name for name, _ in events])


# --- caps ---------------------------------------------------------------------------------------------------------


def test_the_section_cap_stops_reading_and_says_what_was_not_read():
    chunks = ([chunk(DOC_A, i, f"limitation {i}", section=f"A{i}", page=i + 1) for i in range(3)]
              + [chunk(DOC_B, i, f"limitation b{i}", section=f"B{i}") for i in range(2)]
              + [chunk(DOC_C, 0, "limitation c", section="C0")])
    deps = deps_with(chunks, Model(answer="[1]"), exhaustive_max_sections=2)
    events, queue = run(deps, request(DOC_A, DOC_B, DOC_C))
    [cap] = [n for n in queue if n["code"] == "exhaustive_budget"]
    assert "limit of 2 sections (RAG_EXHAUSTIVE_MAX_SECTIONS)" in cap["message"]
    assert "it read 2 of the 6 sections. Not read: a.pdf from A2 (p. 3) on; b.pdf; c.pdf." in cap["message"]
    asked = {n for purpose, m in deps.llm.calls if purpose == "exhaustive map" for n in numbers_in(m)}
    assert asked == {1, 2}
    prompt = deps.llm.streamed[-1]["content"][0]["text"]
    assert "Not read (the request's reading limit): a.pdf from A2 (p. 3) on; b.pdf; c.pdf." in prompt
    assert of(events, "done")[0]["timings"]["sections_read"] == 2


def test_the_token_cap_can_stop_inside_a_section():
    chunks = [chunk(DOC_A, i, f"limitation {WORDS} {i}", section="Long") for i in range(4)]
    one = passages_of(chunks)[0].tokens
    deps = deps_with(chunks, Model(answer="[1]"), exhaustive_group_tokens=one + 1, exhaustive_max_input_tokens=2 * one + 5)
    _, queue = run(deps, request())
    [cap] = [n for n in queue if n["code"] == "exhaustive_budget"]
    assert "RAG_EXHAUSTIVE_MAX_INPUT_TOKENS" in cap["message"] and "a.pdf from Long, part 3 of 4 on" in cap["message"]
    assert sum(1 for purpose, _ in deps.llm.calls if purpose == "exhaustive map") == 2


def test_large_sheets_are_read_through_their_profile_and_the_user_is_told():
    chunks = [(point_id(DOC_A, 0), ChunkPayload(doc_id=DOC_A, doc_name="s.xlsx", doc_type="spreadsheet",
                                                kind="sheet_summary", chunk_index=0, text='Sheet "Big": 5000 rows. limitation',
                                                search_text="x", text_sha="0" * 16, n_tokens=5, sheet="Big")),
              (point_id(DOC_A, 1), ChunkPayload(doc_id=DOC_A, doc_name="s.xlsx", doc_type="spreadsheet",
                                                kind="sheet_summary", chunk_index=1, text='Sheet "Small": 3 rows',
                                                search_text="x", text_sha="0" * 16, n_tokens=5, sheet="Small")),
              (point_id(DOC_A, 2), ChunkPayload(doc_id=DOC_A, doc_name="s.xlsx", doc_type="spreadsheet", kind="table",
                                                chunk_index=2, text="| a | b |", search_text="x", text_sha="0" * 16,
                                                n_tokens=5, sheet="Small"))]
    deps = deps_with(chunks, Model(answer="[1]"))
    events, queue = run(deps, request())
    [note] = [n for n in queue if n["code"] == "exhaustive_sheet_profiles"]
    assert '"Big" (s.xlsx)' in note["message"] and "Small" not in note["message"]
    assert of(events, "step")[0]["args"]["plan"].startswith("2 of 2 sections")  # one per sheet, rows not read


# --- failures -----------------------------------------------------------------------------------------------------


def test_a_failed_map_call_costs_only_its_sections_and_is_a_notice():
    chunks = [chunk(DOC_A, i, f"limitation {WORDS}", section=f"S{i}") for i in range(3)]
    deps = deps_with(chunks, Model(answer="[1][3]", fail={2}), exhaustive_group_tokens=passages_of(chunks)[0].tokens + 1)
    events, queue = run(deps, request())
    maps = [s for s in of(events, "step") if s["tool"] == "map"]
    assert [s["error"] for s in maps] == [None, "RuntimeError: llm: busy", None]
    [failed] = [n for n in queue if n["code"] == "exhaustive_map_failed"]
    assert "1 of the 3 parts failed" in failed["message"] and "a.pdf: S1" in failed["message"]
    assert "Not read (the model call failed): a.pdf: S1." in deps.llm.streamed[-1]["content"][0]["text"]
    assert of(events, "done")[0]["citations"] == [1, 3]


def test_every_map_call_failing_ends_the_turn_with_an_error():
    deps = deps_with([chunk(DOC_A, 0, "limitation", section="S")], Model(fail={1}))
    with pytest.raises(RuntimeError, match="every map call failed"):
        run(deps, request())


def test_a_map_reply_still_cut_after_its_retry_is_a_notice():
    deps = deps_with([chunk(DOC_A, 0, "limitation", section="S"), chunk(DOC_A, 1, "limitation", section="S")],
                     Model(answer="[1]", extra="- half a no", cut={1}))
    events, queue = run(deps, request())
    assert any(n["code"] == "exhaustive_map_cut" for n in queue)
    assert of(events, "done")[0]["citations"] == [1]


# --- reduce -------------------------------------------------------------------------------------------------------


def test_notes_that_do_not_fit_are_merged_in_rounds_until_they_do():
    chunks = [chunk(DOC_A, i, f"limitation {i}", section=f"S{i}") for i in range(40)]
    model = Model(answer="[1]")
    deps = deps_with(chunks, model, exhaustive_notes_tokens=500)
    events, queue = run(deps, request())
    reduces = [s for s in of(events, "step") if s["tool"] == "reduce"]
    assert reduces and reduces[0]["args"]["merge"].startswith("round 1, batch 1 of ")
    merged = [m for purpose, m in model.calls if purpose == "exhaustive merge"]
    assert merged and "From " not in model.streamed[-1]["content"][0]["text"]  # merged notes lose their headings
    timings = of(events, "done")[0]["timings"]
    assert timings["reduce_rounds"] >= 1 and timings["merge_calls"] == len(merged)
    assert timings["tool_calls"] == timings["map_calls"] + timings["merge_calls"]
    assert not any(n["code"] == "exhaustive_notes_cut" for n in queue)
    assert exhaustive.notes_tokens([], {}) == 0


def test_notes_that_merging_cannot_shrink_are_cut_fairly_with_a_notice():
    chunks = [chunk(DOC_A, i, f"limitation {i}", section=f"S{i}") for i in range(40)]
    same = lambda user: {"content": user.split("Notes:\n", 1)[1].split("\n\nQuestion:")[0], "finish_reason": "stop"}
    deps = deps_with(chunks, Model(answer="[1]", merge=same), exhaustive_notes_tokens=500)
    events, queue = run(deps, request())
    [cut] = [n for n in queue if n["code"] == "exhaustive_notes_cut"]
    assert "taken evenly from every part" in cut["message"] and "RAG_EXHAUSTIVE_NOTES_TOKENS" in cut["message"]
    assert 0 < of(events, "done")[0]["timings"]["notes"] < 40


def test_a_failed_merge_keeps_its_notes_and_a_cut_merge_carries_over_what_it_did_not_reach():
    chunks = [chunk(DOC_A, i, f"limitation {i}", section=f"S{i}") for i in range(30)]

    def broken(user):
        raise RuntimeError("llm: down")

    deps = deps_with(chunks, Model(answer="[1]", merge=broken), exhaustive_notes_tokens=500, exhaustive_reduce_rounds=1)
    _, queue = run(deps, request())
    assert {"exhaustive_reduce_failed", "exhaustive_notes_cut"} <= {n["code"] for n in queue}

    async def merge_cut():
        deps = deps_with([], Model(merge=lambda user: {"content": "- merged [1][2]\n- stopped", "finish_reason": "length"}))
        batch = [Note("one [1]", [1], 0), Note("two [2]", [2], 0), Note("three [3]", [3], 1)]
        return await exhaustive.merge_batch(deps, "q", "", batch, 7)
    got = asyncio.run(merge_cut())
    assert [(n.text, n.group) for n in got.notes] == [("merged [1][2]", 7), ("three [3]", 7)] and got.cut


def test_fair_cut_takes_notes_from_every_group_in_turn_and_keeps_their_order():
    notes = [Note(f"a{i} [{i}]", [i], 0) for i in range(1, 6)] + [Note("b [9]", [9], 1)]
    kept = fair_cut(notes, {}, budget=exhaustive.note_cost(notes[0]) * 2 + 2)
    assert [n.text for n in kept] == ["a1 [1]", "b [9]"]


# --- cancellation -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("concurrency", [1, 2])
def test_a_client_that_goes_away_cancels_the_map_calls_still_running(concurrency):
    started, cancelled = [], []

    class Hanging(Model):
        async def complete(self, messages, max_tokens, **kw):
            started.append(numbers_in(messages))
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(numbers_in(messages))
                raise

    chunks = [chunk(DOC_A, i, f"limitation {WORDS}", section=f"S{i}") for i in range(5)]
    deps = deps_with(chunks, Hanging(), exhaustive_group_tokens=passages_of(chunks)[0].tokens + 1,
                     exhaustive_concurrency=concurrency)

    async def scenario():
        notices.begin()
        events = []

        async def consume():
            async for event in exhaustive_answer(deps, request(), [], Conversation(), "q", {}):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        task.cancel()  # what Starlette does when the client disconnects
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        return events

    events = asyncio.run(scenario())
    assert len(started) == concurrency and sorted(cancelled) == sorted(started)  # nothing further started
    assert [data["tool"] for _, data in events] == ["sections"]


# --- scoping ------------------------------------------------------------------------------------------------------


def test_reading_whole_documents_is_scoped_to_the_owner_and_the_selected_documents():
    a, b = new_user_id(), new_user_id()

    async def scenario(user):
        q = AsyncQdrantClient(location=":memory:")
        await q.create_collection(COLLECTION, vectors_config={TEXT_VECTOR: qm.VectorParams(size=2, distance=qm.Distance.COSINE)})
        rows = ([chunk(DOC_A, i, f"a{i}", owner=a) for i in (2, 0, 1)] + [chunk(DOC_C, 0, "a's other doc", owner=a)]
                + [chunk(DOC_A, 3, "a row", kind="row", owner=a), chunk(DOC_A, 4, "a summary", kind="summary", owner=a)]
                + [chunk(DOC_B, i, f"b{i}", owner=b) for i in range(2)])
        await q.upsert(COLLECTION, [qm.PointStruct(id=pid, vector={TEXT_VECTOR: [1.0, 0.5]},
                                                   payload=p.model_dump(exclude_none=True)) for pid, p in rows], wait=True)
        retrieval = Retrieval(q, None, None, api.settings)
        token = current_identity.set(Identity(user, "user", "auth"))
        try:
            return [(p.doc_id, p.chunk_index) for _, p in
                    await retrieval.document_chunks([DOC_B, DOC_A], exhaustive.SKIP_KINDS, page=2)]
        finally:
            current_identity.reset(token)

    assert asyncio.run(scenario(a)) == [(DOC_A, 0), (DOC_A, 1), (DOC_A, 2)]  # B's document and rows: nothing
    assert asyncio.run(scenario(b)) == [(DOC_B, 0), (DOC_B, 1)]


# --- the request ------------------------------------------------------------------------------------------------


def test_the_request_takes_mode_exhaustive_and_refuses_deep_search_fields_with_it():
    assert ChatRequest(question="q", doc_ids=[DOC_A]).mode == "single"  # the default is unchanged
    req = ChatRequest(question="q", doc_ids=[DOC_A], mode="exhaustive")
    assert request_problem(api.settings, req) is None
    with pytest.raises(ValidationError):
        ChatRequest(question="q", doc_ids=[DOC_A], mode="everything")
    bad = ChatRequest(question="q", doc_ids=[DOC_A], mode="exhaustive", research_calls=5, deep_think=True)
    assert request_problem(api.settings, bad) == ("exhaustive mode does not use the Deep search settings deep_think, "
                                                 "research_calls; leave them out")
    python = ChatRequest(question="q", doc_ids=[DOC_A], mode="exhaustive", python=True)  # a Deep search tool
    assert "python" in request_problem(api.settings, python)
    r = TestClient(api.app).post("/v1/chat", json={"question": "q", "doc_ids": [DOC_A], "mode": "exhaustive",
                                                   "fanout": True}, headers=KEYS.headers(new_user_id()))
    assert r.status_code == 422 and "fanout" in r.text


def test_a_stateless_turn_in_exhaustive_mode_reads_every_section():
    deps = deps_with([chunk(DOC_A, 0, "limitation", section="S")], Model(answer="[1]"))
    req = request()

    async def go():
        notices.begin()
        return [e async for e in respond(deps, req, [], Conversation())]

    events = asyncio.run(go())
    assert of(events, "sources")[0]["route"] == "exhaustive" and of(events, "done")[0]["citations"] == [1]
