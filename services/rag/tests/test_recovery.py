"""Limits and recovery: a cut reply is retried once with more room, a busy or failed call once after a backoff,
and every fallback reaches the user as a notice event in the stream (saved with the answer)."""

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from rag_common.chat_events import read_chat
from rag_common.http import MappedTransport
from rag_common.contracts import ColumnSchema, DocStatus, DocType, DocumentInfo, SheetLayout, ThreadContext, TableSchema

import rag.api as api
from rag import notices
from rag.chat import ChatRequest, figure_images, stream_answer
from rag.clients import LLMClient, ModelsClient, RetryPolicy, ServiceError, SqlClient
from rag.config import Settings
from rag.conversation import Conversation
from rag.planner import Planner, schema_cards
from rag.prompts import Source

FAST = RetryPolicy(backoff_s=0.0, tokens_factor=2.5, max_tokens=4096)


def completion(content: str, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}]}


def client(responses: list, retry: RetryPolicy = FAST) -> tuple[LLMClient, list[dict]]:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    http = httpx.AsyncClient(base_url="http://llm:8080", transport=httpx.MockTransport(handler))
    return LLMClient(http, "m", {}, {}, retry=retry), sent


def test_a_cut_reply_is_retried_once_with_more_room(caplog):
    llm, sent = client([(200, completion('{"route": "sq', "length")), (200, completion('{"route": "search"}'))])
    with caplog.at_level(logging.WARNING):
        msg = asyncio.run(llm.complete([{"role": "user", "content": "q"}], max_tokens=400, purpose="SQL planner"))
    assert msg["content"] == '{"route": "search"}' and msg["finish_reason"] == "stop"
    assert [b["max_tokens"] for b in sent] == [400, 1000]
    assert "SQL planner: reply cut at its token limit; retrying once with more room" in caplog.text


def test_a_reply_still_cut_after_the_retry_keeps_finish_reason_length():
    llm, sent = client([(200, completion("a", "length")), (200, completion("ab", "length"))])
    msg = asyncio.run(llm.complete([{"role": "user", "content": "q"}], max_tokens=3000))
    assert msg["finish_reason"] == "length" and [b["max_tokens"] for b in sent] == [3000, 4096]


def test_a_busy_llm_is_retried_once_after_the_backoff_and_a_bad_request_is_not(caplog):
    llm, sent = client([(503, {"detail": "busy"}), (200, completion("ok"))])
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(llm.complete([{"role": "user", "content": "q"}], 10))["content"] == "ok"
    assert len(sent) == 2 and "failed; retrying once" in caplog.text
    llm, sent = client([(400, {"detail": "bad"}), (200, completion("never"))])
    with pytest.raises(ServiceError):
        asyncio.run(llm.complete([{"role": "user", "content": "q"}], 10))
    assert len(sent) == 1


def test_the_answer_stream_retries_only_before_its_first_token():
    ok = 'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n'
    replies = [(503, None), (200, ok)]

    def handler(request):
        status, text = replies.pop(0)
        return httpx.Response(status, text=text or "busy")

    llm = LLMClient(httpx.AsyncClient(base_url="http://llm:8080", transport=httpx.MockTransport(handler)), "m", {}, {},
                    retry=FAST)

    async def run():
        return [e async for e in llm.stream([{"role": "user", "content": "q"}], 10, False)]

    assert asyncio.run(run()) == [("token", "Hi"), ("finish", "stop")]


def test_models_busy_is_retried_and_a_sql_timeout_is_not():
    calls = {"models": 0, "sql": 0}

    def models(request):
        calls["models"] += 1
        return httpx.Response(503, json={"detail": "queue full"}) if calls["models"] == 1 else \
            httpx.Response(200, json={"vectors": [[1.0]], "dim": 1, "instruction": ""})

    def sql(request):
        calls["sql"] += 1
        return httpx.Response(408, json={"detail": "the query took longer than 10 s"})

    m = ModelsClient(httpx.AsyncClient(base_url="http://m", transport=httpx.MockTransport(models)), FAST)
    s = SqlClient(httpx.AsyncClient(base_url="http://s", transport=httpx.MockTransport(sql)), FAST)
    assert asyncio.run(m.embed_query("q", "")) == [1.0] and calls["models"] == 2
    with pytest.raises(ServiceError):
        asyncio.run(s.query("d" * 32, "SELECT 1"))
    assert calls["sql"] == 1


def test_a_sql_read_timeout_is_not_retried_but_a_busy_sandbox_and_a_connect_timeout_are():
    """M9: rag's read timeout means the first query may still run in the sandbox; a second copy would crowd it."""
    def sandbox(fail):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                if isinstance(fail, int):
                    return httpx.Response(fail, json={"detail": "every query slot is busy"})
                raise fail("timed out", request=request)
            return httpx.Response(200, json={"columns": ["n"], "rows": [[1]], "truncated": False})

        transport = MappedTransport(httpx.MockTransport(handler))  # as make_client builds it
        return SqlClient(httpx.AsyncClient(base_url="http://s", transport=transport), FAST), calls

    for fail, retried in ((503, True), (httpx.ConnectTimeout, True), (httpx.ReadTimeout, False)):
        client_, calls = sandbox(fail)
        if retried:
            assert asyncio.run(client_.query("d" * 32, "SELECT 1")).rows == [[1]] and len(calls) == 2
        else:
            with pytest.raises(ServiceError) as err:
                asyncio.run(client_.query("d" * 32, "SELECT 1"))
            assert len(calls) == 1 and (err.value.status, err.value.cause) == (504, "ReadTimeout")
    assert Settings().sql_timeout_s >= 2 * 20 + 5  # the sandbox's own timeout and queue wait answer first


# --- planner ---------------------------------------------------------------------------------------------------

CENSUS = DocumentInfo(
    doc_id="c" * 32, name="census.xlsx", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
    created_at="t", updated_at="t",
    tables=[TableSchema(table="nst_est2025_pop", source='sheet "NST-EST2025-POP"', sheet="NST-EST2025-POP", rows=57,
                        row_filter=True, title="Annual Estimates of the Resident Population",
                        notes="table with row headers in column A. (leading dots indicate sub-parts)",
                        layout=SheetLayout(header_rows=[3, 4], data_rows=[5, 62], raw_table="nst_est2025_pop__raw"),
                        columns=[ColumnSchema(name="Geographic Area", sql_name="geographic_area", type="text",
                                              profile="57 distinct"),
                                 ColumnSchema(name="Solar", sql_name="solar", type="number", profile="min 1, max 2",
                                              header="Electricity Net Generation From Solar, All Sectors",
                                              unit="Million Kilowatthours")])])


def test_schema_card_shows_title_notes_full_header_and_unit():
    card = schema_cards([CENSUS], "growth")
    assert "  title: Annual Estimates of the Resident Population" in card
    assert "leading dots indicate sub-parts" in card
    assert "  - solar (number, Million Kilowatthours) [Electricity Net Generation From Solar, All Sectors]: min 1" in card
    assert "  - geographic_area (text): 57 distinct" in card  # no [label] when the name says the same


class ScriptedLLM:
    def __init__(self, *replies: dict):
        self.replies = list(replies)

    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return self.replies.pop(0)


def test_a_cut_plan_falls_back_to_search_and_tells_the_user():
    async def run():
        queue = notices.begin()
        plan = await Planner(ScriptedLLM({"content": '{"route": "sql", "sql": "SELECT', "finish_reason": "length"}),
                             None, 1500).plan("Which state grew fastest?", [CENSUS])
        return plan, queue

    plan, queue = asyncio.run(run())
    assert (plan.route, plan.detail) == ("search", "planner unavailable")
    assert queue[0]["code"] == "planner_failed" and "cut off at its length limit" in queue[0]["message"]


# --- stream, figures, notices in the protocol -------------------------------------------------------------------


class Ingest:
    def __init__(self, missing: set[str] = frozenset()):
        self.fetched: list[str] = []
        self.missing = missing

    async def asset_data_uri(self, doc_id, name):
        self.fetched.append(name)
        return None if name in self.missing else f"data:image/png;base64,{name}"


def test_each_figure_is_attached_once_and_a_missing_one_is_a_notice():
    sources = [Source(1, "a", "d", "chart.png", "image", "image", "part 1", 10, "fig.png"),
               Source(2, "b", "d", "chart.png", "image", "image", "part 2", 10, "fig.png"),
               Source(3, "c", "e", "other.pdf", "image", "figure", "x", 10, "gone.png")]
    deps = SimpleNamespace(s=Settings(), ingest=Ingest({"gone.png"}))

    async def run():
        queue = notices.begin()
        return await figure_images(deps, sources), queue

    images, queue = asyncio.run(run())
    assert list(images) == [1] and deps.ingest.fetched == ["fig.png", "gone.png"]
    assert [n["code"] for n in queue] == ["figure_missing"]


class StreamLLM:
    async def stream(self, messages, max_tokens, think):
        yield "token", "Half an answer"
        yield "finish", "length"


def _saved_turn(extra_notice: str | None = None):
    async def scenario():
        audit = SimpleNamespace(events=[])

        async def record(action, subject=None, actor="local", **detail):
            audit.events.append((action, detail))

        audit.record = record
        appended = []

        class Threads:
            async def append(self, thread_id, messages, doc_ids):
                appended.append(messages)

            async def context(self, thread_id):
                return ThreadContext(summary="", summary_upto=0, messages=[])

            async def memories(self):
                return []

        deps = SimpleNamespace(s=Settings(), llm=StreamLLM(), ingest=Ingest(), threads=Threads())
        api.state.update(audit=audit, deps=deps)
        sources = [Source(1, "a", "d", "doc.pdf", "text", "p. 1", "text", 10)]

        async def events():
            if extra_notice:
                notices.notice("rewrite_cut", extra_notice)
            yield "sources", {"route": "search", "sources": []}
            async for event in stream_answer(deps, "q", sources, False, {}, Conversation()):
                yield event

        req = ChatRequest(question="q", doc_ids=["d" * 32], thread_id="a" * 32)
        text = "".join([chunk async for chunk in api._sse(events(), req, [])])
        await asyncio.gather(*api._background)
        return text, appended, audit

    return asyncio.run(scenario())


def test_an_answer_cut_at_its_limit_is_a_notice_before_done_and_is_saved_with_the_answer():
    text, appended, audit = _saved_turn()
    reply = read_chat(text.splitlines())
    assert reply.complete and [n["code"] for n in reply.notices] == ["answer_cut"]
    names = [name for name, _ in reply.events]
    assert names.index("notice") == names.index("done") - 1
    assert appended[0][1].meta["notices"][0]["code"] == "answer_cut"
    assert appended[0][1].meta["doc_ids"] == ["d" * 32]  # Retry asks again with the turn's own documents
    assert audit.events[0][1]["notices"] == ["answer_cut"]


def test_a_notice_quoting_a_credential_is_saved_redacted_and_logged_without_its_message(caplog):
    """Notices were logged with their message and saved as they were, while the question
    and the answer are redacted before they are saved."""
    with caplog.at_level(logging.DEBUG):
        _, appended, _ = _saved_turn("The question 'remember my password is hunter2' was cut to fit the search")
    saved = json.dumps(appended[0][1].meta["notices"])
    assert "rewrite_cut" in saved and "answer_cut" in saved and "hunter2" not in saved
    assert "hunter2" not in caplog.text and "notice rewrite_cut" in caplog.text
