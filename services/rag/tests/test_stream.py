import asyncio
import contextlib
from types import SimpleNamespace

from rag_common.contracts import ThreadContext

import rag.api as api
from rag.chat import ChatRequest


class FakeAudit:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def record(self, action, subject=None, actor="local", **detail):
        self.events.append((action, detail))


class FakeThreads:
    def __init__(self):
        self.appended: list[list] = []

    async def append(self, thread_id, messages, doc_ids):
        self.appended.append(messages)

    async def context(self, thread_id):
        return ThreadContext(summary="", summary_upto=0, messages=[])

    async def memories(self):
        return []


class FakeLLM:
    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return {"content": '{"memories": []}'}


def test_disconnect_mid_stream_still_saves_the_partial_answer_and_audits_it():
    async def scenario():
        audit, threads = FakeAudit(), FakeThreads()
        api.state.update(audit=audit, deps=SimpleNamespace(threads=threads, llm=FakeLLM(), s=api.settings))

        async def events():
            yield "sources", {"route": "search", "sources": []}
            yield "token", {"text": "Partial answer"}
            await asyncio.sleep(10)  # the client goes away while the model is still writing
            yield "token", {"text": " never sent"}

        async def drain(stream):
            async for _ in stream:
                pass

        req = ChatRequest(question="q", doc_ids=["d" * 32], thread_id="a" * 32)
        consumer = asyncio.create_task(drain(api._sse(events(), req, [])))
        await asyncio.sleep(0.05)
        consumer.cancel()  # what Starlette does to the response task on disconnect
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        await asyncio.gather(*api._background)
        return audit, threads

    audit, threads = asyncio.run(scenario())
    assert audit.events[0][0] == "chat.failed" and audit.events[0][1]["error"] == api.DISCONNECTED
    assistant = threads.appended[0][1]
    assert assistant.content == "Partial answer" and assistant.meta["interrupted"] is True


def test_a_finished_answer_is_saved_before_the_stream_ends():
    """The user's next message loads the thread; it must already hold this exchange."""
    async def scenario():
        audit, threads = FakeAudit(), FakeThreads()
        api.state.update(audit=audit, deps=SimpleNamespace(threads=threads, llm=FakeLLM(), s=api.settings))

        async def events():
            yield "sources", {"route": "search", "sources": []}
            yield "token", {"text": "South had 340 k$."}
            yield "done", {"citations": []}

        req = ChatRequest(question="q", doc_ids=["d" * 32], thread_id="a" * 32)
        async for _ in api._sse(events(), req, []):
            pass
        saved_when_stream_ended = [m.content for m in threads.appended[0]] if threads.appended else []
        await asyncio.gather(*api._background)
        return saved_when_stream_ended, audit

    saved, audit = asyncio.run(scenario())
    assert saved == ["q", "South had 340 k$."] and audit.events[0][0] == "chat.answered"
