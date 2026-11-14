import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage
from rag_common.contracts import MemoryItem
from rag_common.http import MappedTransport

from rag.agent import cap_tool_calls
from rag.clients import LLMClient, ServiceError
from rag.intent import find_repeat


class CutStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"data: first\n\n"
        raise httpx.ReadError("connection reset")


def test_unreachable_slow_and_cut_off_services_raise_service_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/slow":
            raise httpx.ReadTimeout("slow", request=request)
        if request.url.path == "/down":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, stream=CutStream())

    async def status(path: str) -> int:
        client = httpx.AsyncClient(base_url="http://llm:8080", transport=MappedTransport(httpx.MockTransport(handler)))
        with pytest.raises(ServiceError) as err:
            async with client.stream("GET", path) as response:
                async for _ in response.aiter_lines():
                    pass
        return err.value.status

    assert [asyncio.run(status(p)) for p in ("/slow", "/down", "/cut")] == [504, 502, 502]


def test_duplicate_check_is_skipped_when_the_models_service_is_down():
    async def down(texts):
        raise ServiceError(503, "models loading")

    saved = [MemoryItem(memory_id="b" * 32, text="The user uses EUR.", source="user", created_at="t")]
    assert asyncio.run(find_repeat("The user uses euros.", saved, down, 0.78)) is None


def test_llm_error_frames_and_unfinished_streams_fail_instead_of_looking_complete():
    hi = 'data: {"choices": [{"delta": {"content": "Hi"}}]}'
    frames = {
        "ok": ['data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}', "data: [DONE]"],
        "error": [hi, 'data: {"error": {"message": "context full"}}'],
        "cut": [hi, "data: [DONE]"],  # no finish reason: the answer was cut off
    }
    frames = {kind: "".join(f"{line}\n\n" for line in lines) for kind, lines in frames.items()}

    async def events(kind: str):
        http = httpx.AsyncClient(base_url="http://llm:8080",
                                 transport=httpx.MockTransport(lambda request: httpx.Response(200, text=frames[kind])))
        return [e async for e in LLMClient(http, "m", {}, {}).stream([{"role": "user", "content": "q"}], 10, False)]

    assert asyncio.run(events("ok")) == [("token", "Hi"), ("finish", "stop")]
    for kind in ("error", "cut"):
        with pytest.raises(ServiceError) as err:
            asyncio.run(events(kind))
        assert err.value.status == 502


def test_one_reply_cannot_exceed_the_research_tool_budget():
    calls = [{"name": "search", "args": {"query": f"q{i}"}, "id": f"c{i}", "type": "tool_call"} for i in range(5)]
    capped = cap_tool_calls(AIMessage("", tool_calls=calls), remaining=2)
    assert [c["id"] for c in capped.tool_calls] == ["c0", "c1"]


def test_qdrant_key_file_defaults_to_the_secret_and_an_empty_setting_means_no_key(monkeypatch):
    from rag.config import Settings

    monkeypatch.delenv("RAG_QDRANT_API_KEY_FILE", raising=False)
    assert str(Settings().qdrant_api_key_file).replace("\\", "/") == "/run/secrets/qdrant_api_key"
    monkeypatch.setenv("RAG_QDRANT_API_KEY_FILE", "")
    assert Settings().qdrant_api_key_file is None
