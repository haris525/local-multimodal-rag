"""The live scripts read chat answers with the shared reader (libs/common rag_common.chat_events), so a cut
stream, an error event or events out of order count as failures.

    uv run --with httpx --with pytest pytest -q scripts/tests
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_smoke  # noqa: E402
import multiuser_smoke  # noqa: E402
import smoke_test  # noqa: E402
from rag_common.chat_events import parse_sse  # noqa: E402  (put on the path by the scripts)

COMPLETE = ('event: sources\ndata: {"route": "search", "sources": [{"n": 1, "doc_name": "a.pdf"}]}\n\n'
            'event: token\ndata: {"text": "It is 45 days [1]."}\n\n'
            'event: done\ndata: {"citations": [1], "invalid_citations": [], "timings": {"total_ms": 9}}\n\n')
CUT = 'event: sources\ndata: {"route": "search", "sources": []}\n\nevent: token\ndata: {"text": "It is 4"}\n\n'
FAILED = 'event: memory\ndata: {"action": "saved", "items": []}\n\nevent: error\ndata: {"message": "llm: timeout"}\n\n'


def client(body: str) -> httpx.Client:
    return httpx.Client(base_url="http://gw", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})))


def test_smoke_test_reports_a_cut_answer_as_an_error():
    good = smoke_test.ask(client(COMPLETE), "q", ["a" * 32], "single")
    assert good["error"] is None and good["citations"] == [1] and good["answer"] == "It is 45 days [1]."
    cut = smoke_test.ask(client(CUT), "q", ["a" * 32], "single")
    assert cut["error"] == "the answer stream ended before its done event" and cut["answer"] == "It is 4"


def test_memory_smoke_no_longer_ignores_error_events():
    live = memory_smoke.Live("http://gw", "rag_pat_x")
    live.c = client(FAILED)
    r = live.chat("t" * 32, "remember I like tea", [])
    assert r["error"] == "llm: timeout" and r["memory"] == [{"action": "saved", "items": []}]
    assert live.unfinished == ["'remember I like tea': llm: timeout"]


def test_multiuser_smoke_uses_the_shared_parser():
    assert multiuser_smoke.sse_events is parse_sse
