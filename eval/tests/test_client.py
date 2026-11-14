import httpx
import pytest

from rag_eval.cli import explain
from rag_eval.client import Stack


def test_token_is_required(monkeypatch):
    monkeypatch.delenv("EVAL_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="EVAL_TOKEN"):
        Stack()


def test_calls_go_through_the_gateway_with_the_token(monkeypatch):
    monkeypatch.setenv("EVAL_TOKEN", "rag_pat_abc")
    monkeypatch.setenv("EVAL_URL", "http://gateway:8000/api")
    stack = Stack()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), request.headers.get("authorization")))
        if request.url.path.endswith("/chunks"):
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/chat"):
            return httpx.Response(200, text='event: sources\ndata: {"route": "search", "sources": []}\n\n'
                                            'event: token\ndata: {"text": "hi"}\n\nevent: done\ndata: '
                                            '{"citations": [], "invalid_citations": []}\n\n')
        return httpx.Response(200, json=[] if request.method == "GET" else {"results": []})

    stack.api._transport = httpx.MockTransport(handler)
    stack.documents()
    stack.chunks("a" * 32)
    stack.search("q", ["a" * 32])
    chat = stack.chat("q", ["a" * 32])
    assert chat.answer == "hi" and chat.error is None and chat.route == "search"
    assert [url for _, url, _ in seen] == [
        "http://gateway:8000/api/documents",
        f"http://gateway:8000/api/documents/{'a' * 32}/chunks?offset=0&limit=1000",
        "http://gateway:8000/api/search",
        "http://gateway:8000/api/chat",
    ]
    assert {auth for _, _, auth in seen} == {"Bearer rag_pat_abc"}


def test_unknown_documents_are_explained():
    request = httpx.Request("POST", "http://gateway:8000/api/search")
    response = httpx.Response(404, json={"detail": "unknown documents: abc"}, request=request)
    message = explain(httpx.HTTPStatusError("404", request=request, response=response))
    assert "does not own" in message
    response = httpx.Response(401, json={"detail": "sign in required"}, request=request)
    assert "EVAL_TOKEN" in explain(httpx.HTTPStatusError("401", request=request, response=response))


def test_a_cut_or_failed_answer_is_an_error(monkeypatch):
    """A stream that stops before done, or ends with an error event, is never scored as a complete answer."""
    monkeypatch.setenv("EVAL_TOKEN", "rag_pat_abc")
    stack = Stack()
    sources = 'event: sources\ndata: {"route": "search", "sources": []}\n\n'
    bodies = iter([sources + 'event: token\ndata: {"text": "Half"}\n\n',
                   sources + 'event: error\ndata: {"message": "boom"}\n\n'])
    stack.api._transport = httpx.MockTransport(lambda request: httpx.Response(200, text=next(bodies)))
    cut = stack.chat("q", ["a" * 32])
    assert cut.answer == "Half" and cut.error == "the answer stream ended before its done event"
    assert stack.chat("q", ["a" * 32]).error == "boom"


def test_the_benchmark_gateway_takes_rag_token_or_eval_token(monkeypatch):
    from rag_eval.client import Gateway

    for name in ("RAG_TOKEN", "EVAL_TOKEN", "EVAL_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="RAG_TOKEN"):
        Gateway()
    monkeypatch.setenv("EVAL_TOKEN", "rag_pat_eval")
    gw = Gateway()
    assert gw.api.headers["authorization"] == "Bearer rag_pat_eval" and str(gw.api.base_url) == "http://localhost:8000/api/"
    monkeypatch.setenv("RAG_TOKEN", "rag_pat_bench")
    assert Gateway(url="http://gateway:8000/api/").api.headers["authorization"] == "Bearer rag_pat_bench"


def test_ask_reports_http_and_connection_errors_and_ensure_reports_failed_ingestion(monkeypatch):
    from rag_eval.client import Gateway

    monkeypatch.setenv("RAG_TOKEN", "rag_pat_bench")
    gw = Gateway()
    gw.poll_s = 0

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat"):
            if b"down" in request.content:
                raise httpx.ConnectError("refused")
            return httpx.Response(409, json={"detail": "documents without a complete index"})
        if request.method == "POST":
            return httpx.Response(200, json=[{"name": "a.pdf", "error": None,
                                              "document": {"doc_id": "a" * 32, "name": "a.pdf", "status": "queued"}}])
        return httpx.Response(200, json=[{"doc_id": "a" * 32, "name": "a.pdf", "status": "failed", "error": "parser crashed"}])

    gw.api._transport = httpx.MockTransport(handler)
    t = gw.ask("q", ["a" * 32])
    assert t.reply is None and t.status == 409 and t.error.startswith("HTTP 409")
    t = gw.ask("down", ["a" * 32])
    assert t.reply is None and t.status == 0 and "ConnectError" in t.error
    with pytest.raises(SystemExit, match="a.pdf: parser crashed"):
        gw.ensure({"a.pdf": lambda: b"%PDF-"}, budget_s=5, refresh=True, progress=lambda msg: None)
