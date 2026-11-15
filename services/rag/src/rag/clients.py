"""Thin typed clients for the ingest, models, sql, sandbox and LLM services, with one retry for transient failures
(rag_common.retry) and, for the LLM, one retry with more room for a reply cut at its length limit."""

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx

from rag_common.contracts import DocumentInfo, GrepHit, SqlResult
from rag_common.http import ServiceError  # also raised for unreachable (502) and timed-out (504) services
from rag_common.logging import log_fields
from rag_common.retry import describe, retry_once, transient
from rag_common.sandbox import MEDIA_TYPE, encode
from rag_common.snapshots import generation_parameter, selected_generations

STREAM_CHUNK = 1 << 20
__all__ = ["IngestClient", "LLMClient", "ModelsClient", "RetryPolicy", "SandboxClient", "ServiceError", "SqlClient",
           "TooLarge", "response_json"]
log = logging.getLogger("rag.clients")


async def response_json(response: httpx.Response):
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ServiceError(response.status_code, str(detail))
    return response.json()


class IngestClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http

    async def documents(self, doc_ids: list[str]) -> list[DocumentInfo]:
        data = await response_json(await self.http.post("/v1/documents/status", json={"doc_ids": doc_ids}))
        return [DocumentInfo(**d) for d in data]

    async def grep(self, doc_ids: list[str], terms: list[str], limit: int) -> list[GrepHit]:
        body = {"doc_ids": doc_ids, "terms": terms, "limit": limit}
        if (generations := selected_generations(doc_ids)) is not None:
            body["generations"] = generations
        return [GrepHit(**h) for h in await response_json(await self.http.post("/v1/grep", json=body))]

    async def chunks(self, doc_id: str, offset: int, limit: int) -> list[dict]:
        """One page of a document's chunk records (ingest caps a page at 1000), in chunk order."""
        params = {"offset": offset, "limit": limit}
        params.update(generation_parameter(doc_id))
        return await response_json(await self.http.get(f"/v1/documents/{doc_id}/chunks", params=params))

    async def asset_data_uri(self, doc_id: str, name: str) -> str | None:
        response = await self.http.get(f"/v1/documents/{doc_id}/assets/{name}", params=generation_parameter(doc_id))
        if response.status_code != 200:
            return None
        return "data:image/png;base64," + base64.b64encode(response.content).decode()


@dataclass(frozen=True)
class RetryPolicy:
    """One retry: after `backoff_s` for a busy (503), failed (5xx) or timed-out call; with `tokens_factor` x
    the room (at most `max_tokens`) for a model reply cut at its length limit."""

    backoff_s: float = 2.0
    tokens_factor: float = 2.5
    max_tokens: int = 4096


def _sql_retryable(exc: BaseException) -> bool:
    """The sandbox unreachable, busy (503: every query slot taken) or failing. Not a query that timed out in
    DuckDB (408: it would time out again), and not a read timeout of rag's own: the first query may still be
    running in the sandbox, and a second copy would only crowd it."""
    if isinstance(exc, ServiceError) and (exc.status == 408 or exc.cause in ("ReadTimeout", "WriteTimeout")):
        return False
    if isinstance(exc, httpx.ReadTimeout | httpx.WriteTimeout):
        return False
    return transient(exc)


class TooLarge(Exception):
    """A download passed the bytes the caller allowed; `size` is what was declared or read by then."""

    def __init__(self, size: int):
        super().__init__(f"{size} bytes")
        self.size = size


class SqlClient:
    """The sql sandbox service: read-only queries over a document's tables, and a document's tables file."""

    def __init__(self, http: httpx.AsyncClient, retry: RetryPolicy = RetryPolicy()):
        self.http = http
        self.retry = retry

    async def query(self, doc_id: str, sql: str, max_rows: int = 200) -> SqlResult:
        body = {"doc_id": doc_id, "sql": sql, "max_rows": max_rows}
        body.update(generation_parameter(doc_id))

        async def call():
            return SqlResult(**await response_json(await self.http.post("/v1/query", json=body)))

        return await retry_once(call, what="sql query", backoff_s=self.retry.backoff_s, log=log,
                                should_retry=_sql_retryable)

    async def tables_file(self, doc_id: str, max_bytes: int, reserve: Callable[[int], None] | None = None,
                          unreserve: Callable[[int], None] | None = None) -> bytearray:
        """The document's DuckDB file, whole (sql serves only the signed caller's own), in a buffer of its declared
        size (no second copy). TooLarge past `max_bytes`, from its declared length before anything is read, or while
        reading. `reserve(n)` is called with the size before the body is read (it may refuse by raising); whatever
        was reserved and not kept goes back through `unreserve`, on every exit."""
        def noop(_: int) -> None:
            return None

        reserve, unreserve = reserve or noop, unreserve or noop

        async def call() -> bytearray:
            reserved = 0
            try:
                async with self.http.stream("GET", f"/v1/tables/{doc_id}", params=generation_parameter(doc_id)) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        await response_json(response)  # raises ServiceError with the service's detail
                    declared = response.headers.get("content-length", "")
                    size = int(declared) if declared.isdigit() else None
                    if size is not None and size > max_bytes:
                        raise TooLarge(size)
                    reserve(size if size is not None else max_bytes)
                    reserved = size if size is not None else max_bytes
                    buf = bytearray(size) if size is not None else bytearray()
                    view, filled = memoryview(buf) if size is not None else None, 0
                    async for chunk in response.aiter_bytes():
                        if filled + len(chunk) > (size if size is not None else max_bytes):
                            raise TooLarge(filled + len(chunk))
                        if view is not None:
                            view[filled:filled + len(chunk)] = chunk
                        else:
                            buf += chunk
                        filled += len(chunk)
                    if view is not None:
                        view.release()
                    if filled < len(buf):
                        del buf[filled:]
                unreserve(reserved - len(buf))
                reserved = len(buf)
                return buf
            except BaseException:
                unreserve(reserved)
                raise

        return await retry_once(call, what="tables file", backoff_s=self.retry.backoff_s, log=log)


class SandboxClient:
    """The Python sandbox (rag.sandbox). Its client is unsigned: the sandbox gets no identity, so no assertion it
    could replay, and needs none, since the request carries everything a run may read. No retry: a run that may
    have started is not sent twice (httpx still retries a connection that could not be opened)."""

    def __init__(self, http: httpx.AsyncClient):
        self.http = http

    async def execute(self, code: str, timeout_s: float, files: list[tuple[str, bytes | bytearray]],
                      run_id: str | None = None) -> dict:
        """One run; `run_id` (32 hex characters, made by rag) names it for cancel()."""
        pieces = encode(code, timeout_s, files)

        async def body():
            for piece in pieces:
                view = memoryview(piece)
                for start in range(0, len(view), STREAM_CHUNK):  # 1 MiB copies at a time, never a whole file
                    yield bytes(view[start:start + STREAM_CHUNK])

        headers = {"content-type": MEDIA_TYPE, "content-length": str(sum(len(p) for p in pieces))}
        if run_id:
            headers["x-run-id"] = run_id
        return await response_json(await self.http.post("/v1/execute", content=body(), headers=headers))

    async def cancel(self, run_id: str) -> dict:
        """Ask the sandbox to kill the run `run_id`, or to drop it if it has not started; it answers once the run
        has ended ({"ended": true}) or its own wait passed ({"ended": false})."""
        return await response_json(await self.http.post(f"/v1/runs/{run_id}/cancel"))


class ModelsClient:
    """The models service. Its GPU gate answers 503 when its queue is full: every call waits and retries once."""

    def __init__(self, http: httpx.AsyncClient, retry: RetryPolicy = RetryPolicy()):
        self.http = http
        self.retry = retry

    async def _post(self, path: str, body: dict, what: str) -> dict:
        async def call():
            return await response_json(await self.http.post(path, json=body))

        return await retry_once(call, what=what, backoff_s=self.retry.backoff_s, log=log)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._post("/v1/embed/text", {"texts": [text], "role": "query"}, "query embedding"))["vectors"][0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return (await self._post("/v1/embed/text", {"texts": texts, "role": "document"}, "text embedding"))["vectors"]

    async def embed_image_query(self, text: str) -> list[float]:
        return (await self._post("/v1/embed/image", {"texts": [text]}, "image query embedding"))["vectors"][0]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return (await self._post("/v1/rerank", {"query": query, "documents": documents}, "rerank"))["scores"]


class LLMClient:
    """OpenAI-compatible chat completions (llama-server) with per-request thinking control.

    complete(): a reply cut at its length limit is asked again once with more room; a busy, failed or
    timed-out call is retried once after a backoff. stream(): a failure before the first token is retried
    once; after that, nothing is resent (the user already sees the text)."""

    def __init__(self, http: httpx.AsyncClient, model: str, sampling: dict, think_sampling: dict,
                 retry: RetryPolicy = RetryPolicy()):
        self.http = http
        self.model = model
        self.sampling = sampling
        self.think_sampling = think_sampling
        self.retry = retry

    def _body(self, messages: list[dict], max_tokens: int, think: bool, **extra) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think},
            **(self.think_sampling if think else self.sampling),
            **extra,
        }

    async def complete(self, messages: list[dict], max_tokens: int, temperature: float = 0.0, json_mode: bool = False,
                       tools: list[dict] | None = None, purpose: str = "model call") -> dict:
        """Non-streaming call in instruct mode. Returns the assistant message with "finish_reason" added; a reply
        that is still cut after the retry keeps "length" there, for the caller to handle."""
        extra: dict = {"temperature": temperature}
        if json_mode:
            extra["response_format"] = {"type": "json_object"}
        if tools:
            extra["tools"] = tools

        async def once(room: int) -> dict:
            async def call():
                body = self._body(messages, room, think=False, **extra)
                return await response_json(await self.http.post("/v1/chat/completions", json=body))

            choice = (await retry_once(call, what=purpose, backoff_s=self.retry.backoff_s, log=log))["choices"][0]
            return {**choice["message"], "finish_reason": choice.get("finish_reason")}

        msg = await once(max_tokens)
        if msg["finish_reason"] == "length" and max_tokens < self.retry.max_tokens:
            room = min(self.retry.max_tokens, max(max_tokens + 1, int(max_tokens * self.retry.tokens_factor)))
            log_fields(log, f"{purpose}: reply cut at its token limit; retrying once with more room",
                       logging.WARNING, max_tokens=max_tokens, retry_tokens=room)
            msg = await once(room)
            if msg["finish_reason"] == "length":
                log_fields(log, f"{purpose}: reply cut again after the retry", logging.ERROR, max_tokens=room)
        return msg

    async def stream(self, messages: list[dict], max_tokens: int, think: bool) -> AsyncIterator[tuple[str, object]]:
        """Yield ("reasoning" | "token", text), then ("usage", dict) and ("finish", reason). An error frame
        from the server, or a stream that ends without a finish reason, raises ServiceError (502), so a cut-off
        answer is reported as failed instead of looking complete. A transient failure before anything was
        yielded is retried once after the backoff."""
        body = self._body(messages, max_tokens, think, stream=True, stream_options={"include_usage": True})
        for attempt in (1, 2):
            yielded = False
            try:
                async for item in self._stream_once(body):
                    yielded = True
                    yield item
                return
            except Exception as exc:
                if yielded or attempt == 2 or not transient(exc):
                    if attempt == 2:
                        log_fields(log, "answer stream failed again after a retry", logging.ERROR, error=describe(exc))
                    raise
                log_fields(log, "answer stream failed before its first token; retrying once", logging.WARNING,
                           error=describe(exc), backoff_s=self.retry.backoff_s)
            await asyncio.sleep(self.retry.backoff_s)

    async def _stream_once(self, body: dict) -> AsyncIterator[tuple[str, object]]:
        finished = False
        async with self.http.stream("POST", "/v1/chat/completions", json=body) as response:
            if response.status_code >= 400:
                raise ServiceError(response.status_code, (await response.aread()).decode(errors="replace"))
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("error"):
                    error = event["error"]
                    raise ServiceError(502, f"llm: {error.get('message', error) if isinstance(error, dict) else error}")
                if event.get("usage"):
                    yield "usage", event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        yield "reasoning", delta["reasoning_content"]
                    if delta.get("content"):
                        yield "token", delta["content"]
                    if choice.get("finish_reason"):
                        finished = True
                        yield "finish", choice["finish_reason"]
        if not finished:
            raise ServiceError(502, "llm: the answer stream ended without a finish reason")
