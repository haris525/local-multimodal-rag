"""sql: sandboxed, read-only SQL over the per-document DuckDB files written by ingest.

Only rag calls it, with the caller's identity signed by rag. Tables live under the owner's folder
(tables/<uid>/<doc>.duckdb), so a document of another user does not exist for the caller: 404.

GET /v1/tables/{doc_id} returns that file whole, for rag to ship to the Python sandbox with a run_python call
(rag.sandbox); the same owner check applies. One file streams per user at a time and SQL_MAX_FILE_STREAMS for
everyone; a request waits for its place up to SQL_FILE_QUEUE_WAIT_S, then gets a 503. The file and the places are
given back however the stream ends, also when rag leaves before it starts.

At most SQL_MAX_CONCURRENT queries run at once. A query that finds them all busy waits SQL_QUEUE_WAIT_S at
most, then gets a fast 503 (rag waits and retries once), instead of queueing past rag's own timeout while
the queries ahead of it run their SQL_TIMEOUT_S.
"""

import asyncio
import os
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from rag_common.contracts import DocIdPath, GenerationParameter, SqlQuery, SqlResult
from rag_common.identity import Identity, Verifier, require_user
from rag_common.service import create_app

from sql_sandbox.engine import QueryTimeout, run

TABLES = Path(os.getenv("SQL_TABLES_DIR", "/data/tables"))
TIMEOUT_S = float(os.getenv("SQL_TIMEOUT_S", "20"))  # a query that runs longer is interrupted (408)
QUEUE_WAIT_S = float(os.getenv("SQL_QUEUE_WAIT_S", "5"))  # all slots busy this long: 503
_running = asyncio.Semaphore(int(os.getenv("SQL_MAX_CONCURRENT", "4")))
_streams = asyncio.Semaphore(int(os.getenv("SQL_MAX_FILE_STREAMS", "2")))  # tables files sent at once, for everyone
FILE_QUEUE_WAIT_S = float(os.getenv("SQL_FILE_QUEUE_WAIT_S", "20"))  # a file request waits this long, then 503
_user_streams: dict[str, list] = {}  # user -> [semaphore of 1, requests using it]
MEMORY_LIMIT = os.getenv("SQL_MEMORY_LIMIT", "400MB")  # per query; 4 x 400 MB fits the 2 GB container
PUBLIC_KEYS = Path(os.getenv("SQL_PUBLIC_KEYS_FILE", "/run/secrets/identity_public_keys"))

app = create_app("sql", log_level=os.getenv("SQL_LOG_LEVEL", "INFO"), verifier=Verifier.from_file(PUBLIC_KEYS, {"rag"}))


def table_file(user_id: str, doc_id: str, generation: str | None = None) -> Path | None:
    """The caller's DuckDB file for a document, or None. Both ids are validated (uuid, 32 hex characters);
    the resolved path must still sit under TABLES."""
    path = (TABLES / user_id / doc_id / f"{generation}.duckdb" if generation and generation != "legacy" else
            TABLES / user_id / f"{doc_id}.duckdb")
    if not path.resolve().is_relative_to(TABLES.resolve()) or not path.is_file():
        return None
    return path


@app.post("/v1/query")
async def query(body: SqlQuery, user: Identity = Depends(require_user)) -> SqlResult:
    path = table_file(user.user_id, body.doc_id, body.generation)
    if path is None:
        raise HTTPException(404, "document has no tables")
    try:
        await asyncio.wait_for(_running.acquire(), QUEUE_WAIT_S)
    except TimeoutError as exc:
        raise HTTPException(503, "every query slot is busy (SQL_MAX_CONCURRENT); try again shortly") from exc
    try:
        return await asyncio.to_thread(run, path, body.sql, body.max_rows, TIMEOUT_S, MEMORY_LIMIT)
    except QueryTimeout as exc:  # not a SQL mistake: the caller falls back instead of asking for a repair
        raise HTTPException(408, str(exc)) from exc
    except Exception as exc:  # syntax, binder, permission, memory: the caller repairs or falls back
        raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
    finally:
        _running.release()


class ClosingStream(StreamingResponse):
    """A streamed reply that runs `on_close` once it is over, whatever happened: sent, failed, or the client gone
    before the first byte (a generator that never started would never reach its own finally)."""

    def __init__(self, content, on_close, **kwargs):
        super().__init__(content, **kwargs)
        self.on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.on_close()


async def _stream_place(user_id: str):
    """The user's one place and one of the shared ones, waited for together up to SQL_FILE_QUEUE_WAIT_S (then 503);
    returns the function that gives both back."""
    entry = _user_streams.setdefault(user_id, [asyncio.Semaphore(1), 0])
    entry[1] += 1
    held: list[asyncio.Semaphore] = []

    def give_back() -> None:
        for sem in reversed(held):
            sem.release()
        held.clear()
        entry[1] -= 1
        if entry[1] == 0 and _user_streams.get(user_id) is entry:
            del _user_streams[user_id]

    loop = asyncio.get_running_loop()
    deadline = loop.time() + FILE_QUEUE_WAIT_S
    try:
        for sem in (entry[0], _streams):
            await asyncio.wait_for(sem.acquire(), max(0.0, deadline - loop.time()))
            held.append(sem)
    except BaseException as exc:
        give_back()
        if isinstance(exc, TimeoutError):
            raise HTTPException(503, "a tables file of yours is already streaming, or every stream is busy "
                                     "(SQL_MAX_FILE_STREAMS); try again shortly") from None
        raise
    return give_back


@app.get("/v1/tables/{doc_id}")
async def tables_file(doc_id: DocIdPath, generation: GenerationParameter | None = None,
                      user: Identity = Depends(require_user)) -> StreamingResponse:
    """The caller's DuckDB file of a document, whole. Opened before its size is read, so a reindex that replaces
    the file meanwhile cannot mix two versions: the reply is the version that was opened."""
    path = table_file(user.user_id, doc_id, generation)
    if path is None:
        raise HTTPException(404, "document has no tables")
    give_back = await _stream_place(user.user_id)
    f = None
    try:
        f = open(path, "rb")  # noqa: SIM115 - closed by the reply's on_close
        size = os.fstat(f.fileno()).st_size
    except OSError:
        if f is not None:
            f.close()
        give_back()
        raise HTTPException(404, "document has no tables") from None

    def close() -> None:
        f.close()
        give_back()

    async def chunks():
        while data := await asyncio.to_thread(f.read, 1 << 20):
            yield data

    return ClosingStream(chunks(), close, media_type="application/octet-stream", headers={"content-length": str(size)})
