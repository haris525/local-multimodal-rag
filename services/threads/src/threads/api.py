"""threads: conversations (with running summaries), long-term memories and custom search tasks, per user.

The gateway exposes listing, reading, renaming, deleting, memories and search tasks; appending messages, reading
model context and writing summaries are backend-only (used by rag, so they need an identity signed by rag).
Every route acts for the user in X-Identity; another user's thread, memory or search task is a 404, like a missing
one. rag reads a chat's search task with the chat's identity (rag.search_tasks). THREADS_MAX_MEMORIES and
THREADS_MAX_SEARCH_TASKS apply per user. /internal/* is for auth's system calls (capabilities, legacy adoption,
account purge).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, HTTPException, Path as PathParam, Query, Response
from fastapi.responses import JSONResponse

from rag_common.audit import AuditClient
from rag_common.contracts import (
    SEARCH_TASK_ID,
    THREAD_ID,
    MemoryCreate,
    MigrateLegacy,
    MemoryItem,
    NewMessages,
    SearchTaskItem,
    SearchTaskWrite,
    SummaryUpdate,
    ThreadContext,
    ThreadCreate,
    ThreadDetail,
    ThreadInfo,
    ThreadMessage,
    ThreadUpdate,
)
from rag_common.identity import USER_ID, Identity, Verifier, require_issuer, require_system, require_user
from rag_common.secrets import looks_secret, redact_secrets
from rag_common.service import create_app

from threads.store import SCHEMA_VERSION, OwnerDeleted, Store

store = Store(Path(os.getenv("THREADS_DB", "/data/threads.db")), int(os.getenv("THREADS_MAX_MEMORIES", "50")),
              int(os.getenv("THREADS_MAX_SEARCH_TASKS", "50")))
audit = AuditClient(os.getenv("THREADS_AUDIT_URL", "http://audit:8004") or None, "threads")
ThreadId = PathParam(pattern=THREAD_ID)
TaskId = PathParam(pattern=SEARCH_TASK_ID)
UserId = PathParam(pattern=USER_ID)
FROM_RAG = Depends(require_issuer("rag"))
PUBLIC_KEYS = Path(os.getenv("THREADS_PUBLIC_KEYS_FILE", "/run/secrets/identity_public_keys"))


@asynccontextmanager
async def lifespan(_app):
    yield
    await audit.aclose()


app = create_app("threads", lifespan=lifespan, log_level=os.getenv("THREADS_LOG_LEVEL", "INFO"),
                 verifier=Verifier.from_file(PUBLIC_KEYS, {"auth", "rag"}))


@app.exception_handler(OwnerDeleted)
async def account_deleted(_request, _exc: OwnerDeleted) -> JSONResponse:
    """A write for a deleted account (an identity signed just before the deletion is valid for a while)."""
    return JSONResponse({"detail": "this account has been deleted"}, status_code=403)


def _found(value, what: str = "thread"):
    if value is None:
        raise HTTPException(404, f"{what} not found")
    return value


@app.post("/v1/threads", status_code=201)
def create_thread(body: ThreadCreate, user: Identity = Depends(require_user)) -> ThreadInfo:
    return store.create(user.user_id, body.title, body.doc_ids)


@app.get("/v1/threads")
def list_threads(limit: int = Query(100, ge=1, le=500), user: Identity = Depends(require_user)) -> list[ThreadInfo]:
    return store.recent(user.user_id, limit)


@app.get("/v1/threads/{thread_id}")
def get_thread(thread_id: str = ThreadId, user: Identity = Depends(require_user)) -> ThreadDetail:
    return _found(store.get(user.user_id, thread_id))


@app.patch("/v1/threads/{thread_id}")
def update_thread(body: ThreadUpdate, thread_id: str = ThreadId, user: Identity = Depends(require_user)) -> ThreadInfo:
    return _found(store.update(user.user_id, thread_id, body.title, body.doc_ids))


@app.delete("/v1/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: str = ThreadId, user: Identity = Depends(require_user)) -> Response:
    _found(store.delete(user.user_id, thread_id) or None)
    await audit.record("thread.deleted", thread_id, user_id=user.user_id)
    return Response(status_code=204)


@app.post("/v1/threads/{thread_id}/messages", status_code=201, dependencies=[FROM_RAG])
def append_messages(body: NewMessages, thread_id: str = ThreadId,
                    user: Identity = Depends(require_user)) -> list[ThreadMessage]:
    return _found(store.append(user.user_id, thread_id, body.messages, body.doc_ids))


@app.get("/v1/threads/{thread_id}/context", dependencies=[FROM_RAG])
def thread_context(thread_id: str = ThreadId, user: Identity = Depends(require_user)) -> ThreadContext:
    return _found(store.context(user.user_id, thread_id))


@app.put("/v1/threads/{thread_id}/summary", dependencies=[FROM_RAG])
def update_summary(body: SummaryUpdate, thread_id: str = ThreadId, user: Identity = Depends(require_user)) -> dict:
    return {"updated": store.set_summary(user.user_id, thread_id, body.summary, body.upto)}


@app.get("/v1/memories")
def list_memories(user: Identity = Depends(require_user)) -> list[MemoryItem]:
    return store.memories(user.user_id)


def _no_credentials(text: str) -> None:
    """Every memory write passes here (chat and the Memory panel), so a credential is never stored."""
    if looks_secret(text):
        raise HTTPException(422, "memories never hold passwords, keys, tokens or other credentials")


@app.post("/v1/memories", status_code=201)
async def add_memory(body: MemoryCreate, user: Identity = Depends(require_user)) -> MemoryItem | None:
    _no_credentials(body.text)
    try:
        item = store.add_memory(user.user_id, body.text, body.source, body.thread_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if item:
        await audit.record("memory.added", item.memory_id, user_id=user.user_id, text=item.text, source=item.source,
                           thread_id=item.thread_id)
    return item


@app.put("/v1/memories/{memory_id}")
async def replace_memory(body: MemoryCreate, memory_id: str = PathParam(pattern=THREAD_ID),
                         user: Identity = Depends(require_user)) -> MemoryItem | None:
    """Correct a memory in place; null when another memory already says the same (nothing changes)."""
    _no_credentials(body.text)
    try:
        result = store.replace_memory(user.user_id, memory_id, body.text, body.source)
    except KeyError:
        raise HTTPException(404, "memory not found") from None
    if result is None:
        return None
    item, old = result
    await audit.record("memory.updated", memory_id, user_id=user.user_id, text=item.text, replaces=old,
                       source=item.source)
    return item


@app.delete("/v1/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: str = PathParam(pattern=THREAD_ID),
                        user: Identity = Depends(require_user)) -> Response:
    _found(store.delete_memory(user.user_id, memory_id) or None, "memory")
    await audit.record("memory.deleted", memory_id, user_id=user.user_id)
    return Response(status_code=204)


# --- custom search tasks: a user's own instructions for embedding the question ---------------------
# The audit names the task, never its name or instruction.


def _no_credential_values(body: SearchTaskWrite) -> None:
    """Every search task write passes here: a value that redact_secrets would remove (a password or key after its
    label, a token, a URL with a password) is refused, so none is stored. Words alone are fine: an instruction
    about "password reset policies" holds no credential."""
    if any(redact_secrets(text) != text for text in (body.name, body.instruction)):
        raise HTTPException(422, "search tasks never hold passwords, keys, tokens or other credentials")


@app.get("/v1/search-tasks")
def list_search_tasks(user: Identity = Depends(require_user)) -> list[SearchTaskItem]:
    return store.search_tasks(user.user_id)


@app.get("/v1/search-tasks/{task_id}")
def get_search_task(task_id: str = TaskId, user: Identity = Depends(require_user)) -> SearchTaskItem:
    return _found(store.search_task(user.user_id, task_id), "search task")


@app.post("/v1/search-tasks", status_code=201)
async def add_search_task(body: SearchTaskWrite, user: Identity = Depends(require_user)) -> SearchTaskItem:
    _no_credential_values(body)
    try:
        item = store.add_search_task(user.user_id, body.name, body.instruction)
    except ValueError as exc:  # at the user's limit (THREADS_MAX_SEARCH_TASKS)
        raise HTTPException(409, str(exc)) from exc
    await audit.record("search_task.added", item.task_id, user_id=user.user_id)
    return item


@app.put("/v1/search-tasks/{task_id}")
async def update_search_task(body: SearchTaskWrite, task_id: str = TaskId,
                             user: Identity = Depends(require_user)) -> SearchTaskItem:
    _no_credential_values(body)
    item = _found(store.update_search_task(user.user_id, task_id, body.name, body.instruction), "search task")
    await audit.record("search_task.updated", task_id, user_id=user.user_id)
    return item


@app.delete("/v1/search-tasks/{task_id}", status_code=204)
async def delete_search_task(task_id: str = TaskId, user: Identity = Depends(require_user)) -> Response:
    _found(store.delete_search_task(user.user_id, task_id) or None, "search task")
    await audit.record("search_task.deleted", task_id, user_id=user.user_id)
    return Response(status_code=204)


# --- auth's system calls --------------------------------------------------------------------


@app.get("/internal/capabilities")
def capabilities(_: Identity = Depends(require_system)) -> dict:
    return {"service": "threads", "multi_user": 1, "schema": SCHEMA_VERSION}


@app.delete("/internal/users/{user_id}")
def purge_user(user_id: str = UserId, _: Identity = Depends(require_system)) -> dict:
    """Delete an account's threads, messages, memories and search tasks. Repeatable (a second call deletes nothing)."""
    return store.purge_user(user_id)


@app.post("/internal/migrate-legacy")
def migrate_legacy(body: MigrateLegacy, _: Identity = Depends(require_system)) -> dict:
    """Threads and memories from before accounts existed become this user's. Repeatable. 409 for a deleted
    account."""
    try:
        return store.adopt_legacy(body.owner_id, body.dry_run)
    except OwnerDeleted:
        raise HTTPException(409, "this account has been deleted") from None
