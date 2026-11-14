"""rag: streamed, cited answers over selected documents; /v1/search for evaluation and debugging.

Every request carries the caller's identity (X-Identity, signed by auth). rag re-signs it for its calls to
ingest, sql and threads, so each of them answers with that user's data only; Qdrant reads filter on the
user as well (retrieval.py). Calls to the Python sandbox carry no identity: rag sends the data a run may read,
and every request from the sandbox's network is refused before anything else (rag.guard).
Background work after a response (thread save, compaction, memory extraction, audit) runs in tasks that copy
the request context, identity included.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, HTTPException
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_common.audit import AuditClient
from rag_common.chat_events import ChatEvent
from rag_common.contracts import MAX_SELECTED_DOCS, DocId, DocStatus, DocumentInfo, NewMessage
from rag_common.http import is_up, make_client
from rag_common.identity import (
    Identity,
    IdentityError,
    Signer,
    Verifier,
    load_public_keys,
    require_system,
    require_user,
    require_user_id,
)
from rag_common.secrets import redact_secrets
from rag_common.index import COLLECTION, qdrant_client
from rag_common.service import create_app

from rag.chat import (ChatRequest, Deps, Event, deep_agent, deep_fanout, deep_python, deep_think, request_problem,
                      research_calls)
from rag import notices, tokens
from rag.clients import IngestClient, LLMClient, ModelsClient, RetryPolicy, SandboxClient, ServiceError, SqlClient
from rag.config import get_settings
from rag.conversation import Conversation, ThreadsClient, load, maintain
from rag.guard import SubnetGuard
from rag.planner import Planner
from rag.prompts import location
from rag.retrieval import ALL_RETRIEVERS, Retrieval, SearchOptions
from rag.turn import respond

log = logging.getLogger("rag.api")
settings = get_settings()
state: dict = {}
SCHEMA = 1  # reported by /internal/capabilities: reads are scoped by owner


def load_signer(s) -> Signer:
    """rag's signing key; it must match the rag entry of the public keys every other service verifies with."""
    signer = Signer.from_file(s.signing_key_file, "rag")
    if load_public_keys(s.public_keys_file).get("rag") != signer.public_key_raw():
        raise IdentityError(f"{s.signing_key_file} does not match the rag key in {s.public_keys_file}")
    return signer


@asynccontextmanager
async def lifespan(_app):
    s = settings
    signer = load_signer(s)
    tokens.configure(s.tokenizer_file, s.tokenizer_factor)  # prompt budgets in model tokens (estimate without it)
    qdrant = qdrant_client(s.qdrant_url, s.qdrant_api_key_file, timeout=int(s.qdrant_timeout_s))
    http = {  # calls that touch user data carry the caller's identity, signed by rag; models and llm hold none
        "models": make_client(s.models_url, timeout=s.models_timeout_s),
        "ingest": make_client(s.ingest_url, timeout=s.ingest_timeout_s, signer=signer),
        "llm": make_client(s.llm_url, timeout=s.llm_timeout_s),
        "sql": make_client(s.sql_url, timeout=s.sql_timeout_s, signer=signer),
        "threads": make_client(s.threads_url, timeout=s.threads_timeout_s, signer=signer),
        # unsigned on purpose: the sandbox runs model-written code and gets no identity it could replay
        "sandbox": make_client(s.sandbox_url, timeout=s.sandbox_request_timeout_s),
    }
    retry = RetryPolicy(backoff_s=s.retry_backoff_s, tokens_factor=s.retry_tokens_factor, max_tokens=s.retry_max_tokens)
    ingest = IngestClient(http["ingest"])
    llm = LLMClient(
        http["llm"],
        s.llm_model,
        sampling={"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k, "presence_penalty": s.presence_penalty},
        think_sampling={"temperature": s.think_temperature, "top_p": s.think_top_p, "top_k": s.top_k, "presence_penalty": 0.0},
        retry=retry,
    )
    retrieval = Retrieval(qdrant, ModelsClient(http["models"], retry), ingest, s)
    audit = AuditClient(s.audit_url, "rag", signer=signer)  # audit takes rag events only signed by rag
    sql = SqlClient(http["sql"], retry)
    graph_llm = ChatOpenAI(  # retries are agent.invoke's (logged); the SDK's own are off
        model=s.llm_model, base_url=f"{s.llm_url}/v1", api_key="local", temperature=s.agent_temperature,
        max_tokens=s.agent_tokens, timeout=s.research_llm_timeout_s, max_retries=0,
        extra_body={"top_k": s.top_k, "chat_template_kwargs": {"enable_thinking": False}},
    )
    planner = Planner(llm, sql, s.sql_budget, max_tokens=s.planner_tokens, card_tokens=s.schema_card_tokens,
                      sql_rows=s.sql_max_rows, filter_rows=s.filter_max_rows)
    deps = Deps(s, retrieval, planner, llm, ingest, sql, graph_llm, ThreadsClient(http["threads"]),
                SandboxClient(http["sandbox"]))
    state.update(deps=deps, qdrant=qdrant, http=http,
                 audit=audit)
    yield
    await audit.aclose()
    for client in http.values():
        await client.aclose()
    await qdrant.close()


_ready: dict = {"loop": None, "lock": None, "at": None, "checks": None}


async def readiness() -> dict[str, bool]:
    """The checks, answered from the last ones for RAG_READY_CACHE_S: /readyz needs no identity, and one call probes
    every backend, so callers share one probe instead of each sending their own."""
    loop = asyncio.get_running_loop()
    if _ready["loop"] is not loop:  # one lock and cache per event loop (tests run several)
        _ready.update(loop=loop, lock=asyncio.Lock(), at=None, checks=None)
    async with _ready["lock"]:
        if _ready["checks"] is None or loop.time() - _ready["at"] >= settings.ready_cache_s:
            _ready["checks"] = await _probe()
            _ready["at"] = loop.time()
        return dict(_ready["checks"])


async def _probe() -> dict[str, bool]:
    http = state.get("http", {})
    try:
        index = await state["qdrant"].collection_exists(COLLECTION)
    except Exception:
        index = False
    return {
        "index": index,
        "models": await is_up(http["models"], "/readyz") if http else False,
        "llm": await is_up(http["llm"], "/health") if http else False,
        "ingest": await is_up(http["ingest"], "/healthz") if http else False,
        "sql": await is_up(http["sql"], "/healthz") if http else False,
        "threads": await is_up(http["threads"], "/healthz") if http else False,
        "audit": await is_up(state["audit"].http, "/healthz") if state.get("audit") and state["audit"].http else True,
        # only when the operator turned Python on (request or on)
        **({"sandbox": await is_up(http["sandbox"], "/healthz") if http else False}
           if settings.deep_python != "off" else {}),
    }


app = create_app("rag", lifespan=lifespan, readiness=readiness, log_level=settings.log_level,
                 verifier=Verifier.from_file(settings.public_keys_file, {"auth"}))
app.add_middleware(SubnetGuard, subnet=settings.sandbox_subnet)  # outermost: before the access log and identity


@app.get("/internal/capabilities")
def capabilities(_: Identity = Depends(require_system)) -> dict:
    return {"service": "rag", "multi_user": 1, "schema": SCHEMA}


async def _ready_documents(doc_ids: list[str]) -> list[DocumentInfo]:
    if not doc_ids:
        return []
    uid = require_user_id()
    try:
        listed = await state["deps"].ingest.documents(doc_ids)
    except ServiceError as exc:
        raise HTTPException(502, f"ingest service: {exc.detail}") from exc
    # ingest lists only the caller's documents; anything else is treated as missing (defense in depth)
    docs = {d.doc_id: d for d in listed if d.owner_id == uid}
    if len(docs) != len(listed):
        log.error("ingest listed documents of another owner", extra={"fields": {"count": len(listed) - len(docs)}})
    missing = [d for d in doc_ids if d not in docs]
    if missing:
        raise HTTPException(404, f"unknown documents: {', '.join(missing)}")
    # A document is usable once it has a complete index (n_chunks is set only on completion), so it
    # stays searchable on its previous index while a reindex or a failed retry is in progress.
    pending = [docs[d].name for d in doc_ids if docs[d].status != DocStatus.READY and docs[d].n_chunks == 0]
    if pending:
        raise HTTPException(409, f"documents not ready yet: {', '.join(pending)}")
    return [docs[d] for d in doc_ids]


_background: set[asyncio.Task] = set()  # work that must outlive the response (saving, auditing, compaction)
DISCONNECTED = "client disconnected before the answer finished"


def _redacted(value):
    """Credentials out of stored structures (deep-research steps carry tool arguments)."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: _redacted(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    return value


def _track(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task


def _deep(req: ChatRequest) -> dict:
    """Which deep-research implementation answered, whether its plan and verify steps thought, whether it fanned
    out to researchers on reserved budgets, its research-call budget, and whether it could run Python (deep mode)."""
    if req.mode != "deep":
        return {}
    return {"agent": deep_agent(settings, req), "deep_think": deep_think(settings, req),
            "fanout": deep_fanout(settings, req), "research_calls": research_calls(settings, req),
            "python": deep_python(settings, req)}


async def _persist(req: ChatRequest, answer: str, retrieved: dict, done: dict, steps: list[dict],
                   memory: list[dict], error: str | None, notes: list[dict]) -> None:
    """Save the exchange to its thread (credentials redacted). Compaction and implicit-memory extraction
    then run in the background; turns that already changed memory, or were memory requests, skip extraction."""
    deps: Deps = state["deps"]
    meta = {
        "memory": memory,
        "query": redact_secrets(retrieved.get("query") or ""), "route": retrieved.get("route"),
        "route_detail": retrieved.get("route_detail"), "per_doc": retrieved.get("per_doc"),
        "sources": retrieved.get("sources", []), "steps": _redacted(steps), "citations": done.get("citations"),
        "invalid_citations": done.get("invalid_citations"), "timings": done.get("timings"),
        "mode": req.mode, "think": req.think, **_deep(req), "doc_ids": req.doc_ids, "error": error,
        "interrupted": error == DISCONNECTED,
        "notices": [{**n, "message": redact_secrets(n.get("message", ""))} for n in notes],
        "finish_reason": done.get("finish_reason"),
    }
    try:
        await deps.threads.append(req.thread_id, [NewMessage(role="user", content=redact_secrets(req.question)),
                                                  NewMessage(role="assistant", content=redact_secrets(answer), meta=meta)],
                                  req.doc_ids)
    except Exception:
        log.exception("saving the exchange to its thread failed")
        return
    _track(maintain(
        deps.threads, deps.llm, req.thread_id, redact_secrets(req.question), trigger_tokens=deps.s.summary_trigger_tokens,
        trigger_messages=deps.s.summary_trigger_messages, keep=deps.s.summary_keep_messages,
        auto_memory=deps.s.auto_memory and not memory and not str(retrieved.get("route", "")).startswith("memory"),
        compaction_tokens=deps.s.compaction_tokens, memory_tokens=deps.s.memory_tokens,
        max_fold=deps.s.summary_fold_messages, max_fold_tokens=deps.s.summary_fold_tokens))


async def _finish(req: ChatRequest, docs: list[DocumentInfo], answer: str, retrieved: dict, done: dict,
                  steps: list[dict], memory: list[dict], error: str | None, notes: list[dict] | None = None) -> None:
    """Everything after the stream: thread save and audit record. Runs as its own task."""
    notes = notes or []
    try:
        if req.thread_id:
            await _persist(req, answer, retrieved, done, steps, memory, error, notes)
    finally:
        await state["audit"].record(
            "chat.failed" if error else "chat.answered",
            ",".join(req.doc_ids),
            thread_id=req.thread_id,
            question=redact_secrets(req.question),
            query=redact_secrets(retrieved.get("query") or ""),
            memory=memory,
            documents=[{"doc_id": d.doc_id, "name": d.name} for d in docs],
            mode=req.mode,
            think=req.think,
            **_deep(req),
            route=retrieved.get("route"),
            route_detail=retrieved.get("route_detail"),
            sources=[{k: s.get(k) for k in ("n", "id", "doc_name", "kind", "location", "score")}
                     for s in retrieved.get("sources", [])],
            chunks_per_document=retrieved.get("per_doc"),
            answer=redact_secrets(answer),
            citations=done.get("citations"),
            invalid_citations=done.get("invalid_citations"),
            finish_reason=done.get("finish_reason"),
            usage=done.get("usage"),
            timings=done.get("timings"),
            model=settings.llm_model,
            error=error,
            notices=[n["code"] for n in notes],
        )


async def _sse(events: AsyncIterator[Event], req: ChatRequest, docs: list[DocumentInfo]) -> AsyncIterator[str]:
    """Stream events to the client, then save and audit the exchange in a tracked task. A finished stream
    waits for it (shielded), so the user's next message always sees this exchange in its thread. A client
    that disconnects cancels the generator; the task is not awaited then and finishes on its own.

    Notices (rag.notices) queued while an event was being produced are sent just before it, so the user reads
    what went wrong next to the part of the answer it concerns."""
    notes = notices.begin()
    answer: list[str] = []
    retrieved: dict = {}
    done: dict = {}
    steps: list[dict] = []
    memory: list[dict] = []
    error = None
    try:
        async for name, data in events:
            if name == ChatEvent.TOKEN:
                answer.append(data["text"])
            elif name == ChatEvent.STEP:
                steps.append(data)
            elif name == ChatEvent.MEMORY:
                memory.append(data)
            elif name == ChatEvent.SOURCES:
                retrieved = data
            elif name == ChatEvent.DONE:
                done = data
            for note in notices.pending():
                yield f"event: {ChatEvent.NOTICE.value}\ndata: {json.dumps(note, ensure_ascii=False)}\n\n"
            yield f"event: {ChatEvent(name).value}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        error = DISCONNECTED
        raise
    except Exception as exc:  # the stream has started; report the failure as an event
        log.exception("chat failed")
        error = f"{type(exc).__name__}: {exc}"
        for note in notices.pending():
            yield f"event: {ChatEvent.NOTICE.value}\ndata: {json.dumps(note, ensure_ascii=False)}\n\n"
        yield f"event: {ChatEvent.ERROR.value}\ndata: {json.dumps({'message': error})}\n\n"
    finally:
        saving = _track(_finish(req, docs, "".join(answer), retrieved, done, steps, memory, error,
                                [{"code": n["code"], "message": n["message"]} for n in notes]))
        if error != DISCONNECTED:
            await asyncio.shield(saving)


@app.post("/v1/chat")
async def chat(req: ChatRequest, _user: Identity = Depends(require_user)) -> StreamingResponse:
    problem = request_problem(settings, req)  # fan-out for an arm without it, research_calls over the limit
    if problem:
        raise HTTPException(422, problem)
    req.doc_ids = list(dict.fromkeys(req.doc_ids))
    if not req.doc_ids and not req.thread_id:  # only thread chats may run without documents
        raise HTTPException(422, "select at least one document")
    docs = await _ready_documents(req.doc_ids)
    try:
        conv: Conversation = await load(state["deps"].threads, req.thread_id, [m.model_dump() for m in req.history])
    except ServiceError as exc:
        raise HTTPException(404 if exc.status == 404 else 502, f"thread: {exc.detail}") from exc
    return StreamingResponse(
        _sse(respond(state["deps"], req, docs, conv), req, docs),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    doc_ids: list[DocId] = Field(min_length=1, max_length=MAX_SELECTED_DOCS)
    retrievers: list[Literal["dense", "bm25", "image", "grep"]] = Field(default_factory=lambda: list(ALL_RETRIEVERS))
    rerank: bool = True
    threshold: bool = False
    coverage: bool = True
    mmr: bool = True
    top_k: int | None = Field(default=None, ge=1, le=50)
    planner: bool = False


@app.post("/v1/search")
async def search(req: SearchRequest, _user: Identity = Depends(require_user)) -> dict:
    deps: Deps = state["deps"]
    doc_ids = list(dict.fromkeys(req.doc_ids))
    docs = await _ready_documents(doc_ids)
    plan = await deps.planner.plan(req.question, docs) if req.planner and any(d.tables for d in docs) else None
    opts = SearchOptions(retrievers=tuple(dict.fromkeys(req.retrievers)), rerank=req.rerank, threshold=req.threshold,
                         coverage=req.coverage, mmr=req.mmr, top_k=req.top_k)
    result = await deps.retrieval.search(req.question, doc_ids, opts, plan.scope if plan else None)
    return {
        "route": plan.route if plan else "search",
        "route_detail": plan.detail if plan else "",
        "sql": [s.card() for s in plan.sources] if plan else [],
        "terms": result.terms,
        "hits": result.hits,
        "timings": {k: round(v, 1) for k, v in result.timings.items()},
        "results": [
            {
                "id": c.id, "rank": i, "doc_id": c.payload.doc_id, "doc_name": c.payload.doc_name,
                "chunk_index": c.payload.chunk_index, "kind": c.payload.kind, "location": location(c.payload),
                "page": c.payload.page, "page_end": c.payload.page_end, "sheet": c.payload.sheet,
                "row_start": c.payload.row_start, "row_end": c.payload.row_end, "asset": c.payload.asset,
                "text_sha": c.payload.text_sha,
                "ranks": c.ranks, "rrf": round(c.rrf, 5), "score": c.score, "text": c.payload.text,
            }
            for i, c in enumerate(result.candidates, 1)
        ],
    }
