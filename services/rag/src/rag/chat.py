"""Document answering: plan -> retrieve -> widen -> generate, streamed as events.

Every answer prompt fits one llm slot: the sources get what the slot leaves after the answer's room, the
figures, the rules, memory, conversation and question (never more than RAG_SOURCE_BUDGET), and the answer's
room is never more than the slot leaves after the prompt as counted.
"""

import copy
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from rag_common.chat_events import ChatEvent
from rag_common.contracts import MAX_SELECTED_DOCS, THREAD_ID, DocId, DocumentInfo

from rag.clients import IngestClient, LLMClient, SandboxClient, ServiceError, SqlClient
from rag.config import Settings
from rag.conversation import Conversation, ThreadsClient
from rag.notices import notice
from rag.planner import Plan, Planner
from rag.prompts import Source, build_messages, build_sources, parse_citations, rewrite_messages
from rag.retrieval import Retrieval, SearchOptions
from rag.tokens import count_tokens, image_count, message_tokens

Event = tuple[ChatEvent, dict]  # the order is rag_common.chat_events.NEXT
log = logging.getLogger("rag.chat")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    doc_ids: list[DocId] = Field(default_factory=list, max_length=MAX_SELECTED_DOCS)  # empty: memory/assistant turns only
    history: list[Message] = Field(default_factory=list, max_length=40)
    think: bool = False
    mode: Literal["single", "deep", "exhaustive"] = "single"  # exhaustive: read every section (rag.exhaustive)
    thread_id: str | None = Field(default=None, pattern=THREAD_ID)  # set: history and memories come from the thread
    # deep mode only, for benchmarking: these override RAG_DEEP_AGENT and RAG_DEEP_THINK for this request.
    # `agent` names an arm of rag.arms (langgraph, deepagents, deepagents_stock, optional ones); others are a 422
    agent: str | None = None
    deep_think: bool | None = None
    # deep mode only: `fanout` overrides RAG_DEEP_FANOUT (a 422 for an arm without fan-out); `research_calls`
    # replaces RAG_AGENT_MAX_STEPS for every arm alike (1 to RAG_AGENT_MAX_STEPS_LIMIT; request_problem)
    fanout: bool | None = None
    research_calls: int | None = Field(default=None, ge=1)
    # deep mode only: `python` asks for the run_python tool (rag.sandbox; every arm has it) when RAG_DEEP_PYTHON is
    # request, or declines it when it is on; with RAG_DEEP_PYTHON=off, `python: true` is a 422
    python: bool | None = None

    @field_validator("agent")
    @classmethod
    def _registered_agent(cls, value: str | None) -> str | None:
        from rag import arms  # at validation time: the arms import this module

        return arms.check(value)


def deep_agent(s: Settings, req: ChatRequest) -> str:
    """The deep-research implementation this request runs: its `agent`, else RAG_DEEP_AGENT."""
    return req.agent or s.deep_agent


def deep_think(s: Settings, req: ChatRequest) -> bool:
    """Whether the plan and verify steps of this request's research think: its `deep_think`, else RAG_DEEP_THINK."""
    return s.deep_think if req.deep_think is None else req.deep_think


def deep_fanout(s: Settings, req: ChatRequest) -> bool:
    """Whether this request's research delegates on reserved budgets (rag.fanout): its `fanout`, else
    RAG_DEEP_FANOUT; always False for an arm that does not support fan-out (request_problem refuses a request that
    asks for it there)."""
    from rag import arms  # at call time: the arms import this module

    if not arms.supports_fanout(deep_agent(s, req)):
        return False
    return s.deep_fanout if req.fanout is None else req.fanout


def deep_python(s: Settings, req: ChatRequest) -> bool:
    """Whether this request's research may run Python (rag.sandbox). RAG_DEEP_PYTHON off: never; request: only when
    the request says `python: true`; on: unless the request says `python: false`."""
    if s.deep_python == "off":
        return False
    if s.deep_python == "request":
        return req.python is True
    return req.python is not False


def research_calls(s: Settings, req: ChatRequest) -> int:
    """The research-tool budget of this request's question: its `research_calls`, else RAG_AGENT_MAX_STEPS."""
    return s.agent_max_steps if req.research_calls is None else req.research_calls


DEEP_ONLY = ("agent", "deep_think", "fanout", "research_calls", "python")  # request fields only Deep search reads


def request_problem(s: Settings, req: ChatRequest) -> str | None:
    """Why the request cannot run with these settings (rag answers 422), or None: research_calls above
    RAG_AGENT_MAX_STEPS_LIMIT, fan-out asked of a deep search whose arm does not support it, Python asked of a
    deployment where it is off (RAG_DEEP_PYTHON=off) or has no sandbox (RAG_SANDBOX_URL empty), or a Deep search
    field in an exhaustive request (it would be ignored, so a benchmark of deep-search variants in exhaustive mode
    would compare nothing)."""
    from rag import arms

    if req.mode == "exhaustive":
        given = [name for name in DEEP_ONLY if getattr(req, name) is not None]
        if given:
            return f"exhaustive mode does not use the Deep search settings {', '.join(given)}; leave them out"
    if req.research_calls is not None and req.research_calls > s.agent_max_steps_limit:
        return (f"research_calls must be between 1 and {s.agent_max_steps_limit} (RAG_AGENT_MAX_STEPS_LIMIT), "
                f"not {req.research_calls}")
    if req.mode == "deep" and req.fanout and not arms.supports_fanout(deep_agent(s, req)):
        return arms.no_fanout(deep_agent(s, req))
    if req.mode == "deep" and req.python and s.deep_python == "off":
        return "python is off on this deployment (RAG_DEEP_PYTHON=off; an operator can set it to request or on)"
    if req.mode == "deep" and req.python and not s.sandbox_url:
        return "python is not available here: RAG_SANDBOX_URL is empty"
    return None


def request_settings(s: Settings, req: ChatRequest) -> Settings:
    """The settings one deep search runs with: `s` itself, unless the request sets research_calls. Then a copy with
    RAG_AGENT_MAX_STEPS replaced by it, so every arm reads the same budget the same way; the calls one task()
    reserves (RAG_DEEP_TASK_CALLS) scaled with it, at least 1; and the LangGraph recursion limit raised by two
    graph steps (a model call and its tools) per extra call."""
    n = req.research_calls
    if n is None or n == s.agent_max_steps:
        return s
    lease = max(1, min(n, round(s.deep_task_calls * n / s.agent_max_steps)))
    out = s.model_copy(update={"agent_max_steps": n, "deep_task_calls": lease,
                               "agent_recursion_limit": s.agent_recursion_limit + 2 * max(0, n - s.agent_max_steps)})
    out._budget_from = "the request's research_calls"
    return out


def with_settings(deps: "Deps", s: Settings) -> "Deps":
    """`deps` with these settings (a shallow copy; `deps` itself when they are its own)."""
    if s is deps.s:
        return deps
    out = copy.copy(deps)
    out.s = s
    return out


@dataclass
class Deps:
    s: Settings
    retrieval: Retrieval
    planner: Planner
    llm: LLMClient
    ingest: IngestClient
    sql: SqlClient
    graph_llm: ChatOpenAI  # LangChain client for the deep-research graph (same local llama-server)
    threads: ThreadsClient
    sandbox: "SandboxClient | None" = None  # the run_python tool's service (rag.sandbox)


def answer_room(s: Settings, think: bool) -> int:
    return s.think_tokens if think else s.answer_tokens


def source_budget(s: Settings, question: str, conv: Conversation, think: bool, images: int) -> int:
    """The tokens an answer prompt has for sources: the slot, less the answer's room, `images` figures, the
    rules, memory, conversation and question, and the margin; never more than s.source_budget."""
    fixed = message_tokens(build_messages(question, [], {}, conv.prompt_block(s.conversation_turns), conv.memory_block()))
    room = s.slot_tokens - s.prompt_margin_tokens - answer_room(s, think) - fixed - images * s.image_tokens
    return max(0, min(s.source_budget, room))


def figure_slots(s: Settings, assets: list[tuple[str, str | None]]) -> int:
    """How many figures an answer may attach: distinct images among the candidates, at most max_prompt_images."""
    return min(s.max_prompt_images, len({a for a in assets if a[1]}))


def per_document(sources: list[Source]) -> dict[str, int]:
    """How many sources each document contributed (shown in the UI and the audit log)."""
    counts: dict[str, int] = {}
    for src in sources:
        counts[src.doc_name] = counts.get(src.doc_name, 0) + 1
    return counts


async def standalone_question(deps: Deps, question: str, conv: Conversation) -> str:
    if not conv.turns and not conv.summary:
        return question
    messages = rewrite_messages(conv.recent(deps.s.history_messages), question, conv.summary)
    try:
        msg = await deps.llm.complete(messages, max_tokens=deps.s.rewrite_tokens, purpose="query rewrite")
    except ServiceError as exc:
        notice("rewrite_failed", "The question could not be rewritten with the conversation's context, so it was "
               "searched as written.", error=exc.detail)
        return question
    rewritten = (msg.get("content") or "").strip().strip('"')
    if msg.get("finish_reason") == "length":  # a cut rewrite is worse than the question itself
        notice("rewrite_cut", "The question could not be rewritten with the conversation's context (the rewrite was "
               "cut at its length limit), so it was searched as written.", max_tokens=deps.s.rewrite_tokens)
        return question
    return rewritten or question


async def figure_images(deps: Deps, sources: list[Source]) -> dict[int, str]:
    """The figures to attach: each image once (two chunks of one image share it), at most max_prompt_images.
    A figure that cannot be fetched is left out and the user is told."""
    images: dict[int, str] = {}
    seen: set[tuple[str, str]] = set()
    for src in sources:
        if not src.asset or (src.doc_id, src.asset) in seen or len(images) >= deps.s.max_prompt_images:
            continue
        seen.add((src.doc_id, src.asset))
        uri = await deps.ingest.asset_data_uri(src.doc_id, src.asset)
        if uri:
            images[src.n] = uri
        else:
            notice("figure_missing", f"The image of source [{src.n}] could not be loaded, so the answer uses its "
                   "written description only.", doc_id=src.doc_id)
    return images


async def stream_answer(deps: Deps, question: str, sources: list[Source], think: bool, timings: dict,
                        conv: Conversation) -> AsyncIterator[Event]:
    """Attach figures, stream the model's reasoning and answer, then report citations."""
    s = deps.s
    images = await figure_images(deps, sources)
    messages = build_messages(question, sources, images, conv.prompt_block(s.conversation_turns), conv.memory_block())
    async for event in stream_reply(deps, messages, [src.n for src in sources], think, timings, images):
        yield event


async def stream_reply(deps: Deps, messages: list[dict], numbers: list[int], think: bool, timings: dict,
                       images: dict[int, str] | None = None) -> AsyncIterator[Event]:
    """Stream the model's reasoning and answer to a finished answer prompt, then the done event: the citations
    checked against the source `numbers` the prompt showed (any other [n] is reported invalid). The answer's
    room is never more than the slot leaves after the prompt; a cut answer is a notice. Every mode ends here."""
    s = deps.s
    t = time.perf_counter()
    parts: list[str] = []
    thoughts: list[str] = []
    usage, finish = None, None
    prompt = message_tokens(messages) + image_count(messages) * s.image_tokens
    limit = answer_room(s, think)
    max_tokens = max(64, min(limit, s.slot_tokens - s.prompt_margin_tokens - prompt))
    if max_tokens < limit:
        log.warning("answer room clamped to what the slot leaves", extra={"fields": {
            "prompt_tokens": prompt, "max_tokens": max_tokens, "setting": limit, "slot_tokens": s.slot_tokens}})
    async for kind, value in deps.llm.stream(messages, max_tokens, think):
        if kind == "token":
            parts.append(value)
            yield ChatEvent.TOKEN, {"text": value}
        elif kind == "reasoning":
            thoughts.append(value)
            yield ChatEvent.REASONING, {"text": value}
        elif kind == "usage":
            usage = value
        elif kind == "finish":
            finish = value
    timings["generate_ms"] = round((time.perf_counter() - t) * 1000)
    if isinstance(usage, dict) and (usage.get("prompt_tokens") or 0) > prompt:
        log.warning("prompt larger than counted", extra={"fields": {"counted": prompt, "actual": usage["prompt_tokens"]}})
    if finish == "length":
        notice("answer_cut", _cut_message(s, think, max_tokens, limit, usage, count_tokens("".join(thoughts))),
               max_tokens=max_tokens, setting=limit, prompt_tokens=prompt,
               completion_tokens=(usage or {}).get("completion_tokens") if isinstance(usage, dict) else None)
    cited, invalid = parse_citations("".join(parts), numbers)
    yield ChatEvent.DONE, {"citations": cited, "invalid_citations": invalid, "usage": usage, "finish_reason": finish,
                           "timings": timings, "images": sorted(images or {})}


def _cut_message(s: Settings, think: bool, max_tokens: int, limit: int, usage: object, thought: int) -> str:
    """Why the answer stopped, from what was actually used."""
    used = (usage.get("completion_tokens") if isinstance(usage, dict) else None) or max_tokens
    setting = "RAG_THINK_TOKENS" if think else "RAG_ANSWER_TOKENS"
    if max_tokens < limit:
        why = (f"The answer used all {used} tokens the prompt left in the model's context (its limit is {limit}, "
               f"{setting}; the sources and conversation took the rest) and stops mid-way.")
    else:
        why = f"The answer reached its length limit ({used} of {limit} tokens, {setting}) and stops mid-way."
    if think and thought * 2 > used:
        why += f" Thinking took about {thought} of those tokens."
    return why + " Ask for the rest, or narrow the question."


def nothing_found(timings: dict) -> list[Event]:
    """The answer after an empty sources event."""
    return [(ChatEvent.TOKEN, {"text": "I could not find anything in the selected documents that relates to this question."}),
            (ChatEvent.DONE, {"citations": [], "invalid_citations": [], "usage": None, "finish_reason": "no_sources",
                              "timings": timings, "images": []})]


async def single_shot(deps: Deps, req: ChatRequest, docs: list[DocumentInfo], conv: Conversation,
                      question: str, timings: dict) -> AsyncIterator[Event]:
    t = time.perf_counter()
    plan = await deps.planner.plan(question, docs) if any(d.tables for d in docs) else Plan()
    timings["plan_ms"] = round((time.perf_counter() - t) * 1000)

    result = await deps.retrieval.search(question, req.doc_ids, SearchOptions(), plan.scope)
    timings.update({k: round(v) for k, v in result.timings.items()})
    neighbours = await deps.retrieval.neighbours(result.candidates)
    images = figure_slots(deps.s, [(c.payload.doc_id, c.payload.asset) for c in result.candidates])
    budget = source_budget(deps.s, question, conv, req.think, images)
    sources = build_sources(result.candidates, neighbours, plan.sources, budget)
    if plan.sources and len(sources) < len(plan.sources):
        notice("sources_dropped", "A SQL result did not fit the answer's prompt budget and was left out.")

    yield ChatEvent.SOURCES, {"query": question, "route": plan.route, "route_detail": plan.detail,
                              "terms": result.terms, "hits": result.hits, "per_doc": per_document(sources),
                              "sources": [s.card() for s in sources]}
    if not sources:
        for event in nothing_found(timings):
            yield event
        return
    async for event in stream_answer(deps, question, sources, req.think, timings, conv):
        yield event
