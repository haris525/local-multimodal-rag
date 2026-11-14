"""One chat turn: route the message, carry out memory requests, then answer.

Thread chats are routed first, so memory requests, questions about the assistant and questions about the
conversation never start a document search or a research run, whatever the Think and Deep search
settings are. Stateless calls (evaluation, API) skip routing and never touch memory.

Deep search runs one of the arms of rag.arms (the LangGraph graph, the deepagents harness tuned or out of the
box, and any optional arm), chosen by RAG_DEEP_AGENT or, for benchmarking, by the request's `agent`. A request's
`research_calls` reaches whichever arm runs as its RAG_AGENT_MAX_STEPS (rag.chat.request_settings).

Exhaustive mode (a request's mode "exhaustive", rag.exhaustive) reads every section of the selected documents
instead of searching them; it is routed like the other two, after the memory part of a thread chat.
"""

import time
from collections.abc import AsyncIterator
from typing import assert_never

from rag_common.chat_events import ChatEvent
from rag_common.contracts import DocumentInfo

from rag import arms
from rag.chat import (ChatRequest, Deps, Event, deep_agent, request_settings, single_shot, standalone_question,
                      with_settings)
from rag.conversation import Conversation
from rag.exhaustive import exhaustive_answer
from rag.intent import Intent, apply_memory, memory_reply, route

# Every route a turn can report (the `route` of its sources event): the planner's three, deep research, exhaustive
# mode, and the turns answered without a document search. The web UI has a label for each (app.js ROUTE_LABEL).
DIRECT_ROUTES = ("memory_save", "memory_recall", "memory_forget", "history", "assistant", "no_documents")
ROUTES = ("sql", "filter", "search", "deep", "exhaustive", *DIRECT_ROUTES)


def direct(route: str, query: str, text: str, timings: dict) -> list[Event]:
    """A complete answer that needs no documents: an empty sources event, the text, done."""
    return [
        (ChatEvent.SOURCES, {"query": query, "route": route, "route_detail": "", "terms": [], "hits": {},
                             "per_doc": {}, "sources": []}),
        (ChatEvent.TOKEN, {"text": text}),
        (ChatEvent.DONE, {"citations": [], "invalid_citations": [], "usage": None, "finish_reason": route,
                          "timings": timings, "images": []}),
    ]


_ABOUT = """You are the assistant of a private document app that runs entirely on the user's computer; nothing
is sent to the internet. What you can do: answer questions from the documents the user selects (up to 10),
with numbered citations; read tables (with exact SQL), figures, charts and scanned pages; research step by
step when Deep search is on, and reason longer when Think is on. You also keep a long-term memory of the
user across chats: when they say "remember that ...", it is saved immediately, and they can review or delete
everything in the Memory panel. Reply briefly and warmly. Never invent document content; if they want
information from their documents, invite them to ask the question.

{memory}"""

_HISTORY = """Answer the user's question about this conversation using only the conversation below. If the
conversation does not contain the answer, reply with exactly NEED_SEARCH.

{conversation}

Question: {question}"""


async def respond(deps: Deps, req: ChatRequest, docs: list[DocumentInfo], conv: Conversation) -> AsyncIterator[Event]:
    """One chat turn. Thread chats are routed first, so memory requests, questions about the assistant and
    questions about the conversation never trigger a document search or a research run."""
    timings: dict[str, float] = {}
    t = time.perf_counter()
    if conv.thread_id:
        intent = await route(deps.llm, req.question, conv, documents=len(req.doc_ids), max_tokens=deps.s.intent_tokens)
        outcome = await apply_memory(deps.threads, intent, conv, deps.retrieval.models.embed_documents,
                                     deps.s.memory_duplicate_cosine)
        for event in outcome.events():
            yield event
    else:  # stateless calls (evaluation, API) never touch memory
        intent, outcome = Intent("documents", question=await standalone_question(deps, req.question, conv)), None
    timings["route_ms"] = round((time.perf_counter() - t) * 1000)

    match intent.kind:  # every kind is handled here; a kind added to intent.Kind fails loudly, never searches
        case "memory_save" | "memory_recall" | "memory_forget":
            for event in direct(intent.kind, req.question, memory_reply(intent, outcome, conv), timings):
                yield event
            return
        case "assistant":
            messages = [{"role": "system", "content": _ABOUT.format(memory=conv.memory_block())},
                        {"role": "user", "content": (conv.prompt_block(deps.s.conversation_turns) + "\n\n"
                                                     if conv.turns else "") + req.question}]
            msg = await deps.llm.complete(messages, max_tokens=deps.s.assistant_tokens, temperature=0.5,
                                          purpose="assistant reply")
            reply = (msg.get("content") or "").strip()
            notes = memory_reply(intent, outcome, conv) if outcome and (outcome.saved or outcome.replaced) else ""
            for event in direct("assistant", req.question, "\n\n".join(filter(None, [notes, reply])), timings):
                yield event
            return
        case "history":
            transcript = conv.prompt_block(deps.s.history_messages * 2, max_chars=2000)
            msg = await deps.llm.complete([{"role": "user", "content": _HISTORY.format(
                conversation=transcript or "(empty)", question=req.question)}], max_tokens=deps.s.history_answer_tokens,
                purpose="answer from the chat")
            answer = (msg.get("content") or "").strip()
            if answer and "NEED_SEARCH" not in answer:
                for event in direct("history", req.question, answer, timings):
                    yield event
                return
            # the conversation does not hold it: search the documents instead
        case "documents":
            pass
        case _:
            assert_never(intent.kind)

    notes = memory_reply(intent, outcome, conv) if outcome and (intent.memory_kind or outcome.saved or outcome.replaced
                                                                 or outcome.already or intent.refused_secret) else ""
    if not req.doc_ids:
        text = "\n\n".join(filter(None, [notes, "Select at least one document (up to 10) to search, then ask again."]))
        for event in direct("no_documents", req.question, text, timings):
            yield event
        return
    if notes:  # a mixed message: confirm the memory part, then answer the question (NEXT: start -> notes)
        yield ChatEvent.TOKEN, {"text": notes + "\n\n"}
    if req.mode == "deep":  # every arm reads the request's research budget the same way (request_settings)
        run, deps = arms.get(deep_agent(deps.s, req)).deep_search, with_settings(deps, request_settings(deps.s, req))
    elif req.mode == "exhaustive":
        run = exhaustive_answer
    else:
        run = single_shot
    async for event in run(deps, req, docs, conv, intent.question, timings):
        yield event
