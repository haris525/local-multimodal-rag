"""The event stream of POST /v1/chat, written down once (docs/STATE_MACHINES.md, M12).

rag produces it; the web UI, the scripts and eval read it. A turn is a sequence of server-sent events:

    memory*  [token (the memory notes of a mixed message)]  step*  sources  (token | reasoning)*  done

`memory` events come first (thread chats only); one `token` may carry the confirmation of a memory change
before the answer of a mixed message; `step` events come in deep and exhaustive mode; exactly one `sources`;
then the answer's `token` and `reasoning` events and one `done`. An exception after the stream began becomes
one `error` event instead of `done`. A stream that ends without `done` or `error` was cut (the client went away,
or the connection broke). NEXT is that order as a table; Order follows a stream through it.

A `notice` may come at any point before the end, and leaves the phase as it was: it tells the user plainly
that something did not go as planned and what was done instead (the SQL planner fell back to search, a
model reply was cut at its length limit, a figure could not be attached, deep research hit its budget).
The UI shows notices with the answer; they are saved with it in the thread.

Standard library only: the scripts load this file without installing the package.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum


class ChatEvent(StrEnum):
    MEMORY = "memory"  # {action: saved | forgotten, items: [{memory_id, text}]}
    STEP = "step"  # one deep-research or exhaustive step: {tool, args, sources, error}; `tool` is one of STEP_TOOLS,
    # or a tool an optional deep-search arm documents with itself
    SOURCES = "sources"  # {query, route, route_detail, terms, hits, per_doc, sources}
    REASONING = "reasoning"  # {text}
    TOKEN = "token"  # {text}
    DONE = "done"  # {citations, invalid_citations, usage, finish_reason, timings, images}
    ERROR = "error"  # {message}
    NOTICE = "notice"  # {code, message}: anywhere before the end; the phase stays as it was


# The `tool` of a step event and what its `args` hold. `sources` lists the source numbers a research tool
# produced (null for the other steps); `error` is the error text the model read, or null. Every deep-research
# implementation (RAG_DEEP_AGENT) streams the research tools and its own planning and checking steps.
STEP_TOOLS: dict[str, str] = {
    # research tools, the same in every implementation: their call arguments
    "search": "{query}", "read": "{source}", "list_tables": "{}", "describe_table": "{doc_id, table}",
    "sql": "{doc_id, query}", "peek": "{doc_id, table}", "profile_column": "{table, column, doc_id?}",
    "run_python": "{code}: Python run by the sandbox (RAG_DEEP_PYTHON or the request's python; every implementation)",
    "grep": "{terms}: the document grep; in deepagents_stock the harness's file grep {pattern, path?, glob?, output_mode?}",
    "doc_grep": "{terms}: the document grep (deepagents_stock, where grep is the harness's)",
    "verify": "{missing}: what verify sends the research back for",  # langgraph, deepagents
    "plan": "{sub_questions}",  # langgraph: the plan node
    "write_todos": "{todos}: the plan as 'content (status)' lines",  # deepagents, deepagents_stock
    "read_file": "{file_path, offset?, limit?}: lines of a kept file",  # deepagents, deepagents_stock
    "task": "{description}: a sub-agent's sub-question",  # deepagents (RAG_DEEP_SUBAGENTS or fan-out), deepagents_stock
    "summarize": "{messages, kept_in}: older research messages summarized",  # deepagents
    # the stock harness's other file tools, on the request's in-memory files
    "ls": "{path}", "glob": "{pattern, path?}", "write_file": "{file_path, content}",
    "edit_file": "{file_path, old_string, new_string, replace_all?}", "delete": "{path}",
    # exhaustive mode (rag.exhaustive), which reads every section instead of researching with tools: its plan, one
    # step per map call (`sources`: the passages its notes cite; `error`: the call failed) and per merge call
    "sections": "{plan}: what exhaustive mode will read",
    "map": "{sections, where}: one map call over a group of sections",
    "reduce": "{merge}: one merge call of notes that did not fit the answer",
}


class Phase(StrEnum):
    START = "start"
    MEMORY = "memory"
    NOTES = "notes"  # the memory notes of a mixed message, before its sources
    STEPS = "steps"
    SOURCED = "sourced"
    ANSWER = "answer"
    DONE = "done"
    ERRORED = "errored"


TERMINAL = frozenset({Phase.DONE, Phase.ERRORED})
_M, _S, _SRC, _R, _T, _D, _E = (ChatEvent.MEMORY, ChatEvent.STEP, ChatEvent.SOURCES, ChatEvent.REASONING,
                                ChatEvent.TOKEN, ChatEvent.DONE, ChatEvent.ERROR)
# phase: {event: the phase it leads to}. Anything else is a protocol error. An error event may end any phase
# that is not terminal.
NEXT: dict[Phase, dict[ChatEvent, Phase]] = {
    Phase.START: {_M: Phase.MEMORY, _T: Phase.NOTES, _S: Phase.STEPS, _SRC: Phase.SOURCED},
    Phase.MEMORY: {_M: Phase.MEMORY, _T: Phase.NOTES, _S: Phase.STEPS, _SRC: Phase.SOURCED},
    Phase.NOTES: {_S: Phase.STEPS, _SRC: Phase.SOURCED},
    Phase.STEPS: {_S: Phase.STEPS, _SRC: Phase.SOURCED},
    Phase.SOURCED: {_T: Phase.ANSWER, _R: Phase.ANSWER, _D: Phase.DONE},
    Phase.ANSWER: {_T: Phase.ANSWER, _R: Phase.ANSWER, _D: Phase.DONE},
    Phase.DONE: {},
    Phase.ERRORED: {},
}
for _phase in Phase:
    if _phase not in TERMINAL:
        NEXT[_phase][_E] = Phase.ERRORED
        NEXT[_phase][ChatEvent.NOTICE] = _phase


class ProtocolError(ValueError):
    """An event the protocol does not know, or one that may not come at this point of the stream."""


class Order:
    """Follows one stream through NEXT: step() each event's name; raises ProtocolError on a wrong one."""

    def __init__(self):
        self.phase = Phase.START
        self.seen: list[str] = []

    def step(self, name: str) -> Phase:
        try:
            event = ChatEvent(name)
        except ValueError:
            raise ProtocolError(f"unknown event {name!r} after {self.seen[-1:] or 'the start'}") from None
        nxt = NEXT[self.phase].get(event)
        if nxt is None:
            raise ProtocolError(f"{name!r} may not follow {self.phase.value!r} (events so far: {self.seen})")
        self.seen.append(name)
        self.phase = nxt
        return nxt

    @property
    def terminal(self) -> str | None:
        """'done' or 'error' once the stream ended properly, else None."""
        return {Phase.DONE: "done", Phase.ERRORED: "error"}.get(self.phase)


def check_order(names: Iterable[str]) -> Phase:
    """The phase a complete sequence of event names ends in; ProtocolError when it breaks the order."""
    order = Order()
    for name in names:
        order.step(name)
    return order.phase


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """(event, data) pairs from server-sent event lines; a block ends with an empty line, data lines are
    joined with newlines, and a last block without its empty line still counts."""
    event, data = "message", []
    for line in [*lines, ""]:
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


@dataclass
class ChatReply:
    """One chat answer as a client sees it. terminal is 'done', 'error', or None when the stream was cut;
    problem is the first protocol violation (an unknown event, or one out of order)."""

    events: list[tuple[str, dict]] = field(default_factory=list)
    terminal: str | None = None
    problem: str | None = None
    answer: str = ""
    route: str | None = None
    sources: list[dict] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    memory: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    notices: list[dict] = field(default_factory=list)  # {code, message}
    error: str | None = None  # the error event's message

    @property
    def complete(self) -> bool:
        """The answer ended with done, in order."""
        return self.terminal == "done" and self.problem is None

    @property
    def failure(self) -> str | None:
        """Why this is not a complete answer, or None."""
        if self.error:
            return self.error
        if self.problem:
            return f"protocol: {self.problem}"
        if self.terminal is None:
            return "the answer stream ended before its done event"
        return None


def read_chat(lines: Iterable[str]) -> ChatReply:
    """Read a /v1/chat stream to its end. Every event is kept; the usual fields are collected; the order is
    checked against NEXT; terminal says whether (and which) terminal event arrived first."""
    reply, order = ChatReply(), Order()
    for name, data in parse_sse(lines):
        reply.events.append((name, data))
        if reply.problem is None:
            try:
                order.step(name)
            except ProtocolError as exc:
                reply.problem = str(exc)
        if name == ChatEvent.SOURCES:
            reply.sources, reply.route = data.get("sources", []), data.get("route")
        elif name == ChatEvent.TOKEN:
            reply.answer += data.get("text", "")
        elif name == ChatEvent.MEMORY:
            reply.memory.append(data)
        elif name == ChatEvent.STEP:
            reply.steps.append(data)
        elif name == ChatEvent.NOTICE:
            reply.notices.append(data)
        elif name == ChatEvent.DONE:
            reply.terminal = reply.terminal or "done"
            reply.citations = data.get("citations") or []
            reply.invalid_citations = data.get("invalid_citations") or []
            reply.timings = data.get("timings") or {}
        elif name == ChatEvent.ERROR:
            reply.terminal = reply.terminal or "error"
            reply.error = data.get("message") or "error"
    return reply
