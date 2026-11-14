"""Deep research on the deepagents harness (RAG_DEEP_AGENT=deepagents, or a chat request's agent="deepagents").

    agent (plans with write_todos) <-> tools -> verify (text documents read? todos open?) -> agent | answer

The same tools as rag.agent (its TOOLS registry and ToolContext, so the same per-user scoping, SQL sandbox,
repeated-call answers and token caps), the same numbered source book, and the same cited answer step
(rag.agent.answer_from_book: the same sources and done events). What differs is the loop around the tools: a
langchain create_agent loop with the harness's middleware (deepagents 0.5.9):

- TodoListMiddleware: the agent writes its plan with write_todos (1 to 4 sub-questions, the plan rules of the
  LangGraph agent) and marks each one completed.
- FilesystemMiddleware on a StateBackend: a tool result over RAG_DEEP_EVICT_TOKENS (counted in the model's
  tokens) is kept in /large_tool_results/<call id> and the model sees its first and last lines; read_file reads
  lines of it. The files are a channel of this request's graph state: nothing is written to disk, and nothing
  outlives the request or is visible to another one (each request builds its own graph, with no checkpointer).
- SubAgentMiddleware (RAG_DEEP_SUBAGENTS): task() runs a researcher with the same tools on one sub-question in
  a context of its own and returns its findings with their source numbers.
- SummarizationMiddleware: once the research prompt passes RAG_DEEP_SUMMARY_TOKENS, the oldest messages are
  summarized (figures and source numbers kept) and moved to /conversation_history in the same virtual files.
- PatchToolCallsMiddleware: a tool call left without a result gets one before the loop goes on.

create_deep_agent itself is not called: its built-in prompts would take a large share of the 32,768-token slot
(tests/test_deepagents.py counts them), its filesystem grep would clash with the document grep, and its
Anthropic prompt caching does nothing for llama-server. The same middleware is composed directly, with
trimmed prompts that keep the LangGraph agent's research rules word for word (rag.agent._SYSTEM).

Three middleware of our own complete it. Research (outermost on tool calls): the research-tool budget
(RAG_AGENT_MAX_STEPS, shared with sub-agents), the harness-tool budget (RAG_DEEP_HARNESS_CALLS), tool errors as
messages and the `step` events; with fan-out (rag.fanout) also the leases of the researchers task() starts, their
concurrency, time limit and the cap on their reports. Verify (after a reply without tool calls): for a mixed
selection with no text passage yet it searches the text documents itself; it lists the todos still open and asks
for one check of the evidence against every todo, at most RAG_AGENT_VERIFY_ROUNDS times and never again when the
last round found nothing new. ModelStep (innermost on model calls): the slot guard counted in the model's tokens, one
retry of a busy, failed or cut step, and thinking for the plan step (the first call) and the verify steps when
RAG_DEEP_THINK or a request's deep_think asks. Budgets end the research with a notice, as in rag.agent.

Fan-out (RAG_DEEP_FANOUT or a request's fanout) turns the researcher sub-agent on whatever RAG_DEEP_SUBAGENTS says,
tells the lead its budget and what a researcher reserves, and asks it to delegate first (FANOUT_RULES).
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import openai
from deepagents.backends import StateBackend
from deepagents.backends.utils import sanitize_tool_call_id
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware, hook_config
from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import TOOL_CALL_ERROR_TEMPLATE
from langgraph.types import Command

from rag_common.chat_events import ChatEvent
from rag_common.contracts import DocumentInfo
from rag_common.logging import log_fields
from rag_common.retry import TRANSIENT_STATUS

from rag.agent import (_SYSTEM, SourceBook, TextSearch, ToolContext, _documents, _tools, answer_from_book,
                       text_documents, text_searcher, thinking, tools_tokens)
from rag import fanout
from rag.chat import ChatRequest, Deps, Event, deep_fanout, deep_think
from rag.config import Settings
from rag.conversation import Conversation
from rag.fanout import FanOut
from rag.notices import notice
from rag.sandbox import close as close_sandbox
from rag.tokens import count_tokens, message_tokens, truncate_tokens

log = logging.getLogger("rag.agent_deepagents")

# --- prompts: the harness's built-in ones, trimmed for the slot and a local 27B model ----------------------

BASE = """Be brief: no preamble and no plans in prose; act with tools. Check your work against the question,
not against your own output. When a call fails twice, find out why instead of repeating it."""

TASK = """## The question
{question}

Selected documents:
{documents}"""

TODO_RULES = """## write_todos
Plan first: call write_todos with the smallest set of sub-questions (1 to 4) that together answer the question;
a simple question is its own single sub-question. When the question refers to several documents, write at least
one for each of them. When text documents (PDF, Word, text, images) and tables are both selected, also ask what
the text documents say about it, even when a table seems to hold the answer. Keep the question's own terms,
periods and units. Mark a todo completed as soon as its evidence is found or the documents do not have it,
together with your next tool call; never call write_todos twice in one reply."""

TODO_TOOL = ("Write the research plan, or update it: the whole list of todos, each with content and a status "
             "(pending, in_progress or completed). Each call replaces the list.")

FILES_RULES = """## Large results
A result over {tokens} tokens is kept in /large_tool_results/<call id>: you see its first and last lines with
their line range and its source numbers, which you can cite. read_file(file_path, offset, limit) reads lines of
it (offset counts lines from 0). Read only the lines you need."""

READ_FILE_TOOL = ("Read lines of a kept result (/large_tool_results/...) or of the summarized research "
                  "(/conversation_history/...): offset is the number of lines to skip, limit the number of lines "
                  "(default 100). A '@@ lines A-B of T | next offset N @@' header says which lines came back.")

TASK_RULES = """## task
task(description, subagent_type="researcher") runs a researcher with the same tools on one sub-question in a
context of its own; it returns its findings with source numbers [n] that you can cite. Describe one
sub-question, the documents and tables it concerns, and what to return. Use it for a sub-question that needs
several steps of its own; make single calls yourself. Several task calls in one reply run at the same time."""

FANOUT_RULES = """## task
You have {calls} research calls for this question. task(description, subagent_type="researcher") starts a
researcher with the same tools on one sub-question, in a context of its own, and reserves {lease} of your calls for
it (at most {tasks} researchers); it returns its findings with source numbers [n] that you can cite, and the calls
it does not use come back to you. Delegate first: after write_todos, when the question concerns several documents
or independent sub-questions, call task once per document or sub-question, all in the same reply, before you
search yourself. Describe the sub-question, the documents and tables it concerns, and what to return. Then check
their findings with the calls left. A question about one document stays with you: research it yourself."""

TASK_TOOL = """Run a researcher on one sub-question with the document tools; it returns its findings with their
source numbers. Agent types:
{available_agents}"""

RESEARCHER = """You research one sub-question of the lead researcher's question (below), with the tools above.
When you have the evidence, reply with your findings: every figure and fact with its source number [n], and what
you could not find. Do not reply DONE alone."""

RESEARCHER_BUDGET = """You have {lease} research calls. Your reply reaches the lead cut at {tokens} tokens: findings
first, in short lines."""

SUMMARY = """Summarize this research so far for the researcher who continues it. Keep the question; the todo
list with each item's status; every figure, name and date found, each with its source number [n]; the SQL that
worked with its table and column names; what failed and why; what is still missing. Short plain lines, no
introduction.

<messages>
{messages}
</messages>"""

EVICTED = """Source(s) {sources}: this result is {tokens} tokens, so it was kept in {path} ({lines} lines). Its
first and last lines are below; read_file(file_path="{path}", offset=..., limit=...) reads the others.
{preview}"""

CHECK = ("Check the results above against every todo and the question. Research what is still missing (mark a "
         "todo completed when the documents do not have it), then reply DONE.")


def system_prompt(s: Settings, question: str, docs: list[DocumentInfo]) -> str:
    """The research rules of rag.agent word for word, the harness's base rules trimmed, and the question."""
    return "\n\n".join([_SYSTEM.format(sql_rows=s.sql_max_rows), BASE,
                        TASK.format(question=question, documents=_documents(docs))])


# --- one run ------------------------------------------------------------------------------------------------


@dataclass
class Run:
    """One research run: budgets, flags and the step events, shared by the agent and its sub-agents."""

    s: Settings
    book: SourceBook
    think: bool
    research: frozenset[str]  # names of the research tools (rag.agent.TOOLS); the rest are the harness's
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    tool_calls: int = 0
    harness_calls: int = 0
    model_calls: int = 0  # the lead agent's
    stopped: bool = False  # a budget or the slot is used up: no more steps, verify sends no one back
    rounds: int = 0
    verified_at: int = -1
    verifying: bool = False  # the next model call answers a verify request
    text_searched: bool = False
    fanout: FanOut | None = None  # the researchers' leases (rag.fanout), when fan-out is on

    def step(self, tool: str, args: dict, sources: list[int] | None = None, error: str | None = None) -> None:
        self.events.put_nowait((ChatEvent.STEP, {"tool": tool, "args": args, "sources": sources, "error": error}))


def _todos(value) -> list[dict]:
    """A todo list as dicts whatever the model sent (as rag.agent._strings for its JSON fields): a list of
    todos or of strings, or one string; anything else is empty. Invalid write_todos arguments are refused by
    the tool's schema, but their step event still shows them."""
    items = value if isinstance(value, list) else [value] if isinstance(value, str) and value.strip() else []
    out = []
    for item in items:
        if isinstance(item, dict) and str(item.get("content", "")).strip():
            out.append({"content": str(item["content"]).strip(), "status": str(item.get("status") or "pending")})
        elif isinstance(item, str | int | float) and str(item).strip():
            out.append({"content": str(item).strip(), "status": "pending"})
    return out


def _todo_lines(todos) -> list[str]:
    return [f"{t['content']} ({t['status']})" for t in _todos(todos)]


def open_todos(todos) -> list[str]:
    return [t["content"] for t in _todos(todos) if t["status"] != "completed"]


def flat(message: SystemMessage | None) -> SystemMessage | None:
    """The middleware append their sections as content blocks; the chat template and the token count want one
    string."""
    if message is None or isinstance(message.content, str):
        return message
    return SystemMessage("".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in message.content))


def halt(run: Run) -> None:
    """No room for another step: the research ends; with fan-out, inside a researcher, only that researcher."""
    lease = fanout.current()
    if lease is not None:
        lease.stopped = True
    else:
        run.stopped = True


def request_tokens(system: SystemMessage | None, messages: list[AnyMessage], tools: list) -> int:
    """A research prompt as the model counts it: tool schemas, system message and every message so far."""
    return tools_tokens(tools) + message_tokens(([system] if system is not None else []) + list(messages))


class Research(AgentMiddleware):
    """Budgets, tool errors and step events. Outermost on tool calls, so it sees every result as the model will
    (after eviction)."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        run, s = self.run, self.run.s
        lease = fanout.current()
        if lease is not None:  # a researcher with fan-out: its own lease ends it, not the question's budget
            return {"jump_to": "end"} if run.stopped or lease.done() else None
        if not run.stopped and run.tool_calls >= s.agent_max_steps:
            run.stopped = True
            notice("research_budget", f"Deep search used all {s.agent_max_steps} of its tool calls; the answer uses "
                   f"what it found so far ({s.research_budget_from}).")
        return {"jump_to": "end"} if run.stopped else None

    def _refusal(self, name: str) -> str | None:
        """None when the call may run (and is counted), else why not. Counting happens before any await, so
        calls of one reply, run at the same time, never overrun a budget. With fan-out a researcher spends its
        lease, the lead never spends a reserved call, and a task call needs a lease (rag.fanout)."""
        run, s = self.run, self.run.s
        lease = fanout.current()  # always None without fan-out
        if name in run.research:
            if lease is not None:
                refusal = lease.take()
                if refusal is None:
                    run.tool_calls += 1
                return refusal
            if run.tool_calls >= s.agent_max_steps:
                return "Not run: the research tool budget is used up. Stop calling tools and reply DONE."
            if run.fanout is not None and (refusal := run.fanout.lead_refusal(run.tool_calls)):
                return refusal
            run.tool_calls += 1
            return None
        if lease is not None and name == "task":
            return fanout.NESTED
        if run.harness_calls >= s.deep_harness_calls:
            if not run.stopped:
                run.stopped = True
                notice("research_harness_budget", f"Deep search used all {s.deep_harness_calls} of its planning, "
                       "reading and sub-agent calls; the answer uses what it found so far (RAG_DEEP_HARNESS_CALLS).")
            return "Not run: the budget for write_todos, read_file and task is used up. Reply DONE."
        if name == "task" and run.fanout is not None and (refusal := run.fanout.task_refusal(run.tool_calls)):
            return refusal
        run.harness_calls += 1
        return None

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        name, args = call["name"], call.get("args") or {}
        refusal = self._refusal(name)
        fan = self.run.fanout
        lease = fan.reserve() if refusal is None and name == "task" and fan is not None else None  # no await between
        if refusal:
            result = ToolMessage(refusal, name=name, tool_call_id=call["id"], status="error")
        elif lease is not None:
            result = await self._delegate(request, handler, lease)
        else:
            try:
                result = await handler(request)
            except Exception as exc:  # noqa: BLE001 - the model reads the error and corrects the call (as rag.agent)
                result = ToolMessage(TOOL_CALL_ERROR_TEMPLATE.format(error=repr(exc)), name=name,
                                     tool_call_id=call["id"], status="error")
        self._report(name, args, result)
        return result

    async def _delegate(self, request, handler, lease: fanout.Lease):
        """One researcher on its lease (rag.fanout): it waits for a free place (RAG_DEEP_TASK_CONCURRENCY), runs
        within RAG_DEEP_TASK_TIMEOUT_S, and its report is capped at RAG_DEEP_TASK_RETURN_TOKENS. On a timeout or a
        failure the lead reads an error and the user gets a notice; its sources stay in the book either way."""
        fan, call = self.run.fanout, request.tool_call
        ran = True
        token = fanout.enter(lease)  # the researcher's graph copies the context, so its middleware sees the lease
        try:
            async with fan.gate:
                result = await asyncio.wait_for(handler(request), fan.timeout_s)
            ran = isinstance(result, Command)  # else no researcher started (an unknown subagent type)
            if ran:
                result = self._capped(result, lease)
        except TimeoutError:
            result = self._failed(call, lease, fanout.TIMED_OUT.format(timeout=fan.timeout_s,
                                                                       found=fanout.found_line(lease.sources)),
                                  f"ran past its time limit ({fan.timeout_s:g} s, RAG_DEEP_TASK_TIMEOUT_S) and was "
                                  "stopped", error="timeout")
        except Exception as exc:  # noqa: BLE001 - the lead reads the error; the research goes on
            log.exception("a researcher failed")
            result = self._failed(call, lease, fanout.FAILED.format(error=type(exc).__name__,
                                                                    found=fanout.found_line(lease.sources)),
                                  f"failed ({type(exc).__name__})", error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            fanout.leave(token)
            fan.release(lease, ran)
        return result

    def _capped(self, result: Command, lease: fanout.Lease) -> Command:
        """The researcher's report (the task's tool message) within RAG_DEEP_TASK_RETURN_TOKENS, with a note when
        cut or empty."""
        update = result.update if isinstance(result.update, dict) else {}
        messages = list(update.get("messages") or [])
        if not messages or not isinstance(messages[-1], ToolMessage):
            return result
        fan = self.run.fanout
        text, tokens = fan.report(messages[-1].text, lease)
        if tokens is not None:
            notice("research_task_cut", f"A researcher's report was {tokens} tokens, longer than the "
                   f"{fan.return_tokens} the lead reads (RAG_DEEP_TASK_RETURN_TOKENS), so it was cut; every source it "
                   "found is kept for the answer.", tokens=tokens, limit=fan.return_tokens)
        messages[-1] = messages[-1].model_copy(update={"content": text})
        return dataclasses.replace(result, update={**update, "messages": messages})

    def _failed(self, call: dict, lease: fanout.Lease, text: str, why: str, **fields) -> ToolMessage:
        notice("research_task_failed", f"A researcher {why}; the sources it found are kept and the answer can use "
               "them.", sources=lease.sources, calls=lease.used, **fields)
        return ToolMessage(text, name=call["name"], tool_call_id=call["id"], status="error")

    def _report(self, name: str, args: dict, result: ToolMessage | Command) -> None:
        if isinstance(result, Command):  # write_todos (the list) or task (the sub-agent's report)
            update = result.update if isinstance(result.update, dict) else {}
            if name == "write_todos" and "todos" in update:
                self.run.step("write_todos", {"todos": _todo_lines(update["todos"])})
            else:
                self.run.step(name, {k: v for k, v in args.items() if k != "subagent_type"})
            return
        error = result.text if result.status == "error" else None
        if name == "write_todos":
            args = {"todos": _todo_lines(args.get("todos", []))}
        sources = (result.artifact or []) if name in self.run.research else None  # as rag.agent reports them
        lease = fanout.current()
        if lease is not None:
            lease.found(sources)
        self.run.step(name, args, sources, error)


class Evicting(FilesystemMiddleware):
    """The harness's filesystem on a StateBackend (the request's graph state), with read_file only, and
    eviction of research results measured in the model's tokens (the harness counts 4 characters a token;
    Qwen writes each digit as a token, so a SQL result would be counted at half its size)."""

    def __init__(self, s: Settings, research: frozenset[str]):
        # read_file only: its grep would clash with the document grep
        super().__init__(backend=StateBackend(), system_prompt=FILES_RULES.format(tokens=s.deep_evict_tokens),
                         custom_tool_descriptions={"read_file": READ_FILE_TOOL}, tools=["read_file"],
                         tool_token_limit_before_evict=s.deep_evict_tokens, human_message_token_limit_before_evict=None)
        self.s, self.research = s, research

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if (request.tool_call["name"] not in self.research or not isinstance(result, ToolMessage)
                or result.status == "error" or not isinstance(result.content, str)):
            return result
        text = result.content
        tokens = count_tokens(text)
        if tokens <= self.s.deep_evict_tokens:
            return result
        path = f"{self._large_tool_results_prefix}/{sanitize_tool_call_id(result.tool_call_id)}"
        written = await self.backend.awrite(path, text)
        if written.error:
            log_fields(log, "a large result could not be kept; it stays inline", logging.WARNING, error=written.error)
            return result
        content = EVICTED.format(sources=result.artifact or "none", tokens=tokens, path=path,
                                 lines=len(text.splitlines()), preview=preview(text, self.s.deep_preview_tokens))
        return ToolMessage(content, tool_call_id=result.tool_call_id, name=result.name, id=result.id,
                           artifact=result.artifact, status=result.status)

    def wrap_tool_call(self, request, handler):  # research runs async only; never evict by characters
        return handler(request)


def preview(text: str, budget: int) -> str:
    """The first lines (title, query, header and first rows) and the last 5, each block under the line-range
    header read_file uses ('@@ lines A-B of T @@'), within `budget` tokens."""
    lines = text.splitlines()
    total = len(lines)
    tail = lines[-5:] if total > 20 else []
    head = lines[:total - len(tail)]
    tail_text = f"@@ lines {total - len(tail) + 1}-{total} of {total} @@\n" + "\n".join(tail) if tail else ""
    room = max(0, budget - count_tokens(tail_text) - 30)
    shown: list[str] = []
    for line in head:
        if count_tokens("\n".join(shown + [line])) > room:
            break
        shown.append(line)
    more = f" | next offset {len(shown)}" if len(shown) < total else ""
    out = f"@@ lines 1-{len(shown)} of {total}{more} @@\n" + "\n".join(shown)
    hidden = total - len(shown) - len(tail)
    if hidden:
        out += f"\n... [{hidden} lines not shown] ..."
    if tail_text:
        out += "\n" + tail_text
    return truncate_tokens(out, budget)


class Summarizing(SummarizationMiddleware):
    """The harness's summarization, counted in the model's tokens, with a step event when it summarizes."""

    def __init__(self, run: Run, model, counter: Callable):
        s = run.s
        super().__init__(model, backend=StateBackend(), trigger=("tokens", s.deep_summary_tokens),
                         keep=("tokens", s.deep_summary_keep_tokens), token_counter=counter, summary_prompt=SUMMARY,
                         trim_tokens_to_summarize=s.deep_summary_input_tokens)
        self.run = run

    async def awrap_model_call(self, request, handler):
        response = await super().awrap_model_call(request, handler)
        if isinstance(response, ExtendedModelResponse) and response.command is not None:
            event = (response.command.update or {}).get("_summarization_event") or {}
            self.run.step("summarize", {"messages": event.get("cutoff_index"), "kept_in": event.get("file_path")})
        return response


class ModelStep(AgentMiddleware):
    """Innermost on model calls: the slot guard, retries and thinking (as rag.agent.invoke and its agent node)."""

    def __init__(self, run: Run, lead: bool):
        super().__init__()
        self.run, self.lead = run, lead

    async def awrap_model_call(self, request, handler):
        run, s = self.run, self.run.s
        think = self.lead and run.think and (run.model_calls == 0 or run.verifying)
        if self.lead:
            run.model_calls += 1
            run.verifying = False
        room = s.deep_think_tokens if think else s.agent_tokens
        system = flat(request.system_message)
        prompt = request_tokens(system, request.messages, request.tools)
        if prompt + max(room, s.research_retry_tokens) + s.prompt_margin_tokens > s.slot_tokens:
            halt(run)  # verify sends no one back: there is no room for another step
            notice("research_context_full", "Deep search found more material than one model call can hold; it "
                   "stopped researching and answers from what it has.", prompt_tokens=prompt, slot_tokens=s.slot_tokens)
            return AIMessage("Research prompt full. DONE")
        settings = {**request.model_settings, **(thinking(s) if think else {}), "max_tokens": room}
        request = request.override(system_message=system, model_settings=settings)
        try:
            response = await self._call(handler, request)
            if _finish(response) == "length" and room < s.retry_max_tokens:
                more = min(s.retry_max_tokens, max(room + 1, int(room * s.retry_tokens_factor)))
                log_fields(log, "research step: reply cut at its token limit; retrying once with more room",
                           logging.WARNING, max_tokens=room, retry_tokens=more)
                response = await self._call(handler, request.override(model_settings={**settings, "max_tokens": more}))
        except Exception as exc:  # noqa: BLE001 - the answer step still runs on the sources found so far
            log.exception("research step failed")
            notice("research_step_failed", f"A deep-search step failed ({type(exc).__name__}); the answer uses what "
                   "was found before it.")
            return AIMessage("Research step failed. DONE")
        if _finish(response) == "length":
            notice("research_step_cut", "A deep-search step was cut at its length limit even after a retry with more "
                   f"room ({s.research_retry_tokens} tokens; RAG_AGENT_TOKENS, RAG_RETRY_MAX_TOKENS); research goes on "
                   "with what that step managed to ask for.", prompt_tokens=prompt)
        return response

    async def _call(self, handler, request):
        """A busy, failed or timed-out call is retried once after the backoff (as rag.agent.invoke)."""
        try:
            return await handler(request)
        except Exception as exc:
            if not _transient(exc):
                raise
            log_fields(log, "research step failed; retrying once", logging.WARNING,
                       error=f"{type(exc).__name__}: {exc}"[:300])
            await asyncio.sleep(self.run.s.retry_backoff_s)
            return await handler(request)


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in TRANSIENT_STATUS
    return isinstance(exc, openai.APIConnectionError | openai.APITimeoutError)


def _finish(response) -> str | None:
    messages = response.result if isinstance(response, ModelResponse) else [response]
    last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    return (last.response_metadata or {}).get("finish_reason") if last is not None else None


class Verify(AgentMiddleware):
    """When the agent stops calling tools: cover the text documents of a mixed selection, list the open todos
    and ask for one check against every todo; bounded like rag.agent's verify."""

    def __init__(self, run: Run, cover: Callable):
        super().__init__()
        self.run, self.cover = run, cover

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        last = state["messages"][-1] if state.get("messages") else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        run, s = self.run, self.run.s
        if not run.book.sources or run.stopped or run.rounds >= s.agent_verify_rounds:
            return None  # (a stopped run's text documents are covered after it, deep_search)
        covered = await self.cover()
        if covered is None and run.rounds and len(run.book.sources) == run.verified_at:
            return None  # nothing new since the last check: its list still holds
        run.verified_at = len(run.book.sources)
        missing = open_todos(state.get("todos"))
        parts = [covered[0]] if covered else []
        if missing:
            parts.append("These todos are still open:\n" + "\n".join(f"- {m}" for m in missing))
        parts.append(CHECK)
        run.rounds += 1
        run.verifying = True
        run.step("verify", {"missing": ([covered[1]] if covered else []) + missing or ["check every todo"]})
        return {"messages": [HumanMessage("\n\n".join(parts))], "jump_to": "model"}


def coverage(run: Run, docs: list[DocumentInfo], question: str, text_search: TextSearch) -> Callable:
    """rag.agent's check of a mixed selection: when no passage of a text document was found, search the text
    documents once. Returns (note for the agent, what was missing) or None."""
    text_docs = text_documents(docs)  # as rag.agent: a PDF with a table is text
    mixed = bool(text_docs) and any(d.tables for d in docs)

    async def cover() -> tuple[str, str] | None:
        if not mixed or run.text_searched or any(src.doc_id in text_docs for src in run.book.sources):
            return None
        run.text_searched = True
        try:
            rendered, numbers = await text_search(question)
        except Exception as exc:  # noqa: BLE001 - the answer still runs on the table results
            log.exception("text-document search failed")
            notice("text_search_failed", f"Deep search could not search the text documents ({type(exc).__name__}); "
                   "the answer uses the table results only.")
            return None
        if not numbers:
            return None
        names = ", ".join(sorted(set(text_docs.values())))
        return (f"The text documents ({names}) had not been searched yet. A search of them for the question found "
                f"these sources:\n{rendered}\nCompare them with the table results: when they give a different "
                "figure, both must be reported."), f"what the text documents say ({names})"
    return cover


# --- the agent ----------------------------------------------------------------------------------------------


def build_agent(llm, tools: list[BaseTool], run: Run, docs: list[DocumentInfo], question: str,
                text_search: TextSearch | None = None):
    """The research agent: create_agent with the harness's middleware and ours (module docstring)."""
    s = run.s

    def counter(messages: list[AnyMessage], tools: list | None = None) -> int:
        system = [flat(m) if isinstance(m, SystemMessage) else m for m in messages]
        return message_tokens(system) + (tools_tokens(tools) if tools else 0)

    async def no_cover() -> None:
        return None

    cover = coverage(run, docs, question, text_search) if text_search else no_cover
    middleware: list[AgentMiddleware] = [
        Research(run),
        TodoListMiddleware(system_prompt=TODO_RULES, tool_description=TODO_TOOL),
        Evicting(s, run.research),
    ]
    if s.deep_subagents or run.fanout is not None:
        method, rules = [RESEARCHER], TASK_RULES
        if run.fanout is not None:  # fan-out: the budget and the cost of a researcher, delegate first
            method.append(RESEARCHER_BUDGET.format(lease=run.fanout.lease_size, tokens=run.fanout.return_tokens))
            rules = FANOUT_RULES.format(calls=s.agent_max_steps, lease=run.fanout.lease_size, tasks=s.deep_max_tasks)
        researcher = {"name": "researcher", "description": "researches one sub-question with the document tools",
                      "system_prompt": "\n\n".join([_SYSTEM.format(sql_rows=s.sql_max_rows), BASE, *method,
                                                    TASK.format(question=question, documents=_documents(docs))]),
                      "model": llm, "tools": tools,
                      "middleware": [Research(run), Evicting(s, run.research), PatchToolCallsMiddleware(),
                                     ModelStep(run, lead=False)]}
        middleware.append(SubAgentMiddleware(backend=StateBackend(), subagents=[researcher], system_prompt=rules,
                                             task_description=TASK_TOOL))
    middleware += [Summarizing(run, llm, counter), PatchToolCallsMiddleware(), Verify(run, cover),
                   ModelStep(run, lead=True)]
    return create_agent(llm, tools=tools, system_prompt=system_prompt(s, question, docs), middleware=middleware,
                        name="deep_research")


def recursion_limit(s: Settings) -> int:
    """Graph steps one run may take: every model call (each one calls tools, or ends, or is sent back by verify)
    with its tool step and middleware nodes, at most 6 steps, plus the entry nodes."""
    return (s.agent_max_steps + s.deep_harness_calls + s.agent_verify_rounds + 2) * 6 + 10


Builder = Callable[..., Any]  # (llm, tools, run, docs, question, text_search) -> a compiled agent


async def deep_search(deps: Deps, req: ChatRequest, docs: list[DocumentInfo], conv: Conversation,
                      question: str, timings: dict) -> AsyncIterator[Event]:
    """Research on the harness, then the answer exactly as rag.agent gives it. Research tools only read
    documents; nothing here can write memory (document text is untrusted)."""
    async for event in research(deps, req, docs, conv, question, timings, build_agent):
        yield event


async def research(deps: Deps, req: ChatRequest, docs: list[DocumentInfo], conv: Conversation, question: str,
                   timings: dict, build: Builder, tools_for: Callable[[ToolContext], list[BaseTool]] = _tools,
                   cover_after: bool = True) -> AsyncIterator[Event]:
    """One run of an agent made by `build` around the research tools that `tools_for` makes from the request's
    ToolContext (rag.agent's TOOLS by default), streaming its steps, then the answer from the source book. `cover_after`: rag.agent's search of the text documents
    of a mixed selection when no passage of them was found (off for the stock harness)."""
    started = time.perf_counter()
    s = deps.s
    book = SourceBook()
    ctx = ToolContext(deps, req, docs, book)
    tools = tools_for(ctx)
    run = Run(s, book, deep_think(s, req), frozenset(t.name for t in tools),
              fanout=FanOut.of(s) if deep_fanout(s, req) else None)
    text_search = text_searcher(ctx)
    graph = build(deps.graph_llm, tools, run, docs, question, text_search)
    # the run is a task (it copies the request's context: identity, notices) so steps stream as they happen,
    # sub-agents' included
    task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(question)]},
                                             config={"recursion_limit": recursion_limit(s)}))
    get: asyncio.Future | None = None
    try:
        while True:
            get = asyncio.ensure_future(run.events.get())
            await asyncio.wait({task, get}, return_when=asyncio.FIRST_COMPLETED)
            if not get.done():
                break
            yield get.result()
        while not run.events.empty():
            yield run.events.get_nowait()
        final = task.result()
    finally:  # also when the client went away mid-research
        for pending in (get, task):
            if pending is not None and not pending.done():
                pending.cancel()
        close_sandbox(ctx)  # the request's Python data goes as soon as the research ends
    # as rag.agent's verify does when no round is left: the text documents of a mixed selection are still
    # searched, and the passages reach the answer (a no-op when they were searched or a passage was found)
    if cover_after and book.sources:
        await coverage(run, docs, question, text_search)()
    missing = open_todos(final.get("todos"))
    if missing:
        notice("research_incomplete", "Deep search could not find everything the question needs; the answer says "
               "what is missing. Still missing: " + "; ".join(missing))
    timings["research_ms"] = round((time.perf_counter() - started) * 1000)
    timings.update(tool_calls=run.tool_calls, harness_calls=run.harness_calls, model_calls=run.model_calls)
    if run.fanout is not None:
        timings.update(tasks=run.fanout.started, tasks_refused=run.fanout.refused, task_calls=run.fanout.delegated)
    async for event in answer_from_book(deps, req, conv, question, book, run.tool_calls, timings):
        yield event
