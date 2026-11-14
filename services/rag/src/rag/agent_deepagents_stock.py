"""Deep research on the deepagents harness out of the box (RAG_DEEP_AGENT=deepagents_stock, or a chat request's
agent="deepagents_stock"), for the benchmark next to rag.agent and rag.agent_deepagents.

create_deep_agent (deepagents 0.5.9) with its default system prompts and default middleware: TodoListMiddleware,
FilesystemMiddleware with all its tools (ls, read_file, write_file, edit_file, glob, grep; execute is filtered
out because the backend cannot run commands), SubAgentMiddleware with the general-purpose sub-agent,
SummarizationMiddleware with its defaults, PatchToolCallsMiddleware and AnthropicPromptCachingMiddleware
(which does nothing for a model that is not Anthropic's). The backend is the default in-memory StateBackend:
files live in the request's graph state, nothing on disk, no shell.

What is ours, and only that:
- the research tools of rag.agent (the same scoping, sandbox and caps), with `grep` renamed `doc_grep` (the
  harness's file grep has that name); the research rules of rag.agent, with the same renaming, and the
  question with the selected documents, passed as create_deep_agent's `system_prompt` (the harness puts
  them before its own base prompt, as for any caller);
- the source book and the shared answer step (the same citations, sources and done events);
- rag.agent_deepagents' Research and ModelStep middleware as the caller's `middleware` (and in the
  general-purpose sub-agent, declared with the harness's own name, description and prompt so its stack is
  the default one plus these): the research-tool and harness-tool budgets, tool errors as messages, the step
  events, the slot guard in the model's tokens, one retry of a busy or cut step, and thinking for the first
  call (RAG_DEEP_THINK; the harness has no verify step). They keep a stock run from crashing the request.

The harness's defaults are kept even where they do not fit the 32,768-token slot: its summarization starts at
170,000 tokens (the model has no profile) and its eviction at 20,000 tokens counted as 4 characters a token,
so neither fires here; the slot guard ends a run whose prompt fills the slot, with a notice. No verify step
and no text-document check run after it.
"""

from collections.abc import AsyncIterator

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain_core.tools import BaseTool

from rag_common.contracts import DocumentInfo

from rag.agent import _SYSTEM, _documents, _tools
from rag.agent_deepagents import TASK, ModelStep, Research, Run, research
from rag.chat import ChatRequest, Deps, Event
from rag.config import Settings
from rag.conversation import Conversation

DOC_GREP = "doc_grep"


def _renamed(text: str, *phrases: str) -> str:
    """`text` with the document grep, in each of these phrases, named doc_grep."""
    for old in phrases:
        assert old in text, old  # the rules and descriptions name the document grep exactly so
        text = text.replace(old, old.replace("grep", DOC_GREP), 1)
    return text


def stock_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """The research tools with the document grep renamed (and read's description following it)."""
    out = []
    for t in tools:
        if t.name == "grep":
            t = t.model_copy(update={"name": DOC_GREP})
        elif t.name == "read":
            t = t.model_copy(update={"description": _renamed(t.description, "search or grep")})
        out.append(t)
    return out


def stock_prompt(s: Settings, question: str, docs: list[DocumentInfo]) -> str:
    """The caller's part of the system prompt: rag.agent's research rules (doc_grep for grep), the question and
    the documents. The harness adds its base prompt and its middleware's sections after it."""
    rules = _renamed(_SYSTEM.format(sql_rows=s.sql_max_rows), "grep: exact", "search or grep")
    return "\n\n".join([rules, TASK.format(question=question, documents=_documents(docs))])


def build_stock_agent(llm, tools: list[BaseTool], run: Run, docs: list[DocumentInfo], question: str,
                      text_search=None):
    """create_deep_agent with its defaults; our guards as the caller's middleware, in the lead and in the
    general-purpose sub-agent (the harness's spec, so its default stack is kept)."""
    general_purpose = {**GENERAL_PURPOSE_SUBAGENT, "middleware": [Research(run), ModelStep(run, lead=False)]}
    return create_deep_agent(llm, tools, system_prompt=stock_prompt(run.s, question, docs),
                             middleware=[Research(run), ModelStep(run, lead=True)], subagents=[general_purpose],
                             backend=StateBackend(), name="deep_research_stock")


async def deep_search(deps: Deps, req: ChatRequest, docs: list[DocumentInfo], conv: Conversation,
                      question: str, timings: dict) -> AsyncIterator[Event]:
    """The stock harness's research, then the answer exactly as rag.agent gives it."""
    async for event in research(deps, req, docs, conv, question, timings, build_stock_agent,
                                tools_for=lambda ctx: stock_tools(_tools(ctx)), cover_after=False):
        yield event
