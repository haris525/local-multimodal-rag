"""Search tasks: the instruction the question is embedded with (docs/design/search-tasks.md).

The text embedder (Qwen3-Embedding) turns a question into a vector after an instruction. A search task names one:

- the presets below, the same for everyone. General's instruction is RAG_QUERY_INSTRUCTION, so a deployment can
  change the default without a rebuild; Exact lookup has none (the bare question).
- a user's own custom tasks, kept in the threads service. rag reads one with the caller's identity, so threads
  answers with the caller's own tasks only.

A request's `search_task` is a preset id or a custom task's id. None means General; anything else (an unknown id,
another user's task, a deleted one) is a 422 before anything runs. Every embedding of the question in that turn
uses the task: single-shot search, every search of a deep research (fan-out researchers included) and the
per-document candidates, which reuse the same vector. Exhaustive mode embeds nothing, so it ignores the task without
resolving it. The reranker keeps its fixed instruction, and image queries theirs. models answers with the
instruction it applied, and rag refuses a query vector made without the one it sent (ModelsClient.embed_query).

The text reaches the embedding model only, never the LLM: a hostile instruction can only make its owner's own
retrieval worse. Logs, the audit and the thread record name the task's kind (preset:<id> or custom), never the
text of a custom task.
"""

import logging
import re
from dataclasses import dataclass

from rag_common.contracts import SEARCH_TASK_ID, instruction_text
from rag_common.logging import log_fields

from rag.clients import ServiceError
from rag.config import Settings
from rag.conversation import ThreadsClient

log = logging.getLogger("rag.search_tasks")
GENERAL = "general"


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    description: str  # one line for the UI
    instruction: str | None  # None: RAG_QUERY_INSTRUCTION (General); empty: no instruction, the bare question
    measured: str = ""  # what the experiment measured it to do; the UI shows it when set


# `measured`: part A of docs/design/search-tasks.md (2026-11-15), recall@10 after reranking on the held-out halves of
# HotpotQA and MuSiQue, against the instruction the app used before (docs/benchmarks/embedding-models-2026-11-15.md).
PRESETS: tuple[Preset, ...] = (
    Preset(GENERAL, "General", "The default, for most questions.", None,
           "Finds the right passages more often than the old default (+0.9 and +1.9 points on two public sets)."),
    Preset("multi-step", "Multi-step research", "For questions that chain facts across several passages.",
           "Given a multi-hop question, retrieve documents that can help answer the question",
           "Best on multi-step questions (+1.2 and +2.0 points); about the same as General otherwise."),
    Preset("exact", "Exact lookup", "No instruction: the bare question, close to a keyword search.", "",
           "Close to General (+0.4 and +2.1 points); try it for short, keyword-like questions."),
)
_BY_ID = {p.id: p for p in PRESETS}


@dataclass(frozen=True)
class SearchTask:
    """A resolved search task: what a turn embeds its question with."""

    id: str  # the preset id or the custom task's id, as the request named it (the UI asks again with it)
    kind: str  # "preset:<id>" or "custom": what logs, the audit and the thread record carry
    instruction: str  # empty: the bare question


class BadSearchTask(ValueError):
    """Not a preset and not one of the caller's custom tasks, or a stored task that fails the check (rag answers
    422)."""


def preset_instruction(s: Settings, preset: Preset) -> str:
    return s.query_instruction if preset.instruction is None else preset.instruction


def general(s: Settings) -> SearchTask:
    return SearchTask(GENERAL, f"preset:{GENERAL}", s.query_instruction)


def presets(s: Settings) -> list[dict]:
    """The built-in tasks for the UI, with the instruction each one embeds the question with."""
    return [{"id": p.id, "name": p.name, "description": p.description, "instruction": preset_instruction(s, p),
             "measured": p.measured} for p in PRESETS]


async def resolve(s: Settings, threads: ThreadsClient, task_id: str | None) -> SearchTask:
    """The task a request named: General for None, a preset, or one of the caller's custom tasks (read from threads
    with the caller's identity). BadSearchTask for anything else; ServiceError when threads cannot answer."""
    if task_id is None:
        task = general(s)
    elif task_id in _BY_ID:
        task = SearchTask(task_id, f"preset:{task_id}", preset_instruction(s, _BY_ID[task_id]))
    elif re.fullmatch(SEARCH_TASK_ID, task_id):
        try:
            item = await threads.search_task(task_id)
        except ServiceError as exc:
            if exc.status == 404:  # missing, deleted, or another user's: the same answer
                raise BadSearchTask(f"unknown search task: {task_id} (deleted, or not one of yours)") from None
            raise
        try:
            instruction = instruction_text(item.instruction)  # threads checked it on write; models checks it again
        except ValueError as exc:
            raise BadSearchTask(f"search task {task_id} cannot be used: {exc}; edit it in the Search tasks panel") from None
        task = SearchTask(task_id, "custom", instruction)
    else:
        raise BadSearchTask(f"unknown search task: {task_id}; use one of {', '.join(_BY_ID)} or the id of one of "
                            "your own search tasks")
    log_fields(log, "search task", search_task_kind=task.kind)
    return task
