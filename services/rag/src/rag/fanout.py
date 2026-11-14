"""Fan-out for Deep search: researchers on reserved research budgets (RAG_DEEP_FANOUT, a request's `fanout`).

A question has one research budget, RAG_AGENT_MAX_STEPS (10) or the request's research_calls, shared by the lead
and its researchers; it is the same total for every arm, with or without fan-out. Without fan-out every call
draws on the pool first come, first served, so a lead that spends it all leaves its researchers nothing: a lead
that uses its 10 calls and then calls task twice leaves both researchers with every call refused.

With fan-out a task call reserves a lease of RAG_DEEP_TASK_CALLS (4) research calls before its researcher
starts:

- The lead cannot spend reserved calls: it runs while the calls it used plus the calls still reserved stay
  under the budget. A researcher spends only its own lease.
- A task call is refused at once, with a tool message saying why and no researcher started, when the calls
  neither used nor reserved are fewer than one lease, or RAG_DEEP_MAX_TASKS (2) researchers already ran.
- The calls a researcher did not use go back to the pool when it ends. Used plus reserved never passes the
  budget, so the total stays the budget.
- At most RAG_DEEP_TASK_CONCURRENCY (2, the llama-server slot count) researchers of one request run at once;
  further task calls of the same reply wait for a free place, holding their lease. Researchers never get task
  (no nesting). Across users there is no limiter here: llama-server queues calls first come, first served, so
  another user's call waits for at most about one model call per busy slot.
- A researcher's report reaches the lead cut at RAG_DEEP_TASK_RETURN_TOKENS (800 model tokens), with a note
  when cut; every source it found stays in the run's source book whatever the report says.
- A researcher that runs past RAG_DEEP_TASK_TIMEOUT_S (300 s) or fails gives the lead an error tool message it
  can read and the user the research_task_failed notice; its sources stay, and it is logged.

The lease of the researcher that is running is a context variable: set around its task call, it reaches the
researcher's graph (langgraph copies the context into every node), so the researcher's middleware knows whose
budget it spends. rag.agent_deepagents.Research applies these rules; any arm built on it gets them.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from rag.config import Settings
from rag.tokens import count_tokens, truncate_tokens

REFUSED_FULL = ("Not run: a researcher reserves {lease} research calls and only {free} are left of the {budget} for "
                "this question ({used} used, {held} reserved). Research the rest yourself with the calls left.")
REFUSED_TASKS = ("Not run: {tasks} researchers already ran for this question (RAG_DEEP_MAX_TASKS). Research the "
                 "rest yourself with the {free} research calls left.")
RESERVED = ("Not run: the research calls left are reserved for the researchers started in this reply. Wait for "
            "their findings before calling more tools.")
LEASE_USED = ("Not run: this researcher's {lease} research calls are used up. Reply now with your findings: every "
              "figure and fact with its source number [n], and what you could not find.")
NESTED = "Not run: a researcher cannot start researchers of its own. Research the sub-question with your tools."
EMPTY = "The researcher returned no findings text. {found}"
CUT = ("\n\n[Report cut at {limit} of its {tokens} tokens (RAG_DEEP_TASK_RETURN_TOKENS). {found}]")
TIMED_OUT = ("Error: the researcher did not finish within {timeout:g} s (RAG_DEEP_TASK_TIMEOUT_S), so it returned no "
             "findings. {found} Research what is still missing yourself with the calls left.")
FAILED = ("Error: the researcher failed ({error}) and returned no findings. {found} Research what is still missing "
          "yourself with the calls left.")


def found_line(sources: list[int]) -> str:
    """What the lead is told about a researcher's sources, which stay in the source book in every case."""
    if not sources:
        return "It found no sources."
    return "The sources it found stay in the source book: " + ", ".join(f"[{n}]" for n in sources) + "."


@dataclass(eq=False)  # each lease is its own: two with the same counts are still two researchers
class Lease:
    """The research calls reserved for one researcher."""

    size: int
    used: int = 0
    sources: list[int] = field(default_factory=list)  # source numbers its tools returned, in order
    stopped: bool = False  # its prompt filled the slot: it ends, the run goes on
    last_calls: int = 0  # model calls since its calls ran out

    def left(self) -> int:
        return self.size - self.used

    def take(self) -> str | None:
        """Count one research call against the lease (None), or say why it cannot run."""
        if self.used >= self.size:
            return LEASE_USED.format(lease=self.size)
        self.used += 1
        return None

    def done(self) -> bool:
        """Before each of the researcher's model calls: whether it must end. Once its calls are used up it gets one
        more call to report its findings; a researcher that calls tools instead ends after it."""
        if self.stopped:
            return True
        if self.used < self.size:
            return False
        self.last_calls += 1
        return self.last_calls > 1

    def found(self, sources: list[int] | None) -> None:
        for n in sources or []:
            if n not in self.sources:
                self.sources.append(n)


_current: ContextVar[Lease | None] = ContextVar("rag_research_lease", default=None)


def current() -> Lease | None:
    """The lease of the researcher this code runs in; None in the lead, and always None without fan-out."""
    return _current.get()


def enter(lease: Lease) -> Token:
    return _current.set(lease)


def leave(token: Token) -> None:
    _current.reset(token)


@dataclass
class FanOut:
    """The reservations of one research run. `budget` is the question's research calls (the request's settings);
    the calls spent are the run's (rag.agent_deepagents.Run.tool_calls), passed in as `used`."""

    budget: int
    lease_size: int
    max_tasks: int
    return_tokens: int
    timeout_s: float
    active: list[Lease] = field(default_factory=list)
    started: int = 0  # researchers that ran (or are running)
    refused: int = 0  # task calls refused for the budget or the researcher limit
    delegated: int = 0  # research calls the researchers that ended used
    gate: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))

    @classmethod
    def of(cls, s: Settings) -> FanOut:
        return cls(budget=s.agent_max_steps, lease_size=min(s.deep_task_calls, s.agent_max_steps),
                   max_tasks=s.deep_max_tasks, return_tokens=s.deep_task_return_tokens,
                   timeout_s=s.deep_task_timeout_s, gate=asyncio.Semaphore(s.deep_task_concurrency))

    def held(self) -> int:
        """Calls reserved by running researchers and not yet used."""
        return sum(lease.left() for lease in self.active)

    def free(self, used: int) -> int:
        return self.budget - used - self.held()

    def lead_refusal(self, used: int) -> str | None:
        """For a research call of the lead, which has `used` of the budget: why not, when only reserved calls are
        left (the run's own check covers a budget that is used up)."""
        return RESERVED if self.free(used) <= 0 else None

    def task_refusal(self, used: int) -> str | None:
        """For a task call: why no researcher can start, or None. Counted before any await, as every budget."""
        free = self.free(used)
        if self.started >= self.max_tasks:
            why = REFUSED_TASKS.format(tasks=self.started, free=free)
        elif free < self.lease_size:
            why = REFUSED_FULL.format(lease=self.lease_size, free=free, budget=self.budget, used=used, held=self.held())
        else:
            return None
        self.refused += 1
        return why

    def reserve(self) -> Lease:
        """A lease for a task call that task_refusal let through (no await between the two)."""
        lease = Lease(self.lease_size)
        self.active.append(lease)
        self.started += 1
        return lease

    def release(self, lease: Lease, ran: bool) -> None:
        """The researcher ended: its unused calls go back to the pool. `ran`: a researcher actually started (a task
        call for an unknown subagent type starts none, and is not counted)."""
        self.active.remove(lease)
        self.delegated += lease.used
        if not ran:
            self.started -= 1

    def report(self, text: str, lease: Lease) -> tuple[str, int | None]:
        """The researcher's report as the lead reads it, and its size when it was cut: at most return_tokens model
        tokens with a note, or a note when it returned no text."""
        if not text.strip():
            return EMPTY.format(found=found_line(lease.sources)), None
        tokens = count_tokens(text)
        if tokens <= self.return_tokens:
            return text, None
        return truncate_tokens(text, self.return_tokens) + CUT.format(limit=self.return_tokens, tokens=tokens,
                                                                      found=found_line(lease.sources)), tokens
