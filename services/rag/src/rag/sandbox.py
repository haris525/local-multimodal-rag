"""run_python: Python for Deep search, run by the sandbox service (services/sandbox). Off by default.

RAG_DEEP_PYTHON (off, request or on; chat.deep_python) and a request's `python` offer the tool to whichever arm
runs: rag.agent's TOOLS reach the langgraph, deepagents and deepagents_stock arms. It is a research tool like sql: it
spends the research budget (RAG_AGENT_MAX_STEPS), an identical call is answered from the first one's result, and its
output becomes a numbered source the answer can cite.

A call sends the code, with the selected documents' data, to the sandbox, which runs it in a fresh folder with no
network, no credentials and no access of its own, and answers with what it printed. rag decides what a run may
read; the sandbox only runs it. The data contract (what a run finds in its working folder):

    data/manifest.json             one entry per selected document: its folder, name, doc_id, type, its tables
                                   (name, sheet or page, rows, columns with types and units, raw table) and its files
    data/<n>_<name>/tables.duckdb  the document's tables: the file the sql service queries, fetched from sql with
                                   the caller's signed identity (another user's document does not exist there)
    data/<n>_<name>/text.jsonl     the document's passages, one JSON object per line (chunk_index, kind, page,
                                   page_end, section, text), from ingest-api's chunk listing, with the same
                                   identity; not for spreadsheets, whose rows are in the tables

Only this request's documents (ToolContext.docs: the selected ones, already checked against their owner) are
fetched: once, on the first call, then kept in the request's memory for its later calls and released when the
research ends (close). Data over RAG_SANDBOX_INPUT_MB is refused with a tool error naming the sizes and a notice,
never cut.

Fair use, so one user cannot starve the others: one run per user in flight at a time (RAG_SANDBOX_USER_RUNS; the
sandbox itself runs one at a time for everyone), at most RAG_SANDBOX_REQUEST_SECONDS of run time per question (each
run's limit is what is left of it), and at most RAG_SANDBOX_TOTAL_INPUT_MB of Python data held by rag for all
questions together. Each limit, when hit, is a tool error with a notice.

The shared total is reserved before a download starts, never checked and added afterwards: a tables file's size is
known from its Content-Length before its body is read, and the bytes are counted (synchronously, so no other request
can take the same room) before they arrive; passages are counted page by page. Every exit, a cancellation included,
gives back what the request reserved. Fetching itself goes one at a time per user and RAG_SANDBOX_FETCHES at a time
for everyone, and a tables file is read into a buffer of its known size, so a download never holds twice its size.

A run gets an id rag makes; when the request ends while the run is going (the user left, the research was
cancelled), rag asks the sandbox to kill that run and keeps the user's run turn until the sandbox says the run ended
(or RAG_SANDBOX_CANCEL_WAIT_S passed), so leaving and asking again never leaves an orphan run beside a new one.

What a failure looks like: a run that exits with an error (its traceback), passes its time limit or another limit
of the sandbox, prints nothing, or cannot be sent (sandbox unreachable, busy, refusing) is a tool error: a
ToolException turned into a tool message with status "error" (handle_tool_error), so the model reads what went
wrong, the step event carries it, and the arms' error counting (run health, StallGuard) sees it. Problems of the
sandbox itself, not of the code, also get a notice for the user.
"""

import asyncio
import json
import logging
import re
import uuid
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool, ToolException

from rag_common.contracts import DocType, DocumentInfo
from rag_common.http import ServiceError
from rag_common.identity import current_identity
from rag_common.logging import log_fields

from rag.clients import TooLarge
from rag.notices import notice
from rag.prompts import Source
from rag.tokens import count_tokens, truncate_tokens

log = logging.getLogger("rag.sandbox")
MB = 1024 * 1024
CHUNK_PAGE = 1000  # ingest's largest chunk page
TEXT_FIELDS = ("chunk_index", "kind", "page", "page_end", "section", "text")
SOURCE_CODE_TOKENS = 1500  # the code shown in the source the answer and the Sources panel get

DESCRIPTION = (
    "Run Python 3.12 in a sandbox with no network and return what it prints. Use it for what one SQL query cannot "
    "do: statistics, regressions, correlations, date arithmetic, combining several documents' tables, counting or "
    "extracting over a document's full text. Installed: pandas, numpy, polars, duckdb, scipy, statsmodels, openpyxl, "
    "matplotlib (charts are not shown to anyone). The selected documents are copied into data/ for every run, and "
    "nothing is kept between runs: data/manifest.json lists each document's folder, tables and files; "
    "data/<folder>/tables.duckdb holds its tables, the same ones sql queries "
    "(duckdb.connect(path, read_only=True).sql('SELECT ...').df()); data/<folder>/text.jsonl holds its text passages, "
    "one JSON object per line (chunk_index, page, section, text). Print every result you need, labelled; only "
    "printed output comes back (about {tokens} tokens of it). A run may take {timeout:g} s."
)


@dataclass
class Data:
    files: list[tuple[str, bytes | bytearray]]
    size: int


_held = {"bytes": 0}  # Python data rag holds for all requests (RAG_SANDBOX_TOTAL_INPUT_MB)


def _release(holder: list[int]) -> None:
    _held["bytes"] -= holder[0]
    holder[0] = 0


@dataclass
class Session:
    """The request's run_python state: the documents' data, fetched on the first call, and its run time so far."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    data: Data | None = None
    refused: str | None = None  # why the data cannot be shipped (too large): every later call gets the same error
    seconds: float = 0.0  # run time used by this question (RAG_SANDBOX_REQUEST_SECONDS)
    held: list[int] = field(default_factory=lambda: [0])  # bytes counted in _held for this request

    def __post_init__(self):
        weakref.finalize(self, _release, self.held)  # a request dropped without close() still gives its bytes back


def session(ctx) -> Session:
    if ctx.sandbox is None:
        ctx.sandbox = Session()
    return ctx.sandbox


def close(ctx) -> None:
    """The research ended: drop the request's data and give its bytes back (the arms call it in a finally)."""
    state = ctx.sandbox
    if state is not None:
        state.data = None
        _release(state.held)


def held_bytes() -> int:
    return _held["bytes"]


class TurnTimeout(Exception):
    """The user's earlier runs did not leave a turn within the wait."""


class UserTurns:
    """At most `n` runs of one user in flight at once; entries are dropped when idle."""

    def __init__(self):
        self._users: dict[str, list] = {}  # user -> [semaphore, users of it]

    @asynccontextmanager
    async def turn(self, user: str, n: int, wait_s: float):
        entry = self._users.get(user)
        if entry is None:
            entry = self._users[user] = [asyncio.Semaphore(n), 0]
        entry[1] += 1
        try:
            try:
                await asyncio.wait_for(entry[0].acquire(), wait_s)
            except TimeoutError:
                raise TurnTimeout() from None
            try:
                yield
            finally:
                entry[0].release()
        finally:
            entry[1] -= 1
            if entry[1] == 0 and self._users.get(user) is entry:
                del self._users[user]


user_turns = UserTurns()
fetch_turns = UserTurns()  # data fetching, one at a time per user
_fetch_gate: dict = {"loop": None, "size": None, "sem": None}


@asynccontextmanager
async def fetch_slot(s, user: str):
    """One data fetch of this user at a time, and RAG_SANDBOX_FETCHES for everyone; TurnTimeout past the wait."""
    loop = asyncio.get_running_loop()
    if _fetch_gate["loop"] is not loop or _fetch_gate["size"] != s.sandbox_fetches:
        _fetch_gate.update(loop=loop, size=s.sandbox_fetches, sem=asyncio.Semaphore(s.sandbox_fetches))
    gate = _fetch_gate["sem"]
    async with fetch_turns.turn(user, 1, s.sandbox_request_timeout_s):
        try:
            await asyncio.wait_for(gate.acquire(), s.sandbox_request_timeout_s)
        except TimeoutError:
            raise TurnTimeout() from None
        try:
            yield
        finally:
            gate.release()


def folder(i: int, name: str) -> str:
    """A document's folder under data/: its place in the selection and its name, in safe characters."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[:60] or "document"
    return f"data/{i}_{slug}"


def _mb(size: int) -> str:
    return f"{size / MB:.1f} MB"


def _too_large(s, what: str, size: int) -> str:
    return (f"The selected documents' data is larger than the {s.sandbox_input_mb:g} MB a Python run may take "
            f"(RAG_SANDBOX_INPUT_MB): with {what} it comes to {_mb(size)} or more. Python cannot be used for this "
            "question; use sql and the other tools, or select fewer documents.")


def _columns(table) -> list[dict]:
    return [{"name": c.sql_name, "type": c.type, **({"unit": c.unit} if c.unit else {})} for c in table.columns]


class Busy(Exception):
    """rag already holds RAG_SANDBOX_TOTAL_INPUT_MB of Python data for other questions."""


class Budget:
    """The bytes one gather may still take: its own room (RAG_SANDBOX_INPUT_MB) and the shared total, reserved
    before the bytes arrive. take() checks and counts with no await in between, so two requests can never both be
    given the same room; give_back() returns what was reserved and not used."""

    def __init__(self, s, state: Session):
        self.own = int(s.sandbox_input_mb * MB)
        self.total = int(s.sandbox_total_input_mb * MB)
        self.state = state
        self.used = 0

    def room(self) -> int:
        return max(0, self.own - self.used)

    def take(self, size: int) -> None:
        if self.used + size > self.own:
            raise TooLarge(self.used + size)
        if _held["bytes"] + size > self.total:
            raise Busy()
        _held["bytes"] += size
        self.state.held[0] += size
        self.used += size

    def give_back(self, size: int) -> None:
        size = min(size, self.state.held[0])
        _held["bytes"] -= size
        self.state.held[0] -= size
        self.used -= size


async def _text(ctx, doc: DocumentInfo, budget: Budget) -> bytearray:
    """The document's passages as JSON lines, each page counted in the budget before it is kept."""
    out, offset = bytearray(), 0
    while True:
        page = await ctx.deps.ingest.chunks(doc.doc_id, offset, CHUNK_PAGE)
        block = b"".join(json.dumps({k: record[k] for k in TEXT_FIELDS if record.get(k) not in (None, "")},
                                    ensure_ascii=False).encode() + b"\n" for record in page)
        budget.take(len(block))
        out += block
        offset += len(page)
        if len(page) < CHUNK_PAGE:
            return out


async def gather(ctx) -> Data:
    """The selected documents' files and the manifest (module docstring); ToolException when they are too large,
    Busy when rag holds too much for other questions, ServiceError when sql or ingest fails. What it reserved stays
    counted in the shared total until close(); the caller gives it back when this raises."""
    s, state = ctx.s, session(ctx)
    budget = Budget(s, state)
    files: list[tuple[str, bytes | bytearray]] = []
    entries = []
    for i, doc in enumerate(ctx.docs, 1):
        base = folder(i, doc.name)
        entry: dict = {"folder": base, "name": doc.name, "doc_id": doc.doc_id, "type": doc.doc_type.value,
                       "tables_file": None, "tables": [], "text_file": None}
        if doc.tables:
            before = budget.used
            try:
                data = await ctx.deps.sql.tables_file(doc.doc_id, budget.room(), budget.take, budget.give_back)
            except TooLarge as exc:
                raise ToolException(_too_large(s, f"the tables of {doc.name}", before + exc.size)) from None
            files.append((f"{base}/tables.duckdb", data))
            entry["tables_file"] = f"{base}/tables.duckdb"
            entry["tables"] = [{"table": t.table, "from": t.source, "rows": t.rows, "columns": _columns(t),
                                **({"raw_table": t.layout.raw_table} if t.layout and t.layout.raw_table else {})}
                               for t in doc.tables]
        if doc.doc_type != DocType.SPREADSHEET:
            try:
                data = await _text(ctx, doc, budget)
            except TooLarge as exc:
                raise ToolException(_too_large(s, f"the text of {doc.name}", exc.size)) from None
            files.append((f"{base}/text.jsonl", data))
            entry["text_file"] = f"{base}/text.jsonl"
        entries.append(entry)
    manifest = json.dumps({"about": "The documents selected for this question, copied for this run only.",
                           "documents": entries}, ensure_ascii=False, indent=1).encode()
    try:
        budget.take(len(manifest))
    except TooLarge as exc:
        raise ToolException(_too_large(s, "the manifest with the files", exc.size)) from None
    return Data([("data/manifest.json", manifest), *files], budget.used)


async def documents_data(ctx) -> Data:
    """The request's data, fetched once (later calls, a researcher's included, wait for the first fetch)."""
    state = session(ctx)
    async with state.lock:
        if state.refused:
            raise ToolException(state.refused)
        if state.data is None:
            ident = current_identity.get()
            try:
                async with fetch_slot(ctx.s, ident.user_id if ident else "-"):
                    state.data = await gather(ctx)
            except BaseException as exc:  # every exit gives the reservation back, a cancellation included
                _release(state.held)
                raise _fetch_failed(ctx, state, exc) from None
        return state.data


def _fetch_failed(ctx, state: Session, exc: BaseException) -> BaseException:
    """What a failed fetch raises (the reservation is already given back): a ToolException the model reads, with a
    notice; the exception itself when it is not about the data (a cancellation)."""
    s = ctx.s
    if isinstance(exc, TurnTimeout):
        notice("python_busy", "Another Python data fetch took long enough that this one gave up waiting (one per user, "
               "RAG_SANDBOX_FETCHES for everyone); deep search used its other tools.")
        return ToolException("Your other Python data fetch is still going; try once more later, or use sql and the "
                             "other tools.")
    if isinstance(exc, Busy):
        notice("python_busy", "rag already holds the most Python data it may for other questions "
               "(RAG_SANDBOX_TOTAL_INPUT_MB), so a Python run had to wait for them; deep search used its other "
               "tools meanwhile.", total_mb=s.sandbox_total_input_mb)
        return ToolException("rag holds the most Python data it may for other questions right now "
                             "(RAG_SANDBOX_TOTAL_INPUT_MB). Try once more later, or use sql and the other tools.")
    if isinstance(exc, ToolException):
        state.refused = str(exc)
        notice("python_input_too_large", f"The selected documents' data is larger than the {s.sandbox_input_mb:g} MB "
               "a Python run may take (RAG_SANDBOX_INPUT_MB), so deep search did not run Python and used its other "
               "tools.", limit_mb=s.sandbox_input_mb)
        return exc
    if isinstance(exc, ServiceError):
        notice("python_unavailable", "The documents' data could not be prepared for Python, so deep search used its "
               "other tools for what needed it.", error=f"{exc.status}: {exc.detail}"[:300])
        return ToolException(f"The documents' data could not be fetched for Python ({exc.status}: "
                             f"{exc.detail[:200]}). Use sql and the other tools instead.")
    return exc


def tail_tokens(text: str, limit: int) -> str:
    """The longest end of `text` within `limit` tokens (a traceback ends with what went wrong)."""
    if count_tokens(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[mid:]) <= limit:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def _shown(text: str, limit: int, what: str) -> str:
    """`text` within `limit` tokens, with a line saying so when it was cut."""
    if count_tokens(text) <= limit:
        return text
    return (truncate_tokens(text, max(0, limit - 40)) + f"\n[{what} cut at about {limit} tokens "
            "(RAG_SANDBOX_RESULT_TOKENS): print less, or summarize before printing]")


def _files(reply: dict, limit: int) -> str:
    files = reply.get("files") or []
    if not files:
        return ""
    lines = [f"Files the code wrote ({reply.get('files_total', len(files))}):"]
    for f in files:
        lines.append(f"- {f['path']} ({f['size']} bytes)" + (f":\n{f['text']}" if f.get("text") else ""))
    if reply.get("files_total", 0) > len(files):
        lines.append(f"- ... {reply['files_total'] - len(files)} more not listed (SANDBOX_MAX_FILES)")
    return _shown("\n".join(lines), limit, "the file list")


def failed(reply: dict, s) -> str | None:
    """The tool error for a run that did not produce a result, or None."""
    took = f"{(reply.get('duration_ms') or 0) / 1000:.1f} s"
    out = (reply.get("stdout") or "").strip()
    err = (reply.get("stderr") or "").strip()
    printed = f"\nOutput before that:\n{_shown(out, s.sandbox_result_tokens // 3, 'output')}" if out else ""
    if reply.get("timed_out"):
        limit = reply.get("timeout_s", s.sandbox_timeout_s)
        return (f"The run timed out after {limit:g} s (RAG_SANDBOX_TIMEOUT_S, or what was left of the question's "
                "RAG_SANDBOX_REQUEST_SECONDS) and "
                "was stopped with every process it started. Make it faster: let SQL or duckdb aggregate first, read "
                "fewer rows or columns, avoid loops over rows." + printed)
    if reply.get("error"):
        tail = tail_tokens(err, s.sandbox_result_tokens)
        return f"The run failed after {took}: {reply['error']}.\n{tail}{printed}".rstrip()
    if reply.get("exit_code") not in (0, None):
        return (f"The code failed (exit code {reply['exit_code']}, {took}):\n"
                f"{tail_tokens(err, s.sandbox_result_tokens) or '(nothing on stderr)'}{printed}")
    if not out and not reply.get("files"):
        return "The code ran (exit code 0) but printed nothing: print() every result you need, then run it again."
    return None


def result_text(reply: dict, s) -> str:
    """What the model and the answer read of a run that worked: its output, its warnings, its files."""
    parts = [_shown((reply.get("stdout") or "").rstrip(), s.sandbox_result_tokens, "the output")]
    err = (reply.get("stderr") or "").strip()
    if err:
        parts.append("stderr (warnings):\n" + tail_tokens(err, max(100, s.sandbox_result_tokens // 4)))
    if files := _files(reply, max(100, s.sandbox_result_tokens // 3)):
        parts.append(files)
    return "\n".join(p for p in parts if p)


def python_source(n_hint: int, code: str, reply: dict, s) -> Source:
    """A run as a source: the code (cut for the answer's prompt; the step keeps it whole) and its result."""
    shown = _shown(code.strip(), SOURCE_CODE_TOKENS, "the code")
    # fenced, so the Sources panel shows them as code
    body = f"Code:\n```python\n{shown}\n```\n\nResult:\n```\n{result_text(reply, s)}\n```"
    location = f"Python run, {(reply.get('duration_ms') or 0) / 1000:.1f} s"
    return Source(0, f"python:{n_hint}", "", "Python", "python", location, body, count_tokens(body))


def time_left(ctx) -> float:
    """The run time this question has left (RAG_SANDBOX_REQUEST_SECONDS); ToolException and a notice when none."""
    s = ctx.s
    left = s.sandbox_request_seconds - session(ctx).seconds
    if left < 1:
        notice("python_time_used", f"Deep search used all {s.sandbox_request_seconds:g} s of Python run time this "
               "question may take (RAG_SANDBOX_REQUEST_SECONDS); it went on with its other tools.",
               seconds=s.sandbox_request_seconds)
        raise ToolException(f"This question has used all its {s.sandbox_request_seconds:g} s of Python run time "
                            "(RAG_SANDBOX_REQUEST_SECONDS). Answer with what you have, or use sql and the other tools.")
    return min(s.sandbox_timeout_s, left)


async def send(ctx, code: str, data: Data) -> dict:
    """The sandbox's reply; ToolException (and a notice) when the sandbox cannot run it. One run per user at a
    time (RAG_SANDBOX_USER_RUNS), within the question's run time left; the run's time is counted."""
    client = ctx.deps.sandbox
    if client is None:
        raise ToolException("Python is not available in this deployment (no sandbox client).")
    s = ctx.s
    ident = current_identity.get()
    try:
        async with user_turns.turn(ident.user_id if ident else "-", s.sandbox_user_runs, s.sandbox_request_timeout_s):
            timeout_s = time_left(ctx)
            reply = await _execute(ctx, client, code, timeout_s, data)
    except TurnTimeout:
        notice("python_busy", "Your other Python runs took long enough that a run had to give up waiting for them "
               "(RAG_SANDBOX_USER_RUNS); deep search used its other tools meanwhile.")
        raise ToolException("Your other Python runs are still going (one at a time per user, "
                            "RAG_SANDBOX_USER_RUNS). Use sql and the other tools instead.") from None
    session(ctx).seconds += max(float(reply.get("duration_ms") or 0) / 1000,
                                float(reply.get("timeout_s") or 0) if reply.get("timed_out") else 0.0)
    return reply


async def _cancel(ctx, client, run_id: str) -> None:
    """The request ended mid-run: have the sandbox kill the run and wait (at most RAG_SANDBOX_CANCEL_WAIT_S) until it
    says the run ended, so the user's turn is not given back while the run still holds the sandbox."""
    try:
        reply = await asyncio.wait_for(client.cancel(run_id), ctx.s.sandbox_cancel_wait_s)
        ended = bool(reply.get("ended"))
    except Exception as exc:  # noqa: BLE001 - logged; the request is ending anyway
        log_fields(log, "a run could not be cancelled", logging.WARNING, run_id=run_id,
                   error=f"{type(exc).__name__}: {exc}"[:300])
        return
    if not ended:
        log_fields(log, "a cancelled run did not end within the sandbox's wait", logging.WARNING, run_id=run_id)


async def _execute(ctx, client, code: str, timeout_s: float, data: Data) -> dict:
    run_id = uuid.uuid4().hex
    try:
        return await client.execute(code, timeout_s, data.files, run_id)
    except asyncio.CancelledError:
        await _cancel(ctx, client, run_id)  # still inside the user's turn (send)
        raise
    except ServiceError as exc:
        detail = exc.detail[:300]
        if exc.status == 503:
            message = ("The Python sandbox is busy with other runs and the wait passed its limit "
                       "(SANDBOX_QUEUE_WAIT_S), so deep search used its other tools for what needed Python.")
            told = f"The Python sandbox is busy ({detail}). Try once more later, or use sql and the other tools."
        elif exc.status in (502, 504):
            message = "The Python sandbox could not be reached, so deep search used its other tools."
            told = f"The Python sandbox could not be reached ({detail}). Use sql and the other tools instead."
        else:
            message = "The Python sandbox refused a run, so deep search used its other tools for what needed Python."
            told = f"The Python sandbox refused the run ({exc.status}: {detail})."
        notice("python_unavailable", message, status=exc.status, error=detail)
        raise ToolException(told) from None


def run_python_tool(ctx) -> StructuredTool:
    """The run_python tool for this request (module docstring)."""
    from rag.agent import _tool  # at call time: rag.agent lists this tool in its registry

    s = ctx.s

    async def run_python(code: str) -> tuple[str, list[int]]:
        code = code if isinstance(code, str) else str(code)
        if not code.strip():
            raise ToolException("code is empty: pass the Python source to run.")
        if len(code) > s.sandbox_code_chars:
            raise ToolException(f"The code is {len(code)} characters, more than {s.sandbox_code_chars} "
                                "(RAG_SANDBOX_CODE_CHARS): write shorter code.")

        async def run() -> tuple[str, list[int]]:
            reply = await send(ctx, code, await documents_data(ctx))
            if problem := failed(reply, s):
                raise ToolException(problem)
            src = ctx.book.add(python_source(len(ctx.book.sources), code, reply, s))
            took = f"{(reply.get('duration_ms') or 0) / 1000:.1f} s"
            return f"[{src.n}] Python run (exit code 0, {took})\n{result_text(reply, s)}", [src.n]

        return await ctx.once("run_python:" + code, run)  # exact code: whitespace matters in Python

    tool = _tool(run_python, DESCRIPTION.format(tokens=s.sandbox_result_tokens, timeout=s.sandbox_timeout_s))
    tool.handle_tool_error = True  # a ToolException becomes the tool message, status "error", its text unescaped
    return tool
