"""sandbox: runs the Python that Deep search's run_python tool writes (rag.sandbox), one run at a time.

Only rag reaches it, on the internal `sandbox` network, which holds nothing else. It holds no identity, key or
volume: rag decides what a run may read and sends it with the code (rag_common.sandbox), and everything a run
wrote is removed before the next one starts (executor.py). Requests are not authenticated: nothing but rag can
connect, and a request can do nothing but run code in this container.

    POST /v1/execute   body: a rag_common.sandbox frame, header X-Run-Id (32 hex characters, made by rag)
                       ->  the run's reply (executor.execute)
                       400 a malformed frame or a time limit above SANDBOX_MAX_TIMEOUT_S; 413 code or files over
                       their caps; 503 the run before it took longer than SANDBOX_QUEUE_WAIT_S, or the server is
                       restarting; 507 the files did not fit the scratch space; 500 a run left something behind
                       (the server then stops, and the container starts again from its image); 409 the run
                       was cancelled before it started
    POST /v1/runs/{run_id}/cancel   kill that run if it is going and answer once it ended (at most
                       SANDBOX_CANCEL_WAIT_S): {"ended": bool, "started": bool}; a run not started yet is dropped
                       when its turn comes

One run at a time is a security property, not a tuning choice: every run uses the same user, so two runs at
once could read each other's files and processes. A request waits for the turn before its body is read, so a
queue of requests holds no data in memory. The turn ends when the run and its clean-up have ended, also when
rag gave up waiting meanwhile.

At start-up the server marks itself not dumpable (Linux), so no process of the same user, a run's code included,
can read its memory through ptrace or /proc, and with SANDBOX_SWEEP it removes whatever a previous container
life left behind. In the container it is PID 1 (compose: no init): the only long-lived process, which a run can
neither trace nor kill (init ignores the signals it has no handler for; SIGTERM, which uvicorn handles, only
restarts the container).
"""

import asyncio
import ctypes
import logging
import os
import collections
import ipaddress
import re
import signal
import socket
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from rag_common.logging import log_fields
from rag_common.sandbox import FrameError, decode
from rag_common.service import create_app

from sandbox import executor
from sandbox.config import Limits

log = logging.getLogger("sandbox.api")
limits = Limits.from_env()
HEADER_SLACK = 64 * 1024  # the frame's header beyond the code: file names and sizes
ESCAPED = 6  # bytes one character of code may take in the JSON header (an escape such as \u00e9)
PR_SET_DUMPABLE = 4
_turn = asyncio.Lock()
state: dict = {"restarting": False, "running": None}  # running: (run_id, executor.Control) of the run going now
RUN_ID = re.compile(r"[0-9a-f]{32}")
_cancelled: collections.OrderedDict = collections.OrderedDict()  # run ids cancelled before they started
CANCELLED_KEPT = 256


def stop_server() -> None:
    """Ends the process; Docker starts the container again (restart: unless-stopped) with empty tmpfs mounts."""
    os.kill(os.getpid(), signal.SIGTERM)


terminate: Callable[[], None] = stop_server  # tests replace it


def not_dumpable() -> None:
    """No ptrace or /proc access to this process for anyone without CAP_SYS_PTRACE (Linux)."""
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE, 0) failed")


@asynccontextmanager
async def lifespan(_app):
    not_dumpable()
    limits.scratch_dir.mkdir(parents=True, exist_ok=True)
    if limits.sweep:
        await asyncio.to_thread(executor.cleanup, limits)
    log_fields(log, "sandbox ready", logging.INFO, max_timeout_s=limits.max_timeout_s,
               max_input_mb=limits.max_input_bytes // (1024 * 1024), sweep=limits.sweep)
    yield


app = create_app("sandbox", lifespan=lifespan, log_level=os.getenv("SANDBOX_LOG_LEVEL", "INFO"))


async def _body(request: Request, cap: int) -> bytes:
    """The whole body, in memory, refused past `cap` bytes."""
    parts, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > cap:
            raise HTTPException(413, f"the request is larger than {cap} bytes (SANDBOX_MAX_INPUT_MB, "
                                     "SANDBOX_MAX_CODE_KB)")
        parts.append(chunk)
    return b"".join(parts)


def _address(text: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        addr = ipaddress.ip_address(text or "")
    except ValueError:
        return None
    return addr.ipv4_mapped or addr if isinstance(addr, ipaddress.IPv6Address) else addr


def own_addresses() -> set:
    """This container's addresses (all of them, loopback aside), looked up once."""
    if "own" not in state:
        found = set()
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                if (addr := _address(info[4][0])) is not None:
                    found.add(addr)
        except OSError:
            pass
        state["own"] = found
    return state["own"]


def from_inside(request: Request) -> bool:
    """A connection made from this container (a run's own code): from any loopback address (127.0.0.0/8, ::1, also
    IPv4-mapped), from the address it connected to, or from another address of this container. rag's connections
    come from rag's address. Anything that is not an address at all is treated as inside."""
    client = _address(request.client.host if request.client else None)
    if client is None:
        return True
    server = _address((request.scope.get("server") or (None,))[0])
    return client.is_loopback or client == server or client in own_addresses()


@app.post("/v1/execute")
async def execute(request: Request) -> JSONResponse:
    if from_inside(request):  # a run must not queue a run of its own for after its clean-up
        raise HTTPException(403, "runs are requested by rag only")
    if state["restarting"]:
        raise HTTPException(503, "the sandbox is restarting after a failed clean-up; try again in a minute")
    cap = limits.max_input_bytes + header_cap()
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > cap:
        raise HTTPException(413, f"the request is {declared} bytes, more than {cap} (SANDBOX_MAX_INPUT_MB, "
                                 "SANDBOX_MAX_CODE_KB)")
    try:
        await asyncio.wait_for(_turn.acquire(), limits.queue_wait_s)
    except TimeoutError:
        raise HTTPException(503, f"the sandbox is busy with another run and the wait passed {limits.queue_wait_s:g} s "
                                 "(SANDBOX_QUEUE_WAIT_S)") from None
    work: asyncio.Future | None = None
    run_id = request.headers.get("x-run-id", "")
    run_id = run_id if RUN_ID.fullmatch(run_id) else None
    try:
        if state["restarting"]:  # the run before this one failed its clean-up while this request waited
            raise HTTPException(503, "the sandbox is restarting after a failed clean-up; try again in a minute")
        job = _job(await _body(request, cap))
        if await request.is_disconnected():
            raise HTTPException(409, "the run was cancelled before it started")
        # No await from this check until state["running"] is set: a cancel landing in between would be recorded
        # as "not started" and the run would start anyway, an orphan after rag gave the user's turn back.
        if run_id is not None and _cancelled.pop(run_id, None):
            raise HTTPException(409, "the run was cancelled before it started")
        control = executor.Control()
        state["running"] = (run_id, control)
        work = asyncio.ensure_future(asyncio.to_thread(_run_job, job, control))
        work.add_done_callback(_run_ended)  # the turn ends with the run and its clean-up, never before
        reply = await asyncio.shield(work)
    except executor.CleanupFailed as exc:
        log_fields(log, "a run left something behind; the server stops so the container starts clean",
                   logging.CRITICAL, error=str(exc))
        return JSONResponse({"detail": "the sandbox could not clean up after the run and restarts; try again in a "
                                       "minute"}, status_code=500, background=BackgroundTask(terminate))
    except executor.ScratchFull as exc:
        raise HTTPException(507, "the files did not fit the sandbox's scratch space (SANDBOX_SCRATCH_SIZE)") from exc
    finally:
        if work is None:
            _turn.release()
    log_fields(log, "run", logging.INFO, ms=reply["duration_ms"], exit_code=reply["exit_code"],
               signal=reply["signal"], timed_out=reply["timed_out"], files=reply["files_total"],
               input_bytes=job.input_bytes, stdout_bytes=reply["stdout_bytes"])
    return JSONResponse(reply)


def _run_ended(_) -> None:
    state["running"] = None
    _turn.release()


@app.post("/v1/runs/{run_id}/cancel")
async def cancel(run_id: str, request: Request) -> dict:
    """Kill the run `run_id` (rag's request for it ended) and answer once it ended, so rag can give the user's turn
    back knowing no orphan run holds the sandbox. A run that has not started is dropped when its turn comes."""
    if from_inside(request):
        raise HTTPException(403, "runs are cancelled by rag only")
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(400, "a run id is 32 hex characters")
    running = state["running"]
    if running is None or running[0] != run_id:
        _cancelled[run_id] = True
        while len(_cancelled) > CANCELLED_KEPT:
            _cancelled.popitem(last=False)
        return {"ended": True, "started": False}
    running[1].cancel()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limits.cancel_wait_s
    while state["running"] is running and loop.time() < deadline:
        await asyncio.sleep(0.05)
    ended = state["running"] is not running
    log_fields(log, "run cancelled", logging.INFO, ended=ended)
    return {"ended": ended, "started": True}


def _run_job(job, control=None):
    """One run (in a worker thread). A failed clean-up marks the server as restarting before the turn is given
    back, so no request waiting for the turn runs in a container that is not clean."""
    try:
        return executor.execute(job, limits, control)
    except executor.CleanupFailed:
        state["restarting"] = True
        raise


def header_cap() -> int:
    return ESCAPED * limits.max_code_bytes + HEADER_SLACK


def _job(body: bytes):
    try:
        job = decode(body, max_header_bytes=header_cap(), max_files=limits.max_input_files)
    except FrameError as exc:
        raise HTTPException(400, f"malformed request: {exc}") from None
    if len(job.code.encode("utf-8")) > limits.max_code_bytes:
        raise HTTPException(413, f"the code is longer than {limits.max_code_bytes // 1024} KB (SANDBOX_MAX_CODE_KB)")
    if job.input_bytes > limits.max_input_bytes:
        raise HTTPException(413, f"the files are {job.input_bytes} bytes, more than {limits.max_input_bytes} "
                                 "(SANDBOX_MAX_INPUT_MB)")
    if job.timeout_s > limits.max_timeout_s:
        raise HTTPException(400, f"timeout_s {job.timeout_s:g} is above SANDBOX_MAX_TIMEOUT_S "
                                 f"({limits.max_timeout_s:g})")
    return job
