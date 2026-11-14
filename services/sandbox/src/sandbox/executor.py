"""One run: a fresh folder, the code in a child process with limits, what it printed and wrote within caps, then
everything it left behind removed. The API runs one at a time (api.py), so no run ever shares the container
with another.

    <scratch>/run-<random>/          mode 0700, removed after the run
        main.py                      the code
        stdout, stderr               the child's output: opened by the server before the run and read back only
                                     through those descriptors (the run may replace the names)
        tmp/                         HOME, TMPDIR, MPLCONFIGDIR (matplotlib's prebuilt font cache copied in)
        work/                        the working folder; files the code leaves here are returned (within caps)
        work/data/...                the request's files, at the paths it named (rag_common.sandbox)

The child is `python -I -B -u -X utf8 runner.py` in a session of its own, with a minimal environment (no
variable of the server's), stdin closed, and the RLIMITs of rlimits() applied by runner.py before the code runs.
At the time limit, or when the code ends, its whole process group is killed. With SANDBOX_SWEEP (the image sets
it; the container runs nothing else) the server then also kills every other process its user can signal, which
catches a process that left the group with setsid(), and after the folder it empties SANDBOX_WIPE_DIRS (/tmp and
/dev/shm) and removes every SysV shared memory segment, message queue and semaphore set of the container's IPC
namespace, whoever owns it (POSIX message queues are refused by RLIMIT_MSGQUEUE 0). If anything is left after
that, CleanupFailed: the API stops the server and the container starts again from its image, with empty tmpfs
mounts. In the container the server is PID 1 (no init
process): it reaps the processes a run left, and there is no other long-lived process a run could take over.

Nothing the run could have changed is opened by name afterwards: the output is read from the descriptors the
server opened before the run, and the working folder is walked from a descriptor opened without following links,
each file opened relative to its folder's descriptor, without following a link and without blocking (a FIFO or a
link swapped in by the run is skipped, never read).

On Windows (development and tests only) there are no RLIMITs and the group is killed with taskkill /T.
"""

import errno
import json
import logging
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

from rag_common.logging import log_fields
from rag_common.sandbox import Job

from sandbox.config import Limits
from sandbox.runner import SETUP_FAILED

RUNNER = Path(__file__).with_name("runner.py")
POSIX = os.name == "posix"
LINUX = sys.platform == "linux"
SWEEP_ROUNDS = 20
MAX_SCAN = 10_000  # output files counted at most; a run that leaves more is reported as "10000 or more"
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
log = logging.getLogger("sandbox.executor")


class CleanupFailed(Exception):
    """Something a run left could not be removed; the message lists what."""


class ScratchFull(Exception):
    """The request's files did not fit the scratch space (SANDBOX_SCRATCH_SIZE)."""


class Control:
    """A run's handle for cancel(): set before the child starts, read by the API's cancel route from another thread.
    A run cancelled before its child starts never starts; one cancelled after is killed with its process group."""

    def __init__(self):
        self._lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.cancelled = False

    def started(self, proc: subprocess.Popen) -> bool:
        """Record the child; False (and the child killed) when the run was cancelled meanwhile."""
        with self._lock:
            self.proc = proc
            cancelled = self.cancelled
        if cancelled:
            stop(proc)
        return not cancelled

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            proc = self.proc
        if proc is not None:
            stop(proc)


def stop(proc: subprocess.Popen) -> None:
    """Kill the run's process group now, without waiting (the run's own thread reaps it)."""
    if POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elif proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)


def rlimits(limits: Limits) -> dict[str, int]:
    """The limits runner.py applies to the child (POSIX). RLIMIT_NPROC counts every process and thread of the
    sandbox's user, the server's threads included, which is why the container's pid limit sits above it."""
    if not POSIX:
        return {}
    out = {"RLIMIT_CORE": 0, "RLIMIT_FSIZE": limits.max_file_bytes, "RLIMIT_NOFILE": limits.max_open_files,
           "RLIMIT_NPROC": limits.max_processes}
    if LINUX:
        out["RLIMIT_MSGQUEUE"] = 0  # no POSIX message queue: one would outlive the run
    return out


def child_env(limits: Limits, tmp: Path) -> dict[str, str]:
    """The run's whole environment: nothing of the server's, and no PYTHON* variable (-I ignores them anyway)."""
    threads = str(limits.threads)
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin"]),
           "HOME": str(tmp), "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp), "LANG": "C.UTF-8",
           "MPLCONFIGDIR": str(tmp / "matplotlib"), "MPLBACKEND": "Agg", "XDG_CACHE_HOME": str(tmp / "cache"),
           "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
           "NUMEXPR_MAX_THREADS": threads, "POLARS_MAX_THREADS": threads}
    if not POSIX:
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")  # Python on Windows needs it to start
    return env


def execute(job: Job, limits: Limits, control: Control | None = None) -> dict:
    """Run the job and return the reply (api.py); CleanupFailed when the run left something behind, ScratchFull
    when its files did not fit. `control` lets another thread cancel the run."""
    run = limits.scratch_dir / f"run-{secrets.token_hex(16)}"
    try:
        return _run(job, limits, run, control or Control())
    finally:
        cleanup(limits)


def _run(job: Job, limits: Limits, run: Path, control: Control) -> dict:
    work, tmp = run / "work", run / "tmp"
    for folder in (run, tmp, work):
        folder.mkdir(mode=0o700)
    try:
        for f in job.files:
            _write_new(work, f.path, f.data)
        code_file = run / "main.py"
        _write_new(run, "main.py", job.code.encode("utf-8"))
        if limits.mpl_cache and limits.mpl_cache.is_dir():
            shutil.copytree(limits.mpl_cache, tmp / "matplotlib")
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise ScratchFull(str(exc)) from exc
        raise
    command = [sys.executable, "-I", "-B", "-u", "-X", "utf8", str(RUNNER),
               json.dumps({"rlimits": rlimits(limits)}), str(code_file)]
    # opened here, read back through these descriptors only: the run can unlink or replace the names
    with open(run / "stdout", "w+b") as out, open(run / "stderr", "w+b") as err:
        started = time.monotonic()
        timed_out, proc = False, None
        if not control.cancelled:  # a run cancelled before its turn came never starts
            proc = subprocess.Popen(command, cwd=work, env=child_env(limits, tmp), stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, close_fds=True, **_new_group())
            try:
                if control.started(proc):  # one cancelled while its child started is killed at once
                    proc.wait(timeout=job.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                kill_group(proc)  # also after a normal end: what it started in the background goes with it
                if limits.sweep:
                    sweep_processes()
        duration_ms = round((time.monotonic() - started) * 1000)
        stdout, stdout_bytes = read_output(out, limits.max_output_bytes)
        stderr, stderr_bytes = read_output(err, limits.max_output_bytes)
    code, sig = (proc.returncode if proc is not None else None), None
    if POSIX and code is not None and code < 0:
        try:
            sig = signal.Signals(-code).name
        except ValueError:
            sig = f"signal {-code}"
    files, total = output_files(work, limits)
    return {"exit_code": None if timed_out else code, "signal": None if timed_out else sig, "timed_out": timed_out,
            "duration_ms": duration_ms, "timeout_s": job.timeout_s, "stdout": stdout, "stdout_bytes": stdout_bytes,
            "stderr": stderr, "stderr_bytes": stderr_bytes, "files": files, "files_total": total,
            "error": failure(timed_out, code, sig, stderr, job.timeout_s, limits, control.cancelled)}


def _write_new(root: Path, relative: str, data: bytes | memoryview) -> None:
    """Write a file the request named (already a safe relative path, rag_common.sandbox.check_path), refusing to
    leave `root` or to follow or replace anything that exists."""
    target = root.joinpath(*relative.split("/"))
    if not target.resolve().is_relative_to(root.resolve()):
        raise OSError(f"{relative} leaves the run's folder")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _new_group() -> dict:
    if POSIX:
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def kill_group(proc: subprocess.Popen) -> None:
    """Kill the run's process group (POSIX: the session runner.py leads; Windows: its process tree), then reap."""
    if POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # the group is already gone
    elif proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def failure(timed_out: bool, code: int | None, sig: str | None, stderr: str, timeout_s: float,
            limits: Limits, cancelled: bool = False) -> str | None:
    """Why the run failed when the code's own error does not say it: a limit of the sandbox, or a cancel. None for a
    run that ended by itself (with any exit code) for its own reasons."""
    if cancelled:
        return "the run was cancelled: the request that asked for it ended"
    if timed_out:
        return f"the run passed its time limit ({timeout_s:g} s) and was stopped, with every process it started"
    if code == SETUP_FAILED and "sandbox runner:" in stderr:
        return "the sandbox could not apply its limits to the run, so the code did not run"
    mb = limits.max_file_bytes // (1024 * 1024)
    too_large = f"a file it wrote, or its output, passed {mb} MB (SANDBOX_MAX_FILE_MB)"
    if sig == "SIGXFSZ":  # a program that does not ignore the signal (Python ignores it and raises EFBIG instead)
        return too_large + ", so it was stopped"
    if code and "[Errno 27] File too large" in stderr:
        return too_large
    if sig == "SIGKILL":
        return "it was killed (SIGKILL), most likely for using more memory than the sandbox has (SANDBOX_MEMORY)"
    if sig:
        return f"it was stopped by {sig}"
    return None


def read_output(f, cap: int) -> tuple[str, int]:
    """The text of an output file the server opened before the run, through that descriptor (never by name), within
    `cap` bytes: whole, or its first three quarters and last quarter around a line that says how much was left out.
    Returns (text, the file's size)."""
    size = os.fstat(f.fileno()).st_size
    f.seek(0)
    if size <= cap:
        return f.read(size).decode("utf-8", errors="replace"), size
    head = f.read(cap * 3 // 4)
    f.seek(size - cap // 4)
    tail = f.read(cap // 4)
    left = size - len(head) - len(tail)
    text = (head.decode("utf-8", errors="replace") + f"\n[... {left} bytes left out ...]\n"
            + tail.decode("utf-8", errors="replace"))
    return text, size


def output_files(work: Path, limits: Limits) -> tuple[list[dict], int]:
    """The regular files the code left in its working folder (not data/, its input), sorted by path: at most
    SANDBOX_MAX_FILES of them, each with its size, and its text when it is UTF-8 within SANDBOX_MAX_RETURN_FILE_KB
    and the SANDBOX_MAX_RETURN_KB left. Returns (files, how many there were, at most MAX_SCAN). Links, FIFOs and
    other special files are skipped, never followed or opened; a working folder the run replaced with a link
    returns nothing."""
    if not POSIX:
        return _output_files_by_name(work, limits)
    try:
        root = os.open(work, os.O_RDONLY | os.O_DIRECTORY | NOFOLLOW)
    except OSError:
        return [], 0  # the run replaced or removed its working folder
    try:
        found: list[tuple[str, int]] = []
        for folder, dirs, names, dir_fd in os.fwalk(".", dir_fd=root, follow_symlinks=False):
            if folder == ".":
                dirs[:] = [d for d in dirs if d != "data"]
            for name in names:
                try:
                    info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    found.append((os.path.normpath(os.path.join(folder, name)).replace(os.sep, "/"), info.st_size))
                if len(found) >= MAX_SCAN:
                    break
            if len(found) >= MAX_SCAN:
                break
        found.sort()
        files, budget = [], limits.max_return_bytes
        for relative, size in found[:limits.max_files]:
            text = None
            if size <= min(limits.max_return_file_bytes, budget):
                data = _read_below(root, relative, size)
                if data is not None:
                    try:
                        text = data.decode("utf-8")
                        budget -= len(data)
                    except UnicodeDecodeError:
                        text = None
            files.append({"path": relative, "size": size, "text": text})
        return files, len(found)
    finally:
        os.close(root)


def _read_below(root: int, relative: str, size: int) -> bytes | None:
    """A regular file under the folder descriptor `root`, each folder opened without following a link and the file
    without following a link or blocking; None when anything on the way is not what it should be."""
    fds: list[int] = []
    try:
        current = root
        *folders, name = relative.split("/")
        for part in folders:
            current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | NOFOLLOW, dir_fd=current)
            fds.append(current)
        fd = os.open(name, os.O_RDONLY | NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        fds.append(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        return os.read(fd, size)
    except OSError:
        return None
    finally:
        for fd in fds:
            os.close(fd)


def _output_files_by_name(work: Path, limits: Limits) -> tuple[list[dict], int]:
    """Windows (development and tests only): the same, by name."""
    if not work.is_dir() or work.is_symlink():
        return [], 0
    found: list[tuple[str, int, Path]] = []
    for folder, dirs, names in os.walk(work):
        here = Path(folder)
        if here == work:
            dirs[:] = [d for d in dirs if d != "data"]
        for name in names:
            path = here / name
            info = os.lstat(path)
            if stat.S_ISREG(info.st_mode) and len(found) < MAX_SCAN:
                found.append((path.relative_to(work).as_posix(), info.st_size, path))
    found.sort()
    files, budget = [], limits.max_return_bytes
    for relative, size, path in found[:limits.max_files]:
        text = None
        if size <= min(limits.max_return_file_bytes, budget):
            try:
                with open(path, "rb") as f:
                    text = f.read(size).decode("utf-8")
                budget -= size
            except (OSError, UnicodeDecodeError):
                text = None
        files.append({"path": relative, "size": size, "text": text})
    return files, len(found)


# --- clean-up -----------------------------------------------------------------------------------------------


def cleanup(limits: Limits) -> None:
    """Remove everything a run can leave: the scratch folder's contents, SANDBOX_WIPE_DIRS' contents and, with
    SANDBOX_SWEEP, every other process of this user and every SysV IPC object. CleanupFailed if anything stays."""
    problems = []
    if limits.sweep and (left := sweep_processes()):
        problems.append(f"processes still running: {left[:10]}")
    if limits.sweep:
        reap()
    for folder in (limits.scratch_dir, *limits.wipe_dirs):
        problems += empty(folder)
    if limits.sweep:
        problems += remove_sysv_ipc()
    if problems:
        raise CleanupFailed("; ".join(problems))


def empty(folder: Path) -> list[str]:
    """Remove everything inside `folder` (made writable first: the code may have taken permissions away); the
    problems found, if any."""
    try:
        entries = list(os.scandir(folder))
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [f"{folder} cannot be listed ({exc})"]
    for entry in entries:
        path = Path(entry.path)
        try:
            if entry.is_dir(follow_symlinks=False):
                _writable(path)
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log_fields(log, "a leftover could not be removed", logging.ERROR, path=str(path), error=str(exc))
    try:
        left = [e.name for e in os.scandir(folder)]
    except OSError as exc:
        return [f"{folder} cannot be listed ({exc})"]
    return [f"{folder} still holds {left[:5]}"] if left else []


def _writable(root: Path) -> None:
    """Owner rights on every folder under `root` (links are not followed), so rmtree can list and empty them."""
    os.chmod(root, 0o700)
    for folder, dirs, _ in os.walk(root):
        for d in dirs:
            path = os.path.join(folder, d)
            if not os.path.islink(path):
                os.chmod(path, 0o700)


def _others() -> list[int]:
    """Processes in this container other than PID 1 and this server (in the container they are the same process);
    zombies are reaped by reap()."""
    out = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) in (1, os.getpid()):
            continue
        try:
            with open(f"/proc/{entry.name}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            continue  # gone meanwhile
        if state != "Z":
            out.append(int(entry.name))
    return sorted(out)


def sweep_processes() -> list[int]:
    """Kill every process this user may signal except this server and PID 1, in rounds until none is left (a
    process forking at the time of a round dies in the next); the ones still running after SWEEP_ROUNDS. Only on
    Linux, never as root (kill(-1) as root would reach far more than this container)."""
    if not LINUX:
        return []
    if os.getuid() == 0:
        raise CleanupFailed("SANDBOX_SWEEP is refused for root")
    left: list[int] = []
    for _ in range(SWEEP_ROUNDS):
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            pass  # nothing to signal
        left = _others()
        if not left:
            return []
        time.sleep(0.05)
    return left


def reap() -> int:
    """Collect the exit status of every child that has ended. In the container the server is PID 1, so the processes
    a run left behind become its children when their parents die, and stay zombies (holding a pid) until reaped.
    Called between runs only, when no run's process is being waited for. Returns how many were reaped."""
    if not POSIX:
        return 0
    reaped = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        if pid == 0:
            return reaped
        reaped += 1


_IPC = {"shm": ("shmid", "shmctl"), "msg": ("msqid", "msgctl"), "sem": ("semid", "semctl")}
IPC_RMID = 0


def remove_sysv_ipc(base: Path = Path("/proc/sysvipc")) -> list[str]:
    """Remove every SysV shared memory segment, message queue and semaphore set in the container's IPC namespace
    (private, and used by one run at a time), whatever its owner: a run can hand an object to another uid with
    IPC_SET, but not change its creator, which is always the sandbox's user, so IPC_RMID still applies. Returns
    what is left (any row at all), which fails the clean-up. Linux only."""
    if not LINUX:
        return []
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    problems = []
    for kind, (id_column, call) in _IPC.items():
        for ipc_id in ipc_ids(base / kind, id_column):
            if kind == "sem":
                libc.semctl(ipc_id, 0, IPC_RMID)
            else:
                getattr(libc, call)(ipc_id, IPC_RMID, None)
        if left := ipc_ids(base / kind, id_column):
            problems.append(f"SysV {kind} objects left: {left[:5]}")
    return problems


def ipc_ids(table: Path, id_column: str) -> list[int]:
    """The ids of every object in one /proc/sysvipc table (columns named by its header line), whoever owns it.
    A table that cannot be read counts as one object left, so the clean-up fails closed."""
    try:
        lines = table.read_text().splitlines()
    except FileNotFoundError:
        return []  # no SysV IPC in this kernel
    except OSError:
        return [-1]
    if not lines:
        return []
    at_id = lines[0].split().index(id_column)
    return [int(row[at_id]) for row in (line.split() for line in lines[1:]) if len(row) > at_id]
