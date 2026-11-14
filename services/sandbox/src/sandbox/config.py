"""The sandbox's limits, read once from SANDBOX_* environment variables (compose passes them; DEPLOYMENT section
4.16 lists them with their defaults). A value that is not a number, or is out of range, stops the service at
start-up with a message naming the variable."""

import os
from dataclasses import dataclass
from pathlib import Path


class SettingError(ValueError):
    pass


def _number(env: dict, name: str, default: float, minimum: float, integer: bool = True) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise SettingError(f"{name}={raw!r} is not a {'whole ' if integer else ''}number") from None
    if value < minimum:
        raise SettingError(f"{name}={raw!r} is below its minimum, {minimum:g}")
    return value


def _flag(env: dict, name: str) -> bool:
    raw = env.get(name, "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise SettingError(f"{name}={raw!r} is not true or false")


@dataclass(frozen=True)
class Limits:
    scratch_dir: Path  # SANDBOX_SCRATCH_DIR: the runs' folders; emptied after every run
    wipe_dirs: tuple[Path, ...]  # SANDBOX_WIPE_DIRS (comma list): also emptied after every run (image: /tmp, /dev/shm)
    sweep: bool  # SANDBOX_SWEEP: after every run kill every other process of this user, remove all SysV IPC
    max_timeout_s: float  # SANDBOX_MAX_TIMEOUT_S: the largest time limit a request may ask for (rag sends one)
    queue_wait_s: float  # SANDBOX_QUEUE_WAIT_S: how long a request waits for the run before it (then 503)
    cancel_wait_s: float  # SANDBOX_CANCEL_WAIT_S: how long a cancel waits for the run's end before it answers
    max_input_bytes: int  # SANDBOX_MAX_INPUT_MB: the files shipped with one run, together
    max_input_files: int  # SANDBOX_MAX_INPUT_FILES
    max_code_bytes: int  # SANDBOX_MAX_CODE_KB
    max_output_bytes: int  # SANDBOX_MAX_OUTPUT_KB: stdout and stderr returned, each (the start and the end)
    max_file_bytes: int  # SANDBOX_MAX_FILE_MB: the largest file the code may write, its output included (RLIMIT_FSIZE)
    max_files: int  # SANDBOX_MAX_FILES: output files listed in the reply
    max_return_file_bytes: int  # SANDBOX_MAX_RETURN_FILE_KB: a text file returned with its content up to this size
    max_return_bytes: int  # SANDBOX_MAX_RETURN_KB: the contents returned, together
    threads: int  # SANDBOX_THREADS: threads numpy, polars and DuckDB start (environment of the run)
    max_open_files: int  # SANDBOX_MAX_OPEN_FILES (RLIMIT_NOFILE)
    max_processes: int  # SANDBOX_MAX_PROCESSES: processes and threads of the sandbox's user (RLIMIT_NPROC)
    mpl_cache: Path | None  # SANDBOX_MPL_CACHE: matplotlib's font cache, built in the image, copied into each run

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Limits":
        env = dict(os.environ if env is None else env)
        kb, mb = 1024, 1024 * 1024
        wipe = tuple(Path(p.strip()) for p in env.get("SANDBOX_WIPE_DIRS", "").split(",") if p.strip())
        mpl = env.get("SANDBOX_MPL_CACHE", "").strip()
        return cls(
            scratch_dir=Path(env.get("SANDBOX_SCRATCH_DIR", "").strip() or "/scratch"),
            wipe_dirs=wipe,
            sweep=_flag(env, "SANDBOX_SWEEP"),
            max_timeout_s=_number(env, "SANDBOX_MAX_TIMEOUT_S", 600, 1, integer=False),
            queue_wait_s=_number(env, "SANDBOX_QUEUE_WAIT_S", 300, 0, integer=False),
            cancel_wait_s=_number(env, "SANDBOX_CANCEL_WAIT_S", 30, 0, integer=False),
            max_input_bytes=int(_number(env, "SANDBOX_MAX_INPUT_MB", 512, 1) * mb),
            max_input_files=int(_number(env, "SANDBOX_MAX_INPUT_FILES", 64, 1)),
            max_code_bytes=int(_number(env, "SANDBOX_MAX_CODE_KB", 256, 1) * kb),
            max_output_bytes=int(_number(env, "SANDBOX_MAX_OUTPUT_KB", 256, 1) * kb),
            max_file_bytes=int(_number(env, "SANDBOX_MAX_FILE_MB", 256, 1) * mb),
            max_files=int(_number(env, "SANDBOX_MAX_FILES", 50, 0)),
            max_return_file_bytes=int(_number(env, "SANDBOX_MAX_RETURN_FILE_KB", 64, 0) * kb),
            max_return_bytes=int(_number(env, "SANDBOX_MAX_RETURN_KB", 256, 0) * kb),
            threads=int(_number(env, "SANDBOX_THREADS", 4, 1)),
            max_open_files=int(_number(env, "SANDBOX_MAX_OPEN_FILES", 1024, 16)),
            max_processes=int(_number(env, "SANDBOX_MAX_PROCESSES", 128, 8)),
            mpl_cache=Path(mpl) if mpl else None,
        )
