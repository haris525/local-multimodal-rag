"""The child process of one run: apply the limits to itself, then run the code as __main__. Standard library
only; started as `python -I -B -u -X utf8 runner.py LIMITS_JSON CODE_FILE` in the run's working folder.

Limits (POSIX): every RLIMIT_* in LIMITS_JSON["rlimits"], soft and hard, so the code cannot raise them again;
on Linux also oom_score_adj 1000, so when the container runs out of memory the kernel prefers to kill the run
rather than the server. A preference only: the code can write its score back down to 0, and tmpfs and SysV memory
belong to no process, so the server can still be the one killed (the container then restarts; DEPLOYMENT 4.16).
A limit that cannot be applied stops the run before any code runs (exit code SETUP_FAILED, the reason
on stderr): the run never goes ahead with weaker limits than configured.

A failing run prints its traceback without this file's frames, with the code's own lines in it.
"""

import json
import linecache
import os
import sys
import traceback

SETUP_FAILED = 120
CODE_NAME = "main.py"


def limit(rlimits: dict[str, int]) -> None:
    if os.name != "posix":
        return
    import resource

    for name, value in rlimits.items():
        resource.setrlimit(getattr(resource, name), (value, value))
    if sys.platform == "linux":
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")


def main() -> None:
    try:
        settings = json.loads(sys.argv[1])
        with open(sys.argv[2], encoding="utf-8") as f:
            source = f.read()
        limit(settings.get("rlimits", {}))
    except Exception as exc:  # noqa: BLE001 - reported, and the run does not start
        print(f"sandbox runner: the limits could not be applied ({type(exc).__name__}: {exc})", file=sys.stderr)
        sys.exit(SETUP_FAILED)
    linecache.cache[CODE_NAME] = (len(source), None, source.splitlines(True), CODE_NAME)
    sys.argv = [CODE_NAME]
    namespace = {"__name__": "__main__", "__file__": CODE_NAME, "__builtins__": __builtins__}
    try:
        exec(compile(source, CODE_NAME, "exec"), namespace)  # noqa: S102 - running the code is this process's purpose
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - shown like an uncaught exception, without the runner's frame
        traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next if exc.__traceback__ else None)
        sys.exit(1)


if __name__ == "__main__":
    main()
