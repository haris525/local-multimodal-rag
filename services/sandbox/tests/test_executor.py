"""One run: output, files, time limit, the process group, caps, limits, a fresh folder each time and the clean-up.

The tests run the real child process. RLIMIT tests need POSIX; the time limit and the process tree run everywhere
(Windows kills the tree with taskkill). SANDBOX_SWEEP is never turned on here: kill(-1) would reach every process
of the user running the tests.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rag_common.sandbox import InputFile, Job

from sandbox import executor
from sandbox.config import Limits
from sandbox.runner import SETUP_FAILED

posix = pytest.mark.skipif(os.name != "posix", reason="RLIMITs and process groups are POSIX")


def job(code: str, files: dict[str, bytes] | None = None, timeout_s: float = 30) -> Job:
    return Job(code, timeout_s, tuple(InputFile(p, d) for p, d in (files or {}).items()))


def run(limits: Limits, code: str, **kwargs) -> dict:
    return executor.execute(job(code, **kwargs), limits)


def with_(limits: Limits, **changes) -> Limits:
    import dataclasses

    return dataclasses.replace(limits, **changes)


def test_prints_come_back_with_the_exit_code(limits):
    reply = run(limits, "print('total', 6 * 7)")
    assert reply["exit_code"] == 0 and reply["stdout"].strip() == "total 42" and reply["error"] is None
    assert reply["timed_out"] is False and reply["signal"] is None and reply["stdout_bytes"] == len("total 42") + len(os.linesep)


def test_a_failing_run_shows_its_own_traceback_without_the_runner(limits):
    reply = run(limits, "x = 1\nprint('before')\nraise ValueError('bad value ' + str(x))\n")
    assert reply["exit_code"] == 1 and reply["stdout"].strip() == "before" and reply["error"] is None
    assert "Traceback (most recent call last)" in reply["stderr"] and "ValueError: bad value 1" in reply["stderr"]
    assert 'File "main.py", line 3' in reply["stderr"] and "raise ValueError" in reply["stderr"]
    assert "runner.py" not in reply["stderr"]


def test_a_syntax_error_is_reported_with_its_line(limits):
    reply = run(limits, "print('a')\nif True print('b')\n")
    assert reply["exit_code"] == 1 and "SyntaxError" in reply["stderr"] and "line 2" in reply["stderr"]


def test_sys_exit_keeps_its_code(limits):
    assert run(limits, "import sys; sys.exit(3)")["exit_code"] == 3


def test_the_files_sent_are_readable_where_the_request_put_them(limits):
    files = {"data/manifest.json": b'{"documents": []}', "data/1_sales.csv/text.jsonl": b'{"text": "hello"}\n'}
    code = ("import json, os\nprint(json.load(open('data/manifest.json')))\n"
            "print(open('data/1_sales.csv/text.jsonl').read().strip())\nprint(sorted(os.listdir('data')))")
    reply = run(limits, code, files=files)
    assert reply["stdout"].splitlines() == ["{'documents': []}", '{"text": "hello"}', "['1_sales.csv', 'manifest.json']"]


def test_a_run_reads_a_tables_file_with_duckdb_and_pandas(limits, tmp_path):
    import duckdb

    db = tmp_path / "t.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE sales AS SELECT * FROM (VALUES ('North', 10.0), ('South', 32.0)) t(region, amount)")
    code = ("import duckdb\ncon = duckdb.connect('data/1_s/tables.duckdb', read_only=True)\n"
            "df = con.sql('SELECT * FROM sales').df()\nprint(df['amount'].sum())")
    reply = run(limits, code, files={"data/1_s/tables.duckdb": db.read_bytes()})
    assert reply["exit_code"] == 0, reply["stderr"]
    assert reply["stdout"].strip() == "42.0"


def test_files_left_in_the_working_folder_are_returned_but_not_the_input(limits):
    code = ("import os\nopen('result.csv', 'w', newline='').write('a,b\\n1,2\\n')\nos.makedirs('charts')\n"
            "open('charts/plot.png', 'wb').write(bytes([137, 80, 78, 71, 255, 0]))\n"
            "open('data/extra.txt', 'w').write('not an output')")
    reply = run(limits, code, files={"data/in.txt": b"x"})
    assert reply["files"] == [{"path": "charts/plot.png", "size": 6, "text": None},
                              {"path": "result.csv", "size": 8, "text": "a,b\n1,2\n"}]
    assert reply["files_total"] == 2


def test_the_number_and_size_of_returned_files_are_capped(limits):
    capped = with_(limits, max_files=3, max_return_file_bytes=10, max_return_bytes=15)
    code = "for i in range(6):\n    open(f'f{i}.txt', 'w').write(str(i) * (4 if i < 5 else 50))"
    reply = run(capped, code)
    assert reply["files_total"] == 6 and [f["path"] for f in reply["files"]] == ["f0.txt", "f1.txt", "f2.txt"]
    assert [f["text"] for f in reply["files"]] == ["0000", "1111", "2222"]  # 12 of 15 bytes: the fourth would not fit
    reply = run(with_(limits, max_return_file_bytes=10), "open('big.txt', 'w').write('x' * 50)")
    assert reply["files"] == [{"path": "big.txt", "size": 50, "text": None}]


def test_output_over_the_cap_keeps_its_start_and_end_and_says_how_much_was_left_out(limits):
    capped = with_(limits, max_output_bytes=1000)
    reply = run(capped, "print('START'); print('x' * 5000); print('END')")
    assert reply["stdout_bytes"] == 5 + 5000 + 3 + 3 * len(os.linesep)
    assert reply["stdout"].startswith("START") and reply["stdout"].rstrip().endswith("END")
    assert "bytes left out" in reply["stdout"] and len(reply["stdout"]) < 1100


def test_the_time_limit_stops_the_run(limits):
    started = time.monotonic()
    reply = run(limits, "import time\nprint('working', flush=True)\ntime.sleep(60)", timeout_s=1.5)
    assert time.monotonic() - started < 20
    assert reply["timed_out"] is True and reply["exit_code"] is None and "time limit (1.5 s)" in reply["error"]
    assert reply["stdout"].strip() == "working"  # what it printed before the stop is kept


def test_the_time_limit_kills_every_process_the_run_started(limits, tmp_path):
    marker = tmp_path / "survivor.txt"
    child = f"import time, pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    code = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child!r}])\ntime.sleep(60)"
    reply = run(limits, code, timeout_s=1.5)
    assert reply["timed_out"]
    time.sleep(3.5)
    assert not marker.exists()


@posix
def test_a_finished_run_takes_its_background_processes_with_it(limits, tmp_path):
    marker = tmp_path / "survivor.txt"
    child = f"import time, pathlib; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('alive')"
    reply = run(limits, f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {child!r}])\nprint('done')")
    assert reply["exit_code"] == 0
    time.sleep(2.5)
    assert not marker.exists()


@posix
def test_the_limits_are_set_in_the_run_and_cannot_be_raised(limits):
    code = ("import resource\nfor name in ('RLIMIT_FSIZE', 'RLIMIT_NOFILE', 'RLIMIT_NPROC', 'RLIMIT_CORE'):\n"
            "    print(name, *resource.getrlimit(getattr(resource, name)))\n"
            "try:\n    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))\nexcept ValueError:\n"
            "    print('raise refused')")
    reply = run(with_(limits, max_file_bytes=5 * 1024 * 1024, max_open_files=64, max_processes=4096), code)
    lines = reply["stdout"].splitlines()
    assert lines[:4] == [f"RLIMIT_FSIZE {5 * 1024 * 1024} {5 * 1024 * 1024}", "RLIMIT_NOFILE 64 64",
                         "RLIMIT_NPROC 4096 4096", "RLIMIT_CORE 0 0"]
    assert lines[4] == "raise refused"


@posix
def test_a_file_over_the_size_limit_fails_the_run(limits):
    """Python ignores SIGXFSZ, so the write raises EFBIG; the reply names the limit."""
    reply = run(with_(limits, max_file_bytes=1024 * 1024), "open('big.bin', 'wb').write(b'x' * (3 * 1024 * 1024))")
    assert reply["exit_code"] == 1 and "File too large" in reply["stderr"] and "SANDBOX_MAX_FILE_MB" in reply["error"]
    assert reply["files"][0]["size"] <= 1024 * 1024


@posix
def test_output_is_bounded_by_the_file_size_limit_too(limits):
    reply = run(with_(limits, max_file_bytes=1024 * 1024), "while True:\n    print('x' * 1000)", timeout_s=20)
    assert reply["exit_code"] == 1 and not reply["timed_out"] and reply["stdout_bytes"] <= 1024 * 1024
    assert "SANDBOX_MAX_FILE_MB" in reply["error"]


@posix
def test_too_many_open_files_fail_inside_the_run(limits):
    code = "fs = []\ntry:\n    while True:\n        fs.append(open('data/a.txt'))\nexcept OSError as e:\n    print(len(fs) < 40)"
    reply = run(with_(limits, max_open_files=32), code, files={"data/a.txt": b"a"})
    assert reply["stdout"].strip() == "True"


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX message queues are refused by RLIMIT_MSGQUEUE (Linux)")
def test_a_run_cannot_make_a_posix_message_queue(limits):
    assert executor.rlimits(limits)["RLIMIT_MSGQUEUE"] == 0
    assert run(limits, "import resource; print(resource.getrlimit(resource.RLIMIT_MSGQUEUE))")["stdout"].strip() == "(0, 0)"


def test_the_run_sees_none_of_the_servers_environment(limits, monkeypatch):
    monkeypatch.setenv("SANDBOX_TEST_SECRET", "do-not-leak")
    reply = run(limits, "import os, json; print(json.dumps(dict(os.environ)))")
    env = json.loads(reply["stdout"])
    assert "SANDBOX_TEST_SECRET" not in env and not any("do-not-leak" in v for v in env.values())
    assert env["HOME"] == env["TMPDIR"] and env["HOME"].endswith("tmp") and env["MPLBACKEND"] == "Agg"


def test_each_run_starts_in_a_fresh_folder_and_nothing_of_a_previous_run_is_left(limits):
    first = run(limits, "import os, tempfile\nopen('left.txt', 'w').write('secret')\n"
                        "open(os.path.join(tempfile.gettempdir(), 'tmp.txt'), 'w').write('secret')\nprint(os.getcwd())")
    assert first["exit_code"] == 0
    assert list(limits.scratch_dir.iterdir()) == []  # removed after the run
    second = run(limits, f"import os\nprint(sorted(os.listdir('.')))\nprint(os.path.exists({first['stdout'].strip()!r}))\n"
                         f"print(os.listdir({str(limits.scratch_dir)!r}) == [os.path.basename(os.path.dirname(os.getcwd()))])")
    assert second["stdout"].splitlines() == ["[]", "False", "True"]
    assert list(limits.scratch_dir.iterdir()) == []


def test_the_clean_up_empties_the_wipe_folders_even_when_the_code_took_permissions_away(limits, tmp_path):
    shm = tmp_path / "shm"
    shm.mkdir()
    wiping = with_(limits, wipe_dirs=(shm,))
    code = (f"import os\nbase = {str(shm)!r}\nos.makedirs(os.path.join(base, 'a', 'b'))\n"
            "open(os.path.join(base, 'a', 'b', 'f.txt'), 'w').write('x')\n"
            "open(os.path.join(base, 'loose.txt'), 'w').write('x')\n"
            "os.chmod(os.path.join(base, 'a', 'b'), 0o500)\nos.chmod(os.path.join(base, 'a'), 0o500)")
    assert run(wiping, code)["exit_code"] == 0
    assert list(shm.iterdir()) == [] and list(limits.scratch_dir.iterdir()) == []


def test_a_failed_clean_up_is_reported_not_ignored(limits, monkeypatch):
    monkeypatch.setattr(executor, "empty", lambda folder: [f"{folder} still holds ['x']"])
    with pytest.raises(executor.CleanupFailed, match="still holds"):
        run(limits, "print(1)")


@posix
def test_links_in_the_working_folder_are_never_followed(limits):
    reply = run(limits, "import os\nos.symlink('/etc/hostname', 'link.txt')\nos.mkfifo('pipe')\nopen('real.txt','w').write('r')")
    assert reply["files"] == [{"path": "real.txt", "size": 1, "text": "r"}] and reply["files_total"] == 1


def test_a_request_path_can_never_leave_the_runs_folder(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    for bad in ("../escape.txt", "data/../../escape.txt"):
        with pytest.raises(OSError, match="leaves"):
            executor._write_new(root, bad, b"x")
    assert not (tmp_path / "escape.txt").exists()
    executor._write_new(root, "data/ok.txt", b"x")
    with pytest.raises(FileExistsError):
        executor._write_new(root, "data/ok.txt", b"y")  # never replaces what is there


def test_a_runner_that_cannot_apply_its_limits_runs_no_code(tmp_path):
    code = tmp_path / "main.py"
    code.write_text("print('ran')")
    proc = subprocess.run([sys.executable, "-I", str(executor.RUNNER), "not json", str(code)], capture_output=True,
                          text=True, check=False)
    assert proc.returncode == SETUP_FAILED and "ran" not in proc.stdout and "sandbox runner:" in proc.stderr
    assert executor.failure(False, SETUP_FAILED, None, proc.stderr, 30, Limits.from_env({})).startswith(
        "the sandbox could not apply its limits")


@pytest.mark.parametrize("sig, words", [("SIGKILL", "SANDBOX_MEMORY"), ("SIGSEGV", "stopped by SIGSEGV"),
                                        ("SIGXFSZ", "SANDBOX_MAX_FILE_MB")])
def test_a_run_stopped_by_a_signal_says_why(sig, words):
    assert words in executor.failure(False, -9, sig, "", 30, Limits.from_env({}))
    assert executor.failure(False, 1, None, "Traceback ...", 30, Limits.from_env({})) is None


def test_every_sysv_object_is_found_whoever_owns_it(tmp_path):
    """A run can hand a segment to another uid with IPC_SET (here uid 0); it is removed all the same."""
    table = tmp_path / "shm"
    table.write_text("       key      shmid perms       size  cpid  lpid nattch   uid   gid  cuid  cgid\n"
                     "         0         17   600       4096    10    10      0 10101 10101 10101 10101\n"
                     "         0         18   600       4096    11    11      0     0     0 10101 10101\n")
    assert executor.ipc_ids(table, "shmid") == [17, 18] and executor.ipc_ids(tmp_path / "none", "shmid") == []
    unreadable = tmp_path / "msg"
    unreadable.mkdir()  # reading a folder fails: counted as an object left, so the clean-up fails closed
    assert executor.ipc_ids(unreadable, "msqid") == [-1]


def test_the_sweep_is_refused_for_root(monkeypatch):
    monkeypatch.setattr(executor, "LINUX", True)
    monkeypatch.setattr(os, "getuid", lambda: 0, raising=False)
    with pytest.raises(executor.CleanupFailed, match="root"):
        executor.sweep_processes()


def test_the_sweep_kills_in_rounds_until_no_process_is_left(monkeypatch):
    monkeypatch.setattr(executor, "LINUX", True)
    monkeypatch.setattr(os, "getuid", lambda: 10101, raising=False)
    monkeypatch.setattr(executor.time, "sleep", lambda s: None)
    monkeypatch.setattr(executor.signal, "SIGKILL", 9, raising=False)
    kills, alive = [], [[41, 42], [42], []]
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append(pid))
    monkeypatch.setattr(executor, "_others", lambda: alive.pop(0))
    assert executor.sweep_processes() == [] and kills == [-1, -1, -1]
    monkeypatch.setattr(executor, "_others", lambda: [99])
    assert executor.sweep_processes() == [99]  # reported, so the server restarts the container


@posix
def test_ended_children_are_reaped_between_runs():
    """In the container the server is PID 1 and inherits what runs leave; reap() collects them."""
    procs = [subprocess.Popen([sys.executable, "-c", "pass"]) for _ in range(3)]
    time.sleep(1)
    assert executor.reap() >= 3
    assert executor.reap() == 0
    for proc in procs:
        proc.returncode = 0  # already reaped: keep Popen from waiting again


def _within(seconds: float, fn, *args):
    """fn(*args) in a thread; the test fails instead of hanging when it does not return in time."""
    import threading

    box = {}
    thread = threading.Thread(target=lambda: box.update(value=fn(*args)), daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"did not return within {seconds} s"
    return box["value"]


@posix
def test_a_fifo_swapped_in_for_the_output_neither_blocks_nor_hides_it(limits):
    code = "import os\nprint('kept', flush=True)\nos.unlink('../stdout')\nos.mkfifo('../stdout')\nprint('after')"
    reply = _within(30, executor.execute, job(code), limits)
    assert reply["exit_code"] == 0 and reply["stdout"].splitlines() == ["kept", "after"]
    assert list(limits.scratch_dir.iterdir()) == []


@posix
@pytest.mark.parametrize("target", ["/dev/zero", "/proc/self/environ", "/proc/1/maps", "/etc/hostname"])
def test_a_link_swapped_in_for_the_output_is_never_read(limits, monkeypatch, target):
    monkeypatch.setenv("SANDBOX_TEST_SERVER_ONLY", "server-secret")
    code = ("import os\nprint('mine', flush=True)\nfor n in ('../stdout', '../stderr'):\n    os.unlink(n)\n"
            f"    os.symlink({target!r}, n)")
    reply = _within(30, executor.execute, job(code), limits)
    assert reply["stdout"].strip() == "mine" and reply["stderr"] == "" and "server-secret" not in json.dumps(reply)


@posix
def test_a_working_folder_replaced_with_a_link_returns_nothing(limits):
    code = "import os\nos.chdir('..')\nos.rename('work', 'moved')\nos.symlink('/etc', 'work')\nprint('done')"
    reply = _within(30, executor.execute, job(code), limits)
    assert reply["stdout"].strip() == "done" and reply["files"] == [] and reply["files_total"] == 0


@posix
def test_links_and_fifos_inside_the_working_folder_are_skipped_without_blocking(limits):
    code = ("import os\nos.makedirs('sub')\nos.symlink('/etc', 'sub/etc')\nos.mkfifo('sub/pipe')\n"
            "os.symlink('/etc/passwd', 'passwd')\nopen('sub/real.txt', 'w').write('ok')")
    reply = _within(30, executor.execute, job(code), limits)
    assert reply["files"] == [{"path": "sub/real.txt", "size": 2, "text": "ok"}] and reply["files_total"] == 1


def test_output_read_back_from_the_descriptor_keeps_start_and_end(tmp_path):
    with open(tmp_path / "out", "w+b") as f:
        f.write(b"A" * 300 + b"B" * 100)
        f.flush()
        text, size = executor.read_output(f, 200)
    assert size == 400 and text.startswith("A" * 150) and text.endswith("B" * 50) and "200 bytes left out" in text


def test_a_cancelled_run_is_killed_with_its_processes_and_says_so(limits):
    import threading

    control = executor.Control()
    threading.Timer(1.0, control.cancel).start()
    started = time.monotonic()
    reply = executor.execute(job("import time\nprint('going', flush=True)\ntime.sleep(60)"), limits, control)
    assert time.monotonic() - started < 20 and "cancelled" in reply["error"] and reply["stdout"].strip() == "going"
    assert list(limits.scratch_dir.iterdir()) == []


def test_a_run_cancelled_before_its_child_starts_does_not_run(limits, tmp_path):
    marker = tmp_path / "ran.txt"
    control = executor.Control()
    control.cancel()
    reply = executor.execute(job(f"open({str(marker)!r}, 'w').write('ran')"), limits, control)
    assert "cancelled" in reply["error"]
    time.sleep(0.5)
    assert not marker.exists()
