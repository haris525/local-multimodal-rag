import os
import time

import pytest

from ingest.isolation import IsolatedRunner, ParserCrashed
from ingest.parsers import parse_file


def pid_error():
    raise ValueError(str(os.getpid()))


def test_parsing_runs_in_a_child_that_survives_errors_crashes_and_hangs(tmp_path):
    note = tmp_path / "source.txt"
    note.write_text("# Intro\nThe pump alarm sounded twice.\n", encoding="utf-8")
    runner = IsolatedRunner()
    try:
        doc = runner.run(parse_file, str(note), ".txt", "note.txt", None, 1, timeout=120)  # any format, not only PDF
        assert "pump alarm" in doc.elements[0].text
        with pytest.raises(ValueError):
            runner.run(int, "not a number", timeout=60)  # the parser's own error comes back unchanged
        started = time.monotonic()
        with pytest.raises(ParserCrashed):
            runner.run(os._exit, 3, timeout=600)
        assert time.monotonic() - started < 60  # a dead child is reported at once, not after the timeout
        with pytest.raises(TimeoutError):
            runner.run(time.sleep, 60, timeout=2)
        assert runner.run(abs, -5, timeout=120) == 5  # a fresh child takes over
    finally:
        runner.close()


def test_parser_cache_budget_recycles_after_returning_the_completed_result():
    runner = IsolatedRunner(max_cache_mb=0)
    try:
        first = runner.run(os.getpid, timeout=60)
        second = runner.run(os.getpid, timeout=60)
        assert first != second and os.getpid() not in (first, second)
        assert runner.run(abs, -12, timeout=60) == 12
    finally:
        runner.close()


def test_small_parsers_are_reused_below_the_cache_budget():
    runner = IsolatedRunner(max_cache_mb=1024)
    try:
        assert runner.run(os.getpid, timeout=60) == runner.run(os.getpid, timeout=60)
    finally:
        runner.close()


def test_a_failed_parse_also_releases_its_large_or_disabled_cache():
    runner = IsolatedRunner(max_cache_mb=0)
    try:
        with pytest.raises(ValueError) as error:
            runner.run(pid_error, timeout=60)
        failed_pid = int(str(error.value))
        assert runner.run(os.getpid, timeout=60) != failed_pid
    finally:
        runner.close()


def test_an_idle_parser_death_cannot_poison_every_following_document():
    runner = IsolatedRunner()
    try:
        runner.run(abs, -1, timeout=60)
        process, = runner._pool._processes.values()
        process.kill()
        process.join(timeout=10)
        # Let the pool's manager mark the idle process as broken before the next submit.
        deadline = time.monotonic() + 5
        while not runner._pool._broken and time.monotonic() < deadline:
            time.sleep(0.01)
        with pytest.raises(ParserCrashed):
            runner.run(abs, -2, timeout=60)
        assert runner.run(abs, -3, timeout=60) == 3
    finally:
        runner.close()
