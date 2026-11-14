import os
import time

import pytest

from ingest.isolation import IsolatedRunner, ParserCrashed
from ingest.parsers import parse_file


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
