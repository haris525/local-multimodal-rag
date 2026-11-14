"""The API module reads its limits on import, so a throwaway scratch folder is pointed to before any test imports
sandbox.api. The sweep (which kills every process of the user) and the wiping of /tmp and /dev/shm stay off: they
are for the container only."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_folder = Path(tempfile.mkdtemp(prefix="sandbox-test-"))
atexit.register(shutil.rmtree, _folder, ignore_errors=True)
SCRATCH = _folder / "scratch"
SCRATCH.mkdir()
os.environ["SANDBOX_SCRATCH_DIR"] = str(SCRATCH)
os.environ.pop("SANDBOX_SWEEP", None)
os.environ.pop("SANDBOX_WIPE_DIRS", None)

from sandbox.config import Limits  # noqa: E402


@pytest.fixture
def limits(tmp_path) -> Limits:
    """Limits with a scratch folder of the test's own. RLIMIT_NPROC counts every process of the user running the
    tests, so it is set out of the way here (one test checks that it is applied)."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return Limits.from_env({"SANDBOX_SCRATCH_DIR": str(scratch), "SANDBOX_MAX_PROCESSES": "100000"})
