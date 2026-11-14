"""ingest-api builds its settings, registry and verifier on import, so a throwaway data folder and keys are
pointed to before any test imports ingest.api."""

import atexit
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rag_common.contracts import DocStatus
from rag_common.testing import IdentityKeys

_folder = Path(tempfile.mkdtemp(prefix="ingest-test-"))
atexit.register(shutil.rmtree, _folder, ignore_errors=True)
KEYS = IdentityKeys(_folder / "keys")
DATA = _folder / "data"
os.environ["INGEST_DATA_DIR"] = str(DATA)
os.environ["INGEST_PUBLIC_KEYS_FILE"] = str(KEYS.public_keys_file)
os.environ["INGEST_AUDIT_URL"] = ""
os.environ["INGEST_QDRANT_API_KEY_FILE"] = ""  # no key: the tests use an in-memory Qdrant or none


class WordCounter:
    """Stand-in for the model tokenizer: one token per word or punctuation mark."""

    _TOKEN = re.compile(r"\w+|[^\w\s]")

    def count(self, text: str) -> int:
        return len(self._TOKEN.findall(text))

    def count_many(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]

    def windows(self, text: str, size: int) -> list[str]:
        spans = [m.span() for m in self._TOKEN.finditer(text)]
        return [text[spans[i][0] : spans[min(i + size, len(spans)) - 1][1]] for i in range(0, len(spans), size)]


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()


class Clock:
    """The registry's clock, moved by hand: leases run out when a test says so."""

    def __init__(self):
        self.now = datetime.now(UTC).replace(microsecond=0)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def clock() -> Clock:
    return Clock()


def run_of(reg, doc_id: str) -> str:
    """Claim this queued document for a test run (claim_next takes the oldest); returns the run id."""
    return reg.transition(doc_id, DocStatus.PROCESSING, sources=(DocStatus.QUEUED,), run_id="7" * 32,
                          lease_until=None, attempts=1).run_id


def make_ready(reg, doc_id: str, n_chunks: int = 1, supersede: bool = False) -> list:
    """A queued document through one test run to ready; returns the versions it superseded."""
    done = reg.complete(doc_id, run_of(reg, doc_id), n_chunks, 0, [], [], supersede=supersede)
    assert done is not None
    return done.superseded


def make_failed(reg, doc_id: str, error: str = "parse error") -> None:
    assert reg.fail(doc_id, run_of(reg, doc_id), error)
