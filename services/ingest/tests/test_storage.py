import asyncio

import pytest

from rag_common.contracts import DocStatus, DocType
from rag_common.testing import new_user_id

from conftest import make_failed, make_ready
from ingest import chunkstore
from ingest.config import Settings
from ingest.paths import source_path
from ingest.pipeline import delete_document
from ingest.registry import Registry
from ingest.validation import InvalidUpload, safe_name, validate

U = new_user_id()


def test_upload_validation_checks_magic_bytes_and_strips_paths():
    assert validate("report.PDF", b"%PDF-1.7 ...") == (".pdf", DocType.PDF)
    with pytest.raises(InvalidUpload):
        validate("report.pdf", b"PK\x03\x04 not a pdf")
    with pytest.raises(InvalidUpload):
        validate("tool.exe", b"MZ")
    assert safe_name("..\\..\\etc/pass:wd.txt") == "pass_wd.txt"


def test_registry_claims_each_job_once_and_gives_up_after_max_attempts(tmp_path, clock):
    reg = Registry(tmp_path / "r.db", clock=clock)
    reg.add("a" * 32, U, "a.txt", DocType.TEXT, ".txt", 1)
    assert reg.claim_next().doc_id == "a" * 32
    assert reg.claim_next() is None  # already processing
    assert reg.recover(max_attempts=2) == 0  # its lease is running: a live worker holds it
    clock.advance(seconds=reg.lease_s + 1)  # the worker crashed: nobody renews the lease
    reg.recover(max_attempts=2)  # attempt 1 < 2, back to the queue
    assert reg.get("a" * 32, U).status == DocStatus.QUEUED
    reg.claim_next()
    clock.advance(seconds=reg.lease_s + 1)
    reg.recover(max_attempts=2)  # crashed again: give up
    assert reg.get("a" * 32, U).status == DocStatus.FAILED


def test_grep_matches_whole_terms_and_ranks_by_distinct_terms(tmp_path):
    path = tmp_path / "d.jsonl"
    chunkstore.write(path, [
        {"id": "1", "chunk_index": 0, "text": "SKU INV-42 is in Austin"},
        {"id": "2", "chunk_index": 1, "text": "INV-420 and INV-42 ship from Denver, Denver"},
        {"id": "3", "chunk_index": 2, "text": "nothing relevant"},
    ])
    hits = chunkstore.grep({"d": path}, ["inv-42", "Denver"], limit=10)
    assert [(h.id, h.matched, h.count) for h in hits] == [("2", 2, 3), ("1", 1, 1)]


def test_only_older_finished_versions_are_superseded(tmp_path):
    reg = Registry(tmp_path / "r.db")
    v1, v2, v3 = "1" * 32, "2" * 32, "3" * 32  # three versions of report.pdf, uploaded in this order
    for doc_id in (v1, v2, v3):
        reg.add(doc_id, U, "report.pdf", DocType.PDF, ".pdf", 1)
    assert make_ready(reg, v2, 5, supersede=True) == []  # v1 is not finished yet, v3 is newer
    make_failed(reg, v1)  # v1 finishing late (failed) never deletes v2 or the queued v3
    assert reg.get(v2, U).status == DocStatus.READY
    superseded = make_ready(reg, v3, supersede=True)
    assert [d.doc_id for d in superseded] == [v1, v2] and all(d.run_id for d in superseded)  # each with a claim
    assert [d.doc_id for d in reg.all(U)] == [v3]  # hidden from the owner at once


class Qdrant:
    def __init__(self, down: bool):
        self.down = down

    async def drop(self, owner, doc_id):
        if self.down:
            raise ConnectionError("qdrant is down")


async def _no_points(owner, doc_id):
    return None


def test_delete_hides_at_once_and_removes_data_before_the_row(tmp_path, clock):
    settings = Settings(data_dir=tmp_path, audit_url=None)
    reg = Registry(settings.registry_path, clock=clock)
    ready, busy = "a" * 32, "b" * 32
    for doc_id in (ready, busy):
        reg.add(doc_id, U, f"{doc_id[0]}.txt", DocType.TEXT, ".txt", 1)
    make_ready(reg, ready)
    source = source_path(settings, U, ready, ".txt")
    source.parent.mkdir(parents=True)
    source.write_text("x")

    doc, claim = reg.mark_deleting(ready, U)
    assert doc.status == DocStatus.READY and claim and reg.get(ready, U) is None  # gone for the owner at once
    assert reg.mark_deleting(ready, U) is None  # a second delete finds nothing
    with pytest.raises(ConnectionError):
        asyncio.run(delete_document(settings, reg, Qdrant(down=True).drop, doc, claim))
    assert reg.get_for_worker(ready).status == DocStatus.DELETING and source.exists()  # the row stays until its data is gone
    assert reg.deleting() == [] and reg.claim_deleting(ready) is None  # held by the failed clean-up for now
    clock.advance(seconds=reg.lease_s + 1)
    assert [d.doc_id for d in reg.deleting()] == [ready]  # the claim ran out: the worker's sweep takes it
    retry = reg.claim_deleting(ready)
    assert asyncio.run(delete_document(settings, reg, Qdrant(down=False).drop, doc, retry))
    assert reg.get_for_worker(ready) is None and not source.exists()
    assert not reg.finish_deleting(ready, claim)  # the first claim is long gone

    run = reg.claim_next().run_id  # being processed: its run keeps the row and finishes the deletion
    doc, claim = reg.mark_deleting(busy, U)
    assert doc.status == DocStatus.PROCESSING and claim is None
    assert not asyncio.run(delete_document(settings, reg, _no_points, doc, claim))  # data removed, row kept
    assert reg.get_for_worker(busy).status == DocStatus.DELETING
    assert not reg.complete(busy, run, 1, 0, [], []) and not reg.progress(busy, run, "x")  # the run lost it
    assert reg.finish_deleting(busy, run)  # the run finishes it (Pipeline.abandon)


def test_stopping_the_worker_requeues_the_document_without_costing_an_attempt(tmp_path, clock):
    from ingest.worker import process_until_stopped

    reg = Registry(tmp_path / "r.db", clock=clock)
    reg.add("a" * 32, U, "big.pdf", DocType.PDF, ".pdf", 1)
    doc = reg.claim_next()

    class SlowPipeline:
        async def process(self, doc):
            await asyncio.sleep(60)

    async def scenario():
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(0.05, stop.set)  # SIGTERM while the file is being parsed
        return await process_until_stopped(SlowPipeline(), reg, doc, stop)

    assert asyncio.run(scenario()) is None
    assert reg.get(doc.doc_id, U).status == DocStatus.QUEUED
    reg.claim_next()
    clock.advance(seconds=reg.lease_s + 1)
    reg.recover(max_attempts=2)  # a real crash on the retry still counts as only the first attempt
    assert reg.get(doc.doc_id, U).status == DocStatus.QUEUED


def test_qdrant_key_file_defaults_to_the_secret_and_an_empty_setting_means_no_key(monkeypatch):
    monkeypatch.delenv("INGEST_QDRANT_API_KEY_FILE", raising=False)
    assert str(Settings().qdrant_api_key_file).replace("\\", "/") == "/run/secrets/qdrant_api_key"
    monkeypatch.setenv("INGEST_QDRANT_API_KEY_FILE", "")
    assert Settings().qdrant_api_key_file is None
