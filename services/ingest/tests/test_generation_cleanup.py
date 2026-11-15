"""Old index generations are removed: one replaced by a reindex after the retention period (a question that
started on it can finish), a failed run's at once, the leftovers of other runs that ended without publishing
after a short grace. The current generation, the uploaded file and deleting documents are never touched."""

import asyncio
import sqlite3
import uuid

import pytest
from conftest import Clock
from ingest import paths
from ingest.config import Settings
from ingest.registry import Registry
from ingest.worker import collect_generations, run_job
from rag_common.contracts import DocStatus, DocType
from test_generations import DOC, OWNER, Index, pipeline
from test_lifecycle import V2_SCHEMA, Recorder


class RecordingIndex(Index):
    def __init__(self):
        super().__init__()
        self.deleted = []

    async def delete(self, owner, doc_id, generations=None):
        self.deleted.append((owner, doc_id, generations))


def setup(tmp_path, monkeypatch, retain_h=24.0):
    """The generation tests' pipeline (a CSV whose value the test sets), with a hand-moved clock, a recording
    index and the retention period under test."""
    _settings, registry, pipe, state = pipeline(tmp_path, monkeypatch)
    pipe.settings = Settings(data_dir=tmp_path, audit_url=None, generation_retain_h=retain_h)
    clock = Clock()
    registry.clock = clock
    pipe.indexer = RecordingIndex()
    return pipe.settings, registry, pipe, state, clock


def catalog(registry):
    with sqlite3.connect(registry._path) as db:
        return {g: (p, r, c) for g, p, r, c in db.execute(
            "SELECT generation, published, retired_at, collected_at FROM document_generations WHERE doc_id = ?", (DOC,))}


def files(settings, generation):
    return (paths.table_path(settings, OWNER, DOC, generation).exists(),
            paths.chunk_path(settings, OWNER, DOC, generation).exists())


def reindex(registry, pipe, state, value):
    assert registry.requeue(DOC, OWNER)
    run = registry.claim_next()
    state["value"] = value
    asyncio.run(pipe.process(run))
    return run.run_id


def test_a_reindex_retires_the_replaced_generation_which_stays_readable_for_the_retention_period(tmp_path, monkeypatch):
    settings, registry, pipe, state, clock = setup(tmp_path, monkeypatch, retain_h=24)
    asyncio.run(pipe.process(registry.claim_next()))
    first = registry.get(DOC, OWNER).generation
    second = reindex(registry, pipe, state, 99)
    entries = catalog(registry)
    assert entries[first][0] == 1 and entries[first][1] is not None and entries[first][2] is None  # retired
    assert entries[second] == (1, None, None)  # current: not retired
    assert registry.collectable(settings.generation_retain_s, settings.leftover_grace_s) == []
    clock.advance(hours=23, minutes=59)
    assert asyncio.run(collect_generations(pipe, registry)) == 0
    assert files(settings, first) == (True, True) and registry.is_published(DOC, first)
    clock.advance(minutes=2)
    assert registry.collectable(settings.generation_retain_s, settings.leftover_grace_s) == [(OWNER, DOC, first)]


def test_the_sweep_removes_a_replaced_generation_and_keeps_the_current_one_and_the_upload(tmp_path, monkeypatch):
    settings, registry, pipe, state, clock = setup(tmp_path, monkeypatch, retain_h=1)
    source = paths.source_path(settings, OWNER, DOC, ".csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"value\n11\n")
    asyncio.run(pipe.process(registry.claim_next()))
    first = registry.get(DOC, OWNER).generation
    second = reindex(registry, pipe, state, 99)
    clock.advance(hours=1, seconds=1)
    assert asyncio.run(collect_generations(pipe, registry)) == 1
    assert files(settings, first) == (False, False) and files(settings, second) == (True, True)
    assert source.read_bytes() == b"value\n11\n"
    assert pipe.indexer.deleted == [(OWNER, DOC, [first])]
    assert not registry.is_published(DOC, first) and registry.is_published(DOC, second)
    assert catalog(registry)[first][2] is not None  # collected, kept as a marker
    doc = registry.get(DOC, OWNER)
    assert doc.status == DocStatus.READY and doc.generation == second
    assert asyncio.run(collect_generations(pipe, registry)) == 0  # nothing twice


def test_a_failed_run_is_removed_at_once_and_the_published_generation_stays(tmp_path, monkeypatch):
    settings, registry, pipe, state, _clock = setup(tmp_path, monkeypatch)
    asyncio.run(pipe.process(registry.claim_next()))
    published = registry.get(DOC, OWNER).generation
    assert registry.requeue(DOC, OWNER)
    run = registry.claim_next()
    state["value"] = 99
    pipe.indexer.fail_embedding = True  # after its tables were written
    assert asyncio.run(run_job(pipe, registry, Recorder(), run, asyncio.Event()))
    doc = registry.get(DOC, OWNER)
    assert doc.status == DocStatus.FAILED and doc.generation == published
    assert files(settings, run.run_id) == (False, False) and files(settings, published) == (True, True)
    assert pipe.indexer.deleted == [(OWNER, DOC, [run.run_id])]
    assert catalog(registry)[run.run_id][2] is not None
    assert asyncio.run(collect_generations(pipe, registry)) == 0


def test_runs_that_end_without_publishing_are_retired_and_removed_after_the_grace(tmp_path, monkeypatch):
    settings, registry, _pipe, _state, clock = setup(tmp_path, monkeypatch)
    run = registry.claim_next()
    assert registry.retire_ended_runs() == 0  # the run still holds its document
    assert registry.release(DOC, run.run_id)  # the worker stopped: back in the queue
    assert registry.retire_ended_runs() == 1
    assert registry.retire_ended_runs() == 0
    grace = settings.leftover_grace_s
    assert grace == max(4 * settings.run_lease_s, 600)
    assert registry.collectable(settings.generation_retain_s, grace) == []
    clock.advance(seconds=grace + 1)
    assert registry.collectable(settings.generation_retain_s, grace) == [(OWNER, DOC, run.run_id)]


def test_a_deleting_document_and_the_running_generation_are_never_collected(tmp_path, monkeypatch):
    _settings, registry, pipe, state, clock = setup(tmp_path, monkeypatch, retain_h=0)
    asyncio.run(pipe.process(registry.claim_next()))
    first = registry.get(DOC, OWNER).generation
    second = reindex(registry, pipe, state, 99)
    assert registry.requeue(DOC, OWNER)
    running = registry.claim_next()
    with sqlite3.connect(registry._path) as db:  # even a wrongly retired current or running generation
        db.execute("UPDATE document_generations SET retired_at = '2000-01-01T00:00:00+00:00' WHERE doc_id = ?", (DOC,))
    clock.advance(hours=1)
    assert registry.collectable(0, 0) == [(OWNER, DOC, first)]
    assert registry.mark_deleting(DOC, OWNER) is not None  # processing: its run finishes the deletion
    assert registry.retire_ended_runs() == 0
    assert registry.collectable(0, 0) == []
    assert {second, running.run_id} <= set(catalog(registry))


def test_a_reindex_of_a_legacy_document_removes_the_legacy_files_but_not_the_upload(tmp_path, monkeypatch):
    settings, registry, pipe, state, clock = setup(tmp_path, monkeypatch, retain_h=0)
    run = registry.claim_next()  # published in the pre-generation layout, as before schema v5
    assert registry.complete(DOC, run.run_id, 1, 1, [], []) is not None
    legacy = [paths.table_path(settings, OWNER, DOC), paths.chunk_path(settings, OWNER, DOC),
              paths.asset_dir(settings, OWNER, DOC) / ("a" * 16 + ".png"), paths.source_path(settings, OWNER, DOC, ".csv")]
    for path in legacy:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"legacy")
    second = reindex(registry, pipe, state, 99)
    assert catalog(registry)["legacy"][1] is not None  # retired by the publication
    assert registry.retire_ended_runs() == 1  # the first run's own row, which published the legacy layout
    clock.advance(seconds=settings.leftover_grace_s + 1)
    assert asyncio.run(collect_generations(pipe, registry)) == 2  # the legacy files and the first run's empty row
    assert [path.exists() for path in legacy] == [False, False, False, True]
    assert files(settings, second) == (True, True)
    assert (OWNER, DOC, [None]) in pipe.indexer.deleted
    assert not registry.is_published(DOC, None) and registry.is_published(DOC, second)


def test_a_delete_after_collection_still_targets_every_generation(tmp_path, monkeypatch):
    _settings, registry, pipe, state, _clock = setup(tmp_path, monkeypatch, retain_h=0)
    asyncio.run(pipe.process(registry.claim_next()))
    first = registry.get(DOC, OWNER).generation
    second = reindex(registry, pipe, state, 99)
    assert asyncio.run(collect_generations(pipe, registry)) == 1
    _, claim = registry.mark_deleting(DOC, OWNER)
    _, generations = registry.deletion_targets(DOC, claim)
    assert {first, second} <= set(generations)  # a stalled writer's late files go with the document


def test_v5_upgrade_adds_the_columns_and_retires_the_generations_replaced_before_it(tmp_path):
    path, current, replaced, unpublished = tmp_path / "registry.db", "c" * 32, "d" * 32, "e" * 32
    with sqlite3.connect(path) as db:
        db.executescript(V2_SCHEMA)
        for column in ("run_id TEXT", "lease_until TEXT", "claimed_by TEXT", "index_version INTEGER NOT NULL DEFAULT 1",
                       "generation TEXT", "storage_id TEXT"):
            db.execute("ALTER TABLE documents ADD COLUMN " + column)
        db.execute("CREATE TABLE document_generations (doc_id TEXT NOT NULL, generation TEXT NOT NULL,"
                   " published INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (doc_id, generation))")
        db.execute("INSERT INTO documents (doc_id,owner_id,name,doc_type,ext,size,status,n_chunks,generation,created_at,"
                   "updated_at) VALUES (?,?,'a.csv','spreadsheet','.csv',1,'ready',2,?,'t','t')", (DOC, OWNER, current))
        db.executemany("INSERT INTO document_generations VALUES (?,?,?)",
                       [(DOC, "legacy", 1), (DOC, replaced, 1), (DOC, current, 1), (DOC, unpublished, 0)])
        db.execute("PRAGMA user_version = 5")
    registry = Registry(path)
    entries = catalog(registry)
    assert entries[current] == (1, None, None) and entries[unpublished] == (0, None, None)
    assert entries["legacy"][1] is not None and entries[replaced][1] is not None
    Registry(path)
    assert catalog(registry) == entries  # a second start changes nothing
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 6
    assert registry.retire_ended_runs() == 1  # the unpublished leftover, at the first sweep


def test_a_new_registry_has_the_v6_catalog(tmp_path):
    registry = Registry(tmp_path / "registry.db")
    registry.add(DOC, OWNER, "a.csv", DocType.SPREADSHEET, ".csv", 1, storage_id=uuid.uuid4().hex)
    run = registry.claim_next()
    assert registry.complete(DOC, run.run_id, 1, 0, [], [], generation=run.run_id) is not None
    assert catalog(registry) == {run.run_id: (1, None, None)}


def test_the_retention_period_cannot_be_negative():
    with pytest.raises(ValueError):
        Settings(generation_retain_h=-1)
