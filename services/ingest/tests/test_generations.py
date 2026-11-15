"""Immutable publication: failed/stale runs cannot alter the last published SQL/chunk/index generation."""

import asyncio
import sqlite3
import threading
import uuid

import duckdb
import pandas as pd
import pytest

from conftest import WordCounter
from rag_common.contracts import DocStatus, DocType
from rag_common.testing import new_user_id

from ingest import chunkstore, paths
from ingest.config import Settings
from ingest.elements import ParsedDocument, Sheet
from ingest.pipeline import Pipeline, RunLost
from ingest.registry import Registry
from ingest.indexer import Indexer
from rag_common.index import COLLECTION, TEXT_VECTOR, scope_filter
from rag_common.snapshots import published_snapshots
from qdrant_client import AsyncQdrantClient, models as qm

OWNER = new_user_id()
DOC = "b" * 32


class Index:
    def __init__(self):
        self.fail_embedding = False
        self.payloads = []

    async def embed_texts(self, texts):
        if self.fail_embedding:
            raise RuntimeError("synthetic embedding outage")
        return [[1.0, 0.0] for _ in texts]

    async def embed_images(self, images):
        return []

    async def replace(self, owner, doc_id, payloads, text_vectors, image_vectors):
        self.payloads.extend(payloads)

    async def delete(self, *args, **kwargs):
        pass


def pipeline(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, audit_url=None)
    registry = Registry(settings.registry_path)
    registry.add(DOC, OWNER, "sample.csv", DocType.SPREADSHEET, ".csv", 10)
    index = Index()
    pipe = Pipeline(settings, registry, index, None, WordCounter(), None)
    state = {"value": 11}

    async def parse(doc, ext):
        return ParsedDocument(sheets=[Sheet("sample", pd.DataFrame({"value": [state["value"]]}))])

    async def summary(*args):
        return None

    monkeypatch.setattr(pipe, "parse", parse)
    monkeypatch.setattr(pipe, "summary_chunk", summary)
    return settings, registry, pipe, state


def test_failed_reindex_preserves_published_tables_chunks_and_metadata(tmp_path, monkeypatch):
    settings, registry, pipe, state = pipeline(tmp_path, monkeypatch)
    asyncio.run(pipe.process(registry.claim_next()))
    published = registry.get(DOC, OWNER)
    old_table = paths.table_path(settings, OWNER, DOC, published.generation)
    old_chunks = paths.chunk_path(settings, OWNER, DOC, published.generation)
    before_table, before_chunks = old_table.read_bytes(), old_chunks.read_bytes()
    assert registry.requeue(DOC, OWNER)
    run = registry.claim_next()
    state["value"] = 99
    pipe.indexer.fail_embedding = True
    with pytest.raises(RuntimeError, match="synthetic embedding outage"):
        asyncio.run(pipe.process(run))
    assert registry.fail(DOC, run.run_id, "synthetic embedding outage")
    after = registry.get(DOC, OWNER)
    assert after.status == DocStatus.FAILED and after.generation == published.generation
    assert after.tables == published.tables and after.n_chunks == published.n_chunks
    assert old_table.read_bytes() == before_table
    assert old_chunks.read_bytes() == before_chunks
    with duckdb.connect(str(old_table), read_only=True) as con:
        assert con.sql(f'SELECT value FROM "{published.tables[0].table}"').fetchall() == [(11,)]


def test_publication_switches_all_metadata_and_keeps_the_previous_files(tmp_path, monkeypatch):
    settings, registry, pipe, state = pipeline(tmp_path, monkeypatch)
    asyncio.run(pipe.process(registry.claim_next()))
    old = registry.get(DOC, OWNER)
    assert registry.requeue(DOC, OWNER)
    run = registry.claim_next()
    state["value"] = 99
    original = pipe.indexer.replace

    async def before_publish(*args):
        await original(*args)
        assert registry.get(DOC, OWNER).generation == old.generation
        assert paths.table_path(settings, OWNER, DOC, old.generation).is_file()
        assert paths.chunk_path(settings, OWNER, DOC, old.generation).is_file()
        assert not registry.is_published(DOC, run.run_id)

    monkeypatch.setattr(pipe.indexer, "replace", before_publish)
    asyncio.run(pipe.process(run))
    new = registry.get(DOC, OWNER)
    assert new.generation == run.run_id != old.generation
    for info, expected in ((old, 11), (new, 99)):
        assert registry.is_published(DOC, info.generation)
        with duckdb.connect(str(paths.table_path(settings, OWNER, DOC, info.generation)), read_only=True) as con:
            assert con.sql(f'SELECT value FROM "{info.tables[0].table}"').fetchall() == [(expected,)]
        assert all(row["generation"] == info.generation for row in chunkstore.read(
            paths.chunk_path(settings, OWNER, DOC, info.generation)))


def test_a_writer_thread_finishing_after_lease_takeover_cannot_change_published_files(tmp_path, monkeypatch, clock):
    import ingest.pipeline as module

    settings, registry, pipe, state = pipeline(tmp_path, monkeypatch)
    registry.clock = clock
    asyncio.run(pipe.process(registry.claim_next()))
    assert registry.requeue(DOC, OWNER)
    stale = registry.claim_next()
    entered, release = threading.Event(), threading.Event()
    original = module.write_tables

    def delayed_write(path, sources):
        if stale.run_id in str(path):
            entered.set()
            assert release.wait(10)
        return original(path, sources)

    monkeypatch.setattr(module, "write_tables", delayed_write)

    async def scenario():
        state["value"] = 99
        task = asyncio.create_task(pipe.process(stale))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            clock.advance(seconds=registry.lease_s + 1)
            assert registry.recover(3) == 1
            replacement = registry.claim_next()
            state["value"] = 123
            await pipe.process(replacement)
        finally:
            release.set()
        with pytest.raises(RunLost):
            await task
        return replacement

    replacement = asyncio.run(scenario())
    published = registry.get(DOC, OWNER)
    assert published.generation == replacement.run_id
    assert registry.complete(DOC, stale.run_id, 999, 0, [], [], generation=stale.run_id) is None
    with duckdb.connect(str(paths.table_path(settings, OWNER, DOC, published.generation)), read_only=True) as con:
        assert con.sql(f'SELECT value FROM "{published.tables[0].table}"').fetchall() == [(123,)]


def test_late_multibatch_qdrant_writes_are_invisible_to_both_old_and_new_readers(tmp_path, monkeypatch):
    settings, registry, pipe, state = pipeline(tmp_path, monkeypatch)

    async def scenario():
        qdrant = AsyncQdrantClient(location=":memory:")
        await qdrant.create_collection(COLLECTION, vectors_config={TEXT_VECTOR: qm.VectorParams(size=2, distance=qm.Distance.COSINE)})
        writer = Indexer(qdrant, None, 10, 2, 1)
        original_replace = pipe.indexer.replace

        async def write(owner, doc_id, payloads, texts, images):
            await original_replace(owner, doc_id, payloads, texts, images)
            await writer.replace(owner, doc_id, payloads, texts, images)

        monkeypatch.setattr(pipe.indexer, "replace", write)
        try:
            await pipe.process(registry.claim_next())
            old = registry.get(DOC, OWNER)
            registry.requeue(DOC, OWNER)
            stale = registry.claim_next()
            entered, release = asyncio.Event(), asyncio.Event()
            upsert = qdrant.upsert

            async def delayed_upsert(collection, points, **kwargs):
                if points[0].payload.get("generation") == stale.run_id:
                    entered.set()
                    await release.wait()  # a request already in flight outlives registry ownership
                return await upsert(collection, points, **kwargs)

            monkeypatch.setattr(qdrant, "upsert", delayed_upsert)
            state["value"] = 99
            task = asyncio.create_task(pipe.process(stale))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert registry.release(DOC, stale.run_id)
                replacement = registry.claim_next()
                state["value"] = 123
                await pipe.process(replacement)
            finally:
                release.set()
            with pytest.raises(RunLost):
                await task
            new = registry.get(DOC, OWNER)
            for info in (old, new):
                with published_snapshots([info]):
                    points, _ = await qdrant.scroll(COLLECTION, scroll_filter=scope_filter(OWNER, [DOC]), limit=100)
                assert len(points) == info.n_chunks
                assert {point.payload["generation"] for point in points} == {info.generation}
            assert new.generation == replacement.run_id
            assert not registry.is_published(DOC, stale.run_id)
        finally:
            await qdrant.close()

    asyncio.run(scenario())


def test_delayed_file_cleanup_cannot_remove_a_later_upload_lifetime(tmp_path, monkeypatch):
    import ingest.pipeline as module

    settings, registry, pipe, _ = pipeline(tmp_path, monkeypatch)
    asyncio.run(pipe.process(registry.claim_next()))
    old, claim = registry.mark_deleting(DOC, OWNER)
    new_storage, new_generation = uuid.uuid4().hex, uuid.uuid4().hex
    original = module.remove_artifacts

    def delayed_cleanup(*args, **kwargs):
        assert registry.finish_deleting(DOC, claim)
        registry.add(DOC, OWNER, "sample.csv", DocType.SPREADSHEET, ".csv", 10, storage_id=new_storage)
        for target in (paths.source_path(settings, OWNER, DOC, ".csv", new_storage),
                       paths.table_path(settings, OWNER, DOC, new_generation),
                       paths.chunk_path(settings, OWNER, DOC, new_generation)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"later upload")
        original(*args, **kwargs)

    monkeypatch.setattr(module, "remove_artifacts", delayed_cleanup)
    assert not asyncio.run(module.delete_document(settings, registry, pipe.indexer.delete, old, claim))
    assert registry.get(DOC, OWNER).storage_id == new_storage
    assert paths.source_path(settings, OWNER, DOC, ".csv", new_storage).read_bytes() == b"later upload"
    assert paths.table_path(settings, OWNER, DOC, new_generation).read_bytes() == b"later upload"
    assert paths.chunk_path(settings, OWNER, DOC, new_generation).read_bytes() == b"later upload"


def test_failed_second_qdrant_batch_keeps_the_complete_previous_snapshot(tmp_path, monkeypatch):
    settings, registry, pipe, state = pipeline(tmp_path, monkeypatch)

    async def scenario():
        qdrant = AsyncQdrantClient(location=":memory:")
        await qdrant.create_collection(COLLECTION, vectors_config={TEXT_VECTOR: qm.VectorParams(size=2, distance=qm.Distance.COSINE)})
        writer = Indexer(qdrant, None, 10, 2, 1)
        monkeypatch.setattr(pipe.indexer, "replace", writer.replace)
        try:
            await pipe.process(registry.claim_next())
            old = registry.get(DOC, OWNER)
            registry.requeue(DOC, OWNER)
            failed = registry.claim_next()
            upsert = qdrant.upsert
            batches = 0

            async def fail_second(collection, points, **kwargs):
                nonlocal batches
                batches += 1
                if batches == 2:
                    raise ConnectionError("synthetic second-batch failure")
                return await upsert(collection, points, **kwargs)

            monkeypatch.setattr(qdrant, "upsert", fail_second)
            state["value"] = 99
            with pytest.raises(ConnectionError):
                await pipe.process(failed)
            assert batches == 2
            assert registry.fail(DOC, failed.run_id, "index unavailable")
            current = registry.get(DOC, OWNER)
            assert current.status == DocStatus.FAILED and current.generation == old.generation
            assert current.tables == old.tables and current.n_chunks == old.n_chunks
            with published_snapshots([current]):
                points, _ = await qdrant.scroll(COLLECTION, scroll_filter=scope_filter(OWNER, [DOC]), limit=100)
            assert len(points) == old.n_chunks
            assert {point.payload["generation"] for point in points} == {old.generation}
            assert not registry.is_published(DOC, failed.run_id)
        finally:
            await qdrant.close()

    asyncio.run(scenario())


def test_v4_upgrade_keeps_legacy_publication_and_is_idempotent(tmp_path):
    path = tmp_path / "registry.db"
    # Start from a real pre-generation schema, including a ready document and its original metadata.
    from test_lifecycle import V2_SCHEMA
    with sqlite3.connect(path) as db:
        db.executescript(V2_SCHEMA)
        for column in ("run_id TEXT", "lease_until TEXT", "claimed_by TEXT", "index_version INTEGER NOT NULL DEFAULT 0"):
            db.execute("ALTER TABLE documents ADD COLUMN " + column)
        db.execute("INSERT INTO documents (doc_id,owner_id,name,doc_type,ext,size,status,n_chunks,created_at,updated_at)"
                   " VALUES (?,?,'legacy.txt','text','.txt',12,'ready',3,'original','original')", (DOC, OWNER))
        db.execute("PRAGMA user_version = 4")
    registry = Registry(path)
    legacy = registry.get(DOC, OWNER)
    assert legacy.generation is None and legacy.storage_id is None and legacy.n_chunks == 3
    assert legacy.created_at == legacy.updated_at == "original"
    assert registry.is_published(DOC, None)
    assert Registry(path).get(DOC, OWNER) == legacy
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT generation,published FROM document_generations").fetchall() == [("legacy", 1)]


def test_former_deleter_cannot_capture_a_reuploads_new_generations(tmp_path):
    registry = Registry(tmp_path / "registry.db")
    registry.add(DOC, OWNER, "old.csv", DocType.SPREADSHEET, ".csv", 1)
    old, claim = registry.mark_deleting(DOC, OWNER)
    assert registry.finish_deleting(DOC, claim)
    registry.add(DOC, OWNER, "new.csv", DocType.SPREADSHEET, ".csv", 1, storage_id=uuid.uuid4().hex)
    new = registry.claim_next()
    assert registry.complete(DOC, new.run_id, 1, 0, [], [], generation=new.run_id) is not None
    assert registry.deletion_targets(DOC, claim) is None
    assert registry.get(DOC, OWNER).generation == new.run_id


def test_generation_cleanup_removes_crashed_writer_temporaries_only(tmp_path):
    settings = Settings(data_dir=tmp_path, audit_url=None)
    old, new = uuid.uuid4().hex, uuid.uuid4().hex
    kept = paths.table_path(settings, OWNER, DOC, new)
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"later generation")
    targets = []
    for path in (paths.table_path(settings, OWNER, DOC, old), paths.chunk_path(settings, OWNER, DOC, old)):
        path.parent.mkdir(parents=True, exist_ok=True)
        for artifact in (path, path.with_suffix(".tmp"), path.with_suffix(".tmp.wal"), path.with_suffix(path.suffix + ".wal")):
            artifact.write_bytes(b"abandoned private write")
            targets.append(artifact)
    paths.remove_worker_artifacts(settings, OWNER, DOC, old)
    assert not any(path.exists() for path in targets)
    assert kept.read_bytes() == b"later generation"


def test_a_deletion_claim_uses_its_own_source_lifetime_even_if_the_list_was_stale(tmp_path):
    from ingest.pipeline import delete_document

    settings = Settings(data_dir=tmp_path, audit_url=None)
    registry = Registry(settings.registry_path)
    registry.add(DOC, OWNER, "old.csv", DocType.SPREADSHEET, ".csv", 1, storage_id=uuid.uuid4().hex)
    listed, first_claim = registry.mark_deleting(DOC, OWNER)
    assert registry.finish_deleting(DOC, first_claim)
    storage = uuid.uuid4().hex
    registry.add(DOC, OWNER, "new.csv", DocType.SPREADSHEET, ".csv", 1, storage_id=storage)
    source = paths.source_path(settings, OWNER, DOC, ".csv", storage)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"later upload, now also deleted by its owner")
    _, claim = registry.mark_deleting(DOC, OWNER)

    async def drop(*args):
        pass

    # A sweeper's list can predate a re-upload; its new claim must capture the new source too.
    assert asyncio.run(delete_document(settings, registry, drop, listed, claim))
    assert not source.exists()


def test_pipeline_bounds_vectors_per_batch_and_keeps_global_image_indices(tmp_path, monkeypatch):
    from ingest.elements import Chunk

    settings, registry, pipe, _ = pipeline(tmp_path, monkeypatch)
    settings.upsert_batch = 3
    chunks = [Chunk(kind="image" if i in (4, 8) else "text", text=str(i), search_text=str(i), n_tokens=1,
                    image=b"synthetic" if i in (4, 8) else None,
                    asset="a" * 16 + ".png" if i in (4, 8) else None) for i in range(11)]
    monkeypatch.setattr(pipe, "chunk", lambda *args: chunks)
    calls = []

    async def texts(values):
        assert len(values) <= 3
        return [[float(value), 0.0] for value in values]

    async def images(values):
        return [[1.0, 0.0] for value in values]

    async def replace(owner, doc_id, payloads, vectors, image_vectors):
        calls.append(([p.chunk_index for p in payloads], set(image_vectors)))
        assert [p.chunk_index for p in payloads] == [int(vector[0]) for vector in vectors]

    monkeypatch.setattr(pipe.indexer, "embed_texts", texts)
    monkeypatch.setattr(pipe.indexer, "embed_images", images)
    monkeypatch.setattr(pipe.indexer, "replace", replace)
    asyncio.run(pipe.process(registry.claim_next()))
    assert calls == [([0, 1, 2], set()), ([3, 4, 5], {4}), ([6, 7, 8], {8}), ([9, 10], set())]
    assert registry.get(DOC, OWNER).n_chunks == 11
