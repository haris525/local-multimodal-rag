"""The document lifecycle (docs/STATE_MACHINES.md, M5 and M6): the allowed moves, the run that owns a
processing row, the deleting state, the schema upgrade, and the proofs P1, P2a, P2b, P2c and P4 turned around."""

import asyncio
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from conftest import KEYS, make_ready, run_of
from rag_common.contracts import DocStatus, DocType
from rag_common.testing import new_user_id

from ingest import paths
from ingest.config import Settings
from ingest.elements import Element, ParsedDocument
from ingest.pipeline import Pipeline, RunLost, delete_document
from ingest.registry import NEXT, SCHEMA_VERSION, NewerVersion, Registry, StillDeleting
from ingest.worker import process_until_stopped, run_job, sweep, sweeper

A = new_user_id()
DOC = "a" * 32
STATUSES = list(DocStatus)


class WordCounter:
    _TOKEN = re.compile(r"\w+|[^\w\s]")

    def count(self, text):
        return len(self._TOKEN.findall(text))

    def count_many(self, texts):
        return [self.count(t) for t in texts]

    def windows(self, text, size):
        spans = [m.span() for m in self._TOKEN.finditer(text)]
        return [text[spans[i][0]: spans[min(i + size, len(spans)) - 1][1]] for i in range(0, len(spans), size)]


class FakeRunner:
    def __init__(self):
        self.closed = 0

    def run(self, fn, *args, timeout):
        return ParsedDocument(elements=[Element(kind="text", text="Hello world from the owner.", section="Intro")])

    def close(self):
        self.closed += 1


class FakeIndexer:
    def __init__(self, on_replace=None):
        self.on_replace = on_replace
        self.replaced, self.deleted = [], []

    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_images(self, images):
        return []

    async def replace(self, user_id, doc_id, payloads, text_vectors, image_vectors):
        self.replaced.append((doc_id, payloads[0].doc_name))
        if self.on_replace:
            self.on_replace()

    async def delete(self, user_id, doc_id):
        self.deleted.append(doc_id)


class Recorder:
    def __init__(self):
        self.events = []

    async def record(self, action, subject=None, actor=None, *, user_id=None, **detail):
        self.events.append((action, subject))

    async def aclose(self):
        pass


def setup(tmp_path, indexer=None, clock=None):
    s = Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None)
    reg = Registry(s.registry_path, clock=clock)
    return s, reg, Pipeline(s, reg, indexer or FakeIndexer(), None, WordCounter(), FakeRunner())


def queue(s, reg, doc_id, name="report.txt"):
    reg.add(doc_id, A, name, DocType.TEXT, ".txt", 1)
    source = paths.source_path(s, A, doc_id, ".txt")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("Hello")
    return source


def force(reg, doc_id, status: DocStatus):
    """Put a row in a status directly (a processing row needs a run)."""
    with sqlite3.connect(reg._path) as db:
        db.execute("UPDATE documents SET status = ?, run_id = ? WHERE doc_id = ?",
                   (status.value, "r" * 32 if status in (DocStatus.PROCESSING, DocStatus.DELETING) else None, doc_id))


# --- the table ------------------------------------------------------------------------------


@pytest.mark.parametrize("before", STATUSES)
@pytest.mark.parametrize("after", STATUSES)
def test_every_move_follows_next(tmp_path, before, after):
    reg = Registry(tmp_path / "r.db")
    reg.add(DOC, A, "a.txt", DocType.TEXT, ".txt", 1)
    force(reg, DOC, before)
    fields = {"run_id": "n" * 32} if after == DocStatus.PROCESSING else {"run_id": None}
    allowed = after in NEXT[before]
    moved = reg.transition(DOC, after, **fields)
    assert (moved is not None) == allowed
    assert reg.get_for_worker(DOC).status == (after if allowed else before)


def test_a_narrower_source_than_next_allows_is_refused_up_front(tmp_path):
    reg = Registry(tmp_path / "r.db")
    with pytest.raises(ValueError):
        reg.transition(DOC, DocStatus.READY, sources=(DocStatus.QUEUED,))  # queued never goes straight to ready


def test_the_status_check_matches_the_enum_and_a_processing_row_needs_a_run(tmp_path):
    reg = Registry(tmp_path / "r.db")
    insert = ("INSERT INTO documents (doc_id, owner_id, name, doc_type, ext, size, status, run_id, created_at,"
              " updated_at) VALUES (?, ?, 'x', 'text', '.txt', 1, ?, ?, 't', 't')")
    with sqlite3.connect(reg._path) as db:
        for i, status in enumerate(STATUSES):
            db.execute(insert, (f"{i}" * 32, A, status.value, "r" * 32))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(insert, ("9" * 32, A, "archived", None))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(insert, ("8" * 32, A, "processing", None))


# --- schema upgrade from v2 -----------------------------------------------------------------

V2_SCHEMA = """
CREATE TABLE documents (doc_id TEXT PRIMARY KEY, owner_id TEXT, name TEXT NOT NULL, doc_type TEXT NOT NULL,
    ext TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL, progress TEXT NOT NULL DEFAULT '', error TEXT,
    warnings TEXT NOT NULL DEFAULT '[]', n_chunks INTEGER NOT NULL DEFAULT 0, n_images INTEGER NOT NULL DEFAULT 0,
    tables TEXT NOT NULL DEFAULT '[]', attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE INDEX documents_status ON documents (status, created_at);
CREATE INDEX documents_owner ON documents (owner_id, created_at);
CREATE INDEX documents_owner_name ON documents (owner_id, name);
CREATE TABLE deleted_users (user_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL);
PRAGMA user_version = 2;
"""


def test_a_v2_registry_is_rebuilt_with_the_check_and_keeps_every_row(tmp_path, clock):
    path, gone = tmp_path / "registry.db", new_user_id()
    rows = [("4" * 32, "ready", 3, "report.pdf"), ("2" * 32, "processing", 0, "big.pdf"),
            ("3" * 32, "failed", 0, "bad.pdf"), ("1" * 32, "queued", 0, "later.txt"),
            ("5" * 32, "archived", 0, "odd.txt")]  # a status no code writes
    with sqlite3.connect(path) as db:
        db.executescript(V2_SCHEMA)
        for doc_id, status, n_chunks, name in rows:
            db.execute("INSERT INTO documents (doc_id, owner_id, name, doc_type, ext, size, status, n_chunks, attempts,"
                       " progress, created_at, updated_at) VALUES (?, ?, ?, 'pdf', '.pdf', 1, ?, ?, 1, 'parsing', ?, ?)",
                       (doc_id, A, name, status, n_chunks, f"2026-01-0{len(name) % 9 + 1}", "t"))
        db.execute("INSERT INTO deleted_users VALUES (?, 't')", (gone,))
        before = db.execute("SELECT doc_id FROM documents ORDER BY rowid").fetchall()

    reg = Registry(path, clock=clock)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
        assert db.execute("SELECT doc_id FROM documents ORDER BY rowid").fetchall() == before  # version order kept
        assert {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")} >= {
            "documents_status", "documents_owner", "documents_owner_name"}
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE documents SET status = 'archived'")
    assert reg.is_deleted(gone)
    docs = {d.doc_id: d for d in (reg.get_for_worker(r[0]) for r in rows)}
    assert docs["4" * 32].status == DocStatus.READY and docs["4" * 32].n_chunks == 3
    assert docs["5" * 32].status == DocStatus.FAILED and "unknown status 'archived'" in docs["5" * 32].error
    assert docs["2" * 32].status == DocStatus.PROCESSING  # left processing by a worker that is gone ...
    assert reg.recover(max_attempts=2) == 1  # ... and taken back at the worker's start (no lease)
    assert reg.get("2" * 32, A).status == DocStatus.QUEUED and reg.get("2" * 32, A).progress == ""
    Registry(path)  # a second start changes nothing
    assert len(reg.all(A)) == 5


# --- runs: every write names its run --------------------------------------------------------


def test_P2a_a_stale_run_cannot_complete_a_reupload(tmp_path, clock):
    """Proof P2a turned around: delete while processing, upload the same file again under another name."""
    s, reg, pipe = setup(tmp_path, clock=clock)
    queue(s, reg, DOC, name="old-name.txt")
    stale = reg.claim_next()
    reg.mark_deleting(DOC, A)
    with pytest.raises(StillDeleting):  # the same bytes cannot come back while the old lifetime is being removed
        reg.add(DOC, A, "new-name.txt", DocType.TEXT, ".txt", 1)
    assert not reg.complete(DOC, stale.run_id, 7, 0, [], [])
    assert asyncio.run(pipe.abandon(stale)) == "deleted" and reg.get_for_worker(DOC) is None
    reg.add(DOC, A, "new-name.txt", DocType.TEXT, ".txt", 1)  # now a new lifetime
    assert not reg.complete(DOC, stale.run_id, 7, 0, [], []) and not reg.progress(DOC, stale.run_id, "x")
    fresh = reg.claim_next()
    assert fresh.doc_id == DOC and fresh.name == "new-name.txt" and fresh.run_id != stale.run_id


def test_P2b_a_stale_run_cannot_fail_a_reupload(tmp_path):
    s, reg, pipe = setup(tmp_path)
    queue(s, reg, DOC)
    stale = reg.claim_next()
    reg.mark_deleting(DOC, A)
    assert not reg.fail(DOC, stale.run_id, "error of the old run")
    asyncio.run(pipe.abandon(stale))
    reg.add(DOC, A, "report.txt", DocType.TEXT, ".txt", 1)
    assert not reg.fail(DOC, stale.run_id, "error of the old run") and not reg.release(DOC, stale.run_id)
    assert reg.get(DOC, A).status == DocStatus.QUEUED and reg.claim_next().doc_id == DOC


def test_P2c_a_stale_run_never_supersedes_a_document_the_user_kept(tmp_path):
    """Proof P2c turned around: report.txt v1 is ready; v2 (report.txt) is deleted while it is indexed and the
    same bytes are uploaded as report-final.txt. v1 stays; the re-upload is indexed under its own name."""
    holder = {}

    def delete_and_reupload():
        holder["reg"].mark_deleting(v2, A)
        with pytest.raises(StillDeleting):
            holder["reg"].add(v2, A, "report-final.txt", DocType.TEXT, ".txt", 1)

    indexer = FakeIndexer(on_replace=delete_and_reupload)
    s, reg, pipe = setup(tmp_path, indexer)
    holder["reg"] = reg
    v1, v2 = "1" * 32, "2" * 32
    queue(s, reg, v1, name="report.txt")
    make_ready(reg, v1, 3)
    queue(s, reg, v2, name="report.txt")
    audit = Recorder()
    assert asyncio.run(run_job(pipe, reg, audit, reg.claim_next(), asyncio.Event()))
    assert reg.get(v1, A).status == DocStatus.READY and paths.source_path(s, A, v1, ".txt").exists()
    assert reg.get_for_worker(v2) is None and not paths.source_path(s, A, v2, ".txt").exists()
    assert audit.events == [] and indexer.deleted == [v2]

    indexer.on_replace = None
    queue(s, reg, v2, name="report-final.txt")  # the re-upload, now accepted
    assert asyncio.run(run_job(pipe, reg, audit, reg.claim_next(), asyncio.Event()))
    assert reg.get(v2, A).name == "report-final.txt" and reg.get(v2, A).status == DocStatus.READY
    assert reg.get(v1, A).status == DocStatus.READY  # another name supersedes nothing
    assert indexer.replaced[-1] == (v2, "report-final.txt") and audit.events == [("document.ready", v2)]


def test_P4_a_second_worker_start_leaves_a_running_job_alone(tmp_path, clock):
    """Proof P4 turned around: recover() takes back only jobs whose lease ran out."""
    worker_a = Registry(tmp_path / "r.db", clock=clock)
    worker_a.add(DOC, A, "big.pdf", DocType.PDF, ".pdf", 1)
    run_a = worker_a.claim_next().run_id
    worker_b = Registry(tmp_path / "r.db", clock=clock)
    assert worker_b.recover(max_attempts=2) == 0 and worker_b.claim_next() is None
    clock.advance(seconds=60)
    assert worker_a.renew(DOC, run_a)  # worker A is alive and renews
    clock.advance(seconds=60)
    assert worker_b.recover(max_attempts=2) == 0
    clock.advance(seconds=worker_a.lease_s + 1)  # A died: nobody renewed
    assert worker_b.recover(max_attempts=2) == 1
    run_b = worker_b.claim_next().run_id
    assert run_b != run_a and not worker_a.complete(DOC, run_a, 1, 0, [], [])  # A's late writes change nothing
    assert worker_b.complete(DOC, run_b, 1, 0, [], [])


# --- supersede ------------------------------------------------------------------------------


class DeleteFailsIndexer(FakeIndexer):
    async def delete(self, user_id, doc_id):
        await super().delete(user_id, doc_id)
        raise RuntimeError("qdrant timed out")


def test_P3_a_failed_supersede_clean_up_leaves_the_new_version_ready(tmp_path, clock):
    """Proof P3 turned around: the old version is hidden in the transaction that marks the new one ready; its
    clean-up fails without touching the new version, and the sweep finishes it later."""
    indexer = DeleteFailsIndexer()
    s, reg, pipe = setup(tmp_path, indexer, clock)
    old, new = "1" * 32, "2" * 32
    old_source = queue(s, reg, old)
    make_ready(reg, old)
    queue(s, reg, new)
    audit = Recorder()
    asyncio.run(run_job(pipe, reg, audit, reg.claim_next(), asyncio.Event()))
    after = reg.get(new, A)
    assert after.status == DocStatus.READY and after.n_chunks > 0 and after.error is None
    assert audit.events == [("document.superseded", old), ("document.ready", new)]
    assert reg.get(old, A) is None and reg.get_for_worker(old).status == DocStatus.DELETING  # hidden, kept
    assert old_source.exists() and indexer.deleted == [old]  # points before files: the files wait for a retry
    indexer.delete = FakeIndexer.delete.__get__(indexer)  # Qdrant is back
    clock.advance(seconds=reg.lease_s + 1)
    asyncio.run(sweep(pipe, reg, 2))
    assert reg.get_for_worker(old) is None and not old_source.exists()
    assert reg.get(new, A).status == DocStatus.READY


def test_complete_and_supersede_are_one_transaction(tmp_path):
    """A run that lost its document supersedes nothing; a run that completes hides the old version in the
    same step (a crash right after leaves no moment with both versions listed)."""
    s, reg, _ = setup(tmp_path)
    v1, v2 = "1" * 32, "2" * 32
    queue(s, reg, v1)
    make_ready(reg, v1)
    queue(s, reg, v2)
    run = reg.claim_next().run_id
    reg.mark_deleting(v2, A)
    assert reg.complete(v2, run, 1, 0, [], [], supersede=True) is None
    assert reg.get(v1, A).status == DocStatus.READY
    asyncio.run(Pipeline(s, reg, FakeIndexer(), None, WordCounter(), FakeRunner()).abandon(
        reg.get_for_worker(v2).model_copy(update={"run_id": run})))
    queue(s, reg, v2)
    done = reg.complete(v2, reg.claim_next().run_id, 1, 0, [], [], supersede=True)
    assert [d.doc_id for d in done.superseded] == [v1] and [d.doc_id for d in reg.all(A)] == [v2]


# --- deleting -------------------------------------------------------------------------------


def test_P1_a_document_deleted_while_processing_leaves_nothing_when_the_worker_dies(tmp_path, clock):
    """Proof P1 turned around: the delete removes the points at once and leaves the files to the run; when the
    worker dies instead, the next worker's sweep removes everything once the run's lease has run out."""
    indexer = FakeIndexer()
    s, reg, pipe = setup(tmp_path, indexer, clock)
    source = queue(s, reg, DOC)
    reg.claim_next()
    doc, claim = reg.mark_deleting(DOC, A)  # the user deletes it; ingest-api removes its points
    assert claim is None
    assert not asyncio.run(delete_document(s, reg, indexer.delete, doc, claim))
    assert indexer.deleted == [DOC] and source.exists() and reg.get(DOC, A) is None
    chunk = paths.chunk_path(s, A, DOC)  # written by the run after the delete, before the worker died
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_text("{}")

    restarted = Registry(s.registry_path, clock=clock)
    restarted_pipe = Pipeline(s, restarted, indexer, None, WordCounter(), FakeRunner())
    asyncio.run(sweep(restarted_pipe, restarted, 2))
    assert chunk.exists() and restarted.get_for_worker(DOC)  # the run's lease may still be alive
    clock.advance(seconds=restarted.lease_s + 1)
    asyncio.run(sweep(restarted_pipe, restarted, 2))
    assert restarted.get_for_worker(DOC) is None and restarted.claim_next() is None
    assert not chunk.exists() and not source.exists() and indexer.deleted == [DOC, DOC]


def test_a_run_stops_within_a_renewal_when_its_document_is_deleted(tmp_path):
    """The run's lease renewal sees the deletion; the job is cancelled and the parser child killed."""
    s, reg, _ = setup(tmp_path)
    queue(s, reg, DOC)
    doc = reg.claim_next()
    runner = FakeRunner()

    class SlowPipeline:
        def __init__(self):
            self.runner = runner

        async def process(self, doc):
            await asyncio.sleep(60)

    async def scenario():
        asyncio.get_running_loop().call_later(0.05, reg.mark_deleting, DOC, A)
        await process_until_stopped(SlowPipeline(), reg, doc, asyncio.Event(), renew_s=0.02)

    with pytest.raises(RunLost):
        asyncio.run(asyncio.wait_for(scenario(), 5))
    assert runner.closed == 1


def test_a_failed_clean_up_is_finished_by_the_sweep(tmp_path, clock):
    class DownOnce(FakeIndexer):
        async def delete(self, user_id, doc_id):
            if not self.deleted:
                self.deleted.append("failed")
                raise ConnectionError("qdrant is down")
            await super().delete(user_id, doc_id)

    indexer = DownOnce()
    s, reg, pipe = setup(tmp_path, indexer, clock)
    source = queue(s, reg, DOC)
    make_ready(reg, DOC)
    doc, claim = reg.mark_deleting(DOC, A)
    with pytest.raises(ConnectionError):
        asyncio.run(delete_document(s, reg, indexer.delete, doc, claim))
    asyncio.run(sweep(pipe, reg, 2))
    assert reg.get_for_worker(DOC) is not None  # the failed clean-up still holds it
    clock.advance(seconds=reg.lease_s + 1)
    asyncio.run(sweep(pipe, reg, 2))
    assert reg.get_for_worker(DOC) is None and not source.exists() and indexer.deleted == ["failed", DOC]


# --- through ingest-api ---------------------------------------------------------------------


class FakeQdrant:
    def __init__(self, down=False):
        self.down, self.deleted = down, []

    async def delete(self, collection, points_selector, wait=True):
        if self.down:
            raise ConnectionError("qdrant is down")
        self.deleted.append(points_selector.filter)


@pytest.fixture
def api(monkeypatch):
    import ingest.api as api

    for path in (api.settings.files_dir / ".incoming", api.settings.assets_dir, api.settings.tables_dir,
                 api.settings.chunks_dir):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "audit", Recorder())
    monkeypatch.setattr(api, "qdrant", FakeQdrant())
    return api


def claim(reg, doc_id):
    """Claim this document for a run, with a lease (the shared registry holds other tests' queued rows)."""
    return reg.transition(doc_id, DocStatus.PROCESSING, sources=(DocStatus.QUEUED,), run_id="6" * 32,
                          lease_until=reg._lease(), attempts=1)


def post(client, user, name, data):
    return client.post("/v1/documents", files=[("files", (name, data))], headers=KEYS.headers(user)).json()[0]


def test_the_document_list_leaves_out_column_statistics_that_status_keeps(api):
    """The UI polls the list; the full profiles (top values, quartiles, examples) go to rag through /status."""
    from rag_common.contracts import ColumnSchema, TableSchema

    a = new_user_id()
    client, h = TestClient(api.app), KEYS.headers(a)
    doc_id = post(client, a, "sales.csv", b"Region,Amount\nNorth,1\n" + a.encode())["document"]["doc_id"]
    run = claim(api.registry, doc_id)
    col = ColumnSchema(name="Region", sql_name="region", type="text", profile="2 distinct", pattern="codes",
                       top=[("North", 5)], examples=["North"], distinct=2, empty=0)
    api.registry.complete(doc_id, run.run_id, 1, 0, [], [TableSchema(table="sales", source="x", rows=2, row_filter=False,
                                                                    columns=[col])])
    [listed] = [d for d in client.get("/v1/documents", headers=h).json() if d["doc_id"] == doc_id]
    column = listed["tables"][0]["columns"][0]
    assert column["profile"] == "2 distinct" and column["pattern"] == "codes"
    assert not {"top", "quantiles", "examples", "min", "max", "empty", "distinct", "header"} & set(column)
    assert not {"layout", "notes"} & set(listed["tables"][0])
    [full] = client.post("/v1/documents/status", json={"doc_ids": [doc_id]}, headers=h).json()
    assert full["tables"][0]["columns"][0]["top"] == [["North", 5]]


def test_deleting_a_processing_document_through_the_api(api):
    a = new_user_id()
    client, h = TestClient(api.app), KEYS.headers(a)
    data = b"being indexed " + a.encode()
    doc_id = post(client, a, "big.txt", data)["document"]["doc_id"]
    run = claim(api.registry, doc_id)
    listed = client.get("/v1/documents", headers=h).json()
    assert listed[0]["status"] == "processing" and "run_id" not in listed[0]  # the run id never leaves ingest
    r = client.delete(f"/v1/documents/{doc_id}", headers=h)
    assert r.status_code == 202 and r.json() == {"doc_id": doc_id, "status": "deleting"}
    assert client.get("/v1/documents", headers=h).json() == []  # gone for the user at once
    assert client.get(f"/v1/documents/{doc_id}", headers=h).status_code == 404
    assert client.delete(f"/v1/documents/{doc_id}", headers=h).status_code == 404
    assert len(api.qdrant.deleted) == 1  # the points go at once; the files are the run's to remove
    assert paths.source_path(api.settings, a, doc_id, ".txt").exists()
    again = post(client, a, "big-final.txt", data)  # the run has not finished the deletion yet
    assert again["error"] == api.STILL_DELETING and again["document"] is None
    assert api.registry.finish_deleting(doc_id, run.run_id)  # what the run does when it sees the deletion
    again = post(client, a, "big-final.txt", data)
    assert again["document"]["name"] == "big-final.txt" and again["document"]["status"] == "queued"


def test_a_delete_whose_clean_up_fails_is_accepted_and_finished_later(api, monkeypatch):
    a = new_user_id()
    client, h = TestClient(api.app), KEYS.headers(a)
    data = b"keep trying " + a.encode()
    doc_id = post(client, a, "x.txt", data)["document"]["doc_id"]
    make_ready(api.registry, doc_id)
    monkeypatch.setattr(api, "qdrant", FakeQdrant(down=True))
    assert client.delete(f"/v1/documents/{doc_id}", headers=h).status_code == 202
    assert client.get("/v1/documents", headers=h).json() == []
    assert post(client, a, "x.txt", data)["error"] == api.STILL_DELETING  # the failed clean-up holds it
    with sqlite3.connect(api.registry._path) as db:  # its claim runs out
        db.execute("UPDATE documents SET lease_until = '2000-01-01T00:00:00+00:00' WHERE doc_id = ?", (doc_id,))
    monkeypatch.setattr(api, "qdrant", FakeQdrant())
    again = post(client, a, "x.txt", data)  # the upload finishes the old deletion first
    assert again["document"]["status"] == "queued" and len(api.qdrant.deleted) == 1
    assert paths.source_path(api.settings, a, doc_id, ".txt").exists()


def test_reindex_is_refused_for_a_queued_or_processing_document(api):
    a = new_user_id()
    client, h = TestClient(api.app), KEYS.headers(a)
    doc_id = post(client, a, "q.txt", b"queued " + a.encode())["document"]["doc_id"]
    r = client.post(f"/v1/documents/{doc_id}/reindex", headers=h)
    assert r.status_code == 409 and r.json()["detail"] == "document is already queued"
    claim(api.registry, doc_id)
    assert client.post(f"/v1/documents/{doc_id}/reindex", headers=h).json()["detail"] == "document is being processed"


# --- new versions: an older version waits for the newer upload; the same content again is the same document --


def test_an_older_version_is_not_reindexed_while_a_newer_upload_is_pending(tmp_path):
    """Without the check, reindexing v1 while v2 is processing would leave both ready and listed: v2's
    supersede only reaches versions in ready or failed, and v1's own only reaches versions uploaded before it."""
    reg = Registry(tmp_path / "r.db")
    v1, v2, other = "1" * 32, "2" * 32, "3" * 32
    reg.add(v1, A, "sales.csv", DocType.SPREADSHEET, ".csv", 1)
    make_ready(reg, v1, supersede=True)
    reg.add(other, new_user_id(), "sales.csv", DocType.SPREADSHEET, ".csv", 1)  # another user's: no effect
    assert reg.requeue(v1, A)
    make_ready(reg, v1, supersede=True)
    reg.add(v2, A, "sales.csv", DocType.SPREADSHEET, ".csv", 2)
    with pytest.raises(NewerVersion):  # v2 queued
        reg.requeue(v1, A)
    run = run_of(reg, v2)
    with pytest.raises(NewerVersion):  # v2 processing
        reg.requeue(v1, A)
    assert reg.get(v1, A).status == DocStatus.READY  # untouched, still searchable
    done = reg.complete(v2, run, 1, 0, [], [], supersede=True)
    assert [d.doc_id for d in done.superseded] == [v1]
    assert [(d.doc_id, d.status) for d in reg.all(A)] == [(v2, DocStatus.READY)]


def test_an_older_version_can_be_reindexed_when_the_newer_upload_failed(tmp_path):
    reg = Registry(tmp_path / "r.db")
    v1, v2 = "1" * 32, "2" * 32
    reg.add(v1, A, "sales.csv", DocType.SPREADSHEET, ".csv", 1)
    make_ready(reg, v1, supersede=True)
    reg.add(v2, A, "sales.csv", DocType.SPREADSHEET, ".csv", 2)
    assert reg.fail(v2, run_of(reg, v2), "parse error")
    assert reg.requeue(v1, A)  # the user keeps the old version; v2 stays failed until retried or deleted
    assert reg.requeue(v2, A)  # and the newer one can be retried: it supersedes v1 when it is ready


def test_reindex_of_an_older_version_says_a_newer_one_is_coming(api):
    a = new_user_id()
    client, h = TestClient(api.app), KEYS.headers(a)
    v1 = post(client, a, "sales.csv", b"Region,Amount\nNorth,1\n" + a.encode())["document"]["doc_id"]
    make_ready(api.registry, v1)
    v2 = post(client, a, "sales.csv", b"Region,Amount\nNorth,1\nSouth,2\n" + a.encode())
    assert v2["replaces"] is True and v2["existing"] is False
    r = client.post(f"/v1/documents/{v1}/reindex", headers=h)
    assert r.status_code == 409 and r.json()["detail"] == api.NEWER_VERSION
    other = post(client, a, "regions.csv", b"Region\nEast\n" + a.encode())  # another name: a new document
    assert other["replaces"] is False


def test_with_versions_off_an_older_document_of_the_name_can_be_reindexed(api, monkeypatch):
    monkeypatch.setattr(api.settings, "replace_same_name", False)
    a = new_user_id()
    client, h = TestClient(api.app), KEYS.headers(a)
    v1 = post(client, a, "notes.txt", b"one " + a.encode())["document"]["doc_id"]
    make_ready(api.registry, v1)
    assert post(client, a, "notes.txt", b"two " + a.encode())["replaces"] is False  # both stay
    assert client.post(f"/v1/documents/{v1}/reindex", headers=h).status_code == 202


def test_the_same_content_again_is_the_document_already_there(api):
    """Ids come from the content: the same bytes again, under any name, answer with the document there, marked
    existing so the UI can say so instead of showing the new name's status."""
    a = new_user_id()
    client = TestClient(api.app)
    data = b"Region,Amount\nNorth,1\n" + a.encode()
    first = post(client, a, "sales.csv", data)
    assert first["existing"] is False and first["replaces"] is False and first["document"]["status"] == "queued"
    again = post(client, a, "sales-copy.csv", data)
    assert again["existing"] is True and again["name"] == "sales-copy.csv"
    assert again["document"]["doc_id"] == first["document"]["doc_id"] and again["document"]["name"] == "sales.csv"
    assert len(api.registry.all(a)) == 1


# --- threads of a cancelled run, claims during a slow clean-up, a restarted worker --------------------


def test_shutdown_keeps_the_claim_until_background_writes_finish(tmp_path, monkeypatch):
    import threading

    s, reg, pipe = setup(tmp_path)
    queue(s, reg, DOC)
    doc = reg.claim_next()
    started, finish = threading.Event(), threading.Event()
    release = reg.release

    def checked_release(doc_id, run):
        assert finish.is_set(), "job released while its thread can still write"
        return release(doc_id, run)

    def write():
        started.set()
        assert finish.wait(5)

    async def process(_doc):
        await pipe.in_thread(write)

    monkeypatch.setattr(pipe, "process", process)
    monkeypatch.setattr(reg, "release", checked_release)

    async def scenario():
        stop = asyncio.Event()
        job = asyncio.create_task(process_until_stopped(pipe, reg, doc, stop, renew_s=0.01))
        await asyncio.to_thread(started.wait, 5)
        stop.set()
        try:
            # Let the shutdown path run while its writer is deliberately blocked.
            for _ in range(10):
                await asyncio.sleep(0)
            assert not job.done(), "shutdown finished before its writer"
            assert reg.claim_next() is None
        finally:
            finish.set()
            await pipe.settle()
        assert await job is None
        assert reg.claim_next().run_id != doc.run_id

    asyncio.run(scenario())


def test_cancelling_a_queued_thread_cannot_make_settle_wait_forever(tmp_path):
    import concurrent.futures
    import threading

    _, _, pipe = setup(tmp_path)
    release_pool = threading.Event()
    pool_busy = threading.Event()
    wrote = threading.Event()

    def occupy_pool():
        pool_busy.set()
        assert release_pool.wait(5)

    async def scenario():
        asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        occupying = asyncio.create_task(asyncio.to_thread(occupy_pool))
        while not pool_busy.is_set():
            await asyncio.sleep(0)
        queued = asyncio.create_task(pipe.in_thread(wrote.set))
        for _ in range(5):
            await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release_pool.set()
        await occupying
        try:
            await asyncio.wait_for(pipe.settle(), 1)
            assert wrote.is_set()
        finally:
            # Also let the executor exit when testing the old, broken implementation.
            for finished in list(pipe._threads):
                finished.set()

    try:
        asyncio.run(scenario())
    finally:
        release_pool.set()


def test_workers_serialize_versions_but_can_process_other_documents(tmp_path):
    reg = Registry(tmp_path / "r.db")
    v1, v2, other = "1" * 32, "2" * 32, "3" * 32
    for doc_id, name in ((v1, "report.txt"), (v2, "report.txt"), (other, "other.txt")):
        reg.add(doc_id, A, name, DocType.TEXT, ".txt", 1)
    first = reg.claim_next()
    assert first.doc_id == v1
    second = reg.claim_next()
    assert second.doc_id == other, "another worker must not overtake the earlier version"
    assert reg.claim_next() is None
    assert reg.complete(v1, first.run_id, 1, 0, [], [], supersede=True)
    assert reg.claim_next().doc_id == v2


def test_placement_waiters_keep_the_same_lock():
    from ingest import api

    async def scenario():
        entered = asyncio.Event()
        leave = asyncio.Event()
        active = 0

        async def waiter():
            nonlocal active
            async with api._one_placement(DOC):
                active += 1
                try:
                    entered.set()
                    await leave.wait()
                finally:
                    active -= 1

        first = api._one_placement(DOC)
        await first.__aenter__()
        waiting = asyncio.create_task(waiter())
        await asyncio.sleep(0)  # waiter has registered before the holder leaves
        await first.__aexit__(None, None, None)
        third = asyncio.create_task(waiter())
        try:
            await entered.wait()
            await asyncio.sleep(0)
            assert active == 1, "two uploads entered the same placement concurrently"
        finally:
            leave.set()
            await asyncio.gather(waiting, third)
        assert DOC not in api._placing

    asyncio.run(scenario())


def test_a_cancelled_runs_thread_cannot_leave_a_file_behind(tmp_path, monkeypatch):
    """The run is cancelled while write_tables runs in a thread; the clean-up waits for the thread, so the
    tables file it writes afterwards is removed too."""
    import time

    import ingest.pipeline as pipeline_module

    def slow_write_tables(path, sources):
        time.sleep(0.3)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("duckdb file of a deleted document")
        return []

    monkeypatch.setattr(pipeline_module, "write_tables", slow_write_tables)
    monkeypatch.setattr(Pipeline, "table_sources", lambda self, parsed: ["a sheet"])
    s, reg, pipe = setup(tmp_path)
    queue(s, reg, DOC)
    doc = reg.claim_next()

    async def scenario():
        asyncio.get_running_loop().call_later(0.05, reg.mark_deleting, DOC, A)  # the user deletes it
        with pytest.raises(RunLost):
            await process_until_stopped(pipe, reg, doc, asyncio.Event(), renew_s=0.02)
        assert await pipe.abandon(doc) == "deleted"
        await asyncio.sleep(0.4)

    asyncio.run(scenario())
    assert reg.get_for_worker(DOC) is None and not paths.table_path(s, A, DOC).exists()
    assert not paths.source_path(s, A, DOC, ".txt").exists()


def test_a_clean_up_that_lost_its_claim_removes_no_files(tmp_path, clock):
    """Dropping the points took longer than the lease; meanwhile the row was claimed again, finished, and the
    same file uploaded anew. The first clean-up must not remove the new upload's file."""
    s, reg, _ = setup(tmp_path, clock=clock)
    queue(s, reg, DOC)
    make_ready(reg, DOC)
    doc, claim = reg.mark_deleting(DOC, A)

    async def slow_points(owner, doc_id):
        clock.advance(seconds=reg.lease_s + 1)  # the lease runs out while Qdrant answers
        other = reg.claim_deleting(DOC)  # the sweep, or an upload of the same file, takes over
        assert other and reg.finish_deleting(DOC, other)
        queue(s, reg, DOC, name="again.txt")  # the new upload: its file and its row

    assert not asyncio.run(delete_document(s, reg, slow_points, doc, claim))
    assert paths.source_path(s, A, DOC, ".txt").exists() and reg.get(DOC, A).name == "again.txt"


def test_a_restarted_worker_takes_back_its_own_job_at_once(tmp_path, clock):
    first = Registry(tmp_path / "r.db", clock=clock, holder="worker-1")
    first.add(DOC, A, "big.pdf", DocType.PDF, ".pdf", 1)
    first.add("e" * 32, A, "gone.pdf", DocType.PDF, ".pdf", 1)
    assert first.claim_next().doc_id == DOC  # the process then dies; its lease runs for another 90 s
    run = first.transition("e" * 32, DocStatus.PROCESSING, sources=(DocStatus.QUEUED,), run_id="9" * 32,
                           lease_until=first._lease(), claimed_by="worker-1").run_id
    first.mark_deleting("e" * 32, A)  # deleted while the dead process held it
    other = Registry(tmp_path / "r.db", clock=clock, holder="worker-2")
    assert other.recover(2, mine=True) == 0 and other.deleting(mine=True) == []  # not its jobs
    again = Registry(tmp_path / "r.db", clock=clock, holder="worker-1")
    assert again.recover(2, mine=True) == 1 and again.get(DOC, A).status == DocStatus.QUEUED
    assert [d.doc_id for d in again.deleting(mine=True)] == ["e" * 32] and again.deleting() == []
    assert again.claim_deleting("e" * 32, mine=True) not in (None, run)


def test_the_sweep_runs_on_a_timer(tmp_path, clock):
    indexer = FakeIndexer()
    s, reg, pipe = setup(tmp_path, indexer, clock)
    for doc_id in ("1" * 32, "2" * 32):
        queue(s, reg, doc_id)
        make_ready(reg, doc_id)

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(sweeper(pipe, reg, 2, 0.02, stop))
        reg.mark_deleting("1" * 32, A)
        clock.advance(seconds=reg.lease_s + 1)  # its clean-up died
        await asyncio.sleep(0.1)
        first = reg.get_for_worker("1" * 32)
        reg.mark_deleting("2" * 32, A)
        clock.advance(seconds=reg.lease_s + 1)
        await asyncio.sleep(0.1)
        stop.set()
        await task
        return first

    assert asyncio.run(scenario()) is None and reg.get_for_worker("2" * 32) is None


def test_starting_the_background_sweeper_does_not_reclaim_a_fresh_job(tmp_path, clock):
    s, reg, pipe = setup(tmp_path, clock=clock)
    reg.holder = "worker-startup"
    queue(s, reg, DOC)
    reg.recover(2, mine=True)  # synchronous startup recovery, before accepting work
    doc = reg.claim_next()  # the first job is claimed before the background task gets its turn

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(sweeper(pipe, reg, 2, 0.01, stop))
        await asyncio.sleep(0.03)
        stop.set()
        await task

    asyncio.run(scenario())
    current = reg.get_for_worker(DOC)
    assert current.status == DocStatus.PROCESSING and current.run_id == doc.run_id


def test_an_upload_that_meets_a_deletion_keeps_no_file(api, monkeypatch):
    import hashlib

    a = new_user_id()
    client = TestClient(api.app)
    data = b"placed, then refused " + a.encode()

    def still_deleting(*args, **kwargs):
        raise StillDeleting("x")

    monkeypatch.setattr(api.registry, "add", still_deleting)
    assert post(client, a, "late.txt", data)["error"] == api.STILL_DELETING
    doc_id = api.derive_doc_id(a, hashlib.sha256(data).hexdigest())
    assert not paths.source_path(api.settings, a, doc_id, ".txt").exists()


# --- the v2 to v3 upgrade ---------------------------------------------------------------------


def v2_registry(path):
    """A v2 registry as the services wrote it (WAL), with an unowned row from before accounts and a gap in the
    rowids."""
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(V2_SCHEMA)
        for i, (owner, status) in enumerate([(A, "ready"), (A, "processing"), (None, "ready"), (A, "ready"),
                                             (A, "queued")]):
            db.execute("INSERT INTO documents (doc_id, owner_id, name, doc_type, ext, size, status, created_at,"
                       " updated_at) VALUES (?, ?, ?, 'text', '.txt', 1, ?, ?, 't')",
                       (f"{i}" * 32, owner, f"f{i}.txt", status, f"2026-01-0{i + 1}"))
        db.execute("DELETE FROM documents WHERE doc_id = ?", ("3" * 32,))  # a gap in the rowids


def rows(path):
    with sqlite3.connect(path) as db:
        return (db.execute("PRAGMA user_version").fetchone()[0],
                db.execute("SELECT rowid, doc_id, owner_id, status FROM documents ORDER BY rowid").fetchall(),
                sorted(r[0] for r in db.execute("SELECT name FROM sqlite_master")))


def test_the_upgrade_keeps_an_unowned_row_and_the_rowid_gap(tmp_path):
    path = tmp_path / "registry.db"
    v2_registry(path)
    _, before, _ = rows(path)
    reg = Registry(path)
    version, after, _ = rows(path)
    assert version == SCHEMA_VERSION and [r[:3] for r in after] == [r[:3] for r in before]
    assert [r[0] for r in after] == [1, 2, 3, 5]
    assert reg.legacy_count() == 1 and reg.get_for_worker("2" * 32).owner_id is None
    reg.recover(2)
    claimed = [reg.claim_next(), reg.claim_next(), reg.claim_next()]
    assert {c.doc_id for c in claimed if c} == {"1" * 32, "4" * 32}  # the unowned row is never claimed


def test_an_upgrade_that_fails_halfway_leaves_the_registry_as_it_was(tmp_path, monkeypatch):
    import ingest.registry as registry_module

    path = tmp_path / "registry.db"
    v2_registry(path)
    before = rows(path)
    broken = [*registry_module._V2_TO_V3]
    broken.insert(broken.index("DROP TABLE documents") + 1, "SELECT no_such_column FROM documents_v3")
    monkeypatch.setattr(registry_module, "_V2_TO_V3", broken)
    with pytest.raises(sqlite3.OperationalError):
        Registry(path)
    assert rows(path) == before  # version 2, every row, no documents_v3 left behind
    monkeypatch.undo()
    Registry(path)
    assert rows(path)[0] == SCHEMA_VERSION and len(rows(path)[1]) == 4


def test_services_starting_together_upgrade_once(tmp_path):
    """ingest-api and ingest-worker open a v2 registry at the same moment: one upgrades, the others wait and
    find it done."""
    import threading

    path = tmp_path / "registry.db"
    v2_registry(path)
    start, errors = threading.Barrier(3), []

    def open_registry():
        start.wait()
        try:
            Registry(path)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=open_registry) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    version, after, names = rows(path)
    assert errors == [] and version == SCHEMA_VERSION and len(after) == 4 and "documents_v3" not in names


# --- index versions (schema v4) -----------------------------------------------------------------


def test_requeue_stale_rebuilds_only_ready_documents_an_older_index_version_built(tmp_path, clock):
    reg, gone = Registry(tmp_path / "registry.db", clock=clock), new_user_id()
    for doc_id, owner in [("1" * 32, A), ("2" * 32, A), ("3" * 32, A), ("4" * 32, A), ("5" * 32, gone)]:
        reg.add(doc_id, owner, f"{doc_id[0]}.txt", DocType.TEXT, ".txt", 1)
    for doc_id in ("1" * 32, "5" * 32):
        make_ready(reg, doc_id, n_chunks=4)  # complete's default: version 1, from before versions were kept
    assert reg.complete("2" * 32, run_of(reg, "2" * 32), 4, 0, [], [], index_version=2) is not None
    reg.fail("3" * 32, run_of(reg, "3" * 32), "parse error")
    with sqlite3.connect(tmp_path / "registry.db") as db:  # the account of the fifth is deleted
        db.execute("INSERT INTO deleted_users VALUES (?, 't')", (gone,))

    assert reg.requeue_stale(2) == 1
    status = {d: reg.get_for_worker(d * 32) for d in "12345"}
    assert status["1"].status == DocStatus.QUEUED and status["1"].n_chunks == 4  # old chunks stay searchable
    assert status["2"].status == DocStatus.READY and status["3"].status == DocStatus.FAILED
    assert status["4"].status == DocStatus.QUEUED and status["5"].status == DocStatus.READY
    assert reg.requeue_stale(2) == 0  # queued already: nothing twice
    assert reg.complete("1" * 32, run_of(reg, "1" * 32), 6, 0, [], [], index_version=2) is not None
    assert reg.requeue_stale(2) == 0 and reg.requeue_stale(3) == 2  # the next fix rebuilds both again


def test_a_v3_registry_gets_the_index_version_and_its_ready_rows_count_as_version_1(tmp_path, clock):
    path = tmp_path / "registry.db"
    reg = Registry(path, clock=clock)
    reg.add(DOC, A, "a.txt", DocType.TEXT, ".txt", 1)
    make_ready(reg, DOC)
    with sqlite3.connect(path) as db:  # back to v3 as the services wrote it
        db.execute("ALTER TABLE documents DROP COLUMN index_version")
        db.execute("PRAGMA user_version = 3")
    reg = Registry(path, clock=clock)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 4
        assert db.execute("SELECT index_version FROM documents").fetchall() == [(1,)]
    assert reg.get(DOC, A).status == DocStatus.READY and reg.requeue_stale(2) == 1


def test_a_finished_job_records_the_current_index_version_and_is_not_rebuilt_again(tmp_path):
    from ingest.pipeline import INDEX_VERSION

    s, reg, pipe = setup(tmp_path, FakeIndexer())
    queue(s, reg, DOC, name="a.txt")
    assert asyncio.run(run_job(pipe, reg, Recorder(), reg.claim_next(), asyncio.Event()))
    with sqlite3.connect(s.registry_path) as db:
        assert db.execute("SELECT index_version FROM documents").fetchall() == [(INDEX_VERSION,)]
    assert INDEX_VERSION >= 2 and reg.requeue_stale(INDEX_VERSION) == 0
