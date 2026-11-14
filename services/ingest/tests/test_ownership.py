"""Per-user documents: registry scoping, owner-salted ids, per-owner files and supersede, and the API's 404s."""

import asyncio
import sqlite3

import pytest
from fastapi.testclient import TestClient

from conftest import KEYS, make_ready, run_of
from rag_common.contracts import DocStatus, DocType
from rag_common.index import USER_FIELD
from rag_common.testing import new_user_id

from ingest import paths
from ingest.config import Settings
from ingest.elements import Chunk, Element, ParsedDocument
from ingest.pipeline import Pipeline
from ingest.registry import SCHEMA_VERSION, Registry

A, B = new_user_id(), new_user_id()
DOC = "a" * 32


# --- registry -------------------------------------------------------------------------------


def test_every_registry_method_treats_another_owners_document_as_missing(tmp_path):
    reg = Registry(tmp_path / "r.db")
    reg.add(DOC, A, "report.pdf", DocType.PDF, ".pdf", 1)
    make_ready(reg, DOC, 3)
    assert reg.get(DOC, B) is None and reg.get_deleting(DOC, B) is None
    assert reg.many([DOC], B) == [] and reg.all(B) == []
    assert not reg.requeue(DOC, B) and reg.mark_deleting(DOC, B) is None
    assert reg.purge_owner(B) == []
    assert reg.get(DOC, A).status == DocStatus.READY and reg.get(DOC, A).owner_id == A  # untouched


def test_an_id_held_by_another_owner_is_never_returned(tmp_path):
    reg = Registry(tmp_path / "r.db")
    reg.add(DOC, A, "a.txt", DocType.TEXT, ".txt", 1)
    with pytest.raises(RuntimeError):
        reg.add(DOC, B, "a.txt", DocType.TEXT, ".txt", 1)


def test_registry_refuses_a_missing_or_malformed_owner(tmp_path):
    reg = Registry(tmp_path / "r.db")
    for bad in (None, "", "legacy", "../x"):
        with pytest.raises(ValueError):
            reg.all(bad)
        with pytest.raises(ValueError):
            reg.add(DOC, bad, "a.txt", DocType.TEXT, ".txt", 1)


def test_same_name_supersede_is_scoped_to_the_owner(tmp_path):
    reg = Registry(tmp_path / "r.db")
    a1, b1, a2 = "1" * 32, "2" * 32, "3" * 32
    reg.add(a1, A, "report.pdf", DocType.PDF, ".pdf", 1)
    reg.add(b1, B, "report.pdf", DocType.PDF, ".pdf", 1)  # B's file of the same name
    reg.add(a2, A, "report.pdf", DocType.PDF, ".pdf", 1)
    for doc_id in (a1, b1):
        assert make_ready(reg, doc_id, supersede=True) == []
    assert [d.doc_id for d in make_ready(reg, a2, supersede=True)] == [a1]
    assert reg.get(b1, B) is not None


def test_the_worker_never_claims_a_document_without_owner(tmp_path):
    reg = Registry(tmp_path / "r.db")
    with sqlite3.connect(tmp_path / "r.db") as db:  # a row from before accounts existed
        db.execute("INSERT INTO documents (doc_id, name, doc_type, ext, size, status, created_at, updated_at)"
                   " VALUES (?, 'old.txt', 'text', '.txt', 1, 'queued', '2020', '2020')", ("f" * 32,))
    assert reg.claim_next() is None
    reg.add(DOC, A, "new.txt", DocType.TEXT, ".txt", 1)
    assert reg.claim_next().doc_id == DOC
    assert reg.get_for_worker("f" * 32).owner_id is None and reg.all(A)[0].doc_id == DOC


def test_purge_owner_deletes_only_that_owners_rows(tmp_path):
    reg = Registry(tmp_path / "r.db")
    reg.add("1" * 32, A, "a.txt", DocType.TEXT, ".txt", 1)
    reg.add("2" * 32, A, "b.txt", DocType.TEXT, ".txt", 1)
    reg.add("3" * 32, B, "a.txt", DocType.TEXT, ".txt", 1)
    assert sorted(reg.purge_owner(A)) == ["1" * 32, "2" * 32]
    assert reg.purge_owner(A) == []  # repeatable
    assert [d.doc_id for d in reg.all(B)] == ["3" * 32]


# --- ids, paths and payloads ----------------------------------------------------------------


def test_doc_id_is_salted_with_the_owner():
    from ingest.api import derive_doc_id

    sha = "0" * 64
    assert derive_doc_id(A, sha) != derive_doc_id(B, sha)
    assert derive_doc_id(A, sha) == derive_doc_id(A, sha) and len(derive_doc_id(A, sha)) == 32


def test_every_path_is_under_the_owner(tmp_path):
    s = Settings(data_dir=tmp_path, audit_url=None)
    assert paths.source_path(s, A, DOC, ".pdf") == tmp_path / "files" / A / DOC / "source.pdf"
    assert paths.asset_dir(s, A, DOC) == tmp_path / "assets" / A / DOC
    assert paths.table_path(s, A, DOC) == tmp_path / "tables" / A / f"{DOC}.duckdb"
    assert paths.chunk_path(s, A, DOC) == tmp_path / "chunks" / A / f"{DOC}.jsonl"
    for owner, doc in ((A, "../" + "a" * 29), ("..", DOC), (None, DOC), (A, "A" * 32)):
        with pytest.raises(ValueError):
            paths.chunk_path(s, owner, doc)


def test_remove_owner_leaves_other_owners_alone(tmp_path):
    s = Settings(data_dir=tmp_path, audit_url=None)
    for owner in (A, B):
        for path in (paths.source_path(s, owner, DOC, ".txt"), paths.table_path(s, owner, DOC),
                     paths.chunk_path(s, owner, DOC), paths.asset_dir(s, owner, DOC) / "x.png"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
    paths.remove_owner(s, A)
    assert not any(paths.owner_root(s, kind, A).exists() for kind in paths.KINDS)
    assert all(paths.owner_root(s, kind, B).exists() for kind in paths.KINDS)


def test_chunk_payloads_carry_the_owner():
    from rag_common.contracts import DocumentInfo

    doc = DocumentInfo(doc_id=DOC, name="a.txt", doc_type=DocType.TEXT, size=1, status=DocStatus.PROCESSING,
                       created_at="t", updated_at="t", owner_id=A)
    payload = Pipeline._payload(doc, 0, Chunk(kind="text", text="hello", search_text="hello", n_tokens=1))
    assert payload.model_dump(exclude_none=True)[USER_FIELD] == A


# --- API ------------------------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.events = []

    async def record(self, action, subject=None, actor=None, *, user_id=None, **detail):
        self.events.append((action, subject, user_id))

    async def aclose(self):
        pass


class FakeQdrant:
    """Records delete filters; ingest-api only deletes points."""

    def __init__(self):
        self.deleted = []

    async def delete(self, collection, points_selector, wait=True):
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


def upload(client, user: str, name: str, data: bytes) -> dict:
    r = client.post("/v1/documents", files=[("files", (name, data))], headers=KEYS.headers(user))
    assert r.status_code == 200, r.text
    return r.json()[0]["document"]


def test_another_users_documents_are_404_on_every_route(api):
    a, b = new_user_id(), new_user_id()
    client = TestClient(api.app)
    doc = upload(client, a, "notes.txt", b"private notes of A " + a.encode())
    asset = paths.asset_dir(api.settings, a, doc["doc_id"]) / ("0" * 16 + ".png")
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"\x89PNG")
    hb = KEYS.headers(b)
    d = doc["doc_id"]

    assert d not in {x["doc_id"] for x in client.get("/v1/documents", headers=hb).json()}
    assert client.get(f"/v1/documents/{d}", headers=hb).status_code == 404
    assert client.post("/v1/documents/status", json={"doc_ids": [d]}, headers=hb).json() == []
    assert client.delete(f"/v1/documents/{d}", headers=hb).status_code == 404
    assert client.post(f"/v1/documents/{d}/reindex", headers=hb).status_code == 404
    assert client.get(f"/v1/documents/{d}/chunks", headers=hb).status_code == 404
    assert client.get(f"/v1/documents/{d}/assets/{asset.name}", headers=hb).status_code == 404
    assert client.post("/v1/grep", json={"doc_ids": [d], "terms": ["notes"]}, headers=hb).status_code == 404
    mine = upload(client, b, "other.txt", b"B's own file " + b.encode())["doc_id"]
    assert client.post("/v1/grep", json={"doc_ids": [mine, d], "terms": ["x"]}, headers=hb).status_code == 404
    assert client.post("/v1/grep", json={"doc_ids": [mine], "terms": ["x"]}, headers=hb).status_code == 200

    ha = KEYS.headers(a)  # A still has everything
    assert client.get(f"/v1/documents/{d}", headers=ha).json()["owner_id"] == a
    r = client.get(f"/v1/documents/{d}/assets/{asset.name}", headers=ha)
    assert r.status_code == 200 and r.headers["cache-control"] == "private, max-age=300"
    assert client.get("/v1/documents", headers=ha).headers["cache-control"] == "no-store"
    assert api.qdrant.deleted == []  # B's delete never reached the index


def test_the_same_bytes_from_two_users_are_two_documents(api):
    a, b = new_user_id(), new_user_id()
    client = TestClient(api.app)
    first, again = upload(client, a, "same.txt", b"identical bytes"), upload(client, a, "same.txt", b"identical bytes")
    other = upload(client, b, "same.txt", b"identical bytes")
    assert first["doc_id"] == again["doc_id"] != other["doc_id"]
    assert paths.source_path(api.settings, a, first["doc_id"], ".txt").exists()
    assert paths.source_path(api.settings, b, other["doc_id"], ".txt").exists()
    assert [(e[0], e[2]) for e in api.audit.events] == [("document.uploaded", a)] * 2 + [("document.uploaded", b)]


def test_delete_scopes_the_index_delete_to_the_owner(api):
    a = new_user_id()
    client = TestClient(api.app)
    d = upload(client, a, "gone.txt", b"to be deleted " + a.encode())["doc_id"]
    make_ready(api.registry, d)
    assert client.delete(f"/v1/documents/{d}", headers=KEYS.headers(a)).status_code == 204
    (flt,) = api.qdrant.deleted
    assert {(c.key, getattr(c.match, "value", None)) for c in flt.must} == {(USER_FIELD, a), ("doc_id", d)}
    assert not paths.source_path(api.settings, a, d, ".txt").parent.exists()


def test_routes_need_an_identity_and_internal_routes_need_system(api):
    client = TestClient(api.app)
    assert client.get("/v1/documents").status_code == 401
    assert client.get("/v1/documents", headers={"x-identity": "v1.bogus.token"}).status_code == 401
    assert client.get("/internal/capabilities", headers=KEYS.headers(A)).status_code == 403
    assert client.get("/internal/capabilities", headers=KEYS.headers(A, role="system")).json() == {
        "service": "ingest", "multi_user": 1, "schema": SCHEMA_VERSION}
    assert client.delete(f"/internal/users/{A}", headers=KEYS.headers(A)).status_code == 403


def test_account_purge_removes_rows_points_and_files_and_can_be_repeated(api):
    a, b = new_user_id(), new_user_id()
    client = TestClient(api.app)
    for i in range(2):
        upload(client, a, f"a{i}.txt", f"A file {i} {a}".encode())
    kept = upload(client, b, "b.txt", f"B file {b}".encode())["doc_id"]
    system = KEYS.headers(new_user_id(), role="system")
    assert client.delete(f"/internal/users/{a}", headers=system).json() == {"documents": 2}
    assert client.delete(f"/internal/users/{a}", headers=system).json() == {"documents": 0}
    assert api.registry.all(a) == [] and [d.doc_id for d in api.registry.all(b)] == [kept]
    assert not any(paths.owner_root(api.settings, kind, a).exists() for kind in paths.KINDS)
    assert paths.source_path(api.settings, b, kept, ".txt").exists()
    assert all(len(f.must) == 1 and f.must[0].match.value == a for f in api.qdrant.deleted)


class FakeRunner:
    def run(self, fn, *args, timeout):
        return ParsedDocument(elements=[Element(kind="text", text="Hello world from the owner.", section="Intro")])


class FakeIndexer:
    def __init__(self, on_replace=None):
        self.on_replace = on_replace
        self.replaced, self.deleted = [], []

    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_images(self, images):
        return []

    async def replace(self, user_id, doc_id, payloads, text_vectors, image_vectors):
        self.replaced.append((user_id, doc_id, [p.user_id for p in payloads]))
        if self.on_replace:
            self.on_replace()

    async def delete(self, user_id, doc_id):
        self.deleted.append((user_id, doc_id))


def pipeline(tmp_path, counter, indexer):
    s = Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None)
    reg = Registry(s.registry_path)
    return s, reg, Pipeline(s, reg, indexer, None, counter, FakeRunner())


def queue(s, reg, doc_id, owner, name="report.txt"):
    reg.add(doc_id, owner, name, DocType.TEXT, ".txt", 1)
    source = paths.source_path(s, owner, doc_id, ".txt")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("Hello")


def test_processing_writes_under_the_owner_and_supersedes_only_the_owners_file(tmp_path, counter):
    indexer = FakeIndexer()
    s, reg, pipe = pipeline(tmp_path, counter, indexer)
    old_a, b_doc, new_a = "1" * 32, "2" * 32, "3" * 32
    for doc_id, owner in ((old_a, A), (b_doc, B), (new_a, A)):
        queue(s, reg, doc_id, owner)
    for doc_id in (old_a, b_doc):
        make_ready(reg, doc_id)
    doc = reg.claim_next()  # the oldest queued one: new_a
    replaced = asyncio.run(pipe.process(doc))
    assert [d.doc_id for d in replaced] == [old_a] and reg.get(old_a, A) is None  # hidden at once
    asyncio.run(pipe.remove_superseded(replaced))
    assert indexer.replaced == [(A, new_a, [A])] and indexer.deleted == [(A, old_a)]
    assert paths.chunk_path(s, A, new_a).exists() and not paths.source_path(s, A, old_a, ".txt").exists()
    assert reg.get(b_doc, B) is not None and paths.source_path(s, B, b_doc, ".txt").exists()  # B's file stays


def test_a_job_whose_owner_was_purged_cleans_up_after_itself(tmp_path, counter):
    from ingest.pipeline import RunLost

    holder = {}
    indexer = FakeIndexer(on_replace=lambda: holder["reg"].purge_owner(A))  # account deleted mid-job
    s, reg, pipe = pipeline(tmp_path, counter, indexer)
    holder["reg"] = reg
    queue(s, reg, DOC, A)
    doc = reg.claim_next()
    with pytest.raises(RunLost):
        asyncio.run(pipe.process(doc))
    assert asyncio.run(pipe.abandon(doc)) == "account deleted"
    assert indexer.deleted == [(A, DOC)]
    assert not paths.chunk_path(s, A, DOC).exists() and not paths.source_path(s, A, DOC, ".txt").exists()


def test_a_document_without_owner_is_never_processed(tmp_path, counter):
    from rag_common.contracts import DocumentInfo

    _, _, pipe = pipeline(tmp_path, counter, FakeIndexer())
    orphan = DocumentInfo(doc_id=DOC, name="a.txt", doc_type=DocType.TEXT, size=1, status=DocStatus.PROCESSING,
                          created_at="t", updated_at="t")
    with pytest.raises(ValueError, match="no owner"):
        asyncio.run(pipe.process(orphan))


def test_deletes_succeed_before_anything_was_ever_indexed():
    """No collection yet (the worker never ran): a document or account delete has nothing to remove."""
    import httpx
    from qdrant_client.http.exceptions import UnexpectedResponse

    from ingest.indexer import delete_points, delete_user_points

    class NoCollection:
        async def delete(self, collection, points_selector, wait=True):
            raise UnexpectedResponse(404, "Not Found", b'{"status": {"error": "Not found: Collection"}}', httpx.Headers())

    class Down:
        async def delete(self, collection, points_selector, wait=True):
            raise UnexpectedResponse(500, "Internal Server Error", b"", httpx.Headers())

    asyncio.run(delete_points(NoCollection(), A, DOC))
    asyncio.run(delete_user_points(NoCollection(), A))
    with pytest.raises(UnexpectedResponse):
        asyncio.run(delete_user_points(Down(), A))


# --- deleted accounts (tombstones) and jobs whose row disappeared ---------------------------


def test_a_deleted_owner_never_gets_a_document_again(tmp_path):
    from ingest.registry import OwnerDeleted

    reg = Registry(tmp_path / "r.db")
    reg.add(DOC, A, "a.txt", DocType.TEXT, ".txt", 1)
    assert reg.purge_owner(A) == [DOC] and reg.is_deleted(A) and not reg.is_deleted(B)
    with pytest.raises(OwnerDeleted):
        reg.add("1" * 32, A, "late.txt", DocType.TEXT, ".txt", 1)  # e.g. an upload authorized before the purge
    with pytest.raises(OwnerDeleted):
        reg.adopt_legacy(A)
    assert reg.all(A) == [] and reg.purge_owner(A) == []  # repeatable
    with sqlite3.connect(tmp_path / "r.db") as db:  # a row of the deleted owner, however it got there
        db.execute("INSERT INTO documents (doc_id, owner_id, name, doc_type, ext, size, status, created_at, updated_at)"
                   " VALUES (?, ?, 'x.txt', 'text', '.txt', 1, 'queued', '2020', '2020')", ("2" * 32, A))
    assert reg.claim_next() is None  # never claimed
    assert not reg.complete("2" * 32, run_of(reg, "2" * 32), 1, 0, [], [])  # nor completed, even when run
    reg.add(DOC, B, "b.txt", DocType.TEXT, ".txt", 1)
    assert reg.claim_next().owner_id == B  # other owners are unaffected


def test_a_v1_registry_gets_the_tombstone_table(tmp_path):
    path = tmp_path / "r.db"
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE documents (doc_id TEXT PRIMARY KEY, owner_id TEXT, name TEXT NOT NULL,
            doc_type TEXT NOT NULL, ext TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL,
            progress TEXT NOT NULL DEFAULT '', error TEXT, warnings TEXT NOT NULL DEFAULT '[]',
            n_chunks INTEGER NOT NULL DEFAULT 0, n_images INTEGER NOT NULL DEFAULT 0, tables TEXT NOT NULL DEFAULT '[]',
            attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            PRAGMA user_version = 1;""")
        db.execute("INSERT INTO documents (doc_id, owner_id, name, doc_type, ext, size, status, created_at, updated_at)"
                   " VALUES (?, ?, 'a.txt', 'text', '.txt', 1, 'ready', 't', 't')", (DOC, A))
    reg = Registry(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db.execute("SELECT COUNT(*) FROM deleted_users").fetchone()[0] == 0
    assert reg.get(DOC, A).name == "a.txt"
    reg.purge_owner(A)
    assert reg.is_deleted(A) and Registry(path).is_deleted(A)  # a second start keeps the tombstone


def test_an_upload_authorized_before_the_purge_is_refused_after_it(api):
    a = new_user_id()
    client = TestClient(api.app)
    early = KEYS.headers(a)  # signed just before the admin deleted the account; valid for 2 more minutes
    upload(client, a, "one.txt", b"first document " + a.encode())
    system = KEYS.headers(new_user_id(), role="system")
    assert client.delete(f"/internal/users/{a}", headers=system).json() == {"documents": 1}
    r = client.post("/v1/documents", files=[("files", ("two.txt", b"after the purge " + a.encode()))], headers=early)
    assert r.status_code == 403 and r.json() == {"detail": "this account has been deleted"}
    assert api.registry.all(a) == []
    assert not any(paths.owner_root(api.settings, kind, a).exists() for kind in paths.KINDS)
    owner = {"owner_id": a, "dry_run": True, "force": True}
    assert client.post("/internal/migrate-legacy", json=owner, headers=system).status_code == 409


def test_an_upload_racing_the_purge_keeps_nothing(api, monkeypatch):
    """The purge lands between the upload's tombstone check and its registry insert."""
    a = new_user_id()
    client = TestClient(api.app)
    real_add = api.registry.add

    def purge_first(*args, **kwargs):
        api.registry.purge_owner(a)
        return real_add(*args, **kwargs)

    monkeypatch.setattr(api.registry, "add", purge_first)
    r = client.post("/v1/documents", files=[("files", ("x.txt", b"racing " + a.encode()))], headers=KEYS.headers(a))
    assert r.status_code == 403
    assert not paths.owner_root(api.settings, "files", a).exists() and api.registry.all(a) == []


def test_a_purged_owners_job_leaves_no_folders(tmp_path, counter):
    holder = {}

    def purge_like_ingest_api():  # DELETE /internal/users/{A} while the worker indexes a document of A
        holder["reg"].purge_owner(A)
        paths.remove_owner(holder["s"], A)

    indexer = FakeIndexer(on_replace=purge_like_ingest_api)
    s, reg, pipe = pipeline(tmp_path, counter, indexer)
    holder.update(reg=reg, s=s)
    queue(s, reg, DOC, A)
    from ingest.worker import run_job

    assert asyncio.run(run_job(pipe, reg, Recorder(), reg.claim_next(), asyncio.Event()))
    assert indexer.deleted == [(A, DOC)]  # the points it wrote after the purge
    assert not any(paths.owner_root(s, kind, A).exists() for kind in paths.KINDS)


class FailingIndexer(FakeIndexer):
    async def replace(self, user_id, doc_id, payloads, text_vectors, image_vectors):
        await super().replace(user_id, doc_id, payloads, text_vectors, image_vectors)
        raise RuntimeError("qdrant went away")


def _leftovers(s, owner, doc_id):
    """What a job may have written before it failed: a table file and a figure."""
    paths.table_path(s, owner, doc_id).parent.mkdir(parents=True, exist_ok=True)
    paths.table_path(s, owner, doc_id).write_bytes(b"duckdb")
    paths.asset_dir(s, owner, doc_id).mkdir(parents=True, exist_ok=True)
    (paths.asset_dir(s, owner, doc_id) / ("0" * 16 + ".png")).write_bytes(b"png")
    paths.owner_root(s, "chunks", owner).mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("deleted", ["document", "account"])
def test_a_failed_job_whose_row_is_gone_cleans_up(tmp_path, counter, deleted):
    from ingest.worker import run_job

    holder = {}

    def delete():
        _leftovers(holder["s"], A, DOC)
        if deleted == "document":
            holder["reg"].mark_deleting(DOC, A)  # the user deletes it while it runs
        else:
            holder["reg"].purge_owner(A)

    indexer = FailingIndexer(on_replace=delete)
    s, reg, pipe = pipeline(tmp_path, counter, indexer)
    holder.update(reg=reg, s=s)
    queue(s, reg, DOC, A)
    queue(s, reg, "2" * 32, B)  # a waiting document of another owner stays untouched
    audit = Recorder()
    assert asyncio.run(run_job(pipe, reg, audit, reg.claim_next(), asyncio.Event()))
    assert indexer.deleted == [(A, DOC)]
    assert not paths.table_path(s, A, DOC).exists() and not paths.asset_dir(s, A, DOC).exists()
    assert not paths.source_path(s, A, DOC, ".txt").exists()
    assert not any(paths.owner_root(s, kind, A).exists() for kind in paths.WORKER_KINDS)
    assert audit.events == []  # nothing to report for a document nobody has any more
    assert reg.get("2" * 32, B) and paths.source_path(s, B, "2" * 32, ".txt").exists()


def test_a_failed_job_that_is_still_wanted_is_marked_failed_and_kept(tmp_path, counter):
    from ingest.worker import run_job

    indexer = FailingIndexer()
    s, reg, pipe = pipeline(tmp_path, counter, indexer)
    queue(s, reg, DOC, A)
    audit = Recorder()
    asyncio.run(run_job(pipe, reg, audit, reg.claim_next(), asyncio.Event()))
    assert reg.get(DOC, A).status == DocStatus.FAILED and indexer.deleted == []
    assert paths.source_path(s, A, DOC, ".txt").exists()  # kept for a retry
    assert [e[0] for e in audit.events] == ["document.failed"]
    assert asyncio.run(pipe.abandon(reg.get(DOC, A))) == "held elsewhere"  # a wanted document is left alone
    assert paths.source_path(s, A, DOC, ".txt").exists()


# --- deleted text does not stay readable in the registry file -------------------------------


def _on_disk(path) -> bytes:
    return b"".join(p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists())


def test_registry_deletes_zero_what_they_free(tmp_path):
    path = tmp_path / "r.db"
    reg = Registry(path)
    reg.add(DOC, A, "salary-review-of-john.txt", DocType.TEXT, ".txt", 1)
    reg.add("1" * 32, A, "keep-me.txt", DocType.TEXT, ".txt", 1)
    _, claim = reg.mark_deleting(DOC, A)
    assert reg.finish_deleting(DOC, claim)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert b"salary-review-of-john" not in _on_disk(path) and b"keep-me" in _on_disk(path)


def test_an_account_purge_zeroes_the_rows_and_truncates_the_wal(tmp_path, monkeypatch, caplog):
    import logging

    from ingest import registry as registry_module

    path = tmp_path / "r.db"
    reg = Registry(path)
    reg.add(DOC, A, "private-letter-of-A.pdf", DocType.PDF, ".pdf", 1)
    reg.purge_owner(A)  # no manual checkpoint: purge_owner truncates the WAL itself
    assert b"private-letter-of-A" not in _on_disk(path)
    monkeypatch.setattr(registry_module, "truncate_wal", lambda db, log, **f: log.warning("blocked") or False)
    with caplog.at_level(logging.WARNING, logger="ingest.registry"):
        reg.purge_owner(B)
    assert any(r.name == "ingest.registry" for r in caplog.records)
