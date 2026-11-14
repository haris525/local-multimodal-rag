"""Upgrading an install from before accounts: registry schema v0, the old file layout, points without user_id."""

import asyncio
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from conftest import KEYS
from rag_common.index import COLLECTION, TEXT_VECTOR, USER_FIELD, point_id
from rag_common.testing import new_user_id

from ingest import paths
from ingest.config import Settings
from ingest.registry import SCHEMA_VERSION, Registry

READY, QUEUED, ORPHAN, STRAY = "1" * 32, "2" * 32, "3" * 32, "4" * 32
ASSET = "0123456789abcdef.png"
V0_SCHEMA = """
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY, name TEXT NOT NULL, doc_type TEXT NOT NULL, ext TEXT NOT NULL, size INTEGER NOT NULL,
    status TEXT NOT NULL, progress TEXT NOT NULL DEFAULT '', error TEXT, warnings TEXT NOT NULL DEFAULT '[]',
    n_chunks INTEGER NOT NULL DEFAULT 0, n_images INTEGER NOT NULL DEFAULT 0, tables TEXT NOT NULL DEFAULT '[]',
    attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX documents_status ON documents (status, created_at);
"""


def legacy_install(data):
    """A data folder written before accounts: registry v0 and files keyed by document id only."""
    data.mkdir(parents=True)
    with sqlite3.connect(data / "registry.db") as db:
        db.executescript(V0_SCHEMA)
        db.execute("INSERT INTO documents (doc_id, name, doc_type, ext, size, status, n_chunks, created_at, updated_at)"
                   " VALUES (?, 'report.txt', 'text', '.txt', 5, 'ready', 2, '2026-01-01', '2026-01-01')", (READY,))
        db.execute("INSERT INTO documents (doc_id, name, doc_type, ext, size, status, created_at, updated_at)"
                   " VALUES (?, 'later.txt', 'text', '.txt', 5, 'queued', '2026-01-02', '2026-01-02')", (QUEUED,))
    for doc in (READY, QUEUED):
        (data / "files" / doc).mkdir(parents=True)
        (data / "files" / doc / "source.txt").write_text("hello")
    (data / "assets" / READY).mkdir(parents=True)
    (data / "assets" / READY / ASSET).write_bytes(b"\x89PNG figure")
    (data / "tables").mkdir()
    (data / "tables" / f"{READY}.duckdb").write_bytes(b"duckdb")
    (data / "chunks").mkdir()
    (data / "chunks" / f"{READY}.jsonl").write_text('{"id": "x", "chunk_index": 0, "text": "hello"}\n')
    (data / "chunks" / f"{ORPHAN}.jsonl").write_text("")  # no registry row: reported, never touched


async def legacy_points() -> AsyncQdrantClient:
    q = AsyncQdrantClient(location=":memory:")
    await q.create_collection(COLLECTION, vectors_config={TEXT_VECTOR: qm.VectorParams(size=2, distance=qm.Distance.COSINE)})
    await q.upsert(COLLECTION, [qm.PointStruct(id=point_id(doc, i), vector={TEXT_VECTOR: [1.0, float(i)]},
                                               payload={"doc_id": doc, "chunk_index": i})
                                for doc in (READY, STRAY) for i in range(2)], wait=True)
    return q


def payloads(q: AsyncQdrantClient) -> dict[str, dict]:
    points, _ = asyncio.run(q.scroll(COLLECTION, limit=100, with_payload=True))
    return {str(p.id): p.payload for p in points}


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    import ingest.api as api

    legacy_install(tmp_path / "data")
    settings = Settings(data_dir=tmp_path / "data", audit_url=None)
    monkeypatch.setattr(api, "settings", settings)
    monkeypatch.setattr(api, "registry", Registry(settings.registry_path))  # upgrades v0 to the current schema here
    monkeypatch.setattr(api, "qdrant", asyncio.run(legacy_points()))
    return api


def migrate(client, owner, **flags):
    body = {"owner_id": owner, "dry_run": False, "force": False} | flags
    return client.post("/internal/migrate-legacy", json=body, headers=KEYS.headers(owner, role="system"))


def test_schema_upgrade_keeps_rows_and_hides_them_until_adopted(legacy):
    reg = legacy.registry
    with sqlite3.connect(legacy.settings.registry_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert "owner_id" in {r[1] for r in db.execute("PRAGMA table_info(documents)")}
    assert reg.legacy_count() == 2 and reg.get_for_worker(READY).n_chunks == 2
    assert reg.all(new_user_id()) == [] and reg.claim_next() is None
    Registry(legacy.settings.registry_path)  # a second start changes nothing
    assert reg.legacy_count() == 2


def test_adoption_moves_everything_to_the_owner_once(legacy):
    client = TestClient(legacy.app)
    owner, other = new_user_id(), new_user_id()
    before = payloads(legacy.qdrant)

    dry = migrate(client, owner, dry_run=True).json()
    assert (dry["documents"], dry["files_moved"], dry["points_updated"]) == (2, 5, 2)
    assert legacy.registry.legacy_count() == 2 and payloads(legacy.qdrant) == before  # nothing changed

    first = migrate(client, owner).json()
    assert (first["documents"], first["files_moved"], first["points_updated"]) == (2, 5, 2)
    assert first["orphans"] == [ORPHAN]
    second = migrate(client, owner).json()
    assert (second["documents"], second["files_moved"], second["points_updated"]) == (0, 0, 0)

    s = legacy.settings
    assert sorted(d.doc_id for d in legacy.registry.all(owner)) == [READY, QUEUED]  # ids unchanged
    assert paths.source_path(s, owner, READY, ".txt").read_text() == "hello"
    assert paths.table_path(s, owner, READY).exists() and paths.chunk_path(s, owner, READY).exists()
    assert not (s.files_dir / READY).exists() and (s.chunks_dir / f"{ORPHAN}.jsonl").exists()
    after = payloads(legacy.qdrant)
    assert after.keys() == before.keys()  # point ids unchanged
    assert {after[point_id(READY, i)][USER_FIELD] for i in range(2)} == {owner}
    assert all(USER_FIELD not in after[point_id(STRAY, i)] for i in range(2))  # not the owner's document

    r = client.get(f"/v1/documents/{READY}/assets/{ASSET}", headers=KEYS.headers(owner))
    assert r.status_code == 200 and r.content == b"\x89PNG figure"  # served from the new place
    assert client.get(f"/v1/documents/{READY}", headers=KEYS.headers(other)).status_code == 404
    assert legacy.registry.claim_next().doc_id == QUEUED  # now owned, so the worker takes it


def test_adoption_waits_for_the_worker_to_stop(legacy):
    client = TestClient(legacy.app)
    owner = new_user_id()
    (legacy.settings.data_dir / "worker.heartbeat").write_text(str(time.time()))
    r = migrate(client, owner)
    assert r.status_code == 409 and "stop ingest-worker" in r.json()["detail"]
    assert legacy.registry.legacy_count() == 2
    assert migrate(client, owner, force=True).json()["documents"] == 2
    assert migrate(client, owner).status_code == 200  # no legacy rows left: a running worker is fine


def test_migration_is_for_auth_system_calls_only(legacy):
    client = TestClient(legacy.app)
    owner = new_user_id()
    body = {"owner_id": owner}
    assert client.post("/internal/migrate-legacy", json=body, headers=KEYS.headers(owner)).status_code == 403
    assert client.post("/internal/migrate-legacy", json=body,
                       headers=KEYS.headers(owner, role="admin")).status_code == 403
    assert client.post("/internal/migrate-legacy", json={"owner_id": "legacy"},
                       headers=KEYS.headers(owner, role="system")).status_code == 422
    assert legacy.registry.legacy_count() == 2
