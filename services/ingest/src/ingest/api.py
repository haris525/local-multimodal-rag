"""ingest-api: uploads, document registry, figure assets, chunk listings and grep, per user.

Every route acts for the user in X-Identity (from the gateway, or re-signed by rag). Documents are looked up
with their owner, so another user's document is a 404, like a missing one; the same bytes uploaded by two
users are two documents (the id is salted with the owner). /internal/* is for auth's system calls
(capabilities, legacy adoption, account purge).
"""

import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Depends, File, HTTPException, Path as PathParam, Query, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from qdrant_client import models as qm

from rag_common.audit import AuditClient
from rag_common.contracts import DocIdPath, DocIdsRequest, DocStatus, DocumentInfo, GenerationParameter, GrepHit, GrepRequest, MigrateLegacy
from rag_common.identity import USER_ID, Identity, Verifier, require_system, require_user
from rag_common.index import COLLECTION, USER_FIELD, ensure_user_index, qdrant_client
from rag_common.logging import log_fields
from rag_common.service import create_app

from ingest import chunkstore, paths
from ingest.config import get_settings
from ingest.indexer import delete_points, delete_user_points
from ingest.llmcache import LlmCache
from ingest.pipeline import delete_document
from ingest.registry import SCHEMA_VERSION, NewerVersion, OwnerDeleted, Registry, StillDeleting
from ingest.validation import InvalidUpload, safe_name, validate

settings = get_settings()
registry = Registry(settings.registry_path, lease_s=settings.run_lease_s, holder="ingest-api")
qdrant = qdrant_client(settings.qdrant_url, settings.qdrant_api_key_file, timeout=60)
audit = AuditClient(settings.audit_url, "ingest-api")
log = logging.getLogger("ingest.api")
_ASSET_NAME = re.compile(r"^[0-9a-f]{16}\.png$")
_CHUNK = 1 << 20
ASSET_CACHE = "private, max-age=300"  # short: a signed-out browser must not keep serving a user's figures
ACCOUNT_DELETED = "this account has been deleted"
STILL_DELETING = "this file is still being deleted; upload it again in a minute"
NEWER_VERSION = "a newer version of this file is being processed; it replaces this one when it is ready"


async def _drop_points(owner_id: str, doc_id: str, generations: list[str | None] | None = None) -> None:
    await delete_points(qdrant, owner_id, doc_id, generations)


class UploadResult(BaseModel):
    name: str
    document: DocumentInfo | None = None
    error: str | None = None
    existing: bool = False  # the same content was uploaded before: document is that one (its name may differ)
    replaces: bool = False  # a new version: it replaces the owner's older document of this name once ready


@asynccontextmanager
async def lifespan(_app):
    for path in (settings.files_dir / ".incoming", settings.assets_dir, settings.tables_dir, settings.chunks_dir):
        path.mkdir(parents=True, exist_ok=True)
    yield
    await qdrant.close()
    await audit.aclose()


def _worker_alive() -> bool:
    beat = settings.data_dir / "worker.heartbeat"
    return beat.exists() and time.time() - beat.stat().st_mtime < 60


async def readiness() -> dict[str, bool]:
    try:
        await qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    return {"qdrant": qdrant_ok, "worker": _worker_alive()}


app = create_app("ingest-api", lifespan=lifespan, readiness=readiness, log_level=settings.log_level,
                 verifier=Verifier.from_file(settings.public_keys_file, {"auth", "rag"}))


def derive_doc_id(owner_id: str, content_sha256: str) -> str:
    """The same file uploaded by two users gives two ids, so an upload never reveals another user's document."""
    return hashlib.sha256(f"{owner_id}:{content_sha256}".encode()).hexdigest()[:32]


def _get(doc_id: str, user: Identity) -> DocumentInfo:
    doc = registry.get(doc_id, user.user_id)
    if doc is None:
        raise HTTPException(404, f"document {doc_id} not found")
    return doc


@dataclass
class _Placement:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0  # the holder AND waiters keep this entry alive


_placing: dict[str, _Placement] = {}  # one process; run one ingest-api replica


@asynccontextmanager
async def _one_placement(doc_id: str):
    """Uploads of the same file take turns from the first look at the registry to the inserted row, so one of
    them never removes or replaces the file another has placed."""
    placement = _placing.setdefault(doc_id, _Placement())
    placement.users += 1
    try:
        async with placement.lock:
            yield
    finally:
        placement.users -= 1
        if placement.users == 0:
            del _placing[doc_id]


async def _finish_earlier_deletion(doc_id: str, owner_id: str) -> bool:
    """The same file was uploaded and deleted before, and its deletion is not finished (its row is still
    deleting): finish it now when nobody holds it, so the new upload starts from nothing. False while a run or
    another clean-up holds it."""
    earlier = registry.get_deleting(doc_id, owner_id)
    if earlier is None:
        return True
    claim = registry.claim_deleting(doc_id)
    if claim is None:
        return False
    try:
        return await delete_document(settings, registry, _drop_points, earlier, claim)
    except Exception:
        log.exception("finishing an earlier deletion failed", extra={"fields": {"doc_id": doc_id}})
        return False


async def _store(upload: UploadFile, owner_id: str) -> UploadResult:
    name = upload.filename or "upload"
    try:
        name = safe_name(name)
        head = await upload.read(8192)
        ext, doc_type = validate(name, head)
    except InvalidUpload as exc:
        return UploadResult(name=name, error=str(exc))

    limit = settings.max_upload_mb << 20
    tmp = settings.files_dir / ".incoming" / uuid.uuid4().hex
    digest, size = hashlib.sha256(), 0
    try:
        with tmp.open("wb") as out:
            chunk = head
            while chunk:
                size += len(chunk)
                if size > limit:
                    return UploadResult(name=name, error=f"file exceeds {settings.max_upload_mb} MB")
                digest.update(chunk)
                out.write(chunk)
                chunk = await upload.read(_CHUNK)
        if size == 0:
            return UploadResult(name=name, error="file is empty")
        doc_id = derive_doc_id(owner_id, digest.hexdigest())
        async with _one_placement(doc_id):
            existing = registry.get(doc_id, owner_id)
            if existing:
                return UploadResult(name=name, document=existing, existing=True)
            if not await _finish_earlier_deletion(doc_id, owner_id):
                return UploadResult(name=name, error=STILL_DELETING)
            storage_id = uuid.uuid4().hex
            target = paths.source_path(settings, owner_id, doc_id, ext, storage_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, target)  # the file is in place before the job becomes visible
            try:
                info, created = registry.add(doc_id, owner_id, name, doc_type, ext, size, storage_id=storage_id)
            except OwnerDeleted:  # the account was deleted while this upload streamed: keep nothing of it
                paths.remove_owner(settings, owner_id)
                raise
            except StillDeleting:  # a deletion of the same file holds the row: the file placed here is not wanted
                target.unlink(missing_ok=True)
                return UploadResult(name=name, error=STILL_DELETING)
            if not created:
                target.unlink(missing_ok=True)
                return UploadResult(name=name, document=info, existing=True)
            replaces = settings.replace_same_name and registry.has_older_version(doc_id, owner_id)
            return UploadResult(name=name, document=info, replaces=replaces)
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/v1/documents")
async def upload(files: list[UploadFile] = File(...), user: Identity = Depends(require_user)) -> list[UploadResult]:
    """403 for a deleted account: an identity signed just before the deletion is still valid for a while."""
    if registry.is_deleted(user.user_id):
        raise HTTPException(403, ACCOUNT_DELETED)
    results = []
    for f in files:
        try:
            r = await _store(f, user.user_id)
        except OwnerDeleted:
            raise HTTPException(403, ACCOUNT_DELETED) from None
        if r.error:
            await audit.record("document.rejected", None, user_id=user.user_id, name=r.name, error=r.error)
        else:
            d = r.document
            await audit.record("document.uploaded", d.doc_id, user_id=user.user_id, name=d.name,
                               doc_type=d.doc_type.value, size=d.size, status=d.status.value)
        results.append(r)
    return results


# The document list (the UI polls it every few seconds while a document is processed) carries a compact form of
# each table: its name, source, size and columns (name, SQL name, type, profile line, pattern, warning). rag reads
# the full form (statistics, full headers, layout, notes) through /v1/documents/status for describe_table,
# profile_column and the planner.
LIST_EXCLUDE = {"__all__": {"tables": {"__all__": {
    "layout": True, "notes": True,
    "columns": {"__all__": {"top", "quantiles", "examples", "min", "max", "empty", "distinct", "header"}},
}}}}


@app.get("/v1/documents", response_model_exclude=LIST_EXCLUDE)
def list_documents(user: Identity = Depends(require_user)) -> list[DocumentInfo]:
    return registry.all(user.user_id)


@app.post("/v1/documents/status")
def documents_status(body: DocIdsRequest, user: Identity = Depends(require_user)) -> list[DocumentInfo]:
    """The caller's documents among these ids; ids of other users' documents are left out, like unknown ones."""
    return registry.many(body.doc_ids, user.user_id)


@app.get("/v1/documents/{doc_id}")
def get_document(doc_id: DocIdPath, user: Identity = Depends(require_user)) -> DocumentInfo:
    return _get(doc_id, user)


@app.delete("/v1/documents/{doc_id}", status_code=204)
async def delete(doc_id: DocIdPath, user: Identity = Depends(require_user)) -> Response:
    """The document is gone for its owner at once (status deleting). Its points and files are removed here;
    the row goes with them. 204 when everything is gone; 202 when the rest is left to ingest-worker: for a
    document being processed only the points go here, and its run, which stops within one lease renewal,
    removes the files and the row; a failed clean-up is retried by the worker's sweep."""
    marked = registry.mark_deleting(doc_id, user.user_id)
    if marked is None:
        raise HTTPException(404, f"document {doc_id} not found")
    doc, claim = marked
    await audit.record("document.deleted", doc_id, user_id=user.user_id, name=doc.name)
    try:
        if await delete_document(settings, registry, _drop_points, doc, claim):
            return Response(status_code=204)
    except Exception as exc:
        log_fields(log, "deleting a document's data failed; the worker retries", logging.WARNING, doc_id=doc_id,
                   error=type(exc).__name__)
    return JSONResponse({"doc_id": doc_id, "status": "deleting"}, 202)


@app.post("/v1/documents/{doc_id}/reindex", status_code=202)
async def reindex(doc_id: DocIdPath, user: Identity = Depends(require_user)) -> DocumentInfo:
    doc = _get(doc_id, user)
    try:
        requeued = registry.requeue(doc_id, user.user_id, versions=settings.replace_same_name)
    except NewerVersion:
        raise HTTPException(409, NEWER_VERSION) from None
    if not requeued:
        raise HTTPException(409, "document is being processed" if doc.status == DocStatus.PROCESSING
                            else f"document is already {doc.status.value}")
    await audit.record("document.reindex_requested", doc_id, user_id=user.user_id, name=doc.name)
    return _get(doc_id, user)


def _generation(doc: DocumentInfo, requested: str | None) -> str | None:
    generation = doc.generation if requested is None else (None if requested == "legacy" else requested)
    if generation != doc.generation and not registry.is_published(doc.doc_id, generation):
        raise HTTPException(404, "document generation not found")
    return generation


@app.get("/v1/documents/{doc_id}/assets/{name}")
def asset(doc_id: DocIdPath, name: str, generation: GenerationParameter | None = None,
          user: Identity = Depends(require_user)) -> FileResponse:
    doc = _get(doc_id, user)
    path = paths.asset_dir(settings, user.user_id, doc_id, _generation(doc, generation)) / name
    if not _ASSET_NAME.match(name) or not path.is_file():
        raise HTTPException(404, "asset not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": ASSET_CACHE})


@app.get("/v1/documents/{doc_id}/chunks")
def chunks(doc_id: DocIdPath, offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1),
           generation: GenerationParameter | None = None,
           user: Identity = Depends(require_user)) -> list[dict]:
    doc = _get(doc_id, user)
    return chunkstore.page(paths.chunk_path(settings, user.user_id, doc_id, _generation(doc, generation)),
                           offset, min(limit, 1000))


@app.post("/v1/grep")
def grep(body: GrepRequest, user: Identity = Depends(require_user)) -> list[GrepHit]:
    wanted = list(dict.fromkeys(body.doc_ids))
    docs = registry.many(wanted, user.user_id)
    if len(docs) != len(wanted):
        raise HTTPException(404, "document not found")
    if body.generations is not None and set(body.generations) != set(wanted):
        raise HTTPException(422, "generations must name exactly the selected documents")
    files = {d.doc_id: paths.chunk_path(settings, user.user_id, d.doc_id,
             _generation(d, (body.generations[d.doc_id] or "legacy") if body.generations is not None else None))
             for d in docs}
    return chunkstore.grep(files, body.terms, body.limit)


# --- auth's system calls --------------------------------------------------------------------


@app.get("/internal/capabilities")
def capabilities(_: Identity = Depends(require_system)) -> dict:
    return {"service": "ingest", "multi_user": 1, "schema": SCHEMA_VERSION}


@app.delete("/internal/users/{user_id}")
async def purge_user(user_id: str = PathParam(pattern=USER_ID), _: Identity = Depends(require_system)) -> dict:
    """Delete an account's documents: registry rows, then Qdrant points, then files, then cached model replies.
    Repeatable; a job in progress is cleaned up by the worker when it finds its row gone."""
    ids = await asyncio.to_thread(registry.purge_owner, user_id)  # the WAL truncation may wait
    try:
        await delete_user_points(qdrant, user_id)
    except Exception as exc:  # the rows are gone already; calling again finishes the points and files
        raise HTTPException(502, f"deleting the index entries failed ({type(exc).__name__}); try again") from exc
    paths.remove_owner(settings, user_id)
    try:
        await asyncio.to_thread(LlmCache.from_settings(settings).purge_owner, user_id)
    except Exception as exc:  # auth calls again until it succeeds
        log_fields(log, "deleting an account's cached model replies failed", logging.ERROR,
                   error=f"{type(exc).__name__}: {exc}"[:300])
        raise HTTPException(502, f"deleting cached model replies failed ({type(exc).__name__}); try again") from exc
    return {"documents": len(ids)}


async def _adopt_points(owner_id: str, doc_ids: list[str], dry_run: bool) -> int:
    """Set user_id on the points of these documents that have none; returns how many (would) change. Limited
    to the owner's documents, so an unowned point of any other document is never handed to this user."""
    if not await qdrant.collection_exists(COLLECTION):
        return 0
    if not dry_run:
        await ensure_user_index(qdrant)
    changed = 0
    for i in range(0, len(doc_ids), 256):
        unowned = qm.Filter(must=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key=USER_FIELD)),
                                  qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids[i : i + 256]))])
        count = (await qdrant.count(COLLECTION, count_filter=unowned, exact=True)).count
        if count and not dry_run:
            await qdrant.set_payload(COLLECTION, payload={USER_FIELD: owner_id},
                                     points=qm.FilterSelector(filter=unowned), wait=True)
        changed += count
    return changed


@app.post("/internal/migrate-legacy")
async def migrate_legacy(body: MigrateLegacy, _: Identity = Depends(require_system)) -> dict:
    """Documents from before accounts existed become this user's: registry rows, then files (moved under the
    owner), then Qdrant points (user_id set). Ids and point ids do not change. Every step also covers the
    owner's already adopted documents, so a run that failed halfway resumes; a complete second run changes
    nothing. Refused while the worker runs and legacy rows exist (it could index a file mid-move)."""
    owner = body.owner_id
    if registry.legacy_count() and _worker_alive() and not body.force:
        raise HTTPException(409, "stop ingest-worker first")
    try:
        adopted = registry.adopt_legacy(owner, body.dry_run)
    except OwnerDeleted:
        raise HTTPException(409, ACCOUNT_DELETED) from None
    candidates = list(dict.fromkeys([*adopted, *(d.doc_id for d in registry.all(owner))]))
    files = sum(paths.move_legacy(settings, owner, d, body.dry_run) for d in candidates)
    try:
        points = await _adopt_points(owner, candidates, body.dry_run)
    except Exception as exc:  # rows and files are done; calling again finishes the points
        raise HTTPException(502, f"updating the index failed ({type(exc).__name__}); try again") from exc
    return {"documents": len(adopted), "files_moved": files, "points_updated": points,
            "orphans": paths.orphans(settings, registry.all_ids()), "dry_run": body.dry_run}
