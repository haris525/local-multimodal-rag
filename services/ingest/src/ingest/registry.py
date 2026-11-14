"""Document registry and job queue in SQLite (WAL), shared by ingest-api and ingest-worker.

Every document has an owner. The methods ingest-api uses take the owner and put it in the WHERE clause, so
another user's document reads exactly like a missing one. The worker's methods (claim, renew, progress,
complete, fail, release, get_for_worker) work by doc_id: the worker acts for whoever owns the job it claimed.
Rows from before accounts existed have no owner: nobody sees them and the worker never claims them until
adopt_legacy gives them one.

Deleted accounts (schema v2) leave a tombstone in deleted_users, written in the same transaction that deletes
their rows. A request authorized before the deletion (an upload still streaming, a chat that re-signs its
identity) can still arrive afterwards: add() refuses a tombstoned owner atomically, and the worker never
claims or completes such a job, so nothing can be created for an account after it was purged.

Lifecycle (schema v3). A document's status changes only through transition(), and only along NEXT; the
status column has a CHECK with the same values. A claim gives the row a run id and a lease: every write of
that run (renew, progress, complete, fail, release) names the run and applies only while the row is still
processing under it, so a run that outlived a delete, a re-upload of the same file or a second worker
changes nothing it does not own. The lease is renewed while the run is alive; recover() takes back only
rows whose lease ran out, and, when a worker starts, the rows its own earlier process held (claimed_by: the
worker's id, its host name by default). A delete moves the row to deleting (hidden from its owner at once); the row is
removed only after its points and files, by whoever holds it: ingest-api for an idle document, the run for
a document it was processing, or the worker's sweep for any deleting row nobody holds (a crashed run, a
failed clean-up). While the row exists, the same file cannot be uploaded again (StillDeleting), so two
lifetimes of one doc_id never share its files.

Index versions (schema v4). complete() records which version of the parser and chunker built a document's
index; requeue_stale() re-queues the ready documents of an older version, which the worker does at start.
"""

import json
import logging
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag_common.contracts import DocStatus, DocType, DocumentInfo, TableSchema
from rag_common.identity import USER_ID_RE
from rag_common.sqlite import truncate_wal

log = logging.getLogger("ingest.registry")

Q, P, R, F, D = DocStatus.QUEUED, DocStatus.PROCESSING, DocStatus.READY, DocStatus.FAILED, DocStatus.DELETING
# The allowed moves. A new row starts queued (add); a deleting row is removed once its points and files are
# gone. Everything else is refused, whoever asks.
NEXT: dict[DocStatus, frozenset[DocStatus]] = {
    Q: frozenset({P, D}),        # claimed by the worker; deleted
    P: frozenset({R, F, Q, D}),  # completed; failed; released or recovered; deleted (the run finishes it)
    R: frozenset({Q, D}),        # re-index; deleted or superseded
    F: frozenset({Q, D}),        # retry; deleted or superseded
    D: frozenset(),
}

SCHEMA_VERSION = 4
_STATUSES = ", ".join(f"'{s.value}'" for s in DocStatus)
_COLUMNS = """
    doc_id      TEXT PRIMARY KEY,
    owner_id    TEXT,
    name        TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    ext         TEXT NOT NULL,
    size        INTEGER NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ({statuses})),
    progress    TEXT NOT NULL DEFAULT '',
    error       TEXT,
    warnings    TEXT NOT NULL DEFAULT '[]',
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    n_images    INTEGER NOT NULL DEFAULT 0,
    tables      TEXT NOT NULL DEFAULT '[]',
    attempts    INTEGER NOT NULL DEFAULT 0,
    run_id      TEXT,
    lease_until TEXT,
    claimed_by  TEXT,
    index_version INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    CHECK (status <> 'processing' OR run_id IS NOT NULL)
""".format(statuses=_STATUSES)
_SCHEMA = [
    f"CREATE TABLE IF NOT EXISTS documents ({_COLUMNS})",
    "CREATE INDEX IF NOT EXISTS documents_status ON documents (status, created_at)",
    "CREATE INDEX IF NOT EXISTS documents_owner ON documents (owner_id, created_at)",
    "CREATE INDEX IF NOT EXISTS documents_owner_name ON documents (owner_id, name)",
    "CREATE TABLE IF NOT EXISTS deleted_users (user_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL)",  # v2
]
# v2 -> v3: the CHECK on status and the run columns need a rebuild (rowid kept: it orders the versions of a
# file). A row left processing gets a placeholder run without a lease, so the worker's start takes it back;
# a status no code writes becomes failed with a note instead of breaking its owner's whole list.
_KNOWN = "('queued', 'processing', 'ready', 'failed')"
_V2_TO_V3 = [
    f"CREATE TABLE documents_v3 ({_COLUMNS})",
    f"""INSERT INTO documents_v3 (rowid, doc_id, owner_id, name, doc_type, ext, size, status, progress, error, warnings,
           n_chunks, n_images, tables, attempts, run_id, lease_until, claimed_by, created_at, updated_at)
       SELECT rowid, doc_id, owner_id, name, doc_type, ext, size,
           CASE WHEN status IN {_KNOWN} THEN status ELSE 'failed' END, progress,
           CASE WHEN status IN {_KNOWN} THEN error ELSE 'unknown status ' || quote(status) || ' at the registry upgrade' END,
           warnings, n_chunks, n_images, tables, attempts,
           CASE WHEN status = 'processing' THEN 'before-v3' END, NULL, NULL, created_at, updated_at
       FROM documents ORDER BY rowid""",
    "DROP TABLE documents",
    "ALTER TABLE documents_v3 RENAME TO documents",
]
# v3 -> v4: the index version that built each ready document (pipeline.INDEX_VERSION); rows from before it
# get 1, so the worker's start re-queues them once a parsing or chunking fix raises the version.
_V3_TO_V4 = "ALTER TABLE documents ADD COLUMN index_version INTEGER NOT NULL DEFAULT 1"
_LIVE_OWNER = "NOT EXISTS (SELECT 1 FROM deleted_users d WHERE d.user_id = documents.owner_id)"
_FIELDS = frozenset({"progress", "error", "warnings", "n_chunks", "n_images", "tables", "attempts", "run_id",
                     "lease_until", "claimed_by", "index_version"})


class OwnerDeleted(Exception):
    """The owner's account was deleted: nothing may be stored for it any more."""


class StillDeleting(Exception):
    """This file's previous upload is being deleted (its row exists until its points and files are gone)."""


class Sql(str):
    """A column value that is an SQL expression over the row (attempts + 1), not a parameter."""


@dataclass(frozen=True)
class Completed:
    """A run's document is ready. superseded: the owner's earlier versions of the same name, moved to deleting
    in the same transaction, each with the claim (run_id) to finish its deletion with."""

    superseded: list[DocumentInfo]


def _owner(owner_id: str) -> str:
    """Fail closed: a missing or malformed owner never reaches a query (NULL would mean 'legacy')."""
    if not isinstance(owner_id, str) or not USER_ID_RE.match(owner_id):
        raise ValueError("owner_id is not a user id")
    return owner_id


def new_run() -> str:
    return uuid.uuid4().hex


class Registry:
    def __init__(self, path: Path, *, lease_s: float = 90.0, clock: Callable[[], datetime] | None = None,
                 holder: str | None = None):
        """lease_s: how long a claim stays valid without renewal; the worker renews it every few seconds.
        holder: who this process is (claimed_by of the rows it claims): the worker's id, or "ingest-api"."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self.lease_s = lease_s
        self.holder = holder
        self.clock = clock or (lambda: datetime.now(UTC))
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            self._upgrade(db)

    def _now(self) -> str:
        return self.clock().astimezone(UTC).isoformat(timespec="seconds")

    def _lease(self) -> str:
        return (self.clock() + timedelta(seconds=self.lease_s)).astimezone(UTC).isoformat(timespec="seconds")

    @contextmanager
    def _connect(self, secure: bool = False) -> Iterator[sqlite3.Connection]:
        """secure: deletes overwrite what they free with zeros (PRAGMA secure_delete), so a deleted document's
        name and error text do not stay readable in free pages of the file."""
        db = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        if secure:
            db.execute("PRAGMA secure_delete=ON")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _upgrade(db: sqlite3.Connection) -> None:
        """Bring the schema to SCHEMA_VERSION in one transaction (a second run does nothing). v1 added owner_id;
        v2 the deleted_users table; v3 the CHECK on status and the run columns (a table rebuild); v4 the
        index_version column."""
        if db.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return
        db.execute("BEGIN IMMEDIATE")
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:  # another process may have won
                columns = {r[1] for r in db.execute("PRAGMA table_info(documents)")}
                if columns and "owner_id" not in columns:
                    db.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT")
                if columns and "run_id" not in columns:
                    for statement in _V2_TO_V3:
                        db.execute(statement)
                columns = {r[1] for r in db.execute("PRAGMA table_info(documents)")}  # the v3 rebuild has it
                if columns and "index_version" not in columns:
                    db.execute(_V3_TO_V4)
                for statement in _SCHEMA:
                    db.execute(statement)
                db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    @staticmethod
    def _info(row: sqlite3.Row) -> DocumentInfo:
        return DocumentInfo(
            doc_id=row["doc_id"],
            name=row["name"],
            doc_type=DocType(row["doc_type"]),
            size=row["size"],
            status=DocStatus(row["status"]),
            progress=row["progress"],
            error=row["error"],
            warnings=json.loads(row["warnings"]),
            n_chunks=row["n_chunks"],
            n_images=row["n_images"],
            tables=[TableSchema(**t) for t in json.loads(row["tables"])],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            owner_id=row["owner_id"],
            run_id=row["run_id"],
        )

    # --- the one place a status changes ---------------------------------------------------------

    def _transition(self, db: sqlite3.Connection, doc_id: str, to: DocStatus, *,
                    sources: Iterable[DocStatus] | None = None, owner_id: str | None = None, run: str | None = None,
                    live_owner: bool = False, **fields) -> sqlite3.Row | None:
        """Move one row to `to`, inside the caller's transaction. Allowed only from a status whose NEXT holds
        `to` (sources narrows them further); with `run`, only while the row is processing under that run (the
        worker's writes); with `owner_id`, only the owner's row; with `live_owner`, only while the owner's
        account exists. Returns the row after the move, or None when the row is not (or no longer) in a state
        that allows it: the caller's view of the row is stale and nothing changed."""
        allowed = [s for s in DocStatus if to in NEXT[s]]
        if sources is not None:
            sources = list(sources)
            if any(s not in allowed for s in sources):
                raise ValueError(f"not allowed by NEXT: {[s.value for s in sources]} -> {to.value}")
            allowed = sources
        if run is not None:
            allowed = [s for s in allowed if s == P]
        unknown = set(fields) - _FIELDS
        if unknown or not allowed:
            raise ValueError(f"bad transition to {to.value}: {sorted(unknown) or 'no source state'}")
        values: dict[str, object] = {"to": to.value, "now": self._now(), "doc_id": doc_id}
        sets = ["status = :to", "updated_at = :now"]
        for name, value in fields.items():
            if isinstance(value, Sql):
                sets.append(f"{name} = {value}")
            else:
                sets.append(f"{name} = :f_{name}")
                values[f"f_{name}"] = value
        marks = ", ".join(f":s{i}" for i in range(len(allowed)))
        values |= {f"s{i}": s.value for i, s in enumerate(allowed)}
        where = [f"doc_id = :doc_id AND status IN ({marks})"]
        if run is not None:
            where.append("run_id = :run")
            values["run"] = run
        if owner_id is not None:
            where.append("owner_id = :owner")
            values["owner"] = _owner(owner_id)
        if live_owner:
            where.append(_LIVE_OWNER)
        return db.execute(f"UPDATE documents SET {', '.join(sets)} WHERE {' AND '.join(where)} RETURNING *",
                          values).fetchone()

    def transition(self, doc_id: str, to: DocStatus, **kwargs) -> DocumentInfo | None:
        """_transition in a transaction of its own."""
        with self._connect() as db:
            row = self._transition(db, doc_id, to, **kwargs)
        return self._info(row) if row else None

    # --- ingest-api -----------------------------------------------------------------------------

    def add(self, doc_id: str, owner_id: str, name: str, doc_type: DocType, ext: str,
            size: int) -> tuple[DocumentInfo, bool]:
        """Insert a queued document; returns (info, created). The same bytes from the same owner map to one
        doc_id (derive_doc_id salts it with the owner, so other owners never collide). OwnerDeleted when the
        owner's account was deleted (checked in the same statement as the insert); StillDeleting while an
        earlier upload of the same file is being deleted."""
        now = self._now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cur = db.execute(
                "INSERT OR IGNORE INTO documents (doc_id, owner_id, name, doc_type, ext, size, status, created_at,"
                " updated_at) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?"
                " WHERE NOT EXISTS (SELECT 1 FROM deleted_users WHERE user_id = ?)",
                (doc_id, _owner(owner_id), name, doc_type.value, ext, size, Q.value, now, now, owner_id),
            )
            row = db.execute("SELECT * FROM documents WHERE doc_id = ? AND owner_id = ?", (doc_id, owner_id)).fetchone()
            deleted = self._deleted(db, owner_id)
            db.execute("COMMIT")
        if deleted:
            raise OwnerDeleted(owner_id)
        if row is None:  # the id is taken by a row of another owner (or an unadopted one): never return it
            raise RuntimeError(f"document id {doc_id} belongs to another owner")
        if row["status"] == D.value:
            raise StillDeleting(doc_id)
        return self._info(row), cur.rowcount == 1

    @staticmethod
    def _deleted(db: sqlite3.Connection, owner_id: str) -> bool:
        return db.execute("SELECT 1 FROM deleted_users WHERE user_id = ?", (owner_id,)).fetchone() is not None

    def is_deleted(self, owner_id: str) -> bool:
        """True when this owner's account was deleted (purge_owner left its tombstone)."""
        with self._connect() as db:
            return self._deleted(db, _owner(owner_id))

    def get(self, doc_id: str, owner_id: str) -> DocumentInfo | None:
        """The owner's document; a deleting one is gone for its owner."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM documents WHERE doc_id = ? AND owner_id = ? AND status != ?",
                             (doc_id, _owner(owner_id), D.value)).fetchone()
        return self._info(row) if row else None

    def get_deleting(self, doc_id: str, owner_id: str) -> DocumentInfo | None:
        """The owner's document of this id while it is being deleted."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM documents WHERE doc_id = ? AND owner_id = ? AND status = ?",
                             (doc_id, _owner(owner_id), D.value)).fetchone()
        return self._info(row) if row else None

    def get_for_worker(self, doc_id: str) -> DocumentInfo | None:
        """Unscoped, any status: the worker and the pipeline only."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._info(row) if row else None

    def ext(self, doc_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT ext FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row["ext"] if row else None

    def many(self, doc_ids: list[str], owner_id: str) -> list[DocumentInfo]:
        """The owner's documents among these ids (another owner's ids and deleting ones are left out, like
        unknown ones)."""
        marks = ",".join("?" * len(doc_ids))
        with self._connect() as db:
            rows = db.execute(f"SELECT * FROM documents WHERE owner_id = ? AND status != ? AND doc_id IN ({marks})",
                              [_owner(owner_id), D.value, *doc_ids]).fetchall()
        return [self._info(r) for r in rows]

    def all(self, owner_id: str) -> list[DocumentInfo]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM documents WHERE owner_id = ? AND status != ? ORDER BY created_at DESC",
                              (_owner(owner_id), D.value)).fetchall()
        return [self._info(r) for r in rows]

    def requeue(self, doc_id: str, owner_id: str) -> bool:
        """Re-index a ready or failed document; False for a queued, processing or missing one."""
        with self._connect() as db:
            return self._transition(db, doc_id, Q, sources=(R, F), owner_id=owner_id, attempts=0, error=None,
                                    progress="") is not None

    def requeue_stale(self, index_version: int) -> int:
        """Re-index every ready document an older index version built (a parsing or chunking fix means it may
        be missing content); its old chunks stay searchable until the new ones replace them. Queued, processing,
        failed and deleting rows, and the rows of deleted accounts, are left alone. Returns how many it queued."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = db.execute(f"SELECT doc_id FROM documents WHERE status = ? AND index_version < ?"
                                  f" AND owner_id IS NOT NULL AND {_LIVE_OWNER}", (R.value, index_version)).fetchall()
                moved = [self._transition(db, doc_id, Q, sources=(R,), live_owner=True, attempts=0, error=None,
                                          progress="") for (doc_id,) in rows]
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return sum(row is not None for row in moved)

    def mark_deleting(self, doc_id: str, owner_id: str) -> tuple[DocumentInfo, str | None] | None:
        """The owner deletes a document: it is hidden at once. Returns the row as it was and the claim to
        finish the deletion with, or None when there is no such document. The claim is None for a document
        being processed: its run keeps the row and finishes the deletion when it sees it (within seconds)."""
        with self._connect(secure=True) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                before = db.execute("SELECT * FROM documents WHERE doc_id = ? AND owner_id = ? AND status != ?",
                                    (doc_id, _owner(owner_id), D.value)).fetchone()
                token, row = None, None
                if before is not None:
                    status = DocStatus(before["status"])
                    if status == P:  # the run's id and lease stay: it is the one to finish
                        row = self._transition(db, doc_id, D, sources=(P,), owner_id=owner_id, progress="")
                    else:
                        token = new_run()
                        row = self._transition(db, doc_id, D, sources=(status,), owner_id=owner_id, progress="",
                                               run_id=token, lease_until=self._lease(), claimed_by=self.holder)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return (self._info(before), token) if row is not None else None

    # --- deleting rows: finished by whoever holds them ------------------------------------------

    def _free(self, mine: bool) -> tuple[str, dict]:
        """Rows nobody holds: no lease, or one that ran out; with mine, also the ones this holder's earlier
        process held (it is gone: this one is starting)."""
        values = {"now": self._now(), "holder": self.holder if mine else None}
        return "(lease_until IS NULL OR lease_until < :now OR (:holder IS NOT NULL AND claimed_by = :holder))", values

    def deleting(self, mine: bool = False) -> list[DocumentInfo]:
        """Deleting rows nobody holds (a run or a clean-up that died, see _free)."""
        free, values = self._free(mine)
        with self._connect() as db:
            rows = db.execute(f"SELECT * FROM documents WHERE status = :d AND {free} ORDER BY updated_at",
                              values | {"d": D.value}).fetchall()
        return [self._info(r) for r in rows]

    def claim_deleting(self, doc_id: str, mine: bool = False) -> str | None:
        """Take a deleting row nobody holds, to finish its deletion; returns the claim, or None."""
        token = new_run()
        free, values = self._free(mine)
        with self._connect() as db:
            cur = db.execute(f"UPDATE documents SET run_id = :token, lease_until = :lease, claimed_by = :me,"
                             f" updated_at = :now WHERE doc_id = :doc AND status = :d AND {free}",
                             values | {"token": token, "lease": self._lease(), "me": self.holder, "doc": doc_id,
                                       "d": D.value})
        return token if cur.rowcount == 1 else None

    def renew_claim(self, doc_id: str, claim: str) -> bool:
        """Extend the lease of a deleting row this claim holds; False when someone else holds it now (the
        lease ran out and it was claimed again). A clean-up renews before each step that removes data."""
        with self._connect() as db:
            cur = db.execute("UPDATE documents SET lease_until = ? WHERE doc_id = ? AND status = ? AND run_id = ?",
                             (self._lease(), doc_id, D.value, claim))
        return cur.rowcount == 1

    def finish_deleting(self, doc_id: str, claim: str) -> bool:
        """Remove a deleting row whose points and files are gone; only the holder of the claim can."""
        with self._connect(secure=True) as db:
            cur = db.execute("DELETE FROM documents WHERE doc_id = ? AND status = ? AND run_id = ?",
                             (doc_id, D.value, claim))
        return cur.rowcount == 1

    # --- ingest-worker: one run of one job --------------------------------------------------------

    def claim_next(self) -> DocumentInfo | None:
        """Atomically move the oldest queued document with a live owner to processing, with a new run id and a
        lease (never one of a deleted account). The returned info carries the run id."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                pick = db.execute(f"SELECT doc_id FROM documents WHERE status = ? AND owner_id IS NOT NULL"
                                  f" AND {_LIVE_OWNER} ORDER BY created_at LIMIT 1", (Q.value,)).fetchone()
                row = None
                if pick is not None:
                    row = self._transition(db, pick[0], P, sources=(Q,), live_owner=True, attempts=Sql("attempts + 1"),
                                           run_id=new_run(), lease_until=self._lease(), claimed_by=self.holder,
                                           progress="starting", error=None)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return self._info(row) if row else None

    def _run_write(self, doc_id: str, run: str | None, sets: str, values: tuple) -> bool:
        """A write of a run that keeps the row processing (lease, progress): applies only while it holds it."""
        with self._connect() as db:
            cur = db.execute(f"UPDATE documents SET {sets}, lease_until = ? WHERE doc_id = ? AND status = ?"
                             " AND run_id = ?", (*values, self._lease(), doc_id, P.value, run))
        return cur.rowcount == 1

    def renew(self, doc_id: str, run: str | None) -> bool:
        """Extend the run's lease; False when the run lost the row (deleted, purged, taken back)."""
        return self._run_write(doc_id, run, "updated_at = updated_at", ())

    def progress(self, doc_id: str, run: str | None, text: str) -> bool:
        """Show progress (renews the lease); False when the run lost the row."""
        return self._run_write(doc_id, run, "progress = ?, updated_at = ?", (text, self._now()))

    def complete(self, doc_id: str, run: str | None, n_chunks: int, n_images: int, warnings: list[str],
                 tables: list[TableSchema], supersede: bool = False, index_version: int = 1) -> Completed | None:
        """Mark the run's document ready; None when the run no longer holds it (deleted, taken by another run,
        or its owner's account was deleted). With supersede, the same transaction moves the owner's earlier
        ready or failed versions of the document's name (inserted before it) to deleting: there is never a
        moment with both versions listed or with neither, and the name is the row's own, not the run's copy.
        index_version: the version of what built the index (pipeline.INDEX_VERSION), for requeue_stale."""
        with self._connect(secure=True) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = self._transition(
                    db, doc_id, R, run=run, live_owner=True, progress="", n_chunks=n_chunks, n_images=n_images,
                    warnings=json.dumps(warnings), tables=json.dumps([t.model_dump() for t in tables]), run_id=None,
                    lease_until=None, claimed_by=None, index_version=index_version)
                old = self._supersede(db, doc_id) if row is not None and supersede else []
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return Completed([self._info(r) for r in old]) if row is not None else None

    def _supersede(self, db: sqlite3.Connection, doc_id: str) -> list[sqlite3.Row]:
        """Inside complete's transaction: the earlier ready or failed versions of the same owner and name become
        deleting, each with a claim. A newer upload still queued or processing is never touched, and another
        user's file of the same name never matches."""
        new = db.execute("SELECT rowid, owner_id, name FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        old = db.execute("SELECT doc_id FROM documents WHERE owner_id = ? AND name = ? AND rowid < ?"
                         " AND status IN (?, ?) ORDER BY rowid",
                         (new["owner_id"], new["name"], new["rowid"], R.value, F.value)).fetchall()
        moved = [self._transition(db, r["doc_id"], D, sources=(R, F), progress="", run_id=new_run(),
                                  lease_until=self._lease(), claimed_by=self.holder) for r in old]
        return [r for r in moved if r is not None]

    def fail(self, doc_id: str, run: str | None, error: str) -> bool:
        """Mark the run's document failed; False when the run no longer holds it."""
        with self._connect() as db:
            return self._transition(db, doc_id, F, run=run, progress="", error=error[:2000], run_id=None,
                                    lease_until=None, claimed_by=None) is not None

    def release(self, doc_id: str, run: str | None) -> bool:
        """The worker is shutting down: put the run's document back in the queue, and do not count the attempt
        (attempts guard against files that crash the worker, not against restarts)."""
        with self._connect() as db:
            return self._transition(db, doc_id, Q, run=run, attempts=Sql("MAX(attempts - 1, 0)"), progress="",
                                    run_id=None, lease_until=None, claimed_by=None) is not None

    def recover(self, max_attempts: int, mine: bool = False) -> int:
        """Jobs whose run stopped renewing its lease were interrupted: retry them, or give up after
        max_attempts. A job whose lease is still running belongs to a live worker and is left alone; with mine
        (a worker's start), so are all but the jobs this worker's earlier process held, which are taken back at
        once. Returns how many it took back. The select and the moves are one transaction."""
        free, values = self._free(mine)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = db.execute(f"SELECT doc_id, attempts FROM documents WHERE status = :p AND {free}",
                                  values | {"p": P.value}).fetchall()
                for doc_id, attempts in rows:
                    if attempts < max_attempts:
                        self._transition(db, doc_id, Q, sources=(P,), progress="", run_id=None, lease_until=None,
                                         claimed_by=None)
                    else:
                        self._transition(db, doc_id, F, sources=(P,), progress="", run_id=None, lease_until=None,
                                         claimed_by=None, error="worker stopped while processing this file")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return len(rows)

    # --- accounts -----------------------------------------------------------------------------------

    def purge_owner(self, owner_id: str) -> list[str]:
        """Account deletion: tombstone the owner and delete every document row of it, in one transaction;
        returns their ids. From then on add() refuses the owner. A job in progress is cleaned up by the worker
        when it finds its row gone. Freed pages are zeroed and the WAL is truncated (it may wait for the
        worker's connection: call it off the event loop). Repeatable."""
        with self._connect(secure=True) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO deleted_users (user_id, deleted_at) VALUES (?, ?)",
                       (_owner(owner_id), self._now()))
            rows = db.execute("DELETE FROM documents WHERE owner_id = ? RETURNING doc_id", (owner_id,)).fetchall()
            db.execute("COMMIT")
            truncate_wal(db, log, purge="owner")
        return [r[0] for r in rows]

    # --- legacy migration -------------------------------------------------------------------

    def legacy_count(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM documents WHERE owner_id IS NULL").fetchone()[0]

    def adopt_legacy(self, owner_id: str, dry_run: bool = False) -> list[str]:
        """Give every document from before accounts existed to one owner; returns their ids (unchanged ids).
        OwnerDeleted for a deleted account."""
        if self.is_deleted(owner_id):
            raise OwnerDeleted(owner_id)
        with self._connect() as db:
            if dry_run:
                rows = db.execute("SELECT doc_id FROM documents WHERE owner_id IS NULL").fetchall()
            else:
                rows = db.execute("UPDATE documents SET owner_id = ? WHERE owner_id IS NULL RETURNING doc_id",
                                  (_owner(owner_id),)).fetchall()
        return [r[0] for r in rows]

    def all_ids(self) -> set[str]:
        """Every document id of every owner (the migration's orphan check)."""
        with self._connect() as db:
            return {r[0] for r in db.execute("SELECT doc_id FROM documents")}

    def owned_ids(self) -> set[tuple[str, str]]:
        """Every (owner_id, doc_id) of every owner, any status (the LLM cache sweep: what still exists)."""
        with self._connect() as db:
            rows = db.execute("SELECT owner_id, doc_id FROM documents WHERE owner_id IS NOT NULL")
            return {(owner_id, doc_id) for owner_id, doc_id in rows}
