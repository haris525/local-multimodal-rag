"""A content-addressed cache of the model replies ingest-worker gets: chart data extraction, figure and image
descriptions and document summaries (describe.py, every model call of ingest). Reindexing content seen before
reuses the replies instead of asking the model again.

Key: sha256 over the owner; the call (chart_extraction, image_description, document_summary; the two passes of
a chart's description are two calls) and its prompt version (describe.py); the model identity (the name the
server answers to, its weights file, whose name carries the quantization, its vision projector, and
INGEST_LLM_CACHE_SALT); the retry policy for a cut reply (factor and ceiling); and the exact request body: the
messages with the image as sent (the resized PNG) and the prompt text with the document name, caption, section
and, for a chart's description, the table read in the first pass; max_tokens (the token budget), temperature,
top_p, top_k. Any change to what the model sees, or to how it is asked, gives another key.

Scope: every row names its owner, every statement has the owner in its WHERE clause, and the owner is part of
the key, so two accounts never share an entry, even for identical files.

What is stored: only a whole reply, with finish_reason "stop", not empty and not a refusal. A reply cut at its
token limit, a failed call, an empty or a refused reply is never stored (it is asked again next time), and each
skip is logged. A stored reply carries its sha256; a row that does not match it, or that names another call, is
a corrupt entry: it is removed, logged, and the model is asked.

Lifetime: the table uses records which documents of the owner used each entry. Deleting a document removes its
uses and every entry of the owner that no other document uses; deleting an account removes all its entries;
both zero what they free and truncate the WAL. The worker's sweep removes the uses of documents the registry no
longer has (written by a run after its document was deleted, when the worker died before it cleaned up) and the
entries nobody uses. The size is bounded: past INGEST_LLM_CACHE_MAX_MB or INGEST_LLM_CACHE_MAX_ENTRIES the
least recently used entries go, checked on every write.

Failures: the cache never fails a job and never returns a reply it cannot vouch for. A busy, full or damaged
file makes a lookup a miss and a write a skip, each logged at WARNING, and after INGEST_LLM_CACHE_MAX_ERRORS
errors a job stops using the cache. A file SQLite reports damaged is deleted and started again (it is only a
cache, and deleting it purges everything in it). Any other failure of a purge raises: the document's deletion
is retried by the worker's sweep, the account's by auth.

Storage: its own SQLite file (WAL), llm_cache.db next to registry.db in the ingest data volume, not tables in
the registry. It holds bulk text with its own bound, eviction and reset, which the registry (read on every
request of ingest-api) should not carry, and a damaged or full cache file never touches the registry. The
price is that a purge is not in the registry's transaction: a purge runs before the registry row goes, and
the sweep removes what a crashed run left.
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from rag_common.contracts import DOC_ID
from rag_common.identity import USER_ID_RE
from rag_common.logging import log_fields
from rag_common.retry import retry_once
from rag_common.sqlite import truncate_wal

log = logging.getLogger("ingest.llmcache")

SCHEMA_VERSION = 1  # of llm_cache.db; a file of any other version is emptied and started again (it is a cache)
KEY_FORMAT = 1  # bump when what goes into a key changes shape: every entry is then a miss
_DOC_ID_RE = re.compile(DOC_ID)
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS entries (
        owner_id   TEXT NOT NULL,
        key        TEXT NOT NULL,
        call       TEXT NOT NULL,
        reply      TEXT NOT NULL,
        reply_sha  TEXT NOT NULL,
        bytes      INTEGER NOT NULL,
        created_at REAL NOT NULL,
        used_at    REAL NOT NULL,
        PRIMARY KEY (owner_id, key))""",
    "CREATE INDEX IF NOT EXISTS entries_lru ON entries (used_at, bytes)",
    """CREATE TABLE IF NOT EXISTS uses (
        owner_id TEXT NOT NULL,
        doc_id   TEXT NOT NULL,
        key      TEXT NOT NULL,
        used_at  REAL NOT NULL,
        PRIMARY KEY (owner_id, doc_id, key))""",
    "CREATE INDEX IF NOT EXISTS uses_key ON uses (owner_id, key)",
]
_DROP = ["DROP TABLE IF EXISTS entries", "DROP TABLE IF EXISTS uses"]
_USE = ("INSERT INTO uses (owner_id, doc_id, key, used_at) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (owner_id, doc_id, key) DO UPDATE SET used_at = excluded.used_at")
_UNUSED = "NOT EXISTS (SELECT 1 FROM uses u WHERE u.owner_id = entries.owner_id AND u.key = entries.key)"
_NO_ENTRY = "NOT EXISTS (SELECT 1 FROM entries e WHERE e.owner_id = uses.owner_id AND e.key = uses.key)"
# A reply that starts like a refusal is not stored: asked again next time, the model may describe the image.
_REFUSAL = re.compile(r"\W*(i(\s+am|['\u2019]m)\s+(sorry|unable|not\s+able)|sorry\b|"
                      r"i\s+(cannot|can\s+not|can['\u2019]t|won['\u2019]t|will\s+not)\b|as\s+an\s+ai\b)",
                      re.IGNORECASE)


class CacheReset(Exception):
    """The cache file was damaged and has been deleted: nothing that was in it is left."""


class _Damaged(sqlite3.DatabaseError):
    """PRAGMA quick_check found the file damaged."""


def _damaged(exc: BaseException) -> bool:
    """SQLite says the file is malformed or not a database (not: busy, locked, full, read-only)."""
    name = getattr(exc, "sqlite_errorname", "") or ""
    text = str(exc)
    return (isinstance(exc, _Damaged) or name.startswith(("SQLITE_CORRUPT", "SQLITE_NOTADB"))
            or "malformed" in text or "not a database" in text)


def _check(owner_id: str, doc_id: str | None = None) -> None:
    """Fail closed: a missing or malformed owner or document id never reaches a query."""
    if not isinstance(owner_id, str) or not USER_ID_RE.match(owner_id):
        raise ValueError("owner_id is not a user id")
    if doc_id is not None and (not isinstance(doc_id, str) or not _DOC_ID_RE.match(doc_id)):
        raise ValueError("doc_id is not a document id")


def cache_key(*, owner: str, call: str, version: int, model: str, request: dict, retry: dict) -> str:
    """sha256 over everything that decides the reply (module docstring); `request` is the exact request body."""
    material = {"format": KEY_FORMAT, "owner": owner, "call": call, "version": version, "model": model,
                "retry": retry, "request": request}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def skip_reason(reply: str, finish: Any) -> str | None:
    """Why a reply is not stored, or None to store it: cut at its token limit, not finished normally, empty, or
    a refusal."""
    if finish == "length":
        return "truncated"
    if finish != "stop":
        return "not_finished"
    if not reply.strip():
        return "empty"
    if _REFUSAL.match(reply):
        return "refused"
    return None


def _intact(row: sqlite3.Row | tuple, call: str) -> bool:
    stored_call, reply, sha = row
    return (stored_call == call and isinstance(reply, str) and bool(reply) and isinstance(sha, str)
            and hashlib.sha256(reply.encode()).hexdigest() == sha)


class LlmCache:
    """llm_cache.db. No connection is kept: each operation opens its own (like the registry), so ingest-api and
    any number of workers share the file. Operations raise on failure; JobCache turns that into misses."""

    def __init__(self, path: Path, *, max_bytes: int, max_entries: int, timeout_s: float = 10.0,
                 clock: Callable[[], float] = time.time):
        self.path = path
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.timeout_s = timeout_s
        self.clock = clock
        self.model: str | None = None  # the model identity (model_identity); None: no lookups or writes

    @classmethod
    def from_settings(cls, settings) -> "LlmCache":
        return cls(settings.llm_cache_path, max_bytes=settings.llm_cache_max_mb << 20,
                   max_entries=settings.llm_cache_max_entries, timeout_s=settings.llm_cache_timeout_s)

    def job(self, owner_id: str, doc_id: str, *, run: Callable[..., Awaitable] | None = None,
            max_errors: int = 3) -> "JobCache":
        if not self.model:
            raise ValueError("the cache has no model identity")
        return JobCache(self, owner_id, doc_id, self.model, run=run or asyncio.to_thread, max_errors=max_errors)

    # --- the file ------------------------------------------------------------------------------

    @contextmanager
    def _connect(self, create: bool) -> Iterator[sqlite3.Connection | None]:
        """create: lookups, writes and the worker's start make the file and its tables; a purge or the sweep
        finds no file and has nothing to do (None). Deletes zero what they free (secure_delete): replies are
        document content."""
        if not create and not self.path.exists():
            yield None
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=self.timeout_s, isolation_level=None)
        try:
            db.execute("PRAGMA secure_delete=ON")
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                self._init(db)
            yield db
        finally:
            db.close()

    @staticmethod
    def _init(db: sqlite3.Connection) -> None:
        """Create the tables in one transaction (a second process finds them made). A file of another schema
        version is emptied first: it is a cache, and there is nothing in it worth a migration."""
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("BEGIN IMMEDIATE")
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                if version:
                    log_fields(log, "llm cache file has another schema version; starting it empty", logging.WARNING,
                               version=version, expected=SCHEMA_VERSION)
                    for statement in _DROP:
                        db.execute(statement)
                for statement in _SCHEMA:
                    db.execute(statement)
                db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:  # SQLite rolls back by itself on some errors (a full disk)
                db.execute("ROLLBACK")
            raise

    def _run(self, fn: Callable[[sqlite3.Connection], Any], *, create: bool) -> Any:
        """fn(db) on a connection; None when there is no file and create is off. A file SQLite reports damaged
        is deleted (CacheReset); any other error is raised as it is."""
        try:
            with self._connect(create) as db:
                return None if db is None else fn(db)
        except sqlite3.DatabaseError as exc:
            if _damaged(exc) and self.reset(f"{type(exc).__name__}: {exc}"):
                raise CacheReset(str(exc)) from exc
            raise

    def reset(self, reason: str) -> bool:
        """Delete the damaged cache file. Everything in it is gone, which is safe: it is a cache, and any purge
        is complete. The WAL and shared-memory files go first, so a stale log is never replayed into a new file.
        False when a file could not be deleted (logged with what to do)."""
        log_fields(log, "llm cache file is damaged; deleting it and starting again", logging.ERROR, reason=reason[:300])
        try:
            for suffix in ("-wal", "-shm", ""):
                self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
        except OSError as exc:
            log_fields(log, "deleting the damaged llm cache file failed; stop ingest-worker and delete it",
                       logging.ERROR, path=str(self.path), error=f"{type(exc).__name__}: {exc}"[:300])
            return False
        return True

    @staticmethod
    def _write(db: sqlite3.Connection, fn: Callable[[], Any]) -> Any:
        db.execute("BEGIN IMMEDIATE")
        try:
            result = fn()
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:  # SQLite rolls back by itself on some errors (a full disk)
                db.execute("ROLLBACK")
            raise
        return result

    def _evict(self, db: sqlite3.Connection) -> int:
        """Inside a write transaction: drop the least recently used entries (and their uses) until both bounds
        hold. Returns how many went."""
        count, total = db.execute("SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM entries").fetchone()
        victims: list[tuple[str, str]] = []
        offset = 0
        while count > self.max_entries or total > self.max_bytes:
            batch = db.execute("SELECT owner_id, key, bytes FROM entries ORDER BY used_at, rowid LIMIT 500 OFFSET ?",
                               (offset,)).fetchall()
            if not batch:
                break
            for owner, key, size in batch:
                if count <= self.max_entries and total <= self.max_bytes:
                    break
                victims.append((owner, key))
                count, total = count - 1, total - size
            offset += len(batch)
        db.executemany("DELETE FROM entries WHERE owner_id = ? AND key = ?", victims)
        db.executemany("DELETE FROM uses WHERE owner_id = ? AND key = ?", victims)
        return len(victims)

    def _stats(self, db: sqlite3.Connection) -> tuple[int, int]:
        count, total = db.execute("SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM entries").fetchone()
        return count, total

    def open(self, create: bool = True) -> tuple[int, int]:
        """At the worker's start: create or upgrade the file, check it (PRAGMA quick_check; a damaged file is
        deleted and started again) and evict down to the bounds, which may have been lowered. Returns (entries,
        bytes). Without create, a missing file stays missing: (0, 0)."""
        def check(db):
            result = db.execute("PRAGMA quick_check").fetchone()[0]
            if result != "ok":
                raise _Damaged(f"quick_check: {result}"[:300])
            if evicted := self._write(db, lambda: self._evict(db)):
                log_fields(log, "llm cache evicted entries past its bounds", evicted=evicted)
            return self._stats(db)

        try:
            return self._run(check, create=create) or (0, 0)
        except CacheReset:
            return self._run(self._stats, create=create) or (0, 0)

    def stats(self) -> tuple[int, int]:
        """(entries, bytes); (0, 0) without a file."""
        return self._run(self._stats, create=False) or (0, 0)

    # --- one job's lookups and writes -------------------------------------------------------------

    def get(self, owner_id: str, key: str, doc_id: str, call: str) -> tuple[str | None, bool]:
        """(reply, corrupt). A hit marks the entry used (for eviction) and records that this document uses it. A
        row whose reply does not match its checksum, or that names another call, is removed: (None, True)."""
        _check(owner_id, doc_id)

        def lookup(db):
            def read():
                row = db.execute("SELECT call, reply, reply_sha FROM entries WHERE owner_id = ? AND key = ?",
                                 (owner_id, key)).fetchone()
                if row is None:
                    return None, False
                if not _intact(row, call):
                    db.execute("DELETE FROM entries WHERE owner_id = ? AND key = ?", (owner_id, key))
                    db.execute("DELETE FROM uses WHERE owner_id = ? AND key = ?", (owner_id, key))
                    return None, True
                now = self.clock()
                db.execute("UPDATE entries SET used_at = ? WHERE owner_id = ? AND key = ?", (now, owner_id, key))
                db.execute(_USE, (owner_id, doc_id, key, now))
                return row[1], False

            return self._write(db, read)

        return self._run(lookup, create=True)

    def put(self, owner_id: str, key: str, doc_id: str, call: str, reply: str) -> int:
        """Store a whole reply and record that this document uses it, then evict past the bounds; one
        transaction. Returns how many entries were evicted."""
        _check(owner_id, doc_id)
        data = reply.encode()
        sha = hashlib.sha256(data).hexdigest()

        def store(db):
            def write():
                now = self.clock()
                db.execute("INSERT INTO entries (owner_id, key, call, reply, reply_sha, bytes, created_at, used_at)"
                           " VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (owner_id, key) DO UPDATE SET"
                           " call = excluded.call, reply = excluded.reply, reply_sha = excluded.reply_sha,"
                           " bytes = excluded.bytes, used_at = excluded.used_at",
                           (owner_id, key, call, reply, sha, len(data), now, now))
                db.execute(_USE, (owner_id, doc_id, key, now))
                return self._evict(db)

            return self._write(db, write)

        return self._run(store, create=True)

    # --- purges ---------------------------------------------------------------------------------

    def _purge(self, fn: Callable[[sqlite3.Connection], int], what: str) -> int:
        """A purge in one transaction, then the WAL truncated when something went. No file: nothing to purge. A
        damaged file is deleted, which purges it too."""
        def run(db):
            removed = self._write(db, lambda: fn(db))
            if removed:
                truncate_wal(db, log, purge=what)
            return removed

        try:
            return self._run(run, create=False) or 0
        except CacheReset:
            return 0

    def release_document(self, owner_id: str, doc_id: str) -> int:
        """A deleted document: its uses, and every entry of its owner that no other document uses. Returns how
        many rows went. Raises on a failure, for the caller to retry."""
        _check(owner_id, doc_id)

        def purge(db):
            uses = db.execute("DELETE FROM uses WHERE owner_id = ? AND doc_id = ?", (owner_id, doc_id)).rowcount
            return uses + db.execute(f"DELETE FROM entries WHERE owner_id = ? AND {_UNUSED}", (owner_id,)).rowcount

        return self._purge(purge, "document")

    def purge_owner(self, owner_id: str) -> int:
        """A deleted account: every entry and use of the owner. Returns how many rows went. Raises on a failure,
        for the caller to retry."""
        _check(owner_id)

        def purge(db):
            entries = db.execute("DELETE FROM entries WHERE owner_id = ?", (owner_id,)).rowcount
            return entries + db.execute("DELETE FROM uses WHERE owner_id = ?", (owner_id,)).rowcount

        return self._purge(purge, "owner")

    def collect(self, known: Callable[[], set[tuple[str, str]]]) -> int:
        """The sweep: remove the uses of documents the registry no longer has, then entries no document uses and
        uses without an entry. known() gives every (owner_id, doc_id) of the registry; it is read after the uses
        and only uses written before that are judged, so a use belongs to a row that existed when it was written
        and a row missing afterwards was deleted (a new upload of the same file writes newer uses, which stay).
        Returns how many rows went."""
        cutoff = self.clock()

        def users(db):
            return set(db.execute("SELECT DISTINCT owner_id, doc_id FROM uses WHERE used_at <= ?", (cutoff,)))

        try:
            seen = self._run(users, create=False)
        except CacheReset:
            return 0
        if seen is None:
            return 0
        stale = sorted(seen - known())

        def purge(db):
            removed = 0
            for owner_id, doc_id in stale:
                removed += db.execute("DELETE FROM uses WHERE owner_id = ? AND doc_id = ? AND used_at <= ?",
                                      (owner_id, doc_id, cutoff)).rowcount
            removed += db.execute(f"DELETE FROM entries WHERE {_UNUSED}").rowcount
            return removed + db.execute(f"DELETE FROM uses WHERE {_NO_ENTRY}").rowcount

        return self._purge(purge, "sweep")


class JobCache:
    """One job's use of the cache: its owner, document and model identity, and the counts for the job's log line.
    Lookups and writes run in a thread (run: Pipeline.in_thread, so the clean-up after a cancelled run waits for
    a write still going). Never raises: an error is a miss or a skipped write, logged at WARNING, and after
    max_errors errors the job stops using the cache."""

    def __init__(self, store: LlmCache, owner_id: str, doc_id: str, model: str, *,
                 run: Callable[..., Awaitable] = asyncio.to_thread, max_errors: int = 3):
        _check(owner_id, doc_id)
        self.store, self.owner, self.doc_id, self.model = store, owner_id, doc_id, model
        self.run = run
        self.max_errors = max_errors
        self.counts: Counter[str] = Counter()
        self.off = False

    def key(self, call: str, version: int, request: dict, retry: dict) -> str:
        return cache_key(owner=self.owner, call=call, version=version, model=self.model, request=request, retry=retry)

    def _event(self, event: str, call: str, key: str, level: int = logging.DEBUG, **fields) -> None:
        log_fields(log, f"llm cache {event}", level, doc_id=self.doc_id, call=call, key=key[:12], **fields)

    def _error(self, what: str, call: str, exc: BaseException) -> None:
        self.counts["errors"] += 1
        log_fields(log, f"llm cache {what} failed; the model reply is used as it is", logging.WARNING,
                   doc_id=self.doc_id, call=call, error=f"{type(exc).__name__}: {exc}"[:300])
        if self.counts["errors"] >= self.max_errors and not self.off:
            self.off = True
            log_fields(log, "llm cache off for the rest of this job after repeated errors", logging.WARNING,
                       doc_id=self.doc_id, errors=self.counts["errors"], limit="INGEST_LLM_CACHE_MAX_ERRORS")

    async def get(self, call: str, key: str) -> str | None:
        """The stored reply, or None: ask the model."""
        if self.off:
            return None
        try:
            reply, corrupt = await self.run(self.store.get, self.owner, key, self.doc_id, call)
        except Exception as exc:  # noqa: BLE001 - any cache failure is a miss
            self._error("lookup", call, exc)
            return None
        if corrupt:
            self.counts["corrupt"] += 1
            self._event("entry was corrupt; removed, asking the model", call, key, logging.WARNING)
        if reply is None:
            self.counts["misses"] += 1
            self._event("miss", call, key)
            return None
        self.counts["hits"] += 1
        self._event("hit", call, key)
        return reply

    async def offer(self, call: str, key: str, reply: str, finish: Any) -> bool:
        """Store the model's reply when it is whole (skip_reason); True when stored."""
        if reason := skip_reason(reply, finish):
            self.skip(call, reason, key)
            return False
        if self.off:
            return False
        try:
            evicted = await self.run(self.store.put, self.owner, key, self.doc_id, call, reply)
        except Exception as exc:  # noqa: BLE001 - a failed write leaves the reply uncached
            self._error("write", call, exc)
            return False
        self.counts["writes"] += 1
        self.counts["evicted"] += evicted
        self._event("write", call, key, evicted=evicted)
        return True

    def skip(self, call: str, reason: str, key: str = "") -> None:
        """A reply that is not stored: truncated, failed (the call raised), not finished, empty or refused."""
        self.counts["skipped"] += 1
        self.counts[f"skipped_{reason}"] += 1
        self._event("skip: reply not stored", call, key, logging.INFO, reason=reason)

    def summary(self) -> dict[str, int | bool]:
        c = self.counts
        out: dict[str, int | bool] = {name: c[name] for name in ("hits", "misses", "writes", "skipped", "errors")}
        out |= {name: n for name, n in sorted(c.items())
                if n and (name.startswith("skipped_") or name in ("corrupt", "evicted"))}
        return out | ({"off_after_errors": True} if self.off else {})

    def log_summary(self) -> None:
        """The job's line: hits are model calls saved."""
        log_fields(log, "llm cache summary", doc_id=self.doc_id, **self.summary())


async def model_identity(settings, client: httpx.AsyncClient) -> str | None:
    """The model part of every key: the name the server answers to (INGEST_VISION_MODEL), its weights file
    (INGEST_VISION_MODEL_FILE; the name carries the quantization), its vision projector (INGEST_VISION_MMPROJ),
    the server's build and INGEST_LLM_CACHE_SALT. llama-server's /props names the file it loaded and its build:
    when it names another file, or when nothing names one, the cache stays off (None, logged), so replies of one
    model are never served as another's. When /props cannot be read (one retry for a busy or unreachable
    server), the configured file is used, with no build, and a warning."""

    async def ask() -> dict:
        response = await client.get("/props", timeout=30)
        response.raise_for_status()
        return response.json()

    reported, build = None, ""
    try:
        props = await retry_once(ask, what="asking the vision server for its model file", log=log,
                                 backoff_s=settings.retry_backoff_s)
        reported = PurePosixPath(str(props.get("model_path") or "").replace("\\", "/")).name or None
        build = str(props.get("build_info") or "")
    except Exception as exc:  # noqa: BLE001 - another OpenAI-compatible server may have no /props
        log_fields(log, "could not ask the vision server which model file it loaded", logging.WARNING,
                   error=f"{type(exc).__name__}: {exc}"[:300])
    configured = settings.vision_model_file
    if configured and reported and configured != reported:
        log_fields(log, "the vision server runs another model file than INGEST_VISION_MODEL_FILE; llm cache off",
                   logging.ERROR, configured=configured, reported=reported)
        return None
    file = configured or reported
    if not file:
        log_fields(log, "the vision model's file is unknown (INGEST_VISION_MODEL_FILE unset and /props did not name "
                        "it); llm cache off", logging.WARNING)
        return None
    return json.dumps({"model": settings.vision_model, "file": file, "mmproj": settings.vision_mmproj,
                       "server": build, "salt": settings.llm_cache_salt}, sort_keys=True)
