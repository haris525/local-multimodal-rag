"""Conversations, their running summaries, long-term memories and custom search tasks in SQLite (WAL), per user.

Messages are never deleted when a conversation is compacted: the summary only replaces older turns in
what the model sees, the full history stays here.

Every thread, memory and search task belongs to one user, and every query has the owner in its WHERE clause, so
another user's thread, memory or search task reads exactly like a missing one. Messages have no owner column: each message read
or write checks the thread's owner on the same connection first. Rows written before accounts existed have
no owner and are visible to nobody until adopt_legacy gives them one.

Deleted accounts (schema v2) leave a tombstone in deleted_users, written in the same transaction that deletes
their rows. A chat authorized before the deletion can still be running (rag signs a fresh identity for each
call while it lasts), so every write that creates something (a thread, a message, a memory) checks the
tombstone in its own transaction and raises OwnerDeleted: nothing can be created for a purged account.

Custom search tasks (schema v3) are a user's own instructions for embedding the question (rag.search_tasks): a
name and an instruction, at most THREADS_MAX_SEARCH_TASKS per user. rag reads one by id for the user in the
chat request's identity, so no user can search with another user's task.
"""

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from rag_common.contracts import (
    MemoryItem,
    NewMessage,
    SearchTaskItem,
    ThreadContext,
    ThreadDetail,
    ThreadInfo,
    ThreadMessage,
)
from rag_common.identity import USER_ID_RE
from rag_common.logging import log_fields
from rag_common.secrets import redact_secrets
from rag_common.sqlite import truncate_wal

log = logging.getLogger("threads.store")

SCHEMA_VERSION = 3
_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS threads (
        thread_id    TEXT PRIMARY KEY,
        user_id      TEXT,
        title        TEXT NOT NULL,
        doc_ids      TEXT NOT NULL DEFAULT '[]',
        summary      TEXT NOT NULL DEFAULT '',
        summary_upto INTEGER NOT NULL DEFAULT 0,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS threads_user ON threads (user_id, updated_at)",
    """CREATE TABLE IF NOT EXISTS messages (
        seq        INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id  TEXT NOT NULL REFERENCES threads (thread_id) ON DELETE CASCADE,
        role       TEXT NOT NULL,
        content    TEXT NOT NULL,
        meta       TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS messages_thread ON messages (thread_id, seq)",
    """CREATE TABLE IF NOT EXISTS memories (
        memory_id  TEXT PRIMARY KEY,
        user_id    TEXT,
        text       TEXT NOT NULL,
        norm       TEXT NOT NULL,
        source     TEXT NOT NULL,
        thread_id  TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (user_id, norm)
    )""",
    "CREATE INDEX IF NOT EXISTS memories_user ON memories (user_id)",
    "CREATE TABLE IF NOT EXISTS deleted_users (user_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL)",  # v2
    """CREATE TABLE IF NOT EXISTS search_tasks (
        task_id     TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        name        TEXT NOT NULL,
        instruction TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )""",  # v3
    "CREATE INDEX IF NOT EXISTS search_tasks_user ON search_tasks (user_id)",
]
# v0 -> v1: memories were unique across all users; rebuilt with a per-user unique key, order kept.
_MEMORIES_V1 = [
    """CREATE TABLE memories_v1 (
        memory_id  TEXT PRIMARY KEY,
        user_id    TEXT,
        text       TEXT NOT NULL,
        norm       TEXT NOT NULL,
        source     TEXT NOT NULL,
        thread_id  TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (user_id, norm)
    )""",
    """INSERT INTO memories_v1 (memory_id, user_id, text, norm, source, thread_id, created_at)
       SELECT memory_id, NULL, text, norm, source, thread_id, created_at FROM memories ORDER BY rowid""",
    "DROP TABLE memories",
    "ALTER TABLE memories_v1 RENAME TO memories",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


def _owner(user_id: str) -> str:
    """Fail closed: a missing or malformed owner never reaches a query (NULL would mean 'legacy')."""
    if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
        raise ValueError("user_id is not a user id")
    return user_id


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


class OwnerDeleted(Exception):
    """The user's account was deleted: nothing may be stored for it any more."""


def _refuse_deleted(db: sqlite3.Connection, user_id: str) -> None:
    """Inside a write transaction: roll back and raise when the user's account was deleted."""
    if db.execute("SELECT 1 FROM deleted_users WHERE user_id = ?", (user_id,)).fetchone():
        db.execute("ROLLBACK")
        raise OwnerDeleted(user_id)


class Store:
    def __init__(self, path: Path, max_memories: int = 50, max_search_tasks: int = 50):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self.max_memories = max_memories  # per user
        self.max_search_tasks = max_search_tasks  # per user
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            self._upgrade(db)
            self._sweep_deleted_accounts(db)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _upgrade(db: sqlite3.Connection) -> None:
        """Bring the schema to SCHEMA_VERSION in one transaction (additive; a second run does nothing).
        v1 added the owners (and rebuilt memories); v2 adds deleted_users; v3 adds search_tasks.
        Foreign keys are off during the memories rebuild; the pragma only takes effect outside a transaction."""
        if db.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return
        db.execute("PRAGMA foreign_keys=OFF")
        try:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:  # another process may have won
                    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                    if "threads" in tables and "user_id" not in _columns(db, "threads"):
                        db.execute("ALTER TABLE threads ADD COLUMN user_id TEXT")
                    if "memories" in tables and "user_id" not in _columns(db, "memories"):
                        for statement in _MEMORIES_V1:
                            db.execute(statement)
                    for statement in _SCHEMA:
                        db.execute(statement)
                    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        finally:
            db.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _sweep_deleted_accounts(db: sqlite3.Connection) -> None:
        """At start: delete the search tasks of accounts already deleted. A threads from before schema v3 (after a
        rollback) purges an account without them; this finishes that purge, the same way (zeroed pages, then the
        WAL truncated). A second start finds nothing."""
        if not db.execute("SELECT 1 FROM search_tasks WHERE user_id IN (SELECT user_id FROM deleted_users)"
                          " LIMIT 1").fetchone():
            return
        db.execute("PRAGMA secure_delete=ON")
        db.execute("BEGIN IMMEDIATE")
        swept = db.execute("DELETE FROM search_tasks WHERE user_id IN (SELECT user_id FROM deleted_users)").rowcount
        db.execute("COMMIT")
        truncate_wal(db, log, purge="deleted accounts' search tasks")
        log_fields(log, "deleted the search tasks of accounts deleted before", search_tasks=swept)

    # --- threads ----------------------------------------------------------------------------

    def _info(self, db: sqlite3.Connection, row: sqlite3.Row) -> ThreadInfo:
        count, summarized = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(seq <= ?), 0) FROM messages WHERE thread_id = ?",
            (row["summary_upto"], row["thread_id"]),
        ).fetchone()
        return ThreadInfo(thread_id=row["thread_id"], title=row["title"], doc_ids=json.loads(row["doc_ids"]),
                          message_count=count, summarized=summarized, created_at=row["created_at"],
                          updated_at=row["updated_at"])

    @staticmethod
    def _thread(db: sqlite3.Connection, user_id: str, thread_id: str) -> sqlite3.Row | None:
        return db.execute("SELECT * FROM threads WHERE thread_id = ? AND user_id = ?", (thread_id, user_id)).fetchone()

    def create(self, user_id: str, title: str, doc_ids: list[str]) -> ThreadInfo:
        thread_id, now = uuid.uuid4().hex, _now()
        _owner(user_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _refuse_deleted(db, user_id)
            db.execute("INSERT INTO threads (thread_id, user_id, title, doc_ids, created_at, updated_at)"
                       " VALUES (?, ?, ?, ?, ?, ?)",
                       (thread_id, user_id, redact_secrets(title.strip()), json.dumps(doc_ids), now, now))
            db.execute("COMMIT")
            return self._info(db, self._thread(db, user_id, thread_id))

    def recent(self, user_id: str, limit: int) -> list[ThreadInfo]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM threads WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                              (_owner(user_id), limit)).fetchall()
            return [self._info(db, r) for r in rows]

    def get(self, user_id: str, thread_id: str) -> ThreadDetail | None:
        with self._connect() as db:
            row = self._thread(db, _owner(user_id), thread_id)
            if row is None:
                return None
            msgs = db.execute("SELECT * FROM messages WHERE thread_id = ? ORDER BY seq", (thread_id,)).fetchall()
            return ThreadDetail(thread=self._info(db, row), messages=[self._message(m) for m in msgs])

    def update(self, user_id: str, thread_id: str, title: str | None, doc_ids: list[str] | None) -> ThreadInfo | None:
        _owner(user_id)
        with self._connect() as db:
            if title is not None:
                db.execute("UPDATE threads SET title = ? WHERE thread_id = ? AND user_id = ?",
                           (redact_secrets(title.strip()), thread_id, user_id))
            if doc_ids is not None:
                db.execute("UPDATE threads SET doc_ids = ? WHERE thread_id = ? AND user_id = ?",
                           (json.dumps(doc_ids), thread_id, user_id))
            row = self._thread(db, user_id, thread_id)
            return self._info(db, row) if row else None

    def delete(self, user_id: str, thread_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM threads WHERE thread_id = ? AND user_id = ?",
                              (thread_id, _owner(user_id))).rowcount == 1

    # --- messages and compaction ------------------------------------------------------------

    @staticmethod
    def _message(row: sqlite3.Row) -> ThreadMessage:
        return ThreadMessage(seq=row["seq"], role=row["role"], content=row["content"], meta=json.loads(row["meta"]),
                             created_at=row["created_at"])

    def append(self, user_id: str, thread_id: str, messages: list[NewMessage],
               doc_ids: list[str] | None) -> list[ThreadMessage] | None:
        now = _now()
        _owner(user_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _refuse_deleted(db, user_id)
            if self._thread(db, user_id, thread_id) is None:
                db.execute("ROLLBACK")
                return None
            seqs = [
                db.execute("INSERT INTO messages (thread_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                           (thread_id, m.role, m.content, json.dumps(m.meta, ensure_ascii=False), now)).lastrowid
                for m in messages
            ]
            db.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ? AND user_id = ?", (now, thread_id, user_id))
            if doc_ids is not None:
                db.execute("UPDATE threads SET doc_ids = ? WHERE thread_id = ? AND user_id = ?",
                           (json.dumps(doc_ids), thread_id, user_id))
            db.execute("COMMIT")
            marks = ",".join("?" * len(seqs))
            return [self._message(r) for r in db.execute(f"SELECT * FROM messages WHERE seq IN ({marks}) ORDER BY seq", seqs)]

    def context(self, user_id: str, thread_id: str) -> ThreadContext | None:
        with self._connect() as db:
            row = self._thread(db, _owner(user_id), thread_id)
            if row is None:
                return None
            msgs = db.execute("SELECT * FROM messages WHERE thread_id = ? AND seq > ? ORDER BY seq",
                              (thread_id, row["summary_upto"])).fetchall()
            return ThreadContext(summary=row["summary"], summary_upto=row["summary_upto"],
                                 messages=[self._message(m) for m in msgs])

    def set_summary(self, user_id: str, thread_id: str, summary: str, upto: int) -> bool:
        """Move the summary forward only (a late, stale compaction cannot overwrite a newer one)."""
        with self._connect() as db:
            return db.execute("UPDATE threads SET summary = ?, summary_upto = ?"
                              " WHERE thread_id = ? AND user_id = ? AND summary_upto < ?",
                              (summary, upto, thread_id, _owner(user_id), upto)).rowcount == 1

    # --- long-term memories -----------------------------------------------------------------

    @staticmethod
    def _memory(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(memory_id=row["memory_id"], text=row["text"], source=row["source"], thread_id=row["thread_id"],
                          created_at=row["created_at"])

    def memories(self, user_id: str) -> list[MemoryItem]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM memories WHERE user_id = ? ORDER BY rowid", (_owner(user_id),)).fetchall()
        return [self._memory(r) for r in rows]

    def add_memory(self, user_id: str, text: str, source: str, thread_id: str | None) -> MemoryItem | None:
        """Store a memory unless the user already has an identical one; at the user's cap, their oldest automatic
        memory makes room (memories the user typed are never evicted). A thread_id that is not one of the user's
        threads is dropped, so a memory never points at someone else's chat. OwnerDeleted for a deleted account."""
        _owner(user_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _refuse_deleted(db, user_id)
            if thread_id and self._thread(db, user_id, thread_id) is None:
                thread_id = None
            if db.execute("SELECT 1 FROM memories WHERE user_id = ? AND norm = ?", (user_id, _norm(text))).fetchone():
                db.execute("ROLLBACK")
                return None
            if db.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)).fetchone()[0] >= self.max_memories:
                oldest = db.execute("SELECT memory_id FROM memories WHERE user_id = ? AND source = 'auto'"
                                    " ORDER BY rowid LIMIT 1", (user_id,)).fetchone()
                if oldest is None:
                    db.execute("ROLLBACK")
                    raise ValueError(f"memory is full ({self.max_memories}); delete some first")
                db.execute("DELETE FROM memories WHERE memory_id = ? AND user_id = ?", (oldest[0], user_id))
            item = MemoryItem(memory_id=uuid.uuid4().hex, text=text.strip(), source=source, thread_id=thread_id,
                              created_at=_now())
            db.execute("INSERT INTO memories (memory_id, user_id, text, norm, source, thread_id, created_at)"
                       " VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (item.memory_id, user_id, item.text, _norm(item.text), source, thread_id, item.created_at))
            db.execute("COMMIT")
        return item

    def replace_memory(self, user_id: str, memory_id: str, text: str, source: str) -> tuple[MemoryItem, str] | None:
        """Rewrite one memory in place, in one transaction: it keeps its id and position, and the old fact
        is never lost when the new text is rejected. Returns (new item, old text); None when another memory
        of the user already says the same; KeyError when the memory is gone (or not the user's). A typed memory
        stays typed."""
        _owner(user_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM memories WHERE memory_id = ? AND user_id = ?", (memory_id, user_id)).fetchone()
            if row is None:
                db.execute("ROLLBACK")
                raise KeyError(memory_id)
            if db.execute("SELECT 1 FROM memories WHERE user_id = ? AND norm = ? AND memory_id != ?",
                          (user_id, _norm(text), memory_id)).fetchone():
                db.execute("ROLLBACK")
                return None
            source = "user" if "user" in (source, row["source"]) else "auto"
            db.execute("UPDATE memories SET text = ?, norm = ?, source = ? WHERE memory_id = ? AND user_id = ?",
                       (text.strip(), _norm(text), source, memory_id, user_id))
            db.execute("COMMIT")
        item = MemoryItem(memory_id=memory_id, text=text.strip(), source=source, thread_id=row["thread_id"],
                          created_at=row["created_at"])
        return item, row["text"]

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM memories WHERE memory_id = ? AND user_id = ?",
                              (memory_id, _owner(user_id))).rowcount == 1

    # --- custom search tasks ----------------------------------------------------------------

    @staticmethod
    def _search_task(row: sqlite3.Row) -> SearchTaskItem:
        return SearchTaskItem(task_id=row["task_id"], name=row["name"], instruction=row["instruction"],
                              created_at=row["created_at"], updated_at=row["updated_at"])

    def search_tasks(self, user_id: str) -> list[SearchTaskItem]:
        """The user's search tasks, oldest first."""
        with self._connect() as db:
            rows = db.execute("SELECT * FROM search_tasks WHERE user_id = ? ORDER BY rowid", (_owner(user_id),)).fetchall()
        return [self._search_task(r) for r in rows]

    def search_task(self, user_id: str, task_id: str) -> SearchTaskItem | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM search_tasks WHERE task_id = ? AND user_id = ?",
                             (task_id, _owner(user_id))).fetchone()
        return self._search_task(row) if row else None

    def add_search_task(self, user_id: str, name: str, instruction: str) -> SearchTaskItem:
        """Store a new search task (name and instruction already checked: SearchTaskWrite). ValueError at the user's
        limit (max_search_tasks); OwnerDeleted for a deleted account."""
        _owner(user_id)
        now = _now()
        item = SearchTaskItem(task_id=uuid.uuid4().hex, name=name, instruction=instruction, created_at=now,
                              updated_at=now)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _refuse_deleted(db, user_id)
            count = db.execute("SELECT COUNT(*) FROM search_tasks WHERE user_id = ?", (user_id,)).fetchone()[0]
            if count >= self.max_search_tasks:
                db.execute("ROLLBACK")
                raise ValueError(f"you have {count} search tasks, the most allowed ({self.max_search_tasks}); "
                                 "delete one first")
            db.execute("INSERT INTO search_tasks (task_id, user_id, name, instruction, created_at, updated_at)"
                       " VALUES (?, ?, ?, ?, ?, ?)", (item.task_id, user_id, name, instruction, now, now))
            db.execute("COMMIT")
        return item

    def update_search_task(self, user_id: str, task_id: str, name: str, instruction: str) -> SearchTaskItem | None:
        """Rewrite a search task in place (it keeps its id, so chats that used it ask with it again); None when it is
        gone or not the user's. One statement returns the row it wrote, so a delete at the same moment cannot
        leave it half done."""
        with self._connect() as db:
            rows = db.execute("UPDATE search_tasks SET name = ?, instruction = ?, updated_at = ?"
                              " WHERE task_id = ? AND user_id = ? RETURNING *",
                              (name, instruction, _now(), task_id, _owner(user_id))).fetchall()
        return self._search_task(rows[0]) if rows else None

    def delete_search_task(self, user_id: str, task_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM search_tasks WHERE task_id = ? AND user_id = ?",
                              (task_id, _owner(user_id))).rowcount == 1

    # --- accounts ---------------------------------------------------------------------------

    def purge_user(self, user_id: str) -> dict:
        """Delete every thread (messages cascade), memory and search task of a user, and tombstone the account in
        the same transaction, so a chat still running cannot write anything for it afterwards. Freed pages are
        zeroed and the WAL is truncated, so the text does not linger in the file. Repeatable: a second run deletes
        nothing."""
        _owner(user_id)
        with self._connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO deleted_users (user_id, deleted_at) VALUES (?, ?)", (user_id, _now()))
            messages = db.execute("SELECT COUNT(*) FROM messages WHERE thread_id IN"
                                  " (SELECT thread_id FROM threads WHERE user_id = ?)", (user_id,)).fetchone()[0]
            threads = db.execute("DELETE FROM threads WHERE user_id = ?", (user_id,)).rowcount
            memories = db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,)).rowcount
            search_tasks = db.execute("DELETE FROM search_tasks WHERE user_id = ?", (user_id,)).rowcount
            db.execute("COMMIT")
            truncate_wal(db, log, purge="user")
        return {"threads": threads, "messages": messages, "memories": memories, "search_tasks": search_tasks}

    def adopt_legacy(self, user_id: str, dry_run: bool = False) -> dict:
        """Give the threads and memories from before accounts existed to one user. A legacy memory that says what
        one of the user's memories already says is left unowned (invisible), never merged. Repeatable: a second
        run adopts nothing. OwnerDeleted for a deleted account."""
        _owner(user_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _refuse_deleted(db, user_id)
            threads = db.execute("SELECT COUNT(*) FROM threads WHERE user_id IS NULL").fetchone()[0]
            memories = db.execute("SELECT COUNT(*) FROM memories WHERE user_id IS NULL").fetchone()[0]
            if dry_run:
                db.execute("ROLLBACK")
                return {"threads": threads, "memories": memories, "memories_skipped": 0, "dry_run": True}
            db.execute("UPDATE threads SET user_id = ? WHERE user_id IS NULL", (user_id,))
            adopted = db.execute("UPDATE OR IGNORE memories SET user_id = ? WHERE user_id IS NULL", (user_id,)).rowcount
            db.execute("COMMIT")
        return {"threads": threads, "memories": adopted, "memories_skipped": memories - adopted, "dry_run": False}
