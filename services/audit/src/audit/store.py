"""Append-only, hash-chained audit log in SQLite, per user.

Each event stores sha256(previous hash + canonical event JSON). Triggers reject UPDATE and DELETE, and
verify() recomputes the chain, so an edited or removed record is detectable.

Accounts (schema v1):
- Every event belongs to one user (user_id) and has a per-user number (user_seq). A user reads only their own
  events, numbered 1, 2, 3 ... in their own sequence; the global seq, prev_hash and request_id are never shown
  to users, so nobody learns how active anyone else is.
- Rows written before accounts existed have user_id 'legacy' and keep the v1 hash over _FIELDS. Every newer row
  uses the v2 hash, which also covers user_id and user_seq ("v": 2 in the canonical JSON). Re-labelling a row
  as legacy (or as another user) changes what is hashed, so verify() reports it. The legacy rows belong to the
  user named in meta.legacy_owner (set once, by the legacy migration). When that user is purged, the value
  becomes a tombstone (LEGACY_OWNER_DELETED): nobody can read or adopt the legacy rows again. They cannot be
  crypto-shredded: they predate the per-user keys and are plain text inside the chain, which keeps every row.
- The detail of every newer row is encrypted with a per-user key (AES-256-GCM, the user id, time and action as
  associated data). Hashes cover the stored text, so verify() needs no keys. Deleting a user's key
  (purge_user) makes their details unreadable for good (crypto-shredding) while the chain stays intact; the
  key row is replaced by an empty tombstone, so a late event for that user is stored without its detail.
"""

import base64
import hashlib
import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from rag_common.sqlite import truncate_wal

log = logging.getLogger("audit.store")
GENESIS = "0" * 64
SCHEMA_VERSION = 1
LEGACY = "legacy"
ENCRYPTED = "enc1:"
SHREDDED = {"_shredded": True}
LEGACY_OWNER_DELETED = "deleted"  # meta.legacy_owner after the adopting account was deleted
_FIELDS = ("time", "service", "action", "actor", "request_id", "subject", "detail")
V2_FIELDS = (*_FIELDS, "user_id", "user_seq")
_TABLES = [
    """CREATE TABLE IF NOT EXISTS events (
        seq        INTEGER PRIMARY KEY AUTOINCREMENT,
        time       TEXT NOT NULL,
        service    TEXT NOT NULL,
        action     TEXT NOT NULL,
        actor      TEXT NOT NULL,
        request_id TEXT,
        subject    TEXT,
        detail     TEXT NOT NULL,
        prev_hash  TEXT NOT NULL,
        hash       TEXT NOT NULL,
        user_id    TEXT NOT NULL DEFAULT 'legacy',
        user_seq   INTEGER
    )""",
]
_SCHEMA = [
    "CREATE INDEX IF NOT EXISTS events_action ON events (action, seq)",
    "CREATE INDEX IF NOT EXISTS events_user ON events (user_id, user_seq)",
    """CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
        BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
        BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END""",
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS user_keys (user_id TEXT PRIMARY KEY, dek BLOB NOT NULL, created_at TEXT NOT NULL)",
]
# v0 -> v1. Schema changes only: the UPDATE trigger does not fire, existing rows read as 'legacy'.
_V0_TO_V1 = [
    "ALTER TABLE events ADD COLUMN user_id TEXT NOT NULL DEFAULT 'legacy'",
    "ALTER TABLE events ADD COLUMN user_seq INTEGER",
]


class Conflict(Exception):
    """The legacy rows already belong to another user."""


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(prev_hash: str, row: dict) -> str:
    """v1 for legacy rows (the original fields), v2 for every other row."""
    if row["user_id"] == LEGACY:
        body = {k: row[k] for k in _FIELDS}
    else:
        body = {k: row[k] for k in V2_FIELDS} | {"v": 2}
    return hashlib.sha256((prev_hash + _canonical(body)).encode()).hexdigest()


def _aad(user_id: str, time: str, action: str) -> bytes:
    return f"{user_id}|{time}|{action}".encode()


def _encrypt(dek: bytes, plain: str, user_id: str, time: str, action: str) -> str:
    nonce = os.urandom(12)
    sealed = AESGCM(dek).encrypt(nonce, plain.encode(), _aad(user_id, time, action))
    return ENCRYPTED + base64.b64encode(nonce + sealed).decode("ascii")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class AuditStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            self._upgrade(db)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One connection per call, always closed (sqlite3's own `with` only commits)."""
        db = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _upgrade(db: sqlite3.Connection) -> None:
        """Bring the schema to SCHEMA_VERSION in one transaction; a second run does nothing."""
        if db.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return
        db.execute("BEGIN IMMEDIATE")
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                columns = {r[1] for r in db.execute("PRAGMA table_info(events)")}
                if columns and "user_id" not in columns:
                    for statement in _V0_TO_V1:
                        db.execute(statement)
                for statement in _TABLES + _SCHEMA:
                    db.execute(statement)
                db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise

    # --- helpers ----------------------------------------------------------------------------

    @staticmethod
    def _legacy_owner(db: sqlite3.Connection) -> str | None:
        row = db.execute("SELECT value FROM meta WHERE key = 'legacy_owner'").fetchone()
        return row[0] if row else None

    @staticmethod
    def _keys(db: sqlite3.Connection, user_ids: set[str]) -> dict[str, bytes]:
        if not user_ids:
            return {}
        marks = ",".join("?" * len(user_ids))
        return {r[0]: r[1] for r in db.execute(f"SELECT user_id, dek FROM user_keys WHERE user_id IN ({marks})",
                                               list(user_ids))}

    @staticmethod
    def _detail(row: sqlite3.Row, keys: dict[str, bytes]) -> dict:
        text = row["detail"]
        if not text.startswith(ENCRYPTED):
            return json.loads(text)
        dek = keys.get(row["user_id"])
        if not dek:  # deleted (purged user): the detail is gone for good
            return dict(SHREDDED)
        raw = base64.b64decode(text[len(ENCRYPTED):])
        try:
            plain = AESGCM(dek).decrypt(raw[:12], raw[12:], _aad(row["user_id"], row["time"], row["action"]))
        except InvalidTag:  # altered ciphertext or associated data; verify() reports the row
            return {"_unreadable": True}
        return json.loads(plain)

    @staticmethod
    def _visible(user_id: str, owner: str | None) -> tuple[str, list]:
        """WHERE clause for the rows a user may read: their own, plus the legacy rows if they adopted them."""
        if owner is not None and owner == user_id:
            return "(user_id = ? OR user_id = ?)", [user_id, LEGACY]
        return "user_id = ?", [user_id]

    # --- writing ----------------------------------------------------------------------------

    def append(self, service: str, action: str, actor: str, request_id: str | None, subject: str | None,
               detail: dict, user_id: str) -> dict:
        """Returns {"seq", "user_seq", "hash"}. user_seq numbers the rows this user can read (their legacy rows
        included once adopted), so it continues the sequence they see."""
        if not user_id or user_id == LEGACY:
            raise ValueError("new events need a user; 'legacy' is reserved for rows from before accounts")
        row = {
            "time": _now(),
            "service": service,
            "action": action,
            "actor": actor,
            "request_id": request_id,
            "subject": subject,
            "user_id": user_id,
        }
        plain = json.dumps(detail, sort_keys=True, ensure_ascii=False, default=str)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                key = db.execute("SELECT dek FROM user_keys WHERE user_id = ?", (user_id,)).fetchone()
                if key is None:
                    dek = AESGCM.generate_key(bit_length=256)
                    db.execute("INSERT INTO user_keys (user_id, dek, created_at) VALUES (?, ?, ?)",
                               (user_id, dek, row["time"]))
                else:
                    dek = key[0]
                row["detail"] = (_encrypt(dek, plain, user_id, row["time"], action) if dek
                                 else json.dumps(SHREDDED))  # tombstone: the user was purged
                where, args = self._visible(user_id, self._legacy_owner(db))
                row["user_seq"] = db.execute(f"SELECT COUNT(*) FROM events WHERE {where}", args).fetchone()[0] + 1
                last = db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
                row["prev_hash"] = last["hash"] if last else GENESIS
                row["hash"] = _digest(row["prev_hash"], row)
                cur = db.execute(
                    "INSERT INTO events (time, service, action, actor, request_id, subject, detail, prev_hash, hash,"
                    " user_id, user_seq) VALUES (:time, :service, :action, :actor, :request_id, :subject, :detail,"
                    " :prev_hash, :hash, :user_id, :user_seq)",
                    row,
                )
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return {"seq": cur.lastrowid, "user_seq": row["user_seq"], "hash": row["hash"]}

    # --- reading ----------------------------------------------------------------------------

    def list_for(self, user_id: str, limit: int, before: int | None = None, action: str | None = None,
                 subject: str | None = None) -> list[dict]:
        """A user's own events, newest first. user_seq is the event's number in the user's sequence (legacy rows
        first, once adopted); `before` pages by it. No global seq, prev_hash or request_id."""
        with self._connect() as db:
            visible, args = self._visible(user_id, self._legacy_owner(db))
            where, extra = [], []
            if before:
                where.append("user_seq < ?")
                extra.append(before)
            if action:
                where.append("action LIKE ?")
                extra.append(action.replace("*", "%"))
            if subject:
                where.append("subject LIKE ?")
                extra.append(f"%{subject}%")
            sql = (f"WITH mine AS (SELECT seq, time, service, action, subject, detail, hash, user_id,"
                   f" ROW_NUMBER() OVER (ORDER BY seq) AS user_seq FROM events WHERE {visible})"
                   f" SELECT * FROM mine{' WHERE ' + ' AND '.join(where) if where else ''} ORDER BY seq DESC LIMIT ?")
            rows = db.execute(sql, [*args, *extra, limit]).fetchall()
            keys = self._keys(db, {r["user_id"] for r in rows})
        return [{"user_seq": r["user_seq"], "time": r["time"], "service": r["service"], "action": r["action"],
                 "subject": r["subject"], "detail": self._detail(r, keys), "hash": r["hash"]} for r in rows]

    def _check(self, db: sqlite3.Connection) -> dict:
        """Recompute every hash in order (v1 or v2 per row); report the first record that does not match."""
        prev, count = GENESIS, 0
        for row in db.execute("SELECT * FROM events ORDER BY seq"):
            row = dict(row)
            if row["prev_hash"] != prev or _digest(prev, row) != row["hash"]:
                return {"ok": False, "events": count, "first_bad_seq": row["seq"], "head": None}
            prev, count = row["hash"], count + 1
        return {"ok": True, "events": count, "first_bad_seq": None, "head": prev}

    def verify(self) -> dict:
        """The whole chain: {"ok", "events", "first_bad_seq", "head"}. For admins (no details involved)."""
        with self._connect() as db:
            return self._check(db)

    def verify_for(self, user_id: str) -> dict:
        """{"ok": the whole chain verifies, "events": the user's own count, "first_bad": the user's number of the
        first altered event when it is one of theirs, else None}. Nothing about other users' rows is revealed
        beyond the fact that the log was altered somewhere."""
        with self._connect() as db:
            chain = self._check(db)
            visible, args = self._visible(user_id, self._legacy_owner(db))
            events = db.execute(f"SELECT COUNT(*) FROM events WHERE {visible}", args).fetchone()[0]
            first_bad = None
            if not chain["ok"]:
                bad = chain["first_bad_seq"]
                if db.execute(f"SELECT 1 FROM events WHERE seq = ? AND {visible}", [bad, *args]).fetchone():
                    first_bad = db.execute(f"SELECT COUNT(*) FROM events WHERE seq <= ? AND {visible}",
                                           [bad, *args]).fetchone()[0]
        return {"ok": chain["ok"], "events": events, "first_bad": first_bad}

    def security_events(self, limit: int, before: int | None = None) -> list[dict]:
        """Sign-in and account events (service 'auth') of every user, newest first, for admins. Includes the
        'anonymous' rows (failed sign-ins for unknown names). Details of other services never appear here."""
        where, args = ["service = 'auth'"], []
        if before:
            where.append("seq < ?")
            args.append(before)
        with self._connect() as db:
            rows = db.execute(f"SELECT * FROM events WHERE {' AND '.join(where)} ORDER BY seq DESC LIMIT ?",
                              [*args, limit]).fetchall()
            keys = self._keys(db, {r["user_id"] for r in rows})
        return [{"seq": r["seq"], "time": r["time"], "action": r["action"], "user_id": r["user_id"],
                 "actor": r["actor"], "subject": r["subject"], "detail": self._detail(r, keys)} for r in rows]

    # --- accounts ---------------------------------------------------------------------------

    def purge_user(self, user_id: str) -> dict:
        """Crypto-shred a deleted account: its key is overwritten on disk (secure_delete, WAL truncated) and
        replaced by an empty tombstone. The rows and the chain stay; their details read as {"_shredded": true}.
        If the account had adopted the legacy rows, meta.legacy_owner becomes a tombstone too, so nobody can
        read or adopt them afterwards (they stay in plain text: see the module docstring). Repeatable."""
        if not user_id or user_id == LEGACY:
            raise ValueError("not a user")
        with self._lock, self._connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE")
            try:
                had_key = db.execute("SELECT length(dek) FROM user_keys WHERE user_id = ?", (user_id,)).fetchone()
                db.execute("DELETE FROM user_keys WHERE user_id = ?", (user_id,))
                db.execute("INSERT INTO user_keys (user_id, dek, created_at) VALUES (?, X'', ?)", (user_id, _now()))
                events = db.execute("SELECT COUNT(*) FROM events WHERE user_id = ?", (user_id,)).fetchone()[0]
                legacy = db.execute("UPDATE meta SET value = ? WHERE key = 'legacy_owner' AND value = ?",
                                    (LEGACY_OWNER_DELETED, user_id)).rowcount
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            truncate_wal(db, log, purge="user key")
        return {"events": events, "key_shredded": bool(had_key and had_key[0]), "legacy_closed": bool(legacy)}

    def adopt_legacy(self, owner_id: str, dry_run: bool = False) -> dict:
        """Give the rows from before accounts existed to one user (they then read them as their first events).
        Set once: a second call for the same user changes nothing; another user gets Conflict, and so does
        everyone once the adopting account was deleted."""
        if not owner_id or owner_id == LEGACY:
            raise ValueError("not a user")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                current = self._legacy_owner(db)
                if current == LEGACY_OWNER_DELETED:
                    raise Conflict("the legacy audit rows belonged to an account that was deleted, so nobody can "
                                   "adopt or read them; they stay in the log because the hash chain keeps every row")
                if current is not None and current != owner_id:
                    raise Conflict("the legacy audit rows already belong to another user")
                rows = db.execute("SELECT COUNT(*) FROM events WHERE user_id = ?", (LEGACY,)).fetchone()[0]
                changed = current is None and not dry_run
                if changed:
                    db.execute("INSERT INTO meta (key, value) VALUES ('legacy_owner', ?) ON CONFLICT DO NOTHING",
                               (owner_id,))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return {"legacy_rows": rows, "owner_set": changed, "dry_run": dry_run}
