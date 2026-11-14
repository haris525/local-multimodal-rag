"""Accounts, sessions (browser cookies and API tokens) and login throttling in SQLite (WAL).

Session tokens are never stored: only their SHA-256. Times are ISO-8601 UTC strings with second precision,
so they compare correctly as text.

Credentials version (schema v3): every account has a counter that a password change, a reset, a disable, an
enable and the start of a deletion increase in the same statement. Each session keeps the value it was
created with, and is valid only while the two are equal. A session is inserted only if the counter still
has the value its creator read before checking the password, so a sign-in whose check overlaps a password
change, a reset or a disable gets no session at all.

The account machine (schema v4): an account is in one AccountState, derived in one place (account_state);
every change of state is one action through apply(), one transaction that reads the row, checks ALLOWED, the
self rule and the last-admin rule against that same read, writes the change, bumps the credentials version
and deletes the sessions where the action requires it. CHECK constraints keep the columns to the states
the code writes. A deletion stays in the deleting state until every data service has purged the account
(auth.api resumes it at start-up and every hour); the row records who asked and why the last try failed.
"""

import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

_USERS = """
    user_id                  TEXT PRIMARY KEY,
    username                 TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role                     TEXT NOT NULL CHECK (role IN ('admin','user')),
    password_hash            TEXT NOT NULL,
    disabled                 INTEGER NOT NULL DEFAULT 0 CHECK (disabled IN (0, 1)),
    must_change_password     INTEGER NOT NULL DEFAULT 0 CHECK (must_change_password IN (0, 1)),
    status                   TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','deleting')),
    reset_by_admin_at        TEXT,
    password_changed_at      TEXT NOT NULL,
    created_at               TEXT NOT NULL,
    created_by               TEXT,
    last_login_at            TEXT,
    password_reset_notice_at TEXT,
    credentials_version      INTEGER NOT NULL DEFAULT 0 CHECK (credentials_version >= 0),
    deletion_requested_by    TEXT,
    deletion_error           TEXT,
    deletion_attempted_at    TEXT,
    CHECK (status <> 'deleting' OR disabled = 1)
"""
_SCHEMA = [
    f"CREATE TABLE IF NOT EXISTS users ({_USERS})",
    """CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    token_hash          TEXT NOT NULL UNIQUE,
    user_id             TEXT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    kind                TEXT NOT NULL CHECK (kind IN ('browser','api')),
    name                TEXT,
    created_at          TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    idle_expires_at     TEXT,
    absolute_expires_at TEXT NOT NULL,
    ip                  TEXT,
    user_agent          TEXT,
    credentials_version INTEGER NOT NULL DEFAULT 0
)""",
    "CREATE INDEX IF NOT EXISTS sessions_user ON sessions (user_id, kind)",
    "CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions (absolute_expires_at)",
    """CREATE TABLE IF NOT EXISTS login_throttle (
    key          TEXT PRIMARY KEY,
    failures     INTEGER NOT NULL DEFAULT 0,
    window_start TEXT NOT NULL,
    lock_count   INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT
)""",
]
# v1 -> v2: the admin-reset notice outlives the forced password change (cleared only when the user dismisses
# it); a reset still waiting for that change keeps its notice.
_V1_TO_V2 = [
    "ALTER TABLE users ADD COLUMN password_reset_notice_at TEXT",
    "UPDATE users SET password_reset_notice_at = reset_by_admin_at WHERE reset_by_admin_at IS NOT NULL",
]
# v2 -> v3: the credentials version. Existing sessions keep working (both sides start at 0).
_V2_TO_V3 = {
    "users": "ALTER TABLE users ADD COLUMN credentials_version INTEGER NOT NULL DEFAULT 0",
    "sessions": "ALTER TABLE sessions ADD COLUMN credentials_version INTEGER NOT NULL DEFAULT 0",
}
# v3 -> v4: CHECK constraints and the deletion bookkeeping need a rebuild of users (foreign keys are off
# meanwhile; sessions keep pointing at users by name). A deleting row that a crash left enabled is disabled,
# as every deleting row is meant to be.
_V3_TO_V4 = [
    f"CREATE TABLE users_v4 ({_USERS})",
    """INSERT INTO users_v4 (user_id, username, role, password_hash, disabled, must_change_password, status,
           reset_by_admin_at, password_changed_at, created_at, created_by, last_login_at, password_reset_notice_at,
           credentials_version)
       SELECT user_id, username, role, password_hash,
           CASE WHEN disabled != 0 OR status = 'deleting' THEN 1 ELSE 0 END,
           CASE WHEN must_change_password != 0 THEN 1 ELSE 0 END, status, reset_by_admin_at,
           password_changed_at, created_at, created_by, last_login_at, password_reset_notice_at,
           MAX(credentials_version, 0)
       FROM users ORDER BY rowid""",
    "DROP TABLE users",
    "ALTER TABLE users_v4 RENAME TO users",
]
SCHEMA_VERSION = 4
_BUMP = "credentials_version = credentials_version + 1"


class AccountState(StrEnum):
    ACTIVE = "active"
    MUST_CHANGE = "must_change"  # signed in only to choose a new password
    DISABLED = "disabled"
    DELETING = "deleting"  # disabled for good; the data services' purge is not finished


_A, _M, _D, _X = AccountState.ACTIVE, AccountState.MUST_CHANGE, AccountState.DISABLED, AccountState.DELETING
# action: the states it may start from. Anything else is refused, by whoever asks (admin, user, operator).
ALLOWED: dict[str, frozenset[AccountState]] = {
    "change_password": frozenset({_A, _M}),
    "reset": frozenset({_A, _M, _D}),
    "disable": frozenset({_A, _M}),
    "enable": frozenset({_D}),
    "set_role": frozenset({_A, _M, _D}),
    "delete": frozenset({_A, _M, _D, _X}),  # from deleting: a retry, which resumes the purge
    "unlock": frozenset({_A, _M, _D}),
}
SELF_REFUSED = {"reset": "use change password for your own account", "disable": "you cannot disable your own account",
                "set_role": "you cannot change your own role", "delete": "you cannot delete your own account"}
BUMPS = {"change_password", "reset", "disable", "enable", "delete"}  # increase the credentials version
SIGNS_OUT = {"change_password", "reset", "disable", "delete"}  # delete every session in the same transaction


class Refused(Exception):
    """An action the account's state, the actor or the last-admin rule does not allow (HTTP status, reason)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def account_state(user: "User") -> AccountState:
    """The one derivation of an account's state from its columns; the API, the CLI and the Users panel use it."""
    if user.status == "deleting":
        return AccountState.DELETING
    if user.disabled:
        return AccountState.DISABLED
    if user.must_change_password:
        return AccountState.MUST_CHANGE
    return AccountState.ACTIVE


def _active_admin(user: "User") -> bool:
    return user.role == "admin" and account_state(user) in (_A, _M)


def check(user: "User", action: str, *, actor_id: str | None = None, other_active_admins: int = 1,
          role: str | None = None) -> Refused | None:
    """Why `action` may not be applied to `user` now (or None). The rules: ALLOWED; nobody resets, disables,
    re-roles or deletes their own account; the last active admin is never disabled, demoted or deleted."""
    state = account_state(user)
    if state not in ALLOWED[action]:
        if state == _X:
            return Refused(409, "the account is being deleted; delete it again to finish")
        return Refused(409, f"cannot {action.replace('_', ' ')} an account that is {state.value.replace('_', ' ')}")
    if actor_id is not None and actor_id == user.user_id and action in SELF_REFUSED:
        return Refused(409, SELF_REFUSED[action])
    removes_admin = action in ("disable", "delete") or (action == "set_role" and role == "user")
    if removes_admin and _active_admin(user) and other_active_admins == 0:
        return Refused(409, f"you cannot {'demote' if action == 'set_role' else action} the last active admin")
    return None


# The Users panel's buttons, in order: (button, action, role) where the button is offered when check() allows it.
ADMIN_ACTIONS = (("disable", "disable", None), ("enable", "enable", None), ("make_admin", "set_role", "admin"),
                 ("make_user", "set_role", "user"), ("reset_password", "reset", None), ("unlock", "unlock", None),
                 ("delete", "delete", None))


def admin_actions(user: "User", viewer_id: str | None, other_active_admins: int, locked: bool) -> list[str]:
    """The buttons an admin (viewer_id) may use on this account now, by the same rules as apply()."""
    out = []
    for button, action, role in ADMIN_ACTIONS:
        if role is not None and role == user.role:
            continue
        if button == "unlock" and not locked:
            continue
        if check(user, action, actor_id=viewer_id, other_active_admins=other_active_admins, role=role) is None:
            out.append(button)
    return out


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def parse(text: str) -> datetime:
    return datetime.fromisoformat(text)


def user_throttle_key(username: str) -> str:
    """The login_throttle row of a username (auth.limits counts sign-in failures under it)."""
    return f"u:{username.lower()}"


class UsernameTaken(Exception):
    pass


@dataclass(frozen=True)
class User:
    user_id: str
    username: str
    role: str
    password_hash: str
    disabled: bool
    must_change_password: bool
    status: str
    reset_by_admin_at: str | None
    password_changed_at: str
    created_at: str
    created_by: str | None
    last_login_at: str | None
    password_reset_notice_at: str | None  # an admin reset the password; shown until the user dismisses it
    credentials_version: int = 0  # increased by every password change, reset, disable, enable, deletion
    deletion_requested_by: str | None = None  # the admin who asked; the resumer acts for them
    deletion_error: str | None = None  # why the last purge stopped
    deletion_attempted_at: str | None = None


@dataclass(frozen=True)
class Session:
    session_id: str
    token_hash: str
    user_id: str
    kind: str
    name: str | None
    created_at: str
    last_seen_at: str
    idle_expires_at: str | None
    absolute_expires_at: str
    ip: str | None
    user_agent: str | None
    credentials_version: int = 0  # the account's version when the session was created


@dataclass(frozen=True)
class Throttle:
    key: str
    failures: int
    window_start: str
    lock_count: int
    locked_until: str | None


ThrottleRows = dict[str, Throttle | None]
ThrottleUpdate = Callable[[ThrottleRows], ThrottleRows]


def _user(row: sqlite3.Row) -> User:
    return User(row["user_id"], row["username"], row["role"], row["password_hash"], bool(row["disabled"]),
                bool(row["must_change_password"]), row["status"], row["reset_by_admin_at"],
                row["password_changed_at"], row["created_at"], row["created_by"], row["last_login_at"],
                row["password_reset_notice_at"], row["credentials_version"], row["deletion_requested_by"],
                row["deletion_error"], row["deletion_attempted_at"])


def _session(row: sqlite3.Row) -> Session:
    return Session(row["session_id"], row["token_hash"], row["user_id"], row["kind"], row["name"], row["created_at"],
                   row["last_seen_at"], row["idle_expires_at"], row["absolute_expires_at"], row["ip"],
                   row["user_agent"], row["credentials_version"])


def _throttle(row: sqlite3.Row) -> Throttle:
    return Throttle(row["key"], row["failures"], row["window_start"], row["lock_count"], row["locked_until"])


_SESSION_COLS = ", ".join(f"s.{c}" for c in Session.__dataclass_fields__)
_USER_COLS = ", ".join(f"u.{c}" for c in User.__dataclass_fields__)
_LIVE = "(s.absolute_expires_at > :now AND (s.idle_expires_at IS NULL OR s.idle_expires_at > :now))"


class AuthStore:
    def __init__(self, path: Path, clock: Callable[[], datetime] = utcnow):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self.clock = clock
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            self._upgrade(db)

    @staticmethod
    def _upgrade(db: sqlite3.Connection) -> None:
        """Bring the schema to SCHEMA_VERSION in one transaction (a second run does nothing). v2 added the reset
        notice, v3 the credentials version, v4 the CHECK constraints and the deletion columns (a rebuild of
        users, with foreign keys off: the pragma only takes effect outside a transaction)."""
        if db.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return
        db.execute("PRAGMA foreign_keys=OFF")
        try:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:  # another process may have won
                    columns = {r[1] for r in db.execute("PRAGMA table_info(users)")}
                    if columns and "password_reset_notice_at" not in columns:
                        for statement in _V1_TO_V2:
                            db.execute(statement)
                    for table, statement in _V2_TO_V3.items():
                        existing = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                        if existing and "credentials_version" not in existing:
                            db.execute(statement)
                    if columns and "deletion_error" not in columns:
                        for statement in _V3_TO_V4:
                            db.execute(statement)
                    for statement in _SCHEMA:
                        db.execute(statement)
                    if db.execute("PRAGMA foreign_key_check").fetchall():
                        raise RuntimeError("auth.db upgrade: sessions point at accounts that do not exist")
                    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        finally:
            db.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    def now(self) -> str:
        return iso(self.clock())

    # --- users ------------------------------------------------------------------------------

    def create_user(self, username: str, role: str, password_hash: str, must_change: bool,
                    created_by: str | None) -> User:
        """A new account starts without lockout state: failures counted for the name before it existed (unknown
        names are counted too) are cleared in the same transaction."""
        user_id, now = str(uuid.uuid4()), self.now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO users (user_id, username, role, password_hash, must_change_password,"
                    " password_changed_at, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, username.lower(), role, password_hash, int(must_change), now, now, created_by))
                db.execute("DELETE FROM login_throttle WHERE key = ?", (user_throttle_key(username),))
                db.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                db.execute("ROLLBACK")
                raise UsernameTaken(username) from exc
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return _user(row) if row else None

    def get_user_by_name(self, username: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (username.lower(),)).fetchone()
        return _user(row) if row else None

    def list_users(self) -> list[User]:
        with self._connect() as db:
            return [_user(r) for r in db.execute("SELECT * FROM users ORDER BY username")]

    def count_users(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def count_admins(self) -> int:
        """Admin accounts in any state (create-admin refuses when one exists)."""
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'").fetchone()[0]

    @staticmethod
    def _active_admins(db: sqlite3.Connection) -> set[str]:
        return {r[0] for r in db.execute("SELECT user_id FROM users WHERE role = 'admin' AND disabled = 0"
                                         " AND status = 'active'")}

    def active_admin_ids(self) -> set[str]:
        with self._connect() as db:
            return self._active_admins(db)

    def apply(self, user_id: str, action: str, *, actor_id: str | None = None, password_hash: str | None = None,
              must_change: bool = True, by_admin: bool = True, role: str | None = None,
              expected_version: int | None = None) -> User:
        """Apply one action to one account in one transaction (BEGIN IMMEDIATE): read the row, check() it with
        the admin count of the same transaction, write the change, bump the credentials version and delete the
        sessions where the action requires it. Refused (404, 409) leaves everything as it was.

        change_password and reset take password_hash; reset also must_change and by_admin (the CLI's --keep
        resets with both False); set_role takes role; delete records actor_id as the requester.
        expected_version: the credentials version the caller acted on (the user's own password change passes
        the one its session was checked against); any other version is refused, so a change whose password
        check overlapped a reset or a disable cannot undo it."""
        if action not in ALLOWED:
            raise ValueError(f"unknown action {action!r}")
        if action == "change_password" and expected_version is None:
            raise ValueError("change_password needs the credentials version the user was checked against")
        now = self.now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if row is None:
                    raise Refused(404, "user not found")
                user = _user(row)
                if expected_version is not None and user.credentials_version != expected_version:
                    raise Refused(409, "the account changed meanwhile; sign in again")
                others = len(self._active_admins(db) - {user_id})
                refused = check(user, action, actor_id=actor_id, other_active_admins=others, role=role)
                if refused:
                    raise refused
                sets, values = self._change(action, now, password_hash, must_change, by_admin, role, actor_id)
                if action in BUMPS:
                    sets.append(_BUMP)
                if sets:
                    db.execute(f"UPDATE users SET {', '.join(sets)} WHERE user_id = ?", (*values, user_id))
                if action in SIGNS_OUT:
                    db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
                if action == "unlock":
                    db.execute("DELETE FROM login_throttle WHERE key = ?", (user_throttle_key(user.username),))
                after = _user(db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone())
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return after

    @staticmethod
    def _change(action: str, now: str, password_hash: str | None, must_change: bool, by_admin: bool,
                role: str | None, actor_id: str | None) -> tuple[list[str], list]:
        """The columns one action writes. reset_by_admin_at is the reason for a pending forced change (cleared
        by the next change); password_reset_notice_at is the user's notice of an admin reset, which only
        dismiss_reset_notice clears, so an admin who resets and then signs in as the user cannot make the reset
        invisible by changing the temporary password."""
        if action in ("change_password", "reset"):
            if not password_hash:
                raise ValueError(f"{action} needs a password hash")
            forced = action == "reset" and must_change
            admin = action == "reset" and by_admin
            return (["password_hash = ?", "must_change_password = ?", "password_changed_at = ?", "reset_by_admin_at = ?",
                     "password_reset_notice_at = CASE WHEN ? THEN ? ELSE password_reset_notice_at END"],
                    [password_hash, int(forced), now, now if admin else None, int(admin), now])
        if action in ("disable", "enable"):
            return ["disabled = ?"], [int(action == "disable")]
        if action == "set_role":
            if role not in ("admin", "user"):
                raise ValueError("set_role needs role admin or user")
            return ["role = ?"], [role]
        if action == "delete":
            return (["status = 'deleting'", "disabled = 1", "deletion_requested_by = ?", "deletion_error = NULL"],
                    [actor_id])
        return [], []  # unlock: only the throttle row changes

    def dismiss_reset_notice(self, user_id: str) -> str | None:
        """Clear the admin-reset notice; returns its time, or None when there was none."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT password_reset_notice_at FROM users WHERE user_id = ?", (user_id,)).fetchone()
            db.execute("UPDATE users SET password_reset_notice_at = NULL WHERE user_id = ?", (user_id,))
            db.execute("COMMIT")
        return row[0] if row else None

    def set_password_hash(self, user_id: str, password_hash: str, credentials_version: int) -> bool:
        """Rehash with new parameters: same password, nothing else changes. Only while the account is still at
        the credentials version the caller checked the password against (a reset meanwhile wins)."""
        with self._connect() as db:
            cur = db.execute("UPDATE users SET password_hash = ? WHERE user_id = ? AND credentials_version = ?",
                             (password_hash, user_id, credentials_version))
        return cur.rowcount == 1

    def deleting_users(self) -> list[User]:
        """Accounts whose deletion has not finished."""
        with self._connect() as db:
            return [_user(r) for r in db.execute("SELECT * FROM users WHERE status = 'deleting' ORDER BY username")]

    def deletion_failed(self, user_id: str, error: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE users SET deletion_error = ?, deletion_attempted_at = ? WHERE user_id = ?"
                       " AND status = 'deleting'", (error[:500], self.now(), user_id))

    def delete_user(self, user_id: str) -> bool:
        """The last step of a deletion: the account row (sessions cascade) and its username's lockout state, so
        a later account with the same name does not inherit a lock or its doubling. Only a deleting account."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("DELETE FROM users WHERE user_id = ? AND status = 'deleting' RETURNING username",
                                 (user_id,)).fetchone()
                if row:
                    db.execute("DELETE FROM login_throttle WHERE key = ?", (user_throttle_key(row[0]),))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return row is not None

    def record_login(self, user_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE users SET last_login_at = ? WHERE user_id = ?", (self.now(), user_id))

    # --- sessions ---------------------------------------------------------------------------

    def create_session(self, user_id: str, kind: str, token_hash: str, name: str | None,
                       idle_expires_at: str | None, absolute_expires_at: str, ip: str | None,
                       user_agent: str | None, *, credentials_version: int) -> Session | None:
        """Insert a session only while the account is active, enabled and still at the credentials version
        the caller read before it checked the password; None otherwise (the credentials changed meanwhile)."""
        session_id, now = uuid.uuid4().hex, self.now()
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO sessions (session_id, token_hash, user_id, kind, name, created_at, last_seen_at,"
                " idle_expires_at, absolute_expires_at, ip, user_agent, credentials_version)"
                " SELECT ?, ?, user_id, ?, ?, ?, ?, ?, ?, ?, ?, credentials_version FROM users"
                " WHERE user_id = ? AND credentials_version = ? AND status = 'active' AND disabled = 0",
                (session_id, token_hash, kind, name, now, now, idle_expires_at, absolute_expires_at, ip,
                 (user_agent or "")[:300] or None, user_id, credentials_version))
            if cur.rowcount != 1:
                return None
            row = db.execute(f"SELECT {_SESSION_COLS} FROM sessions s WHERE session_id = ?", (session_id,)).fetchone()
        return _session(row)

    def session_by_hash(self, token_hash: str) -> tuple[Session, User] | None:
        with self._connect() as db:
            row = db.execute(f"SELECT {_SESSION_COLS}, {_USER_COLS} FROM sessions s JOIN users u USING (user_id)"
                             " WHERE s.token_hash = ?", (token_hash,)).fetchone()
        if row is None:
            return None
        n = len(Session.__dataclass_fields__)
        values = tuple(row)
        return Session(*values[:n]), _user(dict(zip(User.__dataclass_fields__, values[n:])))

    def touch(self, session_id: str, last_seen_at: str, idle_expires_at: str | None) -> None:
        with self._connect() as db:
            db.execute("UPDATE sessions SET last_seen_at = ?, idle_expires_at = ? WHERE session_id = ?",
                       (last_seen_at, idle_expires_at, session_id))

    def delete_session(self, session_id: str, user_id: str | None = None, kind: str | None = None) -> bool:
        with self._connect() as db:
            cur = db.execute("DELETE FROM sessions WHERE session_id = ? AND (? IS NULL OR user_id = ?)"
                             " AND (? IS NULL OR kind = ?)", (session_id, user_id, user_id, kind, kind))
        return cur.rowcount > 0

    def delete_session_by_hash(self, token_hash: str) -> Session | None:
        with self._connect() as db:
            row = db.execute(f"SELECT {_SESSION_COLS} FROM sessions s WHERE token_hash = ?", (token_hash,)).fetchone()
            if row:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        return _session(row) if row else None

    def delete_sessions(self, user_id: str, except_session_id: str | None = None, kind: str | None = None) -> int:
        with self._connect() as db:
            cur = db.execute("DELETE FROM sessions WHERE user_id = ? AND session_id IS NOT ?"
                             " AND (? IS NULL OR kind = ?)", (user_id, except_session_id, kind, kind))
        return cur.rowcount

    def list_tokens(self, user_id: str) -> list[Session]:
        with self._connect() as db:
            rows = db.execute(f"SELECT {_SESSION_COLS} FROM sessions s WHERE user_id = :uid AND kind = 'api'"
                              f" AND {_LIVE} ORDER BY created_at DESC", {"uid": user_id, "now": self.now()})
            return [_session(r) for r in rows]

    def count_sessions(self, user_id: str) -> int:
        """Unexpired sessions and API tokens."""
        with self._connect() as db:
            return db.execute(f"SELECT COUNT(*) FROM sessions s WHERE user_id = :uid AND {_LIVE}",
                              {"uid": user_id, "now": self.now()}).fetchone()[0]

    def purge_expired(self) -> int:
        """Delete expired sessions and tokens; run at startup and every hour."""
        with self._connect() as db:
            cur = db.execute(f"DELETE FROM sessions AS s WHERE NOT {_LIVE}", {"now": self.now()})
        return cur.rowcount

    # --- throttle ---------------------------------------------------------------------------

    def get_throttle(self, key: str) -> Throttle | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM login_throttle WHERE key = ?", (key,)).fetchone()
        return _throttle(row) if row else None

    def put_throttle(self, key: str, failures: int, window_start: str, lock_count: int,
                     locked_until: str | None) -> None:
        with self._connect() as db:
            self._put_throttle(db, Throttle(key, failures, window_start, lock_count, locked_until))

    @staticmethod
    def _put_throttle(db: sqlite3.Connection, t: Throttle) -> None:
        db.execute("INSERT INTO login_throttle (key, failures, window_start, lock_count, locked_until)"
                   " VALUES (?, ?, ?, ?, ?) ON CONFLICT (key) DO UPDATE SET failures = excluded.failures,"
                   " window_start = excluded.window_start, lock_count = excluded.lock_count,"
                   " locked_until = excluded.locked_until",
                   (t.key, t.failures, t.window_start, t.lock_count, t.locked_until))

    def update_throttle(self, key: str, update: Callable[[Throttle | None], Throttle]) -> Throttle:
        """Read, change and write one throttle row in a single transaction, so concurrent failures all count."""
        _, after = self.update_throttles([key], lambda rows: {key: update(rows[key])})
        return after[key]

    def update_throttles(self, keys: list[str], update: ThrottleUpdate) -> tuple[ThrottleRows, ThrottleRows]:
        """Read several throttle rows, change them and write them back in one transaction (BEGIN IMMEDIATE, so
        concurrent callers take turns). `update` gets the current rows and returns the changes: a row to write,
        or None to delete it; keys it leaves out stay as they are. Returns (rows before, rows after)."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                before = {}
                for key in keys:
                    row = db.execute("SELECT * FROM login_throttle WHERE key = ?", (key,)).fetchone()
                    before[key] = _throttle(row) if row else None
                changes = update(dict(before))
                for key, t in changes.items():
                    if t is None:
                        db.execute("DELETE FROM login_throttle WHERE key = ?", (key,))
                    else:
                        self._put_throttle(db, t)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return before, before | changes

    def purge_throttle(self, older_than: str) -> int:
        """Forget rows that no longer matter (keeps the table small): no failure counted since `older_than`,
        and no lock that ended after it. The doubling remembers a lock until lockout_max_hours after it ended
        (auth.limits), so the caller passes now - lockout_max_hours and a lock's history is never dropped
        before the doubling would have forgotten it."""
        with self._connect() as db:
            cur = db.execute("DELETE FROM login_throttle WHERE window_start < ? AND"
                             " (locked_until IS NULL OR locked_until < ?)", (older_than, older_than))
        return cur.rowcount
