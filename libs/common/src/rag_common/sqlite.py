"""SQLite helpers shared by the services' stores."""

import logging
import sqlite3

from rag_common.logging import log_fields


def truncate_wal(db: sqlite3.Connection, log: logging.Logger, **fields) -> bool:
    """Copy the write-ahead log into the database and empty it (PRAGMA wal_checkpoint(TRUNCATE)), so rows just
    deleted with secure_delete do not linger in the -wal file. Run it outside a transaction, after the delete.

    Another connection reading or writing at that moment can block it, after the connection's busy timeout:
    then it returns False and logs a WARNING, and the deleted data stays in the -wal file until a later
    checkpoint overwrites it (SQLite's own, or the next account deletion)."""
    busy, wal_frames, checkpointed = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if busy:
        log_fields(log, "WAL checkpoint blocked: deleted data stays in the -wal file until a later checkpoint",
                   logging.WARNING, wal_frames=wal_frames, checkpointed=checkpointed, **fields)
    return not busy
