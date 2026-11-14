"""rag_common.sqlite: the checked WAL truncation after deletes."""

import logging
import sqlite3

from rag_common.sqlite import truncate_wal

log = logging.getLogger("test.sqlite")


def _db(path, timeout=5.0):
    db = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    return db


def test_truncate_wal_empties_the_log(tmp_path):
    db = _db(tmp_path / "a.db")
    db.execute("CREATE TABLE t (x TEXT)")
    db.execute("INSERT INTO t VALUES ('private text')")
    assert (tmp_path / "a.db-wal").stat().st_size > 0
    assert truncate_wal(db, log) is True
    assert (tmp_path / "a.db-wal").stat().st_size == 0
    db.close()


def test_a_blocked_truncation_is_reported(tmp_path, caplog):
    writer = _db(tmp_path / "a.db", timeout=0.05)
    writer.execute("CREATE TABLE t (x TEXT)")
    writer.execute("INSERT INTO t VALUES ('one')")
    reader = _db(tmp_path / "a.db")
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM t").fetchall()  # holds a snapshot of the WAL
    writer.execute("INSERT INTO t VALUES ('two')")
    with caplog.at_level(logging.WARNING, logger="test.sqlite"):
        assert truncate_wal(writer, log, purge="user") is False
    assert any("WAL checkpoint blocked" in r.getMessage() for r in caplog.records)
    reader.execute("COMMIT")
    assert truncate_wal(writer, log) is True
    reader.close()
    writer.close()
