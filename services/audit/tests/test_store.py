import sqlite3

import pytest

from rag_common.testing import new_user_id

from audit.store import AuditStore

U = new_user_id()


def test_chain_verifies_and_detects_tampering(tmp_path):
    store = AuditStore(tmp_path / "a.db")
    for i in range(3):
        store.append("rag", "chat.answered", "local", f"r{i}", "doc", {"question": f"q{i}"}, U)
    assert store.verify() == {"ok": True, "events": 3, "first_bad_seq": None, "head": store.list_for(U, 1)[0]["hash"]}

    db = sqlite3.connect(tmp_path / "a.db")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE events SET detail = '{}' WHERE seq = 2")
    db.execute("DROP TRIGGER events_no_update")  # an attacker with file access
    db.execute("""UPDATE events SET detail = '{"question": "edited"}' WHERE seq = 2""")
    db.commit()
    assert store.verify()["first_bad_seq"] == 2


def test_every_connection_is_closed(tmp_path, monkeypatch):
    opened, connect = [], sqlite3.connect

    def tracked(*args, **kwargs):
        opened.append(connect(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(sqlite3, "connect", tracked)
    store = AuditStore(tmp_path / "a.db")
    store.append("rag", "chat.answered", "local", "r1", "doc", {}, U)
    store.list_for(U, 10)
    store.verify()
    store.verify_for(U)
    store.security_events(10)
    store.purge_user(U)
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError):  # "Cannot operate on a closed database"
            db.execute("SELECT 1")
