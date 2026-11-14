import pytest

from rag_common.contracts import NewMessage
from rag_common.testing import new_user_id

from threads.store import Store

U = new_user_id()


def turn(q: str, a: str) -> list[NewMessage]:
    return [NewMessage(role="user", content=q), NewMessage(role="assistant", content=a, meta={"route": "search"})]


def test_context_returns_summary_plus_unsummarized_turns_and_history_is_kept(tmp_path):
    store = Store(tmp_path / "t.db")
    t = store.create(U, "Budget questions", ["d1"])
    first = store.append(U, t.thread_id, turn("q1", "a1"), None)
    store.append(U, t.thread_id, turn("q2", "a2"), ["d1", "d2"])

    assert store.set_summary(U, t.thread_id, "User asked q1; answer a1.", first[-1].seq)
    assert not store.set_summary(U, t.thread_id, "stale", first[0].seq)  # never moves backwards
    ctx = store.context(U, t.thread_id)
    assert ctx.summary == "User asked q1; answer a1."
    assert [m.content for m in ctx.messages] == ["q2", "a2"]

    detail = store.get(U, t.thread_id)
    assert [m.content for m in detail.messages] == ["q1", "a1", "q2", "a2"]  # full history kept
    assert (detail.thread.message_count, detail.thread.summarized, detail.thread.doc_ids) == (4, 2, ["d1", "d2"])
    assert detail.messages[1].meta == {"route": "search"}

    assert store.delete(U, t.thread_id) and store.get(U, t.thread_id) is None and store.context(U, t.thread_id) is None


def test_memories_dedupe_and_evict_only_automatic_ones_at_the_cap(tmp_path):
    store = Store(tmp_path / "m.db", max_memories=2)
    assert store.add_memory(U, "Works in complaint handling.", "user", None)
    assert store.add_memory(U, "works in  COMPLAINT handling", "auto", None) is None  # same after normalizing
    auto = store.add_memory(U, "Prefers answers as tables.", "auto", None)
    store.add_memory(U, "Uses metric units.", "auto", None)  # full: evicts the oldest automatic memory
    assert [m.text for m in store.memories(U)] == ["Works in complaint handling.", "Uses metric units."]
    assert auto.memory_id not in {m.memory_id for m in store.memories(U)}

    store.delete_memory(U, store.memories(U)[1].memory_id)
    store.add_memory(U, "Second user fact.", "user", None)
    with pytest.raises(ValueError, match="memory is full"):
        store.add_memory(U, "Third user fact.", "user", None)


def test_titles_never_store_credentials(tmp_path):
    store = Store(tmp_path / "s.db")
    thread = store.create(U, "remember my password is hunter2", [])
    assert thread.title == "remember my password is [redacted]"
    assert store.update(U, thread.thread_id, "token: abc123", None).title == "token: [redacted]"


def test_replacing_a_memory_is_atomic_and_never_loses_the_old_fact(tmp_path):
    store = Store(tmp_path / "t.db")
    finance = store.add_memory(U, "The user works in finance.", "user", None)
    store.add_memory(U, "The user uses EUR.", "auto", None)
    item, old = store.replace_memory(U, finance.memory_id, "The user works in legal.", "auto")
    assert (item.memory_id, item.source, old) == (finance.memory_id, "user", "The user works in finance.")
    assert store.replace_memory(U, finance.memory_id, "the user uses eur", "auto") is None  # says what another one says
    assert [m.text for m in store.memories(U)] == ["The user works in legal.", "The user uses EUR."]
    with pytest.raises(KeyError):
        store.replace_memory(U, "f" * 32, "The user is a nurse.", "user")


def test_memory_writes_refuse_credentials(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from threads import api
    with pytest.raises(HTTPException) as err:
        api._no_credentials("Remember my password is hunter2")
    assert err.value.status_code == 422
    api._no_credentials("The user prefers short answers.")
