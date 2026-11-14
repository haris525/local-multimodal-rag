"""The LLM cache (llmcache.py): a model reply is reused for the same request of the same owner, a cut, failed,
empty or refused reply is never stored, entries go with their documents and accounts, the size is bounded, and a
cache failure (busy, full, damaged, a corrupt row) is a model call with a warning, never a failed job."""

import asyncio
import hashlib
import io
import json
import logging
import sqlite3
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import ingest.describe as describe
from conftest import KEYS, WordCounter, make_ready
from rag_common.contracts import DocType
from rag_common.testing import new_user_id

from ingest import paths
from ingest.config import Settings
from ingest.describe import ModelCall, VisionSettings, describe_image, summarize
from ingest.elements import Element, ParsedDocument
from ingest.llmcache import SCHEMA_VERSION, LlmCache, cache_key, model_identity, skip_reason
from ingest.pipeline import Pipeline, delete_document
from ingest.registry import Registry
from ingest.worker import open_cache, run_job, sweep

A, B = new_user_id(), new_user_id()
DOC, DOC2 = "a" * 32, "b" * 32
CFG = VisionSettings(model="m", retry_backoff_s=0.0)
TABLE = "x-axis: 1 to 3\n| x | y |\n|---|---|\n| 1 | 2.00 |\n| 3 | 4.00 |"


def png(color: str = "white", size: tuple[int, int] = (40, 20)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    """An OpenAI-compatible vision server whose reply is a function of the request, like temperature 0: the
    extraction pass reads a table, a description or a summary names a hash of its request. script: (status,
    content, finish_reason) replies to give first, in order. Every call is counted."""

    def __init__(self, script=()):
        self.calls: list[dict] = []
        self.script = list(script)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        if self.script:
            status, content, finish = self.script.pop(0)
            if status != 200:
                return httpx.Response(status, json={"detail": content})
        else:
            content, finish = self.reply(body), "stop"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})

    @staticmethod
    def reply(body: dict) -> str:
        user = body["messages"][-1]["content"]
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        if isinstance(user, str):
            return f"Summary {digest}. Topics: test"
        if user[-1]["text"].startswith("This image is from the document"):
            return TABLE
        return f"Description {digest}"

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="http://llm", transport=httpx.MockTransport(self.handler))


def store(folder, model: str = "model-a", **kwargs) -> LlmCache:
    cache = LlmCache(folder / "llm_cache.db", max_bytes=kwargs.pop("max_bytes", 1 << 30),
                     max_entries=kwargs.pop("max_entries", 1000), **kwargs)
    cache.model = model
    return cache


def describe_with(cache, model: FakeModel, *, owner=A, doc=DOC, cfg=CFG, **changes):
    """One figure described through describe_image; returns ((text, cut), the job's cache or None)."""
    args = {"png": png(), "doc_name": "report.pdf", "caption": "Figure 1", "section": "Results", "standalone": False,
            "max_tokens": 4000} | changes
    job = cache.job(owner, doc) if cache is not None else None
    return asyncio.run(describe_image(model.client(), cfg, cache=job, **args)), job


def sql(path, statement: str, *args) -> list[tuple]:
    """One statement on its own connection, closed afterwards (an open handle would keep a file from being
    deleted on Windows)."""
    db = sqlite3.connect(path, isolation_level=None)
    try:
        return db.execute(statement, args).fetchall()
    finally:
        db.close()


def on_disk(path) -> bytes:
    return b"".join(p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists())


async def no_points(owner_id, doc_id):
    pass


# --- hits and misses ------------------------------------------------------------------------


def test_the_same_request_is_answered_from_the_cache_without_a_model_call(tmp_path):
    cache, model = store(tmp_path), FakeModel()
    first, job = describe_with(cache, model)
    assert len(model.calls) == 2 and job.summary() == {"hits": 0, "misses": 2, "writes": 2, "skipped": 0, "errors": 0}
    again, job = describe_with(cache, model)
    assert len(model.calls) == 2 and again == first
    assert job.summary() == {"hits": 2, "misses": 0, "writes": 0, "skipped": 0, "errors": 0}
    text, cut = first
    assert text.startswith(describe.APPROXIMATE) and text.endswith(f"{describe.DATA_HEADING}\n{TABLE}") and not cut


def test_the_text_is_the_same_with_the_cache_on_or_off(tmp_path):
    off, _ = describe_with(None, FakeModel())
    cache = store(tmp_path)
    assert describe_with(cache, FakeModel())[0] == off == describe_with(cache, FakeModel())[0]


def test_a_summary_is_cached_too(tmp_path):
    cache, model = store(tmp_path), FakeModel()
    runs = [asyncio.run(summarize(model.client(), CFG, "doc.pdf", "pdf", "material", 1200, 3000,
                                  cache=cache.job(A, DOC))) for _ in range(2)]
    assert runs[0] == runs[1] and len(model.calls) == 1 and model.calls[0]["temperature"] == 0.2
    asyncio.run(summarize(model.client(), CFG, "doc.pdf", "pdf", "other material", 1200, 3000, cache=cache.job(A, DOC)))
    assert len(model.calls) == 2


# --- the key ----------------------------------------------------------------------------------

CHANGES = {
    "image bytes": {"png": png("black")},
    "caption": {"caption": "Figure 2"},
    "section": {"section": "Methods"},
    "document name": {"doc_name": "other.pdf"},
    "image file or figure": {"standalone": True},
    "token budget": {"max_tokens": 3000},
    "temperature": {"cfg": VisionSettings(model="m", retry_backoff_s=0.0, temperature=0.3)},
    "image size sent": {"cfg": VisionSettings(model="m", retry_backoff_s=0.0, max_side=10)},
    "first pass budget": {"cfg": VisionSettings(model="m", retry_backoff_s=0.0, extraction_tokens=1500)},
    "retry ceiling": {"cfg": VisionSettings(model="m", retry_backoff_s=0.0, retry_max_tokens=9000)},
    "model name": {"cfg": VisionSettings(model="other", retry_backoff_s=0.0)},
    "owner": {"owner": B},
}


@pytest.mark.parametrize("change", CHANGES)
def test_each_part_of_the_request_is_part_of_the_key(tmp_path, change):
    cache, model = store(tmp_path), FakeModel()
    describe_with(cache, model)
    describe_with(cache, model)
    assert len(model.calls) == 2
    describe_with(cache, model, **CHANGES[change])
    assert len(model.calls) > 2, change


def test_the_model_identity_and_the_prompt_version_are_part_of_the_key(tmp_path, monkeypatch):
    model = FakeModel()
    describe_with(store(tmp_path, "weights-Q4_K_XL"), model)
    describe_with(store(tmp_path, "weights-Q8_0"), model)  # another quantization: both passes asked again
    assert len(model.calls) == 4
    monkeypatch.setattr(describe, "IMAGE_DESCRIPTION", ModelCall("image_description", 2, "image description"))
    describe_with(store(tmp_path, "weights-Q4_K_XL"), model)
    assert len(model.calls) == 5  # the first pass is reused, the second (a new prompt version) is asked again


def test_every_component_changes_the_key():
    base = {"owner": A, "call": "image_description", "version": 1, "model": "m", "request": {"messages": [1]},
            "retry": {"factor": 1.5, "max_tokens": 8000}}
    other = {"owner": B, "call": "chart_extraction", "version": 2, "model": "m2", "request": {"messages": [2]},
             "retry": {"factor": 1.5, "max_tokens": 9000}}
    keys = {cache_key(**base)} | {cache_key(**(base | {name: other[name]})) for name in base}
    assert len(keys) == len(base) + 1


def props(status: int = 200, body: dict | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://llm", transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json=body or {})))


def test_the_model_identity_names_the_files_and_another_loaded_file_turns_the_cache_off(tmp_path, caplog):
    s = Settings(data_dir=tmp_path, vision_model_file="Qwen-UD-Q4_K_XL.gguf", vision_mmproj="mmproj-BF16.gguf",
                 retry_backoff_s=0)
    same = asyncio.run(model_identity(s, props(body={"model_path": "/models/gguf/Qwen-UD-Q4_K_XL.gguf",
                                                     "build_info": "b7000-abc"})))
    assert json.loads(same) == {"model": "qwen3.8-27b", "file": "Qwen-UD-Q4_K_XL.gguf", "mmproj": "mmproj-BF16.gguf",
                                "server": "b7000-abc", "salt": ""}
    newer = asyncio.run(model_identity(s, props(body={"model_path": "/models/gguf/Qwen-UD-Q4_K_XL.gguf",
                                                      "build_info": "b7100-def"})))
    assert newer != same  # another llama.cpp build: replies are not reused across it
    with caplog.at_level(logging.WARNING, logger="ingest.llmcache"):
        assert asyncio.run(model_identity(s, props(body={"model_path": "/models/gguf/Qwen-Q8_0.gguf"}))) is None
        unknown = asyncio.run(model_identity(s, props(404)))  # no /props: the configured file, with a warning
        assert json.loads(unknown) == json.loads(same) | {"server": ""}
        assert asyncio.run(model_identity(s, props(503))) == unknown  # busy twice: one retry, then the same
        bare = Settings(data_dir=tmp_path)
        assert asyncio.run(model_identity(bare, props(404))) is None  # nothing names the weights: off
        reported = asyncio.run(model_identity(bare, props(body={"model_path": "C:\\models\\w.gguf"})))
    assert json.loads(reported)["file"] == "w.gguf"
    messages = [r.getMessage() for r in caplog.records]
    assert any("another model file" in m for m in messages) and any("could not ask" in m for m in messages)
    salted = Settings(data_dir=tmp_path, vision_model_file="Qwen-UD-Q4_K_XL.gguf", vision_mmproj="mmproj-BF16.gguf",
                      llm_cache_salt="2")
    assert asyncio.run(model_identity(salted, props(404))) != unknown


# --- never stored -------------------------------------------------------------------------------


@pytest.mark.parametrize("script, reason", [
    ([(200, "One. Two. Thr", "length"), (200, "One. Two. Three. Fo", "length")], "truncated"),
    ([(500, "boom", ""), (500, "boom", "")], "failed"),
    ([(200, "", "stop")], "empty"),
    ([(200, "I'm sorry, but I can't help with that.", "stop")], "refused"),
    ([(200, "Partial", "content_filter")], "not_finished"),
])
def test_a_cut_failed_empty_or_refused_reply_is_never_stored(tmp_path, script, reason):
    cache, model = store(tmp_path), FakeModel(script)
    job = cache.job(A, DOC)
    try:
        asyncio.run(summarize(model.client(), CFG, "doc.pdf", "pdf", "material", 600, 3000, cache=job))
    except httpx.HTTPStatusError:
        assert reason == "failed"
    assert job.counts[f"skipped_{reason}"] == 1 and job.counts["writes"] == 0 and cache.stats() == (0, 0)
    calls = len(model.calls)
    asyncio.run(summarize(model.client(), CFG, "doc.pdf", "pdf", "material", 600, 3000, cache=cache.job(A, DOC)))
    assert len(model.calls) == calls + 1 and cache.stats()[0] == 1  # asked again; this whole reply is stored


def test_a_description_still_cut_is_asked_again_and_its_first_pass_is_reused(tmp_path):
    cache = store(tmp_path)
    model = FakeModel([(200, TABLE, "stop"), (200, "Half", "length"), (200, "Half and more", "length")])
    (_, cut), job = describe_with(cache, model)
    assert cut and job.counts["writes"] == 1 and job.counts["skipped_truncated"] == 1
    (_, cut), job = describe_with(cache, model)
    assert not cut and len(model.calls) == 4 and job.counts["hits"] == 1 and job.counts["writes"] == 1


def test_what_counts_as_a_reply_to_keep():
    assert skip_reason("A line chart of revenue.", "stop") is None and skip_reason("NOT_A_CHART", "stop") is None
    assert skip_reason("Sorrel fields at dawn.", "stop") is None and skip_reason("I measured 3 values.", "stop") is None
    for refusal in ("I'm sorry, but I can't", "I cannot describe this image.", "Sorry, I am unable to", "As an AI, I",
                    "I am unable to view images", "I can" + chr(0x2019) + "t help with that", "  I won't do that"):
        assert skip_reason(refusal, "stop") == "refused", refusal
    assert skip_reason("  ", "stop") == "empty" and skip_reason("x", "length") == "truncated"
    assert skip_reason("x", None) == "not_finished"


# --- per owner ------------------------------------------------------------------------------------


def test_an_identical_file_of_another_account_never_uses_the_first_accounts_entries(tmp_path):
    cache, model = store(tmp_path), FakeModel()
    describe_with(cache, model, owner=A, doc=DOC)
    describe_with(cache, model, owner=B, doc=DOC2)
    assert len(model.calls) == 4
    assert dict(sql(cache.path, "SELECT owner_id, COUNT(*) FROM entries GROUP BY owner_id")) == {A: 2, B: 2}
    assert cache.purge_owner(A) == 4  # two entries, two uses
    describe_with(cache, model, owner=B, doc=DOC2)
    assert len(model.calls) == 4  # B's entries stay
    with pytest.raises(ValueError):
        cache.get("not-a-user", "k", DOC, "image_description")
    with pytest.raises(ValueError):
        cache.release_document(A, "../escape")


# --- purges ---------------------------------------------------------------------------------------


def test_deleting_a_document_removes_the_entries_no_other_document_of_the_owner_uses(tmp_path):
    s = Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None)
    reg = Registry(s.registry_path)
    cache, model = LlmCache.from_settings(s), FakeModel()
    cache.model = "m"
    describe_with(cache, model, doc=DOC)  # the same figure in both documents
    describe_with(cache, model, doc=DOC2)
    describe_with(cache, model, doc=DOC, caption="Figure 9")  # first pass shared, second only in DOC
    assert cache.stats()[0] == 3
    for doc_id in (DOC, DOC2):
        reg.add(doc_id, A, f"{doc_id[0]}.pdf", DocType.PDF, ".pdf", 1)
        make_ready(reg, doc_id)
    assert asyncio.run(delete_document(s, reg, no_points, *reg.mark_deleting(DOC, A)))
    assert cache.stats()[0] == 2
    calls = len(model.calls)
    describe_with(cache, model, doc=DOC2)
    assert len(model.calls) == calls  # DOC2 keeps its replies
    assert asyncio.run(delete_document(s, reg, no_points, *reg.mark_deleting(DOC2, A)))
    assert cache.stats() == (0, 0)
    assert b"Description" not in on_disk(cache.path) and b"x-axis" not in on_disk(cache.path)  # zeroed, WAL emptied


def test_a_failed_cache_purge_leaves_the_deletion_to_the_sweep(tmp_path, monkeypatch, clock):
    s = Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None)
    reg = Registry(s.registry_path, clock=clock)
    cache = LlmCache.from_settings(s)
    cache.model = "m"
    describe_with(cache, FakeModel(), doc=DOC)
    reg.add(DOC, A, "a.pdf", DocType.PDF, ".pdf", 1)
    make_ready(reg, DOC)
    real = LlmCache.release_document
    monkeypatch.setattr(LlmCache, "release_document", lambda self, o, d: (_ for _ in ()).throw(
        sqlite3.OperationalError("disk I/O error")))
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(delete_document(s, reg, no_points, *reg.mark_deleting(DOC, A)))
    assert reg.get_for_worker(DOC) is not None and cache.stats()[0] == 2  # the row stays deleting, hidden
    monkeypatch.setattr(LlmCache, "release_document", real)
    clock.advance(seconds=reg.lease_s + 1)
    asyncio.run(sweep(Pipeline(s, reg, RecordingIndexer(), None, WordCounter(), None), reg, 2))
    assert reg.get_for_worker(DOC) is None and cache.stats() == (0, 0)


class Recorder:
    def __init__(self):
        self.events = []

    async def record(self, action, subject=None, actor=None, *, user_id=None, **detail):
        self.events.append((action, subject))

    async def aclose(self):
        pass


class FakeQdrant:
    async def delete(self, collection, points_selector, wait=True):
        pass


@pytest.fixture
def api(monkeypatch):
    import ingest.api as api

    for path in (api.settings.files_dir / ".incoming", api.settings.assets_dir, api.settings.tables_dir,
                 api.settings.chunks_dir):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "audit", Recorder())
    monkeypatch.setattr(api, "qdrant", FakeQdrant())
    return api


def owners_in(cache: LlmCache) -> set[str]:
    return {r[0] for r in sql(cache.path, "SELECT owner_id FROM entries")}


def test_a_document_deleted_through_the_api_takes_its_cached_replies(api):
    a = new_user_id()
    client = TestClient(api.app)
    r = client.post("/v1/documents", files=[("files", ("x.txt", b"cached " + a.encode()))], headers=KEYS.headers(a))
    doc_id = r.json()[0]["document"]["doc_id"]
    make_ready(api.registry, doc_id)
    cache = LlmCache.from_settings(api.settings)
    cache.model = "m"
    describe_with(cache, FakeModel(), owner=a, doc=doc_id)
    assert a in owners_in(cache)
    assert client.delete(f"/v1/documents/{doc_id}", headers=KEYS.headers(a)).status_code == 204
    assert a not in owners_in(cache)


def test_an_account_purge_takes_its_cached_replies_and_a_failure_is_a_502_to_retry(api, monkeypatch):
    a, b = new_user_id(), new_user_id()
    cache = LlmCache.from_settings(api.settings)
    cache.model = "m"
    describe_with(cache, FakeModel(), owner=a, doc=DOC)
    describe_with(cache, FakeModel(), owner=b, doc=DOC)
    real, down = LlmCache.purge_owner, {"on": True}

    def purge(self, owner_id):
        if down["on"]:
            raise sqlite3.OperationalError("disk I/O error")
        return real(self, owner_id)

    monkeypatch.setattr(LlmCache, "purge_owner", purge)
    client, system = TestClient(api.app), KEYS.headers(new_user_id(), role="system")
    r = client.delete(f"/internal/users/{a}", headers=system)
    assert r.status_code == 502 and "cached model replies" in r.json()["detail"] and a in owners_in(cache)
    down["on"] = False
    assert client.delete(f"/internal/users/{a}", headers=system).status_code == 200  # auth calls again
    assert a not in owners_in(cache) and b in owners_in(cache)


def test_purges_and_the_sweep_do_nothing_without_a_cache_file(tmp_path):
    cache = store(tmp_path)
    assert cache.release_document(A, DOC) == 0 and cache.purge_owner(A) == 0 and cache.collect(set) == 0
    assert cache.stats() == (0, 0) and not cache.path.exists()


def test_the_sweep_removes_cached_replies_of_documents_the_registry_no_longer_has(tmp_path):
    s = Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None)
    reg = Registry(s.registry_path)
    cache, model = LlmCache.from_settings(s), FakeModel()
    cache.model = "m"
    reg.add(DOC2, A, "kept.pdf", DocType.PDF, ".pdf", 1)  # DOC has no row: deleted while a dead run cached
    describe_with(cache, model, doc=DOC, caption="only in DOC")
    describe_with(cache, model, doc=DOC2)
    asyncio.run(sweep(Pipeline(s, reg, RecordingIndexer(), None, WordCounter(), None), reg, 2))
    assert cache.stats()[0] == 2  # DOC's own description went; the shared first pass and DOC2's stay
    calls = len(model.calls)
    describe_with(cache, model, doc=DOC2)
    assert len(model.calls) == calls


def test_the_sweep_judges_only_uses_written_before_it_read_the_registry(tmp_path):
    now = {"t": 100.0}
    cache = store(tmp_path, clock=lambda: now["t"])
    describe_with(cache, FakeModel(), doc=DOC)
    now["t"] = 50.0  # a use newer than the sweep's look at the registry (a new upload of the same file) stays
    assert cache.collect(set) == 0 and cache.stats()[0] == 2
    now["t"] = 200.0
    assert cache.collect(set) == 4 and cache.stats() == (0, 0)


# --- bounded --------------------------------------------------------------------------------------


def test_the_least_recently_used_entries_go_past_the_bounds(tmp_path):
    now = {"t": 0.0}

    def tick():
        now["t"] += 1
        return now["t"]

    cache = store(tmp_path, max_entries=2, clock=tick)
    cache.put(A, "k1", DOC, "c", "one")
    cache.put(A, "k2", DOC, "c", "two")
    assert cache.get(A, "k1", DOC, "c") == ("one", False)  # k1 used again: k2 is now the oldest
    assert cache.put(A, "k3", DOC, "c", "three") == 1
    assert [k for k in ("k1", "k2", "k3") if cache.get(A, k, DOC, "c")[0]] == ["k1", "k3"]
    assert {r[0] for r in sql(cache.path, "SELECT key FROM uses")} == {"k1", "k3"}  # its uses went with it

    by_size = store(tmp_path / "size", max_bytes=10, clock=tick)
    by_size.put(A, "a", DOC, "c", "12345678")
    assert by_size.put(A, "b", DOC, "c", "123456") == 1 and by_size.stats() == (1, 6)

    lowered = store(tmp_path, max_entries=1, clock=tick)  # a bound lowered between two starts
    assert lowered.open() == (1, len("three"))


def test_a_job_counts_what_eviction_removed(tmp_path):
    cache, model = store(tmp_path, max_entries=1), FakeModel()
    _, job = describe_with(cache, model)
    assert job.summary()["evicted"] == 1 and cache.stats()[0] == 1


# --- failures -------------------------------------------------------------------------------------


def test_a_corrupt_entry_is_removed_and_the_model_is_asked(tmp_path, caplog):
    cache, model = store(tmp_path), FakeModel()
    first, _ = describe_with(cache, model)
    sql(cache.path, "UPDATE entries SET reply = 'tampered' WHERE call = 'image_description'")
    with caplog.at_level(logging.WARNING, logger="ingest.llmcache"):
        again, job = describe_with(cache, model)
    assert again == first and len(model.calls) == 3
    assert job.counts["corrupt"] == 1 and job.counts["hits"] == 1 and job.counts["writes"] == 1
    assert any("corrupt" in r.getMessage() for r in caplog.records)
    _, job = describe_with(cache, model)
    assert job.counts["hits"] == 2  # stored again, whole


def test_a_damaged_cache_file_is_a_model_call_and_starts_again(tmp_path, caplog):
    cache, model = store(tmp_path), FakeModel()
    first, _ = describe_with(cache, model)
    cache.path.write_bytes(b"this is not a database " * 200)
    with caplog.at_level(logging.WARNING, logger="ingest.llmcache"):
        again, job = describe_with(cache, model)
    assert again == first and len(model.calls) == 4 and job.counts["errors"] == 1 and job.counts["writes"] == 2
    assert any("damaged" in r.getMessage() for r in caplog.records)
    _, job = describe_with(cache, model)
    assert job.counts["hits"] == 2  # a fresh file


def test_the_worker_start_resets_a_malformed_file_and_empties_another_schema_version(tmp_path, caplog):
    cache, model = store(tmp_path), FakeModel()
    for i in range(30):
        describe_with(cache, model, caption=f"Figure {i}")
    data = bytearray(cache.path.read_bytes())
    data[4096 + 8:] = b"\xff" * (len(data) - 4096 - 8)  # every page after the first
    cache.path.write_bytes(bytes(data))
    with caplog.at_level(logging.WARNING, logger="ingest.llmcache"):
        assert cache.open() == (0, 0)
    assert any("damaged" in r.getMessage() for r in caplog.records)

    other = tmp_path / "other"
    other.mkdir()
    for statement in ("CREATE TABLE entries (whatever TEXT)", "INSERT INTO entries VALUES ('from a newer version')",
                      "PRAGMA user_version = 7"):
        sql(other / "llm_cache.db", statement)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ingest.llmcache"):
        assert store(other).open() == (0, 0)
    assert any("another schema version" in r.getMessage() for r in caplog.records)
    assert sql(other / "llm_cache.db", "PRAGMA user_version") == [(SCHEMA_VERSION,)]
    describe_with(store(other), model)
    assert store(other).stats()[0] == 2


def test_a_busy_cache_file_is_a_miss_and_the_job_stops_trying_after_three_errors(tmp_path, caplog):
    cache = store(tmp_path, timeout_s=0.05)
    cache.open()
    blocker = sqlite3.connect(cache.path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    model = FakeModel()
    try:
        with caplog.at_level(logging.WARNING, logger="ingest.llmcache"):
            (text, _), job = describe_with(cache, model)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert len(model.calls) == 2 and text.endswith(TABLE)
    assert job.counts["errors"] == 3 and job.off and job.summary()["off_after_errors"]  # lookup, write, lookup
    assert any("off for the rest of this job" in r.getMessage() for r in caplog.records)


def test_a_full_disk_skips_the_write_and_keeps_the_reply(tmp_path, monkeypatch, caplog):
    cache = store(tmp_path)
    cache.open()
    real = sqlite3.connect

    def no_room(*args, **kwargs):  # the file may not grow: SQLITE_FULL, as on a full volume
        db = real(*args, **kwargs)
        db.execute(f"PRAGMA max_page_count = {db.execute('PRAGMA page_count').fetchone()[0]}")
        return db

    monkeypatch.setattr(sqlite3, "connect", no_room)
    long = "A long description. " * 5000
    model = FakeModel([(200, TABLE, "stop"), (200, long, "stop")])
    with caplog.at_level(logging.WARNING, logger="ingest.llmcache"):
        (text, _), job = describe_with(cache, model)
    assert long.strip() in text and job.counts["writes"] == 1 and job.counts["errors"] == 1
    assert any("full" in str(getattr(r, "fields", {}).get("error", "")) for r in caplog.records)


def test_two_workers_share_one_cache_file(tmp_path):
    one, two = store(tmp_path), store(tmp_path)
    one.open()
    errors: list[BaseException] = []

    def work(cache: LlmCache, doc_id: str):
        try:
            for i in range(40):
                key = f"k{i % 10}"
                cache.put(A, key, doc_id, "c", f"reply {key}")
                assert cache.get(A, key, doc_id, "c") == (f"reply {key}", False)
        except BaseException as exc:  # noqa: BLE001 - reported by the assert below
            errors.append(exc)

    pairs = ((one, DOC), (two, DOC2), (one, DOC2), (two, DOC))
    threads = [threading.Thread(target=work, args=pair) for pair in pairs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and one.stats()[0] == 10 and two.stats()[0] == 10


# --- through the pipeline and the worker ----------------------------------------------------------


class FigureRunner:
    """A PDF with text, two figures and more text."""

    def run(self, fn, *args, timeout):
        return ParsedDocument(elements=[
            Element(kind="text", text="The report opens with its findings.", section="Intro"),
            Element(kind="image", image=png("white"), text="Figure 1: revenue", section="Results", page=2),
            Element(kind="image", image=png("navy"), text="Figure 2: costs", section="Results", page=3),
            Element(kind="text", text="Text after the figures stays in the index.", section="Results"),
        ])

    def close(self):
        pass


class RecordingIndexer:
    def __init__(self, on_replace=None):
        self.on_replace = on_replace
        self.texts: list[list[str]] = []

    async def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_images(self, images):
        return [[0.0, 1.0] for _ in images]

    async def replace(self, user_id, doc_id, payloads, text_vectors, image_vectors):
        self.texts.append([p.text for p in payloads])
        if self.on_replace:
            self.on_replace()

    async def delete(self, user_id, doc_id):
        pass


def setup(folder, cache_on: bool = True, indexer=None):
    s = Settings(data_dir=folder, audit_url=None, docling_artifacts=None)
    reg = Registry(s.registry_path)
    model = FakeModel()
    cache = LlmCache.from_settings(s)
    cache.model = "m" if cache_on else None
    pipe = Pipeline(s, reg, indexer or RecordingIndexer(), model.client(), WordCounter(), FigureRunner(), cache)
    reg.add(DOC, A, "report.pdf", DocType.PDF, ".pdf", 1)
    source = paths.source_path(s, A, DOC, ".pdf")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF")
    return s, reg, model, cache, pipe


def test_a_reindex_of_unchanged_content_makes_no_model_call(tmp_path, caplog):
    s, reg, model, cache, pipe = setup(tmp_path / "on")
    asyncio.run(pipe.process(reg.claim_next()))
    assert len(model.calls) == 5  # two figures, two passes each, and the summary
    assert reg.requeue(DOC, A)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ingest.llmcache"):
        asyncio.run(pipe.process(reg.claim_next()))
    assert len(model.calls) == 5
    first, second = pipe.indexer.texts
    assert first == second and any("Description" in t for t in first) and any(t.startswith("Summary") for t in first)
    [line] = [r for r in caplog.records if r.getMessage() == "llm cache summary"]
    assert line.fields == {"doc_id": DOC, "hits": 5, "misses": 0, "writes": 0, "skipped": 0, "errors": 0}

    _, off_reg, off_model, _, off = setup(tmp_path / "off", cache_on=False)  # the cache off: the same index
    for _ in range(2):
        asyncio.run(off.process(off_reg.claim_next()))
        off_reg.requeue(DOC, A)
    assert len(off_model.calls) == 10 and off.indexer.texts == [first, first]
    assert not (tmp_path / "off" / "llm_cache.db").exists()


def test_a_cache_that_cannot_be_used_never_fails_the_job(tmp_path, caplog):
    s, reg, model, cache, pipe = setup(tmp_path)
    s.llm_cache_path.mkdir()  # a directory where the file should be: every operation fails
    with caplog.at_level(logging.INFO, logger="ingest.llmcache"):
        asyncio.run(pipe.process(reg.claim_next()))
    assert reg.get(DOC, A).status.value == "ready" and len(model.calls) == 5
    [line] = [r for r in caplog.records if r.getMessage() == "llm cache summary"]
    assert line.fields["errors"] == 3 and line.fields["off_after_errors"]


def test_a_document_deleted_while_it_is_indexed_leaves_no_cached_replies(tmp_path):
    holder = {}
    s, reg, model, cache, pipe = setup(tmp_path, indexer=RecordingIndexer(
        on_replace=lambda: holder["reg"].mark_deleting(DOC, A)))
    holder["reg"] = reg
    assert asyncio.run(run_job(pipe, reg, Recorder(), reg.claim_next(), asyncio.Event()))
    assert len(model.calls) == 5 and reg.get_for_worker(DOC) is None and cache.stats() == (0, 0)


def test_a_job_whose_account_is_purged_meanwhile_leaves_no_cached_replies(tmp_path):
    holder = {}
    s, reg, model, cache, pipe = setup(tmp_path, indexer=RecordingIndexer(
        on_replace=lambda: holder["reg"].purge_owner(A)))
    holder["reg"] = reg
    assert asyncio.run(run_job(pipe, reg, Recorder(), reg.claim_next(), asyncio.Event()))
    assert len(model.calls) == 5 and cache.stats() == (0, 0)


def test_a_run_whose_document_was_finished_by_someone_else_releases_its_replies(tmp_path):
    s, reg, model, cache, pipe = setup(tmp_path)
    doc = reg.claim_next()
    asyncio.run(pipe.describe(doc, FigureRunner().run(None, timeout=1), pipe._job(doc)))
    assert cache.stats()[0] == 4
    reg.mark_deleting(DOC, A)
    sql(reg._path, "DELETE FROM documents WHERE doc_id = ?", DOC)  # the sweep finished it after the lease ran out
    assert asyncio.run(pipe.abandon(doc)) == "gone" and cache.stats() == (0, 0)


def test_the_worker_serves_the_cache_only_when_on_and_the_model_is_known(tmp_path, caplog):
    off = Settings(data_dir=tmp_path, llm_cache=False, vision_model_file="w.gguf")
    cache = LlmCache.from_settings(off)
    asyncio.run(open_cache(cache, off, props(body={"model_path": "/m/w.gguf"})))
    assert cache.model is None and not cache.path.exists()  # off: no file made
    on = Settings(data_dir=tmp_path, vision_model_file="w.gguf")
    asyncio.run(open_cache(cache, on, props(body={"model_path": "/m/other.gguf"})))
    assert cache.model is None  # the server runs another file
    with caplog.at_level(logging.INFO, logger="ingest.worker"):
        asyncio.run(open_cache(cache, on, props(body={"model_path": "/m/w.gguf"})))
    assert cache.model and cache.path.exists() and any(r.getMessage() == "llm cache on" for r in caplog.records)
    assert Settings(data_dir=tmp_path).llm_cache and Settings(data_dir=tmp_path).llm_cache_max_mb == 1024


def test_a_document_the_cache_cannot_take_is_indexed_without_it(tmp_path, caplog):
    _, _, _, _, pipe = setup(tmp_path)
    with caplog.at_level(logging.WARNING, logger="ingest.pipeline"):
        assert pipe._job(SimpleNamespace(doc_id="not-a-document-id", owner_id=A)) is None
    assert any("llm cache unusable" in r.getMessage() for r in caplog.records)
