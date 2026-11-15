"""Every retrieval/tool arm uses the registry snapshot captured before the answer starts."""

import asyncio
import json

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models as qm

from rag_common.contracts import DocStatus, DocType, DocumentInfo
from rag_common.index import COLLECTION, TEXT_VECTOR, point_id
from rag_common.snapshots import current_generations, published_snapshots, selected_generations

from test_scoping import A, B, DOC_A, FakeModels, as_user, payload
from rag.clients import IngestClient, SqlClient
from rag.retrieval import Retrieval, SearchOptions

OLD, NEW, STAGED = "1" * 32, "2" * 32, "3" * 32


def document(generation):
    return DocumentInfo(doc_id=DOC_A, name="example.txt", doc_type=DocType.TEXT, size=1,
                        status=DocStatus.READY, created_at="t", updated_at="t", owner_id=A, generation=generation)


@pytest.mark.parametrize("generation", [None, OLD, NEW])
def test_search_fetch_neighbours_and_exhaustive_reads_use_only_the_pinned_generation(generation):
    async def scenario():
        qdrant = AsyncQdrantClient(location=":memory:")
        await qdrant.create_collection(COLLECTION, vectors_config={TEXT_VECTOR: qm.VectorParams(size=2, distance=qm.Distance.COSINE)})
        points = [qm.PointStruct(id=point_id(DOC_A, i, version), vector={TEXT_VECTOR: [1.0, float(i)]},
                                 payload={**payload(A, DOC_A, i), **({"generation": version} if version else {})})
                  for version in (None, OLD, NEW, STAGED) for i in range(3)]
        foreign = qm.PointStruct(id="9" * 32, vector={TEXT_VECTOR: [1.0, 0.0]}, payload=payload(B, DOC_A, 0))
        await qdrant.upsert(COLLECTION, [*points, foreign])
        from rag.config import Settings
        retrieval = Retrieval(qdrant, FakeModels(), None, Settings())
        try:
            with published_snapshots([document(generation)]):
                found = await retrieval.search("q", [DOC_A], SearchOptions(retrievers=("dense",), rerank=False,
                                                                           threshold=False, coverage=False))
                fetched = await retrieval.fetch([str(point.id) for point in [*points, foreign]])
                all_chunks = await retrieval.document_chunks([DOC_A], page=1)
                assert found.candidates and fetched and len(all_chunks) == 3
                assert {item.payload.generation for item in [*found.candidates, *fetched]} == {generation}
                assert {item[1].generation for item in all_chunks} == {generation}
                middle = next(item for item in fetched if item.payload.chunk_index == 1)
                neighbours = (await retrieval.neighbours([middle]))[middle.id]
                assert [item.chunk_index for item in neighbours] == [0, 2]
                assert all(item.generation == generation for item in neighbours)
        finally:
            await qdrant.close()
    as_user(A, scenario)
    assert current_generations.get() is None


@pytest.mark.parametrize("generation", [None, OLD])
def test_sql_python_download_grep_chunks_and_assets_preserve_the_captured_snapshot(generation):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/v1/query":
            return httpx.Response(200, json={"columns": ["n"], "rows": [[1]], "truncated": False})
        if request.url.path in ("/v1/grep", f"/v1/documents/{DOC_A}/chunks"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, content=b"fixture", headers={"content-length": "7"})

    async def scenario():
        async with httpx.AsyncClient(base_url="http://fixture", transport=httpx.MockTransport(handler)) as http:
            ingest, sql = IngestClient(http), SqlClient(http)
            doc = document(generation)
            with published_snapshots([doc]):
                doc.generation = NEW  # a later metadata refresh must not mutate this answer's selection
                await sql.query(DOC_A, "SELECT 1")
                assert await sql.tables_file(DOC_A, 100) == b"fixture"
                await ingest.grep([DOC_A], ["needle"], 1)
                await ingest.chunks(DOC_A, 0, 1)
                assert await ingest.asset_data_uri(DOC_A, "0" * 16 + ".png")
    asyncio.run(scenario())
    for request in requests:
        if request.url.path == "/v1/query":
            assert json.loads(request.content)["generation"] == (generation or "legacy")
        elif request.url.path == "/v1/grep":
            assert json.loads(request.content)["generations"] == {DOC_A: generation}
        else:
            assert request.url.params["generation"] == (generation or "legacy")
    assert current_generations.get() is None


def test_concurrent_questions_keep_separate_snapshots_and_refuse_unselected_documents():
    async def request(generation):
        with published_snapshots([document(generation)]):
            await asyncio.sleep(0)
            assert selected_generations([DOC_A]) == {DOC_A: generation}
            with pytest.raises(ValueError, match="outside"):
                selected_generations(["f" * 32])

    async def scenario():
        await asyncio.gather(request(OLD), request(NEW), request(None))
    asyncio.run(scenario())
    assert current_generations.get() is None


def test_stream_snapshot_survives_tokens_and_is_reset_on_disconnect(monkeypatch):
    import rag.api as api
    from rag.chat import ChatRequest

    async def finish(*args):
        pass

    monkeypatch.setattr(api, "_finish", finish)

    async def events():
        assert selected_generations([DOC_A]) == {DOC_A: OLD}
        yield "token", {"text": "partial"}
        await asyncio.sleep(0)
        assert selected_generations([DOC_A]) == {DOC_A: OLD}
        yield "token", {"text": "next"}

    async def scenario():
        selected = document(OLD)
        stream = api._sse(events(), ChatRequest(question="q", doc_ids=[DOC_A]), [selected])
        assert "partial" in await anext(stream)
        assert current_generations.get() is None
        selected.generation = NEW
        assert "next" in await anext(stream)
        await stream.aclose()
        await asyncio.sleep(0)
        assert current_generations.get() is None
    asyncio.run(scenario())


def test_stream_can_be_closed_by_a_different_task_without_losing_cleanup(monkeypatch):
    import rag.api as api
    from rag.chat import ChatRequest
    from rag_common.chat_events import ChatEvent

    finished = []

    async def finish(*args, **kwargs):
        finished.append(True)

    async def events():
        assert selected_generations([DOC_A]) == {DOC_A: OLD}
        yield ChatEvent.TOKEN, {"text": "partial"}
        yield ChatEvent.DONE, {}

    monkeypatch.setattr(api, "_finish", finish)

    async def scenario():
        stream = api._sse(events(), ChatRequest(question="q", doc_ids=[DOC_A]), [document(OLD)])
        await anext(stream)
        await asyncio.create_task(stream.aclose())  # disconnect/finalization can run in another task
        await asyncio.sleep(0)
        assert finished and current_generations.get() is None

    asyncio.run(scenario())
