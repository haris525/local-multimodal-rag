"""Per-user scoping in rag: Qdrant filters, reads by point ID, signed downstream calls, background work."""

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from conftest import KEYS
from rag_common.contracts import DocStatus, DocType, DocumentInfo
from rag_common.http import make_client
from rag_common.identity import Identity, IdentityError, Verifier, current_identity
from rag_common.index import COLLECTION, TEXT_VECTOR, USER_FIELD, ChunkPayload, point_id
from rag_common.testing import new_user_id

import rag.api as api
from rag.chat import ChatRequest
from rag.conversation import ThreadsClient, load
from rag.fusion import Candidate
from rag.retrieval import Retrieval, RowScope, SearchOptions, selection_filter

A, B = new_user_id(), new_user_id()
DOC_A, DOC_B = "a" * 32, "b" * 32
THREAD = "c" * 32


def as_user(user_id: str | None, make_coro):
    """Run a coroutine with this caller in context, as identity_middleware would set it."""
    async def run():
        token = current_identity.set(Identity(user_id, "user", "auth") if user_id else None)
        try:
            return await make_coro()
        finally:
            current_identity.reset(token)
    return asyncio.run(run())


def payload(owner: str, doc: str, i: int) -> dict:
    return ChunkPayload(doc_id=doc, doc_name=f"{doc[0]}.txt", doc_type="text", kind="text", chunk_index=i,
                        text=f"chunk {i} of {doc[0]}", search_text=f"chunk {i}", text_sha="0" * 16, n_tokens=3,
                        section="Intro", user_id=owner).model_dump(exclude_none=True)


async def two_owners() -> AsyncQdrantClient:
    """A local Qdrant holding three chunks of A's document and three of B's."""
    q = AsyncQdrantClient(location=":memory:")
    await q.create_collection(COLLECTION, vectors_config={TEXT_VECTOR: qm.VectorParams(size=2, distance=qm.Distance.COSINE)})
    await q.upsert(COLLECTION, [qm.PointStruct(id=point_id(doc, i), vector={TEXT_VECTOR: [1.0, float(i)]},
                                               payload=payload(owner, doc, i))
                                for owner, doc in ((A, DOC_A), (B, DOC_B)) for i in range(3)], wait=True)
    return q


# --- filters --------------------------------------------------------------------------------


@pytest.mark.parametrize("scope", [None, RowScope(DOC_A, "sales", (1, 2))])
@pytest.mark.parametrize("docs", [[DOC_A], [DOC_A, DOC_B]])
def test_selection_filter_always_requires_the_owner(scope, docs):
    flt = selection_filter(A, docs, scope)
    owner, inner = flt.must
    assert (owner.key, owner.match.value) == (USER_FIELD, A)
    assert flt.should is None and flt.must_not is None  # nothing at the top level can bypass the owner
    if scope and len(docs) > 1:  # the RowScope `should` lives inside the owner condition's sibling
        assert isinstance(inner, qm.Filter) and len(inner.must[-1].should) == 2
    if scope is None:
        assert [branch.must[0].match.value for branch in inner.should] == docs
        assert all(branch.must[1].is_empty.key == "generation" for branch in inner.should)


def test_selection_filter_refuses_a_missing_or_malformed_user():
    for bad in ("", "legacy", "not-a-uuid"):
        with pytest.raises(ValueError):
            selection_filter(bad, [DOC_A], None)


# --- reads by point ID ----------------------------------------------------------------------


def test_fetch_returns_only_the_callers_chunks_in_order():
    async def scenario():
        retrieval = Retrieval(await two_owners(), None, None, api.settings)
        ids = [point_id(DOC_A, 2), point_id(DOC_B, 0), point_id(DOC_A, 0)]
        return [(c.payload.doc_id, c.payload.chunk_index) for c in await retrieval.fetch(ids)]

    assert as_user(A, scenario) == [(DOC_A, 2), (DOC_A, 0)]
    assert as_user(B, scenario) == [(DOC_B, 0)]  # guessing another user's point IDs yields nothing


def test_neighbours_never_read_another_users_chunks():
    async def scenario():
        retrieval = Retrieval(await two_owners(), None, None, api.settings)
        mine = Candidate(point_id(DOC_A, 1), ChunkPayload(**payload(A, DOC_A, 1)))
        forged = Candidate("x", ChunkPayload(**payload(A, DOC_B, 1)))  # claims B's document
        found = await retrieval.neighbours([mine, forged])
        return {cid: tuple(p.chunk_index if p else None for p in pair) for cid, pair in found.items()}

    assert as_user(A, scenario) == {point_id(DOC_A, 1): (0, 2), "x": (None, None)}


class LeakyQdrant:
    """Ignores every filter, as a broken index would: rag must still drop other owners' points."""

    def __init__(self, points):
        self.points = points
        self.filters: list[qm.Filter] = []

    async def query_batch_points(self, collection, requests):
        self.filters += [r.filter for r in requests]
        return [SimpleNamespace(points=self.points) for _ in requests]

    async def query_points_groups(self, collection, **kwargs):
        self.filters.append(kwargs["query_filter"])
        return SimpleNamespace(groups=[SimpleNamespace(hits=self.points)])

    async def scroll(self, collection, scroll_filter, limit, with_payload, with_vectors):
        self.filters.append(scroll_filter)
        return self.points, None


class FakeModels:
    async def embed_query(self, text, instruction):
        return [1.0, 0.0]

    async def embed_image_query(self, text):
        return [1.0, 0.0]

    async def rerank(self, query, documents):
        return [0.9] * len(documents)


def leaky_points():
    return [SimpleNamespace(id=point_id(doc, i), payload=payload(owner, doc, i), vector={TEXT_VECTOR: [1.0, float(i)]})
            for owner, doc in ((A, DOC_A), (B, DOC_B)) for i in range(2)]


def test_search_fetch_and_neighbours_drop_foreign_payloads():
    qdrant = LeakyQdrant(leaky_points())
    retrieval = Retrieval(qdrant, FakeModels(), None, api.settings)

    async def scenario():
        result = await retrieval.search("q", [DOC_A], SearchOptions(retrievers=("dense", "bm25"), rerank=False))
        fetched = await retrieval.fetch([p.id for p in qdrant.points])
        near = await retrieval.neighbours([Candidate(point_id(DOC_A, 1), ChunkPayload(**payload(A, DOC_A, 1)))])
        return result, fetched, near

    result, fetched, near = as_user(A, scenario)
    assert result.candidates and {c.payload.user_id for c in result.candidates} == {A}
    assert {c.payload.user_id for c in fetched} == {A}
    assert near[point_id(DOC_A, 1)][0].user_id == A
    for flt in qdrant.filters:  # every request carried the owner condition as well
        assert any(getattr(c, "key", None) == USER_FIELD and c.match.value == A for c in flt.must)


def test_reads_without_an_identity_fail_closed():
    retrieval = Retrieval(LeakyQdrant(leaky_points()), FakeModels(), None, api.settings)
    for make in (lambda: retrieval.fetch([point_id(DOC_A, 0)]),
                 lambda: retrieval.search("q", [DOC_A], SearchOptions(retrievers=("dense",), rerank=False)),
                 lambda: retrieval.neighbours([Candidate(point_id(DOC_A, 1), ChunkPayload(**payload(A, DOC_A, 1)))])):
        with pytest.raises(IdentityError):
            as_user(None, make)


# --- signed calls to threads ----------------------------------------------------------------


class ThreadsServer:
    """A fake threads service that verifies rag's signature and answers per signed user."""

    def __init__(self):
        self.verifier = Verifier.from_file(KEYS.public_keys_file, {"rag"})
        self.memories = {A: ["The user works in finance."], B: ["The user lives in Oslo."]}
        self.seen: list[tuple[str, str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        ident = self.verifier.verify(request.headers["x-identity"])
        self.seen.append((request.method, request.url.path, ident.user_id))
        path = request.url.path
        if path.endswith("/context"):
            return httpx.Response(200, json={"summary": "", "summary_upto": 0, "messages": []})
        if path.endswith("/messages"):
            return httpx.Response(201, json=[])
        if path == "/v1/memories" and request.method == "GET":
            return httpx.Response(200, json=[{"memory_id": f"{i:032x}", "text": t, "source": "auto", "created_at": "t"}
                                             for i, t in enumerate(self.memories.get(ident.user_id, []))])
        if path == "/v1/memories" and request.method == "POST":
            text = json.loads(request.content)["text"]
            self.memories.setdefault(ident.user_id, []).append(text)
            return httpx.Response(201, json={"memory_id": "f" * 32, "text": text, "source": "auto", "created_at": "t"})
        return httpx.Response(404, json={"detail": "not found"})


def threads_client(server: ThreadsServer) -> ThreadsClient:
    http = make_client("http://threads:8006", signer=KEYS.signer("rag"))
    http._transport = httpx.MockTransport(server)  # keeps the request hooks: request id and signature
    return ThreadsClient(http)


def test_load_uses_only_the_callers_memories():
    server = ThreadsServer()
    threads = threads_client(server)
    texts = {uid: [m.text for m in as_user(uid, lambda: load(threads, THREAD, [])).memories] for uid in (A, B)}
    assert texts == {A: ["The user works in finance."], B: ["The user lives in Oslo."]}


class FakeAudit:
    def __init__(self):
        self.events = []

    async def record(self, action, subject=None, actor=None, *, user_id=None, **detail):
        self.events.append((action, current_identity.get()))


class MemoryLLM:
    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return {"content": json.dumps({"memories": ["The user prefers short answers."]}) if json_mode else "summary"}


def test_background_maintenance_keeps_the_callers_identity():
    server, audit = ThreadsServer(), FakeAudit()
    api.state.update(audit=audit, deps=SimpleNamespace(threads=threads_client(server), llm=MemoryLLM(), s=api.settings))

    async def scenario():
        token = current_identity.set(Identity(A, "user", "auth"))

        async def events():
            yield "sources", {"route": "search", "sources": []}
            yield "token", {"text": "Answer."}
            yield "done", {"citations": []}

        async for _ in api._sse(events(), ChatRequest(question="Keep it short please", thread_id=THREAD), []):
            pass
        current_identity.reset(token)  # the request is over; maintenance and audit still run
        assert current_identity.get() is None
        await asyncio.gather(*api._background)

    asyncio.run(scenario())
    assert ("GET", f"/v1/threads/{THREAD}/context", A) in server.seen  # compaction check, in the background
    assert ("POST", "/v1/memories", A) in server.seen  # the extracted memory is saved for A
    assert {uid for _, _, uid in server.seen} == {A}
    assert server.memories[A][-1] == "The user prefers short answers." and len(server.memories[B]) == 1
    assert audit.events[0][1].user_id == A


# --- API ------------------------------------------------------------------------------------


def doc(doc_id: str, owner: str) -> DocumentInfo:
    return DocumentInfo(doc_id=doc_id, name=f"{doc_id[0]}.txt", doc_type=DocType.TEXT, size=1, status=DocStatus.READY,
                        n_chunks=3, created_at="t", updated_at="t", owner_id=owner)


class FakeIngest:
    def __init__(self, docs):
        self.docs = docs

    async def documents(self, doc_ids):
        return [d for d in self.docs if d.doc_id in doc_ids]


@pytest.fixture
def client():
    api.state.update(deps=SimpleNamespace(ingest=FakeIngest([doc(DOC_A, A), doc(DOC_B, B)])), audit=FakeAudit())
    return TestClient(api.app)  # no lifespan: the fakes above stand in for the real clients


def test_every_route_needs_an_identity_from_auth(client):
    body = {"question": "q", "doc_ids": [DOC_A]}
    assert client.post("/v1/chat", json=body).status_code == 401
    assert client.post("/v1/search", json=body).status_code == 401
    assert client.post("/v1/search", json=body, headers=KEYS.headers(A, issuer="rag")).status_code == 401  # only auth
    assert client.post("/v1/search", json=body, headers=KEYS.headers(A, role="system")).status_code == 401
    assert client.get("/healthz").status_code == 200


def test_documents_of_another_user_are_404_before_anything_runs(client):
    for doc_ids in ([DOC_B], [DOC_A, DOC_B]):
        for path in ("/v1/chat", "/v1/search"):
            r = client.post(path, json={"question": "q", "doc_ids": doc_ids}, headers=KEYS.headers(A))
            assert r.status_code == 404, (path, doc_ids)


def test_a_document_listed_with_the_wrong_owner_is_still_404(client):
    api.state["deps"] = SimpleNamespace(ingest=FakeIngest([doc(DOC_B, B)]))  # an ingest bug: lists B's file to A
    r = client.post("/v1/search", json={"question": "q", "doc_ids": [DOC_B]}, headers=KEYS.headers(A))
    assert r.status_code == 404


def test_capabilities_are_for_auth_system_calls_only(client):
    assert client.get("/internal/capabilities", headers=KEYS.headers(A, role="system")).json() == {
        "service": "rag", "multi_user": 1, "schema": 1}
    assert client.get("/internal/capabilities", headers=KEYS.headers(A)).status_code == 403


def test_the_signing_key_must_match_the_published_rag_key(tmp_path):
    other = KEYS.public_keys_file.read_text()
    doc_ = json.loads(other)
    doc_["keys"]["rag"] = base64.b64encode(b"\x01" * 32).decode()
    (tmp_path / "pub.json").write_text(json.dumps(doc_))
    good = SimpleNamespace(signing_key_file=KEYS.key_files["rag"], public_keys_file=KEYS.public_keys_file)
    assert api.load_signer(good).issuer == "rag"
    with pytest.raises(IdentityError):
        api.load_signer(SimpleNamespace(signing_key_file=KEYS.key_files["rag"], public_keys_file=tmp_path / "pub.json"))
