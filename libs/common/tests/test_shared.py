"""Contracts, Qdrant filters and client, create_app defaults, the audit client and read_secret."""

import asyncio
import json
import logging
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import models as qm

from rag_common import identity
from rag_common.audit import AuditClient
from rag_common.contracts import (
    AuditEvent,
    DocIdPath,
    DocIdsRequest,
    DocumentInfo,
    GrepRequest,
    MigrateLegacy,
    NewMessages,
    ThreadCreate,
    ThreadUpdate,
)
from rag_common.identity import Identity, current_identity
from rag_common.index import USER_FIELD, ChunkPayload, qdrant_client, scope_filter, user_condition
from rag_common.secrets import read_secret
from rag_common.service import create_app

UID = "3f0c9a52-8d0e-4c1b-9a57-2b4e0f6d9c11"
DOC = "0123456789abcdef0123456789abcdef"


# --- contracts ------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../../etc/passwd", "x" * 32, DOC.upper(), DOC + "0", ""])
def test_every_doc_id_list_is_validated(bad):
    for build in (
        lambda ids: DocIdsRequest(doc_ids=ids),
        lambda ids: GrepRequest(doc_ids=ids, terms=["a"]),
        lambda ids: ThreadCreate(doc_ids=ids),
        lambda ids: ThreadUpdate(doc_ids=ids),
        lambda ids: NewMessages(messages=[{"role": "user", "content": "q"}], doc_ids=ids),
    ):
        assert build([DOC])
        with pytest.raises(ValidationError):
            build([DOC, bad])


def test_doc_id_path_parameter():
    app = FastAPI()

    @app.get("/v1/documents/{doc_id}")
    def get(doc_id: DocIdPath) -> dict:
        return {"doc_id": doc_id}

    client = TestClient(app)
    assert client.get(f"/v1/documents/{DOC}").json() == {"doc_id": DOC}
    assert client.get("/v1/documents/..%2F..%2Fsecret").status_code in (404, 422)
    assert client.get("/v1/documents/abc").status_code == 422


def test_audit_event_user_and_migration_contracts():
    for ok in (UID, "anonymous", "unattributed", None):
        assert AuditEvent(service="s", action="a", user_id=ok).user_id == ok
    for bad in ("legacy", "admin", UID + "x", "", UID.upper()):
        with pytest.raises(ValidationError):
            AuditEvent(service="s", action="a", user_id=bad)
    assert MigrateLegacy(owner_id=UID).dry_run is False
    with pytest.raises(ValidationError):
        MigrateLegacy(owner_id="legacy")
    doc = DocumentInfo(doc_id=DOC, name="a", doc_type="text", size=1, status="ready", created_at="t", updated_at="t")
    assert doc.owner_id is None


# --- Qdrant filters -------------------------------------------------------------------------


def test_scope_filter_always_carries_the_owner():
    flt = scope_filter(UID, [DOC])
    keys = {c.key for c in flt.must}
    assert keys == {USER_FIELD, "doc_id"}
    owner = next(c for c in flt.must if c.key == USER_FIELD)
    assert owner.match == qm.MatchValue(value=UID)
    for bad in ("", "legacy", None):
        with pytest.raises(ValueError):
            user_condition(bad)
    assert ChunkPayload(doc_id=DOC, doc_name="a", doc_type="text", kind="text", chunk_index=0, text="t",
                        search_text="t", text_sha="s", n_tokens=1).user_id is None


# --- create_app -----------------------------------------------------------------------------


def access_lines(capsys) -> list[dict]:
    """create_app configures JSON logging to stdout (replacing pytest's log handlers)."""
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    return [line for line in lines if line["logger"] == "access"]


def test_create_app_defaults_to_no_store_and_logs_the_user(capsys):
    app = create_app("svc")

    @app.get("/v1/data")
    def data() -> dict:
        return {}

    @app.get("/v1/asset")
    def asset() -> Response:
        return Response(b"png", headers={"Cache-Control": "private, max-age=300"})

    client = TestClient(app)
    r = client.get("/v1/data", headers={"x-request-id": "rid-1"})
    assert r.headers["cache-control"] == "no-store" and r.headers["x-request-id"] == "rid-1"
    assert client.get("/v1/asset").headers["cache-control"] == "private, max-age=300"
    line = access_lines(capsys)[0]
    assert line["user_id"] is None and line["path"] == "/v1/data" and line["request_id"] == "rid-1"


def test_access_log_reports_the_verified_user(capsys, keys, clock):
    from conftest import pem, raw

    verifier = identity.Verifier({"auth": raw(keys["auth"])}, {"auth"}, clock=clock)
    signer = identity.Signer(pem(keys["auth"]), "auth", clock=clock)
    app = create_app("svc", verifier=verifier)

    @app.get("/v1/data")
    def data() -> dict:
        return {}

    client = TestClient(app)
    client.get("/v1/data", headers={"x-identity": signer.sign(Identity(UID, "user"))})
    client.get("/v1/data")
    lines = access_lines(capsys)
    users, statuses = [line["user_id"] for line in lines], [line["status"] for line in lines]
    assert users == [UID, None] and statuses == [200, 401]  # a rejected request is still access-logged


# --- audit client ---------------------------------------------------------------------------


def record(client: AuditClient, *args, ident: Identity | None = None, **kwargs) -> dict:
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(201, json={})

    async def run():
        client.http._transport = httpx.MockTransport(handler)
        token = current_identity.set(ident)
        try:
            await client.record(*args, **kwargs)
        finally:
            current_identity.reset(token)

    asyncio.run(run())
    return sent


def test_audit_record_single_user_mode_is_unchanged():
    sent = record(AuditClient("http://audit", "rag"), "chat.answered", DOC, question="q")
    assert sent["actor"] == "local" and sent["user_id"] is None and sent["detail"] == {"question": "q"}


def test_audit_record_takes_the_user_from_context_or_argument():
    client = AuditClient("http://audit", "rag")
    sent = record(client, "chat.answered", DOC, ident=Identity(UID, "user", "auth"))
    assert sent["user_id"] == UID and sent["actor"] == UID
    sent = record(client, "auth.login_failed", user_id="anonymous", actor="ops", username="bob")
    assert sent["user_id"] == "anonymous" and sent["actor"] == "ops" and sent["detail"] == {"username": "bob"}


def test_audit_record_without_user_once_identities_are_enforced(monkeypatch, caplog):
    monkeypatch.setattr(identity, "_enforced", True)
    with caplog.at_level(logging.ERROR, logger="audit"):
        sent = record(AuditClient("http://audit", "threads"), "memory.added", "m1")
    assert sent["user_id"] == "unattributed" and sent["actor"] == "unattributed"
    assert any(r.getMessage() == "audit event without user" for r in caplog.records)


def _signed_write(client: AuditClient, *args, ident: Identity | None = None, **kwargs) -> dict:
    """The X-Identity of one write, verified with the public keys (None when the write was unsigned)."""
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["identity"] = request.headers.get("x-identity")
        sent["body"] = json.loads(request.content)
        return httpx.Response(201, json={})

    async def run():
        client.http._transport = httpx.MockTransport(handler)
        token = current_identity.set(ident)
        try:
            await client.record(*args, **kwargs)
        finally:
            current_identity.reset(token)

    asyncio.run(run())
    return sent


def test_audit_writes_of_auth_and_rag_carry_their_own_signature(tmp_path):
    """The audit service takes auth and rag events only with their signature: auth signs as the audit writer
    (it files events under any account, 'anonymous' included), rag as the event's user."""
    from rag_common.identity import AUDIT_WRITER_ID, Verifier
    from rag_common.testing import IdentityKeys

    keys = IdentityKeys(tmp_path)
    verify = Verifier.from_file(keys.public_keys_file, {"auth", "rag"}).verify
    auth = AuditClient("http://audit", "auth", signer=keys.signer("auth"))
    sent = _signed_write(auth, "auth.login_failed", user_id="anonymous", username="bob")
    assert verify(sent["identity"]) == Identity(AUDIT_WRITER_ID, "system", "auth")
    assert sent["body"]["user_id"] == "anonymous"
    rag = AuditClient("http://audit", "rag", signer=keys.signer("rag"))
    sent = _signed_write(rag, "chat.answered", DOC, ident=Identity(UID, "admin", "auth"))
    assert verify(sent["identity"]) == Identity(UID, "admin", "rag") and sent["body"]["user_id"] == UID
    assert _signed_write(rag, "chat.failed", user_id="unattributed")["identity"] is None  # cannot vouch: unsigned
    assert _signed_write(AuditClient("http://audit", "threads"), "memory.added", user_id=UID)["identity"] is None


# --- secrets --------------------------------------------------------------------------------


def test_read_secret(tmp_path):
    assert read_secret(None) is None
    path = tmp_path / "key"
    path.write_text("  s3cret\n")
    assert read_secret(path) == "s3cret"
    path.write_text("\n")
    with pytest.raises(ValueError):
        read_secret(path)
    with pytest.raises(FileNotFoundError):
        read_secret(tmp_path / "missing")


# --- Qdrant client --------------------------------------------------------------------------


class _QdrantStub(BaseHTTPRequestHandler):
    """Answers the client's version check and "does the collection exist", recording each request's api-key."""

    seen: list[tuple[str, str | None]] = []

    def do_GET(self):
        type(self).seen.append((self.path, self.headers.get("api-key")))
        answer = ({"title": "qdrant", "version": "1.20.0"} if self.path == "/"
                  else {"result": {"exists": True}, "status": "ok", "time": 0.0})
        body = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def qdrant_stub():
    _QdrantStub.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _QdrantStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_qdrant_client_sends_the_key_from_the_secret_file(tmp_path, qdrant_stub):
    key_file = tmp_path / "qdrant_api.key"
    key_file.write_text("k3y-from-file\n")

    async def call(path):
        client = qdrant_client(qdrant_stub, path, timeout=5)
        try:
            return await client.collection_exists("chunks")
        finally:
            await client.close()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the insecure-connection warning is silenced, nothing else is raised
        assert asyncio.run(call(key_file)) is True
    assert asyncio.run(call(None)) is True
    # the client's version check (path "/") may run in the background; the collection calls are in order
    keys = [key for path, key in _QdrantStub.seen if path == "/collections/chunks/exists"]
    assert keys == ["k3y-from-file", None]
    assert {key for _, key in _QdrantStub.seen} <= {"k3y-from-file", None}


def test_qdrant_client_fails_without_its_key_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        qdrant_client("http://qdrant:6333", tmp_path / "missing", timeout=5)
    (tmp_path / "empty").write_text("")
    with pytest.raises(ValueError):
        qdrant_client("http://qdrant:6333", tmp_path / "empty", timeout=5)
