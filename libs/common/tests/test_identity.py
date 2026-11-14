import asyncio
import base64
import json

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from conftest import pem, raw
from rag_common import identity
from rag_common.identity import (
    HEADER,
    Identity,
    IdentityError,
    Signer,
    Verifier,
    current_identity,
    require_issuer,
    require_system,
    require_user,
    require_user_id,
    signing_hook,
)
from rag_common.logging import request_id
from rag_common.service import create_app

UID = "3f0c9a52-8d0e-4c1b-9a57-2b4e0f6d9c11"
OTHER = "7b1d2e3f-4a5b-4c6d-8e7f-0a1b2c3d4e5f"


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forge(key: Ed25519PrivateKey, payload: dict | bytes) -> str:
    """Sign an arbitrary payload, to test checks that come after the signature."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    p = b64(body)
    return f"v1.{p}.{b64(key.sign(f'v1.{p}'.encode()))}"


@pytest.fixture
def signer(keys, clock) -> Signer:
    return Signer(pem(keys["auth"]), "auth", clock=clock)


@pytest.fixture
def verifier(keys, clock) -> Verifier:
    return Verifier({k: raw(v) for k, v in keys.items()}, {"auth", "rag"}, clock=clock)


def payload(clock, **over) -> dict:
    base = {"sub": UID, "role": "user", "iss": "auth", "iat": int(clock.now), "exp": int(clock.now) + 120}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


# --- tokens ---------------------------------------------------------------------------------


def test_round_trip_and_format(signer, verifier):
    token = signer.sign(Identity(UID, "admin"), request_id="r" * 80)
    assert token.startswith("v1.") and token.count(".") == 2 and "=" not in token
    assert verifier.verify(token) == Identity(UID, "admin", "auth")
    body = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
    assert set(body) == {"sub", "role", "iss", "iat", "exp", "rid"} and len(body["rid"]) == 64
    assert body["exp"] - body["iat"] == 120


def test_from_files(tmp_path, keys, public_keys_file, clock):
    key_file = tmp_path / "auth_signing.key"
    key_file.write_bytes(pem(keys["auth"]))
    signer = Signer.from_file(key_file, "auth", clock=clock)
    assert base64.b64decode(signer.public_key_b64()) == raw(keys["auth"])
    verifier = Verifier.from_file(public_keys_file, {"auth"}, clock=clock)
    assert verifier.verify(signer.sign(Identity(UID, "user"))).user_id == UID
    with pytest.raises(IdentityError, match="no public key"):
        Verifier({"auth": raw(keys["auth"])}, {"auth", "rag"})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1, "keys": {"auth": base64.b64encode(b"short").decode()}}))
    with pytest.raises(IdentityError, match="32-byte"):
        Verifier.from_file(bad, {"auth"})


def test_tampered_payload_or_signature(signer, verifier, keys, clock):
    token = signer.sign(Identity(UID, "user"))
    v, p, s = token.split(".")
    other = b64(json.dumps(payload(clock, sub=OTHER), separators=(",", ":"), sort_keys=True).encode())
    with pytest.raises(IdentityError, match="signature"):
        verifier.verify(f"{v}.{other}.{s}")
    flipped = s[:-2] + ("A" if s[-2] != "A" else "B") + s[-1]
    with pytest.raises(IdentityError):
        verifier.verify(f"{v}.{p}.{flipped}")
    with pytest.raises(IdentityError, match="signature"):  # signed by rag's key but claiming iss=auth
        verifier.verify(forge(keys["rag"], payload(clock)))


def test_wrong_or_unaccepted_issuer(keys, clock):
    rag = Signer(pem(keys["rag"]), "rag", clock=clock)
    only_auth = Verifier({k: raw(v) for k, v in keys.items()}, {"auth"}, clock=clock)
    with pytest.raises(IdentityError, match="issuer"):
        only_auth.verify(rag.sign(Identity(UID, "user")))
    with pytest.raises(IdentityError, match="issuer"):
        only_auth.verify(forge(keys["auth"], payload(clock, iss="other")))


def test_lifetime(signer, verifier, keys, clock):
    token = signer.sign(Identity(UID, "user"))
    clock.now += 121
    with pytest.raises(IdentityError, match="expired"):
        verifier.verify(token)
    clock.now -= 121
    with pytest.raises(IdentityError, match="not yet valid"):  # iat in the future beyond the skew
        verifier.verify(forge(keys["auth"], payload(clock, iat=int(clock.now) + 6, exp=int(clock.now) + 100)))
    assert verifier.verify(forge(keys["auth"], payload(clock, iat=int(clock.now) + 5, exp=int(clock.now) + 100)))
    with pytest.raises(IdentityError, match="lifetime"):
        verifier.verify(forge(keys["auth"], payload(clock, exp=int(clock.now) + 301)))
    with pytest.raises(IdentityError, match="times"):
        verifier.verify(forge(keys["auth"], payload(clock, iat=True)))
    with pytest.raises(ValueError):
        Signer(pem(keys["auth"]), "auth", ttl_s=301)


@pytest.mark.parametrize("token", [
    "", "v1", "v1.a", "v1.a.b.c", "v2.eyJ9.AAAA", "v1.e30=.AAAA", "v1.e+0.AAAA", "x" * 1025,
])
def test_malformed_tokens(verifier, token):
    with pytest.raises(IdentityError):
        verifier.verify(token)


def test_malformed_payloads(verifier, keys, clock):
    key = keys["auth"]
    for body in (b"not json", b"[1,2]", b"\xff\xfe", b'{"sub":"a","sub":"b"}'):
        with pytest.raises(IdentityError, match="payload"):
            verifier.verify(forge(key, body))
    with pytest.raises(IdentityError, match="keys"):  # extra key
        verifier.verify(forge(key, {**payload(clock), "admin": True}))
    with pytest.raises(IdentityError, match="keys"):  # missing key
        verifier.verify(forge(key, payload(clock, exp=None)))
    with pytest.raises(IdentityError, match="request id"):
        verifier.verify(forge(key, payload(clock, rid="r" * 65)))
    # a second base64 spelling of a valid payload (unused low bits set in the last character) is rejected
    body = json.dumps(payload(clock), separators=(",", ":"), sort_keys=True).encode()
    while len(body) % 3 != 1:  # one leftover byte: two characters, the last one carries 4 unused bits
        body = body[:-1] + b" }"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    p = b64(body)
    assert verifier.verify(f"v1.{p}.{b64(key.sign(f'v1.{p}'.encode()))}")
    p2 = p[:-1] + alphabet[alphabet.index(p[-1]) + 1]
    assert base64.urlsafe_b64decode(p2 + "==") == body
    with pytest.raises(IdentityError, match="base64"):
        verifier.verify(f"v1.{p2}.{b64(key.sign(f'v1.{p2}'.encode()))}")


def test_roles_and_subject(verifier, keys, clock):
    with pytest.raises(IdentityError, match="system role"):
        verifier.verify(forge(keys["rag"], payload(clock, iss="rag", role="system")))
    assert verifier.verify(forge(keys["auth"], payload(clock, role="system"))).role == "system"
    with pytest.raises(IdentityError, match="role"):
        verifier.verify(forge(keys["auth"], payload(clock, role="root")))
    for sub in ("not-a-uuid", UID.upper(), "3f0c9a52-8d0e-1c1b-9a57-2b4e0f6d9c11", 7):
        with pytest.raises(IdentityError, match="subject"):
            verifier.verify(forge(keys["auth"], payload(clock, sub=sub)))


def test_signer_refuses_what_it_cannot_vouch_for(keys, clock):
    rag = Signer(pem(keys["rag"]), "rag", clock=clock)
    with pytest.raises(ValueError):
        rag.sign(Identity(UID, "system"))
    with pytest.raises(ValueError):
        rag.sign(Identity("bob", "user"))
    with pytest.raises(IdentityError):
        Signer(b"not a key", "rag")


# --- context and httpx hook -----------------------------------------------------------------


def test_require_user_id_raises_when_unset():
    with pytest.raises(IdentityError):
        require_user_id()
    token = current_identity.set(Identity(UID, "user", "auth"))
    try:
        assert require_user_id() == UID
    finally:
        current_identity.reset(token)


def test_signing_hook_signs_only_with_an_identity(keys, clock, verifier):
    rag = Signer(pem(keys["rag"]), "rag", clock=clock)
    hook = signing_hook(rag)

    async def run(ident: Identity | None, headers: dict | None = None) -> httpx.Request:
        req = httpx.Request("GET", "http://threads/v1/memories", headers=headers)
        token = current_identity.set(ident)
        rid = request_id.set("req-1")
        try:
            await hook(req)
        finally:
            current_identity.reset(token)
            request_id.reset(rid)
        return req

    assert HEADER not in asyncio.run(run(None)).headers
    signed = asyncio.run(run(Identity(UID, "user", "auth")))
    assert verifier.verify(signed.headers[HEADER]) == Identity(UID, "user", "rag")
    assert asyncio.run(run(Identity(UID, "user", "auth"), {HEADER: "kept"})).headers[HEADER] == "kept"
    assert HEADER not in asyncio.run(run(Identity(UID, "system", "auth"))).headers  # rag cannot vouch for system


def test_make_client_signs_from_context(keys, clock, verifier):
    from rag_common.http import make_client

    seen = {}

    async def run():
        client = make_client("http://threads", signer=Signer(pem(keys["rag"]), "rag", clock=clock))
        client._transport = httpx.MockTransport(lambda r: seen.update(r.headers) or httpx.Response(204))
        token = current_identity.set(Identity(UID, "admin", "auth"))
        try:
            await client.get("/v1/memories")
        finally:
            current_identity.reset(token)
            await client.aclose()

    asyncio.run(run())
    assert verifier.verify(seen[HEADER]) == Identity(UID, "admin", "rag")


# --- middleware and dependencies ------------------------------------------------------------


def app_with(verifier: Verifier, optional=frozenset()) -> FastAPI:
    app = create_app("test", verifier=verifier, identity_optional=optional)

    @app.get("/v1/me")
    def me(request: Request, user: Identity = Depends(require_user)) -> dict:
        ctx = current_identity.get()
        return {"user": user.user_id, "issuer": user.issuer, "ctx": ctx.user_id if ctx else None}

    @app.post("/v1/events")
    def events(request: Request) -> dict:
        ident = request.state.identity
        return {"signed": ident.user_id if ident else None}

    @app.get("/v1/backend")
    def backend(user: Identity = Depends(require_issuer("rag"))) -> dict:
        return {"ok": True}

    @app.get("/internal/capabilities")
    def capabilities(_: Identity = Depends(require_system)) -> dict:
        return {"multi_user": 1}

    return app


def test_middleware_rules(signer, verifier, keys, clock):
    client = TestClient(app_with(verifier, {("POST", "/v1/events")}))
    token = signer.sign(Identity(UID, "user"))
    assert client.get("/healthz").status_code == 200  # exempt
    assert client.get("/readyz").status_code == 200
    r = client.get("/v1/me")
    assert r.status_code == 401 and r.json() == {"detail": "authentication required"}
    assert r.headers["cache-control"] == "no-store"
    r = client.get("/v1/me", headers={HEADER: "v1.junk.junk"})
    assert r.status_code == 401 and r.json() == {"detail": "invalid identity"}
    r = client.get("/v1/me", headers=[(HEADER, token), (HEADER, token)])
    assert r.status_code == 401
    r = client.get("/v1/me", headers={HEADER: token})
    assert r.json() == {"user": UID, "issuer": "auth", "ctx": UID}
    assert current_identity.get() is None  # reset after the request
    # optional route: unsigned is allowed, a bad signature is not
    assert client.post("/v1/events").json() == {"signed": None}
    assert client.post("/v1/events", headers={HEADER: token}).json() == {"signed": UID}
    assert client.post("/v1/events", headers={HEADER: token[:-4] + "AAAA"}).status_code == 401
    assert identity.identity_enforced()


def test_dependencies(signer, verifier, keys, clock):
    client = TestClient(app_with(verifier))
    rag = Signer(pem(keys["rag"]), "rag", clock=clock)
    user = signer.sign(Identity(UID, "user"))
    system = signer.sign(Identity(UID, "system"))
    assert client.get("/v1/backend", headers={HEADER: user}).status_code == 403
    assert client.get("/v1/backend", headers={HEADER: rag.sign(Identity(UID, "user"))}).status_code == 200
    assert client.get("/v1/me", headers={HEADER: system}).status_code == 401  # system is not a user
    assert client.get("/internal/capabilities", headers={HEADER: user}).status_code == 403
    assert client.get("/internal/capabilities", headers={HEADER: system}).json() == {"multi_user": 1}


def test_no_verifier_means_no_identity_checks():
    app = create_app("plain")

    @app.get("/v1/open")
    def open_route(request: Request) -> dict:
        return {"identity": getattr(request.state, "identity", None)}

    @app.get("/v1/needs-user")
    def needs_user(user: Identity = Depends(require_user)) -> dict:
        return {}

    client = TestClient(app)
    assert client.get("/v1/open").json() == {"identity": None}
    assert client.get("/v1/needs-user").status_code == 401  # dependencies still fail closed
    assert not identity.identity_enforced()
