"""Who is asking: Ed25519-signed identity assertions between services.

The auth service checks the browser session or API token and signs a short-lived assertion that the gateway
forwards as `X-Identity`; rag re-signs it for its own downstream calls. Services verify the signature with
public keys only, so a compromised container without a private key cannot act as another user.

Format: `v1.<P>.<S>`, where P = base64url(canonical JSON payload) and S = base64url(Ed25519 signature over
b"v1." + P), both without padding. Payload keys: sub, role, iss, iat, exp and optionally rid (request id).
The algorithm is fixed; there is no `alg` field. Tokens are never logged.
"""

import base64
import binascii
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from rag_common.logging import log_fields, request_id

USER_ID = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
USER_ID_RE = re.compile(USER_ID)
HEADER = "x-identity"
DEFAULT_TTL_S, MAX_TTL_S, SKEW_S = 120, 300, 5
MAX_TOKEN_LEN = 1024
MAX_RID_LEN = 64
ROLES = ("user", "admin", "system")
EXEMPT_PATHS = frozenset({"/healthz", "/readyz"})
# Subject of system calls made by an operator command when no admin account is involved (valid uuid4 form).
OPERATOR_ID = "00000000-0000-4000-8000-000000000000"
# Subject of auth's system assertions on audit event writes, and of nothing else: audit accepts auth events only
# with it, so a system assertion auth sent to another service for an /internal call cannot be replayed as one.
AUDIT_WRITER_ID = "00000000-0000-4000-8000-0000000000a1"

_VERSION = "v1"
_REQUIRED = frozenset({"sub", "role", "iss", "iat", "exp"})
_ALLOWED = _REQUIRED | {"rid"}
_B64URL = re.compile(r"^[A-Za-z0-9_-]*$")

log = logging.getLogger("identity")

Role = Literal["user", "admin", "system"]


@dataclass(frozen=True)
class Identity:
    user_id: str
    role: Role
    issuer: str | None = None


current_identity: ContextVar[Identity | None] = ContextVar("current_identity", default=None)


class IdentityError(Exception):
    """An assertion was missing, malformed, forged, expired or not acceptable here. The message is the reason."""


# --- encoding -------------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(text: str) -> bytes:
    """Strict base64url without padding: only the URL alphabet, and only the canonical encoding."""
    if not _B64URL.match(text) or len(text) % 4 == 1:
        raise IdentityError("bad base64")
    try:
        data = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise IdentityError("bad base64") from exc
    if _b64url(data) != text:  # non-zero padding bits: a second spelling of the same bytes
        raise IdentityError("bad base64")
    return data


def canonical_json(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    out = dict(pairs)
    if len(out) != len(pairs):
        raise IdentityError("duplicate payload key")
    return out


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- keys -----------------------------------------------------------------------------------


def load_public_keys(path: Path) -> dict[str, bytes]:
    """Read `{"version": 1, "keys": {"auth": "<base64 raw 32 bytes>", "rag": "..."}}`."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IdentityError(f"cannot read public keys from {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("keys"), dict):
        raise IdentityError(f"{path}: expected {{\"version\": 1, \"keys\": {{...}}}}")
    keys: dict[str, bytes] = {}
    for issuer, value in doc["keys"].items():
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, TypeError, ValueError) as exc:
            raise IdentityError(f"{path}: key for {issuer!r} is not base64") from exc
        if len(raw) != 32:
            raise IdentityError(f"{path}: key for {issuer!r} is not a raw 32-byte Ed25519 key")
        keys[issuer] = raw
    return keys


def public_key_raw(key: Ed25519PrivateKey | Ed25519PublicKey) -> bytes:
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


# --- signing and verifying ------------------------------------------------------------------


class Signer:
    def __init__(self, private_key_pem: bytes, issuer: str, ttl_s: int = DEFAULT_TTL_S,
                 clock: Callable[[], float] = time.time):
        try:
            key = serialization.load_pem_private_key(private_key_pem, password=None)
        except (ValueError, TypeError) as exc:
            raise IdentityError(f"cannot load the {issuer} signing key: {exc}") from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise IdentityError(f"the {issuer} signing key is not an Ed25519 key")
        if not 0 < ttl_s <= MAX_TTL_S:
            raise ValueError(f"ttl_s must be 1..{MAX_TTL_S}")
        self._key = key
        self.issuer = issuer
        self.ttl_s = ttl_s
        self.clock = clock

    @classmethod
    def from_file(cls, path: Path, issuer: str, **kwargs) -> "Signer":
        try:
            pem = Path(path).read_bytes()
        except OSError as exc:
            raise IdentityError(f"cannot read the {issuer} signing key at {path}: {exc}") from exc
        return cls(pem, issuer, **kwargs)

    def public_key_raw(self) -> bytes:
        return public_key_raw(self._key)

    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key_raw()).decode("ascii")

    def can_sign(self, identity: Identity) -> bool:
        """Only auth vouches for system identities; nobody vouches for a malformed one."""
        return (bool(USER_ID_RE.match(identity.user_id)) and identity.role in ROLES
                and (identity.role != "system" or self.issuer == "auth"))

    def sign(self, identity: Identity, request_id: str | None = None) -> str:
        if not self.can_sign(identity):
            raise ValueError(f"{self.issuer} cannot sign role={identity.role!r} for this subject")
        now = int(self.clock())
        payload: dict = {"sub": identity.user_id, "role": identity.role, "iss": self.issuer,
                         "iat": now, "exp": now + self.ttl_s}
        if request_id and request_id != "-":
            payload["rid"] = request_id[:MAX_RID_LEN]
        p = _b64url(canonical_json(payload).encode("utf-8"))
        signature = self._key.sign(f"{_VERSION}.{p}".encode("ascii"))
        return f"{_VERSION}.{p}.{_b64url(signature)}"


class Verifier:
    def __init__(self, public_keys: dict[str, bytes], accept: set[str] | frozenset[str],
                 clock: Callable[[], float] = time.time):
        self.accept = frozenset(accept)
        missing = sorted(self.accept - public_keys.keys())
        if missing:
            raise IdentityError(f"no public key for issuer(s): {', '.join(missing)}")
        self._keys = {iss: Ed25519PublicKey.from_public_bytes(public_keys[iss]) for iss in self.accept}
        self.clock = clock

    @classmethod
    def from_file(cls, path: Path, accept: set[str] | frozenset[str], **kwargs) -> "Verifier":
        return cls(load_public_keys(path), accept, **kwargs)

    def verify(self, token: str) -> Identity:
        # 1. shape
        if not isinstance(token, str) or len(token) > MAX_TOKEN_LEN:
            raise IdentityError("token too long")
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != _VERSION:
            raise IdentityError("bad format")
        _, p, s = parts
        # 2. payload: strict base64url, a JSON object with exactly the known keys
        try:
            payload = json.loads(_unb64url(p).decode("utf-8"), object_pairs_hook=_no_duplicates)
        except UnicodeDecodeError as exc:
            raise IdentityError("bad payload") from exc
        except ValueError as exc:  # json.JSONDecodeError
            raise IdentityError("bad payload") from exc
        if not isinstance(payload, dict):
            raise IdentityError("bad payload")
        keys = set(payload)
        if not keys <= _ALLOWED or not _REQUIRED <= keys:
            raise IdentityError("bad payload keys")
        rid = payload.get("rid")
        if rid is not None and (not isinstance(rid, str) or len(rid) > MAX_RID_LEN):
            raise IdentityError("bad request id")
        # 3. issuer: only chooses the key
        iss = payload["iss"]
        if not isinstance(iss, str) or iss not in self.accept:
            raise IdentityError("issuer not accepted")
        # 4. signature
        try:
            self._keys[iss].verify(_unb64url(s), f"{_VERSION}.{p}".encode("ascii"))
        except InvalidSignature as exc:
            raise IdentityError("bad signature") from exc
        # 5. lifetime
        iat, exp = payload["iat"], payload["exp"]
        if not _is_int(iat) or not _is_int(exp):
            raise IdentityError("bad times")
        now = self.clock()
        if not iat - SKEW_S <= now <= exp:
            raise IdentityError("expired or not yet valid")
        if not 0 < exp - iat <= MAX_TTL_S:
            raise IdentityError("lifetime too long")
        # 6. role
        role = payload["role"]
        if role not in ROLES:
            raise IdentityError("bad role")
        if role == "system" and iss != "auth":
            raise IdentityError("system role from a non-auth issuer")
        # 7. subject
        sub = payload["sub"]
        if not isinstance(sub, str) or not USER_ID_RE.match(sub):
            raise IdentityError("bad subject")
        return Identity(user_id=sub, role=role, issuer=iss)


# --- context --------------------------------------------------------------------------------


def require_user_id() -> str:
    """The caller's user id, or IdentityError: code that scopes data must fail closed without one."""
    ident = current_identity.get()
    if ident is None:
        raise IdentityError("no identity in this context")
    return ident.user_id


def signing_hook(signer: Signer) -> Callable[[httpx.Request], Awaitable[None]]:
    """httpx request hook: forward the current identity as a fresh assertion signed by this service."""

    async def sign_request(request: httpx.Request) -> None:
        ident = current_identity.get()
        if ident is None or HEADER in request.headers or not signer.can_sign(ident):
            return
        rid = request_id.get()
        request.headers[HEADER] = signer.sign(ident, None if rid == "-" else rid)

    return sign_request


_enforced = False


def identity_enforced() -> bool:
    """True once this process verifies identities (identity_middleware is installed). Until then a service
    runs in single-user mode, and code that would require an identity keeps its old behaviour."""
    return _enforced


def _deny(detail: str) -> Response:
    return JSONResponse({"detail": detail}, status_code=401, headers={"cache-control": "no-store"})


def identity_middleware(app: FastAPI, verifier: Verifier,
                        optional: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset()) -> None:
    """Require a valid X-Identity on every request except the health probes and the (method, path) pairs in
    `optional`. Sets request.state.identity and the current_identity context variable for the request."""
    global _enforced
    _enforced = True
    optional = frozenset(optional)

    @app.middleware("http")
    async def identity(request: Request, call_next) -> Response:
        request.state.identity = None
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        values = request.headers.getlist(HEADER)
        if len(values) > 1:
            log_fields(log, "identity rejected", logging.WARNING, reason="more than one header", path=request.url.path)
            return _deny("invalid identity")
        if not values:
            if (request.method, request.url.path) in optional:
                return await call_next(request)
            return _deny("authentication required")
        try:
            ident = verifier.verify(values[0])
        except IdentityError as exc:
            log_fields(log, "identity rejected", logging.WARNING, reason=str(exc), path=request.url.path)
            return _deny("invalid identity")
        request.state.identity = ident
        token = current_identity.set(ident)
        try:
            return await call_next(request)
        finally:
            current_identity.reset(token)


# --- FastAPI dependencies -------------------------------------------------------------------


def _identity(request: Request) -> Identity | None:
    return getattr(request.state, "identity", None)


def require_user(request: Request) -> Identity:
    ident = _identity(request)
    if ident is None or ident.role not in ("user", "admin"):
        raise HTTPException(401, "authentication required")
    return ident


def require_issuer(*issuers: str) -> Callable[[Request], Identity]:
    allowed = frozenset(issuers)

    def issuer_dependency(request: Request) -> Identity:
        ident = _identity(request)
        if ident is None:
            raise HTTPException(401, "authentication required")
        if ident.issuer not in allowed:
            raise HTTPException(403, "backend only")
        return ident

    return issuer_dependency


def require_system(request: Request) -> Identity:
    ident = _identity(request)
    if ident is None:
        raise HTTPException(401, "authentication required")
    if ident.role != "system" or ident.issuer != "auth":
        raise HTTPException(403, "system only")
    return ident
