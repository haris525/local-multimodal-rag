"""audit: append-only, hash-chained record of document, chat and account activity, per user.

Writes (POST /v1/events) come from the services over the backend network: auth and rag sign theirs (auth as
the audit writer, rag as the event's user), the services without a signing key write unsigned under their own
names (see _filed_under). Reads go through the gateway with an identity signed by auth:
a user sees only their own events and verifies the chain against their own count. /internal/* is for auth's
system calls: the whole-chain check and the sign-in events for admins, and crypto-shredding a deleted account.
"""

import os
from pathlib import Path

from fastapi import Depends, Header, HTTPException, Path as PathParam, Query, Request

from rag_common.contracts import AuditEvent, MigrateLegacy
from rag_common.identity import (
    AUDIT_WRITER_ID,
    USER_ID,
    Identity,
    Verifier,
    require_issuer,
    require_system,
    require_user,
)
from rag_common.service import create_app

from audit.store import SCHEMA_VERSION, AuditStore, Conflict

store = AuditStore(Path(os.getenv("AUDIT_DB", "/data/audit.db")))
PUBLIC_KEYS = Path(os.getenv("AUDIT_PUBLIC_KEYS_FILE", "/run/secrets/identity_public_keys"))
app = create_app("audit", log_level=os.getenv("AUDIT_LOG_LEVEL", "INFO"),
                 verifier=Verifier.from_file(PUBLIC_KEYS, {"auth", "rag"}),
                 identity_optional={("POST", "/v1/events")})
FROM_AUTH = Depends(require_issuer("auth"))


SIGNED_BY = {"auth": "auth", "rag": "rag"}  # services that hold a signing key: their events need its signature
KEYLESS = frozenset({"ingest-api", "ingest-worker", "threads"})  # no key: unsigned, under these names only


def _filed_under(event: AuditEvent, signed: Identity | None) -> str:
    """Who may write which events; returns the user the event is filed under.

    - auth events only from auth itself: signed by auth as the audit writer (a system assertion with
      AUDIT_WRITER_ID, which auth sends nowhere else); auth files them under any account.
    - rag events only signed by rag, for the user in the signature.
    - the services without a key (ingest-api, ingest-worker, threads) write unsigned, under their own names.
      Anything on the backend network can still claim one of these names (the residual risk), but no
      longer write sign-in, account or chat events, which is what admins and users rely on.
    """
    issuer = SIGNED_BY.get(event.service)
    if signed is not None and signed.issuer != issuer:
        raise HTTPException(403, f"an event signed by {signed.issuer} must be one of its own")
    if issuer is None:
        if event.service not in KEYLESS:
            raise HTTPException(403, "unknown service")
        return event.user_id or "unattributed"
    if signed is None:
        raise HTTPException(403, f"{event.service} events must be signed by {issuer}")
    if issuer == "auth":
        if signed.role != "system" or signed.user_id != AUDIT_WRITER_ID:
            raise HTTPException(403, "auth events come from the auth service only")
        return event.user_id or "unattributed"
    if event.user_id and event.user_id != signed.user_id:
        raise HTTPException(403, "user_id does not match the signed identity")
    return signed.user_id


@app.post("/v1/events", status_code=201)
def append(event: AuditEvent, request: Request, x_request_id: str | None = Header(default=None)) -> dict:
    user_id = _filed_under(event, request.state.identity)
    row = store.append(event.service, event.action, event.actor, x_request_id, event.subject, event.detail, user_id)
    return {"user_seq": row["user_seq"]}


@app.get("/v1/events", dependencies=[FROM_AUTH])
def events(
    limit: int = Query(100, ge=1, le=1000),
    before: int | None = Query(None, ge=1),
    action: str | None = Query(None, max_length=60),
    subject: str | None = Query(None, max_length=200),
    user: Identity = Depends(require_user),
) -> list[dict]:
    return store.list_for(user.user_id, limit, before, action, subject)


@app.get("/v1/verify", dependencies=[FROM_AUTH])
def verify(user: Identity = Depends(require_user)) -> dict:
    return store.verify_for(user.user_id)


# --- auth's system calls --------------------------------------------------------------------


@app.get("/internal/capabilities")
def capabilities(_: Identity = Depends(require_system)) -> dict:
    return {"service": "audit", "multi_user": 1, "schema": SCHEMA_VERSION}


@app.get("/internal/verify")
def verify_all(_: Identity = Depends(require_system)) -> dict:
    return store.verify()


@app.get("/internal/security-events")
def security_events(limit: int = Query(100, ge=1, le=500), before: int | None = Query(None, ge=1),
                    _: Identity = Depends(require_system)) -> list[dict]:
    return store.security_events(limit, before)


@app.delete("/internal/users/{user_id}")
def purge_user(user_id: str = PathParam(pattern=USER_ID), _: Identity = Depends(require_system)) -> dict:
    """Crypto-shred a deleted account's event details. Repeatable."""
    return store.purge_user(user_id)


@app.post("/internal/migrate-legacy")
def migrate_legacy(body: MigrateLegacy, _: Identity = Depends(require_system)) -> dict:
    """The rows from before accounts existed become this user's first events (set once)."""
    try:
        return store.adopt_legacy(body.owner_id, body.dry_run)
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
