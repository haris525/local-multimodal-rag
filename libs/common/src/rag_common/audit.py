"""Best-effort audit client: a failed write never fails the user's request; it is logged at ERROR so it
shows up in monitoring. Each write waits for the audit service for at most 5 s.

Services that hold a signing key (auth, rag) pass it, and every write then carries an X-Identity signed by
that service: the audit service accepts auth and rag events only with their own signature. auth signs as
the audit writer (a system assertion: it files events under any account, 'anonymous' sign-in failures
included); rag signs as the event's user. The other services hold no key and write unsigned, under their own
service names only (see services/audit/src/audit/api.py)."""

import logging

import httpx

from rag_common.contracts import AuditEvent
from rag_common.http import ServiceError, make_client
from rag_common.identity import AUDIT_WRITER_ID, HEADER, Identity, Signer, current_identity, identity_enforced
from rag_common.logging import log_fields, request_id

log = logging.getLogger("audit")


class AuditClient:
    def __init__(self, base_url: str | None, service: str, signer: Signer | None = None):
        self.service = service
        self.signer = signer
        self.http = make_client(base_url, timeout=5.0, connect_retries=2) if base_url else None

    def _signature(self, uid: str | None) -> dict[str, str]:
        """The X-Identity header for one write, or none without a signer (or for a user it cannot vouch for,
        such as 'unattributed': such a write is refused by the audit service and logged here)."""
        if self.signer is None:
            return {}
        if self.signer.issuer == "auth":
            ident = Identity(AUDIT_WRITER_ID, "system")
        else:
            current = current_identity.get()
            same = current is not None and current.user_id == uid and current.role != "system"
            ident = Identity(uid or "", current.role if same else "user")
        if not self.signer.can_sign(ident):
            return {}
        rid = request_id.get()
        return {HEADER: self.signer.sign(ident, None if rid == "-" else rid)}

    async def record(self, action: str, subject: str | None = None, actor: str | None = None, *,
                     user_id: str | None = None, **detail) -> None:
        """`user_id` is the account the event belongs to: explicit, else the identity in context. A service that
        verifies identities and still has none logs an ERROR and files the event as "unattributed"; a service
        still in single-user mode sends no user (the audit service decides)."""
        if self.http is None:
            return
        ident = current_identity.get()
        uid = user_id or (ident.user_id if ident else None)
        if uid is None and identity_enforced():
            log_fields(log, "audit event without user", logging.ERROR, action=action, subject=subject)
            uid = "unattributed"
        event = AuditEvent(service=self.service, action=action, subject=subject, actor=actor or uid or "local",
                           user_id=uid, detail=detail)
        try:
            response = await self.http.post("/v1/events", json=event.model_dump(), headers=self._signature(uid))
            response.raise_for_status()
        except (httpx.HTTPError, ServiceError) as exc:
            log_fields(log, "audit write failed", logging.ERROR, action=action, subject=subject, error=str(exc))

    async def aclose(self) -> None:
        if self.http is not None:
            await self.http.aclose()
