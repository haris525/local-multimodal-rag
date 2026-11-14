"""Calls from auth to the data services' /internal routes: capabilities, legacy adoption, account purge and the
admin's audit reads. Every call carries an X-Identity with role=system, signed by auth, whose subject is the
admin (or operator) acting.
"""

import logging
from dataclasses import dataclass

import httpx

from rag_common.contracts import MigrateLegacy
from rag_common.identity import HEADER, Identity, Signer
from rag_common.logging import log_fields, request_id

from auth.config import Settings

CAPABILITY_SERVICES = ("ingest", "threads", "audit", "rag")  # all must be multi-user before a second account
MIGRATE_ORDER = ("audit", "threads", "ingest")
PURGE_ORDER = ("ingest", "threads", "audit")

log = logging.getLogger("auth.services")


class ServiceError(Exception):
    def __init__(self, service: str, detail: str, status: int | None = None):
        super().__init__(f"{service}: {detail}")
        self.service, self.detail, self.status = service, detail, status


class NotReady(Exception):
    def __init__(self, service: str):
        super().__init__(service)
        self.service = service


@dataclass
class PurgeResult:
    ok: bool
    failed_step: str | None = None
    detail: str = ""


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except ValueError:
        pass
    return f"HTTP {response.status_code}"


class DataServices:
    def __init__(self, s: Settings, signer: Signer, transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = 60.0):
        self.urls = {"ingest": s.ingest_url, "threads": s.threads_url, "audit": s.audit_url, "rag": s.rag_url}
        self.signer = signer
        self.transport = transport
        self.timeout = timeout

    async def _call(self, service: str, method: str, path: str, actor_id: str, **kwargs) -> httpx.Response:
        url = self.urls.get(service)
        if not url:
            raise ServiceError(service, "not configured")
        headers = {HEADER: self.signer.sign(Identity(actor_id, "system"), request_id.get())}
        if request_id.get() != "-":
            headers["x-request-id"] = request_id.get()
        try:
            async with httpx.AsyncClient(base_url=url, timeout=self.timeout, transport=self.transport) as client:
                return await client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ServiceError(service, f"unreachable: {type(exc).__name__}") from exc

    async def _json(self, service: str, method: str, path: str, actor_id: str, **kwargs):
        response = await self._call(service, method, path, actor_id, **kwargs)
        if response.status_code >= 300:
            raise ServiceError(service, _detail(response), response.status_code)
        return response.json() if response.content else {}

    async def capabilities(self, actor_id: str) -> dict[str, dict | str]:
        """Each service's /internal/capabilities answer, or why it could not be read."""
        out: dict[str, dict | str] = {}
        for service in CAPABILITY_SERVICES:
            try:
                out[service] = await self._json(service, "GET", "/internal/capabilities", actor_id)
            except ServiceError as exc:
                out[service] = exc.detail
        return out

    async def require_multi_user(self, actor_id: str) -> None:
        caps = await self.capabilities(actor_id)
        for service in CAPABILITY_SERVICES:
            body = caps[service]
            if not isinstance(body, dict) or body.get("multi_user") != 1:
                raise NotReady(service)

    async def migrate_legacy(self, actor_id: str, owner_id: str, dry_run: bool, force: bool) -> dict[str, dict]:
        """Adopt unowned (pre-account) data for one user, in the spec's order; stops at the first failure."""
        body = MigrateLegacy(owner_id=owner_id, dry_run=dry_run, force=force).model_dump()
        results = {}
        for service in MIGRATE_ORDER:
            results[service] = await self._json(service, "POST", "/internal/migrate-legacy", actor_id, json=body)
        return results

    async def purge(self, target_id: str, actor_id: str) -> PurgeResult:
        """Every data service deletes the account's data, in PURGE_ORDER; stops at the first failure. Every
        step can be repeated, so a failed purge resumes from the start when called again. The account is
        already in the deleting state (AuthStore.apply), and auth removes its row after the last step."""
        for service in PURGE_ORDER:
            try:
                await self._json(service, "DELETE", f"/internal/users/{target_id}", actor_id)
            except ServiceError as exc:
                log_fields(log, "account purge step failed", logging.ERROR, step=service, error=exc.detail)
                return PurgeResult(False, service, f"{service}: {exc.detail}")
        return PurgeResult(True)

    async def audit_verify(self, actor_id: str) -> dict:
        return await self._json("audit", "GET", "/internal/verify", actor_id)

    async def security_events(self, actor_id: str, limit: int, before: int | None) -> list[dict]:
        params = {"limit": limit} | ({"before": before} if before is not None else {})
        return await self._json("audit", "GET", "/internal/security-events", actor_id, params=params)
