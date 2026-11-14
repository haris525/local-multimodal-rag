"""Requests from the Python sandbox's network are refused before anything else runs.

rag shares the internal `sandbox` network with the sandbox service, whose code is written by the model. Nothing on
that network has a reason to call rag: rag calls the sandbox, never the other way round. SubnetGuard is the
outermost middleware, so a request from RAG_SANDBOX_SUBNET (compose: SANDBOX_SUBNET) gets a 403 before the access
log, the identity check or any route (the health checks included, so /readyz cannot be used to fan probes out to
the other services). The refusals are counted and logged once a minute at most, so a flood cannot rotate rag's
other log lines away.
"""

import ipaddress
import logging
import time

from rag_common.logging import log_fields

log = logging.getLogger("rag.guard")
_BODY = b'{"detail":"forbidden"}'


def address(text: str | None):
    """The address in `text` (an IPv4-mapped IPv6 address as IPv4), or None."""
    try:
        addr = ipaddress.ip_address(text or "")
    except ValueError:
        return None
    return (addr.ipv4_mapped or addr) if isinstance(addr, ipaddress.IPv6Address) else addr


class SubnetGuard:
    """Pure ASGI middleware: 403 for every HTTP request whose client address is in `subnet` (empty: off)."""

    def __init__(self, app, subnet: str = "", log_every_s: float = 60.0, clock=time.monotonic):
        self.app = app
        self.network = ipaddress.ip_network(subnet) if subnet else None
        self.log_every_s, self.clock = log_every_s, clock
        self.refused = 0  # since the last log line
        self.logged_at: float | None = None

    def blocked(self, scope) -> bool:
        if self.network is None or scope.get("type") != "http":
            return False
        client = scope.get("client")
        addr = address(client[0] if client else None)
        return addr is not None and addr.version == self.network.version and addr in self.network

    async def __call__(self, scope, receive, send) -> None:
        if not self.blocked(scope):
            await self.app(scope, receive, send)
            return
        self.refused += 1
        now = self.clock()
        if self.logged_at is None or now - self.logged_at >= self.log_every_s:
            log_fields(log, "requests from the sandbox network refused", logging.WARNING, count=self.refused,
                       subnet=str(self.network))
            self.refused, self.logged_at = 0, now
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(_BODY)).encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": _BODY})
