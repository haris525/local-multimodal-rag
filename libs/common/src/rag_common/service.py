"""FastAPI scaffolding shared by every Python service: request IDs, access logs, health probes, identity."""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from rag_common.identity import Verifier, identity_middleware
from rag_common.logging import configure_logging, log_fields, request_id

ReadinessCheck = Callable[[], Awaitable[dict[str, bool]]]
log = logging.getLogger("access")
_QUIET = ("/healthz", "/readyz")


class AccessLog:
    """ASGI middleware: one log line per request, written when the response body is complete (a streamed
    answer is timed to its last byte) or when an exception escapes (logged as 500 with the error type).
    Sets the request ID for everything the request logs and returns it in `x-request-id`."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1") or uuid.uuid4().hex
        token = request_id.set(rid)
        start = time.perf_counter()
        status, error = 500, None

        async def send_with_id(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", rid.encode("latin-1"))]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except BaseException as exc:  # crashes and client disconnects are logged too, then re-raised
            error = "disconnected" if isinstance(exc, asyncio.CancelledError) else type(exc).__name__
            raise
        finally:
            if scope["path"] not in _QUIET:
                identity = (scope.get("state") or {}).get("identity")  # set by identity_middleware
                fields = {"request_id": rid, "method": scope["method"], "path": scope["path"], "status": status,
                          "ms": round((time.perf_counter() - start) * 1000, 1),
                          "user_id": identity.user_id if identity else None}
                if error:
                    fields["error"] = error
                level = logging.WARNING if error == "disconnected" else logging.ERROR if error or status >= 500 else logging.INFO
                log_fields(log, "request", level, **fields)
            request_id.reset(token)


class NoStoreDefault:
    """ASGI middleware: responses without their own Cache-Control get `no-store`, so no authenticated answer
    is cached by the browser. Responses that set a policy (document assets) keep it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_default(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(name.lower() == b"cache-control" for name, _ in headers):
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_default)


def create_app(
    title: str,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
    readiness: ReadinessCheck | None = None,
    log_level: str = "INFO",
    verifier: Verifier | None = None,
    identity_optional: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> FastAPI:
    """With a verifier, every request except the health probes needs a valid X-Identity (see identity.py).
    Middleware order, outermost first: access log (sets the request id), no-store default, identity."""
    configure_logging(log_level)
    app = FastAPI(title=title, lifespan=lifespan)
    if verifier is not None:
        identity_middleware(app, verifier, identity_optional)
    app.add_middleware(NoStoreDefault)
    app.add_middleware(AccessLog)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        checks = await readiness() if readiness else {}
        ready = all(checks.values())
        return JSONResponse({"ready": ready, "checks": checks}, status_code=200 if ready else 503)

    return app
