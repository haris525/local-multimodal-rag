"""HTTP client factory: request-ID propagation, optional identity signing, retries on connection errors only,
and one error type.

Every backend call that fails raises ServiceError: an HTTP error status keeps its code (raised by the
callers' response checks), no answer at all becomes 502 (unreachable) or 504 (timed out). Callers that
fall back on failure (the planner, the duplicate check) therefore also fall back on timeouts."""

import httpx

from rag_common.identity import Signer, signing_hook
from rag_common.logging import request_id


class ServiceError(Exception):
    def __init__(self, status: int, detail: str, cause: str = ""):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.cause = cause  # the transport error behind a 502/504 ("ReadTimeout", "ConnectError", ...)


def _mapped(exc: httpx.TransportError, host: str) -> ServiceError:
    if isinstance(exc, httpx.TimeoutException):
        return ServiceError(504, f"{host} timed out ({type(exc).__name__})", type(exc).__name__)
    return ServiceError(502, f"{host} unreachable ({type(exc).__name__}: {exc})", type(exc).__name__)


class _MappedStream(httpx.AsyncByteStream):
    """A response body whose transport errors (a stream cut mid-answer) surface as ServiceError too."""

    def __init__(self, inner: httpx.AsyncByteStream, host: str):
        self.inner = inner
        self.host = host

    async def __aiter__(self):
        try:
            async for chunk in self.inner:
                yield chunk
        except httpx.TransportError as exc:
            raise _mapped(exc, self.host) from exc

    async def aclose(self) -> None:
        await self.inner.aclose()


class MappedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport):
        self.inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            response = await self.inner.handle_async_request(request)
        except httpx.TransportError as exc:
            raise _mapped(exc, request.url.host) from exc
        return httpx.Response(response.status_code, headers=response.headers,
                              stream=_MappedStream(response.stream, request.url.host), extensions=response.extensions)

    async def aclose(self) -> None:
        await self.inner.aclose()


async def _forward_request_id(request: httpx.Request) -> None:
    rid = request_id.get()
    if rid != "-":
        request.headers["x-request-id"] = rid


def make_client(base_url: str, timeout: float = 30.0, connect_retries: int = 3,
                signer: Signer | None = None) -> httpx.AsyncClient:
    """httpx retries only failed connection attempts, never timeouts or HTTP errors,
    so a busy GPU service is not hit by amplified retries.

    With a signer, every request made while an identity is in context carries a fresh X-Identity signed by
    this service (see identity.signing_hook)."""
    hooks = [_forward_request_id]
    if signer is not None:
        hooks.append(signing_hook(signer))
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout, connect=5.0),
        transport=MappedTransport(httpx.AsyncHTTPTransport(retries=connect_retries)),
        event_hooks={"request": hooks},
    )


async def is_up(client: httpx.AsyncClient, path: str) -> bool:
    try:
        return (await client.get(path, timeout=3.0)).status_code == 200
    except (httpx.HTTPError, ServiceError):
        return False
