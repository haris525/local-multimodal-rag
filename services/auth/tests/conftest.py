import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from auth.api import create
from auth.config import Settings

ROOT = Path(__file__).resolve().parents[3]
CSRF = {"X-Requested-With": "local-rag"}
PASSWORD = "correct horse battery staple"


def load_init_secrets():
    spec = importlib.util.spec_from_file_location("init_secrets", ROOT / "scripts" / "init_secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    """Starts at the real time (whole seconds, as stored), so code that reads the real clock, such as the
    auth-admin commands, agrees with the app under test until a test moves this one."""

    def __init__(self):
        self.now = datetime.now(UTC).replace(microsecond=0)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


class FakeServices:
    """The data services' /internal routes, as auth sees them."""

    def __init__(self):
        self.multi_user = {"ingest": True, "threads": True, "audit": True, "rag": True}
        self.fail = set()  # services whose user purge answers 500
        self.calls: list[tuple[str, str, str, str]] = []  # (service, method, path, x-identity)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        service = {"ingest-api": "ingest"}.get(request.url.host, request.url.host)
        self.calls.append((service, request.method, request.url.path, request.headers.get("x-identity", "")))
        if request.url.path == "/internal/capabilities":
            if not self.multi_user[service]:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"service": service, "multi_user": 1, "schema": 1})
        if request.url.path.startswith("/internal/users/"):
            if service in self.fail:
                return httpx.Response(500, json={"detail": "disk full"})
            return httpx.Response(200, json={"deleted": 0})
        if request.url.path == "/internal/migrate-legacy":
            return httpx.Response(200, json={"adopted": 0, "body": json.loads(request.content)})
        if request.url.path == "/internal/verify":
            return httpx.Response(200, json={"ok": True, "events": 3, "first_bad_seq": None, "head": "h"})
        if request.url.path == "/internal/security-events":
            return httpx.Response(200, json=[{"seq": 2, "time": "t", "action": "auth.login_failed",
                                              "user_id": "anonymous", "actor": "anonymous", "detail": {}}])
        return httpx.Response(404)


class Recorder:
    def __init__(self):
        self.events: list[dict] = []

    async def record(self, action, subject=None, actor=None, *, user_id=None, **detail):
        self.events.append({"action": action, "subject": subject, "actor": actor or user_id, "user_id": user_id,
                            "detail": detail})

    def actions(self) -> list[str]:
        return [e["action"] for e in self.events]

    async def aclose(self):
        pass


@pytest.fixture
def secrets_dir(tmp_path) -> Path:
    folder = tmp_path / "secrets"
    load_init_secrets().generate(folder)
    return folder


@pytest.fixture
def settings(tmp_path, secrets_dir) -> Settings:
    return Settings(
        db_path=tmp_path / "auth.db",
        signing_key_file=secrets_dir / "auth_signing.key",
        public_keys_file=secrets_dir / "identity_public.json",
        argon2_time_cost=1, argon2_memory_kib=1024, argon2_parallelism=1,  # fast in tests
        audit_url=None, ingest_url="http://ingest-api", threads_url="http://threads", rag_url="http://rag",
        trusted_proxies="testclient",  # TestClient's peer: tests play the gateway and set X-Real-IP
    )


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def services() -> FakeServices:
    return FakeServices()


class Api:
    """A TestClient that never keeps cookies between calls: each call says which credential it carries."""

    def __init__(self, client: TestClient, settings: Settings):
        self.client, self.s = client, settings
        self.auth = client.app.state.auth
        self.audit = Recorder()
        self.auth.audit = self.audit
        self.auth.services.urls["audit"] = "http://audit"

    def call(self, method: str, path: str, *, cookie: str | None = None, bearer: str | None = None,
             csrf: bool = True, headers: dict | None = None, **kwargs) -> httpx.Response:
        self.client.cookies.clear()
        h = dict(CSRF if csrf else {})
        if cookie:
            h["Cookie"] = f"{self.s.cookie_name}={cookie}"
        if bearer:
            h["Authorization"] = f"Bearer {bearer}"
        h.update(headers or {})
        response = self.client.request(method, path, headers=h, **kwargs)
        self.client.cookies.clear()
        return response

    def add_user(self, username: str, role: str = "user", password: str = PASSWORD, must_change: bool = False):
        return self.auth.store.create_user(username, role, self.auth.passwords.hash_sync(password), must_change, None)

    def login(self, username: str, password: str = PASSWORD, **kwargs) -> httpx.Response:
        return self.call("POST", "/v1/auth/login", json={"username": username, "password": password}, **kwargs)

    def session_token(self, username: str, password: str = PASSWORD) -> str:
        r = self.login(username, password)
        assert r.status_code == 200, r.text
        return cookie_value(r, self.s.cookie_name)

    def api_token(self, cookie: str, name: str = "eval", password: str = PASSWORD) -> str:
        r = self.call("POST", "/v1/auth/tokens", cookie=cookie,
                      json={"name": name, "days": 30, "current_password": password})
        assert r.status_code == 201, r.text
        return r.json()["token"]


def cookie_value(response: httpx.Response, name: str) -> str:
    header = response.headers["set-cookie"]
    assert header.startswith(f"{name}="), header
    return header.split(";")[0].split("=", 1)[1]


@pytest.fixture
def api(settings, clock, services):
    app = create(settings, clock=clock, transport=httpx.MockTransport(services))
    with TestClient(app, base_url="http://testserver") as client:
        yield Api(client, settings)
