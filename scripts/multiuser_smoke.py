"""Live isolation test for accounts and per-user data, through the gateway (docs/design/multi-user.md).

    RAG_ADMIN_PASSWORD='...' uv run --with httpx scripts/multiuser_smoke.py --admin-user NAME
        [--url http://localhost:8000] [--docker] [--compose-dir .] [--skip-lockout] [--cleanup]

It signs in as the admin (the password comes from RAG_ADMIN_PASSWORD, never the command line; an API token cannot
manage accounts by design), creates two temporary accounts A and B (mu-smoke-a-<tag>, mu-smoke-b-<tag>), then:
- A uploads tiny generated files (text, CSV, PNG), chats in a thread and saves memories; B uploads its own;
- B tries every route on A's data: documents, status, chunks, assets, reindex, delete, chat with A's documents
  and thread, deep mode, memory commands, search, threads and their backend-only routes, memories, audit and the
  admin routes; the admin's own views show none of A's data either;
- header spoofing, CSRF and Origin rules, cookie flags, sign-in lockout and the gateway's sign-in rate limit,
  disabling and resetting an account;
- A and B upload, chat and save memories at the same time: no cross-talk in answers, citations, memories or
  thread lists;
- the admin deletes A: A's session and API token get 401, audit verification still passes, the username can be
  reused and the new account starts empty.
With --docker (run from the repository folder, where compose.yaml is) it also checks inside the containers that
llm, models and sql reach nothing on the backend network, that backends refuse calls without a valid identity, and
that the deleted account left no registry rows, index points, files, chats, memories or audit key.

Everything it creates is deleted at the end (--cleanup removes accounts left by an interrupted run). It refuses to
run when other accounts exist or the admin has documents, chats or memories. It prints a pass/fail table.
Exit codes: 0 every check passed, 1 a check failed, 2 refused or usage error.

Needs the full stack with the models loaded and takes several minutes. Sign-ins are paced for the gateway's limit
(10 a minute per address). The lockout checks make 10 failed sign-ins from your address and 20 within 15 minutes
lock the address, so wait 15 minutes between runs or pass --skip-lockout.

Tests of the pure parts: uv run --with httpx --with pytest pytest -q scripts/tests
"""

import argparse
import json
import os
import random
import re
import secrets
import struct
import subprocess
import sys
import time
import zlib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from urllib.parse import urlsplit

import httpx

# The chat stream reader shared with eval and the protocol rag follows (libs/common, standard library only).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libs" / "common" / "src"))
from rag_common.chat_events import parse_sse, read_chat  # noqa: E402

PREFIX = "mu-smoke-"
CSRF = {"X-Requested-With": "local-rag"}
SAFE_METHODS = {"GET", "HEAD"}
LOGIN_GAP_S = 6.5  # nginx: 10 sign-ins a minute per address, burst 5
FOREIGN_ORIGIN = "http://evil.example"
FORGED_IDENTITY = "v1.eyJzdWIiOiJ4In0.AAAA"
PROBE = "What does this document say?"  # B's question on A's data: it carries none of A's facts
WORDS = ["kestrel", "basalt", "juniper", "quartz", "heron", "lantern", "marlin", "cobalt", "saffron", "tundra",
         "falcon", "garnet", "willow", "obsidian", "pelican", "cedar", "ember", "glacier", "harbor", "ivory"]
FRUITS = ["mangosteen", "rambutan", "persimmon", "loquat", "cherimoya", "feijoa", "jabuticaba", "salak"]
_DEFAULT = object()


# --- pure helpers (tested in scripts/tests) ---------------------------------------------------


def origin_of(url: str) -> str:
    """scheme://host[:port] of a URL, the value a browser sends as Origin."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"not an http(s) URL: {url}")
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default = {"http": 80, "https": 443}[parts.scheme]
    return f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port and parts.port != default else "")


def parse_set_cookie(header: str) -> dict:
    """{'name', 'value', 'flags': lowercased attribute names, 'params': {lowercased name: value}}."""
    first, *rest = [p.strip() for p in header.split(";")]
    name, _, value = first.partition("=")
    flags, params = set(), {}
    for part in rest:
        if not part:
            continue
        key, sep, val = part.partition("=")
        flags.add(key.strip().lower())
        if sep:
            params[key.strip().lower()] = val.strip()
    return {"name": name.strip(), "value": value.strip(), "flags": flags, "params": params}


def cookie_problems(header: str, https: bool) -> list[str]:
    """What is wrong with the session cookie of a successful sign-in (spec K1)."""
    c = parse_set_cookie(header)
    problems = [f"missing {flag}" for flag in ("httponly", "path") if flag not in c["flags"]]
    if c["params"].get("samesite", "").lower() != "strict":
        problems.append("SameSite is not Strict")
    if c["params"].get("path") not in (None, "/"):
        problems.append("Path is not /")
    if "domain" in c["flags"]:
        problems.append("Domain is set")
    if https and (c["name"] != "__Host-rag_session" or "secure" not in c["flags"]):
        problems.append("https needs the __Host-rag_session name and Secure")
    if not https and (c["name"] != "rag_session" or "secure" in c["flags"]):
        problems.append("plain http needs the rag_session name without Secure")
    return problems


sse_events = parse_sse  # (event, data) pairs from server-sent event lines (the shared parser)


def png(width: int, height: int, stripes: list[tuple[int, int, int]]) -> bytes:
    """A valid RGB PNG with horizontal stripes, without any imaging library."""
    rows = b""
    for y in range(height):
        r, g, b = stripes[(y * len(stripes)) // height]
        rows += b"\x00" + bytes([r, g, b]) * width

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")


@dataclass(frozen=True)
class Facts:
    """One account's unique, made-up facts: they appear in its files, questions and memories only."""

    project: str
    code: str
    fruit: str
    room: str
    beacon: str
    lucky: str

    @classmethod
    def make(cls, rng: random.Random, taken: set[str] = frozenset()) -> "Facts":
        words = [w for w in WORDS if w not in taken]
        project, beacon = rng.sample(words, 2)
        return cls(project=project, code=f"{rng.choice(WORDS).upper()}-{rng.randint(1000, 9999)}",
                   fruit=rng.choice([f for f in FRUITS if f not in taken]), room=f"{rng.choice('BCDFGH')}{rng.randint(100, 999)}",
                   beacon=beacon, lucky=str(rng.randint(10000, 99999)))

    def needles(self) -> list[str]:
        """Strings that must never show up in another account's views."""
        return [self.code, self.fruit, self.room, self.beacon, self.lucky, f"project {self.project}"]


def sample_files(tag: str, who: str, facts: Facts) -> dict[str, bytes]:
    """Tiny files with one account's facts: a text note, a CSV table and a striped PNG."""
    note = (f"Project {facts.project} field notes ({who}, run {tag}).\n\n"
            f"The vault code for project {facts.project} is {facts.code}.\n"
            f"Only the {facts.project} team uses this vault code, and it changes every quarter.\n")
    table = f"item,quantity\n{facts.project}-bolt,{len(facts.code) * 7}\n{facts.project}-nut,{len(facts.fruit) * 11}\n"
    colours = [(200, 40, 40), (40, 160, 60), (40, 70, 200)] if who == "A" else [(230, 200, 30), (120, 40, 160), (20, 20, 20)]
    return {
        f"{PREFIX}notes-{tag}.txt": note.encode(),
        f"{PREFIX}table-{tag}.csv": table.encode(),
        f"{PREFIX}figure-{tag}.png": png(256, 192, colours),
    }


_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})


def norm(text: str) -> str:
    """Lower case with every dash variant as "-": model answers may typeset a code as KESTREL\u20114721."""
    return text.translate(_DASHES).lower()


def says(text: str, fact: str) -> bool:
    return norm(fact) in norm(text)


def leaks(text: str, needles: Iterable[str]) -> list[str]:
    """The needles that occur in the text (case and dash variants ignored)."""
    return [n for n in needles if n and says(text, n)]


def refusal_reasons(users: list[dict], admin_id: str, library: dict[str, list],
                    allow_other_accounts: bool = False) -> list[str]:
    """Why this install is not a place for the test: other accounts, or data in the admin's own library.
    With allow_other_accounts the test still never reads or changes those accounts; it only creates, reads and
    deletes its own mu-smoke-* accounts. They share this machine's sign-in counter during L1-L3, though."""
    others = [u for u in users if u["user_id"] != admin_id]
    leftovers = sorted(u["username"] for u in others if u["username"].startswith(PREFIX))
    real = sorted(u["username"] for u in others if not u["username"].startswith(PREFIX))
    reasons = []
    if real and not allow_other_accounts:
        reasons.append(f"{len(real)} other account(s) exist ({', '.join(real[:5])}{', ...' if len(real) > 5 else ''}); "
                       "run this test on a test install, or before creating accounts")
    if leftovers:
        reasons.append(f"accounts from an earlier run exist ({', '.join(leftovers)}); remove them with --cleanup")
    for kind, items in library.items():
        if items:
            reasons.append(f"the admin has {len(items)} {kind}; the test needs an empty library")
    return reasons


@dataclass
class Check:
    id: str
    title: str
    ok: bool
    detail: str = ""


class Report:
    def __init__(self, out=print):
        self.checks: list[Check] = []
        self.out = out

    def add(self, id: str, title: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(id, title, bool(ok), "" if ok else detail))
        self.out(f"[{'ok' if ok else 'FAIL'}] {id:5} {title}" + (f"\n        {detail}" if detail and not ok else ""))
        return bool(ok)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def table(self, width: int = 110) -> str:
        id_w = max([len("ID"), *(len(c.id) for c in self.checks)])
        lines = [f"{'ID':{id_w}}  RESULT  CHECK", f"{'-' * id_w}  ------  {'-' * (width - id_w - 10)}"]
        for c in self.checks:
            title = c.title if len(c.title) <= width - id_w - 10 else c.title[: width - id_w - 13] + "..."
            lines.append(f"{c.id:{id_w}}  {'pass' if c.ok else 'FAIL':6}  {title}")
            if c.detail:
                lines.append(f"{'':{id_w}}          {c.detail[: width - id_w - 10]}")
        lines.append(f"\n{len(self.checks) - len(self.failed)} of {len(self.checks)} checks passed")
        return "\n".join(lines)


# --- talking to the gateway -------------------------------------------------------------------


@dataclass
class Account:
    username: str
    password: str = ""
    user_id: str = ""
    cookie: str = ""  # "name=value" of the session cookie
    token: str = ""  # API token
    facts: Facts | None = None
    docs: dict[str, str] = field(default_factory=dict)  # name -> doc_id
    first_login: httpx.Response | None = None  # the sign-in with the temporary password
    before_change: httpx.Response | None = None  # a data request made before the password change


class Gateway:
    """Requests through the gateway with exactly the headers given: no cookie jar, so sessions never mix."""

    def __init__(self, url: str, origin: str, timeout: float = 120, transport: httpx.BaseTransport | None = None):
        jar = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))  # accepts no cookie
        self.http = httpx.Client(base_url=url, timeout=timeout, cookies=jar, follow_redirects=False, transport=transport)
        self.origin = origin
        self.https = url.startswith("https://")
        self._last_login = 0.0

    def close(self) -> None:
        self.http.close()

    def headers(self, method: str, acc: Account | None, origin, csrf: bool, bearer: bool, extra: dict | None) -> dict:
        h: dict[str, str] = {}
        if acc is not None and bearer:
            h["Authorization"] = f"Bearer {acc.token}"
        elif acc is not None:
            h["Cookie"] = acc.cookie
        if origin is _DEFAULT:  # what a browser sends: Origin on unsafe requests; scripts with a token send none
            origin = self.origin if method.upper() not in SAFE_METHODS and not bearer else None
        if origin:
            h["Origin"] = origin
        if csrf and not bearer:
            h.update(CSRF)
        h.update(extra or {})
        return h

    def call(self, method: str, path: str, acc: Account | None = None, *, origin=_DEFAULT, csrf: bool = True,
             bearer: bool = False, headers: dict | None = None, **kwargs) -> httpx.Response:
        return self.http.request(method, path, headers=self.headers(method, acc, origin, csrf, bearer, headers), **kwargs)

    def login(self, username: str, password: str, origin=_DEFAULT, paced: bool = True) -> httpx.Response:
        """POST /api/auth/login, paced under the gateway's rate limit; a 429 without Retry-After is the gateway's
        limit (waited out and retried), a 429 with Retry-After is the auth lockout (returned)."""
        r = None
        for _ in range(8):
            if paced:
                time.sleep(max(0.0, self._last_login + LOGIN_GAP_S - time.monotonic()))
            self._last_login = time.monotonic()
            r = self.call("POST", "/api/auth/login", origin=self.origin if origin is _DEFAULT else origin,
                          json={"username": username, "password": password})
            if not (r.status_code == 429 and "retry-after" not in r.headers and paced):
                return r
            time.sleep(LOGIN_GAP_S)
        return r

    def chat(self, acc: Account, question: str, doc_ids: list[str], thread_id: str | None = None,
             mode: str = "single", timeout: float = 900) -> dict:
        out = {"status": None, "route": None, "answer": "", "sources": [], "citations": [], "memory": [], "error": None, "body": ""}
        body = {"question": question, "doc_ids": doc_ids, "thread_id": thread_id, "mode": mode, "think": False}
        with self.http.stream("POST", "/api/chat", headers=self.headers("POST", acc, _DEFAULT, True, False, None),
                              json=body, timeout=timeout) as r:
            out["status"] = r.status_code
            if r.status_code != 200:
                out["body"] = r.read().decode(errors="replace")
                return out
            reply = read_chat(r.iter_lines())  # error: an error event, a cut stream, or events out of order
        return out | {"sources": reply.sources, "route": reply.route, "answer": reply.answer, "memory": reply.memory,
                      "citations": reply.citations, "error": reply.failure}


def saved_texts(chat: dict) -> list[str]:
    return [i["text"] for e in chat["memory"] if e.get("action") == "saved" for i in e.get("items", [])]


def cookie_pair(response: httpx.Response) -> str:
    header = response.headers.get("set-cookie", "")
    c = parse_set_cookie(header)
    if not c["value"]:
        raise RuntimeError(f"no session cookie in the response ({response.status_code})")
    return f"{c['name']}={c['value']}"


DELETION_WAIT_S = 300
DELETION_POLL_S = 3.0


def settle_deletion(gw, admin, user_id: str, r: httpx.Response, wait_s: float = DELETION_WAIT_S,
                    poll_s: float = DELETION_POLL_S) -> httpx.Response:
    """An account delete that answered 202 goes on in auth: follow the Users list, as the panel does, until the
    account is gone (204) or its deletion stopped (502 with the reason), within wait_s."""
    deadline = time.monotonic() + wait_s
    while r.status_code == 202 and time.monotonic() < deadline:
        time.sleep(poll_s)
        row = next((u for u in gw.call("GET", "/api/admin/users", admin).json() if u["user_id"] == user_id), None)
        if row is None:
            return httpx.Response(204)
        deletion = row.get("deletion") or {}
        if not deletion.get("running"):
            return httpx.Response(502, json={"detail": deletion.get("error") or "deletion stopped", "status": "deleting"})
    return r


def detail_of(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail", body)) if isinstance(body, dict) else str(body)[:200]
    except ValueError:
        return response.text[:200]


# --- docker compose exec checks (--docker) ---------------------------------------------------

NET_PY = """
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=3).close()
    print("open")
except OSError as exc:
    print("closed", type(exc).__name__)
"""

# Inside rag: direct calls to each backend without a valid identity; then B's real rag-signed identity on A's table.
DIRECT_PY = """
import json, sys, time, urllib.error, urllib.request
from pathlib import Path
from rag_common.identity import Identity, Signer
pem = Path("/run/secrets/rag_signing_key").read_bytes()
uid, targets, table_doc = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
tokens = {
    "none": None,
    "garbage": "v1.eyJzdWIiOiJ4In0.AAAA",
    "expired": Signer(pem, "rag", clock=lambda: time.time() - 3600).sign(Identity(uid, "user")),
    "wrong_issuer_key": Signer(pem, "auth").sign(Identity(uid, "user")),
}
def status(method, url, body, token):
    headers = {"content-type": "application/json"}
    if token:
        headers["x-identity"] = token
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError as exc:
        return f"unreachable: {type(exc).__name__}"
out = {f"{name} {kind}": status(m, u, b, t) for name, m, u, b in targets for kind, t in tokens.items()}
out["sql with B's identity on A's table"] = status("POST", "http://sql:8005/v1/query",
    {"doc_id": table_doc, "sql": "select 1"}, Signer(pem, "rag").sign(Identity(uid, "user")))
print(json.dumps(out))
"""

# Inside auth: an identity signed by auth is not enough for the routes only rag may call.
RAG_ONLY_PY = """
import json, sys, urllib.error, urllib.request
from pathlib import Path
from rag_common.identity import Identity, Signer
token = Signer(Path("/run/secrets/auth_signing_key").read_bytes(), "auth").sign(Identity(sys.argv[1], "user"))
req = urllib.request.Request(f"http://threads:8006/v1/threads/{sys.argv[2]}/messages", method="POST",
    data=json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(),
    headers={"content-type": "application/json", "x-identity": token})
try:
    urllib.request.urlopen(req, timeout=10)
    print(json.dumps({"status": 201}))
except urllib.error.HTTPError as exc:
    print(json.dumps({"status": exc.code}))
except OSError as exc:
    print(json.dumps({"status": f"unreachable: {type(exc).__name__}"}))
"""

# Inside ingest-api: what is left of an account in the registry, the index and the files.
INGEST_LEFT_PY = """
import asyncio, json, sys
from qdrant_client import models as qm
from ingest.config import get_settings
from ingest.registry import Registry
from rag_common.index import COLLECTION, qdrant_client
s = get_settings()
async def count(client, uid):
    if not await client.collection_exists(COLLECTION):
        return 0
    flt = qm.Filter(must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=uid))])
    return (await client.count(COLLECTION, count_filter=flt, exact=True)).count
async def main():
    client = qdrant_client(s.qdrant_url, s.qdrant_api_key_file, timeout=30)
    out = {}
    for uid in sys.argv[1:]:
        out[uid] = {"points": await count(client, uid), "documents": len(Registry(s.registry_path).all(uid)),
                    "folders": [k for k in ("files", "assets", "tables", "chunks") if (s.data_dir / k / uid).exists()]}
    await client.close()
    print(json.dumps(out))
asyncio.run(main())
"""

THREADS_LEFT_PY = """
import json, os, sqlite3, sys
db = sqlite3.connect(f"file:{os.getenv('THREADS_DB', '/data/threads.db')}?mode=ro", uri=True)
uid = sys.argv[1]
print(json.dumps({t: db.execute(f"SELECT COUNT(*) FROM {t} WHERE user_id = ?", (uid,)).fetchone()[0] for t in ("threads", "memories")}))
"""

AUDIT_KEY_PY = """
import json, os, sqlite3, sys
db = sqlite3.connect(f"file:{os.getenv('AUDIT_DB', '/data/audit.db')}?mode=ro", uri=True)
row = db.execute("SELECT length(dek) FROM user_keys WHERE user_id = ?", (sys.argv[1],)).fetchone()
print(json.dumps({"key_bytes": None if row is None else row[0],
                  "events": db.execute("SELECT COUNT(*) FROM events WHERE user_id = ?", (sys.argv[1],)).fetchone()[0]}))
"""

PROTECTED = [("auth", 8007), ("threads", 8006), ("audit", 8004), ("ingest-api", 8001), ("qdrant", 6333)]
COMPUTE_ONLY = ["sql", "models", "llm"]
NEEDED = [("rag", "qdrant", 6333), ("rag", "threads", 8006), ("rag", "ingest-api", 8001), ("rag", "audit", 8004),
          ("rag", "sql", 8005), ("rag", "models", 8003), ("rag", "llm", 8080), ("ingest-worker", "qdrant", 6333),
          ("ingest-worker", "audit", 8004), ("ingest-worker", "models", 8003), ("ingest-worker", "llm", 8080)]


def compose_exec(service: str, argv: list[str], compose_dir: str, timeout: float = 120) -> tuple[int, str]:
    """docker compose exec -T <service> <argv>: (exit code, stdout or stderr)."""
    try:
        p = subprocess.run(["docker", "compose", "exec", "-T", service, *argv], cwd=compose_dir, capture_output=True,
                           text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return p.returncode, (p.stdout.strip() or p.stderr.strip())


def reach_argv(service: str, host: str, port: int) -> list[str]:
    """A connection test that works in the service's image: llm has curl but no python."""
    if service == "llm":
        return ["sh", "-c", f"curl -s -m 3 -o /dev/null http://{host}:{port}/ && echo open || echo closed"]
    return ["python", "-c", NET_PY, host, str(port)]


def last_json(text: str):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"no JSON in: {text[-300:]}")


# --- the test ---------------------------------------------------------------------------------


class Smoke:
    def __init__(self, args, report: Report, admin_password: str):
        self.args = args
        self.r = report
        self.gw = Gateway(args.url, args.origin or origin_of(args.url))
        self.admin = Account(args.admin_user.strip().lower(), admin_password)
        self.tag = f"{random.randrange(16**6):06x}"
        self.rng = random.Random(self.tag)
        self.created: dict[str, str] = {}  # username -> user_id, for the cleanup
        self.a = self.b = None

    # -- small helpers --

    def check(self, id: str, title: str, fn: Callable[[], tuple[bool, str] | bool]) -> bool:
        """Run one check; an exception is a failure with its message, and the run goes on."""
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - a live test reports every error as a failed check
            return self.r.add(id, title, False, f"{type(exc).__name__}: {exc}")
        ok, detail = result if isinstance(result, tuple) else (result, "")
        return self.r.add(id, title, ok, detail)

    def status(self, method: str, path: str, acc: Account | None = None, **kw) -> int:
        return self.gw.call(method, path, acc, **kw).status_code

    def json(self, method: str, path: str, acc: Account | None = None, ok: tuple[int, ...] = (200, 201), **kw):
        r = self.gw.call(method, path, acc, **kw)
        if r.status_code not in ok:
            raise RuntimeError(f"{method} {path}: {r.status_code} {detail_of(r)}")
        return r.json() if r.content else None

    def sign_in(self, acc: Account) -> httpx.Response:
        r = self.gw.login(acc.username, acc.password)
        if r.status_code == 200:
            acc.cookie, acc.user_id = cookie_pair(r), r.json()["user"]["user_id"]
        return r

    def change_password(self, acc: Account, new: str) -> None:
        r = self.gw.call("POST", "/api/auth/password", acc, json={"current_password": acc.password, "new_password": new})
        if r.status_code != 200:
            raise RuntimeError(f"password change for {acc.username}: {r.status_code} {detail_of(r)}")
        acc.password, acc.cookie = new, cookie_pair(r)

    def new_password(self) -> str:
        """A fresh password from the system's random source (not the seeded rng: the tag is in the usernames)."""
        return f"{secrets.choice(WORDS)}-{secrets.choice(WORDS)}-{100 + secrets.randbelow(900)}-{secrets.token_hex(4)}"

    def create_account(self, username: str) -> Account:
        body = self.json("POST", "/api/admin/users", self.admin, json={"username": username, "role": "user"})
        acc = Account(username, body["temporary_password"], body["user"]["user_id"])
        self.created[username] = acc.user_id
        return acc

    def delete_account(self, acc_or_name, user_id: str | None = None) -> httpx.Response:
        name = acc_or_name.username if isinstance(acc_or_name, Account) else acc_or_name
        uid = user_id or (acc_or_name.user_id if isinstance(acc_or_name, Account) else self.created[name])
        r = None
        for attempt in range(3):  # a failed purge step leaves the account in "deleting"; calling again resumes it
            r = self.gw.call("DELETE", f"/api/admin/users/{uid}", self.admin, json={"confirm_username": name}, timeout=180)
            r = settle_deletion(self.gw, self.admin, uid, r)
            if r.status_code != 502:
                break
            time.sleep(5 * (attempt + 1))
        if r.status_code in (204, 404):
            self.created.pop(name, None)
        return r

    def upload(self, acc: Account, files: dict[str, bytes]) -> dict[str, str]:
        """One request per file, like the UI; returns {name: doc_id}."""
        out = {}
        for name, data in files.items():
            results = self.json("POST", "/api/documents", acc, files=[("files", (name, data))], timeout=300)
            if results[0].get("error"):
                raise RuntimeError(f"upload of {name}: {results[0]['error']}")
            out[name] = results[0]["document"]["doc_id"]
        return out

    def wait_ready(self, acc: Account, doc_ids: Iterable[str]) -> dict[str, dict]:
        wanted, deadline = set(doc_ids), time.monotonic() + self.args.ingest_budget
        while True:
            docs = {d["doc_id"]: d for d in self.json("GET", "/api/documents", acc) if d["doc_id"] in wanted}
            if len(docs) == len(wanted) and all(d["status"] in ("ready", "failed") for d in docs.values()):
                failed = [f"{d['name']}: {d.get('error')}" for d in docs.values() if d["status"] == "failed"]
                if failed:
                    raise RuntimeError(f"ingestion failed for {failed}")
                return docs
            if time.monotonic() > deadline:
                raise RuntimeError(f"ingestion did not finish within {self.args.ingest_budget} s")
            time.sleep(3)

    def all_audit(self, acc: Account) -> list[dict]:
        events, before = [], None
        while True:
            page = self.json("GET", "/api/audit", acc, params={"limit": 1000, **({"before": before} if before else {})})
            events += page
            if len(page) < 1000:
                return events
            before = page[-1]["user_seq"]

    def start_account(self, letter: str) -> Account:
        """Create, first sign-in with the temporary password (data refused until it is changed), change it."""
        acc = self.create_account(f"{PREFIX}{letter}-{self.tag}")
        r = self.sign_in(acc)
        if r.status_code != 200:
            raise RuntimeError(f"first sign-in of {acc.username}: {r.status_code} {detail_of(r)}")
        acc.first_login = r
        acc.before_change = self.gw.call("GET", "/api/documents", acc)
        self.change_password(acc, self.new_password())
        return acc

    # -- phases --

    def preflight(self) -> list[str]:
        r = self.sign_in(self.admin)
        if r.status_code != 200:
            return [f"admin sign-in failed: {r.status_code} {detail_of(r)}"]
        me = r.json()["user"]
        if me["role"] != "admin":
            return [f"{self.admin.username} is not an admin"]
        if me["must_change_password"]:
            return [f"{self.admin.username} must change their password first (sign in once in the browser)"]
        users = self.json("GET", "/api/admin/users", self.admin)
        library = {"documents": self.json("GET", "/api/documents", self.admin),
                   "chats": self.json("GET", "/api/threads", self.admin),
                   "memories": self.json("GET", "/api/memories", self.admin)}
        return refusal_reasons(users, self.admin.user_id, library, self.args.allow_other_accounts)

    def setup(self) -> None:
        fa = Facts.make(self.rng)
        fb = Facts.make(self.rng, taken={fa.project, fa.beacon, fa.fruit})
        self.a, self.b = self.start_account("a"), self.start_account("b")
        self.a.facts, self.b.facts = fa, fb
        self.check("K1", "session cookie: HttpOnly, SameSite=Strict, Path=/, no Domain, name and Secure per scheme",
                   lambda: (not (p := cookie_problems(self.a.first_login.headers.get("set-cookie", ""), self.gw.https)), "; ".join(p)))
        self.check("E2a", "a temporary password allows sign-in only to change it (data routes 403)",
                   lambda: ((must := self.a.first_login.json()["user"]["must_change_password"])
                            and self.a.before_change.status_code == 403 and detail_of(self.a.before_change) == "password change required",
                            f"must_change={must} {self.a.before_change.status_code} {detail_of(self.a.before_change)}"))
        self.a.docs = self.upload(self.a, sample_files(self.tag, "A", fa))
        self.b.docs = self.upload(self.b, {f"{PREFIX}b-notes-{self.tag}.txt": sample_files(self.tag, "B", fb)[f"{PREFIX}notes-{self.tag}.txt"]})
        self.wait_ready(self.a, self.a.docs.values())
        self.wait_ready(self.b, self.b.docs.values())
        self.a_note, self.a_table, self.a_figure = (self.a.docs[f"{PREFIX}{k}-{self.tag}.{e}"] for k, e in
                                                    (("notes", "txt"), ("table", "csv"), ("figure", "png")))
        self.b_note = next(iter(self.b.docs.values()))
        chunks = self.json("GET", f"/api/documents/{self.a_figure}/chunks", self.a)
        self.a_asset = next((c["asset"] for c in chunks if c.get("asset")), None)
        self.a_thread = self.json("POST", "/api/threads", self.a, json={"title": f"{PREFIX}{self.tag}", "doc_ids": [self.a_note]})["thread_id"]
        self.a_question = f"What is the vault code for project {fa.project}?"
        a_chat = self.gw.chat(self.a, self.a_question, [self.a_note], self.a_thread)
        self.check("ctl1", "A's own chat answers from A's document (control)",
                   lambda: (a_chat["status"] == 200 and says(a_chat["answer"], fa.code) and not a_chat["error"],
                            f"{a_chat['status']} {a_chat['error'] or ''} {a_chat['answer'][:160] or a_chat['body'][:160]}"))
        self.a_mem_text = f"My favourite fruit is the {fa.fruit}."
        self.a_mem = self.json("POST", "/api/memories", self.a, json={"text": self.a_mem_text, "source": "user"})
        remembered = self.gw.chat(self.a, f"Remember that my desk is in room {fa.room}.", [], self.a_thread)
        self.check("ctl2", "A saves a memory through the chat (control)",
                   lambda: (any(says(t, fa.room) for t in saved_texts(remembered)), f"route={remembered['route']} memory={remembered['memory']}"))
        self.b_thread = self.json("POST", "/api/threads", self.b, json={"title": f"{PREFIX}{self.tag}", "doc_ids": [self.b_note]})["thread_id"]
        a_token = self.json("POST", "/api/auth/tokens", self.a, json={"name": f"{PREFIX}{self.tag}", "days": 1, "current_password": self.a.password})
        b_token = self.json("POST", "/api/auth/tokens", self.b, json={"name": f"{PREFIX}{self.tag}", "days": 1, "current_password": self.b.password})
        self.a.token, self.b.token = a_token["token"], b_token["token"]

    def a_ids(self) -> list[str]:
        return [*self.a.docs.values(), self.a_thread, self.a_mem["memory_id"]]

    def a_needles(self) -> list[str]:
        """What B and the admin must never see. A's fruit and room are not included: B saves those same texts
        itself (C7, M4) to show that duplicates are per user."""
        f = self.a.facts
        return [f.code, f"project {f.project}", f.beacon, f.lucky, *self.a_ids(), self.a_question]

    def documents(self) -> None:
        a, b, doc = self.a, self.b, self.a_note
        self.check("D1", "B's document list has none of A's documents",
                   lambda: (not (ids := {d["doc_id"] for d in self.json("GET", "/api/documents", b)} & set(a.docs.values())), str(ids)))
        self.check("D2", "B GET A's document: 404", lambda: (s := self.status("GET", f"/api/documents/{doc}", b)) == 404 or (False, str(s)))
        self.check("D3", "B document status for A's id: []",
                   lambda: (body := self.json("POST", "/api/documents/status", b, json={"doc_ids": [doc]})) == [] or (False, str(body)))
        self.check("D4", "B DELETE A's document: 404, and A still has it",
                   lambda: ((s := self.status("DELETE", f"/api/documents/{doc}", b)) == 404
                            and self.status("GET", f"/api/documents/{doc}", a) == 200, f"B got {s}"))
        self.check("D5", "B reindex A's document: 404", lambda: (s := self.status("POST", f"/api/documents/{doc}/reindex", b)) == 404 or (False, str(s)))
        self.check("D6", "B GET A's chunks: 404", lambda: (s := self.status("GET", f"/api/documents/{doc}/chunks", b)) == 404 or (False, str(s)))

        def assets():
            name = self.a_asset or "0123456789abcdef.png"
            mine = self.status("GET", f"/api/documents/{self.a_figure}/assets/{name}", a) if self.a_asset else None
            theirs = self.status("GET", f"/api/documents/{self.a_figure}/assets/{name}", b)
            return theirs == 404 and mine in (200, None), f"A {mine} B {theirs} asset={self.a_asset}"
        self.check("D7", "B GET A's figure asset: 404 (A: 200)", assets)
        copy_name = f"{PREFIX}copy-{self.tag}.txt"
        same_bytes = sample_files(self.tag, "A", a.facts)[f"{PREFIX}notes-{self.tag}.txt"]

        def same_upload():
            ids = self.upload(b, {copy_name: same_bytes})
            b.docs.update(ids)
            return ids[copy_name] != doc, f"{ids[copy_name]} vs {doc}"
        self.check("D8", "B uploads A's exact bytes: a new document id", same_upload)

        def same_name():
            name = f"{PREFIX}notes-{self.tag}.txt"  # A's file name, B's own content
            ids = self.upload(b, {name: f"Unrelated note by B, run {self.tag}.\n".encode()})
            b.docs.update(ids)
            self.wait_ready(b, ids.values())
            mine = self.json("GET", f"/api/documents/{doc}", a)
            return mine["status"] == "ready", f"A's document is {mine['status']}"
        self.check("D9", "B uploads a file named like A's: A's document stays (no cross-user supersede)", same_name)

    def chats(self) -> None:
        a, b = self.a, self.b
        for id, title, kwargs in (
            ("C1", "B chat with A's document: 404 before streaming", {"doc_ids": [self.a_note]}),
            ("C2", "B chat with B's and A's documents: 404", {"doc_ids": [self.b_note, self.a_note]}),
            ("C3", "B chat in A's thread: 404", {"doc_ids": [self.b_note], "thread_id": self.a_thread}),
            ("C4", "B deep search with A's document: 404", {"doc_ids": [self.a_note], "mode": "deep"}),
        ):
            def run(kwargs=kwargs):
                c = self.gw.chat(b, PROBE, **kwargs)
                return c["status"] == 404, f"{c['status']} {c['body'][:120]}"
            self.check(id, title, run)

        recall = self.gw.chat(b, "What do you remember about me?", [], self.b_thread)
        def answered(c: dict) -> bool:  # a failed chat would make the checks below pass for the wrong reason
            return c["status"] == 200 and not c["error"] and bool(c["answer"].strip())

        self.check("C5", "B asks what is remembered: nothing of A's memories",
                   lambda: (answered(recall) and not (found := leaks(recall["answer"] + json.dumps(recall["memory"]), a.facts.needles())),
                            f"{recall['status']} {recall['error'] or ''} leaked {found if answered(recall) else recall['body'][:120]}"))
        forget = self.gw.chat(b, f"Forget that my favourite fruit is the {a.facts.fruit}.", [], self.b_thread)
        self.check("C6", "B asks to forget A's memory text: A's memory stays",
                   lambda: (answered(forget) and self.a_mem["memory_id"] in {m["memory_id"] for m in self.json("GET", "/api/memories", a)},
                            f"{forget['status']} {forget['error'] or ''} route={forget['route']}"))
        remember = self.gw.chat(b, f"Remember that my desk is in room {a.facts.room}.", [], self.b_thread)
        self.check("C7", "B asks to remember A's memory text: saved for B (not 'already saved')",
                   lambda: (any(says(m["text"], a.facts.room) for m in self.json("GET", "/api/memories", b)),
                            f"route={remember['route']} memory={remember['memory']} answer={remember['answer'][:120]}"))
        self.check("C8", "B search in A's document: 404",
                   lambda: (s := self.status("POST", "/api/search", b, json={"question": PROBE, "doc_ids": [self.a_note]})) == 404 or (False, str(s)))

    def threads_and_memories(self) -> None:
        a, b, t = self.a, self.b, self.a_thread
        self.check("T1", "B's thread list has no thread of A",
                   lambda: t not in {x["thread_id"] for x in self.json("GET", "/api/threads", b)})

        def thread_routes():
            before = self.json("GET", f"/api/threads/{t}", a)
            got = [self.status("GET", f"/api/threads/{t}", b),
                   self.status("PATCH", f"/api/threads/{t}", b, json={"title": "taken over"}),
                   self.status("DELETE", f"/api/threads/{t}", b)]
            after = self.json("GET", f"/api/threads/{t}", a)
            intact = after["thread"]["title"] == before["thread"]["title"] and len(after["messages"]) == len(before["messages"])
            return got == [404, 404, 404] and intact, f"{got} intact={intact}"
        self.check("T2", "B GET, PATCH, DELETE A's thread: 404 each, A's thread intact", thread_routes)

        def backend_only():
            got = [self.status("POST", f"/api/threads/{t}/messages", acc, json={"messages": [{"role": "user", "content": "x"}]})
                   for acc in (b, a)]
            got += [self.status("GET", f"/api/threads/{t}/context", acc) for acc in (b, a)]
            got += [self.status("PUT", f"/api/threads/{t}/summary", acc, json={"summary": "x", "upto": 1}) for acc in (b, a)]
            return all(s == 404 for s in got), str(got)
        self.check("T3", "messages, context and summary routes: 404 at the gateway (B and A)", backend_only)
        self.check("M1", "B's memories have none of A's",
                   lambda: (not (found := leaks(json.dumps(self.json("GET", "/api/memories", b)), [a.facts.fruit, self.a_mem["memory_id"]])), str(found)))
        mem = self.a_mem["memory_id"]
        self.check("M2", "B DELETE and PUT A's memory: 404, A's memory stays",
                   lambda: ((got := [self.status("DELETE", f"/api/memories/{mem}", b),
                                     self.status("PUT", f"/api/memories/{mem}", b, json={"text": "replaced by B", "source": "user"})]) == [404, 404]
                            and mem in {m["memory_id"] for m in self.json("GET", "/api/memories", a)}, str(got)))
        self.check("M3", "B adds a memory naming A's thread: saved with thread_id null",
                   lambda: ((m := self.json("POST", "/api/memories", b, json={"text": f"I like short answers, run {self.tag}.", "thread_id": t}))
                            and m["thread_id"] is None, str(m)))
        self.check("M4", "B adds A's exact memory text: saved for B (per-user duplicates)",
                   lambda: ((m := self.json("POST", "/api/memories", b, json={"text": self.a_mem_text, "source": "user"})) is not None, str(m)))

    def audit(self) -> None:
        a, b = self.a, self.b
        self.check("A1", "B's audit: A's document id matches nothing, and no page holds A's text or ids",
                   lambda: ((hits := self.json("GET", "/api/audit", b, params={"action": "*", "subject": self.a_note})) == []
                            and not (found := leaks(json.dumps(self.all_audit(b)), self.a_needles())),
                            f"subject hits {len(hits)}, leaked {found if hits == [] else '-'}"))

        def counts():
            check, mine, theirs = self.json("GET", "/api/audit/verify", b), len(self.all_audit(b)), len(self.all_audit(a))
            return check["ok"] and check["events"] == mine and mine > 0, f"verify {check} own events {mine} (A has {theirs})"
        self.check("A2", "B's audit verify counts B's events only", counts)

    def admin_views(self) -> None:
        def own_views():
            dumped = json.dumps([self.json("GET", p, self.admin) for p in ("/api/documents", "/api/threads", "/api/memories")]
                                + [self.all_audit(self.admin)])
            return not (found := leaks(dumped, self.a_needles())), str(found)
        self.check("R1", "the admin's documents, threads, memories and audit show none of A's data", own_views)

        def security():
            events = self.json("GET", "/api/admin/audit/security", self.admin, params={"limit": 500})
            bad = sorted({e["action"] for e in events if not re.match(r"^(auth|admin)\.", e["action"])})
            found = leaks(json.dumps(events), [*self.a.facts.needles(), self.a_question])
            return bool(events) and not bad and not found, f"other actions {bad}, leaked {found}"
        self.check("R2", "security events: sign-in and account events only, no content", security)
        uid = self.a.user_id

        def admin_routes(acc, bearer=False):
            calls = [("GET", "/api/admin/users", None), ("POST", "/api/admin/users", {"username": f"{PREFIX}x-{self.tag}"}),
                     ("PATCH", f"/api/admin/users/{uid}", {"disabled": True}), ("POST", f"/api/admin/users/{uid}/reset-password", None),
                     ("POST", f"/api/admin/users/{uid}/unlock", None), ("DELETE", f"/api/admin/users/{uid}", {"confirm_username": self.a.username}),
                     ("GET", "/api/admin/audit/verify", None), ("GET", "/api/admin/audit/security", None)]
            return [self.status(m, p, acc, bearer=bearer, **({"json": body} if body else {})) for m, p, body in calls]
        def b_on_admin_routes():
            got = admin_routes(self.b) + admin_routes(self.b, bearer=True)
            return all(s == 403 for s in got), str(got)
        self.check("R3", "B on every admin route: 403 (session and API token)", b_on_admin_routes)

    def spoofing_and_csrf(self) -> None:
        a, b = self.a, self.b
        spoof = {"X-User-Id": a.user_id, "X-User-Role": "admin", "X-Identity": FORGED_IDENTITY}
        self.check("S1", "anonymous with X-User-Id and a forged X-Identity: 401",
                   lambda: (s := self.status("GET", "/api/documents", headers=spoof)) == 401 or (False, str(s)))

        def b_view():
            replay = {**spoof, "X-Identity": a.cookie.split("=", 1)[1]}
            docs = {d["doc_id"] for d in self.json("GET", "/api/documents", b, headers=replay)}
            return self.b_note in docs and not docs & set(a.docs.values()), f"{len(docs)} documents"
        self.check("S2", "B with A's id and A's session value in identity headers: still B's view", b_view)
        new = {"title": f"{PREFIX}csrf-{self.tag}"}
        self.check("X1", "B cookie POST without Origin: 403",
                   lambda: (s := self.status("POST", "/api/threads", b, origin=None, json=new)) == 403 or (False, str(s)))
        self.check("X2", "B cookie POST with a foreign Origin: 403",
                   lambda: (s := self.status("POST", "/api/threads", b, origin=FOREIGN_ORIGIN, json=new)) == 403 or (False, str(s)))
        self.check("X3", "B cookie POST with our Origin but no X-Requested-With: 403",
                   lambda: ((r := self.gw.call("POST", "/api/threads", b, csrf=False, json=new)).status_code == 403
                            and detail_of(r) == "missing request header", f"{r.status_code} {detail_of(r)}"))
        self.check("X4", "B API token POST /api/search without Origin: 200",
                   lambda: (s := self.status("POST", "/api/search", b, bearer=True, json={"question": "vault code", "doc_ids": [self.b_note]})) == 200
                   or (False, str(s)))
        self.check("X4b", "API token with a foreign Origin: 403",
                   lambda: (s := self.status("POST", "/api/search", b, bearer=True, origin=FOREIGN_ORIGIN,
                                             json={"question": "q", "doc_ids": [self.b_note]})) == 403 or (False, str(s)))
        self.check("X5", "sign-in with a foreign Origin: 403",
                   lambda: (s := self.gw.login(b.username, b.password, origin=FOREIGN_ORIGIN, paced=False).status_code) == 403 or (False, str(s)))
        self.check("X6", "B GET with a foreign Origin: 403",
                   lambda: (s := self.status("GET", "/api/documents", b, origin=FOREIGN_ORIGIN)) == 403 or (False, str(s)))

    def concurrency(self) -> None:
        def work(acc: Account) -> dict:
            gw = Gateway(self.args.url, self.gw.origin)
            try:
                f = acc.facts
                name = f"{PREFIX}conc-{self.tag}.txt"  # the same name for both: supersede stays per owner
                doc = gw_upload(gw, acc, name, f"The beacon word for this run is {f.beacon}.\n".encode())
                deadline = time.monotonic() + self.args.ingest_budget
                while True:
                    d = next((x for x in gw.call("GET", "/api/documents", acc).json() if x["doc_id"] == doc), None)
                    if d and d["status"] in ("ready", "failed"):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("ingestion timed out")
                    time.sleep(2)
                thread = gw.call("POST", "/api/threads", acc, json={"title": name, "doc_ids": [doc]}).json()["thread_id"]
                answer = gw.chat(acc, "What is the beacon word for this run?", [doc], thread)
                memory = gw.chat(acc, f"Remember that my lucky number is {f.lucky}.", [], thread)
                return {"doc": doc, "thread": thread, "answer": answer, "memory": memory,
                        "memories": gw.call("GET", "/api/memories", acc).json(),
                        "threads": [t["thread_id"] for t in gw.call("GET", "/api/threads", acc).json()],
                        "documents": [x["doc_id"] for x in gw.call("GET", "/api/documents", acc).json()]}
            finally:
                gw.close()

        with ThreadPoolExecutor(2) as pool:
            fa, fb = pool.submit(work, self.a), pool.submit(work, self.b)
        try:
            ra, rb = fa.result(), fb.result()
        except Exception as exc:  # noqa: BLE001
            self.r.add("CC", "A and B at the same time", False, f"{type(exc).__name__}: {exc}")
            return
        for me, mine, other, theirs, label in ((self.a, ra, self.b, rb, "A"), (self.b, rb, self.a, ra, "B")):
            ans = mine["answer"]
            self.check(f"CC{label}1", f"{label}'s answer: own beacon word, not the other's",
                       lambda: (says(ans["answer"], me.facts.beacon) and not leaks(ans["answer"], [other.facts.beacon]),
                                f"{ans['status']} {ans['answer'][:140] or ans['body'][:140]}"))
            self.check(f"CC{label}2", f"{label}'s sources and citations: own documents only",
                       lambda: ((src := {s.get("doc_id") for s in ans["sources"]}) <= set(me.docs.values()) | {mine["doc"]} and bool(src),
                                f"sources {src}"))
            self.check(f"CC{label}3", f"{label}'s memories and thread list: nothing of the other",
                       lambda: (not leaks(json.dumps(mine["memories"]), [other.facts.lucky])
                                and theirs["thread"] not in mine["threads"] and theirs["doc"] not in mine["documents"]
                                and any(says(m["text"], me.facts.lucky) for m in mine["memories"]),
                                f"memories {[m['text'] for m in mine['memories']]}"))
            me.docs[f"conc-{label}"] = mine["doc"]

    def lockout(self) -> None:
        a = self.a
        nosuch = f"{PREFIX}nosuch-{self.tag}"

        def attempts(username):
            return [self.gw.login(username, f"wrong-{self.tag}").status_code for _ in range(6)]
        first = attempts(a.username)
        right = self.gw.login(a.username, a.password)
        self.check("L1", "5 wrong passwords lock the account: 429 with Retry-After, also for the right password",
                   lambda: (first == [401] * 5 + [429] and right.status_code == 429 and right.headers.get("retry-after", "").isdigit(),
                            f"{first} then {right.status_code} retry-after={right.headers.get('retry-after')}"))
        unknown = attempts(nosuch)
        self.check("L2", "an unknown username gets the same answers", lambda: (unknown == first, f"{unknown} vs {first}"))
        self.check("L3", "the admin unlocks A, then A signs in",
                   lambda: ((u := self.status("POST", f"/api/admin/users/{a.user_id}/unlock", self.admin)) == 204
                            and (s := self.sign_in(a).status_code) == 200, f"unlock {u}"))

    def rate_limit(self) -> None:
        def burst():  # empty passwords: refused (422) by auth without counting as failed sign-ins
            got = [self.gw.call("POST", "/api/auth/login", origin=self.gw.origin, json={"username": f"{PREFIX}rate", "password": ""})
                   for _ in range(12)]
            limited = [r for r in got if r.status_code == 429 and "retry-after" not in r.headers]
            return bool(limited), str([r.status_code for r in got])
        self.check("L4", "more than 10 sign-ins a minute from one address: 429 from the gateway", burst)
        self.gw._last_login = time.monotonic() + 30  # let the gateway's bucket drain before the next sign-in

    def disable_and_reset(self) -> None:
        a = self.a
        before = Account(a.username, cookie=a.cookie)

        def disable():
            p = self.status("PATCH", f"/api/admin/users/{a.user_id}", self.admin, json={"disabled": True})
            s = self.status("GET", "/api/documents", before)
            t = self.status("GET", "/api/documents", Account(a.username, token=a.token), bearer=True)
            refused = self.gw.login(a.username, a.password).status_code
            e = self.status("PATCH", f"/api/admin/users/{a.user_id}", self.admin, json={"disabled": False})
            return (p == 200 and s == 401 and t == 401 and refused == 401 and e == 200,
                    f"patch {p}, A's session {s}, token {t}, sign-in {refused}, enable {e}")
        self.check("E1", "disabling A ends A's session and API token at once (401) and refuses sign-in", disable)

        def reset():
            r = self.sign_in(a)
            old = Account(a.username, cookie=a.cookie)
            token = self.json("POST", "/api/auth/tokens", a, json={"name": f"{PREFIX}{self.tag}-e2", "days": 1,
                                                                   "current_password": a.password})["token"]
            temp = self.json("POST", f"/api/admin/users/{a.user_id}/reset-password", self.admin)["temporary_password"]
            gone = self.status("GET", "/api/documents", old)
            token_gone = self.status("GET", "/api/documents", Account(a.username, token=token), bearer=True)
            a.password = temp
            login = self.sign_in(a)
            must = login.json()["user"]["must_change_password"] if login.status_code == 200 else None
            session = self.json("GET", "/api/auth/session", a)
            blocked = self.gw.call("GET", "/api/documents", a)
            self.change_password(a, self.new_password())
            back = self.status("GET", "/api/documents", a)
            ok = (r.status_code == 200 and gone == 401 and token_gone == 401 and must is True and session.get("password_reset_by_admin_at")
                  and blocked.status_code == 403 and back == 200)
            return ok, f"old session {gone}, token {token_gone}, must_change {must}, reset_at {session.get('password_reset_by_admin_at')}, data {blocked.status_code}, after change {back}"
        self.check("E2", "resetting A's password: sessions and token 401, must change at sign-in, data 403 until changed", reset)
        tok = self.json("POST", "/api/auth/tokens", a, json={"name": f"{PREFIX}{self.tag}-2", "days": 1, "current_password": a.password})
        a.token = tok["token"]

    def docker(self) -> None:
        d = self.args.compose_dir
        for src in COMPUTE_ONLY:
            for host, port in PROTECTED:
                self.check("N1", f"{src} cannot reach {host}:{port}",
                           lambda src=src, host=host, port=port: ((out := compose_exec(src, reach_argv(src, host, port), d))[1].startswith("closed"),
                                                                  f"exit {out[0]}: {out[1][-200:]}"))
        for src, host, port in NEEDED:
            self.check("N2", f"{src} reaches {host}:{port}",
                       lambda src=src, host=host, port=port: ((out := compose_exec(src, reach_argv(src, host, port), d))[1] == "open",
                                                              f"exit {out[0]}: {out[1][-200:]}"))
        targets = [["ingest-api", "GET", "http://ingest-api:8001/v1/documents", None],
                   ["threads", "GET", "http://threads:8006/v1/threads", None],
                   ["audit", "GET", "http://audit:8004/v1/events", None],
                   ["sql", "POST", "http://sql:8005/v1/query", {"doc_id": self.a_table, "sql": "select 1"}],
                   ["rag", "POST", "http://rag:8002/v1/search", {"question": "q", "doc_ids": [self.a_note]}]]

        def direct():
            code, out = compose_exec("rag", ["python", "-c", DIRECT_PY, self.b.user_id, json.dumps(targets), self.a_table], d)
            got = last_json(out)
            table = got.pop("sql with B's identity on A's table")
            bad = {k: v for k, v in got.items() if v != 401}
            return not bad and table == 404 and len(got) == 20, f"not 401: {bad}; sql with B on A's table {table}"
        self.check("S3S4", "backends refuse missing, garbage, expired and wrongly signed identities (401); B's real identity gets 404 on A's table", direct)
        self.check("S4b", "an identity signed by auth cannot append thread messages (403, rag only)",
                   lambda: ((got := last_json(compose_exec("auth", ["python", "-c", RAG_ONLY_PY, self.b.user_id, self.b_thread], d)[1]))["status"] == 403, str(got)))

    def deletion(self) -> None:
        a, b = self.a, self.b
        old = Account(a.username, cookie=a.cookie, token=a.token)
        r = self.delete_account(a)
        self.check("P1", "the admin deletes A: 204", lambda: (r.status_code == 204, f"{r.status_code} {detail_of(r)}"))
        self.check("P1a", "A's session and API token: 401; A cannot sign in",
                   lambda: ((got := [self.status("GET", "/api/documents", old), self.status("GET", "/api/documents", old, bearer=True),
                                     self.gw.login(a.username, a.password).status_code]) == [401, 401, 401], str(got)))
        self.check("P1b", "A is gone from the account list",
                   lambda: a.user_id not in {u["user_id"] for u in self.json("GET", "/api/admin/users", self.admin)})
        self.check("P1c", "audit verification passes after the deletion (whole log and B's)",
                   lambda: ((w := self.json("GET", "/api/admin/audit/verify", self.admin))["ok"]
                            and (v := self.json("GET", "/api/audit/verify", b))["ok"], f"whole {w} B {v}"))

        def shredded():
            events = [e for e in self.json("GET", "/api/admin/audit/security", self.admin, params={"limit": 500}) if e["user_id"] == a.user_id]
            return bool(events) and all(e["detail"] == {"_shredded": True} for e in events), f"{len(events)} events, details {[e['detail'] for e in events[:3]]}"
        self.check("P1d", "A's sign-in events remain in the chain with their details erased", shredded)
        self.check("P1e", "B's documents, threads and memories are untouched",
                   lambda: (self.b_note in {x["doc_id"] for x in self.json("GET", "/api/documents", b)}
                            and self.b_thread in {t["thread_id"] for t in self.json("GET", "/api/threads", b)}
                            and bool(self.json("GET", "/api/memories", b))))

        def reuse():
            again = self.create_account(a.username)
            if self.sign_in(again).status_code != 200:
                return False, "sign-in with the temporary password failed"
            self.change_password(again, self.new_password())
            views = [self.json("GET", p, again) for p in ("/api/documents", "/api/threads", "/api/memories")]
            ok = again.user_id != a.user_id and views == [[], [], []]
            gone = self.delete_account(again).status_code
            return ok and gone == 204, f"new id differs {again.user_id != a.user_id}, views {[len(v) for v in views]}, delete {gone}"
        self.check("P1f", "the username can be reused and the new account starts empty", reuse)
        if self.args.docker:
            d = self.args.compose_dir

            def ingest_left():
                got = last_json(compose_exec("ingest-api", ["python", "-c", INGEST_LEFT_PY, a.user_id, b.user_id], d)[1])
                return (got[a.user_id] == {"points": 0, "documents": 0, "folders": []} and got[b.user_id]["points"] > 0, str(got))
            self.check("P1g", "no registry rows, index points or folders left for A (B keeps its points)", ingest_left)
            self.check("P1h", "no threads or memories left for A",
                       lambda: ((got := last_json(compose_exec("threads", ["python", "-c", THREADS_LEFT_PY, a.user_id], d)[1]))
                                == {"threads": 0, "memories": 0}, str(got)))
            self.check("P1i", "A's audit key is shredded (rows stay for the chain)",
                       lambda: ((got := last_json(compose_exec("audit", ["python", "-c", AUDIT_KEY_PY, a.user_id], d)[1]))["key_bytes"] == 0, str(got)))

    def cleanup(self) -> list[str]:
        problems = []
        for name, uid in list(self.created.items()):
            try:
                r = self.delete_account(name, uid)
                if r.status_code not in (204, 404):
                    problems.append(f"{name}: {r.status_code} {detail_of(r)}")
            except httpx.HTTPError as exc:
                problems.append(f"{name}: {exc}")
        return problems

    def run(self) -> int:
        try:
            reasons = self.preflight()
        except httpx.HTTPError as exc:
            print(f"error: cannot reach {self.args.url}: {exc}", file=sys.stderr)
            return 2
        if reasons:
            print("refusing to run:\n  - " + "\n  - ".join(reasons), file=sys.stderr)
            return 2
        print(f"run {self.tag}: accounts {PREFIX}a-{self.tag} and {PREFIX}b-{self.tag}\n")
        try:
            try:
                self.setup()
            except Exception as exc:  # noqa: BLE001 - without A and B nothing else can run
                self.r.add("SETUP", "create A and B, upload, chat, save memories", False, f"{type(exc).__name__}: {exc}")
                return 1
            for phase in (self.documents, self.chats, self.threads_and_memories, self.audit, self.admin_views,
                          self.spoofing_and_csrf, self.concurrency):
                self.run_phase(phase)
            if self.args.docker:
                self.run_phase(self.docker)
            if not self.args.skip_lockout:
                self.run_phase(self.lockout)
            self.run_phase(self.rate_limit)
            self.run_phase(self.disable_and_reset)
            self.run_phase(self.deletion)
        finally:
            problems = self.cleanup()
            if problems:
                self.r.add("CLEAN", "delete every account the run created", False, "; ".join(problems))
            self.gw.close()
        return 1 if self.r.failed else 0

    def run_phase(self, phase: Callable[[], None]) -> None:
        try:
            phase()
        except Exception as exc:  # noqa: BLE001 - one broken phase must not hide the others
            self.r.add(phase.__name__.upper()[:5], f"phase {phase.__name__}", False, f"{type(exc).__name__}: {exc}")


def gw_upload(gw: Gateway, acc: Account, name: str, data: bytes) -> str:
    r = gw.call("POST", "/api/documents", acc, files=[("files", (name, data))], timeout=300)
    r.raise_for_status()
    result = r.json()[0]
    if result.get("error"):
        raise RuntimeError(f"upload of {name}: {result['error']}")
    return result["document"]["doc_id"]


def cleanup_only(args, admin_password: str) -> int:
    gw = Gateway(args.url, args.origin or origin_of(args.url))
    admin = Account(args.admin_user.strip().lower(), admin_password)
    try:
        r = gw.login(admin.username, admin.password)
        if r.status_code != 200:
            print(f"error: admin sign-in failed: {r.status_code} {detail_of(r)}", file=sys.stderr)
            return 2
        admin.cookie = cookie_pair(r)
        users = gw.call("GET", "/api/admin/users", admin).json()
        left, failed = [u for u in users if u["username"].startswith(PREFIX)], 0
        for u in left:
            d = gw.call("DELETE", f"/api/admin/users/{u['user_id']}", admin, json={"confirm_username": u["username"]}, timeout=180)
            d = settle_deletion(gw, admin, u["user_id"], d)
            print(f"{u['username']}: {'deleted' if d.status_code == 204 else f'{d.status_code} {detail_of(d)}'}")
            failed += d.status_code != 204
        print(f"{len(left) - failed} of {len(left)} leftover accounts deleted")
        return 1 if failed else 0
    finally:
        gw.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="multiuser_smoke.py",
        description="Live isolation test of accounts and per-user data through the gateway. The admin's password "
                    "is read from RAG_ADMIN_PASSWORD. Exit codes: 0 passed, 1 a check failed, 2 refused or usage error.",
        epilog="Example: RAG_ADMIN_PASSWORD='...' uv run --with httpx scripts/multiuser_smoke.py --admin-user admin --docker")
    p.add_argument("--admin-user", required=True, help="an admin account (signs in with a password: API tokens cannot manage accounts)")
    p.add_argument("--url", default="http://localhost:8000", help="the gateway (default: http://localhost:8000)")
    p.add_argument("--origin", help="Origin header to send (default: derived from --url; set it to PUBLIC_ORIGIN behind a proxy)")
    p.add_argument("--docker", action="store_true", help="also run the in-container checks with docker compose exec")
    p.add_argument("--compose-dir", default=".", help="folder with compose.yaml for --docker (default: current folder)")
    p.add_argument("--skip-lockout", action="store_true", help="skip the sign-in lockout checks (L1-L3), for a second run within 15 minutes")
    p.add_argument("--allow-other-accounts", action="store_true",
                   help="run although real accounts exist; the test never reads or changes them, but L1-L3 add 12 failed sign-ins to this machine's address (limit 20 per 15 minutes; add --skip-lockout to avoid)")
    p.add_argument("--ingest-budget", type=int, default=900, help="seconds to wait for uploads to be indexed (default: 900)")
    p.add_argument("--cleanup", action="store_true", help="only delete accounts left by an interrupted run (mu-smoke-*) and exit")
    args = p.parse_args(argv)
    try:
        origin_of(args.url)
        if args.origin:
            origin_of(args.origin)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    password = os.environ.get("RAG_ADMIN_PASSWORD", "")
    if not password:
        print("error: set RAG_ADMIN_PASSWORD to the admin's password (it is never taken from the command line)", file=sys.stderr)
        return 2
    if args.cleanup:
        return cleanup_only(args, password)
    report = Report()
    code = Smoke(args, report, password).run()
    if report.checks:
        print("\n" + report.table())
    return code


if __name__ == "__main__":
    sys.exit(main())
