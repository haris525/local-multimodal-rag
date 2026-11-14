"""Session and API tokens, cookie headers, and reading credentials from a request.

Tokens are random and shown to their owner only; the database keeps their SHA-256.
"""

import hashlib
import re
import secrets
from typing import Literal

from fastapi import Request

from auth.config import Settings

API_TOKEN_PREFIX = "rag_pat_"
API_TOKEN_RE = re.compile(r"rag_pat_[A-Za-z0-9_-]{32,128}")  # new_token: 43 URL-safe base64 characters
Kind = Literal["browser", "api"]


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token(kind: Kind) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    if kind == "api":
        token = API_TOKEN_PREFIX + token
    return token, hash_token(token)


def _cookie(s: Settings, value: str, max_age: int) -> str:
    parts = [f"{s.cookie_name}={value}", "Path=/", "HttpOnly", "SameSite=Strict", f"Max-Age={max(0, max_age)}"]
    if s.secure_cookies:
        parts.append("Secure")
    return "; ".join(parts)  # never a Domain: the cookie stays on this exact host


def cookie_header(s: Settings, token: str, max_age: int) -> str:
    return _cookie(s, token, max_age)


def clear_cookie_header(s: Settings) -> str:
    return _cookie(s, "", 0)


def read_credentials(request: Request, s: Settings) -> tuple[Kind, str] | None:
    """`Authorization: Bearer rag_pat_...` is an API token; otherwise the session cookie, if any.

    A Bearer header always decides: when it is not a well-formed API token (wrong prefix, characters or
    length), the answer is None (401), never a fall back to the cookie sent with it."""
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer":
        value = value.strip()
        return ("api", value) if API_TOKEN_RE.fullmatch(value) else None
    cookie = request.cookies.get(s.cookie_name)
    if cookie and len(cookie) <= 200:
        return "browser", cookie
    return None
