"""auth: sign-in, sessions, API tokens, account management, and /internal/verify for the gateway.

The gateway maps /api/auth/* to /v1/auth/* and /api/admin/* to /v1/admin/*, and calls /internal/verify (nginx
auth_request) before every other /api request: a valid session or API token gets a 204 with a signed
X-Identity, which nginx forwards to the backend in place of the cookie or token.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from rag_common.audit import AuditClient
from rag_common.identity import OPERATOR_ID, USER_ID, Identity, Signer
from rag_common.logging import log_fields
from rag_common.secrets import read_secret
from rag_common.service import create_app

from auth.config import Settings, load_signer, valid_username
from auth.limits import Throttler, TrustedProxies
from auth.passwords import Passwords, PolicyError, check_policy, generate_temp_password, same_password
from auth.services import DataServices, NotReady, PurgeResult, ServiceError
from auth.sessions import clear_cookie_header, cookie_header, hash_token, new_token, read_credentials
from auth.store import (AuthStore, Refused, Session, User, UsernameTaken, account_state, admin_actions, iso, parse,
                        utcnow)

CSRF_HEADER, CSRF_VALUE = "x-requested-with", "local-rag"
SAFE_METHODS = frozenset({"GET", "HEAD"})
UserIdPath = Annotated[str, PathParam(pattern=USER_ID)]
TokenIdPath = Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")]

log = logging.getLogger("auth")


# --- request bodies -------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=1, max_length=1024)


class TokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    days: int | None = Field(default=None, ge=1, le=365)
    current_password: str = Field(min_length=1, max_length=1024)


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    role: Literal["admin", "user"] = "user"


class UserPatch(BaseModel):
    disabled: bool | None = None
    role: Literal["admin", "user"] | None = None


class UserDelete(BaseModel):
    confirm_username: str = Field(min_length=1, max_length=64)


# --- core -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    session: Session
    user: User


def _signin() -> HTTPException:
    return HTTPException(401, "sign in required")  # one answer for missing, unknown, expired and revoked


class Auth:
    """Everything a request handler needs, created at startup and kept in app.state.auth."""

    def __init__(self, s: Settings, store: AuthStore, passwords: Passwords, throttle: Throttler, signer: Signer,
                 audit: AuditClient, services: DataServices, clock: Callable[[], datetime]):
        self.s, self.store, self.passwords, self.throttle = s, store, passwords, throttle
        self.signer, self.audit, self.services, self.clock = signer, audit, services, clock
        self.proxies = TrustedProxies(s.trusted_proxies)
        self.deletions: dict[str, asyncio.Task] = {}  # account purges running in this process, by user id

    def authenticate(self, request: Request, *, csrf: bool, allow_must_change: bool,
                     require_kind: str | None = None, method: str | None = None) -> Principal:
        creds = read_credentials(request, self.s)
        if creds is None:
            raise _signin()
        kind, token = creds
        found = self.store.session_by_hash(hash_token(token))
        if found is None:
            raise _signin()
        session, user = found
        if session.kind != kind:  # a cookie must be a browser session, a bearer token an API token
            raise _signin()
        if session.credentials_version != user.credentials_version:  # password changed, reset, disabled since
            self.store.delete_session(session.session_id)
            raise _signin()
        now = self.clock()
        if now >= parse(session.absolute_expires_at) or (
                session.kind == "browser" and session.idle_expires_at and now >= parse(session.idle_expires_at)):
            self.store.delete_session(session.session_id)
            raise _signin()
        if user.disabled or user.status != "active":
            raise _signin()
        method = (method or request.method).upper()
        if csrf and session.kind == "browser" and method not in SAFE_METHODS \
                and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
            raise HTTPException(403, "missing request header")
        if user.must_change_password and not allow_must_change:
            raise HTTPException(403, "password change required")
        if require_kind and session.kind != require_kind:
            raise HTTPException(403, "browser session required")
        if (now - parse(session.last_seen_at)).total_seconds() >= self.s.touch_interval_s:
            idle = iso(now + timedelta(minutes=self.s.idle_timeout_min)) if session.kind == "browser" else None
            self.store.touch(session.session_id, iso(now), idle)
        return Principal(session, user)

    def user_info(self, user: User) -> dict:
        return {"user_id": user.user_id, "username": user.username, "role": user.role,
                "must_change_password": user.must_change_password}

    def admin_view(self, user: User, viewer: User, active_admins: set[str] | None = None) -> dict:
        """One account as the Users panel shows it: its state and the actions the viewing admin may take now,
        computed by the same rules as AuthStore.apply, so the panel and the CLI never re-derive them."""
        admins = self.store.active_admin_ids() if active_admins is None else active_admins
        locked = self.throttle.locked_until(user.username)
        state = account_state(user)
        running = user.user_id in self.deletions and not self.deletions[user.user_id].done()
        return self.user_info(user) | {
            "disabled": user.disabled, "status": user.status, "locked_until": locked,
            "created_at": user.created_at, "last_login_at": user.last_login_at,
            "active_sessions": self.store.count_sessions(user.user_id), "state": state.value,
            "actions": admin_actions(user, viewer.user_id, len(admins - {user.user_id}), bool(locked)),
            "deletion": {"running": running, "error": user.deletion_error, "attempted_at": user.deletion_attempted_at}
            if user.status == "deleting" else None}

    # --- account deletion: started by an admin, resumed at start-up and every hour -----------------

    def start_deletion(self, user_id: str, resumed: bool = False) -> asyncio.Task:
        """The purge of an account already in the deleting state, as a task that outlives the request; one per
        account at a time (a retry while one runs joins it)."""
        task = self.deletions.get(user_id)
        if task is None or task.done():
            task = asyncio.create_task(self._purge(user_id, resumed))
            self.deletions[user_id] = task
        return task

    async def _purge(self, user_id: str, resumed: bool) -> PurgeResult:
        """Every data service purges the account (each step repeatable), then the row goes. A failure, of any
        kind, is kept on the row (deletion_error) for the Users panel and retried by the resumer; it never
        escapes the task."""
        try:
            user = self.store.get_user(user_id)
            if user is None:
                return PurgeResult(True)
            if user.status != "deleting":
                return PurgeResult(False, None, "the account is not being deleted")
            actor = user.deletion_requested_by or OPERATOR_ID
            result = await self.services.purge(user_id, actor)
            if not result.ok:
                self.store.deletion_failed(user_id, result.detail)  # "<step>: <reason>"
                return result
            self.store.delete_user(user_id)
            await self.audit.record("admin.user_deleted", user_id, user_id=actor, username=user.username,
                                    **({"resumed": True} if resumed else {}),
                                    **({} if user.deletion_requested_by else {"requested_by": "unknown"}))
            return result
        except Exception as exc:
            log.exception("account purge failed", extra={"fields": {"user_id": user_id}})
            detail = f"auth: {type(exc).__name__}: {exc}"
            try:
                self.store.deletion_failed(user_id, detail)
            except Exception:
                log.exception("recording the failed purge failed", extra={"fields": {"user_id": user_id}})
            return PurgeResult(False, "auth", detail)

    async def resume_deletions(self) -> int:
        """Finish every deletion that stopped (a data service was down, auth restarted mid-purge); returns how
        many it tried. Each failed step is logged."""
        tried = 0
        for user in self.store.deleting_users():
            if user.user_id in self.deletions and not self.deletions[user.user_id].done():
                continue
            tried += 1
            try:  # one account's trouble never stops the others
                result = await self.start_deletion(user.user_id, resumed=True)
            except Exception as exc:
                result = PurgeResult(False, "auth", f"{type(exc).__name__}: {exc}")
            if result.ok:
                log_fields(log, "resumed account deletion finished", user_id=user.user_id)
            else:
                log_fields(log, "resumed account deletion stopped", logging.ERROR, user_id=user.user_id,
                           failed_step=result.failed_step, error=result.detail)
        return tried

    def start_browser_session(self, user: User, request: Request) -> str | None:
        """Create a browser session for the account as `user` read it (its credentials version); returns its
        Set-Cookie header, or None when the credentials changed since (no session is created)."""
        now = self.clock()
        absolute = now + timedelta(days=self.s.absolute_timeout_days)
        token, token_hash = new_token("browser")
        created = self.store.create_session(user.user_id, "browser", token_hash, None,
                                            iso(now + timedelta(minutes=self.s.idle_timeout_min)), iso(absolute),
                                            client_ip(request), request.headers.get("user-agent"),
                                            credentials_version=user.credentials_version)
        return cookie_header(self.s, token, int((absolute - now).total_seconds())) if created else None

    def purge(self) -> None:
        sessions = self.store.purge_expired()
        counters = self.store.purge_throttle(iso(self.clock() - timedelta(hours=self.s.lockout_max_hours)))
        if sessions or counters:
            log_fields(log, "purged expired sessions", sessions=sessions, throttle_rows=counters)

    async def check_current_password(self, user: User, password: str, request: Request) -> None:
        """For password change and token creation: wrong passwords count toward the lockout (the attempt is
        counted before the password is verified, as at sign-in)."""
        ip = client_ip(request)
        attempt = self.throttle.begin(user.username, ip)
        if attempt.retry_after:
            raise HTTPException(429, "too many attempts, try again later",
                                headers={"Retry-After": str(attempt.retry_after)})
        if not await self.passwords.verify(user.password_hash, password):
            for lock in attempt.locks:
                await self.audit.record("auth.locked", user_id=user.user_id, scope=lock.scope, until=lock.until, ip=ip)
            raise HTTPException(401, "current password is wrong")
        self.throttle.succeed(attempt)


def client_ip(request: Request) -> str | None:
    """The client's address: X-Real-IP from the gateway, otherwise whoever connected."""
    peer = request.client.host if request.client else None
    return auth_of(request).proxies.client_address(peer, request.headers.get("x-real-ip"))


def auth_of(request: Request) -> Auth:
    return request.app.state.auth


def principal(*, csrf: bool = True, allow_must_change: bool = False, require_kind: str | None = None,
              admin: bool = False):
    def dependency(request: Request) -> Principal:
        p = auth_of(request).authenticate(request, csrf=csrf, allow_must_change=allow_must_change,
                                          require_kind=require_kind)
        if admin and p.user.role != "admin":
            raise HTTPException(403, "admin only")
        return p

    return Depends(dependency)


ADMIN = principal(require_kind="browser", admin=True)  # API tokens cannot manage accounts


# --- sign-in and own account ----------------------------------------------------------------

router = APIRouter()


@router.post("/v1/auth/login")
async def login(request: Request) -> JSONResponse:
    auth = auth_of(request)
    if request.headers.get(CSRF_HEADER) != CSRF_VALUE:  # login CSRF
        raise HTTPException(403, "missing request header")
    try:
        body = LoginRequest.model_validate_json(await request.body())
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_url=False, include_context=False, include_input=False))
    username, ip = body.username.strip().lower(), client_ip(request)
    attempt = auth.throttle.begin(username, ip)  # counted as a failure until the password proves right
    if attempt.retry_after:
        return JSONResponse({"detail": "too many attempts, try again later"}, 429,
                            headers={"Retry-After": str(attempt.retry_after)})
    user = auth.store.get_user_by_name(username)  # its credentials version is the one the session must match
    if user is not None:
        ok = await auth.passwords.verify(user.password_hash, body.password)
    else:
        await auth.passwords.dummy_verify(body.password)
        ok = False
    if not ok or user.disabled or user.status != "active":
        subject = user.user_id if user else "anonymous"
        reason = "unknown_user" if user is None else "bad_password" if not ok else "disabled"
        await auth.audit.record("auth.login_failed", user_id=subject, ip=ip, reason=reason,
                                **({"username": username} if user is None else {}))
        for lock in attempt.locks:
            await auth.audit.record("auth.locked", user_id=subject, scope=lock.scope, until=lock.until, ip=ip)
        raise HTTPException(401, "invalid username or password")
    auth.throttle.succeed(attempt)
    if auth.passwords.needs_rehash(user.password_hash):
        auth.store.set_password_hash(user.user_id, await auth.passwords.hash(body.password), user.credentials_version)
    old = read_credentials(request, auth.s)
    if old and old[0] == "browser":  # never keep a pre-login session id alive (session fixation)
        found = auth.store.session_by_hash(hash_token(old[1]))
        if found and found[0].kind == "browser":
            auth.store.delete_session(found[0].session_id)
    cookie = auth.start_browser_session(user, request)
    if cookie is None:  # the password was changed or reset, or the account disabled, during the check
        await auth.audit.record("auth.login_failed", user_id=user.user_id, ip=ip, reason="credentials_changed")
        raise HTTPException(401, "invalid username or password")
    auth.store.record_login(user.user_id)
    await auth.audit.record("auth.login_succeeded", user_id=user.user_id, ip=ip,
                            user_agent=(request.headers.get("user-agent") or "")[:300])
    return JSONResponse({"user": auth.user_info(auth.store.get_user(user.user_id))}, headers={"set-cookie": cookie})


@router.post("/v1/auth/logout", status_code=204)
async def logout(request: Request) -> Response:
    auth = auth_of(request)
    token = request.cookies.get(auth.s.cookie_name)
    if token:
        found = auth.store.session_by_hash(hash_token(token))
        if found and found[0].kind == "browser":
            auth.store.delete_session(found[0].session_id)
            await auth.audit.record("auth.logout", user_id=found[1].user_id)
    return Response(status_code=204, headers={"set-cookie": clear_cookie_header(auth.s), "clear-site-data": '"cache"'})


@router.get("/v1/auth/session")
def session(request: Request):
    auth = auth_of(request)
    try:
        p = auth.authenticate(request, csrf=False, allow_must_change=True)
    except HTTPException as exc:
        if exc.status_code == 401:
            return JSONResponse({"detail": "sign in required", "setup_required": auth.store.count_users() == 0}, 401)
        raise
    notice = p.user.password_reset_notice_at
    return {"user": auth.user_info(p.user), "password_reset_by_admin_at": p.user.reset_by_admin_at,
            "security_notice": {"kind": "password_reset_by_admin", "at": notice} if notice else None}


@router.post("/v1/auth/security-notice/dismiss", status_code=204)
async def dismiss_security_notice(request: Request, p: Principal = principal(require_kind="browser")) -> Response:
    """The user has read the admin-reset notice. Only this clears it (a password change does not)."""
    auth = auth_of(request)
    seen = auth.store.dismiss_reset_notice(p.user.user_id)
    if seen:
        await auth.audit.record("auth.security_notice_dismissed", user_id=p.user.user_id, reset_at=seen,
                                ip=client_ip(request))
    return Response(status_code=204)


@router.post("/v1/auth/password")
async def change_password(request: Request, body: PasswordChange,
                          p: Principal = principal(allow_must_change=True, require_kind="browser")) -> JSONResponse:
    auth, user = auth_of(request), p.user
    await auth.check_current_password(user, body.current_password, request)
    try:
        check_policy(body.new_password, user.username, auth.s)
    except PolicyError as exc:
        raise HTTPException(400, str(exc)) from exc
    if same_password(body.new_password, body.current_password):
        raise HTTPException(400, "Choose a different password.")
    try:  # every session and API token goes, this one included
        changed = auth.store.apply(user.user_id, "change_password", expected_version=user.credentials_version,
                                   password_hash=await auth.passwords.hash(body.new_password))
    except Refused:  # reset, disabled or being deleted while the current password was checked
        raise _signin() from None
    await auth.audit.record("auth.password_changed", user_id=user.user_id)
    cookie = auth.start_browser_session(changed, request) if changed else None
    if cookie is None:  # changed again (an admin reset or disable) in the meantime: sign in again
        raise _signin()
    return JSONResponse({"user": auth.user_info(auth.store.get_user(user.user_id))}, headers={"set-cookie": cookie})


@router.get("/v1/auth/tokens")
def list_tokens(request: Request, p: Principal = principal(require_kind="browser")) -> list[dict]:
    return [{"token_id": t.session_id, "name": t.name, "created_at": t.created_at, "last_used_at": t.last_seen_at,
             "expires_at": t.absolute_expires_at} for t in auth_of(request).store.list_tokens(p.user.user_id)]


@router.post("/v1/auth/tokens", status_code=201)
async def create_token(request: Request, body: TokenCreate, p: Principal = principal(require_kind="browser")) -> dict:
    auth = auth_of(request)
    days = body.days or auth.s.token_default_days
    if days > auth.s.token_max_days:
        raise HTTPException(400, f"a token can last at most {auth.s.token_max_days} days")
    await auth.check_current_password(p.user, body.current_password, request)
    token, token_hash = new_token("api")
    expires = iso(auth.clock() + timedelta(days=days))
    t = auth.store.create_session(p.user.user_id, "api", token_hash, body.name.strip(), None, expires,
                                  client_ip(request), request.headers.get("user-agent"),
                                  credentials_version=p.user.credentials_version)
    if t is None:  # the password was changed or reset during the check: this session is over too
        raise _signin()
    await auth.audit.record("auth.token_created", user_id=p.user.user_id, token_id=t.session_id, name=t.name,
                            expires_at=expires)
    return {"token_id": t.session_id, "name": t.name, "expires_at": expires, "token": token}  # shown once


@router.delete("/v1/auth/tokens/{token_id}", status_code=204)
async def revoke_token(request: Request, token_id: TokenIdPath,
                       p: Principal = principal(require_kind="browser")) -> Response:
    auth = auth_of(request)
    if not auth.store.delete_session(token_id, p.user.user_id, kind="api"):
        raise HTTPException(404, "token not found")
    await auth.audit.record("auth.token_revoked", user_id=p.user.user_id, token_id=token_id)
    return Response(status_code=204)


# --- account management (admins) ------------------------------------------------------------


def _target(auth: Auth, user_id: str) -> User:
    user = auth.store.get_user(user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    return user


def _apply(auth: Auth, target: User, action: str, admin: User, **values) -> User:
    try:
        return auth.store.apply(target.user_id, action, actor_id=admin.user_id, **values)
    except Refused as exc:
        raise HTTPException(exc.status, exc.detail) from None


async def _both_trails(auth: Auth, admin: User, target: User, user_event: str | None, admin_event: str,
                       **detail) -> None:
    """Account events go into the target's own trail (actor = the admin) and the admin's trail (subject = target)."""
    if user_event:
        await auth.audit.record(user_event, user_id=target.user_id, actor=admin.user_id, **detail)
    await auth.audit.record(admin_event, target.user_id, user_id=admin.user_id, username=target.username, **detail)


@router.get("/v1/admin/users")
def list_users(request: Request, p: Principal = ADMIN) -> list[dict]:
    auth = auth_of(request)
    admins = auth.store.active_admin_ids()
    return [auth.admin_view(u, p.user, admins) for u in auth.store.list_users()]


@router.post("/v1/admin/users", status_code=201)
async def create_user(request: Request, body: UserCreate, p: Principal = ADMIN) -> dict:
    auth = auth_of(request)
    try:
        username = valid_username(body.username)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:  # data stays shared until every service scopes it by owner: no second account before then
        await auth.services.require_multi_user(p.user.user_id)
    except NotReady as exc:
        raise HTTPException(409, f"multi-user not ready: {exc.service}") from exc
    if auth.store.get_user_by_name(username):
        raise HTTPException(409, "username taken")
    temporary = generate_temp_password()
    try:
        user = auth.store.create_user(username, body.role, await auth.passwords.hash(temporary), must_change=True,
                                      created_by=p.user.user_id)
    except UsernameTaken as exc:
        raise HTTPException(409, "username taken") from exc
    await _both_trails(auth, p.user, user, "auth.account_created", "admin.user_created", role=user.role)
    return {"user": auth.admin_view(user, p.user), "temporary_password": temporary}


@router.patch("/v1/admin/users/{user_id}")
async def update_user(request: Request, user_id: UserIdPath, body: UserPatch, p: Principal = ADMIN) -> dict:
    """Disable, enable or change the role; each change is one action (AuthStore.apply)."""
    auth = auth_of(request)
    target = _target(auth, user_id)
    if body.disabled is not None and (body.disabled != target.disabled or target.status == "deleting"):
        _apply(auth, target, "disable" if body.disabled else "enable", p.user)
        await _both_trails(auth, p.user, target, "auth.disabled_by_admin" if body.disabled else "auth.enabled_by_admin",
                           "admin.user_disabled" if body.disabled else "admin.user_enabled")
    if body.role is not None and body.role != target.role:
        _apply(auth, target, "set_role", p.user, role=body.role)
        await _both_trails(auth, p.user, target, "auth.role_changed", "admin.role_changed", role=body.role)
    return auth.admin_view(_target(auth, user_id), p.user)


@router.post("/v1/admin/users/{user_id}/reset-password")
async def reset_password(request: Request, user_id: UserIdPath, p: Principal = ADMIN) -> dict:
    auth = auth_of(request)
    target = _target(auth, user_id)
    temporary = generate_temp_password()
    _apply(auth, target, "reset", p.user, password_hash=await auth.passwords.hash(temporary))
    await _both_trails(auth, p.user, target, "auth.password_reset_by_admin", "admin.password_reset")
    return {"temporary_password": temporary}


@router.post("/v1/admin/users/{user_id}/unlock", status_code=204)
async def unlock_user(request: Request, user_id: UserIdPath, p: Principal = ADMIN) -> Response:
    """Clears the username's lock; a lock on a network address ends on its own."""
    auth = auth_of(request)
    target = _target(auth, user_id)
    _apply(auth, target, "unlock", p.user)
    await _both_trails(auth, p.user, target, None, "admin.user_unlocked")
    return Response(status_code=204)


@router.delete("/v1/admin/users/{user_id}", status_code=204)
async def delete_user(request: Request, user_id: UserIdPath, body: UserDelete, p: Principal = ADMIN) -> Response:
    """The account is locked at once (deleting: disabled, signed out), then every data service purges it.
    204 when that finished within admin_delete_wait_s, 502 with failed_step when a step failed (the account
    stays deleting; a retry or the hourly resumer finishes it), 202 while it is still running (the Users
    panel polls the list). Never longer than the wait, so the gateway never times out a running purge."""
    auth = auth_of(request)
    target = _target(auth, user_id)
    if body.confirm_username.strip().lower() != target.username:
        raise HTTPException(400, "the confirmation does not match the username")
    _apply(auth, target, "delete", p.user)
    task = auth.start_deletion(target.user_id)
    done, _ = await asyncio.wait({task}, timeout=auth.s.admin_delete_wait_s)
    if not done:
        return JSONResponse({"detail": "the deletion is still running", "status": "deleting"}, 202)
    result = task.result()
    if not result.ok:
        return JSONResponse({"detail": result.detail, "status": "deleting", "failed_step": result.failed_step}, 502)
    return Response(status_code=204)


@router.get("/v1/admin/audit/verify")
async def audit_verify(request: Request, p: Principal = ADMIN) -> dict:
    try:
        return await auth_of(request).services.audit_verify(p.user.user_id)
    except ServiceError as exc:
        raise HTTPException(502, f"audit service: {exc.detail}") from exc


@router.get("/v1/admin/audit/security")
async def audit_security(request: Request, limit: int = Query(100, ge=1, le=500), before: int | None = Query(None, ge=1),
                         p: Principal = ADMIN) -> list[dict]:
    auth = auth_of(request)
    try:
        events = await auth.services.security_events(p.user.user_id, limit, before)
    except ServiceError as exc:
        raise HTTPException(502, f"audit service: {exc.detail}") from exc
    names = {u.user_id: u.username for u in auth.store.list_users()}
    return [{k: e.get(k) for k in ("seq", "time", "action", "user_id", "actor", "detail")}
            | {"username": names.get(e.get("user_id"))} for e in events]


# --- gateway --------------------------------------------------------------------------------


@router.get("/internal/verify", include_in_schema=False)
def verify(request: Request) -> Response:
    """nginx auth_request: 204 + X-Identity, or 401/403 (X-Auth-Detail carries the reason for the JSON body)."""
    auth = auth_of(request)
    try:
        # The original method decides the CSRF rule; a request without one is treated as unsafe.
        p = auth.authenticate(request, csrf=True, allow_must_change=False,
                              method=request.headers.get("x-original-method") or "POST")
    except HTTPException as exc:
        return Response(status_code=exc.status_code, headers={"x-auth-detail": str(exc.detail)})
    assertion = auth.signer.sign(Identity(p.user.user_id, p.user.role), request.headers.get("x-request-id"))
    return Response(status_code=204, headers={"x-identity": assertion})


# --- startup --------------------------------------------------------------------------------


async def bootstrap_admin(auth: Auth) -> None:
    """Unattended first admin from AUTH_BOOTSTRAP_ADMIN and a password file, only while no account exists."""
    s = auth.s
    if auth.store.count_users() or not (s.bootstrap_admin and s.bootstrap_password_file):
        return
    username = valid_username(s.bootstrap_admin)
    password = read_secret(s.bootstrap_password_file)
    check_policy(password, username, s)
    user = auth.store.create_user(username, "admin", await auth.passwords.hash(password), must_change=True,
                                  created_by=None)
    await auth.audit.record("auth.account_created", user_id=user.user_id, actor="bootstrap", role="admin")
    log_fields(log, "bootstrap admin created; delete the bootstrap password file now", logging.WARNING,
               username=username, file=str(s.bootstrap_password_file))


async def _maintenance(auth: Auth) -> None:
    """Resume stopped account deletions shortly after start-up (the data services may still be starting),
    then every hour, after purging expired sessions and throttle rows."""
    await asyncio.sleep(auth.s.deletion_resume_delay_s)
    while True:
        try:
            await auth.resume_deletions()
        except Exception:
            log.exception("resuming account deletions failed")
        await asyncio.sleep(3600)
        try:
            await asyncio.to_thread(auth.purge)
        except Exception:
            log.exception("purging expired sessions failed")


def create(settings: Settings | None = None, *, clock: Callable[[], datetime] = utcnow,
           transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Build the app; keys, database and the first-admin bootstrap are loaded at startup (fail fast)."""
    s = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        signer = load_signer(s, clock=lambda: clock().timestamp())
        store = AuthStore(s.db_path, clock)
        audit = AuditClient(s.audit_url, "auth", signer=signer)  # audit takes auth events only signed by auth
        auth = Auth(s, store, Passwords(s), Throttler(store, s, clock), signer, audit,
                    DataServices(s, signer, transport=transport), clock)
        app.state.auth = auth
        await bootstrap_admin(auth)
        auth.purge()
        maintenance = asyncio.create_task(_maintenance(auth))
        try:
            yield
        finally:
            maintenance.cancel()
            for task in auth.deletions.values():  # a purge cut short here is resumed at the next start
                task.cancel()
            await audit.aclose()

    app = create_app("auth", lifespan=lifespan, log_level=s.log_level)
    app.include_router(router)
    return app


app = create()
