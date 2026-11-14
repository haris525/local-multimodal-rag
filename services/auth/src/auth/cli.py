"""auth-admin: operator commands, run in the auth container (`docker compose run --rm auth auth-admin ...`).

They work on the auth database directly, so they recover access even when nobody can sign in.
Exit codes: 0 ok, 1 error, 2 usage.
"""

import argparse
import asyncio
import getpass
import sys
from datetime import timedelta

from rag_common.audit import AuditClient
from rag_common.identity import OPERATOR_ID

from auth.config import USERNAME_RULE, Settings, load_signer, valid_username
from auth.limits import Throttler
from auth.passwords import Passwords, PolicyError, check_policy
from auth.services import DataServices, NotReady, ServiceError
from auth.sessions import new_token
from auth.store import AccountState, AuthStore, Refused, User, account_state, iso, utcnow

STATE_LABEL = {AccountState.ACTIVE: "active", AccountState.MUST_CHANGE: "must change password",
               AccountState.DISABLED: "disabled", AccountState.DELETING: "deleting"}


class Fail(Exception):
    """An error for the operator: printed without a traceback, exit code 1."""


def _read_password(args, username: str, s: Settings) -> str:
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("New password: ")
        if getpass.getpass("Repeat it: ") != password:
            raise Fail("the passwords do not match")
    try:
        check_policy(password, username, s)
    except PolicyError as exc:
        raise Fail(str(exc)) from exc
    return password


def _user(store: AuthStore, username: str) -> User:
    user = store.get_user_by_name(username.strip())
    if user is None:
        raise Fail(f"no user named {username!r}")
    return user


async def _record(s: Settings, action: str, user_id: str, subject: str | None = None, **detail) -> None:
    audit = AuditClient(s.audit_url, "auth", signer=load_signer(s) if s.audit_url else None)
    try:
        await audit.record(action, subject, actor="operator", user_id=user_id, **detail)
    finally:
        await audit.aclose()


async def _services(s: Settings) -> DataServices:
    return DataServices(s, load_signer(s))


# --- commands -------------------------------------------------------------------------------


async def create_admin(args, s: Settings, store: AuthStore) -> int:
    try:
        username = valid_username(args.username)
    except ValueError as exc:
        raise Fail(str(exc)) from exc
    if store.count_admins():
        raise Fail("an admin already exists; use reset-password to recover access")
    if store.get_user_by_name(username):
        raise Fail(f"the username {username!r} is taken")
    password = _read_password(args, username, s)
    user = store.create_user(username, "admin", Passwords(s).hash_sync(password), must_change=False, created_by=None)
    await _record(s, "auth.account_created", user.user_id, role="admin")
    print(f"created admin {username}: {user.user_id}")
    if not args.no_adopt_legacy:
        await _adopt(s, user, dry_run=False, force=False, quiet_if_not_ready=True)
    return 0


async def create_user(args, s: Settings, store: AuthStore) -> int:
    """Operator path for automation (smoke tests, scripted setups). Admins normally use the Users panel.
    Same guard as the admin API: no accounts beyond the first admin until every data service is multi-user."""
    try:
        username = valid_username(args.username)
    except ValueError as exc:
        raise Fail(str(exc)) from exc
    if store.get_user_by_name(username):
        raise Fail(f"the username {username!r} is taken")
    try:
        await (await _services(s)).require_multi_user(OPERATOR_ID)
    except NotReady as exc:
        raise Fail(f"data services are not multi-user yet ({exc.service}); only the first admin can be created") from exc
    password = _read_password(args, username, s)
    user = store.create_user(username, args.role, Passwords(s).hash_sync(password), must_change=args.must_change,
                             created_by=None)
    await _record(s, "auth.account_created", user.user_id, role=args.role, by="operator")
    print(f"created {args.role} {username}: {user.user_id}")
    return 0


async def _adopt(s: Settings, owner: User, dry_run: bool, force: bool, quiet_if_not_ready: bool = False) -> None:
    services = await _services(s)
    try:
        await services.require_multi_user(owner.user_id)
    except NotReady as exc:
        message = (f"data services are not multi-user yet ({exc.service}); data stays shared for now. After the "
                   f"upgrade run: auth-admin migrate-legacy --to {owner.username}")
        if quiet_if_not_ready:
            print(message)
            return
        raise Fail(message) from exc
    try:
        results = await services.migrate_legacy(owner.user_id, owner.user_id, dry_run, force)
    except ServiceError as exc:
        raise Fail(f"migrate-legacy stopped at {exc.service}: {exc.detail}") from exc
    for service, result in results.items():
        print(f"{service}: {result}")


async def migrate_legacy(args, s: Settings, store: AuthStore) -> int:
    await _adopt(s, _user(store, args.to), args.dry_run, args.force)
    return 0


def _apply(store: AuthStore, user: User, action: str, **values) -> User:
    """The account actions of the Users panel, by the same rules (AuthStore.apply). The operator is not an
    account, so no self rule applies to it."""
    try:
        return store.apply(user.user_id, action, **values)
    except Refused as exc:
        raise Fail(f"{user.username}: {exc.detail}") from None


async def reset_password(args, s: Settings, store: AuthStore) -> int:
    user = _user(store, args.username)
    password = _read_password(args, user.username, s)
    revoked = store.count_sessions(user.user_id)
    _apply(store, user, "reset", password_hash=Passwords(s).hash_sync(password), must_change=not args.keep,
           by_admin=not args.keep)
    await _record(s, "auth.password_reset_by_admin", user.user_id)
    print(f"password reset for {user.username}; {revoked} session(s) and token(s) revoked"
          + ("" if args.keep else "; a new password must be chosen at the next sign-in"))
    if Throttler(store, s).locked_until(user.username):
        print(f"note: {user.username} is locked out; run: auth-admin unlock --username {user.username}")
    return 0


async def unlock(args, s: Settings, store: AuthStore) -> int:
    user = _user(store, args.username)
    _apply(store, user, "unlock")
    await _record(s, "admin.user_unlocked", user.user_id, user.user_id, username=user.username)
    print(f"unlocked {user.username}")
    return 0


async def revoke_sessions(args, s: Settings, store: AuthStore) -> int:
    user = _user(store, args.username)
    print(f"revoked {store.delete_sessions(user.user_id)} session(s) and token(s) of {user.username}")
    return 0


async def list_users(args, s: Settings, store: AuthStore) -> int:
    throttle = Throttler(store, s)
    rows = [("USERNAME", "ROLE", "STATE", "LOCKED UNTIL", "LAST LOGIN", "SESSIONS", "USER ID")]
    for u in store.list_users():
        state = STATE_LABEL[account_state(u)]  # the same derivation as the API and the Users panel
        if u.deletion_error:
            state += f" ({u.deletion_error})"
        rows.append((u.username, u.role, state, throttle.locked_until(u.username) or "-", u.last_login_at or "-",
                     str(store.count_sessions(u.user_id)), u.user_id))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip())
    return 0


async def create_token(args, s: Settings, store: AuthStore) -> int:
    user = _user(store, args.username)
    if user.disabled or user.status != "active":
        raise Fail(f"{user.username} is disabled")
    if not 1 <= args.days <= s.token_max_days:
        raise Fail(f"--days must be 1..{s.token_max_days}")
    token, token_hash = new_token("api")
    expires = iso(utcnow() + timedelta(days=args.days))
    t = store.create_session(user.user_id, "api", token_hash, args.name, None, expires, None, "auth-admin",
                             credentials_version=user.credentials_version)
    if t is None:
        raise Fail(f"the account {user.username} changed meanwhile; run the command again")
    await _record(s, "auth.token_created", user.user_id, token_id=t.session_id, name=args.name, expires_at=expires)
    print(f"token for {user.username} ({args.name}), expires {expires}. It is shown once:")
    print(token)
    return 0


async def check(args, s: Settings, store: AuthStore) -> int:
    services = await _services(s)
    admins = [u for u in store.list_users() if u.role == "admin" and u.status == "active"]
    caps = await services.capabilities(admins[0].user_id if admins else OPERATOR_ID)
    for service, answer in caps.items():
        print(f"{service}: {answer}")
    ready = all(isinstance(a, dict) and a.get("multi_user") == 1 for a in caps.values())
    print("multi-user: ready" if ready else "multi-user: not ready (accounts beyond the first admin are refused)")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="auth-admin", description="Manage local-rag accounts from the server.")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create-admin", help="create the first admin (asks for the password twice)")
    c.add_argument("--username", required=True, help=USERNAME_RULE)
    c.add_argument("--password-stdin", action="store_true", help="read the password from one line of stdin")
    c.add_argument("--no-adopt-legacy", action="store_true", help="do not give pre-account data to this admin")
    u = sub.add_parser("create-user", help="create an account from the server (automation; admins use the Users panel)")
    u.add_argument("--username", required=True, help=USERNAME_RULE)
    u.add_argument("--role", choices=("user", "admin"), default="user")
    u.add_argument("--password-stdin", action="store_true", help="read the password from one line of stdin")
    u.add_argument("--must-change", action="store_true", help="require a new password at the first sign-in")
    r = sub.add_parser("reset-password", help="operator recovery: set a new password and revoke sessions")
    r.add_argument("--username", required=True)
    r.add_argument("--password-stdin", action="store_true")
    r.add_argument("--keep", action="store_true", help="do not require a new password at the next sign-in")
    for name, text in (("unlock", "clear a login lockout"), ("revoke-sessions", "sign a user out everywhere")):
        sub.add_parser(name, help=text).add_argument("--username", required=True)
    sub.add_parser("list-users", help="list accounts")
    t = sub.add_parser("create-token", help="create an API token for a user (printed once)")
    t.add_argument("--username", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--days", type=int, default=30)
    m = sub.add_parser("migrate-legacy", help="give data created before accounts existed to one user")
    m.add_argument("--to", required=True, metavar="NAME")
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--force", action="store_true")
    sub.add_parser("check", help="print whether every data service is multi-user")
    return p


COMMANDS = {"create-admin": create_admin, "create-user": create_user, "reset-password": reset_password, "unlock": unlock,
            "revoke-sessions": revoke_sessions, "list-users": list_users, "create-token": create_token,
            "migrate-legacy": migrate_legacy, "check": check}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)  # usage errors exit 2
    try:
        s = Settings()
        store = AuthStore(s.db_path)
        return asyncio.run(COMMANDS[args.command](args, s, store))
    except Fail as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # configuration, keys, database
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
