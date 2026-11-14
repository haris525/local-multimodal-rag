import io
import sys
from datetime import timedelta

import pytest

from auth import cli
from auth.passwords import Passwords
from auth.store import AuthStore, utcnow

from conftest import load_init_secrets

STRONG = "operator passphrase 2026"


@pytest.fixture
def env(monkeypatch, settings):
    for key, value in {"AUTH_DB_PATH": settings.db_path, "AUTH_SIGNING_KEY_FILE": settings.signing_key_file,
                       "AUTH_PUBLIC_KEYS_FILE": settings.public_keys_file, "AUTH_AUDIT_URL": "",
                       "AUTH_ARGON2_TIME_COST": 1, "AUTH_ARGON2_MEMORY_KIB": 1024, "AUTH_ARGON2_PARALLELISM": 1,
                       "AUTH_INGEST_URL": "", "AUTH_THREADS_URL": "", "AUTH_RAG_URL": ""}.items():  # unreachable
        monkeypatch.setenv(key, str(value))
    return settings


def run(monkeypatch, *argv: str, stdin: str = "") -> int:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    return cli.main(list(argv))


def test_create_admin_refuses_a_weak_password(env, monkeypatch, capsys):
    assert run(monkeypatch, "create-admin", "--username", "root", "--password-stdin", stdin="password123\n") == 1
    assert "Use at least 12 characters." in capsys.readouterr().err
    assert run(monkeypatch, "create-admin", "--username", "root", "--password-stdin",
               stdin="unbelievable\n") == 1
    assert AuthStore(env.db_path).count_users() == 0


def test_create_admin_once(env, monkeypatch, capsys):
    code = run(monkeypatch, "create-admin", "--username", "Root", "--password-stdin", stdin=STRONG + "\n")
    out = capsys.readouterr().out
    assert code == 0 and "created admin root:" in out
    assert "not multi-user yet" in out  # data services unreachable here: legacy data is left for later
    user = AuthStore(env.db_path).get_user_by_name("root")
    assert user.role == "admin" and not user.must_change_password and user.user_id in out
    assert Passwords(env).verify_sync(user.password_hash, STRONG)
    assert run(monkeypatch, "create-admin", "--username", "second", "--password-stdin", stdin=STRONG + "\n") == 1
    assert "use reset-password" in capsys.readouterr().err


def test_create_user_needs_multi_user_services(env, monkeypatch, capsys):
    # data services are unreachable in this fixture, so the guard refuses (fail closed)
    assert run(monkeypatch, "create-user", "--username", "ops-admin", "--role", "admin", "--password-stdin",
               stdin=STRONG + "\n") == 1
    assert "not multi-user yet" in capsys.readouterr().err
    assert AuthStore(env.db_path).get_user_by_name("ops-admin") is None


def test_create_user_when_ready(env, monkeypatch, capsys):
    async def ready(self, actor_id):
        return None
    monkeypatch.setattr(cli.DataServices, "require_multi_user", ready)
    assert run(monkeypatch, "create-user", "--username", "Ops-Admin", "--role", "admin", "--password-stdin",
               stdin=STRONG + "\n") == 0
    assert "created admin ops-admin:" in capsys.readouterr().out
    user = AuthStore(env.db_path).get_user_by_name("ops-admin")
    assert user.role == "admin" and not user.must_change_password
    assert Passwords(env).verify_sync(user.password_hash, STRONG)
    assert run(monkeypatch, "create-user", "--username", "ops-admin", "--password-stdin", stdin=STRONG + "\n") == 1
    assert "taken" in capsys.readouterr().err
    assert run(monkeypatch, "create-user", "--username", "bob", "--must-change", "--password-stdin",
               stdin=STRONG + "\n") == 0
    assert AuthStore(env.db_path).get_user_by_name("bob").must_change_password
    assert run(monkeypatch, "create-user", "--username", "weak", "--password-stdin", stdin="password123\n") == 1


def test_create_admin_asks_twice(env, monkeypatch, capsys):
    answers = iter([STRONG, "something else entirely"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(answers))
    assert run(monkeypatch, "create-admin", "--username", "root", "--no-adopt-legacy") == 1
    assert "do not match" in capsys.readouterr().err


def test_the_fake_clock_agrees_with_the_cli_clock(clock):
    """The commands use the real clock and the app under test the fake one: a lock set through the app must
    still be in force when a command looks at it, at any time of day."""
    assert abs(clock() - utcnow()) < timedelta(minutes=1)


def test_reset_password_and_unlock(env, monkeypatch, capsys, api):
    store = AuthStore(env.db_path)
    api.add_user("alice")
    cookie = api.session_token("alice")
    for _ in range(5):
        api.login("alice", "wrong password!")
    assert run(monkeypatch, "reset-password", "--username", "alice", "--password-stdin",
               stdin="alice gets a fresh one\n") == 1  # contains the username
    assert run(monkeypatch, "reset-password", "--username", "alice", "--password-stdin",
               stdin="a fresh passphrase\n") == 0
    out = capsys.readouterr().out
    assert "1 session(s)" in out and "auth-admin unlock --username alice" in out
    user = store.get_user_by_name("alice")
    assert user.must_change_password and user.reset_by_admin_at
    assert api.call("GET", "/v1/auth/session", cookie=cookie).status_code == 401
    assert api.login("alice", "a fresh passphrase").status_code == 429
    assert run(monkeypatch, "unlock", "--username", "alice") == 0
    assert api.login("alice", "a fresh passphrase").status_code == 200
    assert run(monkeypatch, "reset-password", "--username", "nobody", "--password-stdin", stdin=STRONG) == 1


def test_reset_password_keep(env, monkeypatch, api):
    api.add_user("alice")
    assert run(monkeypatch, "reset-password", "--username", "alice", "--password-stdin", "--keep",
               stdin="a fresh passphrase\n") == 0
    user = AuthStore(env.db_path).get_user_by_name("alice")
    assert not user.must_change_password and user.reset_by_admin_at is None


def test_list_users_and_create_token(env, monkeypatch, capsys, api):
    api.add_user("alice")
    assert run(monkeypatch, "create-token", "--username", "alice", "--name", "eval", "--days", "7") == 0
    token = capsys.readouterr().out.strip().splitlines()[-1]
    assert token.startswith("rag_pat_")
    assert api.call("GET", "/v1/auth/session", bearer=token).json()["user"]["username"] == "alice"
    assert run(monkeypatch, "list-users") == 0
    out = capsys.readouterr().out
    assert "USERNAME" in out and "alice" in out and token not in out
    assert run(monkeypatch, "revoke-sessions", "--username", "alice") == 0
    assert api.call("GET", "/v1/auth/session", bearer=token).status_code == 401
    assert run(monkeypatch, "create-token", "--username", "alice", "--name", "x", "--days", "400") == 1


def test_check_and_migrate_without_multi_user_services(env, monkeypatch, capsys, api):
    api.add_user("root", role="admin")
    assert run(monkeypatch, "check") == 0
    assert "multi-user: not ready" in capsys.readouterr().out
    assert run(monkeypatch, "migrate-legacy", "--to", "root", "--dry-run") == 1
    assert "not multi-user yet" in capsys.readouterr().err


def test_usage_errors_exit_2(env, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, "create-admin")
    assert exc.value.code == 2


def test_init_secrets_refuses_to_overwrite(tmp_path):
    module = load_init_secrets()
    written = module.generate(tmp_path / "s")
    assert sorted(p.name for p in written) == sorted(module.FILES)
    assert 'api_key: "' in (tmp_path / "s" / "qdrant_local.yaml").read_text()
    with pytest.raises(FileExistsError):
        module.generate(tmp_path / "s")
    before = (tmp_path / "s" / "auth_signing.key").read_bytes()
    module.generate(tmp_path / "s", force=True)
    assert (tmp_path / "s" / "auth_signing.key").read_bytes() != before
