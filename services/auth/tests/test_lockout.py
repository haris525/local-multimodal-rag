import asyncio
from datetime import timedelta

import httpx
import pytest

from auth.api import create
from auth.config import Settings
from auth.limits import Throttler
from auth.store import AuthStore, UsernameTaken, iso, parse

from conftest import CSRF, PASSWORD, cookie_value


def throttler(tmp_path, clock) -> Throttler:
    s = Settings(db_path=tmp_path / "a.db")
    return Throttler(AuthStore(s.db_path, clock), s, clock)


@pytest.mark.parametrize("route", ["login", "token"])
def test_concurrent_guesses_cannot_outrun_the_lockout(settings, clock, route):
    """40 guesses at once, the right one last: at most 5 passwords are verified, the rest are refused before
    any verification, and the right one does not get through after 39 wrong ones."""
    async def burst() -> list[int]:
        app = create(settings, clock=clock)
        async with app.router.lifespan_context(app):  # the whole app in this event loop
            auth = app.state.auth
            auth.store.create_user("alice", "user", auth.passwords.hash_sync(PASSWORD), False, None)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver", headers=CSRF) as client:
                guesses = [f"wrong guess number {i:03d}" for i in range(39)] + [PASSWORD]
                if route == "login":
                    calls = [client.post("/v1/auth/login", json={"username": "alice", "password": pw}) for pw in guesses]
                else:  # the current-password check of token creation (and password change)
                    signed_in = await client.post("/v1/auth/login", json={"username": "alice", "password": PASSWORD})
                    cookie = {"cookie": f"{settings.cookie_name}={cookie_value(signed_in, settings.cookie_name)}"}
                    client.cookies.clear()
                    calls = [client.post("/v1/auth/tokens", headers=cookie,
                                         json={"name": "t", "current_password": pw}) for pw in guesses]
                return [r.status_code for r in await asyncio.gather(*calls)]

    codes = asyncio.run(burst())
    assert codes.count(429) == 35 and codes.count(401) + codes.count(200) + codes.count(201) == 5
    assert codes[-1] == 429


def test_attempts_are_counted_before_the_password_is_checked(tmp_path, clock):
    t = throttler(tmp_path, clock)
    started = [t.begin("alice", "10.0.0.1") for _ in range(12)]  # none has finished yet
    assert [a.retry_after is None for a in started] == [True] * 5 + [False] * 7
    assert started[4].locks and started[4].locks[0].scope == "user"
    assert t.check("alice", None) == 15 * 60


def test_a_right_password_at_the_threshold_signs_in_and_lifts_its_own_lock(tmp_path, clock):
    t = throttler(tmp_path, clock)
    for _ in range(4):
        t.begin("alice", None)  # four wrong passwords
    right = t.begin("alice", None)  # its advance count reaches the threshold and locks
    assert right.retry_after is None and t.check("alice", None)
    t.succeed(right)
    assert t.check("alice", None) is None and t.store.get_throttle("u:alice") is None


def test_success_keeps_the_lock_history(tmp_path, clock):
    t = throttler(tmp_path, clock)
    for _ in range(5):
        t.begin("alice", None)
    clock.advance(minutes=16)  # the first lock (15 min) is over
    t.succeed(t.begin("alice", None))  # the real user signs in
    assert t.store.get_throttle("u:alice").lock_count == 1
    for _ in range(5):
        t.begin("alice", None)
    assert t.check("alice", None) == 30 * 60  # still doubling: a success does not reset the history


def test_lock_history_expires(tmp_path, clock):
    t = throttler(tmp_path, clock)
    for _ in range(5):
        t.fail("alice", None)
    clock.advance(hours=24, minutes=16)  # the lock ended 24 hours ago
    for _ in range(5):
        t.fail("alice", None)
    assert t.check("alice", None) == 15 * 60 and t.store.get_throttle("u:alice").lock_count == 1


def test_successes_give_back_the_address_count(tmp_path, clock):
    t = throttler(tmp_path, clock)
    for i in range(30):  # many users signing in from one address (a household behind one router)
        attempt = t.begin(f"user{i}", "10.0.0.1")
        assert attempt.retry_after is None
        t.succeed(attempt)
    assert t.store.get_throttle("ip:10.0.0.1") is None
    for i in range(3):
        t.begin(f"other{i}", "10.0.0.1")  # failures still count
    t.succeed(t.begin("user0", "10.0.0.1"))
    assert t.store.get_throttle("ip:10.0.0.1").failures == 3


def test_five_failures_lock_with_retry_after(api, clock):
    api.add_user("alice")
    for _ in range(5):
        assert api.login("alice", "wrong password!").status_code == 401
    r = api.login("alice")  # even the right password is refused while locked
    assert r.status_code == 429 and r.json() == {"detail": "too many attempts, try again later"}
    assert int(r.headers["retry-after"]) == 15 * 60
    assert "auth.locked" in api.audit.actions()
    clock.advance(minutes=15, seconds=1)
    assert api.login("alice").status_code == 200


def test_unknown_username_behaves_the_same(api):
    answers = [api.login("nosuchuser", "wrong password!") for _ in range(6)]
    assert [r.status_code for r in answers] == [401] * 5 + [429]
    assert answers[0].json() == {"detail": "invalid username or password"}
    assert int(answers[-1].headers["retry-after"]) == 15 * 60
    failed = [e for e in api.audit.events if e["action"] == "auth.login_failed"]
    assert failed[0]["user_id"] == "anonymous" and failed[0]["detail"]["username"] == "nosuchuser"
    assert failed[0]["detail"]["reason"] == "unknown_user"


def test_doubling_and_cap(tmp_path, clock):
    s = Settings(db_path=tmp_path / "a.db")
    t = Throttler(AuthStore(s.db_path, clock), s, clock)
    minutes = []
    for _ in range(9):
        locks = []
        for _ in range(5):
            locks += t.fail("alice", None)
        assert t.check("alice", None) is not None
        minutes.append((parse(locks[-1].until) - clock.now).total_seconds() / 60)
        clock.now = parse(locks[-1].until) + timedelta(seconds=1)
    assert minutes == [15, 30, 60, 120, 240, 480, 960, 1440, 1440]  # doubles, capped at 24 h


def test_A1_the_hourly_purge_keeps_the_doubling_after_a_24h_lock(tmp_path, clock):
    """Proof A1 turned around: the purge forgets a lock only once the doubling has forgotten it too."""
    s = Settings(db_path=tmp_path / "a.db")
    store = AuthStore(s.db_path, clock)
    t = Throttler(store, s, clock)
    minutes = []
    for _ in range(9):
        locks = []
        for _ in range(5):
            locks += t.fail("alice", None)
        minutes.append((parse(locks[-1].until) - clock.now).total_seconds() / 60)
        clock.now = parse(locks[-1].until) + timedelta(seconds=1)
        store.purge_throttle(iso(clock.now - timedelta(hours=s.lockout_max_hours)))  # what Auth.purge runs hourly
    assert minutes == [15, 30, 60, 120, 240, 480, 960, 1440, 1440]
    clock.advance(hours=s.lockout_max_hours)  # a day after the last lock ended: history and row both go
    assert store.purge_throttle(iso(clock.now - timedelta(hours=s.lockout_max_hours))) == 1
    assert store.get_throttle("u:alice") is None


def test_window_resets_failures(tmp_path, clock):
    s = Settings(db_path=tmp_path / "a.db")
    t = Throttler(AuthStore(s.db_path, clock), s, clock)
    for _ in range(4):
        t.fail("alice", None)
    clock.advance(minutes=16)  # outside the 15-minute window: counting starts again
    assert t.fail("alice", None) == [] and t.check("alice", None) is None


def test_success_resets_the_counter(api):
    api.add_user("alice")
    for _ in range(4):
        api.login("alice", "wrong password!")
    assert api.login("alice").status_code == 200
    for _ in range(4):
        assert api.login("alice", "wrong password!").status_code == 401
    assert api.login("alice").status_code == 200


def test_per_ip_limit(api):
    api.add_user("alice")
    ip = {"headers": {"X-Real-IP": "10.0.0.9"}}
    for i in range(20):  # 20 different usernames from one address
        assert api.login(f"user{i}", "wrong password!", **ip).status_code == 401
    r = api.login("alice", **ip)
    assert r.status_code == 429 and int(r.headers["retry-after"]) == 15 * 60
    assert api.login("alice", headers={"X-Real-IP": "10.0.0.10"}).status_code == 200  # another address is fine
    assert any(e["detail"].get("scope") == "ip" for e in api.audit.events if e["action"] == "auth.locked")


def test_x_real_ip_is_believed_only_from_the_gateway(settings, clock, services):
    """A caller that is not the gateway (another container on the backend network) is counted under its own
    address, whatever X-Real-IP it sends, so it cannot spread guesses over made-up addresses."""
    from fastapi.testclient import TestClient

    from conftest import Api

    direct = settings.model_copy(update={"trusted_proxies": "gateway"})  # the default: TestClient is not it
    app = create(direct, clock=clock, transport=httpx.MockTransport(services))
    with TestClient(app, base_url="http://testserver") as client:
        api = Api(client, direct)
        api.add_user("alice")
        for i in range(20):
            r = api.login(f"user{i}", "wrong password!", headers={"X-Real-IP": f"10.0.{i}.1"})
            assert r.status_code == 401
        assert api.login("alice", headers={"X-Real-IP": "10.9.9.9"}).status_code == 429
        assert api.auth.store.get_throttle("ip:testclient").locked_until
        assert not any(api.auth.store.get_throttle(f"ip:10.0.{i}.1") for i in range(20))


def test_trusted_proxies_resolve_names_and_follow_a_restarted_gateway():
    from auth.limits import TrustedProxies

    now, answers, asked = [0.0], {"gateway": {"172.20.0.5"}}, []

    def resolve(name):
        asked.append(name)
        from ipaddress import ip_address
        return {ip_address(a) for a in answers.get(name, set())}

    proxies = TrustedProxies("gateway, 192.168.7.0/24", resolve=resolve, clock=lambda: now[0])
    assert proxies.trusts("172.20.0.5") and proxies.trusts("192.168.7.40") and proxies.trusts("::ffff:172.20.0.5")
    assert not proxies.trusts("172.20.0.9") and not proxies.trusts(None) and not proxies.trusts("not an address")
    assert len(asked) == 1  # the miss came within retry_s: no second lookup
    answers["gateway"] = {"172.20.0.9"}  # the gateway restarted with a new address
    now[0] = 6.0
    assert proxies.trusts("172.20.0.9") and not proxies.trusts("172.20.0.5")
    assert proxies.client_address("172.20.0.9", "203.0.113.7") == "203.0.113.7"
    assert proxies.client_address("172.20.0.9", "garbage") == "172.20.0.9"
    assert proxies.client_address("172.20.0.77", "203.0.113.7") == "172.20.0.77"
    assert TrustedProxies("gateway", resolve=lambda name: set()).client_address("10.1.1.1", "1.2.3.4") == "10.1.1.1"


def test_lockout_state_does_not_outlive_or_predate_an_account(tmp_path, clock):
    t = throttler(tmp_path, clock)
    old = t.store.create_user("carol", "user", "x", False, None)
    for _ in range(10):
        t.fail("carol", None)  # two locks: 30 minutes, lock_count 2
    t.store.apply(old.user_id, "delete")
    assert t.store.delete_user(old.user_id)
    assert t.store.get_throttle("u:carol") is None
    t.store.create_user("carol", "user", "x", True, None)  # a new person with the old name
    assert t.locked_until("carol") is None
    for _ in range(5):
        t.fail("dave", None)  # someone locks a name before the account exists
    t.store.create_user("Dave", "user", "x", True, None)
    assert t.locked_until("dave") is None and t.store.get_throttle("u:dave") is None
    t.fail("erin", None)
    with pytest.raises(UsernameTaken):
        t.store.create_user("carol", "user", "x", True, None)  # taken: nothing else changes
    assert t.store.get_throttle("u:erin").failures == 1


def test_admin_create_and_delete_clear_the_names_lockout(api):
    admin = api.session_token(api.add_user("root", role="admin").username)
    for _ in range(5):
        api.login("newbie", "wrong password!")
    r = api.call("POST", "/v1/admin/users", cookie=admin, json={"username": "newbie"})
    assert r.status_code == 201 and r.json()["user"]["locked_until"] is None
    user_id, temporary = r.json()["user"]["user_id"], r.json()["temporary_password"]
    for _ in range(5):
        api.login("newbie", "wrong password!")
    assert api.call("DELETE", f"/v1/admin/users/{user_id}", cookie=admin,
                    json={"confirm_username": "newbie"}).status_code == 204
    assert api.auth.store.get_throttle("u:newbie") is None
    assert api.login("newbie", temporary).status_code == 401  # gone, and not locked: a plain failure


def test_unlock(api):
    api.add_user("alice")
    for _ in range(5):
        api.login("alice", "wrong password!")
    assert api.login("alice").status_code == 429
    api.auth.store.apply(api.auth.store.get_user_by_name("alice").user_id, "unlock")
    assert api.login("alice", PASSWORD).status_code == 200


def test_stale_counters_are_purged(api, clock):
    api.login("ghost", "wrong password!")
    assert api.auth.store.get_throttle("u:ghost")
    clock.advance(hours=25)
    api.auth.purge()
    assert api.auth.store.get_throttle("u:ghost") is None
