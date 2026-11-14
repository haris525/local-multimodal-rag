"""Login throttling: per-username lockout with doubling, and a per-IP limit.

Usernames are counted whether or not the account exists, so the answers never reveal which accounts exist.

Every password check is an Attempt: begin() checks the locks and counts the attempt as a failure in one
transaction, before the password is verified, so concurrent guesses cannot all pass the check before any
failure is recorded (at most `lockout_threshold` verifications per lock). A right password then takes its
count back with succeed(); a wrong one, a crash or a dropped connection leaves it counted.
"""

import ipaddress
import math
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from auth.config import Settings
from auth.store import AuthStore, Throttle, ThrottleRows, iso, parse, user_throttle_key, utcnow

Address = ipaddress.IPv4Address | ipaddress.IPv6Address


def _address(text: str | None) -> Address | None:
    try:
        addr = ipaddress.ip_address((text or "").strip())
    except ValueError:
        return None
    return addr.ipv4_mapped or addr if addr.version == 6 else addr


def _resolve(name: str) -> set[Address]:
    try:
        infos = socket.getaddrinfo(name, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):  # not resolvable (yet): trusts nobody under that name
        return set()
    return {a for a in (_address(info[4][0]) for info in infos) if a}


class TrustedProxies:
    """Whose X-Real-IP auth believes (AUTH_TRUSTED_PROXIES, comma-separated: host names, addresses or networks;
    default the gateway). Any other caller, such as another container on the backend network, is counted
    under its own address, so it cannot spread guesses over made-up addresses to dodge the per-address limit.
    Host names are resolved again every `refresh_s` seconds, and on a miss at most every `retry_s` seconds,
    so a restarted gateway with a new address is followed. A name that equals the peer as given (a test
    client reports a name, not an address) also counts."""

    def __init__(self, entries: str, refresh_s: float = 60.0, retry_s: float = 5.0,
                 resolve: Callable[[str], set[Address]] = _resolve, clock: Callable[[], float] = time.monotonic):
        self.networks, self.names = [], []
        for entry in filter(None, (e.strip() for e in entries.split(","))):
            try:
                self.networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                self.names.append(entry)
        self.refresh_s, self.retry_s, self.resolve, self.clock = refresh_s, retry_s, resolve, clock
        self._resolved: set[Address] = set()
        self._at = float("-inf")

    def trusts(self, peer: str | None) -> bool:
        if not peer:
            return False
        if peer in self.names:
            return True
        addr = _address(peer)
        if addr is None:
            return False
        if any(addr in net for net in self.networks):
            return True
        if not self.names:
            return False
        age = self.clock() - self._at
        if age >= self.refresh_s or (addr not in self._resolved and age >= self.retry_s):
            self._resolved = set().union(*(self.resolve(name) for name in self.names))
            self._at = self.clock()
        return addr in self._resolved

    def client_address(self, peer: str | None, real_ip: str | None) -> str | None:
        """The address to count failures under: X-Real-IP when the peer is a trusted proxy, else the peer."""
        if real_ip and self.trusts(peer):
            forwarded = _address(real_ip)
            if forwarded:
                return str(forwarded)
        return peer


@dataclass(frozen=True)
class Lock:
    scope: str  # "user" | "ip"
    until: str


def user_key(username: str) -> str:
    return user_throttle_key(username)


def ip_key(ip: str) -> str:
    return f"ip:{ip}"


def _scope(key: str) -> str:
    return "ip" if key.startswith("ip:") else "user"


def _active(t: Throttle | None, now: datetime) -> bool:
    return bool(t and t.locked_until and parse(t.locked_until) > now)


def _sets_lock(before: Throttle | None, after: Throttle | None) -> bool:
    return bool(after and after.locked_until and after.locked_until != (before.locked_until if before else None))


@dataclass(frozen=True)
class Attempt:
    """One password check. Refused (retry_after set, nothing counted) or counted as a failure in advance."""

    retry_after: int | None = None
    before: ThrottleRows = field(default_factory=dict)
    after: ThrottleRows = field(default_factory=dict)

    @property
    def locks(self) -> list[Lock]:
        """The locks this attempt set (for the audit log once it has failed)."""
        return [Lock(_scope(k), a.locked_until) for k, a in self.after.items() if _sets_lock(self.before.get(k), a)]


class Throttler:
    """The spec's `Throttle` (renamed: `Throttle` is the stored row)."""

    def __init__(self, store: AuthStore, s: Settings, clock: Callable[[], datetime] = utcnow):
        self.store = store
        self.s = s
        self.clock = clock

    def _keys(self, username: str, ip: str | None) -> list[str]:
        return ([ip_key(ip)] if ip else []) + [user_key(username)]

    def _counted(self, key: str, t: Throttle | None, now: datetime) -> Throttle:
        """The row with one more failure; at the threshold it is locked (doubling per lock for usernames).
        A username's lock history counts only locks that ended less than lockout_max_hours ago."""
        is_ip = key.startswith("ip:")
        window = timedelta(minutes=self.s.ip_window_min if is_ip else self.s.lockout_window_min)
        threshold = self.s.ip_failure_limit if is_ip else self.s.lockout_threshold
        failures, start = (t.failures, parse(t.window_start)) if t else (0, now)
        lock_count, locked_until = (t.lock_count, t.locked_until) if t else (0, None)
        if now - start > window:
            failures, start = 0, now
        failures += 1
        if failures >= threshold:
            if is_ip:
                duration = timedelta(minutes=self.s.ip_block_min)
            else:
                if locked_until and now - parse(locked_until) >= timedelta(hours=self.s.lockout_max_hours):
                    lock_count = 0  # the last lock is long over: start doubling again from the base
                lock_count += 1
                duration = min(timedelta(minutes=self.s.lockout_base_min * 2 ** (lock_count - 1)),
                               timedelta(hours=self.s.lockout_max_hours))
            locked_until, failures, start = iso(now + duration), 0, now
        return Throttle(key, failures, iso(start), lock_count, locked_until)

    def _retry(self, rows: ThrottleRows, now: datetime) -> int | None:
        waits = [math.ceil((parse(t.locked_until) - now).total_seconds()) for t in rows.values() if _active(t, now)]
        return max(1, max(waits)) if waits else None

    def check(self, username: str, ip: str | None) -> int | None:
        """Seconds until the caller may try again, or None. Read-only: password checks use begin()."""
        now = self.clock()
        return self._retry({key: self.store.get_throttle(key) for key in self._keys(username, ip)}, now)

    def locked_until(self, username: str) -> str | None:
        t = self.store.get_throttle(user_key(username))
        return t.locked_until if _active(t, self.clock()) else None

    def begin(self, username: str, ip: str | None) -> Attempt:
        """Before verifying a password: refuse while the username or the address is locked, else count this
        attempt as a failure, both in one transaction."""
        now = self.clock()
        refused: list[int] = []

        def reserve(rows: ThrottleRows) -> ThrottleRows:
            retry = self._retry(rows, now)
            if retry:
                refused.append(retry)
                return {}
            return {key: self._counted(key, t, now) for key, t in rows.items()}

        before, after = self.store.update_throttles(self._keys(username, ip), reserve)
        return Attempt(refused[0]) if refused else Attempt(None, before, after)

    def succeed(self, attempt: Attempt) -> None:
        """The password was right: take the attempt's count back. The username's failure counter starts again
        at zero but its lock history (lock_count) stays until it expires; an address keeps the failures others
        added. A lock set by this attempt's own count is lifted, unless something changed it since."""
        now = self.clock()

        def refund(rows: ThrottleRows) -> ThrottleRows:
            changes: ThrottleRows = {}
            for key, t in rows.items():
                before, after = attempt.before.get(key), attempt.after.get(key)
                if t is None or after is None:  # unlocked or purged meanwhile: nothing to give back
                    continue
                own_lock = _sets_lock(before, after) and t.locked_until == after.locked_until
                if own_lock:
                    t = before or Throttle(key, 0, t.window_start, 0, None)
                elif key.startswith("ip:") and t.window_start == after.window_start and t.failures > 0:
                    t = Throttle(key, t.failures - 1, t.window_start, t.lock_count, t.locked_until)
                if not key.startswith("ip:"):
                    t = Throttle(key, 0, t.window_start, t.lock_count, t.locked_until)
                keep = t.failures or t.lock_count or _active(t, now)
                changes[key] = t if keep else None
            return changes

        self.store.update_throttles(list(attempt.after), refund)

    def fail(self, username: str, ip: str | None) -> list[Lock]:
        """Count one failure without checking the locks; returns the locks it set. (Password checks use
        begin(), which counts in advance.)"""
        now = self.clock()
        before, after = self.store.update_throttles(
            self._keys(username, ip), lambda rows: {k: self._counted(k, t, now) for k, t in rows.items()})
        return Attempt(None, before, after).locks
