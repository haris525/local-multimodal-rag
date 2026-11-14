"""argon2id password hashing and the password policy.

Passwords are NFKC-normalised before hashing and checking, so the same password typed on different keyboards
or systems (composed or decomposed accents) always matches.
"""

import asyncio
import secrets
import string
import unicodedata
from functools import lru_cache
from importlib.resources import files

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from auth.config import Settings


class PolicyError(ValueError):
    pass


def normalize(password: str) -> str:
    return unicodedata.normalize("NFKC", password)


class Passwords:
    def __init__(self, s: Settings):
        self.hasher = PasswordHasher(time_cost=s.argon2_time_cost, memory_cost=s.argon2_memory_kib,
                                     parallelism=s.argon2_parallelism, hash_len=32, salt_len=16, type=Type.ID)
        self._slots = asyncio.Semaphore(s.max_concurrent_hashes)  # bounds memory: each hash uses memory_kib
        self._dummy = self.hasher.hash(secrets.token_urlsafe(16))  # same cost as a real verify

    def hash_sync(self, password: str) -> str:
        return self.hasher.hash(normalize(password))

    def verify_sync(self, password_hash: str, password: str) -> bool:
        try:
            return self.hasher.verify(password_hash, normalize(password))
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    async def hash(self, password: str) -> str:
        async with self._slots:
            return await asyncio.to_thread(self.hash_sync, password)

    async def verify(self, password_hash: str, password: str) -> bool:
        async with self._slots:
            return await asyncio.to_thread(self.verify_sync, password_hash, password)

    def needs_rehash(self, password_hash: str) -> bool:
        try:
            return self.hasher.check_needs_rehash(password_hash)
        except InvalidHashError:
            return True

    async def dummy_verify(self, password: str) -> None:
        """Spend the time of a real verify, so an unknown username answers as slowly as a wrong password."""
        await self.verify(self._dummy, password)


@lru_cache(maxsize=1)
def blocklist() -> frozenset[str]:
    text = files("auth").joinpath("data/common_passwords.txt").read_text(encoding="utf-8")
    return frozenset(line.strip().lower() for line in text.splitlines() if line.strip() and not line.startswith("#"))


_TRAILING = string.digits + string.punctuation + " "


def check_policy(password: str, username: str, s: Settings) -> None:
    """Length, not common, no username. No composition rules (NIST SP 800-63B)."""
    password = normalize(password)
    if len(password) < s.min_password_length:
        raise PolicyError(f"Use at least {s.min_password_length} characters.")
    if len(password) > s.max_password_length:
        raise PolicyError(f"Use at most {s.max_password_length} characters.")
    lowered = password.lower()
    # A listed password with digits or symbols appended ("Password1234!") is just as guessable.
    if lowered in blocklist() or lowered.rstrip(_TRAILING) in blocklist():
        raise PolicyError("This password is too common.")
    if username and username.lower() in lowered:
        raise PolicyError("Don't include your username.")


def same_password(a: str, b: str) -> bool:
    return secrets.compare_digest(normalize(a).encode(), normalize(b).encode())


def generate_temp_password() -> str:
    return secrets.token_urlsafe(12)  # 16 characters, 96 bits
