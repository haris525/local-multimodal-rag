import asyncio

import pytest

from auth.config import Settings
from auth.passwords import Passwords, PolicyError, blocklist, check_policy, generate_temp_password, same_password


def test_default_parameters_are_argon2id():
    p = Passwords(Settings())
    h = p.hash_sync("a long enough passphrase")
    assert h.startswith("$argon2id$v=19$m=65536,t=3,p=4$")
    assert p.verify_sync(h, "a long enough passphrase") and not p.verify_sync(h, "wrong")
    assert not p.verify_sync("not a hash", "x") and not p.needs_rehash(h)


def test_rehash_when_parameters_change(settings):
    weak = Passwords(settings).hash_sync("a long enough passphrase")
    assert Passwords(Settings()).needs_rehash(weak)
    assert Passwords(settings).needs_rehash("garbage")


def test_async_hash_verify_and_dummy(settings):
    p = Passwords(settings)

    async def run():
        h = await p.hash("a long enough passphrase")
        assert await p.verify(h, "a long enough passphrase")
        assert not await p.verify(h, "other passphrase")
        await p.dummy_verify("anything")

    asyncio.run(run())


def test_nfkc_normalisation(settings):
    p = Passwords(settings)
    composed, decomposed = "caf\u00e9 au lait 1234", "cafe\u0301 au lait 1234"
    assert p.verify_sync(p.hash_sync(composed), decomposed)
    assert same_password(composed, decomposed)


@pytest.mark.parametrize("password, message", [
    ("short", "Use at least 12 characters."),
    ("x" * 257, "Use at most 256 characters."),
    ("unbelievable", "This password is too common."),
    ("UNBELIEVABLE", "This password is too common."),
    ("Password123456!", "This password is too common."),  # a listed word with digits and symbols appended
    ("my name is alice smith", "Don't include your username."),
])
def test_policy_messages(password, message):
    with pytest.raises(PolicyError) as exc:
        check_policy(password, "alice", Settings())
    assert str(exc.value) == message


def test_policy_accepts_long_uncommon_passwords():
    check_policy("correct horse battery staple", "alice", Settings())
    check_policy("\u00fcn\u00efc\u00f6d\u00e9 pa\u00dfw\u00f6rd", "alice", Settings())  # counts characters


def test_blocklist_is_loaded_without_its_licence_header():
    words = blocklist()
    assert len(words) > 9000 and "password" in words and not any(w.startswith("#") for w in words)


def test_temporary_passwords():
    pw = generate_temp_password()
    assert len(pw) == 16 and pw != generate_temp_password()
