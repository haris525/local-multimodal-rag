import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rag_common import identity


def pem(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def raw(key: Ed25519PrivateKey) -> bytes:
    return identity.public_key_raw(key)


class Clock:
    def __init__(self, now: float = 1_800_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def keys() -> dict[str, Ed25519PrivateKey]:
    return {"auth": Ed25519PrivateKey.generate(), "rag": Ed25519PrivateKey.generate()}


@pytest.fixture
def public_keys_file(tmp_path, keys):
    path = tmp_path / "identity_public.json"
    path.write_text(json.dumps({"version": 1, "keys": {k: base64.b64encode(raw(v)).decode() for k, v in keys.items()}}))
    return path


@pytest.fixture(autouse=True)
def single_user_mode(monkeypatch):
    """identity_middleware flips a process-wide flag; every test starts in single-user mode."""
    monkeypatch.setattr(identity, "_enforced", False)
