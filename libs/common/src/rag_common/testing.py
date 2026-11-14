"""Test helpers for the services: throwaway identity keys and signed X-Identity headers. Not used at runtime.

    keys = IdentityKeys(tmp_folder)          # writes identity_public.json and both signing keys
    client.get("/v1/threads", headers=keys.headers(USER_A))             # as the gateway would send it
    client.post(..., headers=keys.headers(USER_A, issuer="rag"))         # as rag would send it
    client.get("/internal/capabilities", headers=keys.headers(ADMIN, role="system"))
"""

import base64
import json
import uuid
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rag_common.identity import HEADER, Identity, Signer, public_key_raw


def new_user_id() -> str:
    return str(uuid.uuid4())


class IdentityKeys:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self._keys = {"auth": Ed25519PrivateKey.generate(), "rag": Ed25519PrivateKey.generate()}
        self.public_keys_file = folder / "identity_public.json"
        self.public_keys_file.write_text(json.dumps({"version": 1, "keys": {
            issuer: base64.b64encode(public_key_raw(key)).decode() for issuer, key in self._keys.items()}}))
        self.key_files = {}
        for issuer, key in self._keys.items():
            path = folder / f"{issuer}_signing.key"
            path.write_bytes(self.pem(issuer))
            self.key_files[issuer] = path

    def pem(self, issuer: str) -> bytes:
        return self._keys[issuer].private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                serialization.NoEncryption())

    def signer(self, issuer: str = "auth", **kwargs) -> Signer:
        return Signer(self.pem(issuer), issuer, **kwargs)

    def token(self, user_id: str, role: str = "user", issuer: str = "auth") -> str:
        return self.signer(issuer).sign(Identity(user_id, role))

    def headers(self, user_id: str, role: str = "user", issuer: str = "auth") -> dict[str, str]:
        return {HEADER: self.token(user_id, role, issuer)}
