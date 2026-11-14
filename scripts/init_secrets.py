"""Generate the keys the stack needs into secrets/ (never into the repository or .env).

    uv run --with cryptography scripts/init_secrets.py [--dir secrets] [--force]

Writes:
  auth_signing.key, rag_signing.key   Ed25519 private keys (PEM PKCS#8): auth and rag sign identity assertions
  identity_public.json                their public keys, read by every service that verifies assertions
  qdrant_api.key                      the Qdrant API key
  qdrant_local.yaml                   Qdrant config fragment that enables that key

Refuses to overwrite existing files without --force. Rotating keys (--force) needs a restart of every service;
sessions survive because they live in the auth database. Compose file secrets keep the host permissions and
the containers run as non-root users, so on POSIX the folder is 0700 and the files 0644.
"""

import argparse
import base64
import json
import os
import secrets
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

FILES = ("auth_signing.key", "rag_signing.key", "identity_public.json", "qdrant_api.key", "qdrant_local.yaml")


def _private_pem(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _public_b64(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def generate(folder: Path, force: bool = False) -> list[Path]:
    existing = [folder / name for name in FILES if (folder / name).exists()]
    if existing and not force:
        raise FileExistsError(f"{', '.join(str(p) for p in existing)} already exist; pass --force to replace them "
                              "(then restart every service)")
    folder.mkdir(parents=True, exist_ok=True)
    keys = {"auth": Ed25519PrivateKey.generate(), "rag": Ed25519PrivateKey.generate()}
    qdrant_key = secrets.token_urlsafe(32)
    contents = {
        "auth_signing.key": _private_pem(keys["auth"]),
        "rag_signing.key": _private_pem(keys["rag"]),
        "identity_public.json": (json.dumps({"version": 1, "keys": {k: _public_b64(v) for k, v in keys.items()}},
                                            indent=2) + "\n").encode(),
        "qdrant_api.key": (qdrant_key + "\n").encode(),
        "qdrant_local.yaml": f'service:\n  api_key: "{qdrant_key}"\n'.encode(),
    }
    written = []
    for name, data in contents.items():
        path = folder / name
        path.write_bytes(data)
        written.append(path)
    if os.name == "posix":
        folder.chmod(0o700)
        for path in written:
            path.chmod(0o644)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", default="secrets", help="output folder (default: secrets)")
    parser.add_argument("--force", action="store_true", help="replace existing files (rotates every key)")
    args = parser.parse_args()
    try:
        written = generate(Path(args.dir), args.force)
    except FileExistsError as exc:
        sys.exit(f"error: {exc}")
    for path in written:
        print(f"wrote {path}")
    print("Keep this folder out of backups you share and out of the repository; never copy keys into .env.")


if __name__ == "__main__":
    main()
