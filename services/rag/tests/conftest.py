"""rag reads its key paths at import (the verifier is built with the app), so throwaway keys are made and
pointed to before any test module imports rag.api."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from rag_common.testing import IdentityKeys

_folder = Path(tempfile.mkdtemp(prefix="rag-test-keys-"))
atexit.register(shutil.rmtree, _folder, ignore_errors=True)
KEYS = IdentityKeys(_folder)
os.environ["RAG_PUBLIC_KEYS_FILE"] = str(KEYS.public_keys_file)
os.environ["RAG_SIGNING_KEY_FILE"] = str(KEYS.key_files["rag"])
os.environ["RAG_AUDIT_URL"] = ""
os.environ["RAG_QDRANT_API_KEY_FILE"] = ""  # no key: the tests use an in-memory Qdrant or none
