"""The API module opens its store and builds its verifier on import, so throwaway keys and a database are
pointed to before any test imports audit.api."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from rag_common.testing import IdentityKeys

_folder = Path(tempfile.mkdtemp(prefix="audit-test-"))
atexit.register(shutil.rmtree, _folder, ignore_errors=True)
KEYS = IdentityKeys(_folder)
os.environ["AUDIT_PUBLIC_KEYS_FILE"] = str(KEYS.public_keys_file)
os.environ["AUDIT_DB"] = str(_folder / "audit.db")
