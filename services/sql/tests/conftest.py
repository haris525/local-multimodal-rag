"""The API module builds its verifier and reads its tables folder on import, so throwaway keys and a folder are
pointed to before any test imports sql_sandbox.api."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from rag_common.testing import IdentityKeys

_folder = Path(tempfile.mkdtemp(prefix="sql-test-"))
atexit.register(shutil.rmtree, _folder, ignore_errors=True)
KEYS = IdentityKeys(_folder / "keys")
TABLES = _folder / "tables"
TABLES.mkdir()
os.environ["SQL_PUBLIC_KEYS_FILE"] = str(KEYS.public_keys_file)
os.environ["SQL_TABLES_DIR"] = str(TABLES)
