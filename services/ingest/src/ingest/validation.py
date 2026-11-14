"""Upload validation: extension allow-list, magic bytes, and safe file names."""

import re
import unicodedata
from pathlib import PurePath

from rag_common.contracts import DocType

ALLOWED: dict[str, DocType] = {
    ".txt": DocType.TEXT,
    ".md": DocType.TEXT,
    ".pdf": DocType.PDF,
    ".xlsx": DocType.SPREADSHEET,
    ".csv": DocType.SPREADSHEET,
    ".png": DocType.IMAGE,
    ".jpg": DocType.IMAGE,
    ".jpeg": DocType.IMAGE,
    ".webp": DocType.IMAGE,
}

_MAGIC: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".xlsx": (b"PK\x03\x04",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}


class InvalidUpload(ValueError):
    pass


def safe_name(filename: str) -> str:
    name = PurePath(filename.replace("\\", "/")).name
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"[\x00-\x1f\x7f/:*?\"<>|]", "_", name).strip(" .")
    if not name:
        raise InvalidUpload("empty file name")
    stem, dot, ext = name.rpartition(".")
    return (stem[: 200 - len(ext)] + dot + ext) if dot else name[:200]


def validate(name: str, head: bytes) -> tuple[str, DocType]:
    """Return (extension, doc type) or raise InvalidUpload. `head` is the first bytes of the file."""
    ext = PurePath(name).suffix.lower()
    if ext not in ALLOWED:
        raise InvalidUpload(f"unsupported file type '{ext or name}'; allowed: {', '.join(sorted(ALLOWED))}")
    if ext == ".webp":
        ok = head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    elif ext in _MAGIC:
        ok = head.startswith(_MAGIC[ext])
    else:  # text formats: reject binaries (UTF-16 files carry a BOM and are fine)
        ok = head.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" not in head[:8192]
    if not ok:
        raise InvalidUpload(f"file content does not match its '{ext}' extension")
    return ext, ALLOWED[ext]
