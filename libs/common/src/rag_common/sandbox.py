"""The body of the sandbox service's POST /v1/execute, written by rag and read by the sandbox (ARCHITECTURE
section 7.10). One frame holds the code, its time limit and the files the code may read:

    4 bytes   header length H, unsigned, big-endian
    H bytes   header, UTF-8 JSON: {"version": 1, "code": str, "timeout_s": number,
                                   "files": [{"path": str, "size": int}, ...]}
    the files' bytes, one after another in the header's order, and nothing after them

The sandbox reads the frame whole into its own memory, never into a file, so the code of a run in progress
cannot read the next run's data from a temporary file. Paths are relative and made of safe parts (check_path):
no absolute paths, no "..", no hidden names, at most MAX_DEPTH parts, so a path can only name a file inside
the run's folder. Standard library only.
"""

import json
import re
import struct
from dataclasses import dataclass

VERSION = 1
MEDIA_TYPE = "application/octet-stream"
MAX_DEPTH = 4
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")  # whole part only (fullmatch): no trailing newline
_LENGTH = struct.Struct(">I")


class FrameError(ValueError):
    """The frame is malformed or breaks a rule; the message says which."""


@dataclass(frozen=True)
class InputFile:
    path: str
    data: bytes | memoryview


@dataclass(frozen=True)
class Job:
    code: str
    timeout_s: float
    files: tuple[InputFile, ...]

    @property
    def input_bytes(self) -> int:
        return sum(len(f.data) for f in self.files)


def check_path(path: object) -> str:
    """`path` if it is a safe relative path ("data/1_sales.xlsx/tables.duckdb"), else FrameError."""
    if not isinstance(path, str):
        raise FrameError("a file path must be a string")
    parts = path.split("/")
    if not 1 <= len(parts) <= MAX_DEPTH or not all(_PART.fullmatch(p) for p in parts):
        raise FrameError(f"unsafe file path {path[:120]!r}: use 1 to {MAX_DEPTH} parts of letters, digits, '.', '_' "
                         "and '-', each starting with a letter or digit")
    return path


def encode(code: str, timeout_s: float, files: list[tuple[str, bytes]]) -> list[bytes]:
    """The frame in pieces: the header, then each file's bytes as given (a large input is not copied again)."""
    for path, _ in files:
        check_path(path)
    header = json.dumps({"version": VERSION, "code": code, "timeout_s": timeout_s,
                         "files": [{"path": path, "size": len(data)} for path, data in files]}).encode()
    return [_LENGTH.pack(len(header)) + header, *(data for _, data in files)]


def decode(body: bytes, max_header_bytes: int, max_files: int) -> Job:
    """The job in a frame; FrameError for anything malformed: a bad header, an unsafe or repeated path, a file
    inside another file's path, or sizes that do not add up to the body."""
    view = memoryview(body)
    if len(view) < _LENGTH.size:
        raise FrameError("the body is shorter than its header length")
    (length,) = _LENGTH.unpack(view[:_LENGTH.size])
    if length > max_header_bytes:
        raise FrameError(f"the header is {length} bytes, more than {max_header_bytes}")
    start = _LENGTH.size + length
    if start > len(view):
        raise FrameError("the body ends inside its header")
    try:
        header = json.loads(bytes(view[_LENGTH.size:start]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FrameError(f"the header is not UTF-8 JSON: {exc}") from None
    if not isinstance(header, dict) or header.get("version") != VERSION:
        raise FrameError(f"the header must be a JSON object with version {VERSION}")
    code, timeout_s, listed = header.get("code"), header.get("timeout_s"), header.get("files", [])
    if not isinstance(code, str) or not code.strip():
        raise FrameError("code must be a non-empty string")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float) or not timeout_s > 0:
        raise FrameError("timeout_s must be a positive number")
    if not isinstance(listed, list) or len(listed) > max_files:
        raise FrameError(f"files must be a list of at most {max_files}")
    files, offset, seen = [], start, set()
    for item in listed:
        if not isinstance(item, dict) or set(item) != {"path", "size"}:
            raise FrameError("each file must be {path, size}")
        path, size = check_path(item["path"]), item["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise FrameError(f"file {path}: size must be a whole number of bytes")
        if path in seen:
            raise FrameError(f"file {path} is listed twice")
        seen.add(path)
        if offset + size > len(view):
            raise FrameError(f"file {path}: the body ends before its {size} bytes")
        files.append(InputFile(path, view[offset:offset + size]))
        offset += size
    if offset != len(view):
        raise FrameError(f"{len(view) - offset} bytes follow the last file")
    folders = {"/".join(p.split("/")[:i]) for p in seen for i in range(1, p.count("/") + 1)}
    if clash := sorted(seen & folders):
        raise FrameError(f"file {clash[0]} is also a folder of another file")
    return Job(code, float(timeout_s), tuple(files))
