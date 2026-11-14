"""The run_python frame (rag_common.sandbox): what rag writes, the sandbox reads; nothing else gets through."""

import json
import struct

import pytest

from rag_common.sandbox import MAX_DEPTH, FrameError, check_path, decode, encode


def frame(header: dict, tail: bytes = b"") -> bytes:
    raw = json.dumps(header).encode()
    return struct.pack(">I", len(raw)) + raw + tail


def test_a_frame_round_trips_with_its_files_in_order():
    files = [("data/manifest.json", b"{}"), ("data/1_sales.xlsx/tables.duckdb", b"\x00DUCK\xff"), ("data/e", b"")]
    job = decode(b"".join(encode("print(1)", 12.5, files)), max_header_bytes=10_000, max_files=10)
    assert job.code == "print(1)" and job.timeout_s == 12.5
    assert [(f.path, bytes(f.data)) for f in job.files] == files and job.input_bytes == 8


def test_encode_refuses_an_unsafe_path_before_anything_is_sent():
    with pytest.raises(FrameError, match="unsafe"):
        encode("print(1)", 5, [("../x", b"")])


@pytest.mark.parametrize("path", ["../etc/passwd", "/etc/passwd", "data/../../x", ".hidden", "data/.env", "a//b",
                                  "data\\x", "", "data/", "a/" * MAX_DEPTH + "b", "x" * 101, "data/na me", "~/x",
                                  "C:/x", "data/\u00e9t\u00e9"])
def test_unsafe_paths_are_refused(path):
    with pytest.raises(FrameError):
        check_path(path)


@pytest.mark.parametrize("path", ["data/manifest.json", "data/1_sales_2024.xlsx/text.jsonl", "a/b/c/d", "x.y-z_1"])
def test_safe_paths_pass(path):
    assert check_path(path) == path


@pytest.mark.parametrize("body, words", [
    (b"\x00", "shorter"),
    (struct.pack(">I", 99) + b"{}", "ends inside its header"),
    (struct.pack(">I", 4) + b"\xff\xfe{}", "not UTF-8 JSON"),
    (frame({"version": 2, "code": "x", "timeout_s": 1}), "version 1"),
    (frame({"version": 1, "code": "  ", "timeout_s": 1}), "code"),
    (frame({"version": 1, "code": "x", "timeout_s": 0}), "timeout_s"),
    (frame({"version": 1, "code": "x", "timeout_s": True}), "timeout_s"),
    (frame({"version": 1, "code": "x", "timeout_s": 1, "files": [{"path": "a", "size": 1, "mode": 7}]}, b"x"),
     "{path, size}"),
    (frame({"version": 1, "code": "x", "timeout_s": 1, "files": [{"path": "a", "size": -1}]}), "size"),
    (frame({"version": 1, "code": "x", "timeout_s": 1, "files": [{"path": "a", "size": 5}]}, b"abc"), "ends before"),
    (frame({"version": 1, "code": "x", "timeout_s": 1, "files": [{"path": "a", "size": 1}]}, b"ab"), "follow"),
    (frame({"version": 1, "code": "x", "timeout_s": 1, "files": [{"path": "a", "size": 1}, {"path": "a", "size": 1}]},
           b"ab"), "twice"),
    (frame({"version": 1, "code": "x", "timeout_s": 1, "files": [{"path": "d", "size": 1}, {"path": "d/f", "size": 1}]},
           b"ab"), "folder of another file"),
])
def test_malformed_frames_are_refused_with_the_reason(body, words):
    with pytest.raises(FrameError, match=words.replace("{", r"\{").replace("}", r"\}")):
        decode(body, max_header_bytes=10_000, max_files=10)


def test_header_and_file_count_caps():
    files = [(f"data/f{i}", b"") for i in range(3)]
    with pytest.raises(FrameError, match="at most 2"):
        decode(b"".join(encode("x", 1, files)), max_header_bytes=10_000, max_files=2)
    with pytest.raises(FrameError, match="more than 10"):
        decode(b"".join(encode("x" * 100, 1, [])), max_header_bytes=10, max_files=2)


@pytest.mark.parametrize("path", ["data/x\n", "data\n/x", "data/x\r", "data/x\x00"])
def test_a_trailing_newline_or_control_character_is_refused(path):
    with pytest.raises(FrameError):
        check_path(path)
