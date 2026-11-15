import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_models


def test_copying_a_huggingface_cache_ignores_only_nested_metadata(tmp_path):
    src = tmp_path / ".cache" / "huggingface" / "snapshot"
    src.mkdir(parents=True)
    (src / "weights").write_bytes(b"weights")
    (src / ".cache").mkdir()
    (src / ".cache" / "metadata").write_text("do not copy")
    dest = tmp_path / "installed"
    fetch_models.copy_tree(src, dest)
    assert (dest / "weights").read_bytes() == b"weights"
    if os.name == "posix":
        assert (dest / "weights").stat().st_mode & 0o777 == 0o644
    assert not (dest / ".cache").exists()


def test_interrupted_copy_preserves_the_previous_file(tmp_path, monkeypatch):
    src, dest = tmp_path / "source", tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    (src / "weights").write_bytes(b"new weights")
    (dest / "weights").write_bytes(b"old")

    def interrupted(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(fetch_models.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        fetch_models.copy_tree(src, dest)
    assert (dest / "weights").read_bytes() == b"old"
    assert list(dest.iterdir()) == [dest / "weights"]
