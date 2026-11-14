"""Where a document's files live: one folder per owner under each kind, so no path is ever built from a
document id alone and another user's document simply does not exist under the caller's folder.

    files/<uid>/<doc>/source<ext>    assets/<uid>/<doc>/<sha16>.png
    tables/<uid>/<doc>.duckdb         chunks/<uid>/<doc>.jsonl

Owner and document ids are validated before any path is built (user uuid, 32 hex characters).
"""

import os
import re
import shutil
from pathlib import Path
from typing import Literal

from rag_common.contracts import DOC_ID
from rag_common.identity import USER_ID_RE

from ingest.config import Settings

Kind = Literal["files", "assets", "tables", "chunks"]
KINDS: tuple[Kind, ...] = ("files", "assets", "tables", "chunks")
DOC_ID_RE = re.compile(DOC_ID)


def kind_dir(settings: Settings, kind: Kind) -> Path:
    return {"files": settings.files_dir, "assets": settings.assets_dir, "tables": settings.tables_dir,
            "chunks": settings.chunks_dir}[kind]


def owner_root(settings: Settings, kind: Kind, owner_id: str) -> Path:
    if not isinstance(owner_id, str) or not USER_ID_RE.match(owner_id):
        raise ValueError("owner_id is not a user id")
    return kind_dir(settings, kind) / owner_id


def _doc(doc_id: str) -> str:
    if not isinstance(doc_id, str) or not DOC_ID_RE.match(doc_id):
        raise ValueError("doc_id is not a document id")
    return doc_id


def source_path(settings: Settings, owner_id: str, doc_id: str, ext: str) -> Path:
    return owner_root(settings, "files", owner_id) / _doc(doc_id) / f"source{ext}"


def asset_dir(settings: Settings, owner_id: str, doc_id: str) -> Path:
    return owner_root(settings, "assets", owner_id) / _doc(doc_id)


def table_path(settings: Settings, owner_id: str, doc_id: str) -> Path:
    return owner_root(settings, "tables", owner_id) / f"{_doc(doc_id)}.duckdb"


def chunk_path(settings: Settings, owner_id: str, doc_id: str) -> Path:
    return owner_root(settings, "chunks", owner_id) / f"{_doc(doc_id)}.jsonl"


def remove_artifacts(settings: Settings, owner_id: str, doc_id: str) -> None:
    for path in (source_path(settings, owner_id, doc_id, "").parent, asset_dir(settings, owner_id, doc_id)):
        shutil.rmtree(path, ignore_errors=True)
    for path in (table_path(settings, owner_id, doc_id), chunk_path(settings, owner_id, doc_id)):
        path.unlink(missing_ok=True)


def remove_worker_artifacts(settings: Settings, owner_id: str, doc_id: str) -> None:
    """What a job writes (figures, tables, chunk file); the uploaded file stays."""
    shutil.rmtree(asset_dir(settings, owner_id, doc_id), ignore_errors=True)
    for path in (table_path(settings, owner_id, doc_id), chunk_path(settings, owner_id, doc_id)):
        path.unlink(missing_ok=True)


def remove_owner(settings: Settings, owner_id: str) -> None:
    """Every file of one owner (account deletion)."""
    for kind in KINDS:
        shutil.rmtree(owner_root(settings, kind, owner_id), ignore_errors=True)


WORKER_KINDS: tuple[Kind, ...] = ("assets", "tables", "chunks")  # written only by ingest-worker


def remove_empty_owner_dirs(settings: Settings, owner_id: str) -> None:
    """The owner's folders that only the worker writes, when nothing is left in them (after a job's
    clean-up). files/<uid> is left alone: ingest-api may be placing an upload there right now."""
    for kind in WORKER_KINDS:
        try:
            owner_root(settings, kind, owner_id).rmdir()
        except OSError:  # missing, or not empty
            pass


# --- legacy migration -----------------------------------------------------------------------


def _legacy_moves(settings: Settings, owner_id: str, doc_id: str) -> list[tuple[Path, Path]]:
    """(old place, new place) for each kind: before accounts, files were keyed by the document id only."""
    return [
        (settings.files_dir / _doc(doc_id), source_path(settings, owner_id, doc_id, "").parent),
        (settings.assets_dir / doc_id, asset_dir(settings, owner_id, doc_id)),
        (settings.tables_dir / f"{doc_id}.duckdb", table_path(settings, owner_id, doc_id)),
        (settings.chunks_dir / f"{doc_id}.jsonl", chunk_path(settings, owner_id, doc_id)),
    ]


def move_legacy(settings: Settings, owner_id: str, doc_id: str, dry_run: bool = False) -> int:
    """Move a legacy document's files under its owner; returns how many moved (or would move). Only moves
    what exists at the old place and not yet at the new one, so a second run moves nothing."""
    moved = 0
    for old, new in _legacy_moves(settings, owner_id, doc_id):
        if old.exists() and not new.exists():
            if not dry_run:
                new.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old, new)
            moved += 1
    return moved


def orphans(settings: Settings, known: set[str]) -> list[str]:
    """Document ids left at the kind roots (the legacy layout) that no registry row knows. Reported, never
    touched."""
    found: set[str] = set()
    for kind in KINDS:
        folder = kind_dir(settings, kind)
        if folder.is_dir():
            for entry in folder.iterdir():
                stem = entry.name.split(".", 1)[0]
                if DOC_ID_RE.match(stem) and stem not in known:
                    found.add(stem)
    return sorted(found)
