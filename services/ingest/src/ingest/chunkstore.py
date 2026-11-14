"""Per-document JSONL copy of every chunk payload: serves chunk listings and exact-match grep."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from rag_common.contracts import GrepHit


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


@lru_cache(maxsize=64)
def _load(path: str, mtime: float) -> tuple[dict, ...]:
    with open(path, encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f)


def read(path: Path) -> tuple[dict, ...]:
    if not path.exists():
        return ()
    return _load(str(path), path.stat().st_mtime)


def _term_pattern(term: str) -> re.Pattern:
    """Literal, case-insensitive; anchored at word boundaries where the term starts or ends
    with a word character, so 'INV-42' does not match 'INV-420'."""
    body = re.escape(term)
    start = r"(?<!\w)" if term[:1].isalnum() else ""
    end = r"(?!\w)" if term[-1:].isalnum() else ""
    return re.compile(start + body + end, re.IGNORECASE)


def grep(paths: dict[str, Path], terms: list[str], limit: int) -> list[GrepHit]:
    """Rank chunks by distinct terms matched, then by total occurrences."""
    patterns = [_term_pattern(t) for t in terms]
    hits: list[GrepHit] = []
    for doc_id, path in paths.items():
        for record in read(path):
            counts = [len(p.findall(record["text"])) for p in patterns]
            matched = sum(1 for c in counts if c)
            if matched:
                hits.append(
                    GrepHit(id=record["id"], doc_id=doc_id, chunk_index=record["chunk_index"],
                            matched=matched, count=sum(counts))
                )
    hits.sort(key=lambda h: (-h.matched, -h.count, h.doc_id, h.chunk_index))
    return hits[:limit]
