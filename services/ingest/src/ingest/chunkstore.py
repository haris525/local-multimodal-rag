"""Per-document JSONL copy of every chunk payload: serves chunk listings and exact-match grep."""

import heapq
import json
import os
import re
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from rag_common.contracts import GrepHit


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def records(path: Path) -> Iterator[dict]:
    """Read one record at a time; published files are immutable and never cached in full."""
    try:
        f = path.open(encoding="utf-8")
    except FileNotFoundError:
        return
    with f:
        for line in f:
            yield json.loads(line)


def read(path: Path) -> tuple[dict, ...]:
    return tuple(records(path))


def page(path: Path, offset: int, limit: int) -> list[dict]:
    return list(islice(records(path), offset, offset + limit))


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
    def hits():
        for doc_id, path in paths.items():
            for record in records(path):
                counts = [sum(1 for _ in p.finditer(record["text"])) for p in patterns]
                matched = sum(1 for c in counts if c)
                if matched:
                    yield GrepHit(id=record["id"], doc_id=doc_id, chunk_index=record["chunk_index"],
                                  matched=matched, count=sum(counts))

    return heapq.nsmallest(limit, hits(), key=lambda h: (-h.matched, -h.count, h.doc_id, h.chunk_index))
