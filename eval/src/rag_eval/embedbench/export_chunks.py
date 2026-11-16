"""Export one account's indexed chunks, exactly as search sees them, for the embedding benchmark.

Runs inside the ingest-worker container (it has the registry and Qdrant), read-only:

    docker compose exec -T -e EXPORT_OWNER=<user id> ingest-worker python - < export_chunks.py > chunks.jsonl

For every ready document of that owner it reads the published generation from the registry (opened read-only),
then scrolls Qdrant with the owner, document and generation filters, and prints one JSON line per chunk:
doc_name, doc_id, chunk_index, kind, text_sha and search_text (the text the app embeds and indexes for BM25).
No other account is read and nothing is written. The output holds that account's document text: keep it out of
the repository (.gitignore covers chunks*.jsonl).
"""

import asyncio
import json
import os
import sqlite3
import sys

from ingest.config import get_settings
from ingest.registry import Registry
from qdrant_client import models as qm
from rag_common.index import (
    COLLECTION,
    generation_condition,
    qdrant_client,
    user_condition,
)


def published(path, owner: str) -> list:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("SELECT * FROM documents WHERE owner_id = ? AND status = 'ready' ORDER BY name",
                          (owner,)).fetchall()
    finally:
        db.close()
    return [Registry._info(r) for r in rows]


async def main() -> None:
    owner = os.environ["EXPORT_OWNER"]
    s = get_settings()
    client = qdrant_client(s.qdrant_url, s.qdrant_api_key_file, timeout=int(s.qdrant_timeout_s))
    total = 0
    for doc in published(s.registry_path, owner):
        flt = qm.Filter(must=[user_condition(owner), qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc.doc_id)),
                              generation_condition(doc.generation)])
        offset, rows = None, []
        while True:
            points, offset = await client.scroll(COLLECTION, scroll_filter=flt, limit=1000, offset=offset,
                                                 with_payload=True, with_vectors=False)
            rows += [p.payload for p in points]
            if offset is None:
                break
        for p in sorted(rows, key=lambda p: p["chunk_index"]):
            if p.get("user_id") != owner:  # the filter already guarantees it; never print another owner's text
                raise SystemExit("a point of another owner came back")
            print(json.dumps({"doc_name": p["doc_name"], "doc_id": p["doc_id"], "chunk_index": p["chunk_index"],
                              "kind": p["kind"], "text_sha": p["text_sha"], "search_text": p["search_text"]},
                             ensure_ascii=False))
        total += len(rows)
        print(f"{doc.name}: {len(rows)} chunks (generation {doc.generation or 'legacy'})", file=sys.stderr)
    print(f"exported {total} chunks", file=sys.stderr)
    await client.close()


asyncio.run(main())
