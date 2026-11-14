"""Embedding calls to the models service and point writes to Qdrant. Every write and delete names the owner:
points carry it in their payload (user_id) and every delete filter includes it. An embedding call that the
models service's GPU gate turns away (503, queue full), that fails (5xx) or times out is retried once after
a backoff (rag_common.retry); a second failure fails the run, which the document's Retry (reindex) repeats."""

import base64
import logging

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_common.contracts import ModelsInfo
from rag_common.index import COLLECTION, IMAGE_VECTOR, TEXT_VECTOR, ChunkPayload, point_id, user_condition
from rag_common.retry import retry_once

log = logging.getLogger("ingest.indexer")


class Indexer:
    def __init__(self, qdrant: AsyncQdrantClient, models: httpx.AsyncClient, text_batch: int, image_batch: int,
                 upsert_batch: int, retry_backoff_s: float = 5.0):
        self.qdrant = qdrant
        self.models = models
        self.text_batch = text_batch
        self.image_batch = image_batch
        self.upsert_batch = upsert_batch
        self.retry_backoff_s = retry_backoff_s

    async def info(self) -> ModelsInfo:
        response = await self.models.get("/v1/info")
        response.raise_for_status()
        return ModelsInfo(**response.json())

    async def _post(self, path: str, body: dict) -> list[list[float]]:
        async def call():
            response = await self.models.post(path, json=body)
            response.raise_for_status()
            return response.json()["vectors"]

        return await retry_once(call, what=f"embedding ({path})", backoff_s=self.retry_backoff_s, log=log)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.text_batch):
            vectors += await self._post("/v1/embed/text", {"texts": texts[i : i + self.text_batch], "role": "document"})
        return vectors

    async def embed_images(self, images: list[bytes]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(images), self.image_batch):
            batch = [base64.b64encode(img).decode() for img in images[i : i + self.image_batch]]
            vectors += await self._post("/v1/embed/image", {"images_b64": batch})
        return vectors

    async def replace(
        self,
        user_id: str,
        doc_id: str,
        payloads: list[ChunkPayload],
        text_vectors: list[list[float]],
        image_vectors: dict[int, list[float]],
    ) -> None:
        """Upsert the new points first, then delete chunk indexes beyond the new count, so a
        re-indexed document is never partly missing."""
        points = [
            qm.PointStruct(
                id=point_id(doc_id, p.chunk_index),
                vector={TEXT_VECTOR: tv} | ({IMAGE_VECTOR: image_vectors[p.chunk_index]} if p.chunk_index in image_vectors else {}),
                payload=p.model_dump(exclude_none=True),
            )
            for p, tv in zip(payloads, text_vectors, strict=True)
        ]
        for i in range(0, len(points), self.upsert_batch):
            await self.qdrant.upsert(COLLECTION, points[i : i + self.upsert_batch], wait=True)
        stale = qm.Filter(
            must=[
                user_condition(user_id),
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                qm.FieldCondition(key="chunk_index", range=qm.Range(gte=len(payloads))),
            ]
        )
        await self.qdrant.delete(COLLECTION, points_selector=qm.FilterSelector(filter=stale), wait=True)

    async def delete(self, user_id: str, doc_id: str) -> None:
        await delete_points(self.qdrant, user_id, doc_id)


async def _delete(qdrant: AsyncQdrantClient, flt: qm.Filter) -> None:
    try:
        await qdrant.delete(COLLECTION, points_selector=qm.FilterSelector(filter=flt), wait=True)
    except UnexpectedResponse as exc:
        if exc.status_code != 404:  # no collection yet (nothing was ever indexed): nothing to delete
            raise


async def delete_points(qdrant: AsyncQdrantClient, user_id: str, doc_id: str) -> None:
    await _delete(qdrant, qm.Filter(must=[user_condition(user_id),
                                          qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]))


async def delete_user_points(qdrant: AsyncQdrantClient, user_id: str) -> None:
    """Every point of one owner (account deletion)."""
    await _delete(qdrant, qm.Filter(must=[user_condition(user_id)]))
