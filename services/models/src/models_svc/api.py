"""models: text embeddings, image embeddings and reranking over HTTP."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from rag_common.contracts import (
    EmbedImageRequest,
    EmbedResponse,
    EmbedTextRequest,
    ModelsInfo,
    RerankRequest,
    RerankResponse,
)
from rag_common.service import create_app

from models_svc.config import get_settings
from models_svc.encoders import Encoders
from models_svc.gate import Busy, GpuGate

settings = get_settings()
state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["encoders"] = Encoders(settings)
    state["gate"] = GpuGate(settings.max_waiting)
    yield
    state.clear()


async def readiness() -> dict[str, bool]:
    return {"models_loaded": "encoders" in state}


app = create_app("models", lifespan=lifespan, readiness=readiness, log_level=settings.log_level)


@app.exception_handler(Busy)
async def busy(_request: Request, _exc: Busy) -> JSONResponse:
    return JSONResponse({"detail": "GPU queue full"}, status_code=503, headers={"Retry-After": "2"})


def _encoders() -> Encoders:
    if "encoders" not in state:
        raise HTTPException(503, "models are loading")
    return state["encoders"]  # type: ignore[return-value]


async def _run(fn, *args):
    return await state["gate"].run(fn, *args)  # type: ignore[union-attr]


@app.get("/v1/info")
def info() -> ModelsInfo:
    enc = _encoders()
    text_dim, image_dim = enc.dims
    return ModelsInfo(**enc.names, text_dim=text_dim, image_dim=image_dim)


@app.post("/v1/embed/text")
async def embed_text(body: EmbedTextRequest) -> EmbedResponse:
    vectors = await _run(_encoders().embed_text, body.texts, body.role)
    return EmbedResponse(vectors=vectors, dim=len(vectors[0]))


@app.post("/v1/embed/image")
async def embed_image(body: EmbedImageRequest) -> EmbedResponse:
    enc = _encoders()
    try:
        if body.images_b64 is not None:
            vectors = await _run(enc.embed_images, body.images_b64)
        else:
            vectors = await _run(enc.embed_image_queries, body.texts)
    except (ValueError, OSError) as exc:  # bad base64 or undecodable image
        raise HTTPException(400, f"invalid image: {exc}") from exc
    return EmbedResponse(vectors=vectors, dim=len(vectors[0]))


@app.post("/v1/rerank")
async def rerank(body: RerankRequest) -> RerankResponse:
    return RerankResponse(scores=await _run(_encoders().rerank, body.query, body.documents))
