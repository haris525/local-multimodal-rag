"""Qwen3 text embedder, Qwen3-VL image embedder and Qwen3 reranker on one GPU."""

import base64
import io

import torch
from PIL import Image, ImageOps
from sentence_transformers import CrossEncoder, SentenceTransformer

from models_svc.config import Settings


class Encoders:
    def __init__(self, s: Settings):
        self.s = s
        dtype = (torch.bfloat16 if s.device.startswith("cuda") else
                 torch.float16 if s.device.startswith("mps") else torch.float32)
        kwargs = {"dtype": dtype}
        self.text = SentenceTransformer(str(s.text_model_path), device=s.device, model_kwargs=kwargs)
        self.text.max_seq_length = s.max_seq_tokens
        self.image = SentenceTransformer(str(s.image_model_path), device=s.device, model_kwargs=kwargs)
        self.reranker = CrossEncoder(
            str(s.rerank_model_path),
            device=s.device,
            model_kwargs=kwargs,
            prompts={"rag": s.rerank_instruction},
            default_prompt_name="rag",
            max_length=s.max_seq_tokens,
        )

    @property
    def names(self) -> dict[str, str]:
        return {
            "text_model": self.s.text_model_path.name,
            "image_model": self.s.image_model_path.name,
            "rerank_model": self.s.rerank_model_path.name,
        }

    @property
    def dims(self) -> tuple[int, int]:
        return self.text.get_sentence_embedding_dimension(), self.image.get_sentence_embedding_dimension()

    def embed_text(self, texts: list[str], role: str) -> list[list[float]]:
        prompt = f"Instruct: {self.s.query_instruction}\nQuery:" if role == "query" else ""
        vectors = self.text.encode(texts, prompt=prompt, batch_size=self.s.text_batch, normalize_embeddings=True)
        return vectors.tolist()

    def embed_image_queries(self, texts: list[str]) -> list[list[float]]:
        vectors = self.image.encode(
            texts, prompt=self.s.image_query_instruction, batch_size=self.s.text_batch, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_images(self, images_b64: list[str]) -> list[list[float]]:
        images = [self._decode(b) for b in images_b64]
        try:
            vectors = self.image.encode(images, batch_size=self.s.image_batch, normalize_embeddings=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # image batches spike memory; give it back to the LLM's headroom
            elif self.s.device.startswith("mps"):
                torch.mps.empty_cache()  # return unused Metal allocations to the native answer model
        return vectors.tolist()

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        scores = self.reranker.predict(
            [(query, d) for d in documents], batch_size=self.s.rerank_batch, activation_fn=torch.nn.Sigmoid()
        )
        return [float(x) for x in scores]

    def _decode(self, b64: str) -> Image.Image:
        image = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((self.s.image_max_side, self.s.image_max_side))
        return image
