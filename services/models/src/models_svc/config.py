from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODELS_", env_file=".env", extra="ignore")

    text_model_path: Path = Path("/models/hf/Qwen3-Embedding-0.6B")
    image_model_path: Path = Path("/models/hf/Qwen3-VL-Embedding-2B")
    rerank_model_path: Path = Path("/models/hf/Qwen3-Reranker-0.6B")
    device: str = "cuda"
    log_level: str = "INFO"

    query_instruction: str = "Given a question, retrieve passages that answer it"
    image_query_instruction: str = "Find images, figures or charts that help answer the question"
    rerank_instruction: str = "Given a question about the user's documents, judge whether the passage helps answer it"

    max_seq_tokens: int = 1024  # chunks are at most 512 tokens; this bounds activation memory
    text_batch: int = 32
    image_batch: int = 2
    rerank_batch: int = 16
    image_max_side: int = 1024
    max_waiting: int = 32  # GPU requests running or waiting (the running one counts) before answering 503


@lru_cache
def get_settings() -> Settings:
    return Settings()
