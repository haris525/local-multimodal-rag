from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ingest.chunking import ChunkingConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", env_file=".env", extra="ignore")

    data_dir: Path = Path("/data")
    qdrant_url: str = "http://qdrant:6333"
    models_url: str = "http://models:8003"
    vision_url: str = "http://llm:8080"  # any OpenAI-compatible vision endpoint
    vision_model: str = "qwen3.8-27b"
    # The vision server's weights file (its name carries the quantization) and vision projector: with the model
    # name, the model identity in every LLM cache key. compose passes LLM_GGUF and LLM_MMPROJ.
    vision_model_file: str = ""
    vision_mmproj: str = ""
    audit_url: str | None = "http://audit:8004"
    tokenizer_file: str = "/models/hf/Qwen3-Embedding-0.6B/tokenizer.json"
    docling_artifacts: str | None = "/opt/docling-models"
    log_level: str = "INFO"
    public_keys_file: Path = Path("/run/secrets/identity_public_keys")  # ingest-api verifies X-Identity with it
    qdrant_api_key_file: Path | None = Path("/run/secrets/qdrant_api_key")  # Qdrant refuses requests without it

    max_upload_mb: int = 200
    parse_timeout_s: int = 1800
    parse_threads: int = 8
    parser_cache_mb: int = 1024  # recycle an idle parser above this RSS; 0 recycles after every parse
    max_attempts: int = 2
    replace_same_name: bool = True  # a new upload with an existing file name supersedes it once ready
    poll_interval_s: float = 1.0
    run_lease_s: float = 90.0  # a running job's claim lasts this long without renewal (renewed every 10 s)
    sweep_interval_s: float = 60.0  # take back expired claims, finish deletions nobody holds (also during a job)
    # a replaced index generation (after a reindex) stays readable this long for questions that started on it,
    # then the sweep removes its points and files; 0 removes it at the next sweep
    generation_retain_h: float = Field(default=24.0, ge=0)
    worker_id: str = ""  # claimed_by of this worker's jobs; empty: the host name (a container keeps it on restart)

    max_figures_per_doc: int = 150
    # figure descriptions stop (the rest searchable by caption only, with a warning) after this many failures
    # in a row, or once describing has taken this long: each failure can take two vision timeouts
    figure_failure_limit: int = 3
    figure_deadline_s: float = 3600.0
    figure_description_tokens: int = 4000
    image_description_tokens: int = 4000
    chart_extraction: bool = True  # read a chart's data points first, then describe it with them (two passes)
    chart_extraction_tokens: int = 2000
    describe_temperature: float = 0.0  # repeatable descriptions (was 0.2)
    describe_max_side: int = 1536  # pixels; llama-server also caps image tokens (--image-max-tokens)
    summary_tokens: int = 1200  # the document summary (a plain paragraph needs about 200)
    summary_retry_tokens: int = 3000  # a cut summary is asked again once with 2.5x the room, at most this
    retry_max_tokens: int = 8000  # a description or summary cut at its limit is retried once, up to this
    retry_backoff_s: float = 5.0  # wait before the one retry of a busy (503), failed (5xx) or timed-out call
    # The LLM cache (llmcache.py): model replies reused for content seen before, per owner. Off: every call goes to
    # the model; entries already stored are still purged with their documents and accounts.
    llm_cache: bool = True
    llm_cache_max_mb: int = 1024  # past either bound the least recently used entries go
    llm_cache_max_entries: int = 200_000
    llm_cache_timeout_s: float = 10.0  # a lookup or write waits this long for the file's lock, then is skipped
    llm_cache_max_errors: int = 3  # cache errors in one job after which that job stops using the cache
    llm_cache_salt: str = ""  # part of every key: change it to start afresh (a new llm image, other llm flags)
    vision_timeout_s: float = 600.0
    models_timeout_s: float = 600.0
    qdrant_timeout_s: float = 120.0
    raw_table_max_rows: int = 100_000  # <table>__raw rows; the parse child reads INGEST_RAW_TABLE_MAX_ROWS
    max_rows_indexed: int = 20_000
    min_pdf_table_rows: int = 3

    chunk_target_tokens: int = 256
    chunk_max_tokens: int = 384
    table_max_tokens: int = 512

    embed_batch: int = 32
    image_embed_batch: int = 2
    upsert_batch: int = 128

    @field_validator("qdrant_api_key_file", mode="before")
    @classmethod
    def _no_key_file(cls, value):
        return None if value == "" else value  # an empty setting means no key (tests, runs outside Docker)

    @property
    def vision_settings(self) -> "VisionSettings":
        from ingest.describe import VisionSettings

        return VisionSettings(model=self.vision_model, temperature=self.describe_temperature,
                              max_side=self.describe_max_side, chart_extraction=self.chart_extraction,
                              extraction_tokens=self.chart_extraction_tokens, retry_max_tokens=self.retry_max_tokens,
                              retry_backoff_s=self.retry_backoff_s)

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            text_target=self.chunk_target_tokens,
            text_max=self.chunk_max_tokens,
            table_max=self.table_max_tokens,
        )

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def tables_dir(self) -> Path:
        return self.data_dir / "tables"

    @property
    def chunks_dir(self) -> Path:  # chunk texts for grep
        return self.data_dir / "chunks"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.db"

    @property
    def llm_cache_path(self) -> Path:
        return self.data_dir / "llm_cache.db"

    @property
    def generation_retain_s(self) -> float:
        return self.generation_retain_h * 3600

    @property
    def leftover_grace_s(self) -> float:
        """How long the unpublished generation of a run that ended stays before the sweep removes it: no question
        reads it, but a run taken back by lease expiry may still have a writer finishing (four leases, at least
        10 minutes). A run that failed in this worker is removed at once instead."""
        return max(4 * self.run_lease_s, 600.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
