from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_common.contracts import instruction_text


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    models_url: str = "http://models:8003"
    llm_url: str = "http://llm:8080"
    ingest_url: str = "http://ingest-api:8001"
    audit_url: str | None = "http://audit:8004"
    sql_url: str = "http://sql:8005"
    threads_url: str = "http://threads:8006"
    llm_model: str = "qwen3.8-27b"
    log_level: str = "INFO"
    # Identity: rag verifies the gateway's assertions (signed by auth) and re-signs its own downstream calls.
    signing_key_file: Path = Path("/run/secrets/rag_signing_key")
    public_keys_file: Path = Path("/run/secrets/identity_public_keys")
    qdrant_api_key_file: Path | None = Path("/run/secrets/qdrant_api_key")  # Qdrant refuses requests without it

    # the General search task's instruction (rag.search_tasks): the question is embedded after it; empty embeds the
    # bare question. Query side only, so changing it needs no reindex.
    query_instruction: str = "Given a question about the user's documents, retrieve passages with the facts needed to answer it"
    # retrieval depth per retriever
    dense_k: int = 40
    bm25_k: int = 40
    image_k: int = 10
    grep_k: int = 20
    image_min_cosine: float = 0.30
    # fusion and selection
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_bm25: float = 1.0
    weight_grep: float = 1.0
    weight_image: float = 0.5
    rerank_candidates: int = 30
    min_relevance: float = 0.05
    coverage_min_relevance: float = 0.005  # floor for a document's single best chunk (overview questions)
    relative_relevance: float = 0.25  # threshold = min(min_relevance, 0.25 x best score), adapts for overviews
    per_doc_k: int = 2
    mmr_lambda: float = 0.7  # 1 = pure relevance, 0 = pure diversity
    max_per_doc: int = 0  # cap on chunks from one document (0 = let MMR decide)
    min_keep: int = 2
    duplicate_cosine: float = 0.95
    min_sources: int = 8
    max_sources: int = 16
    # prompt budget. One llama-server slot holds llm_ctx / llm_parallel tokens (65,536 / 2 = 32,768) and a prompt
    # plus its output room must fit in it. Tokens are counted with tokenizer_file (the Qwen3 tokenizer.json the
    # ingest worker counts chunks with), or a conservative estimate when it is missing. A request's source
    # budget is the smaller of source_budget and what the slot leaves after the answer room, the figures, the
    # rules, memory, conversation and question; the answer's room is never more than the slot leaves.
    tokenizer_file: str = "/models/hf/Qwen3-Embedding-0.6B/tokenizer.json"
    tokenizer_factor: float = 1.05  # the served model's tokenizer counts up to 3.4% more than this file
    llm_ctx: int = 65536  # LLM_CTX, passed by compose
    llm_parallel: int = 2  # llama-server --parallel
    image_tokens: int = 1024  # one attached figure (llama-server --image-max-tokens)
    prompt_margin_tokens: int = 512  # kept free in every prompt (chat template, tokenizer differences)
    source_budget: int = 16000
    sql_budget: int = 2500  # one SQL result in a prompt (about 60-80 rows of a 6-column table)
    max_prompt_images: int = 2
    answer_tokens: int = 4096
    think_tokens: int = 8192
    # output budgets of the smaller model calls; each is retried once with retry_tokens_factor x the room when
    # the reply is cut at the limit (finish_reason "length"), never above retry_max_tokens
    planner_tokens: int = 600
    rewrite_tokens: int = 160
    intent_tokens: int = 600
    assistant_tokens: int = 800
    history_answer_tokens: int = 1000
    memory_tokens: int = 400
    compaction_tokens: int = 800
    retry_tokens_factor: float = 2.5
    retry_max_tokens: int = 4096
    retry_backoff_s: float = 2.0  # wait before the one retry of a busy (503), failed (5xx) or timed-out call
    # timeouts of the calls rag makes, in seconds
    llm_timeout_s: float = 900.0
    research_llm_timeout_s: float = 600.0
    models_timeout_s: float = 120.0
    ingest_timeout_s: float = 60.0
    sql_timeout_s: float = 60.0  # at least 2 x SQL_TIMEOUT_S (20) + SQL_QUEUE_WAIT_S (5), so the sandbox answers first
    threads_timeout_s: float = 30.0
    qdrant_timeout_s: float = 60.0
    # SQL from the planner and the research tools
    sql_max_rows: int = 200
    filter_max_rows: int = 5000
    schema_card_tokens: int = 6000  # schema cards in the planner prompt and describe_table (full cards first)
    list_tables_tokens: int = 3000
    profile_tokens: int = 1500  # one profile_column result
    # value structure in profile_column (rag.value_structure): for a text column whose values mostly contain a
    # separator, every distinct last part with its count while there are at most profile_parts_max of them (else
    # the most frequent and how many more), within profile_parts_tokens
    profile_parts_max: int = Field(default=150, ge=1, le=4000)  # the parts query returns at most this + 50 rows
    profile_parts_tokens: int = Field(default=2000, ge=100)
    # total and subtotal rows (rag.totals): describe_table and profile_column sum each measure per value for up to
    # totals_columns text columns of a table with at most totals_labels_max distinct
    # values; a value is a total when the others add up to 1 to 4 times its sum within totals_tolerance
    totals_labels_max: int = Field(default=500, ge=3, le=4000)
    totals_tolerance: float = Field(default=0.005, gt=0, le=0.1)
    totals_columns: int = Field(default=4, ge=0, le=20)  # 0 turns the sums check off; names are still marked
    history_messages: int = 6  # turns the query rewrite sees
    conversation_turns: int = 4  # turns (citations stripped) shown to the answer prompt
    summary_trigger_tokens: int = 3000  # compact a thread when its unsummarized turns pass this size
    summary_trigger_messages: int = 12  # ... or this many messages
    summary_keep_messages: int = 6  # most recent messages always kept verbatim
    summary_fold_messages: int = 24  # oldest messages folded into the summary in one run, at most ...
    summary_fold_tokens: int = 12000  # ... and within this many tokens; the rest wait for the next turn
    auto_memory: bool = True  # extract lasting user facts and preferences after each turn
    memory_duplicate_cosine: float = 0.78  # a new memory this close to a saved one is the same fact
    # deep research
    agent_max_steps: int = 10  # tool calls per deep-research question (a request's research_calls overrides it)
    agent_max_steps_limit: int = Field(default=30, ge=1)  # the largest research_calls a request may ask for
    agent_verify_rounds: int = 2  # times verify may send the agent back for missing facts
    agent_tokens: int = 2048  # output room of one research step (tool calls with long SQL)
    agent_temperature: float = 0.3  # the research loop; plan and verify use 0
    agent_preview_chars: int = 1000  # per passage in a tool result; SQL and peek results are shown whole
    # the research prompt (rules, tool schemas, question and every tool result so far) may grow until it and
    # a retried step's room (min(retry_max_tokens, agent_tokens x retry_tokens_factor)) fill the slot; then the
    # research stops with a notice
    agent_recursion_limit: int = 80
    verify_evidence_tokens: int = 16000  # shared fairly between the sources, so later ones are checked too
    grep_limit: int = 10
    peek_rows: int = 15  # raw rows from the top of a sheet the peek tool shows (and 8 from the bottom)
    peek_columns: int = 30  # columns shown; the rest are named in a "... N more" line
    peek_cell_chars: int = 40
    peek_tokens: int = 4000  # the whole peek result; rows are dropped from the middle to fit
    # which deep-research implementation runs (a chat request's `agent` overrides it): an
    # arm of rag.arms, langgraph (rag.agent), deepagents (rag.agent_deepagents, on the deepagents harness),
    # deepagents_stock (rag.agent_deepagents_stock, the harness out of the box) or an optional arm
    deep_agent: str = "langgraph"
    deep_think: bool = False  # thinking in the plan and verify steps of both (a request's `deep_think` overrides it)
    deep_think_tokens: int = 4096  # output room of a plan or verify step that thinks, the thought included
    # the deepagents harness only
    deep_evict_tokens: int = 2000  # a research result larger than this is kept in the request's virtual files
    deep_preview_tokens: int = 600  # ... and the model sees its first and last lines, within this many tokens
    deep_summary_tokens: int = 20000  # the research prompt is summarized when it passes this ...
    deep_summary_keep_tokens: int = 6000  # ... keeping the most recent messages within this many tokens verbatim
    deep_summary_input_tokens: int = 12000  # older messages one summary call reads, at most
    deep_subagents: bool = False  # offer task(): a researcher sub-agent with the same tools per sub-question
    deep_harness_calls: int = 12  # write_todos, read_file and task calls per question (research tools: agent_max_steps)
    # fan-out (rag.fanout; arms that support it): each task() reserves research calls from the question's
    # budget before its researcher starts, and the calls it does not use come back (a request's `fanout`
    # overrides deep_fanout)
    deep_fanout: bool = False
    deep_task_calls: int = Field(default=4, ge=1)  # research calls one task() reserves (scaled by research_calls)
    deep_max_tasks: int = Field(default=2, ge=1)  # researchers per question
    deep_task_concurrency: int = Field(default=2, ge=1)  # researchers running at once per request (the slot count)
    deep_task_return_tokens: int = Field(default=800, ge=1)  # a researcher's report to the lead is cut here
    deep_task_timeout_s: float = Field(default=300.0, gt=0)  # one researcher's time limit
    # exhaustive mode (rag.exhaustive; a request's mode "exhaustive"): every section of the selected documents is
    # read in map calls of at most exhaustive_group_tokens of passages, and the notes they return are merged
    # until they fit one answer prompt. The caps stop the reading with a notice that says what was not read.
    exhaustive_group_tokens: int = Field(default=8000, ge=500)  # passages in one map call (small sections packed)
    exhaustive_map_tokens: int = Field(default=1500, ge=64)  # output room of one map call (retried once with more)
    exhaustive_concurrency: int = Field(default=1, ge=1, le=2)  # map calls at once; 1 leaves a slot for other users
    exhaustive_max_input_tokens: int = Field(default=300_000, ge=1)  # passages one request may read in total
    exhaustive_max_sections: int = Field(default=500, ge=1)  # sections one request may read
    exhaustive_notes_tokens: int = Field(default=16000, ge=500)  # notes in the answer prompt, and in one merge call
    exhaustive_reduce_tokens: int = Field(default=4096, ge=64)  # output room of one merge call
    exhaustive_reduce_rounds: int = Field(default=3, ge=1)  # merge rounds before the notes are cut to fit

    # Python in deep search (rag.sandbox): the run_python tool, run by the sandbox service, for every arm. off: never
    # (a request's `python: true` is a 422); request: only for a request with `python: true` (the benchmark); on: for
    # every deep search unless the request says `python: false`. rag ships the selected documents' data with each run.
    deep_python: Literal["off", "request", "on"] = "off"
    sandbox_url: str = ""  # http://sandbox:8008 with the compose profile "sandbox"; empty: no sandbox
    sandbox_subnet: str = ""  # the sandbox network (compose: SANDBOX_SUBNET); every request from it is refused
    sandbox_timeout_s: float = Field(default=120.0, gt=0)  # one run's time limit (at most SANDBOX_MAX_TIMEOUT_S)
    sandbox_request_timeout_s: float = Field(default=600.0, gt=0)  # the whole call: the wait for the run before it
    # (SANDBOX_QUEUE_WAIT_S), the transfer and the run itself
    sandbox_input_mb: float = Field(default=256.0, gt=0)  # the documents' data shipped with one run, together
    sandbox_code_chars: int = Field(default=20000, ge=100)  # the longest code one call may send
    sandbox_result_tokens: int = Field(default=3000, ge=200)  # a run's output (or error) the model reads
    sandbox_user_runs: int = Field(default=1, ge=1)  # runs of one user in flight at once (the sandbox: one at a time)
    sandbox_request_seconds: float = Field(default=600.0, gt=0)  # Python run time one question may use, together
    sandbox_total_input_mb: float = Field(default=1024.0, gt=0)  # Python data rag holds for all questions together
    sandbox_fetches: int = Field(default=2, ge=1)  # data fetches for Python at once, for everyone (one per user)
    sandbox_cancel_wait_s: float = Field(default=60.0, gt=0)  # how long a cancelled run's end is waited for
    ready_cache_s: float = Field(default=5.0, ge=0)  # /readyz answers from its last check this long
    # sampling (model card; presence penalty 0 so citations and figures can repeat)
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 0.0
    think_temperature: float = 1.0
    think_top_p: float = 0.95
    # what set agent_max_steps: the setting, or a request's research_calls (rag.chat.request_settings)
    _budget_from: str = PrivateAttr(default="RAG_AGENT_MAX_STEPS")

    @property
    def slot_tokens(self) -> int:
        """One request's share of the llm context."""
        return self.llm_ctx // max(1, self.llm_parallel)

    @property
    def research_retry_tokens(self) -> int:
        """The room a cut research step is retried with (agent.invoke)."""
        return min(self.retry_max_tokens, max(self.agent_tokens + 1, int(self.agent_tokens * self.retry_tokens_factor)))

    @property
    def research_budget_from(self) -> str:
        """Where agent_max_steps came from, for the notices: RAG_AGENT_MAX_STEPS or the request's research_calls."""
        return self._budget_from

    @field_validator("deep_agent")
    @classmethod
    def _registered_agent(cls, value: str) -> str:
        from rag import arms  # at validation time: the arms import this module

        return arms.check(value)

    @model_validator(mode="after")
    def _fanout_arm(self) -> "Settings":
        from rag import arms

        if self.deep_fanout and not arms.supports_fanout(self.deep_agent):
            raise ValueError(f"RAG_DEEP_FANOUT is on, but {arms.no_fanout(self.deep_agent)} (RAG_DEEP_AGENT)")
        return self

    @model_validator(mode="after")
    def _sandbox(self) -> "Settings":
        if self.deep_python != "off" and not self.sandbox_url:
            raise ValueError(f"RAG_DEEP_PYTHON is {self.deep_python}, but RAG_SANDBOX_URL is empty (set it to "
                             "http://sandbox:8008 and start the compose profile sandbox)")
        if self.sandbox_subnet:
            import ipaddress

            try:
                ipaddress.ip_network(self.sandbox_subnet)
            except ValueError as exc:
                raise ValueError(f"RAG_SANDBOX_SUBNET {self.sandbox_subnet!r} is not a network: {exc}") from None
        if self.sandbox_request_timeout_s <= self.sandbox_timeout_s:
            raise ValueError(f"RAG_SANDBOX_REQUEST_TIMEOUT_S ({self.sandbox_request_timeout_s:g}) must be longer than "
                             f"RAG_SANDBOX_TIMEOUT_S ({self.sandbox_timeout_s:g}): it also covers the wait and the "
                             "transfer")
        return self

    @field_validator("query_instruction")
    @classmethod
    def _one_line(cls, value: str) -> str:
        try:
            return instruction_text(value)
        except ValueError as exc:
            raise ValueError(f"RAG_QUERY_INSTRUCTION: {exc}") from None

    @field_validator("qdrant_api_key_file", mode="before")
    @classmethod
    def _no_key_file(cls, value):
        return None if value == "" else value  # an empty setting means no key (tests, runs outside Docker)


@lru_cache
def get_settings() -> Settings:
    return Settings()
