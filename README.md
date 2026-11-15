# Local Multimodal RAG

Ask questions about your own documents (PDFs with tables and figures, Excel and CSV files, text,
Markdown and images) and get cited answers, entirely on one machine. Use an NVIDIA workstation or
the [Apple Silicon Mac setup](docs/MACOS.md) (native Metal inference with the application in Docker).

Answers can include **inline Mermaid diagrams**, with legends and source citations. Rendering stays
on this machine; diagrams also appear when saved chats are reopened.

- **Accounts:** everyone signs in with an account an admin created. Each user's documents, search
  index entries, chats, memories and audit trail are their own; admins manage accounts but cannot read
  anyone's content. The port stays bound to 127.0.0.1: read
  [DEPLOYMENT.md, section 5](docs/DEPLOYMENT.md#5-network-exposure) before you give other computers
  access.
- **Models (all local):** Qwen3.8-27B (quantized; reads images and calls tools) for answers, image
  reading and research; Qwen3-Embedding-0.6B for text; Qwen3-VL-Embedding-2B for images;
  Qwen3-Reranker-0.6B.
- **Retrieval:** dense vectors, native BM25, exact-match grep and image vectors, fused with reciprocal
  rank fusion, reranked, then per-document coverage and maximal marginal relevance over the up to 10
  documents you select.
- **Structured data:** every sheet and PDF table is also a SQL table; a planner chooses SQL, SQL-filtered
  search or plain search, and SQL runs in a read-only sandbox.
- **Deep search:** a LangGraph ReAct agent (plan, use tools, verify) for multi-document and
  calculation-heavy questions, the default; behind a switch (`RAG_DEEP_AGENT`), the same tools and answer step
  on LangChain's open-source [deepagents](https://pypi.org/project/deepagents/) package, tuned or as it ships,
  to compare the two. Optionally (off unless an operator enables it) it can run Python on the selected
  documents' tables and text in a locked-down sandbox with no network.
- **Read all (exhaustive mode):** for questions about the documents as a whole ("list every limitation",
  "which documents mention X"), the model reads every section of the selection, takes cited notes on each
  part, merges them and answers from all of them, with progress steps and notices for anything not read.
- **Chats and memory:** saved chats, follow-up questions that understand earlier turns, automatic
  summaries for long chats, and long-term memories of your role and preferences (view and delete in the UI).
- **Operations:** microservices behind one localhost port, signed identities between services, a
  durable ingestion queue, a hash-chained audit log per user, hardened containers, and evaluation with
  MRR, Recall and nDCG with confidence intervals.

## Documentation

| Read | For |
|---|---|
| [Getting started](docs/GETTING_STARTED.md) | the first install and your first questions |
| [Apple Silicon Mac setup](docs/MACOS.md) | install, build, launch, memory limits, stop/restart and troubleshooting on macOS |
| [Code review](docs/REVIEW.md) | confirmed concurrency bugs, fixes, validation and remaining risks |
| [Analysis performance](docs/ANALYSIS_PERFORMANCE.md) | ReAct tools, Python sandbox, and measured DuckDB/Polars/pandas comparison |
| [How it works](docs/HOW_IT_WORKS.md) | what happens inside the app, in plain words |
| [Sample documents](examples/documents/README.md) | openly licensed files to try, with questions and expected answers |
| [Deployment](docs/DEPLOYMENT.md) | running it on a server, backups, updates and security |
| [Architecture](docs/ARCHITECTURE.md) | the technical design, settings and measurements |
| [State machines](docs/STATE_MACHINES.md) | every entity that moves through states (documents, accounts, sessions, the chat stream and more), with diagrams and known limitations |
| [Plan: video, audio and more formats](docs/design/video-and-new-formats.md) | not built yet: how videos, audio and other file types could be ingested |
| [Changelog](CHANGELOG.md) | what changed |

## Quick start

**Apple Silicon Mac:** follow [MACOS.md](docs/MACOS.md). The commands below use NVIDIA CUDA and do not
run unchanged on a Mac.

Requirements: Docker with NVIDIA GPU support (Docker Engine 26 or newer, Docker Desktop 4.29 or newer)
and Compose v2.24 or newer, [uv](https://docs.astral.sh/uv/), a GPU with 32 GB of memory, 80 GB of free
disk (models and images take about 50 GB before your first document). New to Docker or GPUs? Follow
[Getting started](docs/GETTING_STARTED.md) instead. Upgrading an install from before accounts? Follow
[DEPLOYMENT.md, section 4.10](docs/DEPLOYMENT.md#410-upgrade-an-install-from-before-accounts).

```powershell
docker compose --profile setup run --rm fetch-models   # once: downloads about 25 GB of models
uv run --with cryptography scripts/init_secrets.py     # once: signing keys and the Qdrant api_key in secrets/
docker compose up -d --build                           # build and start everything
docker compose run --rm auth auth-admin create-admin --username <you>   # once: the first admin
```

Open http://localhost:8000, sign in, add documents, select up to 10, ask. Create accounts for other
people under your username > *Users*.

## Services

| Service | Role |
|---|---|
| `gateway` | nginx: UI and `/api` proxy with a sign-in check on every request, the only published port (127.0.0.1:8000) |
| `auth` | accounts, sign-in, sessions and API tokens; signs the caller's identity for the other services |
| `ingest-api` | uploads, document registry, figures, exact-match grep |
| `ingest-worker` | parse (Docling), SQL tables, describe images, chunk, summarize, embed, index |
| `sql` | sandboxed read-only DuckDB queries |
| `sandbox` | Deep search's model-written Python (`run_python`; compose profile `sandbox`, off unless enabled), one run at a time, reachable only from `rag`, no network, no secrets |
| `rag` | planner, hybrid retrieval, fusion, rerank, coverage and MMR, streamed cited answers, Deep search (LangGraph agent, or the deepagents package), exhaustive map and reduce |
| `audit` | append-only, hash-chained event log, per user, with details encrypted per user |
| `threads` | saved conversations, running summaries, long-term memories, per user |
| `models` | text and image embeddings, reranking (GPU) |
| `llm` | llama-server with Qwen3.8-27B (GPU) |
| `qdrant` | vector and BM25 index; every entry carries its owner, and every request needs the `api_key` |

## Development

Each Python service is its own uv project with its own lock file and tests. Run them from the
repository root:

```powershell
uv run --directory services/rag pytest
uv run --directory services/ingest pytest
uv run --directory services/sql pytest
uv run --directory services/audit pytest
uv run --directory services/threads pytest
uv run --directory services/auth pytest
docker build --target test -f services/sandbox/Dockerfile .  # sandbox: Linux resource limits, also on Mac/Windows
uv run --directory libs/common pytest
uv run --directory services/models pytest    # installs PyTorch, several GB
uv run --directory eval pytest
node --test services/gateway/tests/*.test.mjs
```

Live checks against a running stack. They run as the user of an API token (your username >
*API tokens*), kept in the shell and never in `.env`. Generate the sample files first; every check
deletes what it creates when it finishes:

```powershell
$env:RAG_TOKEN = "rag_pat_..."                                          # bash: export RAG_TOKEN=rag_pat_...
uv run --with fpdf2 --with openpyxl --with pillow scripts/make_samples.py samples
uv run --with httpx scripts/smoke_test.py --samples samples [--deep]   # all 11 questions should pass
uv run --with httpx scripts/memory_smoke.py --samples samples          # 23 memory and routing checks
```

For large synthetic CSVs, messy workbooks, long documents, concurrent reindexing and sustained memory
measurements in a disposable account, follow [Workload validation](docs/WORKLOAD_VALIDATION.md).

The isolation test signs in as an admin (password from `RAG_ADMIN_PASSWORD`) and checks that two
temporary accounts cannot reach each other's data:
`uv run --with httpx scripts/multiuser_smoke.py --admin-user <admin> [--docker]`
([DEPLOYMENT.md, section 9.7](docs/DEPLOYMENT.md#97-isolation-test)).

The Python sandbox's containment (no network, no other service, limits, nothing left between runs) is checked
on the running stack with `python scripts/check_sandbox_isolation.py`
([DEPLOYMENT.md, section 4.16](docs/DEPLOYMENT.md#416-update-python-for-deep-search-in-a-locked-down-sandbox)).

Evaluation: `docker compose --profile eval run --rm eval generate|retrieval|answers` with `EVAL_TOKEN`
set in the shell (see
[Getting started, section 8](docs/GETTING_STARTED.md#8-measure-retrieval-quality-optional)).

The Deep search benchmark compares the implementations (`langgraph`, `deepagents` and `deepagents_stock`, alone
or with options such as `+think`) on fixed questions with checked answers: the questions of the
[sample documents](examples/documents/README.md) plus subsets of HotpotQA and MuSiQue. It scores each reply
without a judge model (numbers within a tolerance, required facts, citations of the right documents) and runs
every variant on the same questions, interleaved in a randomized order. With the API token of a separate
benchmark account in `RAG_TOKEN`: `uv run --project eval python -m rag_eval bench-setup --set
local,hotpotqa,musique`, then `bench --variants langgraph,langgraph+think,deepagents,deepagents+think,deepagents_stock`
([Getting started, section 8](docs/GETTING_STARTED.md#8-measure-retrieval-quality-optional) has every step).

Settings live in `.env` (copy `.env.example`); every setting has a default. The ones that change a
deployment are listed in [DEPLOYMENT.md, section 2.3](docs/DEPLOYMENT.md#23-settings-that-change-the-deployment).

## License

MIT: see [LICENSE](LICENSE). Bundled third-party files (the Markdown and HTML sanitizer libraries of the web UI,
the common-password list, the sample documents and one test question) keep their own licenses, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
