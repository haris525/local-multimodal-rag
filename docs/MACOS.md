# Run on an Apple Silicon Mac

This guide starts with a fresh checkout and ends with signing in and asking about a file. You do not
need a data-science environment, Python notebooks, CUDA, or an NVIDIA GPU.

This configuration has been exercised on a 64 GB Apple Silicon Mac with real Qwen text/vision,
text and image embeddings, reranking, and ingestion of the six sample document formats. See the
[review report](REVIEW.md) for the exact tests and remaining consistency limits. Recheck memory and
answer quality for your own documents; the sample checks are not a sustained-load test.

The application and its databases run in Docker. Two model servers run directly on macOS so they can
use Apple's GPU through Metal. Ordinary Linux CUDA containers cannot use that GPU. Small containers
forward model requests to fixed local ports; they publish no ports themselves. The Windows/Linux
configuration remains in `compose.yaml`; the Mac changes are in `compose.macos.yaml`.

## 1. Requirements and memory

- An Apple Silicon Mac (M series), **64 GB unified memory recommended for this 27B configuration**.
  Smaller machines need a separately validated smaller model/configuration.
- At least **80 GB free disk**, plus room for your documents. Setup downloads roughly 25 GB of model
  files and builds several GB of application images and tools. Existing caches can occupy more space.
- [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/), running.
  Use Linux ARM64 containers. Compose must be **2.24.4 or newer** (`docker compose version`), and Docker
  Engine must be **28 or newer** (`docker version`), for the Compose tags and isolated sandbox network.
- Apple's Command Line Tools. In **Terminal** (Applications → Utilities → Terminal), run
  `xcode-select --install` and finish the installer if they are missing. This supplies Git and the compiler.
- [uv](https://docs.astral.sh/uv/getting-started/installation/). Install it with the official command
  below, then open a new Terminal and check `uv --version`. uv installs the required Python version
  and project packages locally; do not change the system Python.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

In Docker Desktop → Settings → Resources, give the Linux VM **16 GB memory** to start. The container
limits total about 12 GB without the optional Python sandbox; their actual use is usually lower.
The model servers use **additional host memory outside that VM**. Allow roughly 20–25 GB for the LLM
and 7–12 GB for the embedding/reranking service as initial estimates, not measured guarantees.
Keep room for macOS and other applications. Model weights, context, document parsing, and concurrent
requests all contribute. A container memory cap can terminate a busy service; it does not make a model smaller.

Before starting, inspect `docker ps` and Activity Monitor → Memory. Stop other large workloads yourself
if needed. The launcher does not stop other projects or change Docker Desktop's global settings.
Start with one user, one small document, and no optional Python sandbox.

## 2. Get the repository

Run these commands in Terminal. A command block can be copied as a whole:

```sh
git clone https://github.com/haris525/local-multimodal-rag.git
cd local-multimodal-rag
python3 scripts/macos.py doctor
```

Keep this folder: all following commands run from it. `doctor` lists tools, missing model files,
cached Qwen folders, Docker's memory allocation, and validates the merged configuration. A cache
folder alone does not prove that a model finished downloading.

Copy settings only if you want to change them: `cp .env.example .env`. The default web address is
`http://localhost:8000`. Ports 18080 and 18083 are reserved for this project's native model servers.

## 3. Prepare and build once

```sh
python3 scripts/macos.py setup
```

Leave Terminal open. Allow an hour or more for a first install on a slower connection: model downloads
are large, and compilation/building adds several minutes. Later starts reuse those files. The command:

1. Builds a pinned llama.cpp revision with Metal in `.runtime/llama.cpp/`. CMake is supplied through uv;
   Homebrew is not required.
2. Installs the model service's locked Mac dependencies into `services/models/.venv/`, without CUDA.
3. Downloads the exact GGUF model and its vision projector into `models/gguf/`, and the three retrieval
   models into `models/hf/`. It reuses Hugging Face's cache for the retrieval models before copying them
   into this layout. Repeat setup after an interrupted download; the downloader reuses completed files
   and handles partial transfers according to its cache state.
4. Generates `secrets/` on a first install. A complete existing set is reused, never rotated automatically.
5. Builds this project's Docker images, including document-parsing models. This step does not start the app.

The required files are:

| Role | Exact model / format |
|---|---|
| Answers, tools, image descriptions | `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| Vision projector for that GGUF | `mmproj-BF16.gguf` from the same repository |
| Text search | `Qwen/Qwen3-Embedding-0.6B` |
| Image search | `Qwen/Qwen3-VL-Embedding-2B` |
| Ranking search results | `Qwen/Qwen3-Reranker-0.6B` |

**Already have Qwen in MLX or Ollama?** A different file format cannot simply be renamed to GGUF. The
launcher uses llama.cpp for compatibility with this repo's image, tool-calling, reasoning, and token
budget behavior. MLX quantized weights and the tiny MTP draft are not inputs for that runtime.
Qwen3 embedding models with 4B or 8B parameters are also not substitutes for the configured 0.6B model:
changing an embedder requires checking dimensions/semantics and rebuilding the index.

If setup fails, read the final error, fix it, and rerun. Do not delete `secrets/` or use `--force` to get
past an error on an existing installation. An incomplete secrets folder needs restoration; see
[backup and restore](DEPLOYMENT.md#62-back-up).

## 4. Start the model servers

Use **two Terminal tabs** and leave both running. In each tab, first enter the repository folder with
`cd /the/path/to/local-multimodal-rag` (replace this with your actual path).

**Tab 1 — answers and image reading:**

```sh
python3 scripts/macos.py llm
```

Wait for the server to listen on port 18080. It loads the GGUF weights once. It uses two request slots
with a total context of 32,768 tokens, so each request gets 16,384. A token is a small piece of text;
larger context means more memory. Only this one answer model should be running for this project.

**Tab 2 — search models:**

```sh
python3 scripts/macos.py models
```

Wait for `Application startup complete`. Loading all three models takes time. Work is serialized with
small batches to bound temporary memory, and the PyTorch MPS allocator has a high-watermark limit.
That limit covers this process's MPS allocations; it is not a cap on all Mac memory or the LLM process.

Both servers bind to **127.0.0.1**. Do not change them to `0.0.0.0` to fix a connection problem.
To confirm they are ready, use a third tab:

```sh
curl --fail http://127.0.0.1:18080/health
curl --fail http://127.0.0.1:18083/readyz
```

## 5. Launch the containers and create your account

In the third tab, from the repository folder:

```sh
python3 scripts/macos.py compose up -d --wait --wait-timeout 600
python3 scripts/macos.py compose ps
```

All default services should become healthy. The launcher always combines both Compose files and uses
the separate project name **`local-rag-mac`**, so its volumes are separate from the NVIDIA setup.
Always use this helper for Mac commands; plain `docker compose up` selects the NVIDIA configuration.

Create the first admin, replacing `yourname` with a lowercase username of at least three characters:

```sh
python3 scripts/macos.py compose run --rm auth auth-admin create-admin --username yourname
```

Enter the password twice when prompted (at least 12 characters; typing is hidden).
Open **[http://localhost:8000](http://localhost:8000)** and sign in.

For a first test, upload `examples/documents/reference/us-constitution-1787.txt`, wait for **ready**,
select it, turn **Deep search** off for the first question, and ask “What are the three branches of
government?” Open a citation to check its source. Then try the
[sample documents and questions](../examples/documents/README.md). PDFs and images take longer.
For the automated smoke test and API tokens, follow [Getting started, section 7](GETTING_STARTED.md#7-check-that-it-works-optional).

### Enable Python analysis for Deep search

The ReAct agent chooses among hybrid search, literal `grep`, nearby-passage reads, table inspection,
column profiles, read-only DuckDB SQL, and optional `run_python`. Python is available to every Deep
search agent when enabled; ordinary single-shot search does not call it.

To enable it, add these settings to `.env` in the repository folder (create that text file if absent;
update an existing setting instead of keeping conflicting copies):

```dotenv
COMPOSE_PROFILES=sandbox
RAG_DEEP_PYTHON=on
RAG_SANDBOX_URL=http://sandbox:8008
SANDBOX_MEMORY=2g
SANDBOX_CPUS=2
SANDBOX_THREADS=2
RAG_SANDBOX_INPUT_MB=128
RAG_SANDBOX_TOTAL_INPUT_MB=256
```

Then build and start it with the rest of the app:

```sh
python3 scripts/macos.py compose build sandbox
python3 scripts/macos.py compose up -d --wait --wait-timeout 600
```

The sandbox adds up to **2 GB** to the container budget above. Docker's VM and other running projects
must have room for it. It runs one analysis at a time, has no secrets or document-volume mounts,
receives only the selected documents' tables/text, and removes each run's files and processes afterward.
The installed libraries include DuckDB, Polars, pandas, NumPy, SciPy, statsmodels, openpyxl, and matplotlib.
Printed analysis results become citable evidence; generated charts are currently not displayed in the UI.

Use **Deep search** to test analysis. The agent decides when to call Python; enabling the tool does not
force a Python call for every question. For a comparison of the installed tabular engines, run:

```sh
uv run --python 3.12 scripts/check_sandbox_isolation.py --macos
uv run --python 3.12 scripts/benchmark_analysis.py --macos --output /tmp/analysis-benchmark.json
```

See the [measured comparison and tool guidance](ANALYSIS_PERFORMANCE.md). The benchmark
uses synthetic inputs inside the live sandbox, checks that engines agree, and reports filtering,
literal text matching, grouping, and join times plus process memory. It measures data operations;
model generation and semantic-retrieval performance require separate measurements.

## 6. Stop, restart, and update

Stop this project's containers first, then press **Control-C** in each model-server tab:

```sh
python3 scripts/macos.py compose stop
```

Closing the browser does not stop the containers or native model servers.
Documents, accounts and chats stay in Docker volumes. To restart, repeat steps 4 and 5; skip account
creation and setup. `compose down` also keeps volumes but removes this project's containers/networks.
**Do not add `-v`**: it deletes this project's stored data. Back up Docker volumes and `secrets/`
together as described in [Deployment](DEPLOYMENT.md#6-data-backups-and-restore).

After pulling code updates, stop this project, rerun setup, and start again. Setup retains model files
and existing keys. The pinned inference revision is recorded in `scripts/macos.py`.

## 7. Settings and troubleshooting

### Waiting for an answer

An active answer shows a spinner, elapsed time and a message such as **Reading your documents**,
**Thinking** or **Writing your answer**. Text appears as the model generates the final answer.
The elapsed time is time spent waiting, not an estimate of time remaining. A completed answer or an
error removes the loader; a failed or interrupted answer offers **Retry**.

For introductory questions, start with **Think**, **Deep search** and **Read all** unchecked.
Read all first makes separate reading calls over all selected sections, then writes the answer; a call
that reaches its length limit may be repeated once with more room. This can take several minutes
before the first answer text appears. Use it when coverage of every section matters. Deep search adds
tool-driven research and can also take several minutes. These modes still stream the final answer.

For speed comparisons, plug in the Mac and check its energy mode. Keep other memory-heavy work in
mind: native models, Docker's VM and other applications share the same unified memory. Container
memory limits alone do not bound all usage. Do not assume a visible loader means a particular token
rate or change other projects' settings to diagnose this app.

### Launcher settings

Mac overrides are shell variables. Set the same values in every tab that uses the launcher:

```sh
export MAC_MODELS_DIR="$HOME/local-rag-models"   # optional: model storage outside the checkout
export MAC_LLM_CTX=32768                       # total for two slots; keep the default initially
```

The helper fixes the model filenames/alias and passes the same context to the LLM and RAG service.
Changing `LLM_MODEL` in `.env` does not convert or replace weights. The smaller Mac prompt defaults are
in `compose.macos.yaml`; only increase context, batches, or worker memory after measuring headroom.
`MAC_WORKER_MEM_LIMIT` (default `4g`) and `MAC_RAG_MEM_LIMIT` (default `2g`) adjust the Docker caps;
they do not resize Docker Desktop's VM. The optional Python sandbox adds its own memory budget.

| Symptom | What to do |
|---|---|
| “Cannot connect to Docker” / socket error | Open Docker Desktop and wait for the engine, then rerun `doctor`. |
| Image-registry lookup times out during a build | Check internet/proxy access to Docker Hub and `ghcr.io`; retry setup when the connection works. Completed models and keys are reused. |
| “unknown tag !override” | Update Docker Desktop/Compose; the Mac override requires Compose 2.24.4+. |
| Model/proxy stays unhealthy | Keep both model tabs open; inspect their errors and the two `curl` checks above. Docker Desktop must resolve `host.docker.internal`. |
| Port already in use | Check which application owns 8000, 18080 or 18083; stop that application's service yourself. The launcher refuses to replace it. |
| CUDA/NVIDIA error on Mac | Use `python3 scripts/macos.py compose ...`; plain Compose is the NVIDIA path. |
| MPS out of memory / high memory pressure | Stop the project; close other model workloads. Restart at the defaults. Inspect Activity Monitor; do not disable the MPS watermark. |
| Worker restarts / document fails | Inspect its logs below. A large PDF can exceed the 4 GB parser cap; raise it only if VM and Mac have room. |
| Chat works but image questions fail | Confirm the matching projector and image embedder are present and inspect both native model logs. |

```sh
python3 scripts/macos.py compose logs --tail 100 ingest-worker rag llm models
python3 scripts/macos.py compose stats --no-stream
```

Native inference logs are in its two Terminal tabs. Native model memory does not appear in `compose stats`.
Keep one `ingest-api` instance: upload placement locks are local to its process. Worker claims serialize
versions of the same filename, but this is a workstation deployment, not a distributed worker system.

## 8. Locality and validation

Setup needs internet access. Runtime model loading is offline and documents go to local processes.
Unlike the NVIDIA deployment, the native model processes are outside Docker's network and filesystem
isolation; the two proxies join a bridge that can reach the host. Docker's “no network egress” guarantee
does not extend to those native processes. Use the NVIDIA deployment or an OS-enforced network policy
if enforced isolation of inference is a requirement.

Primary references: [llama.cpp Metal build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[Docker Desktop networking](https://docs.docker.com/desktop/features/networking/),
[Docker GPU support](https://docs.docker.com/desktop/features/gpu/),
[the GGUF files](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main).

See [the review report](REVIEW.md) for the checks actually performed, test counts, and remaining risks.
