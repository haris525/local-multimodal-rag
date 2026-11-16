# Deployment

Apple Silicon Mac installation and operation use [MACOS.md](MACOS.md). The NVIDIA deployment below
has different inference isolation and memory requirements. See [REVIEW.md](REVIEW.md) for current
concurrency fixes and remaining index-publication limitations before enabling concurrent reindex/query work.

This guide is for whoever runs the app: on one workstation for yourself, or on a Linux server that
other people reach over the network. First-time setup on your own machine is in
[GETTING_STARTED.md](GETTING_STARTED.md). How the system works is in [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
(plain words) and [ARCHITECTURE.md](ARCHITECTURE.md) (technical design). Changes that affect stored
data are listed in [CHANGELOG.md](../CHANGELOG.md).

> **Accounts.** Everyone signs in. An admin creates the accounts (section 9), and each user sees only
> their own documents, chats, memories and audit trail. Admins manage accounts but cannot read anyone's
> content. Whoever has Docker or root access on the host can read every volume and reset any password
> (section 7.2), so the host itself must be trusted.

Commands are for bash on Linux unless marked PowerShell. In Windows PowerShell 5.1, type `curl.exe`,
not `curl` (which is an alias for `Invoke-WebRequest`).

## 1. Deployment modes at a glance

The same `compose.yaml` runs in both modes. Only the host setup and the way people reach the port
differ.

| | Workstation (local) | Linux server (used over the network) |
|---|---|---|
| Host | Windows 11 + Docker Desktop (WSL2), or a Linux desktop | headless Linux + Docker Engine + NVIDIA Container Toolkit |
| Users | you, as the first admin; more accounts if others share the PC | several people on their own computers, each with an account and a private library (section 9) |
| Access | http://localhost:8000 | `https://<name>` through a TLS reverse proxy on the host (section 5.3), or an SSH tunnel to the server's 127.0.0.1:8000 (section 5.1) |
| Starts | when you sign in and Docker Desktop starts | at boot, with the Docker service |
| GPU memory for the stack | the card minus the desktop (about 1.6 GiB on Windows) | the whole card |
| Concurrency | 2 LLM slots | 2 LLM slots shared by everyone; ingestion can hold one (section 8.5) |
| Backups | your choice | scheduled (section 6) |
| Steps | GETTING_STARTED, then section 3 | section 4 and section 5 |

## 2. Requirements

### 2.1 Hardware

| | Requirement | Notes |
|---|---|---|
| GPU | one NVIDIA GPU with 32 GB | all four models stay loaded together (section 2.2) |
| RAM | 32 GB, 64 GB recommended | `ingest-worker` is capped at 12 GB (`WORKER_MEM_LIMIT`), `sql` at 2 GB. Docker Desktop needs at least 24 GB ([GETTING_STARTED section 2](GETTING_STARTED.md#2-check-that-docker-can-see-your-gpu)) |
| CPU | 8 cores or more recommended | PDF parsing uses 8 threads (`OMP_NUM_THREADS=8` in the ingest image, `INGEST_PARSE_THREADS=8`); `sql` is capped at 2 CPUs |
| Disk | 80 GB free; 100 GB or more on a server, plus your documents | about 50 GB is used before the first document (below) |

Disk use, measured with Docker Desktop on Windows:

| What | Size | Where |
|---|---|---|
| Models | 25.3 GB: GGUF 17.66 GB, mmproj 0.93 GB, Qwen3-VL-Embedding-2B 4.0 GB, Qwen3-Embedding-0.6B 1.2 GB, Qwen3-Reranker-0.6B 1.2 GB | volume `local-rag_models` |
| Images | about 23 GB: `local-rag/models` 9.6 GB, llama.cpp 7.0 GB, `local-rag/ingest` 4.5 GB, the rest under 2 GB | Docker image store |
| Build cache | grows with every rebuild | Docker; prune it (section 8.4) |
| Your data | originals, figure crops, DuckDB tables, chunk text, vectors, chats, audit log | data volumes (section 6) |

So about 50 GB is used before the first document.

### 2.2 GPU memory budget

Measured with `nvidia-smi` on a 32 GB GPU (32,607 MiB) with Docker Desktop on Windows 11
([ARCHITECTURE section 3](ARCHITECTURE.md#3-models-and-gpu-budget)):

| State | Whole card | Added |
|---|---|---|
| Windows desktop only (none on a headless server) | 1.6 GiB | |
| + `llm`: Qwen3.8-27B UD-Q4_K_XL (17.66 GB) + mmproj-BF16 (0.93 GB) + KV cache (2.1 GiB) + buffers | 20.6 GiB | 19.0 GiB |
| + `models`: Qwen3-Embedding-0.6B, Qwen3-VL-Embedding-2B, Qwen3-Reranker-0.6B (BF16) | 28.2 GiB | 7.6 GiB |
| Whole stack, other desktop apps open | 29,427 of 32,607 MiB (28.7 of 31.8 GiB) | |

The stack itself needs about 27 GiB. Peaks stay small: an image request adds about 0.2 GiB, and
`models` caps images at 1024 px and batches at 2 images or 32 texts. A 24 GB card does not fit the
defaults, and the repository has no smaller configuration. The architecture notes that the Q6_K file
(22.4 GB) does not leave room for the embedders; `fetch-models` downloads only UD-Q4_K_XL.

**KV cache.** With the q8_0 cache it costs 34 KiB per token. `LLM_CTX=65536` (2 slots of 32,768
tokens) takes 2.1 GiB; every 32,768 tokens more or less changes that by about 1.06 GiB.

### 2.3 Settings that change the deployment

Settings live in `.env` next to `compose.yaml` (copy `.env.example`). Every one has a default.

| Variable | Default | Changes |
|---|---|---|
| `GATEWAY_PORT` | `8000` | the host port. The gateway's Origin check accepts `http://localhost:<port>`, `http://127.0.0.1:<port>` and `PUBLIC_ORIGIN` (section 5) |
| `GATEWAY_BIND` | `127.0.0.1` | the host address the port is published on. Keep it: a TLS proxy on the same host reaches the gateway there (section 5.3) |
| `PUBLIC_ORIGIN` | `http://localhost:${GATEWAY_PORT}` | the address users open, with no path or trailing slash. Its host name joins the gateway's Host allow-list and the origin its Origin allow-list. `https://...` gives the `__Host-rag_session` cookie with `Secure`. Plain `http` for a host other than `localhost` or `127.0.0.1` stops `auth` at startup unless `AUTH_ALLOW_INSECURE_HTTP=true`. Set this, not `AUTH_PUBLIC_ORIGIN`, which compose derives from it |
| `TRUSTED_PROXY_CIDR` | `127.0.0.1/32` (trusts no proxy) | the address whose `X-Forwarded-For` the gateway believes. With a TLS proxy on the host: `172.31.240.1/32` (section 5.3) |
| `FRONTEND_SUBNET` | `172.31.240.0/24` | subnet of the `frontend` network. Connections to the published port come from its first address. Change it only if it clashes with a network on the host |
| `AUTH_*` | see `services/auth/src/auth/config.py` | sign-in: `AUTH_IDLE_TIMEOUT_MIN` (720), `AUTH_ABSOLUTE_TIMEOUT_DAYS` (7), `AUTH_LOCKOUT_THRESHOLD` (5), `AUTH_LOCKOUT_BASE_MIN` (15), `AUTH_LOCKOUT_MAX_HOURS` (24), `AUTH_IP_FAILURE_LIMIT` (20), `AUTH_MIN_PASSWORD_LENGTH` (12), `AUTH_TOKEN_MAX_DAYS` (365), `AUTH_TRUSTED_PROXIES` (`gateway`: whose `X-Real-IP` is believed) (section 9.3) |
| `MAX_UPLOAD_MB` | `200` | gateway request size limit (`client_max_body_size`, for the whole request) and the per-file limit in `ingest-api` (`INGEST_MAX_UPLOAD_MB`) |
| `LLM_GGUF` | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | the weights file llama-server loads from `/models/gguf/`. It must exist in the models volume. compose also passes it to `ingest-worker` (`INGEST_VISION_MODEL_FILE`), where its name, with the quantization, is part of every LLM cache key (section 4.17) |
| `LLM_MMPROJ` | `mmproj-BF16.gguf` | the vision projector; image reading needs it. compose passes it to `ingest-worker` too (`INGEST_VISION_MMPROJ`, part of the LLM cache key) |
| `LLM_CTX` | `65536` | llama-server `--ctx-size`, split over 2 slots, so each request gets `LLM_CTX / 2` tokens. Changes GPU memory as in section 2.2 |
| `LLM_MODEL` | `qwen3.8-27b` | only the name llama-server answers to; compose passes the same value to `ingest-worker`, `rag` and `eval`. It does not change the model |
| `WORKER_MEM_LIMIT` | `12g` | memory cap of `ingest-worker`, which parses untrusted files |
| `MODELS_SEED_DIR` | `./scripts` | host folder mounted at `/seed` for `fetch-models --from-dir /seed` (section 4.3) |
| `INGEST_*` | see `services/ingest/src/ingest/config.py` | chunk sizes, figure and row caps, vision endpoint, figure description budgets and chart extraction, retry and timeouts (section 4.12), the LLM cache (section 4.17) |
| `RAG_*` | see `services/rag/src/rag/config.py` | retrieval depth, fusion weights, thresholds, prompt budgets, every model-call budget, timeouts, retry policy, deep-search budgets, sampling (section 4.12, [ARCHITECTURE section 7.8](ARCHITECTURE.md#78-limits-retries-and-notices)) |
| `SQL_*` | see `services/sql/src/sql_sandbox/api.py` and `engine.py` | `SQL_TIMEOUT_S` (20), `SQL_MAX_CONCURRENT` (4), `SQL_MEMORY_LIMIT` (400MB), `SQL_MAX_CELL_CHARS` (2,000), `SQL_MAX_RESPONSE_CHARS` (200,000), `SQL_MAX_FILE_STREAMS` (2: tables files sent to rag for Python at once) |
| `SANDBOX_*` | see `compose.yaml` and `services/sandbox/src/sandbox/config.py` | the Python sandbox's container limits and per-run caps (section 4.16). Compose passes them to `sandbox` one by one; it gets no `.env` |

- **Fixed in `compose.yaml`, not settings:** `--parallel 2`, the q8_0 KV cache, `--image-max-tokens 1024`,
  `--n-gpu-layers 999` and flash attention. Changing them means editing `compose.yaml`. A change there, or
  to the `llm` image, can change the model's replies: also set a new `INGEST_LLM_CACHE_SALT`, so ingest does
  not reuse replies cached before it (section 4.17).
- **Lowering `LLM_CTX`.** `rag` budgets each prompt for a 32,768-token slot: up to 16,000 tokens of
  sources, 1,500 of SQL results, 2,048 of figures, plus 8,192 output tokens in *Think* mode
  ([ARCHITECTURE section 7.5](ARCHITECTURE.md#75-prompt-budget-and-neighbour-expansion)). With a smaller
  slot, also lower `RAG_SOURCE_BUDGET` and `RAG_THINK_TOKENS` so that sources, figures, the question
  and the answer still fit. For example, `LLM_CTX=32768` gives 16,384 tokens per slot, which cannot
  hold 16,000 tokens of sources plus an 8,192-token *Think* answer, so long requests would fail.
- **Which containers read `.env`.** Compose reads it for the variables above and passes it whole to
  every Python service (`ingest-api`, `ingest-worker`, `rag`, `models`, `threads`, `audit`, `sql`,
  `auth`), so `INGEST_*`, `RAG_*`, `MODELS_*`, `THREADS_*`, `AUDIT_*`, `SQL_*` and `AUTH_*` settings in
  `.env` take effect. `gateway`, `llm` and `qdrant` take only the compose variables above, and `sandbox`
  only the `SANDBOX_*` variables compose names for it: it runs model-written code, so nothing else of `.env`
  reaches it.
- **No secrets in `.env`.** Every service receives it. Signing keys and the Qdrant `api_key` live in
  `secrets/` (written by `scripts/init_secrets.py`, mounted as Docker secrets); API tokens for scripts
  go in the shell (`RAG_TOKEN`, `EVAL_TOKEN`).
- **Applying changes.** `docker compose up -d` recreates the services whose settings changed. A
  change to an `LLM_*` setting reloads `llm`, which takes a few minutes.

### 2.4 Check that it fits

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv    # before starting
docker compose up -d
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5    # watch; Ctrl+C to stop
```

Watch while `llm` and `models` become healthy, then while a PDF with figures is ingested and a *Think*
question is asked at the same time. A 32 GB GPU on Windows keeps about 3 GiB free when idle. If
`docker compose logs llm` or `docker compose logs models` shows a CUDA out-of-memory error, close other
GPU programs or lower `LLM_CTX`. On Linux,
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` shows each process. On
Windows the per-process column says N/A, so use the whole-card numbers.

### 2.5 Software

| | Windows workstation | Linux server |
|---|---|---|
| OS | Windows 11 | a 64-bit distribution supported by Docker and the NVIDIA Container Toolkit. The commands below are written for Ubuntu; no Linux distribution has been tested yet |
| NVIDIA driver | a current Windows driver. WSL2 uses it; do not install a Linux driver inside WSL | new enough for CUDA 13.2, because the `models` image ships PyTorch built for CUDA 13.2 (`llm` uses CUDA 12.8). NVIDIA's CUDA release notes list the minimum driver version; no Linux driver has been tested with this app yet |
| Docker | Docker Desktop 4.29 or newer with the WSL2 backend (GPU support is built in) | Docker Engine 26 or newer with the Compose plugin, plus the NVIDIA Container Toolkit. `sql` mounts one folder of a volume (`volume.subpath`), which older engines refuse |
| Compose | v2.24 or newer: `compose.yaml` uses `env_file` with `required: false` | same |
| Other | Git; [uv](https://docs.astral.sh/uv/) for `scripts/init_secrets.py` and the smoke tests | Git, curl, python3 (for the reindex loop in section 6.4), uv |

## 3. Local setup (workstation)

Follow [GETTING_STARTED.md](GETTING_STARTED.md). It covers the GPU check, WSL memory, cloning, the
model download, the first start and troubleshooting. Some things it does not cover:

- **Ports.** Only the gateway publishes a port: `127.0.0.1:${GATEWAY_PORT}` (8000). The backend
  ports (8001 to 8007, 8080, 6333) exist only on the internal Docker networks. Reach them with
  `docker compose exec` (section 8.1). Open the app as `http://localhost:<port>` or
  `http://127.0.0.1:<port>`. Any other host name, such as the machine's name or a LAN address, gets no
  response at all unless `PUBLIC_ORIGIN` names it. The right host name on another port loads the page,
  but uploads and chat get `403` (section 5).
- **Settings file.** `.env` is optional and ignored by Git. After editing it, run `docker compose up -d`.
- **Secrets folder.** `scripts/init_secrets.py` writes the signing keys and the Qdrant `api_key` into
  `secrets/`, which Git and Docker ignore. Back it up with the volumes (section 6.2).
- **Internet use.** Only the image build (base images, Python packages, Docling weights) and the
  one-time model download (`docker compose --profile setup run --rm fetch-models`, about 25 GB) use
  the internet. At runtime the `backend` and `compute` networks have no route out.
- **Health.** `docker compose ps` should show every service `healthy`, and
  `docker compose run --rm auth auth-admin check` should end with `multi-user: ready`. Signed in, the
  dot at the top right of the page turns green.
- **Start with Windows.** In Docker Desktop, open Settings > General and turn on *Start Docker Desktop
  when you sign in*. Every service has `restart: unless-stopped`, so the stack comes back when Docker
  Desktop starts. Services you stopped with `docker compose stop` stay stopped until
  `docker compose start`. Docker Desktop runs only while a user is signed in, so a Windows PC is not a
  headless server.

## 4. Server setup (Linux)

### 4.1 Prerequisites

1. **NVIDIA driver.** Install it from your distribution (Ubuntu: `sudo ubuntu-drivers install`),
   reboot, and check `nvidia-smi`.
2. **Docker Engine and Compose.** Install them from Docker's repository
   (https://docs.docker.com/engine/install/), then:

   ```bash
   sudo systemctl enable --now docker
   docker compose version            # v2.24 or newer
   ```

3. **NVIDIA Container Toolkit.** Add NVIDIA's repository as its install guide describes
   (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then:

   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi   # must list the GPU
   ```

4. **Operator account.** Add only administrators to the `docker` group
   (`sudo usermod -aG docker <user>`). Membership is equivalent to root and gives read access to every
   volume.
5. **Log rotation.** `compose.yaml` already rotates every service's logs (json-file, 5 files of 10 MB
   each). Set a daemon-wide default in `/etc/docker/daemon.json` only for other containers on the host,
   keeping the `runtimes` entry that `nvidia-ctk` added.

### 4.2 Code and settings

```bash
sudo mkdir -p /opt/local-rag && sudo chown "$USER" /opt/local-rag
git clone <repo-url> /opt/local-rag
cd /opt/local-rag
cp .env.example .env && chmod 600 .env
uv run --with cryptography scripts/init_secrets.py    # signing keys and the Qdrant api_key, in secrets/
```

Replace `<repo-url>` with the repository's address. Review `.env`: the
defaults suit a 32 GB card, and `GATEWAY_PORT` only needs changing if 8000 is taken on the server.
Set `PUBLIC_ORIGIN` to the address users will open (section 5.3).

`init_secrets.py` refuses to overwrite existing files. On Linux it makes `secrets/` readable by its
owner only (0700); the files inside are 0644, because the containers run as other users.

### 4.3 Models

All models live in the `local-rag_models` volume. Its layout is `gguf/` (the two GGUF files) and
`hf/<name>/` (the three Hugging Face models). `fetch-models` (profile `setup`) is the only container
that fills it.

| The server has | Do this |
|---|---|
| Internet access at setup | `docker compose --profile setup run --rm fetch-models` |
| A copy of the models, no Hugging Face access | seed folder, below |
| No internet at all | air-gapped steps, below |

**Seed folder.** This is the repository's offline path. To export the models from a machine that
already has them (PowerShell on a Windows workstation; replace `<models-folder>` with an empty host folder):

```powershell
docker run --rm -v local-rag_models:/models:ro -v "<models-folder>:/out" python:3.12-slim-bookworm cp -a /models/gguf /models/hf /out/
```

Copy that folder to the server (for example to `/srv/models-seed` with `rsync` or a disk). Then:

```bash
MODELS_SEED_DIR=/srv/models-seed docker compose --profile setup run --rm fetch-models --from-dir /seed
```

It copies `gguf/` and `hf/` into the volume and skips files that are already there with the same
size. With `--from-dir` the container skips the `huggingface_hub` install, so it needs neither PyPI nor
Hugging Face, only the `python:3.12-slim-bookworm` image (pull it once, or load it as in the
air-gapped steps).

**Air-gapped (not yet tested).** These steps have not been run end to end yet. Build on a connected
machine and move the images, then copy the models with a plain container:

```bash
# connected machine, in the repository
docker compose build
docker save -o local-rag-images.tar local-rag/models:dev local-rag/ingest:dev local-rag/rag:dev \
  local-rag/sql:dev local-rag/threads:dev local-rag/audit:dev local-rag/gateway:dev \
  qdrant/qdrant:v1.20.0 ghcr.io/ggml-org/llama.cpp:server-cuda python:3.12-slim-bookworm

# server, in /opt/local-rag
docker load -i local-rag-images.tar
docker compose up --no-start --no-build --pull never      # creates the volumes
docker run --rm -v local-rag_models:/models -v /srv/models-seed:/seed:ro python:3.12-slim-bookworm \
  cp -a /seed/gguf /seed/hf /models/
docker compose up -d --no-build --pull never
```

`compose.yaml` pins `qdrant` and `llm` by digest (`image: ...@sha256:...`), and `--pull never` stops
with an error if the loaded images do not match those references. If that happens, save those two
images under the exact references written in `compose.yaml` instead of their tags.

### 4.4 Build and start

```bash
cd /opt/local-rag
docker compose up -d --build
docker compose ps
docker compose run --rm auth auth-admin create-admin --username <you>   # the first admin (section 9.2)
```

The first build takes 10 to 20 minutes. `llm` and `models` need a few minutes to load onto the GPU;
their health checks allow 300 seconds. Then sign in at `PUBLIC_ORIGIN` and create the other accounts
(section 9.4).

### 4.5 Health checks

```bash
curl -s http://127.0.0.1:8000/healthz                    # ok
docker compose run --rm auth auth-admin check            # each service: 'multi_user': 1; last line: multi-user: ready
export RAG_TOKEN=rag_pat_...                             # an API token (section 9.5)
curl -s -H "Authorization: Bearer $RAG_TOKEN" http://127.0.0.1:8000/api/ready          # {"ready":true,"checks":{...}}
curl -s -H "Authorization: Bearer $RAG_TOKEN" http://127.0.0.1:8000/api/audit/verify   # {"ok":true,"events":...}
```

Without a token, every `/api` route except sign-in answers `401 {"detail":"sign in required"}`. For an
end-to-end test, run the smoke test from
[GETTING_STARTED section 7](GETTING_STARTED.md#7-check-that-it-works-optional) on the server, or
through a tunnel from a workstation (section 5.1), with `--url http://localhost:8000`, and the
isolation test (section 9.7).

### 4.6 Start on boot

Docker starts at boot (`systemctl enable docker`) and every service has `restart: unless-stopped`, so
nothing else is required. The one exception is services you stopped by hand, which stay stopped. If you
want the stack started at every boot regardless, add a systemd unit:

```ini
# /etc/systemd/system/local-rag.service
[Unit]
Description=Local multimodal RAG (docker compose)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/local-rag
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose stop
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now local-rag
```

`docker compose up -d` waits for each dependency to be healthy before starting the services that need
it, which can take several minutes on a cold GPU. `TimeoutStartSec` allows for that.

### 4.7 Logs

```bash
docker compose logs -f --tail 100 rag                 # one service
docker compose logs --since 1h | grep <request_id>    # one request across services
```

Every service writes one JSON line per request. The gateway sets `X-Request-ID`, and every Python
service logs it and passes it on.

### 4.8 Updates

```bash
cd /opt/local-rag
OLD=$(git rev-parse HEAD)
# 1. back up the data volumes (section 6.2; the backup also saves this commit in commit.txt),
#    then keep the current images for a rollback
for i in models ingest rag sql threads audit auth gateway; do docker tag local-rag/$i:dev local-rag/$i:previous; done
# 2. fetch, and see what changed
git pull --ff-only
git diff --name-only "$OLD" HEAD
git diff "$OLD" HEAD -- CHANGELOG.md     # read every "Data changes" line before going on
# 3. rebuild and recreate only the affected services, for example:
docker compose up -d --build --no-deps rag
```

If an entry in [CHANGELOG.md](../CHANGELOG.md) lists a data change, plan for its reindex or keep the
backup from step 1 ready for a rollback. An install from before accounts (no `auth` service, no
`secrets/` folder) follows section 4.10 instead.

| Changed path | Run |
|---|---|
| `services/rag/` | `docker compose up -d --build --no-deps rag` |
| `services/threads/` | `docker compose up -d --build --no-deps threads` |
| `services/ingest/` | `docker compose up -d --build --no-deps ingest-api ingest-worker` (one image, built by `ingest-api`; always recreate both) |
| `services/models/` | `docker compose up -d --build --no-deps models` |
| `services/sql/`, `services/audit/`, `services/auth/`, `services/gateway/` | the same with `sql`, `audit`, `auth` or `gateway` |
| `libs/common/` | every Python service: `ingest-api ingest-worker rag sql threads audit auth models` |
| `compose.yaml` or `.env` | `docker compose up -d` (recreates the services whose configuration changed) |
| `scripts/fetch_models.py` | run `fetch-models` again if the model list changed |
| `eval/` | `docker compose --profile eval build eval` |
| `docs/`, `README.md`, `CHANGELOG.md` | nothing |

- `--no-deps` stops Compose from recreating dependencies. Otherwise an update to `rag` could also
  restart `llm` and `models`, which take minutes to reload.
- The gateway looks up backend addresses again every 10 seconds, so it does not need a restart when a
  backend is recreated.
- To pick up security fixes in the base images, add `--pull` to the build:
  `docker compose build --pull <services>`, then `docker compose up -d --no-deps <services>`.

### 4.9 Rollback

```bash
cd /opt/local-rag
git checkout "$(cat /srv/backups/local-rag/<set>/commit.txt)"   # the commit saved by the backup in step 1
for i in models ingest rag sql threads audit auth gateway; do docker tag local-rag/$i:previous local-rag/$i:dev; done
docker compose up -d --no-deps --force-recreate rag      # the services you updated
```

The checkout leaves a detached HEAD; run `git checkout main` before the next update. If the update
changed stored data (CHANGELOG.md says so under "Data changes"), also restore the backup from step 1
(section 6.3). Services refuse to start when the Qdrant collection was built with different embedding
models. Going back to a version from before accounts always means restoring the backup taken in
section 4.10.

### 4.10 Upgrade an install from before accounts

Run these once, from the repository folder (the commands are the same in bash and PowerShell). They
take about as long as a backup and a rebuild. For the backup, follow section 6.2 but leave out
`auth-data` and `secrets/`, which do not exist yet, and its final `docker compose start`.

```bash
git pull                                           # or fetch the release
docker compose stop ingest-worker
# back up the volumes (section 6.2, as above)
uv run --with cryptography scripts/init_secrets.py
docker compose up -d --build --scale ingest-worker=0
docker compose run --rm auth auth-admin check                                   # each service: 'multi_user': 1; last line: multi-user: ready
docker compose run --rm auth auth-admin create-admin --username <you> --no-adopt-legacy
docker compose run --rm auth auth-admin migrate-legacy --to <you> --dry-run    # what would change
docker compose run --rm auth auth-admin migrate-legacy --to <you>
docker compose up -d                                                            # starts ingest-worker again
```

Then check:

1. Sign in at http://localhost:8000 (or `PUBLIC_ORIGIN`) as `<you>`.
2. *Audit log* reads "Hash chain verified: N of your events, unaltered.", where N counts the events
   from before the upgrade plus your new ones.
3. *Audit log* > *Whole log* verifies.
4. Optional, on an install whose library was empty or on a test copy: the isolation test (section
   9.7). It refuses to run while the admin's library holds data.

What the steps do:

- `docker compose up -d --build` upgrades the schemas of `registry.db`, `threads.db` and `audit.db`
  as the services start, adds the `auth` service and the `auth-data` volume, and recreates the
  `frontend` network with its fixed subnet. No `docker compose down` is needed.
- `migrate-legacy` gives everything created before accounts existed (documents, files, index entries,
  chats, memories and audit events) to that admin. Document ids and index point ids do not change. A
  second run changes nothing. While the worker runs (its heartbeat is less than 60 seconds old) and
  unadopted documents exist, it stops with `409 stop ingest-worker first`; `--force` skips that check.
- Data that is not adopted stays in place, visible to nobody.
- The audit events from before accounts can go to one account only, and they stay unencrypted (later
  events are encrypted per user, section 8.3).
- Evaluation golden sets belong to the user who owns their documents. A set from before the upgrade
  keeps working with a token of the admin who adopted the documents; other users generate their own
  ([GETTING_STARTED section 8](GETTING_STARTED.md#8-measure-retrieval-quality-optional)).

A fresh install skips the migration: `init_secrets.py`, `docker compose up -d --build`,
`auth-admin create-admin --username <you>` (it asks for the password twice, at least 12 characters),
then sign in.

### 4.11 Update: explicit state machines (registry v3, auth v4)

This update ([CHANGELOG](../CHANGELOG.md), [STATE_MACHINES.md](STATE_MACHINES.md)) upgrades two
databases when the services first start: `registry.db` from schema v2 to v3 (`ingest-api` or
`ingest-worker`, whichever starts first) and `auth.db` from v2 to v4 (`auth`). Each upgrade is one
transaction that rebuilds a table and keeps every row; sessions stay signed in. The earlier code cannot
work with the new schemas (its worker fails to claim a job, its account deletion fails a CHECK), so the
only way back is the backup.

Before you start, look for accounts that a deletion left unfinished:

```bash
docker compose run --rm auth auth-admin list-users     # accounts whose STATUS is deleting
```

The new `auth` finishes such deletions on its own a minute after it starts, acting as the operator,
because the earlier code did not record which admin asked. If one of them should not be purged, sort it
out before the update; a deleting account cannot be taken back.

```bash
cd /opt/local-rag
# 1. back up every volume (section 6.2), then keep the current images for a rollback
for i in models ingest rag sql threads audit auth gateway; do docker tag local-rag/$i:dev local-rag/$i:previous; done
# 2. update the code and build the images while the old services keep running
git pull --ff-only
docker compose build ingest-api auth rag gateway       # ingest-worker uses the ingest-api image
# 3. stop ingest completely: the old worker releases its running job, and no old ingest-api or worker
#    touches the registry after the new one has upgraded it
docker compose stop ingest-worker ingest-api
# 4. start the new services
docker compose up -d --no-deps ingest-api ingest-worker auth rag gateway
# 5. the GPU gate fix: models reloads its embedders and reranker, a few minutes; plan it for a quiet moment
docker compose build models && docker compose up -d --no-deps models
docker compose --profile eval build eval               # only if you use eval
```

Then check:

1. `docker compose logs auth ingest-api ingest-worker | grep -i error` shows nothing new.
2. `docker compose run --rm auth auth-admin list-users` lists every account with a STATE column.
3. Sign in; the document list and your chats are as before. Upload a small file, delete it while it
   shows *processing*: it leaves the list at once.
4. Admins: *Users* shows each account's state and only the buttons allowed for it.

`sql`, `threads` and `audit` changed only through `libs/common` (new definitions they do not use); rebuild
them at your next update. New settings, all optional: `INGEST_RUN_LEASE_S` (90: a running job's claim
without renewal), `INGEST_SWEEP_INTERVAL_S` (60: how often the worker takes back expired jobs and finishes
deletions), `INGEST_WORKER_ID` (the worker's name on the jobs it claims, the host name by default; a
restarted worker takes its own jobs back at once), `AUTH_ADMIN_DELETE_WAIT_S` (60: how long an account
deletion request waits before answering 202; keep it below the gateway's 120 s) and
`AUTH_DELETION_RESUME_DELAY_S` (60: when after start-up stopped deletions are resumed; then hourly).

**Rollback is a restore of the backup from step 1, nothing else.** Everything written after the update
(uploads, chats, memories, account changes, audit events) is lost. In this order:

```bash
cd /opt/local-rag
# 1. stop everything that writes
docker compose stop gateway rag ingest-worker ingest-api threads audit auth qdrant
# 2. restore all five volumes and secrets/ from the set of step 1 (section 6.3: its cp and its for loop)
# 3. go back to the code and the images of that backup, then start them
git checkout "$(cat /srv/backups/local-rag/<set>/commit.txt)"
for i in models ingest rag sql threads audit auth gateway; do docker tag local-rag/$i:previous local-rag/$i:dev; done
docker compose up -d --no-deps --force-recreate gateway rag ingest-worker ingest-api threads audit auth qdrant
# models only if you rebuilt it in step 5 (it stores nothing): docker compose up -d --no-deps --force-recreate models
```

### 4.12 Update: answer quality (sheet layout, chart descriptions, deep search, limits and notices)

This update ([CHANGELOG](../CHANGELOG.md)) changes how spreadsheets and images are read at ingestion, how
deep search researches, how every prompt is sized (in the model's own tokens), and how every limit and
failure is handled. No database schema changes: the new fields of the table schemas are optional, and
documents ingested before keep working as they are. **Existing documents benefit only after a reindex:**

- every spreadsheet (`.xlsx`, `.csv`): the new sheet layout (title, multi-row headers, footnotes, groups
  as a `Section` column, key-only rows kept, a second table on a sheet read as `<sheet> (2)`, or
  `<sheet> (table 2)` when a sheet has that name; two-cell preamble rows such as `Source, Census Bureau`
  kept out of the header; tables side by side read separately), real number
  and date types (ISO timestamps with a time zone as UTC `TIMESTAMP`, counts as numbers, postal codes as
  text), the column profiles, patterns and warnings that `describe_table`, the planner and
  `profile_column` show, the `<table>__raw` table for the `peek` tool, and no lost rows (a sheet with long
  header names could lose every data row from the search index);
- every image, and every PDF with figures: the two-pass chart descriptions at temperature 0;
- every document: the shorter summary (a plain paragraph instead of a Markdown report that could be cut).

SQL column names and types of reindexed sheets can change (for example `population_estimate_as_of_july_1_2025`
instead of a 250-character name, `begin_date` as a DATE instead of a number, USGS `time` as a TIMESTAMP
instead of text, a new `section` column in grouped tables): SQL saved outside the app against the old
names must be rewritten. Text, Markdown and PDFs without figures need no reindex for their search index,
but get the new summary only when reindexed.

`rag` now mounts one folder of the `models` volume read-only, `hf/Qwen3-Embedding-0.6B` (a volume subpath:
Docker Engine 26 or newer, as for `sql`), to read its `tokenizer.json` and count prompt tokens as the model
does, and gets `RAG_LLM_CTX` from `LLM_CTX`; both come with the new `compose.yaml`. The folder must exist in
the volume before `rag` starts (`fetch-models` puts it there; the embedding service needs it too). Without
the tokenizer file `rag` logs a warning and counts with a conservative estimate. The document list
(`GET /api/documents`) now sends a compact form of each table (no column statistics, full headers, layout
or notes); scripts that need the full schema read one document (`GET /api/documents/{id}`) instead. **Rebuild `eval`
in this update if you use it:** an `eval` image built before it cannot read the new `notice` chat event
and fails on the first answer that carries one.

```bash
cd /opt/local-rag
# 1. back up every volume (section 6.2) and keep the current images for a rollback
for i in ingest rag sql gateway; do docker tag local-rag/$i:dev local-rag/$i:previous; done
# 2. update the code, build, and recreate the services that changed
git pull --ff-only
docker compose build ingest-api rag sql gateway        # ingest-worker uses the ingest-api image
docker compose --profile eval build eval               # if you use eval: an older image rejects notice events
docker compose up -d --no-deps ingest-api ingest-worker rag sql gateway
# 3. reindex the affected documents of each account (the loop of section 6.4, filtered by type):
export RAG_TOKEN=rag_pat_...                            # a token of the account whose documents to reindex
H="Authorization: Bearer $RAG_TOKEN"
for id in $(curl -s -H "$H" http://127.0.0.1:8000/api/documents | python3 -c 'import json,sys; print(*[d["doc_id"] for d in json.load(sys.stdin) if d["doc_type"] in ("spreadsheet", "image") or d["n_images"]])'); do
  curl -s -o /dev/null -X POST -H "$H" http://127.0.0.1:8000/api/documents/$id/reindex
done
```

`auth`, `threads`, `audit` and `models` changed only through `libs/common` (new optional fields, the
`notice` chat event, the transport cause on service errors); rebuild them at your next update.

**Evaluation golden sets after the reindex.** A golden set labels each question's relevant passage by
chunk id and text hash (plus page, sheet and rows). Reindexing a spreadsheet or a figure-bearing document
rewrites its chunks (new row chunks with the `Section` column, new descriptions and summaries), so its
labels stop matching and its retrieval scores drop for that reason alone. Regenerate the set after the
reindex, as the account that owns the documents, review it, and only then compare runs:

```bash
EVAL_TOKEN=$RAG_TOKEN docker compose --profile eval run --rm eval generate --n 120   # writes eval/datasets/golden.jsonl
# open the file, fix or delete weak questions, then
EVAL_TOKEN=$RAG_TOKEN docker compose --profile eval run --rm eval retrieval
```

A run of the old set against the reindexed documents measures label drift, not quality; keep the old set
(rename it) if you want to compare the before and after runs of documents that were not reindexed. Figure descriptions take about twice as long
as before (two model calls per chart; one short extra call per other image), so a figure-heavy library
takes longer to reindex. Rollback: the images tagged `previous` and the backup; documents reindexed by the
new code also work with the old code, but their SQL column names stay the new ones until reindexed again.

New settings, all optional ([ARCHITECTURE section 7.8](ARCHITECTURE.md#78-limits-retries-and-notices) lists
every limit and what happens when it is hit):

| Setting | Default | What it sets |
|---|---|---|
| `RAG_ANSWER_TOKENS` | 4096 (was 2048) | the answer's output budget; a cut answer ends with a notice |
| `RAG_PLANNER_TOKENS`, `RAG_INTENT_TOKENS`, `RAG_REWRITE_TOKENS`, `RAG_ASSISTANT_TOKENS`, `RAG_HISTORY_ANSWER_TOKENS`, `RAG_MEMORY_TOKENS`, `RAG_COMPACTION_TOKENS` | 600, 600, 160, 800, 1000, 400, 800 | output budgets of the smaller model calls |
| `RAG_RETRY_TOKENS_FACTOR`, `RAG_RETRY_MAX_TOKENS` | 2.5, 4096 | a reply cut at its limit is asked again once with this much more room, at most this many tokens |
| `RAG_RETRY_BACKOFF_S` | 2 | wait before the one retry of a busy, failed or timed-out llm, models or sql call |
| `RAG_LLM_TIMEOUT_S`, `RAG_RESEARCH_LLM_TIMEOUT_S`, `RAG_MODELS_TIMEOUT_S`, `RAG_INGEST_TIMEOUT_S`, `RAG_SQL_TIMEOUT_S`, `RAG_THREADS_TIMEOUT_S`, `RAG_QDRANT_TIMEOUT_S` | 900, 600, 120, 60, 60, 30, 60 | how long rag waits for each service (`RAG_SQL_TIMEOUT_S` at least 2 x `SQL_TIMEOUT_S` + `SQL_QUEUE_WAIT_S`) |
| `RAG_TOKENIZER_FILE`, `RAG_TOKENIZER_FACTOR` | `/models/hf/Qwen3-Embedding-0.6B/tokenizer.json`, 1.05 | prompt tokens counted as the model counts them (the factor covers llama-server counting up to 3.4% more) |
| `RAG_LLM_CTX`, `RAG_LLM_PARALLEL`, `RAG_IMAGE_TOKENS`, `RAG_PROMPT_MARGIN_TOKENS` | `LLM_CTX` (65536), 2, 1024, 512 | one request's slot (ctx / parallel), an attached figure's cost, the room kept free in every prompt |
| `RAG_SOURCE_BUDGET`, `RAG_SQL_BUDGET` | 16000, 2500 (was 1500, estimated) | sources in an answer (less when the slot leaves less), one SQL result in a prompt |
| `RAG_SQL_MAX_ROWS`, `RAG_FILTER_MAX_ROWS`, `RAG_SCHEMA_CARD_TOKENS`, `RAG_LIST_TABLES_TOKENS` | 200, 5000, 6000, 3000 | SQL row caps and schema card sizes in tokens (replace `RAG_SCHEMA_CARD_CHARS` and `RAG_LIST_TABLES_CHARS`) |
| `RAG_AGENT_MAX_STEPS`, `RAG_AGENT_VERIFY_ROUNDS`, `RAG_AGENT_TOKENS`, `RAG_AGENT_TEMPERATURE`, `RAG_AGENT_PREVIEW_CHARS`, `RAG_AGENT_RECURSION_LIMIT`, `RAG_VERIFY_EVIDENCE_TOKENS`, `RAG_GREP_LIMIT` | 10 (was 8), 2, 2048 (was 1024), 0.3, 1000 (was 700), 80 (was 60), 16000, 10 (was 8) | deep search; the research prompt is bounded by the slot (`RAG_AGENT_CONTEXT_TOKENS` and `RAG_VERIFY_EVIDENCE_CHARS` are gone) |
| `RAG_PEEK_ROWS`, `RAG_PEEK_COLUMNS`, `RAG_PEEK_CELL_CHARS`, `RAG_PEEK_TOKENS`, `RAG_PROFILE_TOKENS` | 15, 30, 40, 4000, 1500 | the `peek` and `profile_column` tools |
| `INGEST_CHART_EXTRACTION`, `INGEST_CHART_EXTRACTION_TOKENS`, `INGEST_DESCRIBE_TEMPERATURE`, `INGEST_DESCRIBE_MAX_SIDE` | true, 2000, 0 (was 0.2), 1536 | figure descriptions |
| `INGEST_FIGURE_FAILURE_LIMIT`, `INGEST_FIGURE_DEADLINE_S` | 3, 3600 | figure descriptions stop after this many failures in a row, or after this long |
| `INGEST_SUMMARY_TOKENS`, `INGEST_SUMMARY_RETRY_TOKENS` | 1200 (was 350), 3000 | the document summary and its one retry at 2.5x the room |
| `INGEST_RETRY_MAX_TOKENS`, `INGEST_RETRY_BACKOFF_S` | 8000, 5 | the one retry of a cut description or summary, and of a busy, failed or timed-out vision or embedding call |
| `INGEST_VISION_TIMEOUT_S`, `INGEST_MODELS_TIMEOUT_S`, `INGEST_QDRANT_TIMEOUT_S` | 600, 600, 120 | how long the worker waits for each service |
| `INGEST_RAW_TABLE_MAX_ROWS` | 100000 | rows kept as text in `<table>__raw` |
| `INGEST_MAX_TABLES_PER_SHEET` | 50 | tables read from one sheet; the rest stay part of the last, with a warning |
| `RAG_SUMMARY_FOLD_MESSAGES`, `RAG_SUMMARY_FOLD_TOKENS` | 24, 12000 | the oldest turns one thread compaction folds at most |
| `SQL_TIMEOUT_S` | 20 (was 10) | a query that runs longer is interrupted (408, not retried) |
| `SQL_QUEUE_WAIT_S` | 5 | a query that finds every slot busy this long gets 503 (rag retries once) |
| `SQL_MAX_CELL_CHARS`, `SQL_MAX_RESPONSE_CHARS` | 2000, 200000 | cell and result caps (were fixed) |

Check: ask a question over a sheet whose answer needs SQL; the footer shows `SQL`. Ask one that cannot
be planned (a table without the needed column): a note under the answer says it came from text search.
In the document list, a failed document has a *Retry* button. With *Deep search*, select a spreadsheet
and a PDF that both state a figure: the steps include a search of the PDF, and the answer cites both.
`docker compose logs rag | grep tokenizer` shows nothing (a line there means rag counts with the
estimate).

### 4.13 Update: deep search on the deepagents harness, behind a switch

This update ([CHANGELOG](../CHANGELOG.md)) adds two implementations of *Deep search* on the deepagents
harness, tuned for the model's context (`rag/agent_deepagents.py`) and out of the box
(`rag/agent_deepagents_stock.py`; ARCHITECTURE section 7.6), next to the LangGraph one, so the three can be
compared on the same deployment. The LangGraph agent is the default; nothing changes for users until
`RAG_DEEP_AGENT` is set. **Data changes: none.** Only
the `rag` image changes (a new pinned dependency, `deepagents==0.5.9`, with `langchain`,
`langchain-anthropic` and `langchain-google-genai`, all pure Python): rebuild it with
`docker compose up -d --build --no-deps rag`. It stays offline: the harness's files live in each request's
memory, no model of another provider is ever built, and tracing stays off (`LANGSMITH_TRACING=false` in
the image).

| Setting | Default | What it sets |
|---|---|---|
| `RAG_DEEP_AGENT` | `langgraph` | which implementation runs deep search: `langgraph`, `deepagents`, `deepagents_stock` or an optional arm the rag registers; any other value stops rag at start-up, and the message lists the registered names |
| `RAG_DEEP_THINK`, `RAG_DEEP_THINK_TOKENS` | false, 4096 | thinking in the plan and verify steps (`deepagents_stock`: the first call), and its room (thought and reply) |
| `RAG_DEEP_EVICT_TOKENS`, `RAG_DEEP_PREVIEW_TOKENS` | 2000, 600 | deepagents: a research result larger than this is kept in the request's virtual files; the model sees its first and last lines within this many tokens and reads the rest with `read_file` |
| `RAG_DEEP_SUMMARY_TOKENS`, `RAG_DEEP_SUMMARY_KEEP_TOKENS`, `RAG_DEEP_SUMMARY_INPUT_TOKENS` | 20000, 6000, 12000 | deepagents: the research prompt is summarized past this size, keeping the newest messages within this many tokens, reading at most this much of the older ones |
| `RAG_DEEP_SUBAGENTS` | false | deepagents: offer `task`, a researcher sub-agent with the same tools and budget per sub-question (fan-out, section 4.14, offers it too) |
| `RAG_DEEP_HARNESS_CALLS` | 12 | deepagents and deepagents_stock: calls of the harness's own tools (`write_todos`, the file tools, `task`) per question (the research tools keep `RAG_AGENT_MAX_STEPS`) |

`deepagents_stock` takes none of the other `RAG_DEEP_*` settings: it runs the harness's defaults (its
prompts, all its file tools, the general-purpose sub-agent, its summarization and eviction thresholds), with
only the budgets and the slot guard of the others around it.

For a benchmark, leave the settings alone and send each request with `"agent": "langgraph"`,
`"agent": "deepagents"` or `"agent": "deepagents_stock"` and `"deep_think": true` or `false` in the body of
`POST /api/chat` (with `"mode": "deep"`): the six arms run on one deployment, one model and the same
documents. A wrong value is
a 422. The thread and the audit record of each answer keep `agent` and `deep_think`.

Check: set `RAG_DEEP_AGENT=deepagents`, recreate `rag`, and ask a deep question over a spreadsheet: the steps
start with `write_todos` (the plan), and `verify` comes after the research. Set it back to `langgraph` (or
remove it) to return to the LangGraph agent.

### 4.14 Update: fan-out for Deep search, and a research budget per request

This update ([CHANGELOG](../CHANGELOG.md); ARCHITECTURE section 7.6, "Fan-out") lets a deep-search arm that
supports it hand sub-questions to researchers, each with research calls reserved for it before it starts, so a
lead that searches first can no longer leave its researchers nothing. It is off by default and changes nothing
until it is turned on; the total research budget stays `RAG_AGENT_MAX_STEPS`. A request may also set its own
research budget, the same way for every arm, for a fair comparison at a larger budget. **Data changes: none.**
Only the `rag` image changes: rebuild it with `docker compose up -d --build --no-deps rag`. eval runs from the
repository with uv and needs no rebuild.

| Setting | Default | What it sets |
|---|---|---|
| `RAG_DEEP_FANOUT` | false | fan-out for every deep search whose arm supports it (deepagents, and an optional arm that declares it); a request's `fanout` overrides it. With an arm that does not support it as `RAG_DEEP_AGENT` (langgraph, deepagents_stock), rag stops at start-up and names the arms that do |
| `RAG_DEEP_TASK_CALLS` | 4 | research calls one `task` reserves for its researcher, taken from the question's budget before it starts; the calls it does not use go back. Scaled with a request's `research_calls` (8 of 20) |
| `RAG_DEEP_MAX_TASKS` | 2 | researchers per question; a further `task` is refused with a message to the lead |
| `RAG_DEEP_TASK_CONCURRENCY` | 2 | researchers of one request running at once (llama-server's slot count); further `task` calls of the same reply wait |
| `RAG_DEEP_TASK_RETURN_TOKENS` | 800 | a researcher's report reaches the lead cut at this many Qwen tokens, with a note (notice `research_task_cut`); its sources all stay for the answer |
| `RAG_DEEP_TASK_TIMEOUT_S` | 300 | one researcher's time limit; past it the lead reads an error, the user gets the notice `research_task_failed`, and the sources it found are kept |
| `RAG_AGENT_MAX_STEPS_LIMIT` | 30 | the largest `research_calls` a chat request may ask for (1 to this; anything else is a 422) |

Two optional fields of `POST /api/chat` (deep mode): `"fanout": true` or `false` (a 422 for an arm without
fan-out), and `"research_calls": N`, which replaces `RAG_AGENT_MAX_STEPS` for that request whichever arm
runs. The thread and the audit record of each answer keep `fanout` and `research_calls` next to `agent` and
`deep_think`. For the benchmark, leave the settings alone and use the variant suffixes `+fanout` and
`+calls<N>` (GETTING_STARTED section 8).

No limiter across users was added: llama-server already queues model calls first come, first served, so
another user's call waits for at most about one model call per busy slot while researchers run.

Check: ask a deep question over two documents with `"agent": "deepagents", "fanout": true`: the steps show two
`task` steps, each researcher's own steps between them, and `done.timings` has `tasks` 2. With
`"agent": "langgraph", "fanout": true` the answer is a 422 that names the arms with fan-out.

### 4.15 Update: text after figures and tables (index version 2)

This update ([CHANGELOG](../CHANGELOG.md)) fixes a chunking bug present since the first version: in a
section that had a figure or a table, every paragraph after the first figure or table was left out of the
search index (PDFs and other documents with figures or tables; plain text, Markdown and spreadsheets were
not affected). **Data changes: every affected document is reindexed, on its own.** The registry moves to
schema v4 when `ingest-api` or `ingest-worker` first starts (one added column, every row kept), and at its
start the new worker re-queues every ready document of every account that an older version indexed, and logs
`re-queued documents an older index version built` with the count. Nobody needs a user's token.

```bash
cd /opt/local-rag
docker tag local-rag/ingest:dev local-rag/ingest:previous
git pull --ff-only
docker compose build ingest-api            # ingest-worker runs the same image
docker compose stop ingest-worker ingest-api
docker compose up -d --no-deps ingest-api ingest-worker
docker compose logs ingest-worker | grep "re-queued"
```

While the queue drains, documents stay searchable on their old index (`n_chunks > 0`), and each one switches
to the new index when its reindex finishes. Figure descriptions are generated again, so the reindex takes
about as long as the first ingest of the same files and shares the model with chat. Rollback: the
`previous` image works with schema v4 (it ignores the new column). Documents it reindexes keep the version
the new code wrote, so reindex them by hand after rolling forward again.

### 4.16 Update: Python for Deep search, in a locked-down sandbox

This update ([CHANGELOG](../CHANGELOG.md); [ARCHITECTURE section 7.10](ARCHITECTURE.md#710-python-in-deep-search-run_python-the-sandbox-service))
gives every Deep search arm a `run_python` tool, run by a new `sandbox` service, so research can compute what one
SQL query cannot. **Nothing changes until an operator turns it on**: the `sandbox` container starts only with the
compose profile `sandbox`, rag uses it only with `RAG_DEEP_PYTHON=request` or `on` and `RAG_SANDBOX_URL` set, and
with the default `RAG_DEEP_PYTHON=off` a request's `"python": true` is a 422. **Data changes: none.** The sandbox
stores nothing: its scratch space is tmpfs, emptied after every run.

What changes: a new image (`local-rag/sandbox`, an estimated 0.8 GB, not measured yet: Python 3.12 with pandas,
numpy, polars, pyarrow, duckdb, scipy, statsmodels, openpyxl and matplotlib, pinned by
`services/sandbox/uv.lock`); a new internal network `sandbox` with a fixed subnet (`SANDBOX_SUBNET`,
172.31.241.0/24), which holds only `rag` and `sandbox`; `rag` joined to it, refusing every request that comes from
it, with a memory limit (`RAG_MEM_LIMIT`, 6g) and a 5-second cache on `/readyz`; and a new route on `sql`
(`GET /v1/tables/{doc_id}`: the caller's DuckDB file of a document, for rag only, with the same owner check as
queries, at most `SQL_MAX_FILE_STREAMS` at once). `rag` does not wait for `sandbox` at start: a sandbox that is
down makes `run_python` a tool error with a notice, nothing else.

```bash
cd /opt/local-rag
docker tag local-rag/rag:dev local-rag/rag:previous && docker tag local-rag/sql:dev local-rag/sql:previous
git pull --ff-only
docker build --target test -f services/sandbox/Dockerfile .   # the sandbox's tests on Linux, unprivileged
docker compose build rag sql && docker compose --profile sandbox build sandbox
docker compose up -d --no-deps rag sql                        # creates the sandbox network; Python stays off
# Python for this deployment of rag only (the shell wins over .env for compose's interpolation):
COMPOSE_PROFILES=sandbox RAG_DEEP_PYTHON=request RAG_SANDBOX_URL=http://sandbox:8008 \
  docker compose up -d --no-deps sandbox rag
python3 scripts/check_sandbox_isolation.py                    # must end with ALL PASSED
```

compose passes `RAG_DEEP_PYTHON` (default `off`) and `RAG_SANDBOX_URL` (default empty) to rag from the shell or
from `.env`. Set in the shell, they last only for that `docker compose up`: a later plain `docker compose up` of rag
(after an update, for example) turns Python off again unless `.env` sets them; for a lasting setting put
`COMPOSE_PROFILES=sandbox`, `RAG_DEEP_PYTHON` and `RAG_SANDBOX_URL` in `.env`. Turning it off: `docker compose up -d
--no-deps rag` without them, and `docker compose stop sandbox`.

`RAG_DEEP_PYTHON=request` is what the benchmark needs (`bench --variants deepagents,deepagents+python`,
GETTING_STARTED section 8): only requests that send `"python": true` get the tool. **Any signed-in account can send
`"python": true`**, so `request` is meant for a benchmark window or a deployment whose users are all trusted, not as
a standing setting for everyone; `on` offers it to every deep search (a request may still send `"python": false`)
and is for a deployment that gives Python to all users. Without `docker build --target test` succeeding and the
isolation check passing, leave it `off`.

| Setting (rag) | Default | What it sets |
|---|---|---|
| `RAG_DEEP_PYTHON` | `off` | `off`: never offered, a request's `"python": true` is a 422. `request`: offered only to a request with `"python": true`. `on`: offered to every deep search unless the request sends `"python": false`. Not `off`: `/api/ready` also checks the sandbox, and `RAG_SANDBOX_URL` must be set (rag refuses to start otherwise) |
| `RAG_SANDBOX_URL` | empty | where rag sends runs: `http://sandbox:8008` with the profile `sandbox` |
| `RAG_SANDBOX_SUBNET` | `SANDBOX_SUBNET` (compose) | every request from this network is refused with 403 before rag's access log, identity check or routes (health checks included); the refusals are logged once a minute with their count |
| `RAG_SANDBOX_TIMEOUT_S` | 120 | one run's time limit; past it the run and every process it started are killed and the model reads "timed out after N s". At most `SANDBOX_MAX_TIMEOUT_S` (a higher value makes every run a refused-run error) |
| `RAG_SANDBOX_REQUEST_TIMEOUT_S` | 600 | the whole call: the wait for the user's earlier run and for the run before it, the transfer and the run. Keep it above `SANDBOX_QUEUE_WAIT_S` + `RAG_SANDBOX_TIMEOUT_S` |
| `RAG_SANDBOX_INPUT_MB` | 256 | the selected documents' data sent with each run, together. More is a tool error naming the document and the sizes (notice `python_input_too_large`), never a cut |
| `RAG_SANDBOX_CODE_CHARS` | 20000 | the longest code one call may send |
| `RAG_SANDBOX_RESULT_TOKENS` | 3000 | the output (or error) the model reads of a run, in Qwen tokens; the rest is cut with a note |
| `RAG_SANDBOX_USER_RUNS` | 1 | runs of one user in flight at once; another waits (at most `RAG_SANDBOX_REQUEST_TIMEOUT_S`, then a tool error and the notice `python_busy`) |
| `RAG_SANDBOX_REQUEST_SECONDS` | 600 | run time one question may use, together; each run's limit is what is left, and past it a tool error and the notice `python_time_used` |
| `RAG_SANDBOX_TOTAL_INPUT_MB` | 1024 | Python data rag holds for all questions at once (each question's data is dropped when its research ends). The room is reserved before a download starts (a tables file's size is known from its Content-Length), so concurrent questions can never pass it together; a question that would pass it gets a tool error and the notice `python_busy`, and can try again later. Every exit, a cancelled request included, gives its reservation back |
| `RAG_SANDBOX_FETCHES` | 2 | data fetches for Python at once, for everyone; one per user at a time. A tables file is read into a buffer of its known size, so a fetch never holds twice its size |
| `RAG_SANDBOX_CANCEL_WAIT_S` | 60 | when a request ends while its run is going, rag asks the sandbox to kill the run (by an id rag made for it) and keeps the user's run turn until the sandbox says the run ended, at most this long |
| `RAG_READY_CACHE_S` | 5 | `/readyz` answers from its last check this long, so callers share one probe of the backends |
| `RAG_MEM_LIMIT` (compose) | 6g | rag's memory limit. rag loads no model (the tokenizer only); it holds conversations, prompts and at most `RAG_SANDBOX_TOTAL_INPUT_MB` of Python data. Not measured yet: check `docker stats local-rag-rag-1` under load and raise it if rag comes near it |

| Setting (sandbox, compose) | Default | What it sets |
|---|---|---|
| `COMPOSE_PROFILES` | unset | `sandbox` starts the sandbox container with every `docker compose up` |
| `SANDBOX_SUBNET` | 172.31.241.0/24 | the sandbox network's subnet, passed to rag as `RAG_SANDBOX_SUBNET`. Change it if it clashes with a network on the host |
| `SANDBOX_MEMORY`, `SANDBOX_CPUS`, `SANDBOX_PIDS` | 4g, 4, 256 | the container's memory (no swap; the tmpfs mounts and SysV memory count toward it), CPUs and processes |
| `SANDBOX_SCRATCH_SIZE`, `SANDBOX_TMP_SIZE`, `SANDBOX_SHM_SIZE` | 2g, 512m, 64m | the tmpfs mounts for the runs' folders, `/tmp` and `/dev/shm` |
| `SANDBOX_SYSV_SHM_BYTES`, `SANDBOX_SYSV_SHM_PAGES` | 64 MiB, 16384 (64 MiB) | SysV shared memory in the container's own IPC namespace: the largest segment, all segments together (sysctls `kernel.shmmax`, `kernel.shmall`). Fixed alongside: `kernel.shm_rmid_forced=1` (a segment goes once no process is attached, whoever owns it), 32 segments, 16 message queues of 64 KiB, small semaphore sets. Docker allows these sysctls with a private IPC namespace |
| `SANDBOX_QUEUE_WAIT_S` | 300 | runs go one at a time; a request waits this long for the run before it, then gets a 503 ("busy": a tool error and the notice `python_unavailable`) |
| `SANDBOX_MAX_TIMEOUT_S` | 600 | the largest time limit a request may ask for |
| `SANDBOX_CANCEL_WAIT_S` | 30 | a cancel from rag kills the run's process group and answers once the run and its clean-up ended, at most this long; a run cancelled before its turn never starts (409) |
| `SANDBOX_MAX_INPUT_MB`, `SANDBOX_MAX_INPUT_FILES`, `SANDBOX_MAX_CODE_KB` | 512, 64, 256 | what one request may carry (413 past them); keep `SANDBOX_MAX_INPUT_MB` above `RAG_SANDBOX_INPUT_MB` |
| `SANDBOX_MAX_OUTPUT_KB` | 256 | stdout and stderr returned, each: past it, the first three quarters and the last quarter with a line saying how much was left out |
| `SANDBOX_MAX_FILE_MB` | 256 | the largest file a run may write, its output included (RLIMIT_FSIZE); a larger write fails and the run is reported as over the limit |
| `SANDBOX_MAX_FILES`, `SANDBOX_MAX_RETURN_FILE_KB`, `SANDBOX_MAX_RETURN_KB` | 50, 64, 256 | files a run left in its folder: how many are listed (with the total, counted up to 10,000), and the text returned of small UTF-8 ones |
| `SANDBOX_MAX_PROCESSES` | 128 | processes and threads of the sandbox's user, the server's included (RLIMIT_NPROC); keep it below `SANDBOX_PIDS` so the server always has room |
| `SANDBOX_MAX_OPEN_FILES`, `SANDBOX_THREADS` | 1024, 4 | open files per process; threads numpy, polars and OpenBLAS start |
| `SQL_MAX_FILE_STREAMS`, `SQL_FILE_QUEUE_WAIT_S` | 2, 20 | tables files `sql` sends to rag at once, one per user at a time, so one user's fetches cannot take every place; a request waits up to `SQL_FILE_QUEUE_WAIT_S` for its place, then gets a 503 (rag retries once) |

**What a run can see** (the data contract, ARCHITECTURE section 7.10): `data/manifest.json`, and for each selected
document `data/<n>_<name>/tables.duckdb` (its tables, as `sql` queries them) and, except for spreadsheets,
`data/<n>_<name>/text.jsonl` (its passages). rag fetches them from `sql` and `ingest-api` as the signed caller, so
another user's document cannot be among them; the sandbox holds no identity and no key. Nothing a run writes
outlives it.

**Containment.** The sandbox's only network is `sandbox`, internal, with `rag` as the only other member: no
internet, no DNS for outside names (Docker Engine 26 or newer does not forward them for internal networks), no
route to `auth`, `threads`, `audit`, Qdrant, `llm`, `models`, `sql`, `ingest-api` or `gateway`, and rag answers
every request from that network with 403 before anything else. It runs as its own uid 10101 on a read-only root,
with `/scratch` and `/tmp` as noexec tmpfs, every capability dropped, `no-new-privileges`, Docker's default
seccomp profile (which refuses the kernel keyring and, on current engines, io_uring), `ipc: private` with small
SysV limits, the server as PID 1 (no init process a run could trace and keep alive between runs) and the limits
above; no secrets, no volumes, no `.env`. Runs go one at a time, each in a fresh folder; the server reads a run's
output only through descriptors it opened before the run and walks its folder without following links. At the end
of a run it kills the run's process group and every other process of its user (setsid escapes included), empties
the scratch folder, `/tmp` and `/dev/shm`, and removes every SysV object of the namespace whoever owns it; if
anything is left, it answers 503 to everything and stops, and Docker starts the container again. The server is
not dumpable (no run can read its memory) and refuses runs requested from inside the container (any loopback
address or one of its own).

**The host side of the sandbox network.** Docker filters forwarded traffic of an internal network, not traffic
to the host itself: when the bridge holds an address on the host (172.31.241.1 by default), a run can reach a host
service that listens on every interface there. `compose.yaml` therefore puts the network in isolated gateway mode
(`com.docker.network.bridge.gateway_mode_ipv4: isolated`, Docker Engine 28 or newer, internal networks only): the
bridge gets no address on the host side at all, and IPv6 is off on the network. An older engine refuses to create
the network; update Docker rather than removing the option. With no host address, Docker may give the subnet's
first address to a container (often the sandbox itself). The isolation check fails if the network has a gateway
address or the sandbox has a route through one, and scans every TCP port and probes common UDP ports (DNS, DHCP,
TFTP, NTP, NetBIOS, SNMP, IKE, SSDP, mDNS, memcached) of any host-side address it finds: a route's gateway, or the
first address when no container holds it. Any answer fails it. As a second layer on a Linux host, drop the traffic on the host:

```bash
# iptables (nftables: the same rule in the inet filter input chain); persist it with your firewall's tooling
sudo iptables -I INPUT -s 172.31.241.0/24 -j DROP
```

**Docker Desktop** (Windows or macOS): the containers run in Docker Desktop's Linux VM (on Windows a WSL2 distro),
so "the host" of the paragraph above is that VM, not Windows or macOS; a firewall rule belongs inside the VM, and
the host check of uid 10101 is skipped. Desktop also offers ways from containers to the real host:
`host.docker.internal` and `gateway.docker.internal` (and `kubernetes.docker.internal`), and the VM's own network
192.168.65.0/24. The isolation check verifies that from the sandbox none of these names resolve, that no route leads
to 192.168.65.0/24, and that its usual addresses answer on no common TCP or UDP port. Isolated gateway mode needs
a Docker Desktop release with Engine 28 or newer.

**What is accepted.** One run at a time under one uid with a clean-up between runs is the isolation, not a kernel
or uid of its own per run (DEPLOYMENT section 7.2). A run can see in `/proc/net/tcp` that rag holds connections to
the sandbox (another user's request waiting, and roughly how many bytes are queued), not what they carry. A run
can lower its own OOM score back to 0 and cannot be stopped from sending SIGTERM to the server; and tmpfs and SysV
memory belong to no process, so at the memory limit the kernel may kill the server instead of the run. Each of
these fails closed: the container restarts, the run in progress is lost, and rag reports a tool error. uid 10101
must belong to nothing else on the host (RLIMIT_NPROC counts every process of a uid); the isolation check verifies
it on a Linux host.

Check: `python3 scripts/check_sandbox_isolation.py` ends with `ALL PASSED` (about two minutes; `--skip-port-scan`
leaves out the TCP port scan). Two of its lines are weaker than they read: the SysV shared memory line also passes
because the kernel removes unattached segments (`kernel.shm_rmid_forced`), so the message queue and semaphore lines,
which nothing but the server's clean-up removes, are the proof; and the OOM-score line sees the scores at the run's
start only, since a run can lower its own later. Then, with `RAG_DEEP_PYTHON=request`, ask a deep question over a spreadsheet with
`"python": true` (for example "the correlation between two columns"): the steps show `run_python`, and a source of
kind `python` holds the code and its output. With the `sandbox` container stopped, the same question gets the
notice `python_unavailable` and an answer from the other tools. Without `"python": true`, nothing changes.

Rollback: set `RAG_DEEP_PYTHON=off` (or remove it), remove `sandbox` from `COMPOSE_PROFILES`, then
`docker tag local-rag/rag:previous local-rag/rag:dev` and the same for `sql`,
`docker compose up -d --no-deps rag sql`, and `docker compose rm -sf sandbox`. The new `sql` route is unused by the
old `rag`.

### 4.17 Update: a cache for ingest's model replies

This update ([CHANGELOG](../CHANGELOG.md); ARCHITECTURE section 5.1, "LLM cache") keeps the replies of the
model calls ingest makes (chart data extraction, figure and image descriptions, document summaries) in a
cache keyed on the exact request, so reprocessing unchanged content makes no model call: a reindex, the next
`INDEX_VERSION` raise (section 4.15; its time goes mostly to model calls) or a new
version of a file with the same figures. Entries belong to one account, are never shared, and go with their
document and their account. **Data changes: one new file, `llm_cache.db` in `ingest-data`, created by the
worker at its first start; the registry stays at schema v4.** Only the `ingest` image changes, and
`compose.yaml` passes two more variables to `ingest-worker` (`INGEST_VISION_MODEL_FILE` and
`INGEST_VISION_MMPROJ`, from `LLM_GGUF` and `LLM_MMPROJ`). Update `ingest-api` together with the worker: it
removes the entries of a deleted document or account.

```bash
cd /opt/local-rag
docker tag local-rag/ingest:dev local-rag/ingest:previous
git pull --ff-only
docker compose build ingest-api            # ingest-worker runs the same image
docker compose up -d --no-deps ingest-api ingest-worker
docker compose logs ingest-worker | grep "llm cache"
```

At start the worker logs `llm cache on` with the model identity it keys on: the model name (`LLM_MODEL`), the
weights and projector files (their names carry the quantization), llama-server's build (from its `/props`)
and the salt. When llama-server reports another weights file than `INGEST_VISION_MODEL_FILE`, the worker
logs an ERROR and runs with the cache off, so replies of one model are never served as another's. The cache
starts empty: the first reindex after the update still asks the model for everything.

| Setting | Default | What it sets |
|---|---|---|
| `INGEST_LLM_CACHE` | true | false: every reply comes from the model. Entries already stored stay, and are still removed with their documents and accounts |
| `INGEST_LLM_CACHE_MAX_MB`, `INGEST_LLM_CACHE_MAX_ENTRIES` | 1024, 200000 | past either bound the least recently used entries go, checked on every write and at the worker's start (a description takes a few KB) |
| `INGEST_LLM_CACHE_TIMEOUT_S` | 10 | how long a lookup or write waits for the file's lock; then it counts as an error: a model call, or a reply not stored |
| `INGEST_LLM_CACHE_MAX_ERRORS` | 3 | cache errors in one job after which that job stops using the cache (logged); the next job tries again |
| `INGEST_LLM_CACHE_SALT` | empty | part of every key: set a new value to start afresh, for example after changing the `llm` image or its flags in `compose.yaml` |
| `INGEST_VISION_MODEL_FILE`, `INGEST_VISION_MMPROJ` | `LLM_GGUF`, `LLM_MMPROJ` (set by compose) | the model files in every key. Outside compose, with neither set and no `/props` to name them, the cache stays off |

What the worker logs: one `llm cache summary` line per job, with `hits` (model calls saved), `misses`,
`writes`, `skipped`, `errors` and, when not zero, `skipped_truncated`, `skipped_failed`, `skipped_empty`,
`skipped_refused`, `skipped_not_finished`, `corrupt` and `evicted`; `llm cache skip: reply not stored` with
its reason (a reply cut at its token limit, a failed call, an empty, refused or unfinished reply is never
stored and is asked again next time); a WARNING for a busy or full file and for a corrupt entry, each a model
call instead; an ERROR for a damaged file, which the worker deletes and starts again. Each hit, miss and
write is a DEBUG line (`INGEST_LOG_LEVEL=DEBUG`).

Check: reindex a PDF with figures, wait until it is ready, and reindex it again. The second job's
`llm cache summary` shows `misses` 0 and as many `hits` as the first job's `writes`, and the document is
ready in seconds instead of minutes. Delete it and upload it again: its job asks the model for everything,
because its entries went with it.

Rollback: the `previous` image ignores `llm_cache.db`. Documents deleted while it runs keep their entries
until the new worker is back: its sweep removes the entries of documents that no longer exist within a
minute of its start. To empty the cache by hand, stop ingest and delete the file (or set a new
`INGEST_LLM_CACHE_SALT`, which makes every entry a miss):

```bash
docker compose stop ingest-worker ingest-api
docker compose run --rm --no-deps ingest-worker sh -c 'rm -f /data/llm_cache.db /data/llm_cache.db-wal /data/llm_cache.db-shm'
docker compose up -d --no-deps ingest-api ingest-worker
```

### 4.18 Update: exhaustive mode (read every section)

This update ([CHANGELOG](../CHANGELOG.md); ARCHITECTURE section 7.9) adds an answer mode that reads every
section of the selected documents before answering, for questions about the documents as a whole ("list every
limitation the papers state", "which documents mention X and what does each say"). It runs only when a request
asks for it (`"mode": "exhaustive"`, the new *Read all* toggle in the UI); single and deep answers do not change.
**Data changes: none**; it reads the index as it is, no reindex. The `rag` image changes (the mode) and the
`gateway` image (the toggle). `libs/common` changes only in the names of the new step events
(`chat_events.STEP_TOOLS`), which only `rag` and eval read, so the other services need no rebuild. eval runs from
the repository with uv and needs no rebuild.

```bash
cd /opt/local-rag
for i in rag gateway; do docker tag local-rag/$i:dev local-rag/$i:previous; done
git pull --ff-only
docker compose up -d --build --no-deps rag gateway
```

| Setting | Default | What it sets |
|---|---|---|
| `RAG_EXHAUSTIVE_GROUP_TOKENS` | 8000 | tokens of passages in one map call; small sections are packed up to it, a larger one is split. Never more than the slot leaves after the map prompt and a retried reply |
| `RAG_EXHAUSTIVE_MAP_TOKENS` | 1500 | output room of one map call (its notes); a cut reply is retried once with more room (up to `RAG_RETRY_MAX_TOKENS`), then kept with the notice `exhaustive_map_cut` |
| `RAG_EXHAUSTIVE_CONCURRENCY` | 1 | map and merge calls of one request at once, 1 or 2 (llama-server's slot count; anything else stops rag at start). 1 leaves the other slot to other users |
| `RAG_EXHAUSTIVE_MAX_INPUT_TOKENS` | 300000 | tokens of passages one request may read; reading stops there with the notice `exhaustive_budget`, which names what was not read |
| `RAG_EXHAUSTIVE_MAX_SECTIONS` | 500 | sections one request may read; the same stop and notice |
| `RAG_EXHAUSTIVE_NOTES_TOKENS` | 16000 | notes in the answer prompt (less when the slot leaves less after the answer's room), and in one merge call |
| `RAG_EXHAUSTIVE_REDUCE_TOKENS` | 4096 | output room of one merge call |
| `RAG_EXHAUSTIVE_REDUCE_ROUNDS` | 3 | merge rounds before notes that still do not fit are cut evenly across the documents (notice `exhaustive_notes_cut`) |

Time and GPU: about one map call per 8,000 tokens of selected text (a 20-page paper takes two to four), each a
model call of about the length of a short answer, then the answer. With the default concurrency of 1 the other
llama-server slot stays free, so other users' questions wait at most about one model call, as with deep
search. A request with `"mode": "exhaustive"` and any Deep search field (`agent`, `deep_think`, `fanout`,
`research_calls`) is a 422. The thread and the audit record keep `mode: exhaustive` and `route: exhaustive`;
`done.timings` adds the map and merge counts and times.

For the benchmark, upload the global set's documents once (`bench-setup --set global`: eight of the sample
papers and the Constitution, already there if the local set was set up) and run the set in both modes, then
compare (GETTING_STARTED section 8):

```bash
uv run --project eval python -m rag_eval bench-setup --set global
uv run --project eval python -m rag_eval bench --set global --mode exhaustive --variant exhaustive --repeats 2
uv run --project eval python -m rag_eval bench --set global --mode deep --agent langgraph --variant deep --repeats 2
uv run --project eval python -m rag_eval compare eval/reports/bench-exhaustive-<time>.json eval/reports/bench-deep-<time>.json
```

Check: tick one paper, turn on *Read all* and ask "List every limitation this paper states." The steps show
`sections` with the plan, then one `map` per part ("1-3 of 12 | paper.pdf: 1 Introduction (p. 1), ...") and
the answer cites passages from all over the paper; turning on *Deep search* turns *Read all* off. Rollback: the
`previous` images (section 4.9); nothing stored changed.

### 4.19 Update: name forms in deep search

This update ([CHANGELOG](../CHANGELOG.md); ARCHITECTURE section 7.6) helps Deep search see how a column writes
a name before it filters or counts by it. Every arm's research rules gain one rule: look at how the column
writes a place, country, organization or product (its distinct values or its profile), match every form that
means the same (abbreviations, codes, short or official names), and check what a substring or case-insensitive
match caught before counting it. The profile tool (`profile_column`) adds a value structure for text columns
whose values mostly contain a separator: every
distinct last part with its row count, and the parts that look like abbreviations or codes. **Data changes:
none**; the summary is computed from the tables as they are, no reindex. Only the `rag` image changes; the
queries are ordinary read-only SELECTs, so `sql` needs no rebuild.

```bash
cd /opt/local-rag
docker tag local-rag/rag:dev local-rag/rag:previous
git pull --ff-only
docker compose up -d --build --no-deps rag
```

| Setting | Default | What it sets |
|---|---|---|
| `RAG_PROFILE_PARTS_MAX` | 150 | distinct last parts listed in full, each with its count (1 to 4,000); past it, the 30 with the most rows and how many more |
| `RAG_PROFILE_PARTS_TOKENS` | 2000 | the value structure of one column in `profile_column`; the rarest parts are dropped first, the line of possible codes stays |

Each profile of such a column runs one more query in the SQL sandbox. A query that fails leaves the profile as it was,
with a line saying so, and the notice `profile_parts_failed`. Check: in Deep search, ask about a spreadsheet with
a column like "12 km NNE of Town, Region"; after a `profile_column` step, its source ("profile of column ... in
table ...") ends with `last part after ','`, the parts with their counts and `possible abbreviations or codes`. Rollback: the `previous` image (section 4.9);
nothing stored changed.

### 4.20 Update: total rows in deep search

This update ([CHANGELOG](../CHANGELOG.md); ARCHITECTURE section 7.6) helps Deep search keep total and subtotal
rows out of the sums of their parts. Every arm's research rules gain one rule: before adding up, counting,
averaging or ranking rows, look at the label column for totals (a label such as Total or All ..., a country above
its states, regions above their states, a parent code above its parts) and total columns, leave out a total that
covers rows already included, and use the total row when the question asks for the total or a share of it. The
schema cards name listed values and number columns whose words say total, and `describe_table` and
`profile_column` add what the sums show: a value whose sum equals the
sum of the others (or a half, third or quarter of it), the subtotals between, and codes that begin longer codes.
**Data changes: none**; the checks read the tables as they are, no reindex. Only the `rag` image changes.

```bash
cd /opt/local-rag
docker tag local-rag/rag:dev local-rag/rag:previous
git pull --ff-only
docker compose up -d --build --no-deps rag
```

| Setting | Default | What it sets |
|---|---|---|
| `RAG_TOTALS_LABELS_MAX` | 500 | a text column with more distinct values is not summed (3 to 4,000) |
| `RAG_TOTALS_TOLERANCE` | 0.005 | how close the sums must be (a share of the larger; above 0, at most 0.1) |
| `RAG_TOTALS_COLUMNS` | 4 | text columns of a table summed by `describe_table` (0 to 20); 0 turns the sums off and keeps the names |

Each `describe_table` call runs one more read-only query per checked text column, once per table; a
`profile_column` call on a text column runs one more.
A query that fails leaves the result as it was, with a line saying so, and the notice `totals_check_failed`.
Check: in Deep search, ask about a table with a total row (for example a row "United States" among the states);
the `describe_table` step's result ends with `possible totals in column ...` and the line `(leave a total out when
adding up the rows or columns it covers; use it when the total is asked for)`. Rollback: the `previous` image
(section 4.9); nothing stored changed.

### 4.21 Update: text in PDF margins (index version 3)

This update ([CHANGELOG](../CHANGELOG.md)) fixes PDF text that was left out of the search index. Docling marks
text in the top and bottom margins as page headers and footers, and all of it was dropped. Text that is part of
the document but runs into a margin was lost with it, for example the last line of a resume's education section
at the foot of its last page. Now a margin line is dropped only when it repeats on another page (a running
title, digits ignored), is a page number, or has fewer than 3 words. **Data changes: every ready document is
reindexed, on its own**, the same way as in section 4.15: the worker re-queues them at its first start and logs
`re-queued documents an older index version built`. They stay searchable on their current index until the new
one is ready, and the replaced index is removed after `INGEST_GENERATION_RETAIN_H` hours. Figure descriptions
come from the model reply cache (section 4.17), so the reindex is much faster than the first ingest.

```bash
docker tag local-rag/ingest:dev local-rag/ingest:previous
docker compose build ingest-api            # ingest-worker runs the same image
docker compose stop ingest-worker ingest-api
docker compose up -d --no-deps ingest-api ingest-worker
docker compose logs ingest-worker | grep "re-queued"
```

### 4.22 Update: search tasks (threads schema v3)

This update ([CHANGELOG](../CHANGELOG.md); ARCHITECTURE sections 7.2 and 7.7; design:
[design/search-tasks.md](design/search-tasks.md)) lets each question choose the instruction it is embedded with
before the meaning search: a *Search task* picker next to *Think*, three built-in tasks (General, Multi-step
research, Exact lookup) and each user's own, managed in the *Search tasks* panel.

General's instruction changes with this update, from `Given a question, retrieve passages that answer it` to
`Given a question about the user's documents, retrieve passages with the facts needed to answer it`. The new one was
chosen by a measured experiment ([benchmarks/embedding-models-2026-11-15.md](benchmarks/embedding-models-2026-11-15.md)):
- **Retrieval:** better recall@10 after reranking on held-out halves of two public multi-hop sets (+0.9 and +1.9
  points), and the same on the example documents.
- **End to end:** not significantly different on the local and hard benchmark sets.

Some answers can therefore differ after the update. To keep the old instruction, set
`RAG_QUERY_INSTRUCTION=Given a question, retrieve passages that answer it` in `.env`.

**Data changes:** `threads.db` goes from schema v2 to v3 when the new `threads` starts: one transaction that adds
the `search_tasks` table and keeps every row. Back up first. Documents and the search index are untouched: only
the question's side of the search changes, so there is no reindex.

Rebuild `models`, `threads`, `rag` and `gateway`. Start `models` before `rag`: the new `rag` sends the instruction
with every question and checks that `models` applied it; the earlier `models` ignores it, so questions would fail
with "models does not apply search tasks" until `models` is updated. `ingest`, `sql`, `audit`, `auth` and `sandbox`
change only through `libs/common`: a refused request (422) now names the field and the reason without echoing the
value sent. Rebuild them too, in step 4.

```bash
cd /opt/local-rag
# 1. back up every volume (section 6.2), then keep the current images for a rollback
for i in models rag threads gateway auth audit ingest sql sandbox; do docker tag local-rag/$i:dev local-rag/$i:previous; done
# 2. update the code and build the images while the old services keep running
git pull --ff-only
docker compose build models threads rag gateway auth audit ingest-api ingest-worker sql sandbox
# 3. models first: it reloads its embedders and reranker, a few minutes; plan it for a quiet moment
docker compose up -d --no-deps models
# 4. then threads (it upgrades threads.db at start), rag and the gateway, and the services that share libs/common
docker compose up -d --no-deps threads rag gateway
docker compose up -d --no-deps auth audit ingest-api ingest-worker sql sandbox
docker compose --profile eval build eval               # only if you run eval in Docker
```

Then check:

1. `docker compose logs threads rag models | grep -i error` shows nothing new.
2. `docker compose exec threads python -c "import sqlite3; print(sqlite3.connect('/data/threads.db').execute('PRAGMA user_version').fetchone())"`
   prints `(3,)`.
3. Sign in: the *Search task* picker under the question box lists General, Multi-step research and Exact lookup.
   Add a task in *Search tasks* (top bar), pick it, and ask a question: the answer's footer shows
   `search task: <its name>`. Your chats and memories are as before.

| Setting | Default | What it sets |
|---|---|---|
| `RAG_QUERY_INSTRUCTION` | `Given a question about the user's documents, retrieve passages with the facts needed to answer it` | General's instruction; empty embeds the bare question. One line, at most 300 characters, or rag does not start. Query side only: set it in `.env` and recreate `rag` (`docker compose up -d --no-deps rag`), no reindex |
| `THREADS_MAX_SEARCH_TASKS` | 50 | custom search tasks per user; adding one past it is refused with a message that says so |

`MODELS_QUERY_INSTRUCTION` stays as the fallback for a caller that sends no instruction; rag always sends one. The
reranker's instruction does not change. If you set `MODELS_QUERY_INSTRUCTION`, copy it to `RAG_QUERY_INSTRUCTION`,
because rag now always sends General's instruction. If `rag` reports "models does not apply search tasks", the
running `models` is the earlier one: rebuild and restart it (step 3).

Rollback: restore the backup from step 1, as at the end of section 4.11, which loses what was written since. To
keep the chats and memories written since, put back the `previous` images of the four services instead (section
4.9): the earlier `threads` ignores the `search_tasks` table and keeps working with a v3 `threads.db`. It does not
delete a user's search tasks when it deletes the account; the next start of the new `threads` deletes them (the
log says `deleted the search tasks of accounts deleted before`).

## 5. Network exposure

> **Keep the gateway port on 127.0.0.1** (`GATEWAY_BIND`, the default). Give other computers access
> through a TLS reverse proxy on the same host (section 5.3) or SSH tunnels (sections 5.1 and 5.2).
> Do not publish the port on `0.0.0.0` or a LAN address, forward it on a router, or expose it with
> `tailscale serve` or `funnel`: passwords and session cookies would cross the network in plain text,
> and the gateway could no longer tell a trusted proxy from a client.
>
> A host firewall does not catch this mistake. Docker writes its own iptables rules, so a port
> published on `0.0.0.0` is reachable even when `ufw` denies it. The `127.0.0.1` binding in
> `compose.yaml` is what keeps the gateway private.

The gateway applies three checks (`services/gateway/default.conf.template`):

- **Host check.** It answers only requests addressed to `localhost`, `127.0.0.1`, `gateway` (the
  name other containers use) or the host name in `PUBLIC_ORIGIN`. Any other host name, such as the
  server's name or a LAN address, gets no response at all. This also stops a malicious web page that
  points its own host name at 127.0.0.1 (DNS rebinding).
- **Origin check.** Browser requests must come from `PUBLIC_ORIGIN`, `http://localhost:${GATEWAY_PORT}`
  or `http://127.0.0.1:${GATEWAY_PORT}`. A `GET` without an `Origin` header passes; a `POST`, `PATCH`
  or `DELETE` without one passes only with an API token. Anything else gets `403`: pages still load,
  but uploads and chat fail.
- **Sign-in check.** Every `/api` request except sign-in itself goes to the `auth` service first
  (nginx `auth_request`). A valid session cookie or API token comes back as a signed identity, which
  replaces the cookie and the token on the way to the backends; anything else gets `401`. Sign-in
  attempts are limited to 10 per minute per client address (burst 5), on top of the lockouts in
  section 9.3.

With the default `PUBLIC_ORIGIN` the browser must use `localhost` or `127.0.0.1`, which is what an SSH
tunnel gives it. With a TLS proxy, users open `PUBLIC_ORIGIN`.

The Python sandbox (section 4.16, compose profile `sandbox`) publishes nothing and is not reachable from the host
network or from any container but `rag`. Never publish its port or attach it to another network: it runs
model-written code and accepts runs without authentication, because only `rag` can reach it. In the other
direction, rag refuses every request from the sandbox network (`SANDBOX_SUBNET`), and a run may reach the host's
address on that network if a host service listens on every interface: the isolation check scans it, and section
4.16 gives a firewall rule.

### 5.1 SSH tunnel

Each user runs this on their own computer. It works the same in Windows PowerShell, which has an
OpenSSH client built in:

```bash
ssh -N -o ServerAliveInterval=30 -L 8000:127.0.0.1:8000 tunnel@rag-server
```

They then open http://localhost:8000. The local port (the first `8000`) must equal the server's
`GATEWAY_PORT`, because of the Origin check. If 8000 is taken on users' computers, change
`GATEWAY_PORT` on the server and have everyone use the new port.

On the server, use key-only SSH (`PasswordAuthentication no`). Give tunnel users an account that can
forward to the gateway and nothing else, with this in `~tunnel/.ssh/authorized_keys`:

```
restrict,port-forwarding,permitopen="127.0.0.1:8000" ssh-ed25519 AAAA... alice@laptop
```

The port in `permitopen` must be the server's `GATEWAY_PORT`.

A tunnel only carries the connection: users still sign in, and each sees only their own data
(section 9). Every tunnel reaches the gateway from the same address, so the per-address sign-in limit
(20 failures in 15 minutes) counts all tunnel users together. Tunnels assume the default
`PUBLIC_ORIGIN` (`http://localhost:<GATEWAY_PORT>`).

### 5.2 VPN, then SSH tunnel

WireGuard or Tailscale gives the server a private address. Users run the same SSH tunnel to that
address or name (for example `tunnel@rag-server` over Tailscale MagicDNS), so SSH does not have to be
reachable from the internet. Allow SSH only on the VPN interface:

```bash
sudo ufw allow in on tailscale0 to any port 22 proto tcp     # or wg0 for WireGuard
```

Do not bind the gateway to the VPN address. Keep it on loopback and use the tunnel, or a TLS proxy
(section 5.3) that listens on the VPN interface.

### 5.3 TLS reverse proxy (the supported setup)

This is the supported way to give other computers access. A TLS reverse proxy on the host, such as
Caddy or nginx, serves HTTPS for a name such as `rag.example.lan` and forwards to
`127.0.0.1:${GATEWAY_PORT}`. The gateway itself stays on loopback. The proxy needs:

- **Body size** at least `MAX_UPLOAD_MB`. The UI sends one request per file, so this is the size of
  the largest file (scripts such as the smoke test send several files in one request).
- **Timeouts** of at least 900 seconds, matching the gateway's `proxy_read_timeout` and
  `proxy_send_timeout` for `/api/`. *Deep search* answers stream for minutes.
- **No buffering on `/api/chat`**, which is a server-sent-events stream.
- **The `Host` header passed on unchanged** (Caddy does this by default; nginx needs
  `proxy_set_header Host $host`), and `X-Forwarded-For` set to the client's address.
- **HSTS** (`Strict-Transport-Security`), set by the proxy; the gateway does not send it.
- **A firewall** that allows only 443 and SSH.

Then set three lines in `.env` and run `docker compose up -d`:

```ini
PUBLIC_ORIGIN=https://rag.example.lan
TRUSTED_PROXY_CIDR=172.31.240.1/32
GATEWAY_BIND=127.0.0.1
```

- **`PUBLIC_ORIGIN`** puts the name on the gateway's Host and Origin allow-lists and switches the
  session cookie to `__Host-rag_session` with `Secure`, so everyone signs in again once.
- **`TRUSTED_PROXY_CIDR`.** Connections to the published port reach the gateway from the first
  address of `FRONTEND_SUBNET`, 172.31.240.1 by default (checked on Docker Desktop). Trusting that
  address lets the per-address sign-in lockout (`AUTH_IP_FAILURE_LIMIT`), the gateway's sign-in rate
  limit and the `ip` field of sign-in audit events see each browser's address instead of one shared
  address. Trust it only while `GATEWAY_BIND` stays `127.0.0.1`: on `0.0.0.0`, LAN clients arrive from
  the same address and could fake `X-Forwarded-For`. A proxy that runs as a container on the
  `frontend` network uses its own address instead. To check, sign in through the proxy and open
  *Audit log* > *Security events*: the `ip` of your `auth.login_succeeded` event should be your
  computer's address.

Caddy with its offline internal CA:

```
# /etc/caddy/Caddyfile
rag.example.lan {
    tls internal                     # certificate from Caddy's own CA; nothing is requested online
    header Strict-Transport-Security "max-age=31536000"
    request_body {
        max_size 200MiB              # MAX_UPLOAD_MB
    }
    reverse_proxy 127.0.0.1:8000 {   # GATEWAY_PORT
        flush_interval -1            # pass streamed answers on at once
    }
}
```

Caddy passes the `Host` header, sets `X-Forwarded-For` and has no proxy timeouts by default.
`tls internal` issues and renews the certificate from Caddy's local CA without internet access.
Clients must trust that CA's root certificate, which the Debian and Ubuntu packages keep at
`/var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt`. Copy it to every client:

- **Windows** (PowerShell as administrator): `Import-Certificate -FilePath root.crt -CertStoreLocation Cert:\LocalMachine\Root`
- **Linux:** copy it to `/usr/local/share/ca-certificates/caddy-local.crt`, then run
  `sudo update-ca-certificates`.
- **Firefox** keeps its own store; turn on `security.enterprise_roots.enabled` or import the file there.

The name must resolve on every client, through a DNS entry on the LAN or a line in each hosts file.
This example follows Caddy's documentation; it has not been run end to end against this stack yet.

Firewall:

```bash
sudo ufw default deny incoming
sudo ufw allow OpenSSH
sudo ufw allow 443/tcp          # the TLS proxy
sudo ufw enable
```

### Consistent reindex upgrade

Registry schema v5 adds a published `generation`, an upload `storage_id`, and a generation catalog.
Existing documents keep their legacy files and point IDs and need no automatic reindex. A new run writes
alongside them and publishes only after every artifact is ready. New questions pin the published
generation; questions already in progress finish on their original selection.

Schema v6 adds the clean-up of old generations (applied automatically at start). After a reindex the
replaced generation is retired: it stays readable for `INGEST_GENERATION_RETAIN_H` hours (24 by default;
a question that started on it can finish), then the worker's sweep removes its points, tables, chunk file
and figures. A run that fails removes what it wrote at once; the leftovers of a run that stopped or was
taken back go after four run leases (at least 10 minutes). The uploaded file and the current generation
are never touched. Generations replaced before v6 count as retired from the first start of v6. The
worker logs each removal (`removed an old index generation`). Saved chats keep their source text; a
figure of a removed generation shows "Figure not available".

Do not mix old readers or writers with generation-aware services. Before updating, let active answers
finish, close the app in other browsers, and back up all five data volumes and `secrets/` together using
section 6.2. Build the images while the old app runs, then stop its writers and readers before recreating
them. For the standard Compose deployment:

```sh
docker compose build ingest-api ingest-worker rag sql gateway auth threads audit
docker compose stop gateway ingest-worker ingest-api rag sql auth threads audit
docker compose up -d --no-deps --no-build audit threads auth ingest-api sql rag ingest-worker gateway
docker compose up -d --no-deps --no-build --wait --wait-timeout 180 audit threads auth ingest-api sql rag ingest-worker gateway
```

On a Mac use these commands instead; leave the two native inference servers running:

```sh
python3 scripts/macos.py compose build ingest-api ingest-worker rag sql gateway auth threads audit
python3 scripts/macos.py compose stop gateway ingest-worker ingest-api rag sql auth threads audit
python3 scripts/macos.py compose up -d --no-deps --no-build audit threads auth ingest-api sql rag ingest-worker gateway
python3 scripts/macos.py compose up -d --no-deps --no-build --wait --wait-timeout 180 audit threads auth ingest-api sql rag ingest-worker gateway
```

The first `up` starts all updated services, including the worker; the second waits for health. Waiting
for RAG/ingest readiness before starting the worker can otherwise wait on its missing heartbeat.
The first migration is automatic and retains accounts, sessions and existing document metadata.
Every new reindex receives new point IDs even when its text is unchanged. Regenerate and review any
evaluation golden set tied to those IDs after reindexing; a score change against old IDs is label drift.
Rollback to pre-generation code requires restoring the matching data backup, including Qdrant, and
the previous images. Merely checking out old code would make it read incompatible point selections.
Never use `down -v` as an upgrade or rollback step. Validate a disposable synthetic workload using
[WORKLOAD_VALIDATION.md](WORKLOAD_VALIDATION.md).

## 6. Data, backups and restore

### 6.1 What is stored where

Compose names volumes `local-rag_<name>` (project `name: local-rag`).

| Volume | Mounted in | Holds | Back up |
|---|---|---|---|
| `local-rag_ingest-data` | `ingest-api`, `ingest-worker` (`/data`); `sql` sees only `tables/`, read-only | one folder per user under each kind: `files/<user_id>/` (the uploaded originals), `assets/<user_id>/` (figure crops), `tables/<user_id>/<doc_id>.duckdb`, `chunks/<user_id>/<doc_id>.jsonl` (chunk text for grep); `registry.db` (SQLite, WAL: registry with each document's owner, the job queue, and the ids of deleted accounts), `llm_cache.db` (SQLite, WAL: cached model replies per owner, section 4.17), `worker.heartbeat` | yes. It is the only copy of the originals |
| `local-rag_qdrant-data` | `qdrant` (`/qdrant/storage`) | collection `chunks`: vectors, BM25 index, payloads (each with its owner's `user_id`) | yes, from the same moment as `ingest-data` |
| `local-rag_threads-data` | `threads` (`/data`) | `threads.db` (SQLite, WAL): chats, running summaries, memories, each with its owner, and the ids of deleted accounts | yes |
| `local-rag_audit-data` | `audit` (`/data`) | `audit.db` (SQLite, WAL): the hash-chained event log, and each user's encryption keys for event details | yes. Never edit it |
| `local-rag_auth-data` | `auth` (`/data`) | `auth.db` (SQLite, WAL): accounts, argon2id password hashes, sessions and API tokens (SHA-256 hashes only), sign-in lockout counters | yes. Without it every account is gone, and the data stays owned by user ids nobody can sign in as |
| `local-rag_models` | `llm`, `models`, `ingest-worker` (read-only); `fetch-models` | `gguf/` and `hf/` (section 4.3), about 25 GB | optional: can be downloaded again, or kept once as a seed folder |

The `sandbox` container has no volume: what a run receives and writes lives on tmpfs and is removed after the
run (section 4.16), so there is nothing of it to back up.

For generation-aware documents, the paths within `ingest-data` are
`files/<user>/<doc>/<storage_id>/source<ext>`, `tables/<user>/<doc>/<generation>.duckdb`,
`chunks/<user>/<doc>/<generation>.jsonl` and `assets/<user>/<doc>/<generation>/<figure>.png`.
The table above also describes the legacy paths, which remain readable. Back up the entire volume,
including all generations and the registry; copying only the latest file is insufficient.

Host files outside the volumes, all ignored by Git: `secrets/` (the Ed25519 signing keys of `auth`
and `rag`, their public keys, and the Qdrant `api_key`; back it up and keep it private), `.env`,
`eval/datasets/golden*.jsonl` and `eval/reports/` (both contain passages from users' documents), and
`samples/`.

To find a volume on disk, run `docker volume inspect local-rag_ingest-data --format '{{ .Mountpoint }}'`.
On Linux this is `/var/lib/docker/volumes/<name>/_data`. With Docker Desktop, volumes live inside its
virtual disk (Settings > Resources > Advanced > Disk image location). Do not edit files there while
the stack runs.

### 6.2 Back up

`ingest-data` and `qdrant-data` describe the same documents, so copy them together. The four SQLite
databases run in WAL mode, which means recent writes sit in the `-wal` file next to each database. So
stop everything that writes, and leave `llm`, `models` and `sql` running: they only read, and the two
GPU services take minutes to reload. The pause lasts a minute or two, and nobody can sign in during it.
Copy `secrets/` with the volumes.

```bash
cd /opt/local-rag
docker compose stop gateway rag ingest-worker ingest-api threads audit auth qdrant
dest=/srv/backups/local-rag/$(date +%F-%H%M); mkdir -p "$dest"
for v in ingest-data qdrant-data threads-data audit-data auth-data; do
  docker run --rm -v local-rag_$v:/v:ro -v "$dest":/backup python:3.12-slim-bookworm \
    tar -C /v -czf /backup/$v.tgz .
done
cp -a secrets "$dest/"; cp .env "$dest/" 2>/dev/null; git rev-parse HEAD > "$dest/commit.txt"
docker compose start
```

PowerShell (workstation; replace `<backup-folder>` with the folder that keeps the backup sets):

```powershell
docker compose stop gateway rag ingest-worker ingest-api threads audit auth qdrant
$dest = "<backup-folder>\local-rag\$(Get-Date -Format yyyy-MM-dd-HHmm)"; New-Item -ItemType Directory -Force $dest | Out-Null
foreach ($v in "ingest-data","qdrant-data","threads-data","audit-data","auth-data") {
  docker run --rm -v "local-rag_${v}:/v:ro" -v "${dest}:/backup" python:3.12-slim-bookworm tar -C /v -czf "/backup/$v.tgz" .
}
Copy-Item -Recurse secrets $dest
if (Test-Path .env) { Copy-Item .env $dest }
git rev-parse HEAD | Out-File -Encoding ascii "$dest\commit.txt"
docker compose start
```

- If the Docker data directory is on LVM, ZFS or btrfs, you can take a filesystem snapshot instead of
  tar. Stop the same services, snapshot, start them again, and copy from the snapshot. The pause is a
  few seconds.
- Run backups from cron or a systemd timer at a quiet hour, and delete old sets on a schedule, for
  example `find /srv/backups/local-rag -mindepth 1 -maxdepth 1 -mtime +30 -exec rm -rf {} +`.
- Keep copies off the machine, and encrypt them: they contain every user's documents, chats and
  memories, the password hashes and `secrets/`.
- **Deleting an account does not reach older backups.** A set made before the deletion still holds
  that user's documents, chats and memories, and its `audit-data` copy predates the crypto-shredding,
  so their audit details stay readable there. Expire old sets on a schedule that matches how
  long a deleted user's data may be kept.
- Test a restore now and then.
- `llm_cache.db` in `ingest-data` is only a cache of model replies (section 4.17): a set restored without it
  costs model calls at the next reindex, nothing else.

### 6.3 Restore

Restore all five data volumes and `secrets/` from the same backup set. Mixing sets breaks the link
between the registry and the vectors, or between accounts and the data they own. On a new machine,
fill `local-rag_models` first (section 4.3).

```bash
cd /opt/local-rag
src=/srv/backups/local-rag/<set>
git checkout "$(cat "$src/commit.txt")"         # optional: the code that wrote the backup
cp -a "$src/secrets" .                          # before compose: every service needs these files
docker compose up --no-start                    # new machine: creates the volumes (needs the images)
docker compose stop gateway rag ingest-worker ingest-api sql threads audit auth qdrant
for v in ingest-data qdrant-data threads-data audit-data auth-data; do
  docker run --rm -e V=$v -v local-rag_$v:/v -v "$src":/backup:ro python:3.12-slim-bookworm \
    sh -c 'find /v -mindepth 1 -delete && tar -C /v -xzpf "/backup/$V.tgz"'
done
docker compose up -d
docker compose run --rm auth auth-admin check   # last line: multi-user: ready
```

Then sign in: *Audit log* should read "Hash chain verified", and *Whole log* should verify for an
admin. `tar` running as root restores the original file owners (uid 10001 for the app's files).

Without the backup's `secrets/`, create new keys with
`uv run --with cryptography scripts/init_secrets.py --force` and run
`docker compose up -d --force-recreate`. Accounts, sessions and API tokens survive, because they live in
`auth.db`; the same two commands rotate the keys at any time.

### 6.4 Rebuild the search index without a Qdrant backup

The originals in `ingest-data` are enough to rebuild `qdrant-data`. You might need this after losing
the volume, or after the embedding models changed and services report
`collection 'chunks' was built with ...`. These steps follow the code but have not been run end to
end yet, so take a backup first (section 6.2).

```bash
cd /opt/local-rag
docker compose rm -sf qdrant
docker volume rm local-rag_qdrant-data
docker compose up -d
docker compose restart ingest-worker      # the worker creates the collection when it starts
export RAG_TOKEN=rag_pat_...              # a token of the account whose documents to reindex
H="Authorization: Bearer $RAG_TOKEN"
for id in $(curl -s -H "$H" http://127.0.0.1:8000/api/documents | python3 -c 'import json,sys; print(*[d["doc_id"] for d in json.load(sys.stdin)])'); do
  curl -s -o /dev/null -X POST -H "$H" http://127.0.0.1:8000/api/documents/$id/reindex
done
```

Each account sees only its own documents, so run the loop once per account. An operator can create a
one-day token for any account with `auth-admin create-token --username <name> --name reindex --days 1`
(section 9.6); it shows in that user's *API tokens* list and audit trail. Documents are re-processed
one at a time, including every figure description, so a large library takes hours.

## 7. Security checklist

### 7.1 Already in place

Checked against `compose.yaml`, the Dockerfiles and the gateway template, and with `docker inspect` on
the stack from before accounts. The `compute` network, the Qdrant `api_key` and the `sql` mount were
tried with throwaway containers on Docker Desktop.

| Area | What |
|---|---|
| Network | Only `gateway` publishes a port, on `GATEWAY_BIND` (127.0.0.1). `backend` and `compute` are `internal: true`, with no route to the internet. `frontend` is a normal network, because Docker publishes ports only from one; it holds only `gateway` (nginx with a fixed configuration, storing nothing) and `fetch-models` (profile `setup`, the only container that downloads anything). `llm`, `models` and `sql`, which handle untrusted input or model-written SQL, sit only on `compute`: they cannot resolve or reach `auth`, `threads`, `audit`, `ingest-api` or Qdrant, and the only HTTP service in their reach is `rag`, which refuses any request without an identity signed by `auth` |
| Gateway | Host allow-list (`localhost`, `127.0.0.1`, `gateway` and the host of `PUBLIC_ORIGIN`; any other host name gets no response); Origin allow-list (`403` for other origins, and for a `POST`, `PATCH` or `DELETE` without an Origin unless it carries an API token); sign-in check on every `/api` request (`auth_request`), with the cookie, the token and any client-sent `X-Identity` or `X-User-*` header removed before the backends; sign-in limited to 10 per minute per address; CSP `default-src 'self'` with no inline scripts, `form-action 'self'`, `base-uri 'none'` and `frame-ancestors 'none'`; `nosniff`; `Referrer-Policy: no-referrer`; upload size limit; audit log GET-only; thread routes reserved for `rag` return `404`; unknown `/api` paths return `404`; UI libraries vendored, no CDN |
| Accounts | Passwords hashed with argon2id (12 to 256 characters, offline list of common passwords refused); per-username lockout and per-address limit, with every attempt counted before its password is checked and the client address taken from `X-Real-IP` only when the gateway sends it (section 9.3); session cookie `HttpOnly` and `SameSite=Strict`, with `Secure` and the `__Host-` prefix over https; sessions and API tokens stored only as SHA-256 hashes; sign-out sends `Clear-Site-Data` |
| Identity between services | `auth` signs a 2-minute Ed25519 assertion of the caller (`X-Identity`) for each `/api` request, and `rag` re-signs it for its calls to `ingest-api`, `sql` and `threads`. `ingest-api`, `rag`, `threads`, `audit` and `sql` refuse a request without a valid assertion on every route except the health checks (and `audit`'s event writes from the services without a key, section 7.2). `audit` accepts sign-in, account and chat events only signed by `auth` or `rag`. The private keys sit in `secrets/`, mounted as Docker secrets into `auth` and `rag` only |
| Per-user data | Every registry query, Qdrant read, thread, memory and audit query includes the owner, so another user's document, chat or memory answers `404`, like a missing one. Document ids are salted with the owner. Admins manage accounts but have no route to anyone's content. A deleted account is tombstoned in `ingest-api` and `threads`, which refuse to store anything more for it |
| Qdrant | Answers `401` without its `api_key` (health checks stay open); only `ingest-api`, `ingest-worker` and `rag` get it, as a Docker secret |
| Containers | Python services run as uid 10001 and the gateway as nginx's unprivileged user; these drop all Linux capabilities. Every service has `no-new-privileges`. Root filesystems are read-only for `rag`, `sql`, `threads`, `audit` and `auth` |
| Model-written Python | Off unless enabled: the compose profile `sandbox`, `RAG_DEEP_PYTHON=request` or `on` and `RAG_SANDBOX_URL`; with `off` (the default) a request's `"python": true` is a 422. The `sandbox` service sits alone with `rag` on the internal `sandbox` network (fixed subnet; rag refuses every request from it), runs as its own uid on a read-only root with noexec tmpfs, no capabilities, `no-new-privileges`, memory, CPU, pid, RLIMIT and SysV caps, no secrets, volumes or `.env`; one run at a time, its output read only through descriptors opened before it, each run killed with every process it started at its time limit and followed by a clean-up of processes, files, `/tmp`, `/dev/shm` and every SysV object (a failed clean-up restarts the container). It gets only the caller's selected documents' data, fetched by rag with the caller's identity, and no identity of its own; rag limits each user to one run at once, each question to a run-time budget, and the data it holds. `scripts/check_sandbox_isolation.py` checks this live (section 4.16) |
| Untrusted input | `ingest-worker` is capped at 12 GB and parses every file (Docling for PDFs) in a child process that is killed after 30 minutes and replaced when it crashes. `sql` mounts only the `tables` folder of `ingest-data`, read-only (no registry, originals or chunk text), with 2 GB, 2 CPUs, and DuckDB with no file or network access. Uploads go through an extension allow-list, magic-byte check, size limit and name sanitizing; images are re-encoded and spreadsheet formulas are never evaluated |
| Credentials in chat | passwords and API keys typed into a question are redacted before anything is stored: saved chats, research steps, summaries, memory, thread titles and the audit log (`libs/common/src/rag_common/secrets.py`). The local model and the intent router read the message as typed. The threads service refuses memories that look like credentials. The matching uses patterns, so it will not catch every secret |
| Offline | `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` in the `models` and `ingest` images; Qdrant and Hugging Face telemetry off; LangSmith tracing off in `rag` |
| Supply chain | `qdrant` and `llm` images pinned by digest; Python dependencies installed from `uv.lock` with `--frozen`; Docling weights baked in at build and checked offline |
| Audit | append-only, hash-chained log; each user verifies the chain against their own events, and admins verify the whole log (section 8.3); event details are encrypted per user and erased with the account |

### 7.2 Known gaps

- **Host access sees everything.** Isolation is enforced by the application. Anyone with Docker or
  root access on the host can read every volume and `secrets/`, and reset any password with
  `auth-admin` (section 9.6).
- **Admin password resets.** The admin sees the temporary password and could sign in as that user
  once. The forced password change, the revoked sessions and tokens, the
  `auth.password_reset_by_admin` event in the user's own audit trail and a notice shown at every
  sign-in until the user dismisses it (changing the temporary password does not clear it) make this
  visible, not impossible.
- **Shared parser and stores.** `ingest-worker`, `sql` and Qdrant handle every user's documents, so a
  parser exploit could expose documents across users. It would not expose chats, memories or the
  audit log: the worker holds no signing keys.
- **Some audit writes are not authenticated.** `ingest-api`, `ingest-worker` and `threads` hold no
  signing key and write their events unsigned, so a compromised container on the `backend` network
  could append forged document, chat-deletion or memory events under those service names, for any
  user, but not read any. It cannot write sign-in, account or chat events: `audit` takes those only
  signed by `auth` or `rag`. `llm`, `models` and `sql` are not on that network.
- **Side channels.** BM25 statistics are collection-wide, llama-server keeps a prompt cache, and all
  users share the GPU. There are no per-user quotas or fair scheduling.
- **Revocation lag.** An answer already streaming continues after its session is revoked, for up to 15
  minutes. A leaked identity assertion could be replayed inside the backend for up to 2 minutes;
  assertions are never logged.
- **Lockout can be abused.** Anyone can lock a known username for a while by failing sign-ins. The
  per-address limit and an admin clearing the lockout (section 9.4) are the mitigations. A lock never
  passes to a new account of the same name: creating or deleting an account clears its counters.
- **Backups.** Deleting an account (crypto-shredding and purges) does not reach backups made before
  the deletion (section 6.2).
- **Soft deletes in Qdrant.** A deleted document's or account's points are only marked deleted: their
  vectors and chunk text stay in `local-rag_qdrant-data` (the segment files and Qdrant's write-ahead
  log) until Qdrant's optimizer rewrites the segment, by default once 20% of it is deleted, or the log
  rotates. The SQLite stores zero what they delete (`secure_delete`) and truncate their `-wal` file
  after an account deletion; if another connection blocks that, the service logs `WAL checkpoint
  blocked`, and the deleted rows stay in the `-wal` file until a later checkpoint.
- `qdrant` runs as root and `llm` as the image's default user (root). Both keep the default
  capabilities (only `no-new-privileges` is set). They are reachable only on the internal network.
- Base images are pinned by tag, not digest: `python:3.12-slim-bookworm`,
  `nginxinc/nginx-unprivileged:1.29.3-alpine`, `ghcr.io/astral-sh/uv:0.13.0`. `fetch-models` installs
  `huggingface_hub` from PyPI when it downloads (not with `--from-dir`).
- The app does not encrypt data at rest, except audit details. Originals, chats, memories and the
  index are plain files and SQLite databases in Docker volumes. The audit encryption keys sit in
  `audit.db` itself, so they make a deleted user's details unreadable, not a copied volume. Audit
  events from before accounts stay unencrypted and cannot be crypto-shredded (section 9.4).
- The audit log cannot be pruned (by design) and holds full questions and answers.
- TLS ends at the proxy (section 5.3). Containers talk plain HTTP on the internal networks.
- **The Python sandbox** (off unless enabled, section 4.16). Its isolation between runs rests on running one at
  a time as one uid and on the clean-up after each run, not on a separate kernel or uid per run; a kernel exploit
  from model-written code would reach the container's host. Runs share the container's CPU and memory and go one
  at a time for everyone; rag allows one run per user at once, a run-time budget per question and a cap on the
  data it holds, but there is no quota per user over time, so a busy user can still make another user's Python
  wait up to `SANDBOX_QUEUE_WAIT_S` and then fall back to the other tools. A run can open connections to `rag`, the
  only other member of its network, which answers every one of them with 403 before anything else and logs them
  once a minute. On engine versions where an internal network's bridge holds an address on the host, a run could
  reach a host service listening on every interface there: the isolation check scans every port of that address,
  and section 4.16 gives the firewall rule. A run can see in `/proc/net/tcp` that another request is waiting and
  roughly its size, not its content. A run can lower its OOM score back to 0 and can send SIGTERM to the server;
  tmpfs and SysV memory belong to no process, so at the memory limit the kernel may kill the server rather than
  the run: each fails closed (the container restarts and the run is lost). Seccomp is Docker's default profile;
  a daemon set to `seccomp=unconfined` weakens it (the check reports it). Charts a run draws are not shown.

### 7.3 Operator checklist

- [ ] Leave the gateway bound to 127.0.0.1 (`GATEWAY_BIND`); do not edit `ports:` in `compose.yaml`.
- [ ] Remote access only through the TLS proxy (section 5.3) or SSH tunnels (section 5.1), with SSH keys and tunnel-only accounts.
- [ ] `PUBLIC_ORIGIN` set to the https address users open; `TRUSTED_PROXY_CIDR` set only for a proxy on this host.
- [ ] Host firewall set to deny inbound except SSH and 443, remembering that Docker bypasses it for published ports.
- [ ] Only administrators in the `docker` group or with `sudo`.
- [ ] Full-disk encryption on the host (LUKS on Linux, BitLocker on Windows).
- [ ] `chmod 600 .env`; `secrets/` private to the operator, and never copied into `.env` or the repository.
- [ ] Encrypted backups of the volumes and `secrets/` kept off the machine, and a tested restore (section 6).
- [ ] Host OS, NVIDIA driver and Docker patched. Images rebuilt with `docker compose build --pull` from time to time.
- [ ] *Audit log* > *Whole log* checked regularly, and *Security events* read for failed sign-ins and lockouts.
- [ ] Never run `docker compose down -v` or `docker volume prune -a` on a host with data: both delete the volumes.
- [ ] Python stays `RAG_DEEP_PYTHON=off` until `docker build --target test -f services/sandbox/Dockerfile .` passes and `python3 scripts/check_sandbox_isolation.py` ends with `ALL PASSED`; run the check again after every deploy or change to `compose.yaml` or the sandbox image. Never publish the sandbox's port or add it to another network.
- [ ] With Python enabled: host services bound to 127.0.0.1, or the firewall rule of section 4.16 for `SANDBOX_SUBNET`; uid 10101 used by nothing else on the host.
- [ ] Users told what admins can and cannot see: accounts and sign-in events, never documents, chats or memories; a password reset shows the admin a temporary password.

## 8. Operations

### 8.1 Health

| Check | Command | Healthy |
|---|---|---|
| All containers | `docker compose ps` | every service `(healthy)` |
| Gateway | `curl -s http://127.0.0.1:8000/healthz` | `ok` |
| Whole app, as `rag` sees it | `curl -s -H "Authorization: Bearer $RAG_TOKEN" http://127.0.0.1:8000/api/ready` | `"ready":true`; `checks` lists `index`, `models`, `llm`, `ingest`, `sql`, `threads`, `audit` |
| Audit chain | `curl -s -H "Authorization: Bearer $RAG_TOKEN" http://127.0.0.1:8000/api/audit/verify` | `"ok":true` (the whole chain); `events` counts the token user's own events |
| Accounts and per-user data | `docker compose run --rm auth auth-admin check` | each service answers `'multi_user': 1`; last line `multi-user: ready` |
| LLM | `docker compose exec llm curl -fsS http://127.0.0.1:8080/health` | `{"status":"ok"}` |
| Embedders and reranker | `docker compose exec models python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8003/readyz').read().decode())"` | `"ready":true` |
| Python sandbox (profile `sandbox`) | `docker compose exec sandbox python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8008/healthz').read().decode())"`; with `RAG_DEEP_PYTHON` `request` or `on`, `/api/ready` lists `sandbox` too | `{"status":"ok"}`; `python3 scripts/check_sandbox_isolation.py` for the containment |
| Ingestion (Qdrant reachable, worker heartbeat under 60 s old) | `docker compose exec ingest-api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8001/readyz').read().decode())"` | `"ready":true` |

The UI polls `/api/ready` every 10 seconds; the dot at the top right shows the result. `$RAG_TOKEN` is
an API token (section 9.5); without one, `/api` answers `401`.

### 8.2 Logs

`docker compose logs` as in section 4.7. The gateway logs one JSON line per request (time, request ID,
method, path, status, bytes, seconds). Python services log one line per request (health probes
excluded), plus errors with stack traces. `ingest-worker` logs every document it processes with its
time. Question and answer text goes to the audit log and `threads`, not to the service logs.

### 8.3 Audit log

Every user sees only their own events under *Audit log*, numbered in their own sequence (#1 is their
first event). The chain check reads "Hash chain verified: N of your events, unaltered.", "The log was
altered at your event #k." or "The audit log was altered (not in your events). Tell your
administrator." Admins also get two buttons there:

- **Whole log** verifies every account's events, without showing their content.
- **Security events** lists the sign-in and account events of all users: sign-ins, failures,
  lockouts, API tokens and admin actions, with the address and browser of each sign-in. Document,
  chat and memory content never appears there.

Both need a browser session. With an API token, from the host, a user reads their own events:

```bash
H="Authorization: Bearer $RAG_TOKEN"
curl -s -H "$H" "http://127.0.0.1:8000/api/audit?limit=20"                    # newest first, up to 1000
curl -s -H "$H" "http://127.0.0.1:8000/api/audit?action=document.*&limit=100" # * is a wildcard
curl -s -H "$H" "http://127.0.0.1:8000/api/audit?subject=<doc_id or thread_id>"
curl -s -H "$H" "http://127.0.0.1:8000/api/audit?before=<n>&limit=1000"       # page back by event number, for an export
```

Events (`auth.*` and `admin.*` are written by the `auth` service):

```
document.uploaded | rejected | deleted | reindex_requested | ready | failed | superseded
thread.deleted    memory.added | updated | deleted    search_task.added | updated | deleted    chat.answered | failed
auth.login_succeeded | login_failed | locked | logout | password_changed | token_created | token_revoked
auth.account_created | password_reset_by_admin | security_notice_dismissed | disabled_by_admin
auth.enabled_by_admin | role_changed
admin.user_created | user_disabled | user_enabled | role_changed | password_reset | user_unlocked | user_deleted
```

Chat events carry the question, documents, sources, the full answer and timings
([ARCHITECTURE section 10](ARCHITECTURE.md#10-audit-log)). Account events done by an admin appear in
the account's own trail (`auth.*`, the admin as actor) and in the admin's trail (`admin.*`); those run
with `auth-admin` have actor `operator`. Deleting a document or a chat does not remove its audit
events. Deleting an account erases the details of its events (crypto-shredding): the events stay and
the chain still verifies. Writes are best-effort: if `audit` is down, users are not blocked, and the
other services log the failed write at ERROR.

### 8.4 Disk growth

```bash
docker system df -v | grep -E 'local-rag|VOLUME NAME'   # volume and image sizes
```

| Grows with | Where | Reclaim |
|---|---|---|
| Documents: originals, figures, tables, chunk text | `ingest-data` | delete documents in the UI. Older versions of a re-uploaded file are deleted once the new one is ready |
| Vectors and BM25 index | `qdrant-data` | follows document deletions |
| Cached model replies | `ingest-data` (`llm_cache.db`) | bounded by `INGEST_LLM_CACHE_MAX_MB` (1 GB) and follows document and account deletions; empty it as in section 4.17 |
| Chats and memories | `threads-data` | delete chats and memories in the UI |
| Audit events, including full answers | `audit-data` | none: append-only by design. Archiving and resetting it is a policy decision, not routine upkeep |
| Container logs | Docker | rotated by `compose.yaml`: at most 5 files of 10 MB per service |
| Build cache and old images, after every rebuild | Docker | `docker builder prune` and `docker image prune` |

### 8.5 GPU sharing between ingestion and chat

- `llm` has 2 slots of 32,768 tokens. Ingestion makes one LLM call at a time (figure descriptions,
  document summaries), so a chat always has a slot. Both still share the GPU, though, so answers stream
  more slowly while a figure-heavy PDF is processed. With several users, a third concurrent request
  waits for a free slot.
- After each answer, the memory check uses the LLM, and so does the running summary once a chat is
  long. *Deep search* makes several LLM calls per question.
- Embedding and reranking requests to `models` share a bounded queue. With more than 32 waiting, it
  answers `503 GPU queue full`.
- To keep chat fast, ingest large batches at quiet times, or pause ingestion with
  `docker compose stop ingest-worker` and resume with `docker compose start ingest-worker`. Queued
  jobs wait in the durable queue. A job interrupted by the stop goes back to the queue but counts as an
  attempt; after two attempts it is marked *failed*, so retry it with the Reindex (circular-arrow) button.
- Nothing else should use the GPU. Another program's allocations can push `llm` or `models` out of
  memory.

### 8.6 Common failures

| Symptom | Cause | Fix |
|---|---|---|
| The browser gets no response at all | Host check: the address is not `localhost`, `127.0.0.1` or the host of `PUBLIC_ORIGIN` (for example the server's bare name or a LAN address) | open `PUBLIC_ORIGIN` through the TLS proxy (section 5.3), or `http://localhost:<GATEWAY_PORT>` through an SSH tunnel (section 5.1) |
| The page loads but uploads and chat fail with `403` | Origin check: the browser is not on `PUBLIC_ORIGIN`, or on `localhost` or `127.0.0.1` with `GATEWAY_PORT` | open `PUBLIC_ORIGIN`; make a tunnel's local port equal `GATEWAY_PORT` |
| A script or `curl` gets `401 {"detail":"sign in required"}` | no API token, or it expired or was revoked (a password change, disable or reset revokes tokens) | create one (section 9.5) and send `Authorization: Bearer <token>` |
| Compose stops on a missing file under `./secrets/` | `scripts/init_secrets.py` was not run | `uv run --with cryptography scripts/init_secrets.py`, then `docker compose up -d` |
| `auth` exits with `refusing plain http` | `PUBLIC_ORIGIN` is `http://` with a host other than `localhost` | use `https://` behind the TLS proxy (section 5.3) |
| `auth` or `rag` exits with `... does not match ...` naming `/run/secrets/identity_public_keys` | the files in `secrets/` come from different runs | `init_secrets.py --force`, then `docker compose up -d --force-recreate` |
| `sql` does not start, with a volume subpath error | Docker Engine older than 26, or `ingest-api` has not created `tables/` yet | update Docker (Docker Desktop 4.29 or newer); check `docker compose logs ingest-api` |
| Sign-in says "Too many attempts" | lockout: 5 failures in 15 minutes for a username, or 20 for an address (section 9.3) | wait, or have an admin clear it (section 9.4 or 9.6) |
| Creating a user fails with `multi-user not ready: <service>` | that service runs code from before accounts | rebuild it (section 4.8); `auth-admin check` must end with `multi-user: ready` |
| `migrate-legacy stopped at ingest: stop ingest-worker first` | the worker ran less than 60 seconds ago | `docker compose stop ingest-worker`, wait a minute, run it again |
| Upload fails with `413` | a file is larger than `MAX_UPLOAD_MB` (the UI sends one request per file) | raise `MAX_UPLOAD_MB` in `.env` and run `docker compose up -d` |
| Compose rejects `env_file` or `required` | Compose older than v2.24 | update Docker Compose |
| `llm` never becomes healthy, with CUDA out-of-memory in its log | another GPU process, or a larger `LLM_CTX` or `LLM_GGUF` | free the GPU (check `nvidia-smi`), or undo the setting |
| `llm` log says the model file is missing | models volume empty, or `LLM_GGUF` names a file that is not there | run `fetch-models`; list the files with `docker compose exec models ls /models/gguf` |
| `/api/ready` shows `"index": false` | the worker has not created the Qdrant collection yet; it does so at start, after `models` and `llm` are healthy | wait, then check `docker compose logs ingest-worker` |
| `ingest-worker` unhealthy | heartbeat older than 60 seconds: the worker is stuck or crashed | `docker compose logs ingest-worker`; `docker compose restart ingest-worker` |
| The worker logs `the vision server runs another model file than INGEST_VISION_MODEL_FILE; llm cache off` | `llm` loads another weights file than the one compose passed to `ingest-worker`, usually after `LLM_GGUF` changed and only `llm` was recreated | `docker compose up -d` (it recreates `ingest-worker` with the new file name); the cache keys then name the new model |
| The worker logs `llm cache file is damaged; deleting it and starting again` | SQLite found `llm_cache.db` damaged (a disk fault, a copy taken while it was written) | nothing: the worker deleted it, and replies are asked again. If it logs `deleting the damaged llm cache file failed`, empty it by hand (section 4.17) |
| A service exits with `collection 'chunks' was built with ...` | the embedding models changed after indexing | restore `qdrant-data` with `ingest-data` (section 6.3), or rebuild the index (section 6.4) |
| Answers are slow or fail with `GPU queue full` during big uploads | ingestion and chat share the GPU | section 8.5 |
| The disk fills up | logs, build cache, audit log, documents | section 8.4 |
| Deep search notices `python_unavailable` | `sandbox` is down or restarting (its log shows `a run left something behind` before a restart), or busy with other runs past `SANDBOX_QUEUE_WAIT_S` | `docker compose ps sandbox`, `docker compose logs sandbox`; raise `SANDBOX_QUEUE_WAIT_S` (and `RAG_SANDBOX_REQUEST_TIMEOUT_S`) for longer waits |
| Deep search notices `python_input_too_large` | the selected documents' tables and text are over `RAG_SANDBOX_INPUT_MB` | select fewer documents, or raise `RAG_SANDBOX_INPUT_MB` (and `SANDBOX_MAX_INPUT_MB`, `SANDBOX_SCRATCH_SIZE`, `SANDBOX_MEMORY` with it) |
| Every Python run fails with "refused the run (400 ... SANDBOX_MAX_TIMEOUT_S)" | `RAG_SANDBOX_TIMEOUT_S` is above `SANDBOX_MAX_TIMEOUT_S` | lower the one or raise the other |
| A request with `"python": true` gets a 422 `python is off on this deployment` | `RAG_DEEP_PYTHON=off` (the default) | set it to `request` (or `on`), with `RAG_SANDBOX_URL` and the profile `sandbox` (section 4.16) |
| rag does not start: `RAG_DEEP_PYTHON is request, but RAG_SANDBOX_URL is empty` | Python enabled without a sandbox address | `RAG_SANDBOX_URL=http://sandbox:8008` and `COMPOSE_PROFILES=sandbox` |
| Deep search notices `python_busy` or `python_time_used` | the user's other run is still going, rag holds `RAG_SANDBOX_TOTAL_INPUT_MB` of Python data, or the question used `RAG_SANDBOX_REQUEST_SECONDS` | wait, or raise the setting named in the notice |
| rag logs `requests from the sandbox network refused` | a run's code calls rag (expected to be refused; the line counts them per minute) | nothing to fix; many of them come from model-written code probing the network |

For first-run problems, see [GETTING_STARTED section 10](GETTING_STARTED.md#10-troubleshooting).

## 9. Users and accounts

Everyone signs in with an account that an admin created. There is no self-registration, no default
password, and no "first visitor becomes admin".

### 9.1 Roles and what each one sees

| Role | Can | Cannot |
|---|---|---|
| user | upload, search and chat over their own documents; keep their own chats and memories; read and verify their own audit trail; change their password; create API tokens | see anything of another user: their documents, index entries, chats, memories and audit events answer `404`, like missing ones |
| admin | everything a user can, with their own data; create, disable, enable, promote, demote, reset and delete accounts and clear lockouts; verify the whole audit log; read security events (section 8.3) | read anyone else's documents, chats, memories or audit details: there is no route for it |
| operator (Docker or root on the host) | run `auth-admin` (section 9.6); read every volume and `secrets/` | nothing on the host is hidden from the operator |

### 9.2 The first admin

On a fresh install, after `scripts/init_secrets.py` and the first `docker compose up -d --build`:

```bash
docker compose run --rm auth auth-admin create-admin --username <you>
```

It asks for the password twice (at least 12 characters), prints the new account's id, and refuses
once any admin exists. Until then the sign-in page shows this command. Usernames have 3 to 64
characters: lowercase letters, digits, `.`, `_` and `-`. An install from before accounts follows
section 4.10 instead.

### 9.3 Signing in

- **Sessions.** The browser gets a session cookie: `rag_session`, or `__Host-rag_session` with
  `Secure` when `PUBLIC_ORIGIN` is https; always `HttpOnly` and `SameSite=Strict`. A session ends after
  12 hours without use, 7 days after sign-in, or at *Sign out* (`AUTH_IDLE_TIMEOUT_MIN`,
  `AUTH_ABSOLUTE_TIMEOUT_DAYS`).
- **Passwords.** 12 to 256 characters; not on the offline list of common passwords; not containing
  the username; no composition rules. User menu > *Change password* signs out every other session and
  revokes the user's API tokens.
- **Temporary passwords.** New accounts and admin resets get a temporary password. The sign-in page
  asks for a new password before anything else works; after a reset it says "An administrator reset
  your password on [date]. Choose a new one." The app then shows that notice at every sign-in until
  the user clicks *Dismiss*, even after the temporary password was changed.
- **Lockout.** 5 failed sign-ins within 15 minutes lock that username for 15 minutes, doubling with
  each further lock up to 24 hours. 20 failures from one address within 15 minutes block that address
  for 15 minutes. Unknown usernames are counted the same way, so the answer never reveals whether an
  account exists. Every attempt is counted before its password is checked, so guesses sent all at once
  still get at most 5 checked. A correct password clears the username's failure count, not its lock
  history: the next lock still doubles until 24 hours after the last one ended. A wrong current password
  when changing the password or creating a token counts too. The address is the one the gateway saw; a
  caller on the backend network that bypasses the gateway is counted under its own address, whatever
  `X-Real-IP` it sends (`AUTH_TRUSTED_PROXIES`). The gateway also allows at most 10 sign-in requests per
  minute per address (section 5).

### 9.4 Managing accounts (admins)

User menu (your username, top right) > *Users*. *Create user* takes a username and a role (user or
admin). The temporary password is shown once, with a *Copy* button; give it to the user over a
channel you trust, and they choose their own at the first sign-in.

The list shows each account's username, role, status (active, disabled, locked until [time], must
change password, deleting), last sign-in and active sessions. Actions:

| Action | Effect |
|---|---|
| Disable / Enable | disabling signs the user out everywhere and revokes their API tokens; their data stays |
| Make admin / Make user | changes the role |
| Reset password | shows a new temporary password once; signs the user out and revokes their API tokens |
| Unlock | shown while the username is locked out; clears the lockout |
| Delete | after you type the username: removes their documents, index entries, files, chats and memories, and erases the details of their audit events (crypto-shredding). Anything still on its way for the account (an upload, an answer being written) is refused afterwards. The account is locked at once; when the removal takes more than a minute the row shows *deleting* and the list updates itself until it is gone. If a step fails, the row says which; *Retry delete* resumes at once, and `auth` retries on its own a minute after it starts and every hour |

- The panel shows only the actions `auth` allows for each account at that moment: none of the account
  changes on your own row (change your own password from the user menu), none that would disable,
  demote or delete the last active admin, and only *Retry delete* for an account being deleted.
  `auth` checks the same rules again in the transaction that makes the change.
- Each action is recorded in your audit trail; account changes also go into the user's own trail.
- A deleted username can be used again; the new account starts empty and without any lockout.
- Deletion does not reach backups made before it (section 6.2). Audit events from before accounts
  stay unencrypted, so deleting the admin who adopted them hides them but does not erase them: they
  stay in `audit.db` (the hash chain keeps every row), and nobody can read or adopt them again.
- Accounts cannot be created until every data service reports that it scopes data per user
  (`multi-user not ready: <service>` otherwise; section 8.6).

### 9.5 API tokens for scripts and evaluation

Scripts use an API token instead of a browser session. User menu > *API tokens*: a name, a lifetime
(1 to 365 days, default 30) and your password. The token (`rag_pat_...`) is shown once. Send it as
`Authorization: Bearer <token>`. It acts as you and sees only your data; it cannot manage accounts or
tokens. Revoke it in the same panel. Changing your password, or an admin disabling your account or
resetting its password, revokes all of your tokens.

Keep tokens in the shell, never in `.env`, which every service receives:

```bash
export RAG_TOKEN=rag_pat_...                 # PowerShell: $env:RAG_TOKEN = "rag_pat_..."
uv run --with httpx scripts/smoke_test.py --samples samples --url http://localhost:8000
EVAL_TOKEN=$RAG_TOKEN docker compose --profile eval run --rm eval retrieval
```

Evaluation golden sets belong to the token's user
([GETTING_STARTED section 8](GETTING_STARTED.md#8-measure-retrieval-quality-optional)). The benchmark that compares
Deep search variants runs on the host as its own account, with that account's token in `RAG_TOKEN`:
`uv run --project eval python -m rag_eval bench-setup --set local,hotpotqa,musique`, then
`bench --variants ...` (same section).

### 9.6 Recovery from the server (`auth-admin`)

These commands work on the auth database directly, so they restore access when nobody can sign in.
Run each as `docker compose run --rm auth auth-admin <command>` from the repository folder; they need
Docker access on the host.

```bash
auth-admin create-admin --username NAME [--no-adopt-legacy]       # first admin; refused once any admin exists
auth-admin reset-password --username NAME [--keep]                # asks twice; revokes sessions and tokens; a new
                                                                  #   password is required at next sign-in unless --keep
auth-admin unlock --username NAME                                 # clear a sign-in lockout
auth-admin revoke-sessions --username NAME                        # sign out everywhere, API tokens included
auth-admin list-users                                             # role, status, lockout, last sign-in, sessions, id
auth-admin create-token --username NAME --name LABEL [--days 30]  # an API token for that user, printed once
auth-admin migrate-legacy --to NAME [--dry-run] [--force]         # pre-account data to one user (section 4.10)
auth-admin check                                                  # is every data service multi-user?
```

- A forgotten admin password: `reset-password --username <admin>`. It warns when the account is also
  locked out and names the command that clears it.
- `create-admin` and `reset-password` read the password from one line of standard input with
  `--password-stdin` (use `docker compose run --rm -T ...` when piping).
- Every command except `revoke-sessions`, `list-users`, `migrate-legacy` and `check` writes an audit
  event with actor `operator`.
- Exit codes: 0 done, 1 error (printed as `error: ...`), 2 usage error.

### 9.7 Isolation test

`scripts/multiuser_smoke.py` checks the isolation on a running stack with the models loaded. It takes
a few minutes:

```bash
RAG_ADMIN_PASSWORD='<admin password>' uv run --with httpx scripts/multiuser_smoke.py --admin-user <admin> [--url http://localhost:8000] [--docker]
```

In PowerShell, set `$env:RAG_ADMIN_PASSWORD = '<admin password>'` first. The password comes from the
environment, never the command line, and an API token will not do: tokens cannot manage accounts.

- It signs in as the admin and creates two temporary accounts (`mu-smoke-a-...`, `mu-smoke-b-...`). As
  A it uploads small generated files, chats and saves memories. B then tries every route on A's data:
  documents, chunks, figures, chat with A's documents and thread, Deep search, memory commands,
  search, threads, memories, audit and the admin routes.
- It also checks header spoofing, the CSRF and Origin rules, the sign-in lockout, disable and reset,
  A and B working at the same time with no cross-talk, and deleting A: A's sessions get `401`, audit
  verification still passes, and the username can be used again.
- `--docker` adds checks inside the containers through `docker compose exec`, so run it from the
  repository folder (or pass `--compose-dir`): network isolation, direct backend calls without a valid
  identity, and that the deleted account left no Qdrant points or files.
- It deletes everything it created, and refuses to run on an install that has other accounts or data
  in the admin's library. `--cleanup` removes accounts left by an interrupted run.
- It prints a pass/fail table. Exit code 0: all passed; 1: a failure; 2: refused, or a usage error.
- Its sign-ins are paced for the gateway's limit. The lockout checks make about 10 failed sign-ins
  from your address, so a second run within 15 minutes would reach the per-address limit of 20: wait,
  or pass `--skip-lockout`. Behind a TLS proxy, pass `--origin` with `PUBLIC_ORIGIN`.
