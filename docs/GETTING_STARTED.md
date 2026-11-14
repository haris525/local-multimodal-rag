# Getting started

This guide takes you from nothing to asking questions about your own documents. It assumes you have
never run the app. Everything runs on your computer. Setup downloads the models and the app's building
blocks once (about 50 GB in all); after that nothing goes to the internet.

Expect 30 to 45 minutes the first time on a fast connection, mostly downloads, and longer on a slow one.

Everyone signs in with an account. You create the first one, an admin account, in step 5; from it you
can create accounts for other people, and each person sees only their own documents, chats and
memories.

## 1. What you need

| | Minimum | Notes |
|---|---|---|
| GPU | NVIDIA, 32 GB VRAM | the 27B model and three small models stay loaded together, using about 27 GB |
| RAM | 32 GB (64 GB recommended) | Docker on Windows needs at least 24 GB of it |
| Disk | 80 GB free | models about 25 GB, images about 23 GB, plus your documents |
| OS | Windows 11 or Linux | macOS cannot run the NVIDIA containers |
| Software | Docker Desktop (Windows) or Docker Engine and the NVIDIA Container Toolkit (Linux), a recent NVIDIA driver, Git, [uv](https://docs.astral.sh/uv/) | Docker Desktop 4.29 or newer, or Docker Engine 26 or newer; Docker Compose v2.24 or newer (`docker compose version`). On Linux the driver must support CUDA 13.2 |

## 2. Check that Docker can see your GPU

- **Windows:** install Docker Desktop and start it. It must use the WSL 2 backend, which is the default.
- **Linux:** install the NVIDIA driver, Docker Engine and the NVIDIA Container Toolkit as in
  [DEPLOYMENT.md, section 4.1](DEPLOYMENT.md#41-prerequisites), steps 1 to 3.

Then open PowerShell (Windows) or a terminal (Linux) and run:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

You should see a table that names your GPU. If you get an error, update the NVIDIA driver and Docker,
restart, and try again. Nothing below works until this does. This check uses a CUDA 12.8 image; the
`models` service needs a driver that supports CUDA 13.2, which any current driver does.

**Windows only, give Docker enough memory.** Create or edit `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
```

Then run `wsl --shutdown` and restart Docker Desktop.

## 3. Get the code

Clone the repository (replace `<repo-url>` with its address) into a folder named `local-rag`:

```powershell
git clone <repo-url> local-rag
cd local-rag
```

Optional: copy the settings file if you want to change ports or limits. Every setting has a working
default.

```powershell
Copy-Item .env.example .env      # Linux: cp .env.example .env
```

## 4. Download the models (once)

This is the biggest download: about 25 GB into a Docker volume.

```powershell
docker compose --profile setup run --rm fetch-models
```

It ends with a list that includes `gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf` and three folders under `hf/`.

Already have the files in a folder with `gguf\` and `hf\` sub-folders? Copy instead of downloading
(replace `<models-folder>` with that folder's path):

```powershell
$env:MODELS_SEED_DIR = "<models-folder>"
docker compose --profile setup run --rm fetch-models --from-dir /seed
```

On Linux, put the variable in front of the command:
`MODELS_SEED_DIR=/path/to/models docker compose --profile setup run --rm fetch-models --from-dir /seed`.

## 5. Build and start

Already ran the app before accounts existed? Follow
[DEPLOYMENT.md, section 4.10](DEPLOYMENT.md#410-upgrade-an-install-from-before-accounts) instead of
this step, so your documents, chats and memories become your account's.

Generate the keys the services use to trust each other, once:

```powershell
uv run --with cryptography scripts/init_secrets.py
```

It writes them into the `secrets/` folder, which Git ignores. Keep that folder with your backups
([DEPLOYMENT.md, section 6.2](DEPLOYMENT.md#62-back-up)) and never copy its contents into `.env`.

```powershell
docker compose up -d --build
```

The first build takes 10 to 20 minutes (it downloads Python packages and the document-parsing models).
Later starts take about a minute.

Check that everything is up:

```powershell
docker compose ps
```

All services should say `healthy`. `llm` and `models` take longest because they load the models onto
the GPU.

Create your account, the first admin. It asks for a password twice (at least 12 characters):

```powershell
docker compose run --rm auth auth-admin create-admin --username <you>
```

Usernames have 3 to 64 characters: lowercase letters, digits, `.`, `_` and `-`.

## 6. Use it

Open **http://localhost:8000** and sign in with the account from step 5. The dot at the top right turns
green with *all services ready*.

1. **Add documents.** Drag files onto *Add documents* (PDF, TXT, MD, XLSX, CSV, PNG, JPG, WEBP).
   Each file shows *queued*, then *processing* with a progress note, then *ready*. A text file takes
   seconds; a long PDF with many charts can take several minutes, because every figure is read by the
   vision model. No documents at hand? Try the files in
   [examples/documents](../examples/documents/README.md); its README lists questions with expected
   answers.
2. **Select documents.** Tick up to 10 ready documents. Only the selected documents are searched.
3. **Ask.** Type a question and press Enter. The answer streams in with citations like [1]. Click a
   citation to see the exact passage, table rows or figure it came from.
4. **Optional toggles.**
   - *Think*: the model reasons before answering. It is slower, and better for hard questions.
   - *Deep search*: the model researches step by step with tools (search, exact match, tables, SQL)
     and checks its evidence before answering. It is slower, and best for questions that span several
     documents or need calculations.
   - *Read all*: the model reads every section of the selected documents, takes notes on each part, then
     answers from all the notes. It is the slowest (about one step per few pages; the steps show how far it
     is), and best for questions about the documents as a whole: "list every limitation the paper states",
     "which of these documents mention X, and what does each say", "how many articles mention Y". Large
     spreadsheets are read through their column summaries, not row by row, so ask figures over their rows
     without it. If the selection is too large to read in one request, the answer says which parts were not
     read. *Read all* and *Deep search* exclude each other.
5. **Chats** (tab at the top left): every conversation is saved. Reopen one to continue where you left
   off; its document selection comes back too. *New chat* starts a fresh one. Follow-up questions
   ("and the West region?") understand the earlier turns, and long chats are summarized automatically
   so they keep working.
6. **Memory** (top right): things the app remembers about you in every chat, like your role or "answer
   with tables". It learns them when you mention them, or when you ask ("Remember that I prefer short
   answers"). Each item it saves or forgets appears above the answer with an *Undo* button. Open the
   panel to see, add or delete them. Memory shapes how answers are written, never what the documents
   say.
7. **Audit log** (top right): your own uploads, deletions, questions, memory changes and sign-ins, with
   the documents and sources used, numbered from #1. "Hash chain verified: N of your events, unaltered."
   means no record was altered.
8. **Your username** (top right) opens your account menu: *Change password*, *API tokens* (for scripts:
   section 7), *Users* (admins only: create and manage accounts) and *Sign out*.

Good to know:

- **Tables and spreadsheets** can be asked for totals, counts and filters ("total sales by region",
  "orders that mention battery overheating"); the app runs real SQL for those.
- **Python in Deep search** (off unless your operator enables it). With `COMPOSE_PROFILES=sandbox`,
  `RAG_DEEP_PYTHON=on` and `RAG_SANDBOX_URL=http://sandbox:8008` in `.env` (then `docker compose up -d`), and the
  isolation check of DEPLOYMENT section 4.16 passing, Deep search may also write and run Python on your selected
  documents' tables and text: statistics, trends, combining several spreadsheets. It runs in the separate `sandbox` container, which has
  no network and sees only the documents you selected for that question; its runs appear in the steps as
  `run_python`, and each result is a source you can open to read the code and its output. Charts are not shown
  ([DEPLOYMENT.md, section 4.16](DEPLOYMENT.md#416-update-python-for-deep-search-in-a-locked-down-sandbox)).
- **Updated a file?** Upload it again with the same name. The new version replaces the old one as soon
  as it is ready, and the old one stays searchable until then. A ticked file stays ticked on its new
  version. A file with another name is a separate document, so delete the old one yourself.
- **Reindex** (the circular-arrow button next to a document) re-processes it after a settings change.
  The document stays usable meanwhile. The trash-can button deletes a document.
- **Other people.** As an admin, open *Users* from your account menu and create an account for each
  person. The temporary password is shown once; they choose their own at the first sign-in. Each
  person sees only their own documents, chats, memories and audit log, and admins cannot read them
  either ([DEPLOYMENT.md, section 9](DEPLOYMENT.md#9-users-and-accounts)).

## 7. Check that it works (optional)

The test runs as you, with an API token. Create one under your username > *API tokens*: a name (for
example `smoke`), how many days it lasts, and your password. It is shown once; copy it into your
shell, never into `.env`. Then generate sample files of every type and run the end-to-end test:

```powershell
$env:RAG_TOKEN = "rag_pat_..."     # Linux: export RAG_TOKEN=rag_pat_...
uv run --with fpdf2 --with openpyxl --with pillow scripts/make_samples.py samples
uv run --with httpx scripts/smoke_test.py --samples samples
uv run --with httpx scripts/smoke_test.py --samples samples --deep
```

Each prints `11/11 passed` when everything is healthy. The test files are named `smoke-...` (for
example `smoke-aurora_report.pdf`), so they never replace your own documents, and the test deletes
them when it finishes, even after a failure. Add `--keep` to leave them in. The app itself never adds
documents: only what you upload appears in the library. Revoke the token in the same panel when you no
longer need it. `scripts/multiuser_smoke.py` checks that accounts cannot see each other's data
([DEPLOYMENT.md, section 9.7](DEPLOYMENT.md#97-isolation-test)).

## 8. Measure retrieval quality (optional)

```powershell
$env:EVAL_TOKEN = "rag_pat_..."    # an API token (section 7); Linux: export EVAL_TOKEN=rag_pat_...
docker compose --profile eval run --rm eval generate --n 120
docker compose --profile eval run --rm eval retrieval
docker compose --profile eval run --rm eval answers --limit 30
```

The evaluation reads documents and asks questions as the token's user, so a golden set belongs to the
user who owns its documents. A set from before accounts keeps working with a token of the admin who
adopted those documents; any other user generates their own. `generate` writes
`eval/datasets/golden.jsonl` from your indexed documents; open it and fix or delete weak questions
before trusting the numbers. Reports (MRR, Recall, nDCG with confidence intervals,
answer faithfulness and correctness) land in `eval/reports/`.

### Compare Deep search variants (benchmark)

The benchmark asks fixed questions with checked answers and scores the replies without a judge model: numbers
within a tolerance, required facts, citations of the right documents
([ARCHITECTURE section 9](ARCHITECTURE.md#9-evaluation)). Its local set is the 65 questions of
[examples/documents/README.md](../examples/documents/README.md) plus two on the smoke samples; the HotpotQA and
MuSiQue subsets are the held-out check, because the local hard cases shaped both agents' shared rules. Run it as a
separate account, so its documents never mix with anyone's: an admin creates the account
([DEPLOYMENT section 9](DEPLOYMENT.md#9-users-and-accounts)), you sign in as it once and create an API token. From
the repository folder:

```powershell
$env:RAG_TOKEN = "rag_pat_..."     # the benchmark account's token; Linux: export RAG_TOKEN=rag_pat_...
uv run --with fpdf2 --with openpyxl --with pillow scripts/make_samples.py samples    # the two smoke-* files
uv run --project eval python -m rag_eval fetch-public                               # once: HotpotQA and MuSiQue
uv run --project eval python -m rag_eval bench-setup --set local,hotpotqa,musique   # once: upload and wait
uv run --project eval python -m rag_eval bench --variants langgraph,langgraph+think,deepagents,deepagents+think,deepagents_stock
```

- A variant is an agent name, sent as the chat request's `agent` field (`langgraph`, `deepagents`,
  `deepagents_stock`), with optional suffixes in any order: `+think` for thinking in deep search's plan and verify
  steps (`deep_think`), `+fanout` for researchers on reserved budgets (`fanout`; deepagents only among the core
  arms, a 422 for the others), `+calls<N>` for a research budget of N calls (`research_calls`, 1 to 30) and
  `+python` for the `run_python` tool (`python`; the stack needs `RAG_DEEP_PYTHON=request`, `RAG_SANDBOX_URL` and
  the compose profile `sandbox`, DEPLOYMENT section 4.16), for example
  `deepagents+fanout,deepagents+calls20,deepagents+python`. A `rag` that has only one implementation ignores these
  fields.
- The run is fair by construction: one run, interleaved, in a randomized order (`--order-seed`, default 7): each
  repeat shuffles the questions and each question runs every variant back to back in a shuffled order, so GPU
  load, cache warmth and drift reach every variant alike. Every variant gets the same question text and documents,
  answer thinking off (`--think` is refused here) and the same rag container. Every report records the rag image
  id, the git commit (`--app-commit` adds the commit the app was built from), the model and `LLM_CTX`.
- `--variants` runs `local,hotpotqa,musique` unless you pass `--set`; `--repeats` (default 3, or `EVAL_REPEATS`),
  `--tags`, `--limit` and `--seed` (the public subset) narrow it. It does not create chats. Each run prints one line:
  where it is (question i of N), the answer, the pass rate so far per variant and the time left.
- It writes `eval/reports/bench-<variant>-<time>.json` and `.md` per variant and `bench-combined-<time>.md`, whose
  first lines name the variant that scores higher on correctness and whether the difference is significant, for
  all questions, the local set and the held-out sets. Ctrl+C still writes the reports, marked incomplete.
- Time: 5 variants x 127 questions x 3 repeats is 1,905 deep runs, about 11 to 16 hours at 20 to 30 s each
  (`--repeats 2`: 7 to 11 hours). The first `bench-setup` also waits for the 10 PDFs (the vision model reads every
  figure) and 600 small HotpotQA and MuSiQue documents.
- `--concurrency 2` keeps two requests in flight, so both slots of the llama server work and a run takes roughly
  half the time (default 1: one request at a time). Requests are still sent in the seeded order
  and every variant runs under the same load; records arrive as they finish and carry `order` (their place in the
  run order) and `in_flight` (the other requests running when they were sent), and the reports keep the run order.
  The time left in the progress line is the time per finished run so far, so it counts both slots. The reports
  say that latencies were measured under load: compare latencies only between runs with the same concurrency
  (`compare` notes a difference). Fan-out variants use both slots themselves, so at 2 their researchers queue
  behind the other request in flight; a variant never runs alongside itself on the same question, so a fan-out
  variant loads the others slightly more than they load it. More than `RAG_LLM_PARALLEL` (2) only adds queueing.
- `--request-timeout` (default 900 s x `--concurrency`) is how long a request may wait for the next bytes of its
  reply before it is recorded as failed. The gateway also cuts a reply after 900 s without bytes, and rag's own
  model calls give up after 900 s (600 s for a researcher), queue time included. A request that times out or
  fails on either side is an ERROR run: the progress line shows it, it scores 0 and it counts under failed runs;
  nothing is retried and nothing is dropped. After a Ctrl+C at concurrency 2 the requests still in flight are
  left out of the reports.
- Two reports, of any kind, compare question by question:
  `uv run --project eval python -m rag_eval compare eval/reports/bench-langgraph-<time>.json eval/reports/bench-deepagents-<time>.json`
  gives, overall and per tag, the questions each side scores higher on and the ties, the difference with its
  interval and p-value, latency median
  and p90, tool errors, notices and limit hits, and a one-line verdict. A single variant runs with
  `bench --agent deepagents [--deep-think]`.
- `--set hard` runs the 33 harder questions (multi-step, cross-document, distractor documents, false premises) on
  the same documents as the local set, for when the local set cannot tell the variants apart:
  `bench --set hard --variants langgraph,deepagents+think --repeats 2`.
- `--set global` runs 17 questions that need whole documents (every item of a list, every place something comes
  up, counts across sections), for *Read all* against Deep search. Run it once in each mode and compare the two
  reports; `--mode exhaustive` takes no `--variants` (rag refuses the deep-search fields in that mode):

  ```powershell
  uv run --project eval python -m rag_eval bench-setup --set global      # eight papers and the Constitution
  uv run --project eval python -m rag_eval bench --set global --mode exhaustive --variant exhaustive --repeats 2
  uv run --project eval python -m rag_eval bench --set global --mode deep --agent langgraph --variant deep --repeats 2
  uv run --project eval python -m rag_eval compare eval/reports/bench-exhaustive-<time>.json eval/reports/bench-deep-<time>.json
  ```

  Each answer's evidence (page and exact quote for every fact) is in `eval/benchmarks/global_evidence/`; the quote
  check against the PDFs runs with `uv run --project eval --with pypdf python -m pytest eval/tests/test_global_set.py`.
- `--set forms` runs 12 questions where names are written in more than one form (a short form or code, an
  official or alternate name, or a name inside a longer different name), on the local set's data files:
  `bench --set forms --variants langgraph,deepagents+think --repeats 2`. The script that recomputes each answer
  is in `eval/benchmarks/forms_evidence/`.
- `--set totals` runs 12 questions where a total or subtotal row changes the answer (a total such as "All
  causes" or "United States" counted with the rows it sums, or a total row the answer needs), on the local
  set's data files: `bench --set totals --variants langgraph,deepagents+think --repeats 2`. The script that
  recomputes each answer is in `eval/benchmarks/totals_evidence/`.
- `bench-clean --set local,hotpotqa,musique` lists the benchmark documents; add `--yes` to delete them.

## 9. Stop, start, update, reset

| Task | Command |
|---|---|
| Stop (keeps everything) | `docker compose stop` |
| Start again | `docker compose start` |
| Update after pulling new code | `git pull` then `docker compose up -d --build` |
| Upgrade an install from before accounts | [DEPLOYMENT.md, section 4.10](DEPLOYMENT.md#410-upgrade-an-install-from-before-accounts) |
| Remove containers, keep data and models | `docker compose down` |
| Delete everything, including documents, accounts and downloaded models | `docker compose down -v` |

## 10. Troubleshooting

| Symptom | What to do |
|---|---|
| `could not select device driver "nvidia"` | Docker cannot see the GPU: repeat step 2 |
| `llm` or `models` never becomes healthy | `docker compose logs llm` or `docker compose logs models`. A message about a missing `.gguf` means step 4 was skipped. Out-of-memory means something else uses the GPU: close games or other AI apps. A message that the CUDA driver is too old (mostly on Linux) means the NVIDIA driver must be updated to one that supports CUDA 13.2 |
| The page does not load | Something else uses port 8000: set `GATEWAY_PORT=8010` in `.env`, run `docker compose up -d`, open http://localhost:8010 |
| A document stays *queued* | One document is processed at a time; a large PDF with many figures can take a while. Check `docker compose logs -f ingest-worker` |
| A document shows *failed* | The reason is shown in red under the file name. Click the Reindex (circular-arrow) button to retry; files that are password-protected or corrupt cannot be read |
| The page shows nothing at all | Open it as http://localhost:8000 or http://127.0.0.1:8000. The app answers no other address, not even this computer's name or network address, unless `PUBLIC_ORIGIN` names it ([DEPLOYMENT.md, section 5.3](DEPLOYMENT.md#53-tls-reverse-proxy-the-supported-setup)) |
| `docker compose up` stops on a missing file under `./secrets/` | Run `uv run --with cryptography scripts/init_secrets.py` (step 5), then `docker compose up -d` again |
| `sql` does not start | It needs Docker Desktop 4.29 or newer (Docker Engine 26 or newer). Update Docker |
| The sign-in page asks you to run `create-admin` | No account exists yet: run the command from step 5 |
| "Too many attempts" at sign-in | 5 wrong passwords in 15 minutes lock that username for 15 minutes or more. Wait, or have an admin clear it in *Users* |
| A forgotten password | An admin resets it in *Users* and gives you a temporary one. If no other admin can, see [DEPLOYMENT.md, section 9.6](DEPLOYMENT.md#96-recovery-from-the-server-auth-admin) |
| A notice says an administrator reset your password | It stays until you click *Dismiss*. If you did not ask for a reset, tell your administrator, and check *Audit log* and your API tokens |
| The smoke test or evaluation gets `401` | The token is missing, mistyped, expired or revoked (changing your password revokes every token). Create a new one |
| Answers are slow | Normal while documents are being processed (they share the GPU). *Think* and *Deep search* are slower by design |
| "documents not ready yet" | Wait until the selected documents show *ready* |
| Anything else | `docker compose logs --tail 50 <service>`: every service logs one JSON line per request with a request ID you can follow across services |
