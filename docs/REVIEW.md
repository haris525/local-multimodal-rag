# Repository review, November 14, 2026

Scope: ingestion/job ownership, cancellation, version replacement, model loading, service boundaries,
account/session handling, SQL execution, chat persistence, and the Docker/first-install path. This is a
targeted engineering review, not a guarantee that every possible race or security defect is absent.

The goal is a private, local document assistant: users upload documents, select their evidence, and
receive cited answers through hybrid search, structured SQL, optional research tools, and saved chats.
Per-user isolation, durable ingestion, and keeping resource use within a workstation are central.

## Confirmed bugs fixed

All five regressions below were observed failing before their respective fixes, then passing afterward.

| Priority | Trigger and impact | Fix and regression |
|---|---|---|
| P1 | An upload releases its placement lock while another upload is waiting. `asyncio.Lock` is temporarily unlocked before its waiter resumes, so cleanup drops the shared entry. A third upload creates another lock, allowing two placements for the same document. This can race replacement/deletion of the same source file. | [`api.py`](../services/ingest/src/ingest/api.py): count the holder and waiters; retire the entry only when nobody references it. `test_placement_waiters_keep_the_same_lock`. This is still one-process coordination. |
| P1 | Shutdown cancels a pipeline awaiting a thread and immediately releases the registry claim. Cancellation does not stop that thread. A second worker can claim the same document while the old thread still writes tables/chunks. | [`worker.py`](../services/ingest/src/ingest/worker.py): stop parsing, wait for background writes, then release; keep renewing the lease during that wait. `test_shutdown_keeps_the_claim_until_background_writes_finish`. |
| P1 | A thread-pool job is queued behind busy threads. Canceling its awaiting coroutine cancels the queued executor Future before its wrapper runs, so its completion event is never set. `settle()` then waits indefinitely, stalling cleanup/shutdown. | [`pipeline.py`](../services/ingest/src/ingest/pipeline.py): shield queued/running executor work, retain completion tracking, and retrieve late failures. `test_cancelling_a_queued_thread_cannot_make_settle_wait_forever`. |
| P1 | The first job is claimed before the scheduled startup sweeper runs. Its `mine=True` recovery mistakes that fresh claim for work held by the previous process, requeues it and invalidates its run; a retry can exhaust its attempt limit. This was observed during the real Mac launch and reproduced with a named worker in a regression. | [`worker.py`](../services/ingest/src/ingest/worker.py): await own-holder cleanup before any new claims; the background sweeper always honors live leases. `test_starting_the_background_sweeper_does_not_reclaim_a_fresh_job`. |
| P2 | Two workers claim different content versions of the same owner's filename. The shorter, newer version finishes first; the older one later becomes ready beside it because supersession only looks backward in upload order. | [`registry.py`](../services/ingest/src/ingest/registry.py): serialize queued/processing versions of one owner's filename in the claim transaction; other documents remain independently claimable. `test_workers_serialize_versions_but_can_process_other_documents`. |

The five regression tests are in [`test_lifecycle.py`](../services/ingest/tests/test_lifecycle.py).
No schema migration or deletion of existing documents is required by these fixes.

A further streaming UI race was reproduced during the loader update: if a token schedules a render,
then an error or interrupted stream appends its failure message before that render executes, the queued
render erases the failure and **Retry** button. Synchronous final/error renders now cancel pending frames
and flush partial text before appending the failure. The UI regression waits for the delayed frame;
it failed before the fix and passes afterward.

The setup work also corrects cache export from a parent directory named `.cache`, makes copied model
files publish atomically, excludes host models/build artifacts from Docker's context, and adds bounded
retries for the OCR provider's observed download timeout. The Mac override also explicitly caps audit
and saved-chat containers, which otherwise inherited no memory cap. Cache export/interrupted-copy behavior has
two additional regression tests in [`test_fetch_models.py`](../scripts/tests/test_fetch_models.py).

## Reindex consistency fixes

The former P5 publication defect and external writes after lease expiry are fixed with immutable run
generations. A run writes its DuckDB file, JSONL chunks, figures and Qdrant point IDs into its own UUID
namespace. Only `Registry.complete()` publishes that generation, together with its counts, warnings and
table schemas, in one transaction that checks the run still owns the document. A failed or late run can
leave private artifacts; it cannot replace the previous published artifacts.

Each question captures the published generation of every selected document once. Dense/BM25/image
search, grep, neighbour reads, exhaustive reads, SQL, Python input downloads and figure URLs preserve
that selection. Owner checks remain mandatory. Legacy documents retain their existing paths and IDs;
their first new reindex creates a generation without removing the legacy copy.

Deletion captures its source lifetime and generations under the matching deletion claim, including
when a sweeper's earlier document listing is stale. Source uploads also have unique
storage IDs, so a delayed cleanup cannot remove a later re-upload of the same bytes. The regression
tests hold an uncancellable file writer and a Qdrant batch until after a replacement run publishes,
and overlap old cleanup with a new upload. See
[`test_generations.py`](../services/ingest/tests/test_generations.py),
[`test_snapshots.py`](../services/rag/tests/test_snapshots.py) and the versioned SQL/API tests.

Published generations are retained until document/account deletion so an in-flight question never needs
an expiring reader lease. Disk usage therefore grows with reindexing; unpublished leftovers of failed or
crashed runs are also removed at deletion. Update: schema v6 adds the clean-up of old generations (a
replaced one after `INGEST_GENERATION_RETAIN_H`, a failed run's at once; see ARCHITECTURE.md). Monitor
the data volume and Qdrant disk space. Keep the supported single ingest-api process and single worker;
these regressions do not certify a horizontally replicated deployment.

Paging and grep now stream JSONL records and retain only the page/top hits. The former 64-file cache
retained 73.6 MiB in a probe that requested only 100 rows from each of eight 10,000-row synthetic files.
The same probe after the change peaked at about 0.1 MiB of traced Python allocations for paging and
grep. These are allocation measurements for that probe, not whole-service RSS figures. Reproduce the
broader live workload using [WORKLOAD_VALIDATION.md](WORKLOAD_VALIDATION.md).

The sustained workload then reproduced a parser OOM at the Mac worker's existing 4 GiB cap after
successive large files. Model requests were batched, but the pipeline retained all returned vectors
until indexing. It now embeds and upserts one bounded batch at a time. PDF stage queues are capped
at eight items, and parsers exceeding 1,024 MiB idle RSS are recycled before embedding/the next file,
including after failed parses. The cache threshold is configurable and does not limit a single parse's
peak. A separate regression also fixes an idle parser death poisoning the process pool for every
following document: errors from `submit()` now clear the broken pool just like errors while awaiting
a parse. See [`test_isolation.py`](../services/ingest/tests/test_isolation.py).

Snapshot context is bound while advancing or closing the answer generator and reset before yielding
an event. A regression finalizes a stream in another task, proving it can still save the interrupted
answer without resetting a context token created elsewhere. These fixes live in shared application
code and the Linux containers used on both Windows and macOS; this follow-up was run on macOS,
not on a Windows host.

The final live stream check also exposed eager consumption in the shared command-line SSE reader:
it expanded every response line into a list before yielding the first event. Parsing now consumes
lines lazily; a regression makes any read beyond the first complete event fail immediately.

## Mac support and model inventory

The inspected machine has Apple Silicon and 64 GB unified memory. Docker Desktop's Linux VM was
allocated about 16 GB. Other projects were running; the review did not stop, remove, or reconfigure them.

The cache contained:

| Model | Observed state before setup | Compatibility |
|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` | three weight shards, about 16.05 GB total, no broken shard symlinks | MLX format; cannot be loaded by this repo's llama.cpp GGUF runtime |
| `Qwen/Qwen3.8-27B` | config present, no weight files | not a complete base-model install |
| `mlx-community/Qwen3.8-27B-MTP-4bit` | about 0.24 GB of weights | draft component, not a standalone 27B answer model |
| `Qwen/Qwen3-Embedding-0.6B` | one weight file, about 1.19 GB | matches the text retrieval model; setup can reuse it |
| `Qwen3-VL-Embedding-2B`, `Qwen3-Reranker-0.6B` | not found in the inspected Hugging Face cache | required downloads for full retrieval |

These checks establish file presence/sizes, not checksums or successful inference. Other Qwen models
were cached, including larger embedders; those are not direct replacements for the configured models.

The first real MPS launch also exposed an invalid allocator-watermark pairing: PyTorch's default low
watermark exceeded the smaller high cap. The launcher now derives a lower cleanup threshold, with
checks covering both the default cap and a customized one. All three models then loaded successfully.

The Mac port adds a pinned native llama.cpp Metal build, native MPS embedding/reranking, and two fixed
upstream Docker proxies. Application/data services remain containerized, with explicit Mac memory/CPU
limits and a separate Compose project. The CUDA dependency index is retained on Windows/Linux;
macOS uses the locked PyPI builds. See [`MACOS.md`](MACOS.md) for the full novice setup.

Native inference is outside Docker's memory and network sandbox. Limits on containers do not cap those
processes. The guide states the changed isolation boundary and distinguishes estimates from measured
memory requirements. The NVIDIA deployment remains the path with container-isolated inference.

## ReAct tools and Python analysis

Deep search already selects among hybrid search, literal matching, passage reads, table/schema tools,
column profiling, SQL and optional Python. The local test deployment now enables its existing
`run_python` integration in a separate Linux sandbox, capped at 2 GiB and two CPUs. This is stored in
this machine's ignored `.env`; the repository-wide default stays off.

The actual agent tool registry was inspected with a synthetic table descriptor: all nine tools,
including `run_python`, were offered. A synthetic comparison inside the live sandbox checked DuckDB,
Polars and pandas on 1,000/100,000/1,000,000 rows. All 36 engine/workload/size results agreed.
The measured guidance is early SQL filtering/aggregation, Polars for suitable dataframe operations,
and pandas for compatibility or small results; the Python tool description now says this explicitly.
See [analysis performance](ANALYSIS_PERFORMANCE.md) for timings, memory, caveats, and reproducible commands.
These data-operation measurements do not test model reasoning or semantic search quality.

A subsequent Deep search run guessed an original CSV path in `run_python`, which failed because the
sandbox receives extracted tables and text rather than original uploads. Each Python tool now lists
its selected documents' exact input paths before code generation and explicitly directs the model
to the manifest and read-only DuckDB-to-pandas loading. Missing-file errors include recovery guidance.
Two regressions check that the advertised paths match the actual manifest without fetching data early,
and that a missing-file failure remains an error with actionable feedback rather than citable evidence.
A synthetic live check using the deployed tool description, Qwen model and Docker sandbox then
generated and executed pandas code successfully on its first Python attempt. Both regional groups'
counts, sums and means matched the known 120-row fixture, and the tool registered a Python source.
This focused check offered only `run_python`; it does not establish full-agent tool-selection quality.

The live sandbox containment check passed **89 checks**, with the Linux-host UID check skipped on
Docker Desktop (90 checks accounted for). It exercised network restrictions, process/file cleanup,
run timeouts, memory-limit killing and successful execution afterward. The other projects were untouched.

## Validation

The generation/memory follow-up passed **1,740 automated tests, with 1 skipped**, across ingestion,
RAG, SQL, auth, shared code, threads, audit, scripts, UI and evaluation. The real browser Mermaid check
also passed. Standard Mac Docker builds and the offline parser-model warmup passed; the updated
process-memory dependency was already pinned transitively, and no model weights/version changed.
The final live gateway check saved a completed answer with its pinned source generation, then closed
a second answer after its first token and verified the partial answer was saved as interrupted.
Its temporary account and documents were removed afterward.
The 14.7-minute live workload passed, including repeated PDF parsing, concurrent reindex/search and
old/new pandas inputs, without new OOMs or service restarts. Its measurements and limits are recorded
in [the benchmark report](benchmarks/reindex-memory-mac-2026-11-14.md); reproduce it using
[WORKLOAD_VALIDATION.md](WORKLOAD_VALIDATION.md).

The original Mac support pass totaled **1,779 passed and 2 skipped**, including Linux sandbox and model
queue checks that were not repeated for the generation follow-up. Those earlier completed checks:

- Ingestion: 654 tests, including all five new race regressions.
- Retrieval/chat: 336 tests; authentication: 143; shared library: 140 passed, 1 skipped.
- Saved chats/memory: 22; SQL: 17; audit: 19; model queue: 3.
- Script checks: 82; browser/UI checks: 36; evaluation/benchmarks: 256 passed, 1 skipped.
- Optional Python sandbox: 71 tests passed in its Linux Docker build target. A direct macOS run
  produced 28 failures, 42 passes, 1 skip because the configured process resource limits could not be
  applied. This is not a supported native Mac runtime; the Mac deployment keeps this service in Linux.
- Both Compose files merge successfully: Mac services have no NVIDIA device reservations; LLM and RAG
  use the same 32,768-token total context and the smaller Mac prompt limits.
- The pinned llama.cpp revision built successfully for Darwin ARM64 with Metal.
- The locked PyTorch 2.15.0 Mac build reported MPS available and executed a small Metal tensor operation.
- The actual cached Qwen3-Embedding-0.6B model loaded on MPS and returned a finite `(1, 1024)` embedding.
- A temporary Linux container reached a temporary macOS server bound only to 127.0.0.1 via
  `host.docker.internal`; the test server/container were then stopped.
- The actual inference proxy image forwarded a request to Mac loopback under its 64 MiB cap and used
  about 10 MiB at idle during that check. This is proxy memory, not model-server memory.
- The ARM64 ingestion image completed its real PDF/OCR warm-up with Hugging Face offline mode enabled.
- All default Mac application images built successfully (inference proxies, ingestion, RAG, SQL,
  accounts, saved chats, audit, and gateway).
- The complete `python3 scripts/macos.py setup` command finished, reusing completed files and keys.
- Real Qwen text generation returned `4` for `2 + 2`; the vision model read the sample sign as 27 degrees C.
- All three retrieval models loaded on MPS. Text and image/query vectors were finite and approximately
  unit length (1,024 and 2,048 dimensions); reranking assigned the relevant passage 0.9999 and the
  unrelated passage about 0.00006.
- The full Mac stack, including the optional Python sandbox, started healthy. An admin
  account verified browser-style login through the gateway. Credentials and local account details
  are not included in repository files.
- Six synthetic sample formats ingested successfully. All **11/11** cited-answer smoke questions
  passed: PDF facts/tables/chart/OCR, text, Markdown, spreadsheet SQL and semantic filtering, CSV,
  and an image. The helper removed its six sample uploads afterward.
- A real Deep search request used `run_python` and `scipy.stats.linregress` on all 120 rows of the
  analysis fixture, reported slope **3**, intercept **7**, Pearson **r = 1**, and cited Python evidence.
  The full request took about **402 s** while other smoke questions were also running. This verifies
  tool execution and citation; it does not establish generally correct model reasoning.
- SciPy and statsmodels also reproduced the known regression in a direct sandbox check.
- Sample question latency ranged from roughly **22-275 s** in this run, including concurrent agent
  work. Model reasoning, prefill and generation dominate the millisecond-scale dataframe operations.
- The final rollout rebuilt all Mac images through the repository's normal Dockerfiles, including
  the real offline OCR warm-up, then recreated the application containers. All 12 services became
  healthy; the running UI files match the reviewed source. Existing model files and data volumes
  were retained.

The application and both native model servers are left running for local testing. Other projects were
not stopped or reconfigured. Container limits were verified, including 512 MiB for audit/saved chats
and 2 GiB for the optional Python sandbox. During sample ingestion the worker used about 2 GiB of its
4 GiB cap in an observed snapshot. Native model allocations sit outside these container limits;
process RSS alone does not capture all MPS GPU allocations.

An interactive Read all request exposed poor waiting feedback: it finished non-streaming reading
calls before the final answer stream began. One reading call reached 1,500 output tokens and was
automatically repeated with more room; generation was about six tokens/second during this request.
The UI now shows a spinner, request elapsed time, mode-specific waiting guidance and stage labels,
then renders answer tokens as they arrive. It stops the loader on completion, errors and interrupted
streams, and never animates saved answers. Three additional UI tests cover these behaviors, including
rendering tokens before the server closes its stream. This UI change does not improve model throughput.
After changing the Mac's power mode, the same live request's recent generation rate rose
to roughly 14-15 tokens/second. This is an observed rate change, not a controlled benchmark.

Mermaid 12.3.0 is bundled locally, with its package integrity verified and licenses retained. Diagrams
render after streaming finishes and when saved answers are opened. Lazy-loaded opaque-origin frames
isolate rendering from the chat and session cookies. Only the diagram document allows generated
inline styles; its CSP blocks network connections, images, fonts, forms and nested frames. Mermaid
runs in strict mode with protected settings and bounded text/edges; output is sanitized again.
The chat/login CSP is unchanged. A real Chromium check rendered the complete bill-passing fixture,
kept its legend/citation, switched themes, rejected cookie access and hostile markup, and showed
the original code for invalid syntax. Five focused diagram tests accompany the existing UI tests.

A deployment exposed a mixed-cache failure: chat code called a separately loaded diagram helper that
was absent from the page, blocking saved-chat display. The parent renderer now ships inside the chat
script, and rendering failures fall back to code without interrupting chat loading or completion.
Static UI files require cache revalidation; the former helper URL remains available for cached pages.
Two regressions cover the missing helper and answer completion after a rendering error. The real
browser check also opens saved diagrams with the former helper explicitly absent.

**Limits:** broad model quality, horizontal scaling and arbitrary user-document performance remain
unestablished. Generation isolation has targeted regression coverage; the finite synthetic workload
is described in [WORKLOAD_VALIDATION.md](WORKLOAD_VALIDATION.md). Passing those checks does not establish
general correctness or memory leak freedom.

## Public repository audit

Before the Mac-support check-in, Gitleaks 8.30.1 scanned the proposed publication tree and all five existing commits.
Its six matches were individually reviewed: a public-key helper name, a public upstream GPG fingerprint
in a test fixture, three CLI-test arguments and a synthetic JWT used to test redaction. No operational
credential was identified; no broad scanner exclusions were added. A separate scan covered personal
paths/identifiers and spreadsheet archive contents. Local account details were removed from these notes.
Ignored local settings, keys, weights, databases, logs and runtime files are outside the publication
tree. These checks do not guarantee that every possible secret format can be detected.
The later Mermaid bundle scan added one match in upstream JavaScript identifiers; its bytes match
the integrity-verified public npm distribution. It is also a reviewed false positive.

For a new install or restart, follow [MACOS.md](MACOS.md). Completed models and keys can be reused;
never delete `secrets/` or volumes to work around a startup/download error.
