# Changelog

Newest first. Each entry says under "Data changes" whether it changes what is stored in the Docker volumes
and what to do about it (see [DEPLOYMENT.md, section 4.8](docs/DEPLOYMENT.md#48-updates)).

## Unreleased: consistent reindexing and search tasks

- Reindex into immutable run generations; publish files, point selection and metadata through one
  registry pointer switch. Questions preserve their selected generations across every retrieval tool,
  SQL, Python input download and figure read. Stale workers cannot overwrite a published generation.
- Give each upload lifetime a unique source directory and scope deletion to captured generations.
- Stream chunk pages and retain only top grep hits instead of caching entire files in memory.
- Embed and publish vectors one bounded batch at a time; cap PDF inter-stage buffers and recycle
  idle parser processes above a configurable memory budget after a sustained workload exposed an OOM.
- Add a reproducible isolated large/messy document workload and memory measurements.
- Keep snapshot context scoped to each streamed operation so disconnect cleanup can save partial
  answers even when the stream is finalized in another task.
- Yield parsed chat events as they arrive in command-line clients instead of buffering the full stream.
- Remove old generations: after a reindex the replaced generation stays readable for
  `INGEST_GENERATION_RETAIN_H` hours (24 by default) so a question that started on it can finish, then
  the worker's sweep removes its points, tables, chunk file and figures. A failed run removes what it
  wrote at once; the leftovers of a run that stopped or was taken back go after four run leases (at
  least 10 minutes). The current and running generations, deleting documents and the uploaded file are
  never collected.
- Files picked while an upload is being sent join it: sent one after another in the order picked,
  counted together ("Uploading 2 of 3") and listed together. Before, the page's message showed only
  the upload that finished last, and the first file looked gone while it was still on its way.
- Keep PDF text in the top and bottom margins unless it repeats on another page (digits ignored), is a
  page number, or has fewer than 3 words. Docling marks margin text as page headers and footers, and all
  of it was dropped, so content running into a margin (such as the last line of a document) was never
  indexed. `INDEX_VERSION` 3 reindexes every ready document once at the worker's next start.
- Add a plan, not implemented, for video, audio and more file formats:
  [docs/design/video-and-new-formats.md](docs/design/video-and-new-formats.md).
- Show the MMR formula in ARCHITECTURE.md section 7.4 as math on github.com. GitHub leaves a ` ```math `
  block inside a list as a code block when the list also holds inline formulas, so the block now sits
  between items 4 and 5 and the list goes on at 5.
- Search tasks: the instruction the question is embedded with before the meaning search is chosen per question
  ([design](docs/design/search-tasks.md), ARCHITECTURE section 7.2). A *Search task* picker sits next to *Think*,
  *Deep search* and *Read all*, with three built-in tasks (`services/rag/src/rag/search_tasks.py`):
  - **General**, the default: `RAG_QUERY_INSTRUCTION`, now `Given a question about the user's documents,
    retrieve passages with the facts needed to answer it`. The earlier `Given a question, retrieve passages that
    answer it` measured below no instruction at all.
  - **Multi-step research**: Qwen's multi-hop instruction.
  - **Exact lookup**: no instruction, the bare question.

  Each user can add their own in the *Search tasks* panel: private, deleted with the account, at most
  `THREADS_MAX_SEARCH_TASKS` (50), never holding a credential value. *Retry* and reopened chats use the task an
  answer was asked with. How General was chosen:
  [benchmarks/embedding-models-2026-11-15.md](docs/benchmarks/embedding-models-2026-11-15.md).
- `models` embeds a query with the instruction its request sends and answers with the one it applied; rag
  refuses a vector made without the one it sent, so an outdated `models` is an error, not a silent fallback. A
  refused request (422) from any service names the field and the reason without echoing the value sent.
- New command `rag-eval embed-bench` (`eval/src/rag_eval/embedbench/`): compares text embedders on recall@k,
  retrieval only, with the app's fusion and reranker, over the example documents' own chunks (gold chunks in
  `eval/benchmarks/local_passages.jsonl`), HotpotQA and MuSiQue. The larger Qwen3 embedders (4B, 8B, VL-8B) add
  at most about 3 points of recall@10 after reranking and need 6 to 14 GiB more GPU memory, so the 0.6B stays
  ([findings](docs/benchmarks/embedding-models-2026-11-15.md)). `bench --search-task` sends a search task with
  every question. Its chunk export holds an account's document text, so `.gitignore` keeps `chunks*.jsonl` and
  `candidates*.txt` out of git.

**Data changes:** registry schema v5 adds `generation`, `storage_id` and a generation catalog. Existing
documents keep their paths/point IDs and remain readable without reindexing. New derived data uses
generation directories/IDs. Schema v6 adds `retired_at` and `collected_at` to the catalog; generations
replaced before it count as retired from its first start. Index version 3 reindexes
every ready document once (DEPLOYMENT.md section 4.21). Use the coordinated upgrade in [Deployment](docs/DEPLOYMENT.md#consistent-reindex-upgrade);
rolling back after new generations are published requires the matching data backup. Threads schema v3 adds a
`search_tasks` table, and General's instruction changes on the question's side only, so neither needs a
reindex (DEPLOYMENT.md section 4.22).

## 0.1.0: initial public release

The first public version. Data changes: none (a new install). Its features:

- **Accounts and per-user data.** Sign-in with accounts an admin creates (argon2id passwords, lockout,
  sessions, API tokens, the `auth-admin` command line). Every document, index entry, file, chat, memory and
  audit event belongs to one account; admins manage accounts but cannot read anyone's content. Deleting an
  account removes its data and erases its audit details.
- **Ingestion.** PDFs with tables and figures, Excel and CSV files, text, Markdown and images, parsed in an
  isolated child process by a separate worker with a durable queue. Figures and charts are described by the
  local vision model, every sheet and PDF table becomes a SQL table with column profiles, and a new upload with
  the same name replaces the older version. Model replies are cached per account.
- **Retrieval.** Dense vectors, native BM25, exact-match grep and image vectors, fused with reciprocal rank
  fusion, reranked, then per-document coverage and maximal marginal relevance over the selected documents.
- **Structured data.** A planner chooses SQL, SQL-filtered search or plain search; SQL runs in a read-only
  sandbox. Schema cards and profiles mark total rows and summarize text values written in several forms.
- **Answers.** Streamed answers with numbered citations, an optional thinking step, and notices for anything
  that was cut, skipped or failed.
- **Deep search.** A LangGraph agent (plan, use tools, verify) by default, and two implementations on
  LangChain's `deepagents` package (tuned, and out of the box) behind `RAG_DEEP_AGENT`. Optional fan-out to
  researchers with reserved budgets, a research budget per request, and optional Python over the selected
  documents in a locked-down sandbox with no network (off by default).
- **Read all (exhaustive mode).** Reads every section of the selected documents, takes cited notes and
  answers from all of them.
- **Chats and memory.** Saved chats, follow-up questions, summaries of long chats and long-term memories that
  the user can view and delete.
- **Operations and security.** Microservices behind one localhost port (nginx), Ed25519-signed identities
  between services, internal networks, hardened containers, a hash-chained and encrypted audit log per user,
  and file-based secrets.
- **Evaluation.** Retrieval metrics (MRR, Recall, nDCG) with bootstrap intervals, and a benchmark without a
  judge model: fixed question sets over the sample documents (local, hard, global, forms and totals) and
  HotpotQA and MuSiQue subsets, several Deep search variants in one interleaved run, and a paired comparison
  of two reports.
- **Documentation.** Getting started, how it works, deployment, architecture and state machines, and openly
  licensed sample documents with questions and expected answers.
