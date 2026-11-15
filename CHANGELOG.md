# Changelog

Newest first. Each entry says under "Data changes" whether it changes what is stored in the Docker volumes
and what to do about it (see [DEPLOYMENT.md, section 4.8](docs/DEPLOYMENT.md#48-updates)).

## Unreleased: consistent reindexing

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

**Data changes:** registry schema v5 adds `generation`, `storage_id` and a generation catalog. Existing
documents keep their paths/point IDs and remain readable without reindexing. New derived data uses
generation directories/IDs. Retain old generations until deletion, so repeated reindexing grows disk
usage. Use the coordinated upgrade in [Deployment](docs/DEPLOYMENT.md#consistent-reindex-upgrade);
rolling back after new generations are published requires the matching data backup.

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
