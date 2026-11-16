# Design: search tasks (instructions for embedding the question)

Status: implemented. [ARCHITECTURE.md](../ARCHITECTURE.md) (sections 7.2 and 7.7) describes the current behavior;
where this note and the code differ, the code and ARCHITECTURE win. The measurements are in
[benchmarks/embedding-models-2026-11-15.md](../benchmarks/embedding-models-2026-11-15.md).

Decisions:
- built-in tasks plus custom tasks per user;
- the reranker instruction stays fixed (measured only);
- the default instruction changes only when the experiment in section 2 and an end-to-end check support it.

## 1. Why

The text embedder (Qwen3-Embedding) turns a question into a vector after an instruction. The app used one fixed
instruction, `Given a question, retrieve passages that answer it` (`services/models` config `query_instruction`).
The embedding benchmark (`rag-eval embed-bench`) showed that this instruction does worse than no instruction for
every model tried:

- **Dense recall@10 on HotpotQA, current 0.6B:** 0.680 with the app's instruction, 0.797 with none, 0.790 with
  the model's own default prompt. MuSiQue: 0.455, 0.587 and 0.600.
- **After the reranker,** the gap shrinks: HotpotQA +0.3 points, MuSiQue +2.5 points. On the example documents,
  no change.
- **Instructions written for a task do better still.** Qwen's multi-hop instruction reaches 0.812 on HotpotQA.
  But it was written for these datasets, so it is an upper bound, not a fair choice for the default.

So the best general instruction was not known, and different kinds of questions want different ones. This design
has two parts:
1. Choose a better default instruction properly.
2. Let each user pick a task per question, and add their own.

## 2. Part A: choose the default (experiment)

An experiment with `embed-bench`, retrieval only, then one end-to-end check.

- **Candidates, fixed before the run** (query instruction for Qwen3-Embedding-0.6B):

  | Id | Instruction |
  |---|---|
  | C0 | `Given a question, retrieve passages that answer it` (the instruction used before) |
  | C1 | none (the bare question) |
  | C2 | `Given a web search query, retrieve relevant passages that answer the query` (model default) |
  | C3 | `Given a multi-hop question, retrieve documents that can help answer the question` (task: multi-step) |
  | C4 | `Given a question, retrieve relevant passages that help answer it` |
  | C5 | `Given a question about the user's documents, retrieve passages with the facts needed to answer it` |
  | C6 | `Retrieve passages relevant to the question` |

- **No overfitting.** HotpotQA and MuSiQue questions are split in half by a hash of their id: a tune half and a
  test half.
  - **Choosing General.** General is the general candidate (C1, C2, C4, C5, C6) with the best tune-half
    rerank recall@10, averaged over HotpotQA and MuSiQue with equal weight. C3 is written for one kind of
    question, so it competes only for the Multi-step preset.
  - **Reporting.** Every comparison is then reported on the test half only.
  - **Example documents.** The 36 example-document questions are too few to choose by, so they are reported
    only, as a check that nothing gets worse.
- **Reranker instruction.** Measured as a secondary result, with General as the query instruction, on the test
  half:
  - R0, the current `Given a question about the user's documents, judge whether the passage helps answer it`;
  - R1, the reranker's own default `Given a web search query, retrieve relevant passages that answer the query`;
  - R2, `Judge whether the passage contains information that helps answer the question`.

  It stays R0.
- **End to end.** The best default runs through `rag-eval bench` on the local and hard sets with the LLM, against
  the earlier instruction. The scores are model-free.
- **Rule, fixed before the run:**
  - The new default must not be worse on any tier. On the example documents that means within the interval;
    on the public halves, the interval of the difference must not go below 0.
  - It must not be significantly worse end to end.
  - Otherwise the default stays.
- **Cost.** Passage vectors are cached, so each candidate costs only question embeddings and new rerank pairs:
  about 15 to 30 minutes of GPU time per run with the LLM stopped. Then the bench run, a few hours.

The change is query-side only. Documents keep their vectors, so a new default needs no reindex.

## 3. Part B: search tasks in the app

A **search task** is a named instruction used to embed the question.

- **Built-in tasks.** These are presets in the repository, each listed with what it was measured to do:
  - **General** (the default from part A);
  - **Multi-step research** (the multi-hop instruction, for questions that chain facts across passages);
  - **Exact lookup** (no instruction, close to keyword-style questions).

  A preset that measures worse than General everywhere is dropped.
- **Custom tasks.** Each user can add their own: a name of at most 40 characters and an instruction of at most
  300 characters. For example: "Contracts: given a question about an agreement, retrieve the clauses that
  answer it".
  - They are stored per user in the threads service, next to memories.
  - They are private to the user, deleted with the account, and not readable by admins, like the rest of a
    user's content.
- **Per question.** A "Search task" choice sits next to the mode and Think controls. Each turn stores the task it
  was asked with, as it stores mode and Think, so follow-ups and Retry use the same one. The default is General.
- **API:**
  - `POST /api/chat` and `POST /api/search` take `search_task`: a preset id or a custom task id. A missing one
    means General; an unknown one is a 422.
  - New endpoints manage custom tasks: `GET`, `POST`, `PUT` and `DELETE /api/search-tasks`, with the same
    owner checks as memories.

### Flow

1. The gateway passes `search_task` on to rag.
2. rag resolves it to instruction text. A preset comes from config; a custom task comes from threads, for this
   user only.
3. rag calls `ModelsClient.embed_query(question, instruction)`.
4. The `models` service accepts `instruction` on `/v1/embed/text` for role `query`, and builds
   `Instruct: {instruction}\nQuery:`. Documents are unchanged.
5. The same vector serves dense search and the per-document candidates.
6. In deep mode, every search of that turn uses the task. Exhaustive mode reads every section, so it does not
   use one.
7. Image queries keep their own instruction, which has not been measured.

### Validation and safety

- **Format:** a single line, NFC, no control characters, 1 to 300 characters. Leading and trailing spaces are
  trimmed. Validation is the same in rag, `threads` and `models`.
- **Reach:** the text only reaches the embedding model, never the LLM. A hostile instruction can therefore only
  make that user's own retrieval worse; it cannot steer an answer or read anything new. Owner scoping still
  applies to every read.
- **Logs and audit:** these record the task kind (`preset:<id>` or `custom`), never custom text.
- **Limits:** at most 50 custom tasks per user, configurable.

### Configuration

- `RAG_QUERY_INSTRUCTION` sets the General instruction, so a deployment can change it without a rebuild.
- The presets live in a file in `services/rag`, each with an id, a name, an instruction and a measured-effect
  note.
- The `models` default stays as a fallback for callers that send no instruction.

## 4. What changes

- **contracts:** `EmbedTextRequest` gains `instruction: str | None`.
- **models:** the query prompt is built from the request's instruction when given.
- **rag:**
  - `search_task` on `ChatRequest` and `SearchRequest`;
  - preset loading and resolution;
  - `embed_query(text, instruction)`;
  - the task in turn records and logs.
- **threads:** a custom search task store (schema change with an upgrade step), with CRUD and owner checks,
  deleted with the account.
- **gateway UI:**
  - the task selector next to mode and Think;
  - a "Search tasks" panel to add, edit and delete custom tasks, next to Memory;
  - turns show and reuse their task.
- **eval:** `embed-bench` takes any query instruction and compares runs on one half of the public questions;
  `bench` gains `--search-task`.
- **Docs and tests:** unit tests for validation and resolution, API tests for CRUD and owner isolation, UI tests
  for the selector and the panel; ARCHITECTURE (retrieval, threads), HOW_IT_WORKS, a DEPLOYMENT update section
  (rebuild models, rag, threads and gateway; threads schema upgrade; no reindex), STATE_MACHINES and the
  CHANGELOG.

**As built, where it differs from the above:**

- **The limit is `THREADS_MAX_SEARCH_TASKS`**: `threads` enforces it, and its settings take the `THREADS_`
  prefix, like `THREADS_MAX_MEMORIES`. Past the limit, adding a task is a 409.
- **The UI reads the built-in tasks from rag:** `GET /api/search-tasks/presets` (read-only, routed to `rag`), so
  the presets file and `RAG_QUERY_INSTRUCTION` stay the only source of their names and instructions. The other
  `/api/search-tasks` routes go to `threads`.
- **The presets file is a Python module**, `services/rag/src/rag/search_tasks.py`. Each preset carries its
  measured-effect note, which the UI shows.
- **Exhaustive mode accepts a task and ignores it** (no 422), since the UI sends the picker's choice with every
  question. It does not even resolve it, so a stale id or `threads` being down cannot fail it, and its records
  carry no task kind.
- **models answers with the instruction it applied**, and rag refuses a vector made without the one it sent, so an
  earlier `models` (which ignores the field) is an error, not a silent fallback to its default.
- **Custom tasks never hold a credential value** (as redaction finds them), and a name needs a letter or digit and
  no invisible formatting characters.
- **The thread record keeps the task's id as well as its kind**, so follow-ups and Retry can ask with it again;
  the audit and the logs keep the kind only. The audit also records `search_task.added | updated | deleted` with
  the task's id, never its text.
- **That a custom task changes the query vector** is covered hop by hop by the tests: the models prompt, the
  instruction rag sends, and the vector serving dense search and the per-document candidates.
- **embed-bench** takes `--query-instruction-text` (any instruction), `--rerank-instruction` and `--label`;
  `python -m rag_eval.embedbench.compare --half tune|test` scores one half of the public questions.
