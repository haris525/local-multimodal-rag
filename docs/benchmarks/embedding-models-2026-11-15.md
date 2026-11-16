# Text embedding models and query instructions, 2026-11-15

**Short answer:** keep Qwen3-Embedding-0.6B. The larger Qwen3 embedders (4B, 8B, VL-8B) find the right
passages more often with embeddings alone. After this app's keyword fusion and reranker, the best of them
adds only about 2.5 to 3 points of recall@10 on hard and multi-hop questions, and nothing measurable on the
example documents. They also need 6 to 14 GiB more GPU memory than a 32 GB card has free next to the 27B LLM.

The bigger lever is the query instruction. The one the app used, `Given a question, retrieve passages that
answer it`, did worse than no instruction for every model. A measured replacement is now the default, and users can
pick a search task per question.

Measured with `rag-eval embed-bench` (retrieval only, no LLM) on one 32 GB GPU. The method is in
[ARCHITECTURE.md](../ARCHITECTURE.md), section 9 under "Embedding benchmark".

## What was measured

- **Models.** Qwen3-Embedding-0.6B (the app's default), Qwen3-Embedding-4B, Qwen3-Embedding-8B, and
  Qwen3-VL-Embedding-8B used as a text embedder. Every model ran with the app's settings: BF16, the same
  prompts and `max_seq_length` 1024. Qwen has no 9B text embedder; the 8B (7.6B parameters) is the
  closest.
- **Pipelines,** over the same passages for every model:
  - exact dense search;
  - dense plus BM25, fused with the app's reciprocal rank fusion (top 40 each, k = 60);
  - that fusion with its top 30 reranked by Qwen3-Reranker-0.6B, as the app does.
- **Question sets:**

  | Tier | Questions | Passages searched | Gold passages |
  |---|---|---|---|
  | Example documents | 36 text questions from `eval/benchmarks/local_samples.jsonl` | the app's own chunks of the 22 benchmark documents (24,398) | hand-checked labels in `eval/benchmarks/local_passages.jsonl` |
  | HotpotQA dev (2 hops) | 7,405 | one shared pool of every question's paragraphs (66,635) | the 2 supporting paragraphs |
  | MuSiQue-Ans dev (2 to 4 hops) | 2,417 | one shared pool (21,100) | the 2 to 4 supporting paragraphs |

- **Metric.** Recall@10: the share of gold passages in the top 10, with 95% bootstrap intervals, and the paired
  difference from the 0.6B on the same questions.

## Results with the app's instruction

Recall@10 after the reranker (the pipeline the app runs), with the query instruction the app used at the time,
`Given a question, retrieve passages that answer it` (C0 below). `--query-instruction app` now uses the new
default, so reproduce this table with `--query-instruction-text "Given a question, retrieve passages that answer it"`.

| Tier | 0.6B | 4B | 8B | VL-8B |
|---|---|---|---|---|
| Example documents, selected documents | 0.903 | 0.847 | 0.861 | 0.861 |
| HotpotQA | 0.889 | 0.902 | 0.909 | 0.915 |
| MuSiQue | 0.627 | 0.640 | 0.657 | 0.667 |

The example-document tier has only 36 questions, so its intervals are about +/-0.1: it cannot separate the
models.

## The query instruction

With dense search only, the 4B and 8B first scored below the 0.6B, which goes against Qwen's published
results.
- **Not a setup bug.** Batched, single and Qwen's reference embeddings agree.
- **The cause is the instruction.** Dense recall@10 on 1,500 HotpotQA questions:

| Model | app instruction | no instruction | Qwen's multi-hop instruction |
|---|---|---|---|
| 0.6B | 0.680 | 0.797 | 0.812 |
| 4B | 0.616 | 0.852 | 0.876 |
| 8B | 0.663 | 0.873 | 0.898 |

With no instruction, or with each model's own default prompt, the order is as published: 8B > 4B > 0.6B.
After the reranker the instruction matters less. For the 0.6B, dropping it moves HotpotQA from 0.889 to
0.893 and MuSiQue from 0.627 to 0.649, and leaves the example documents unchanged. It changes only how the
question is embedded, so documents need no reindex.

## Cost on a 32 GB GPU

| Model | Dimensions | Weights / peak GiB | Passages per second | Question embedding p50 |
|---|---|---|---|---|
| 0.6B | 1,024 | 1.1 / 2.1 | 510 | 25 ms |
| 4B | 2,560 | 7.5 / 10.0 | 121 | 29 ms |
| 8B | 4,096 | 14.1 / 17.6 | 69 | 100 ms |
| VL-8B | 4,096 | 15.2 / 18.7 | 60 | 61 ms |

## Choosing a better default instruction

Seven query instructions were run on the 0.6B. Both the candidates and the selection rule were written down
before the run ([design, section 2](../design/search-tasks.md)):

| Id | Instruction |
|---|---|
| C0 | `Given a question, retrieve passages that answer it` (the old default) |
| C1 | none |
| C2 | `Given a web search query, retrieve relevant passages that answer the query` (model default) |
| C3 | `Given a multi-hop question, retrieve documents that can help answer the question` |
| C4 | `Given a question, retrieve relevant passages that help answer it` |
| C5 | `Given a question about the user's documents, retrieve passages with the facts needed to answer it` |
| C6 | `Retrieve passages relevant to the question` |

- **How the choice was made.** HotpotQA and MuSiQue were split in half by a hash of the question id. The
  general candidates (C1, C2, C4, C5, C6) were ranked on one half, by recall@10 after the reranker, averaged
  over the two sets. C5 came first, but the five were within 0.0014 of each other: in practice any of them
  would do.
- **Results** on the other half, recall@10 and the difference from C0:

| Pipeline, tier | C0 | C5 (new General) | C3 (multi-step) |
|---|---|---|---|
| reranked, HotpotQA (3,684) | 0.884 | 0.893 (+0.009) | 0.896 (+0.012) |
| reranked, MuSiQue (1,191) | 0.639 | 0.658 (+0.019) | 0.659 (+0.020) |
| reranked, example documents (36) | 0.903 | 0.903 (0.000) | 0.903 (0.000) |
| dense only, HotpotQA | 0.669 | 0.793 (+0.124) | 0.804 (+0.136) |
| dense only, MuSiQue | 0.463 | 0.603 (+0.140) | 0.615 (+0.152) |
| dense only, example documents, all 22 | 0.662 | 0.755 (+0.093) | 0.741 (+0.079) |

- **The reranker's own instruction** was also varied, with C5 for the question. The two alternatives moved
  recall@10 by -0.2 to +0.8 points and lowered MRR on the example documents, so the reranker keeps its
  instruction.

**End-to-end answer check.** The repository's benchmark sets were run through the whole app with the LLM, 2
repeats each, under C0 and then under C5. Answers are scored without a judge.

| Set | C0 | C5 | Questions won / lost by C5 | Significance |
|---|---|---|---|---|
| Local, single search (67 questions) | 0.866 | 0.828 | 0 / 4 | p = 0.12, not significant |
| Hard, deep search (33 questions) | 0.803 | 0.894 | 7 / 2 | p = 0.15, not significant |

Two identical C0 runs of the local set also differed on 4 questions (2 each way). Two of C5's four local losses
are among those, so most of the local difference is run-to-run variation.

Repeated asks of single questions after the change showed one possible trade-off. The question asks how two papers
define faithfulness. Answers under C5 mentioned Ragas' statement-based definition in 4 of 9 runs, against 9 of 10
under C0 (p about 0.06). The passage is retrieved under both instructions, one rank lower under C5, so the
difference is in what the model chooses to say.

**Decision:** C5 is the new default for the General search task (`RAG_QUERY_INSTRUCTION`). C3 is the Multi-step
research task. Setting `RAG_QUERY_INSTRUCTION` back to C0 restores the old behavior. Neither change needs a
reindex.

## Rerun it

`rag-eval embed-bench` runs in the `local-rag/models` image on the GPU, with the LLM stopped:
1. Export the benchmark account's chunks with `eval/src/rag_eval/embedbench/export_chunks.py` inside
   `ingest-worker`.
2. Download the candidate models into their own volume.
3. Run `python -m rag_eval.embedbench.run --arms base 4b 8b vl8b --local-chunks chunks.jsonl`.

`--query-instruction` and `--query-instruction-text` compare instructions. `compare.py` splits the public
questions into a tune half and a test half.

Datasets: HotpotQA (CC BY-SA 4.0, Yang et al., 2018) and MuSiQue (CC BY 4.0, Trivedi et al., 2022), fetched
with `rag-eval fetch-public`.
