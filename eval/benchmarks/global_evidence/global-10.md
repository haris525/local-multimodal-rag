# global-10

Question: In which experiments does the best-practices RAG paper use HotpotQA? For each, say what it is used for, which split or how many examples, and which metric is reported.

Gold answer: Four. (1) Summarization (Section 3.7, Table 10, Appendix A.4): HotpotQA is one of three benchmarks, with NQ and TriviaQA, for comparing summarization methods; its development set is used, with Llama3-8B-Instruct as the generator, and the metrics are F1 and the number of tokens. (2) Generator fine-tuning (Section 3.8, Appendix A.5, Tables 12 and 13): HotpotQA is one of seven training datasets and one of four evaluation datasets, with 15,000 training and 7,405 evaluation examples; the metric is ground-truth coverage. (3) Comprehensive evaluation (Section 4.1, Appendix A.6, Table 11): HotpotQA is a MultiHop QA dataset with 2WikiMultiHopQA and MuSiQue; 500 entries are sub-sampled per dataset and token-level F1 and (lenient) EM are reported. (4) RAG capabilities: 500 entries collected evenly from NQ, TriviaQA, HotpotQA, 2WikiMultiHopQA and MuSiQue are scored with the RAGAs metrics (faithfulness, context relevancy, answer relevancy, answer correctness) with GPT-4 as the judge.

Documents: rag-best-practices-2024.pdf

## Evidence

- Table 10 compares summarization methods on NQ, TQA and HotPotQA
  - rag-best-practices-2024.pdf, page 9
  - "Method NQ TQA HotPotQA Avg. Avg. Token F1 #token F1 #token F1 #token"

- summarization benchmarks
  - rag-best-practices-2024.pdf, page 10
  - "We evaluate these methods on three benchmark datasets: NQ, TriviaQA, and HotpotQA."

- summarization metrics
  - rag-best-practices-2024.pdf, page 20
  - "Metrics Evaluation metrics include the F1 score and the number of tokens changed after summa- rization to measure conciseness."

- summarization split
  - rag-best-practices-2024.pdf, page 20
  - "Experiments are conducted on the NQ test set, TriviaQA test set, and HotpotQA development set."

- generator fine-tuning: training datasets
  - rag-best-practices-2024.pdf, page 20
  - "We fine-tune our model on several question answering(QA) and reading comprehension datasets, including ASQA [60], HotpotQA [59], NarrativeQA [61], NQ [57], SQuAD [62], Trivi- aQA [58], TruthfulQA [63]."

- generator fine-tuning: evaluation datasets
  - rag-best-practices-2024.pdf, page 21
  - "For evaluation, ASQA [60], HotpotQA [59], NQ [57], TriviaQA [58] are used."

- Table 13: HotpotQA 15,000 train, 7,405 eval (the PDF text puts a space after each comma)
  - rag-best-practices-2024.pdf, page 21
  - "HotpotQA 15, 000 7, 405"

- generator fine-tuning metric
  - rag-best-practices-2024.pdf, page 21
  - "Metrics We use the ground-truth coverage as our evaluation metric"

- comprehensive evaluation: MultiHop QA datasets
  - rag-best-practices-2024.pdf, page 21
  - "(4) MultiHop QA: Our evaluation included the HotPotQA [59], 2WikiMultiHopQA [71], and MuSiQue [52] datasets."

- 500 entries per dataset
  - rag-best-practices-2024.pdf, page 21
  - "In each dataset, we randomly sub-sample 500 entries from the test set for our experiments."

- RAG capabilities subset
  - rag-best-practices-2024.pdf, page 21
  - "To assess RAG capabilities, we evenly collect a total of 500 entries from NQ, TriviaQA, HotPotQA, 2WikiMultiHopQA and MuSiQue."

- MultiHop QA metrics
  - rag-best-practices-2024.pdf, page 21
  - "Metrics We use token-level F1 score and EM score for Open-Domain QA and MultiHop QA tasks, and accuracy for others."

- RAG capability metrics
  - rag-best-practices-2024.pdf, page 21
  - "Towards RAG capabilities evaluation, we adopt four metrics from RAGAs, including Faithfulness, Context Relevancy, Answer Relevancy, and Answer Correctness."

- RAGAs judge
  - rag-best-practices-2024.pdf, page 22
  - "All these metrics are evaluated using the RAGAs framework, with GPT-4 serving as the judge."

## Arithmetic and notes

A case-insensitive search for HotpotQA (also spelled HotPotQA) finds it on pages 9, 10, 17 (the reference list), 20 and 21 only. Table 11 (p11) reports the MultiHop QA EM and F1 columns without naming the datasets. Table 13 is extracted as "15, 000 7, 405", that is 15,000 and 7,405.

## What makes it global

The four uses are in the summarization section (p9-10), the fine-tuning appendix (p20-21), the comprehensive evaluation appendix (p21) and the RAGAs setup (p21-22). A short answer usually names the multi-hop evaluation and misses the fine-tuning sizes or the summarization split.
