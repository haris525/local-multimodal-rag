# global-9

Question: BM25 shows up in several different roles in the best-practices RAG paper's experiments. List every role it plays, with the module or experiment and a result or setting reported for it.

Gold answer: Four roles. (1) Retrieval baseline: BM25 is the unsupervised sparse retrieval baseline (with Contriever as the dense one) in the retrieval-method comparison on TREC DL19/20 (Table 6; on DL19 mAP 30.13, nDCG@10 50.58, latency 0.07 s); its index is built with Lucene on MS MARCO through Pyserini (Appendix A.2). (2) Hybrid search: the recommended Hybrid Search combines BM25 sparse scores with dense scores, weighted by alpha = 0.3 (Tables 6 and 8), and is part of the best 'Hybrid with HyDE' retrieval. (3) Reranking: on the MS MARCO passage ranking dev set, BM25 retrieves the top-1000 candidate passages that the rerankers reorder, and it is the no-reranking baseline (MRR@10 11.65; Table 9, Appendix A.3). (4) Summarization: BM25 is one of the extractive compression methods compared in Table 10 (average F1 29.47 with 54 tokens).

Documents: rag-best-practices-2024.pdf

## Evidence

- retrieval baseline: BM25 sparse, Contriever dense
  - rag-best-practices-2024.pdf, page 7
  - "In this study, we use BM25 for sparse retrieval and Contriever [45], an unsupervised contrastive encoder, for dense retrieval, serving as two robust baselines based on Thakur et al. [46]."

- Table 6 row (TREC DL19 then DL20: mAP, nDCG@10, R@50, R@1k, latency)
  - rag-best-practices-2024.pdf, page 7
  - "BM25 30.13 50.58 38.32 75.01 0.07 28.56 47.96 46.18 78.63 0.29"

- Hybrid Search combines BM25 with dense retrieval
  - rag-best-practices-2024.pdf, page 7
  - "Hybrid Search combines sparse retrieval (BM25) and dense retrieval (Original embedding) and achieves notable performance with relatively low latency."

- hybrid weighting of sparse and dense scores
  - rag-best-practices-2024.pdf, page 8
  - "where Ss, Sd are the normalized relevance scores from sparse retrieval and dense retrieval respectively, and Sh is the total retrieval score."

- Table 8, alpha = 0.3 row (the alpha symbol is left out of the quote)
  - rag-best-practices-2024.pdf, page 8
  - "= 0.3 47.14 72.50 51.13 89.08 3.20 47.72 69.80 64.32 88.04 0.77"

- reranking: BM25 supplies the top-1000 candidates
  - rag-best-practices-2024.pdf, page 8
  - "For each query, the top-1000 candidate passages retrieved by BM25 are reranked."

- Table 9 row without reranking (MRR@1, MRR@10, MRR@1k, Hit Rate@10)
  - rag-best-practices-2024.pdf, page 8
  - "BM25 - - 6.52 11.65 12.59 24.63 -"

- Table 10: BM25 as an extractive summarization method
  - rag-best-practices-2024.pdf, page 9
  - "Extractive Method BM25 27.97 40 32.44 59 28.00 63 29.47 54"

- implementation: BM25 via Pyserini, Lucene index
  - rag-best-practices-2024.pdf, page 19
  - "For sparse retrieval, we use the BM25 algorithm, which relies on the TF- IDF algorithm."

- same appendix: the Lucene index
  - rag-best-practices-2024.pdf, page 19
  - "The BM25 index is constructed using Lucene on MS MARCO collections"

- reranking baseline
  - rag-best-practices-2024.pdf, page 19
  - "We compare our results with a randomly shuffled ordering and the BM25 retrieval baseline."

## Arithmetic and notes

A search of the whole extracted text for BM25 finds it on pages 2 (Figure 1, under Retrieval and under Extractive summarization), 7, 8, 9 and 19 only; every mention belongs to one of the four roles. Table rows are quoted as extracted; "TF- IDF" is the PDF's line-break hyphen.

## What makes it global

Each role lives in a different module section (3.4 retrieval, 3.4.3 hybrid weights, 3.5 reranking, 3.7 summarization) and the appendix. Most answers give the retrieval and hybrid roles and miss that BM25 also feeds the rerankers and is a summarization baseline.
