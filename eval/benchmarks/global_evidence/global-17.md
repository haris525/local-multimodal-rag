# global-17

Question: Both the BGE-M3 paper and the best-practices RAG paper use Faiss, Lucene and Pyserini. For each paper, say what each of the three is used for. Which vector database does the best-practices paper pick for its final system instead, and why?

Gold answer: BGE-M3: on MIRACL it follows the official benchmark and evaluates with Pyserini; the Dense method builds a dense index with Faiss to search the top-1000 candidates, and the Sparse method builds a sparse index with Lucene for its top-1000; in Appendix C.2 the Lucene Analyzer is used as BM25's tokenizer and beats BM25 with the XLM-RoBERTa tokenizer (MIRACL 38.5 vs 31.9, MKQA 40.9 vs 39.9, MLDR 64.1 vs 53.6). Best practices: the BM25 and Contriever baselines use Pyserini's default implementations, and the retrieval experiments and evaluation are run with the Pyserini toolkit; the BM25 index is built with Lucene on the MS MARCO collections and the dense vector index with Faiss in a Flat configuration (Appendix A.2). Faiss is also one of five vector databases compared in Table 5 (with Weaviate, Chroma, Qdrant and Milvus); it has multiple index types but no billion-scale support, hybrid search or cloud-native features. The paper picks Milvus, the only one meeting all four criteria (multiple index types, billion-scale vectors, hybrid search, cloud-native), and builds its vector database in Milvus with 10 million English Wikipedia texts and 4 million medical texts.

Documents: bge-m3-embedding-2024.pdf, rag-best-practices-2024.pdf

## Evidence

- BGE-M3: evaluation with Pyserini
  - bge-m3-embedding-2024.pdf, page 6
  - "Following the of- ficial benchmark, we evaluate our method using Pyserini (Lin et al., 2021)"

- BGE-M3: Faiss dense index
  - bge-m3-embedding-2024.pdf, page 6
  - "we first use it to generate the embeddings of the corpus and then build the dense index for search- ing top-1000 candidates with Faiss."

- BGE-M3: Lucene sparse index
  - bge-m3-embedding-2024.pdf, page 6
  - "we first use it to gen- erate the weights of the corpus and then build the sparse index for searching top-1000 candidates with Lucene."

- BGE-M3: Lucene Analyzer for BM25 ("Lucene9" is Lucene with footnote 9)
  - bge-m3-embedding-2024.pdf, page 16
  - "Using the Analyzer from Lucene9 can signif- icantly enhance the effectiveness of BM25."

- BGE-M3 Table 11: BM25 with the Analyzer (MIRACL, MKQA, MLDR)
  - bge-m3-embedding-2024.pdf, page 16
  - "BM25 Analyzer 38.5 40.9 64.1"

- BGE-M3 Table 11: BM25 with the XLM-R tokenizer
  - bge-m3-embedding-2024.pdf, page 16
  - "BM25 XLM-R 31.9 39.9 53.6"

- best practices: Pyserini implementations
  - rag-best-practices-2024.pdf, page 19
  - "We use the default implementation of BM25 and Contriever from Pyserini [56]."

- best practices: Lucene BM25 index and Faiss Flat dense index
  - rag-best-practices-2024.pdf, page 19
  - "The BM25 index is constructed using Lucene on MS MARCO collections, while the dense vector index is generated with Faiss employing Flat configuration on the same dataset."

- best practices: Pyserini toolkit
  - rag-best-practices-2024.pdf, page 19
  - "Retrieval experiments and evaluation are conducted using the Pyserini toolkit."

- best practices: the four criteria
  - rag-best-practices-2024.pdf, page 6
  - "we evaluated several options based on four key criteria: multiple index types, billion-scale vector support, hybrid search, and cloud-native"

- best practices: five vector databases compared
  - rag-best-practices-2024.pdf, page 6
  - "Table 5 presents a detailed comparison of five open-source vector databases: Weaviate, Faiss, Chroma, Qdrant, and Milvus."

- best practices: Milvus chosen
  - rag-best-practices-2024.pdf, page 6
  - "Our evaluation indicates that Milvus stands out as the most comprehensive solution among the databases evaluated, meeting all the essential criteria and outperforming other open-source options."

- best practices: the Milvus database used
  - rag-best-practices-2024.pdf, page 11
  - "We used Milvus to build a vector database that includes 10 million text of English Wikipedia and 4 million text of medical data."

## Arithmetic and notes

Table 5 (p6) marks Faiss with a check for multiple index types and crosses for billion-scale, hybrid search and cloud-native, and Milvus with four checks; the check marks are not ASCII, so the rows are described here rather than quoted. Searches for Faiss, Lucene, Pyserini and Milvus find no other uses in either paper (Figure 1 of the best-practices paper lists the five databases). "of- ficial", "search- ing", "gen- erate" and "signif- icantly" are the PDF's line-break hyphens.

## What makes it global

Two papers, each with the tools in two or three distant places (BGE-M3 p6 and p16; best practices p6, p11 and p19). The Analyzer comparison is in an appendix and the Milvus choice in a different section from the Faiss index, so a few chunks give part of each paper's story.
