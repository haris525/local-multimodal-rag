# hard-paper-2

Question: Ragas computes its answer relevance score with embeddings from one OpenAI embedding model, and the best-practices RAG paper uses the same model for its chunk-size experiment. Which model is it, and what maximum input length and NarrativeQA nDCG@10 does the BGE-M3 paper report for it? How many points behind M3-Embedding (All) is it on NarrativeQA?

Gold answer: text-embedding-ada-002. The BGE-M3 paper lists it with a maximum length of 8191 tokens and 41.1 nDCG@10 on NarrativeQA (Table 4), 20.6 points behind M3-Embedding All at 61.7. (On MLDR, Table 3, it averages 32.5.)

Documents: ragas-rag-evaluation-2023.pdf, rag-best-practices-2024.pdf, bge-m3-embedding-2024.pdf

## Evidence

- Ragas uses text-embedding-ada-002 for answer relevance
  - ragas-rag-evaluation-2023.pdf, page 3
  - "We then obtain embeddings for all questions us- ing the text-embedding-ada-002 model, avail- able from the OpenAI API."

- this is the answer relevance metric
  - ragas-rag-evaluation-2023.pdf, page 3
  - "Answer relevance We say that the answer as(q) is relevant if it directly addresses the question in an appropriate way."

- best practices uses the same model in the chunk-size experiment (trailing 2 is footnote 2)
  - rag-best-practices-2024.pdf, page 5
  - "For embedding, we use the text-embedding-ada-0022 model, which supports long input length."

- that paragraph is the chunk-size experiment
  - rag-best-practices-2024.pdf, page 5
  - "The impact of different chunk sizes is shown in Table 3."

- 8191 and 41.1 on NarrativeQA (Table 4)
  - bge-m3-embedding-2024.pdf, page 8
  - "text-embedding-ada-002 8191 41.1"

- M3-Embedding All 61.7 on NarrativeQA
  - bge-m3-embedding-2024.pdf, page 8
  - "All 8192 61.7"

- MLDR average 32.5 (context only)
  - bge-m3-embedding-2024.pdf, page 8
  - "text-embedding-ada-002 8191 32.5"

## Arithmetic and notes

61.7 - 41.1 = 20.6 points.

## What makes it hard

Three documents: Ragas (p3) names the model, the best-practices paper (p5) uses the same one, and BGE-M3 Table 4 (p8) scores it. The best-practices text extracts as 'text-embedding-ada-0022' because footnote 2 follows the name.
