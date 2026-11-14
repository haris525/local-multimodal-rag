# global-12

Question: The BGE-M3 paper combines its dense, sparse (lexical) and multi-vector scores with weights w1, w2 and w3. List every setting of these three weights that the paper gives and what each setting is used for.

Gold answer: Four distinct settings, used in five places. Training (Section 3.3): the teacher score for self-knowledge distillation uses w1 = 1, w2 = 0.3, w3 = 1 (w2 is low because the randomly initialised lexical weights start out poor). MIRACL (Section 4.1): Dense+Sparse uses w1 = 1, w2 = 0.3, w3 = 0 to re-rank the union of the top-1000 dense and top-1000 sparse candidates, and All uses w1 = 1, w2 = 0.3, w3 = 1 to re-rank the top-200 dense candidates. MKQA (Section 4.2): the same weights as MIRACL. Long-document retrieval (Section 4.3, MLDR and NarrativeQA): Dense+Sparse uses w1 = 0.2, w2 = 0.8, w3 = 0, and All uses w1 = 0.15, w2 = 0.5, w3 = 0.35.

Documents: bge-m3-embedding-2024.pdf

## Evidence

- training: w1 = 1, w2 = 0.3, w3 = 1
  - bge-m3-embedding-2024.pdf, page 5
  - "In order to reduce the impact of this, we set w1 = 1, w2 = 0.3, w3 = 1,"

- MIRACL Dense+Sparse: 1, 0.3, 0
  - bge-m3-embedding-2024.pdf, page 6
  - "we set w1 = 1, w2 = 0.3 and w3 = 0 in equation(1) to re-rank the union set of top-1000 candidates from Dense"

- MIRACL All: 1, 0.3, 1
  - bge-m3-embedding-2024.pdf, page 6
  - "For the hybrid retrieval of all three methods (denoted as All), we set w1 = 1, w2 = 0.3 and w3 = 1 in equation(1) to re-rank the top-200 candidates from Dense."

- MKQA: same as MIRACL
  - bge-m3-embedding-2024.pdf, page 7
  - "For Dense+Sparse method and All method, we set the same weights as in MIRACL dataset."

- long-doc (MLDR, NarrativeQA): 0.2, 0.8, 0 and 0.15, 0.5, 0.35
  - bge-m3-embedding-2024.pdf, page 8
  - "For Dense+Sparse method, we set w1 = 0.2, w2 = 0.8 and w3 = 0 in equation(1). For All method, we set w1 = 0.15, w2 = 0.5 and w3 = 0.35 in equation(1)."

- Section 4.3 covers both MLDR and NarrativeQA
  - bge-m3-embedding-2024.pdf, page 7
  - "We evaluate the retrieval performance with longer sequences with two benchmarks: MLDR (Multi- lingual Long-Doc Retrieval), which is curated by"

## Arithmetic and notes

A search of the whole extracted text for "w1 =" finds exactly five places (p5, p6 twice, p8 twice); the MKQA sentence (p7) refers back to MIRACL. The training sentence goes on with the loss weights (lambda1 = 1, lambda2 = 0.1, lambda3 = 1), which are not w1 to w3 and are left out of the quote because the PDF writes them with Greek letters. Distinct settings: (1, 0.3, 1), (1, 0.3, 0), (0.2, 0.8, 0), (0.15, 0.5, 0.35).

## What makes it global

The settings are in Section 3.3 (p5), Section 4.1 (p6), Section 4.2 (p7) and Section 4.3 (p8). A retrieval for the weights tends to return the MIRACL paragraph and misses the different long-document weights or the training setting.
