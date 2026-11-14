# hard-paper-3

Question: The best-practices RAG paper and BGE-M3 both combine a sparse (lexical) score and a dense score into one relevance score with a weighted sum. With the dense weight fixed at 1, what weight does each put on the sparse score in its main setting (best-practices: the alpha it selects; BGE-M3: Dense+Sparse on MIRACL)? And which dense and sparse weights does BGE-M3 use instead for Dense+Sparse on the MLDR long-document benchmark?

Gold answer: Both use 0.3 on the sparse score. Best practices: S_h = alpha * S_s + S_d, and alpha = 0.3 is selected (Table 8 tries 0.1 to 0.9). BGE-M3 on MIRACL: w1 = 1 (dense), w2 = 0.3 (sparse), w3 = 0 (multi-vector). For MLDR, BGE-M3 switches to w1 = 0.2 (dense) and w2 = 0.8 (sparse), w3 = 0, so the sparse score gets the larger weight on long documents.

Documents: rag-best-practices-2024.pdf, bge-m3-embedding-2024.pdf

## Evidence

- alpha weights the sparse score (S_h = alpha * S_s + S_d; the formula itself uses Greek alpha and a middle dot in the PDF, so only the ASCII parts are quoted)
  - rag-best-practices-2024.pdf, page 8
  - "where Ss, Sd are the normalized relevance scores from sparse retrieval and dense retrieval respectively, and Sh is the total retrieval score."

- alpha controls the sparse/dense weighting
  - rag-best-practices-2024.pdf, page 8
  - "controls the weighting between sparse retrieval and dense retrieval components."

- alpha = 0.3 selected (the PDF text reads 'we selected [alpha] = 0.3 for our retrieval and main experiments')
  - rag-best-practices-2024.pdf, page 8
  - "= 0.3 for our retrieval and main experiments."

- Table 8 is the alpha sweep
  - rag-best-practices-2024.pdf, page 8
  - "Table 8: Results of hybrid search with different alpha values."

- BGE-M3 Dense+Sparse on MIRACL: w1 = 1, w2 = 0.3
  - bge-m3-embedding-2024.pdf, page 6
  - "For the hybrid retrieval of dense method and sparse method (denoted as Dense+Sparse), we set w1 = 1, w2 = 0.3 and w3 = 0 in equation(1)"

- that paragraph is the MIRACL experiment
  - bge-m3-embedding-2024.pdf, page 6
  - "We evaluate the multi-lingual retrieval performance with MIRACL"

- BGE-M3 Dense+Sparse on MLDR: w1 = 0.2, w2 = 0.8
  - bge-m3-embedding-2024.pdf, page 8
  - "For Dense+Sparse method, we set w1 = 0.2, w2 = 0.8 and w3 = 0 in equation(1)."

- that paragraph is the MLDR experiment
  - bge-m3-embedding-2024.pdf, page 8
  - "The evaluation result on MLDR is presented in Table 3."

## Arithmetic and notes

Equation (1) of BGE-M3 (p4) is s_rank = w1 * s_dense + w2 * s_lex + w3 * s_mul, so w2 is the sparse (lexical) weight and w1 the dense weight.

## What makes it hard

Three places: the best-practices formula and chosen alpha (p8), the BGE-M3 MIRACL weights (p6) and the different MLDR weights (p8). The MLDR weights are easy to miss because MIRACL is the main setting.
