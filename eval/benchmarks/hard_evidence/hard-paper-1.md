# hard-paper-1

Question: The BGE-M3 paper compares M3-Embedding against one baseline whose text encoder is a much larger 7B decoder language model. Name that baseline and give its NarrativeQA nDCG@10 next to M3-Embedding's Dense and All scores. Then, from the paper that introduced the underlying 7B model, give that model's number of layers and its sliding window size.

Gold answer: The baseline is E5mistral-7b, which uses Mistral-7B as its text encoder. On NarrativeQA (BGE-M3 Table 4) it reaches 49.9 nDCG@10 with a max length of 8192, slightly above M3-Embedding Dense (48.7) and well below M3-Embedding All (61.7). The Mistral 7B paper (Table 1) gives 32 layers (n_layers) and a sliding window of 4096 tokens (window_size).

Documents: bge-m3-embedding-2024.pdf, mistral-7b-2023.pdf

## Evidence

- E5mistral-7b is the baseline built on a 7B decoder (Mistral-7B)
  - bge-m3-embedding-2024.pdf, page 6
  - "Even compared with E5mistral-7b, which leverages a much larger Mistral-7B model as the text encoder and specifi- cally trained with English data"

- Table 4 is NarrativeQA nDCG@10
  - bge-m3-embedding-2024.pdf, page 8
  - "Table 4: Evaluation on NarrativeQA (nDCG@10)."

- 49.9 (E5mistral-7b, max length 8192)
  - bge-m3-embedding-2024.pdf, page 8
  - "E5mistral-7b 8192 49.9"

- 48.7 (M3-Embedding Dense)
  - bge-m3-embedding-2024.pdf, page 8
  - "Dense 8192 48.7"

- 61.7 (M3-Embedding All)
  - bge-m3-embedding-2024.pdf, page 8
  - "All 8192 61.7"

- 32 layers
  - mistral-7b-2023.pdf, page 2
  - "n_layers 32"

- window 4096
  - mistral-7b-2023.pdf, page 2
  - "window_size 4096"

- these rows are the model architecture table
  - mistral-7b-2023.pdf, page 2
  - "Table 1: Model architecture."

## Arithmetic and notes

Table rows are quoted as extracted (one row per line in the PDF text). Dense 48.7 < 49.9 < 61.7 All.

## What makes it hard

Chain of three lookups: the BGE-M3 text (p6) names the baseline built on Mistral-7B, its row in Table 4 (p8) gives the scores, and the Mistral paper's Table 1 (p2) gives the architecture. The baseline beats M3 Dense on this benchmark, which a hurried answer may get backwards.
