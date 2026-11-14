# hard-paper-9

Question: The best-practices RAG paper evaluates fact checking on two datasets. Which of them is also used in ReAct, what three labels can a claim receive in it, and what accuracy does PaLM-540B reach on it with ReAct prompting versus Act-only prompting?

Gold answer: FEVER (the other one is PubHealth). Each FEVER claim is annotated SUPPORTS, REFUTES or NOT ENOUGH INFO. In ReAct's Table 1 (PaLM-540B), ReAct reaches 60.9% accuracy on FEVER and Act 58.9%.

Documents: rag-best-practices-2024.pdf, react-reasoning-and-acting-2022.pdf

## Evidence

- the two fact-checking datasets
  - rag-best-practices-2024.pdf, page 21
  - "(2) Fact Checking: Our evaluation encompassed the FEVER [68] and PubHealth [69] datasets."

- FEVER labels (the PDF text breaks 'verification' at a ligature: 'a fact veri' + 'cation benchmark ...')
  - react-reasoning-and-acting-2022.pdf, page 4
  - "cation benchmark where each claim is annotated SUPPORTS, REFUTES, or NOT ENOUGH INFO"

- Table 1 is PaLM-540B prompting on HotpotQA (EM) and Fever (Acc)
  - react-reasoning-and-acting-2022.pdf, page 5
  - "Table 1: PaLM-540B prompting results on HotpotQA and Fever."

- Act: Fever 58.9
  - react-reasoning-and-acting-2022.pdf, page 5
  - "Act 25.7 58.9"

- ReAct: Fever 60.9
  - react-reasoning-and-acting-2022.pdf, page 5
  - "ReAct 27.4 60.9"

## Arithmetic and notes

The second number of each Table 1 row is Fever accuracy.

## What makes it hard

Chain: the best-practices appendix (p21) gives the dataset pair, ReAct's setup (p4) the labels and Table 1 (p5) the scores. The ReAct-vs-Act numbers are only in the table.
