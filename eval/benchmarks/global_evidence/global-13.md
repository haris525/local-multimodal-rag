# global-13

Question: The BGE-M3 paper gives several different numbers of languages. List every number of languages it states in its text and what each one refers to.

Gold answer: Five. (1) More than 100 working (world) languages: what M3-Embedding supports (abstract, introduction, Figure 1 '100+ Languages', and the limitations). (2) 194 languages: the unsupervised training data, 1.2 billion text pairs with 2,655 cross-lingual correspondences (Section 3.1). (3) 18 languages: the MIRACL multilingual retrieval benchmark (Section 4.1). (4) 25 non-English languages: the queries of the MKQA cross-lingual benchmark (Section 4.2). (5) 105 languages: the 184 million text samples (from Pile, Wudao and mC4) used to further pre-train XLM-RoBERTa with RetroMAE (Appendix B.1). MLDR's 13 languages appear only as table rows, not as a number in the text.

Documents: bge-m3-embedding-2024.pdf

## Evidence

- more than 100 working languages (abstract)
  - bge-m3-embedding-2024.pdf, page 1
  - "It provides a uniform support for the semantic re- trieval of more than 100 working languages."

- more than 100 world languages (introduction)
  - bge-m3-embedding-2024.pdf, page 2
  - "linguality, which is able to support more than 100 world languages."

- 194 languages in the unsupervised data (Section 3.1)
  - bge-m3-embedding-2024.pdf, page 3
  - "In total, it brings in 1.2 billion text pairs of 194 languages and 2655 cross-lingual correspondences."

- MIRACL: 18 languages (Section 4.1)
  - bge-m3-embedding-2024.pdf, page 6
  - "We evaluate the multi-lingual retrieval performance with MIRACL (Zhang et al., 2023c), which con- sists of ad-hoc retrieval tasks in 18 languages."

- MKQA: 25 non-English languages (Section 4.2)
  - bge-m3-embedding-2024.pdf, page 7
  - "We make evaluation for the cross-lingual retrieval performance with the MKQA benchmark (Longpre et al., 2021), which includes queries in 25 non- English languages."

- limitations repeat more than 100
  - bge-m3-embedding-2024.pdf, page 9
  - "Furthermore, we claim support for more than 100 working languages in M3-Embedding."

- 105 languages in the RetroMAE pre-training data (Appendix B.1)
  - bge-m3-embedding-2024.pdf, page 14
  - "We sampled a total of 184 million text samples from these sources, covering 105 languages."

## Arithmetic and notes

A search of the whole extracted text for "languages" preceded by a number finds exactly 100 (p1, p2, p9), 194 (p3), 18 (p6; also the Table 12 caption on p17 and a reference title on p13), 25 (p7; also the Table 13 caption on p17) and 105 (p14). MLDR (Table 3, Table 7) has 13 language rows but the text never states 13. "re- trieval", "con- sists" and "non- English" are the PDF's line-break hyphens.

## What makes it global

The numbers are spread over the abstract, Section 3.1, Section 4.1, Section 4.2, the limitations and Appendix B.1 (pages 1 to 14). The 105 is only in the appendix and the 194 only in the data section, so a few retrieved chunks give two or three of the five.
