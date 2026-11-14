# global-6

Question: How many documents and how many questions does the Seven Failure Points paper say its BioASQ experiment used? The paper does not give the same document count everywhere: list every document count it states and where each appears.

Gold answer: 1000 questions throughout, but two different document counts. The introduction (the research questions, p1) and the conclusion (p5) say the empirical experiment involved 15,000 documents and 1000 questions (question and answer pairs). Section 4.3 (Biomedical Question and Answer, p3) says the authors downloaded 4017 open access documents from the BioASQ dataset with a total of 1000 questions, and Table 1 (p4) lists the BioASQ dataset size as 4017. The paper does not explain the difference.

Documents: rag-seven-failure-points-2024.pdf

## Evidence

- introduction: 15,000 documents and 1000 question (and answer pairs)
  - rag-seven-failure-points-2024.pdf, page 1
  - "We present an empirical experiment using the BioASQ data set to report on potential failure points. The experiment involved 15,000 documents and 1000 question"

- the sentence continues on page 2
  - rag-seven-failure-points-2024.pdf, page 2
  - "and answer pairs. We indexed all documents then ran the queries and stored the generated responses using GPT-4."

- Section 4.3: 4017 documents, 1000 questions
  - rag-seven-failure-points-2024.pdf, page 3
  - "We downloaded 4017 open access documents from the BioASQ dataset and had a total of 1000 questions."

- Table 1 columns
  - rag-seven-failure-points-2024.pdf, page 4
  - "Case Study Domain Doc Types Dataset Size RAG Stages Sample Questions"

- Table 1: BioASQ dataset size 4017
  - rag-seven-failure-points-2024.pdf, page 4
  - "BioASQ Biomedical Scientific PDFs 4017 Chunker, Retriever, Reader"

- conclusion: 15,000 documents and 1000 questions
  - rag-seven-failure-points-2024.pdf, page 5
  - "This paper presented the lessons learned from 3 case studies including an empirical investigation involving 15,000 documents and 1000 questions."

## Arithmetic and notes

A search of the whole extracted text for 15,000, 4017 and 1000 finds only these places (pages 1, 3, 4 and 5); Table 1 gives 38 for the AI Tutor and no size for Cognitive Reviewer. The paper never reconciles 15,000 with 4017.

## What makes it global

The two counts are in four places on four pages (introduction, Section 4.3, Table 1, conclusion). Each passage alone looks complete, so an answer from a few chunks reports one count and misses the other.
