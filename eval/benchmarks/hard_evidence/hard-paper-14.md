# hard-paper-14

Question: The Seven Failure Points paper scored the answers of its BioASQ RAG system with the Ragas faithfulness and answer relevance metrics. What did that evaluation find?

Gold answer: The premise is wrong: the paper does not use Ragas. Its BioASQ responses were generated with GPT-4 and validated with OpenAI evals (the OpenEvals technique implemented by OpenAI). The authors manually inspected 40 issues plus all issues OpenEvals flagged as inaccurate, and found the automated evaluation more pessimistic than a human rater, with the caveat that the reviewers were not experts in this biomedical domain.

Documents: rag-seven-failure-points-2024.pdf, ragas-rag-evaluation-2023.pdf
Expected (cited) documents: rag-seven-failure-points-2024.pdf

## Evidence

- GPT-4 generated the responses; OpenAI evals validated them
  - rag-seven-failure-points-2024.pdf, page 2
  - "We indexed all documents then ran the queries and stored the generated responses using GPT-4. All question and answer pairs were then validated with OpenAI evals 1."

- OpenEvals, 40 issues inspected, automated evaluation more pessimistic (trailing 6 is footnote 6)
  - rag-seven-failure-points-2024.pdf, page 4
  - "OpenEvals technique implemented by OpenAI6. From the gener- ated questions we manually inspected 40 issues and all issues that the OpenEvals flagged as inaccurate. We found that the automated evaluation was more pessimistic than a human rater for this domain."

- caveat
  - rag-seven-failure-points-2024.pdf, page 4
  - "the reviewers were not experts"

- the BioASQ setup (context)
  - rag-seven-failure-points-2024.pdf, page 3
  - "We downloaded 4017 open access documents from the BioASQ dataset and had a total of 1000 questions."

## Arithmetic and notes

A case-insensitive search of the extracted Seven Failure Points text for 'ragas', 'faithful' and 'answer relevan' finds 0 matches.

## What makes it hard

False premise with the Ragas paper selected as a distractor: the Seven Failure Points paper never mentions Ragas, faithfulness or answer relevance (checked in the extracted text). The evaluation is described twice, in the introduction (p2) and in Section 4.3 (p3-4).
