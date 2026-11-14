# hard-paper-11

Question: Three papers generate questions with an LLM: Ragas for its WikiEval dataset, Gorilla for its APIBench instructions, and BGE-M3 for its MultiLongDoc data. Which LLM does each use, and how many items result in each case: WikiEval pages, Gorilla instruction-API pairs, and MultiLongDoc training examples?

Gold answer: Ragas: ChatGPT wrote one question for each of 50 Wikipedia pages. Gorilla: GPT-4 (self-instruct) generated 10 instructions per API for 1,645 APIs, 16,450 instruction-API pairs (the limitations section says over 11,000). BGE-M3: GPT-3.5 generated the MultiLongDoc questions, 41,434 training examples (Table 7; 41.4K in Table 8).

Documents: ragas-rag-evaluation-2023.pdf, gorilla-llm-api-calls-2023.pdf, bge-m3-embedding-2024.pdf

## Evidence

- 50 Wikipedia pages
  - ragas-rag-evaluation-2023.pdf, page 4
  - "To construct the dataset, we first selected 50 Wikipedia pages cov- ering events that have happened since the start of 20225."

- ChatGPT writes the questions
  - ragas-rag-evaluation-2023.pdf, page 4
  - "For each of the 50 pages, we then asked ChatGPT to suggest a question that can be answered based on the introductory section of the page"

- GPT-4 generates Gorilla's instructions
  - gorilla-llm-api-calls-2023.pdf, page 4
  - "Guided by the self-instruct paradigm [42], we employed GPT-4 to generate synthetic instruction data."

- 10 pairs per API, 1,645 APIs
  - gorilla-llm-api-calls-2023.pdf, page 4
  - "For each of our 1,645 API datapoints, we sample 3 of 6 corresponding instruction examples to generate a total of 10 instruction-api pairs"

- 16,450 pairs (Figure 3)
  - gorilla-llm-api-calls-2023.pdf, page 4
  - "Self-instruct with in-context examples to generate 16,450 {instruction,API} pairs"

- alternative count in the limitations
  - gorilla-llm-api-calls-2023.pdf, page 9
  - "consisting of over 11,000 instruction-API pairs."

- GPT-3.5 generates the MultiLongDoc questions
  - bge-m3-embedding-2024.pdf, page 3
  - "Then we use GPT- 3.5 to generate questions based on these paragraphs."

- Table 7 columns
  - bge-m3-embedding-2024.pdf, page 15
  - "Language Source #train #dev #test #cropus Avg. Length of Docs"

- Table 7 total row: 41,434 train (the row starts with 'Total' and an en dash)
  - bge-m3-embedding-2024.pdf, page 15
  - "41,434 2,600 3,800 493,709 4,737"

- Table 7 caption
  - bge-m3-embedding-2024.pdf, page 15
  - "Table 7: Specifications of MultiLongDoc dataset."

- Table 8 rounds it to 41.4K
  - bge-m3-embedding-2024.pdf, page 15
  - "MultiLongDoc Multi-Lingual 41.4K"

## Arithmetic and notes

1,645 x 10 = 16,450. In the Ragas quote, '20225' is '2022' followed by footnote 5.

## What makes it hard

Six facts from three papers, one of them in an appendix table (BGE-M3 Table 7, p15). Gorilla gives 16,450 in Figure 3 (p4) and 'over 11,000' in its limitations (p9); both count.
