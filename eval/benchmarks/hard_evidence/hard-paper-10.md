# hard-paper-10

Question: The best-practices RAG paper scores RAG capabilities with four metrics taken from RAGAs. Which of the four is not one of the three quality aspects defined in the Ragas paper itself, which LLM serves as the judge in the best-practices paper, and which model did the Ragas authors use to evaluate all their prompts?

Gold answer: Answer Correctness. The four metrics are Faithfulness, Context Relevancy, Answer Relevancy and Answer Correctness; the Ragas paper defines only faithfulness, answer relevance and context relevance. The best-practices paper uses GPT-4 as the judge, while the Ragas paper evaluates all prompts with gpt-3.5-turbo-16k.

Documents: rag-best-practices-2024.pdf, ragas-rag-evaluation-2023.pdf

## Evidence

- the four RAGAs metrics
  - rag-best-practices-2024.pdf, page 21
  - "Towards RAG capabilities evaluation, we adopt four metrics from RAGAs, including Faithfulness, Context Relevancy, Answer Relevancy, and Answer Correctness."

- same list in Section 4.1
  - rag-best-practices-2024.pdf, page 11
  - "employing the metrics recommended in RA- GAs [51], including Faithfulness, Context Relevancy, Answer Relevancy, and Answer Correctness."

- GPT-4 judge
  - rag-best-practices-2024.pdf, page 22
  - "All these metrics are evaluated using the RAGAs framework, with GPT-4 serving as the judge."

- Ragas has three quality aspects
  - ragas-rag-evaluation-2023.pdf, page 3
  - "We focus in particular three quality aspects, which we argue are of central importance."

- aspect 1: faithfulness
  - ragas-rag-evaluation-2023.pdf, page 3
  - "First, Faithfulness refers to the idea that the an- swer should be grounded in the given context."

- aspects 2 and 3: answer relevance and context relevance
  - ragas-rag-evaluation-2023.pdf, page 3
  - "Sec- ond, Answer Relevance refers to the idea that the generated answer should address the actual ques- tion that was provided. Finally, Context Relevance refers to the idea that the retrieved context should be focused"

- Ragas prompts run on gpt-3.5-turbo-16k (trailing 2 is footnote 2)
  - ragas-rag-evaluation-2023.pdf, page 3
  - "In our implementation and experiments, all prompts are evaluated using the gpt-3.5-turbo-16k model, which is available through the OpenAI API2."

## Arithmetic and notes

{Faithfulness, Context Relevancy, Answer Relevancy, Answer Correctness} minus {faithfulness, answer relevance, context relevance} = {Answer Correctness}.

## What makes it hard

A set difference across two papers plus two model names in different sections (best practices p21-22, Ragas p3).
