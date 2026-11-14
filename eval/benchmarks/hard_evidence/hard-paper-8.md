# hard-paper-8

Question: ReAct's appendix reruns its prompting experiments with the same OpenAI model that Gorilla uses as its GPT-Index retriever. Which model is that, and what HotpotQA exact match and ALFWorld success rate did it reach in ReAct?

Gold answer: The premise is wrong: the papers use different models. ReAct's GPT-3 runs use text-davinci-002 (greedy decoding), while Gorilla's GPT-Index retriever is text-davinci-003. With text-davinci-002, ReAct reaches 30.8 exact match on HotpotQA (500 sampled validation questions) and a 78.4% success rate on ALFWorld (134 unseen tasks), against 29.4 and 70.9 for PaLM-540B.

Documents: react-reasoning-and-acting-2022.pdf, gorilla-llm-api-calls-2023.pdf

## Evidence

- ReAct's GPT-3 model is text-davinci-002
  - react-reasoning-and-acting-2022.pdf, page 14
  - "Table 5: ReAct prompting results using PaLM-540B vs. GPT-3 (text-davinci-002, greedy decoding)."

- HotpotQA EM 30.8 and ALFWorld 78.4 for GPT-3 (PaLM-540B: 29.4 and 70.9)
  - react-reasoning-and-acting-2022.pdf, page 14
  - "PaLM-540B GPT-3 HotpotQA (exact match) 29.4 30.8 ALFWorld (success rate %) 70.9 78.4"

- evaluation subsets
  - react-reasoning-and-acting-2022.pdf, page 14
  - "On HotpotQA, we randomly sample a subset of 500 validation questions. On ALFWorld, we use all 134 unseen validation task instances"

- Gorilla's GPT-Index is text-davinci-003
  - gorilla-llm-api-calls-2023.pdf, page 6
  - "Similarly, GPT-Index refers to the retrieval model text-davinci-003 from OpenAI."

## What makes it hard

False premise corrected by two model names from different papers (ReAct p14, Gorilla p6). An answer that trusts the premise names one model for both.
