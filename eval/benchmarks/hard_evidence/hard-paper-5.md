# hard-paper-5

Question: The agent-architectures survey names two LLM benchmarks as examples that are solved in a single iteration with no tool calls. What does Mistral 7B score on each of them, with how many shots is each evaluated in the Mistral paper, and how many points ahead of Llama 2 7B is Mistral 7B on each?

Gold answer: MMLU and GSM8K. Mistral 7B scores 60.1% on MMLU (5-shot) and 52.2% on GSM8K (8-shot, maj@8). Llama 2 7B scores 44.4% and 16.0%, so Mistral 7B is 15.7 and 36.2 points ahead.

Documents: ai-agent-architectures-survey-2024.pdf, mistral-7b-2023.pdf, rag-best-practices-2024.pdf, llava-1-5-improved-baselines-2023.pdf
Expected (cited) documents: ai-agent-architectures-survey-2024.pdf, mistral-7b-2023.pdf

## Evidence

- the two benchmarks are MMLU and GSM8K
  - ai-agent-architectures-survey-2024.pdf, page 10
  - "Many language model benchmarks are designed to be solved in a single iteration, with no tool calls, such as MMLU or GSM8K [3, 10]."

- GSM8K is 8-shot with maj@8
  - mistral-7b-2023.pdf, page 3
  - "Math: GSM8K [10] (8-shot) with maj@8"

- MMLU is 5-shot
  - mistral-7b-2023.pdf, page 3
  - "Popular aggregated results: MMLU [12] (5-shot)"

- Table 2 column order (MMLU first, GSM8K last)
  - mistral-7b-2023.pdf, page 4
  - "Model Modality MMLU HellaSwag WinoG PIQA Arc-e Arc-c NQ TriviaQA HumanEval MBPP MATH GSM8K"

- Mistral 7B: MMLU 60.1%, GSM8K 52.2%
  - mistral-7b-2023.pdf, page 4
  - "Mistral 7B Pretrained 60.1% 81.3% 75.3% 83.0% 80.0% 55.5% 28.8% 69.9% 30.5% 47.5% 13.1% 52.2%"

- Llama 2 7B: MMLU 44.4%, GSM8K 16.0%
  - mistral-7b-2023.pdf, page 4
  - "LLaMA 2 7B Pretrained 44.4% 77.1% 69.5% 77.9% 68.7% 43.2% 24.7% 63.8% 11.6% 26.1% 3.9% 16.0%"

## Arithmetic and notes

60.1 - 44.4 = 15.7; 52.2 - 16.0 = 36.2.

## What makes it hard

Three places: the survey sentence (p10), the Mistral shot settings (p3 list) and Table 2 (p4). The best-practices paper also evaluates on MMLU and LLaVA reports MMBench, so both distract.
