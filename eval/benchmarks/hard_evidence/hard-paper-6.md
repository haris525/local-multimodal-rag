# hard-paper-6

Question: Gorilla is fine-tuned from one open base LLM. In the LLaVA-1.5 paper's comparison table on academic-task-oriented benchmarks, which model is built on that same LLM, how many pretraining samples does it use, and what are its VQAv2 and TextVQA scores next to LLaVA-1.5 7B's?

Gold answer: Gorilla is fine-tuned from LLaMA-7B. In LLaVA-1.5's Table 3 the model built on LLaMA-7B is IDEFICS-9B: 353M pretraining samples (1M fine-tuning), VQAv2 50.9 and TextVQA 25.9. LLaVA-1.5 7B (Vicuna-7B, 558K pretraining and 665K fine-tuning samples) gets 78.5 and 58.2.

Documents: gorilla-llm-api-calls-2023.pdf, llava-1-5-improved-baselines-2023.pdf, mistral-7b-2023.pdf, react-reasoning-and-acting-2022.pdf
Expected (cited) documents: gorilla-llm-api-calls-2023.pdf, llava-1-5-improved-baselines-2023.pdf

## Evidence

- Gorilla is LLaMA-7B based
  - gorilla-llm-api-calls-2023.pdf, page 2
  - "We then finetune Gorilla, a LLaMA-7B-based model with document retrieval using our dataset."

- and fine-tuned from the base LLaMA-7B
  - gorilla-llm-api-calls-2023.pdf, page 4
  - "We then perform standard instruction finetuning on the base LLaMA-7B model."

- Table 3 column order: Pretrain, Finetune, VQAv2, GQA, VisWiz, SciQA-IMG, TextVQA
  - llava-1-5-improved-baselines-2023.pdf, page 5
  - "Method LLM Image Sample Size VQAv2 GQA VisWiz SciQA- TextVQA Size Pretrain Finetune"

- IDEFICS-9B on LLaMA-7B: 353M / 1M / VQAv2 50.9; the row continues with an en dash for SciQA-IMG and then TextVQA 25.9
  - llava-1-5-improved-baselines-2023.pdf, page 5
  - "IDEFICS-9B [22] LLaMA-7B 2242 353M 1M 50.9 38.4 35.5"

- the other IDEFICS row uses a different LLM
  - llava-1-5-improved-baselines-2023.pdf, page 5
  - "IDEFICS-80B [22] LLaMA-65B"

- LLaVA-1.5 7B: VQAv2 78.5; the row continues '62.0* 50.0 66.8 58.2' (the star is a superscript asterisk in the PDF), so TextVQA is 58.2
  - llava-1-5-improved-baselines-2023.pdf, page 5
  - "LLaVA-1.5 Vicuna-7B 3362 558K 665K 78.5"

- table caption
  - llava-1-5-improved-baselines-2023.pdf, page 5
  - "Table 3. Comparison with SoTA methods on academic-task-oriented datasets."

## Arithmetic and notes

'2242' and '3362' in the extracted rows are 224^2 and 336^2 with the superscript lost.

## What makes it hard

Chain: Gorilla's base model (p2, p4) is the key to a row of LLaVA-1.5 Table 3 (p5). IDEFICS-80B on LLaMA-65B sits in the next row, and the Mistral paper's many Llama rows are distractors.
