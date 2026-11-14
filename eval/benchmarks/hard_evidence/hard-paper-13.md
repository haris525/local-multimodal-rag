# hard-paper-13

Question: LLaVA-1.5's instruction-tuning data mixture lists a size for each source. How many examples come from VQAv2 and GQA together? How many pretraining samples does the paper's comparison table list for LLaVA-1.5, and what do pretraining plus instruction-tuning samples add up to, the figure the abstract rounds to 1.2M?

Gold answer: VQAv2 83K + GQA 72K = 155K of the 665K instruction-tuning mixture (Table 7). Table 3 lists 558K pretraining and 665K fine-tuning samples for LLaVA-1.5, which add up to 1,223K (about 1.22M), rounded to 1.2M in the abstract.

Documents: llava-1-5-improved-baselines-2023.pdf

## Evidence

- VQAv2 83K and GQA 72K (GQA shares the same formatting prompt)
  - llava-1-5-improved-baselines-2023.pdf, page 9
  - "VQAv2 [19] 83K Answer the question using a single word or phrase. GQA [21] 72K"

- mixture total 665K
  - llava-1-5-improved-baselines-2023.pdf, page 9
  - "Total 665K Table 7. Instruction-following Data Mixture of LLaVA-1.5."

- Table 3 columns
  - llava-1-5-improved-baselines-2023.pdf, page 5
  - "Method LLM Image Sample Size VQAv2 GQA VisWiz SciQA- TextVQA Size Pretrain Finetune"

- 558K pretraining, 665K fine-tuning (same in the Vicuna-7B row)
  - llava-1-5-improved-baselines-2023.pdf, page 5
  - "LLaVA-1.5 Vicuna-13B 3362 558K 665K 80.0"

- the abstract's rounded figure
  - llava-1-5-improved-baselines-2023.pdf, page 1
  - "13B checkpoint uses merely 1.2M publicly available data"

## Arithmetic and notes

83K + 72K = 155K. 558K + 665K = 1,223K, about 1.22M, which rounds to 1.2M.

## What makes it hard

Three places in one paper: the appendix data table (p9), the comparison table (p5) and the abstract (p1), plus two sums. The introduction's '600K image-text pairs' is the original LLaVA's rounded figure, which the question rules out by naming the table.
