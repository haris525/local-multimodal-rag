# hard-paper-4

Question: The agent-architectures survey states that on HotpotQA the ReAct method hallucinated only 6% of the time, compared with 14% for chain-of-thought (CoT). In ReAct's own human analysis, which group of trajectories do those two percentages describe, how many trajectories were labeled in total, and what share of the failures did hallucination account for in ReAct and in CoT?

Gold answer: In ReAct's Table 2, 6% and 14% are the false-positive rates within the success mode: correct answers reached with a hallucinated reasoning trace or facts (true positives are 94% and 86%). The study labeled 200 trajectories, 50 correct and 50 incorrect from each of ReAct and CoT. Among failures, hallucination accounted for 0% for ReAct and 56% for CoT (CoT's major failure mode); ReAct's largest failure mode was reasoning error at 47%.

Documents: ai-agent-architectures-survey-2024.pdf, react-reasoning-and-acting-2022.pdf

## Evidence

- the survey's claim
  - ai-agent-architectures-survey-2024.pdf, page 4
  - "When evaluated on the HotpotQA dataset, the ReAct method only hallucinated 6% of the time, compared to 14% using the chain of thought (CoT) method"

- 6% and 14% are false positives under Success
  - react-reasoning-and-acting-2022.pdf, page 6
  - "Success True positive Correct reasoning trace and facts 94% 86% False positive Hallucinated reasoning trace or facts 6% 14%"

- hallucination as a failure mode: 0% ReAct, 56% CoT
  - react-reasoning-and-acting-2022.pdf, page 6
  - "Hallucination Hallucinated reasoning trace or facts 0% 56%"

- the text confirms the 6%/14% are in success mode and 56% is CoT's major failure mode
  - react-reasoning-and-acting-2022.pdf, page 6
  - "Hallucination is a serious problem for CoT, resulting in much higher false positive rate than ReAct (14% vs. 6%) in success mode, and make up its major failure mode (56%)."

- 200 trajectories
  - react-reasoning-and-acting-2022.pdf, page 6
  - "we randomly sampled 50 trajectories with correct and incorrect answers (judged by EM) from ReAct and CoT respectively (thus 200 examples in total)"

- ReAct reasoning error 47% (context)
  - react-reasoning-and-acting-2022.pdf, page 6
  - "Reasoning error Wrong reasoning trace (including failing to recover from repetitive steps) 47% 16%"

## Arithmetic and notes

50 x 2 outcomes x 2 methods = 200.

## What makes it hard

The survey reads a success-mode false-positive rate as an overall hallucination rate. The answer needs the survey sentence (p4), ReAct's Table 2 and the sampling sentence (both p6).
