# global-14

Question: How many in-context examples (few-shot exemplars) does ReAct put in its prompts for each of its four benchmarks: HotpotQA, FEVER, ALFWorld and WebShop?

Gold answer: HotpotQA: 6 and FEVER: 3, manually composed ReAct-format trajectories from the training set (more examples did not improve performance). ALFWorld: 2 per prompt; the authors annotate 3 trajectories per task type and build 6 prompts per task type from the permutations of 2 of the 3. WebShop: one (one-shot prompting). The paper sums this up as learning from one to six in-context examples.

Documents: react-reasoning-and-acting-2022.pdf

## Evidence

- HotpotQA 6, FEVER 3
  - react-reasoning-and-acting-2022.pdf, page 4
  - "ReAct Prompting For HotpotQA and Fever, we randomly select 6 and 3 cases2 from the training set and manually compose ReAct-format trajectories to use as few-shot exemplars in the prompts."

- footnote 2
  - react-reasoning-and-acting-2022.pdf, page 4
  - "2We find more examples do not improve performance."

- one to six in-context examples overall
  - react-reasoning-and-acting-2022.pdf, page 4
  - "ReAct shows strong generalization to new task instances while learning solely from one to six in-context examples"

- ALFWorld: three annotated trajectories per task type
  - react-reasoning-and-acting-2022.pdf, page 7
  - "To prompt ReAct, we randomly annotate three trajectories from the training set for each task type,"

- ALFWorld: 2 per prompt, 6 prompts per task type
  - react-reasoning-and-acting-2022.pdf, page 7
  - "For robustness, we construct 6 prompts for each task type through each permutation of 2 annotated trajectories from the 3 we annotate."

- WebShop: one-shot
  - react-reasoning-and-acting-2022.pdf, page 8
  - "On Webshop, one-shot Act prompting already performs on par with IL and IL+RL methods."

- introduction: two-shot (ALFWorld) or one-shot (WebShop)
  - react-reasoning-and-acting-2022.pdf, page 3
  - "On ALFWorld and WebShop, two or even one-shot ReAct prompting is able to outperform imitation or reinforcement learning methods"

- the WebShop prompt (one example)
  - react-reasoning-and-acting-2022.pdf, page 22
  - "Table 6: Prompts used for the Webshop experiments in Section 4. We use the prompt on the left for Act, and use the prompt on the right (add reasoning) for ReAct."

## Arithmetic and notes

"cases2" is "cases" followed by footnote marker 2. A search of the whole extracted text for "shot", "exemplar", "in-context example" and "annotate" finds no other prompt size. ALFWorld's 6 is the number of prompts per task type, not of examples per prompt.

## What makes it global

The counts are given once each, in three sections on pages 4, 7 and 8, and phrased differently ("6 and 3 cases", "permutation of 2 annotated trajectories", "one-shot"). An answer from one chunk gets HotpotQA and FEVER and guesses the rest.
