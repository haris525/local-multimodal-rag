# global-7

Question: The Mistral 7B paper reports several efficiency results as multiplicative factors (written like "2x"). List every such factor in the paper and say what each one measures.

Gold answer: Four. (1) 2x: for a sequence length of 16K with window W = 4096, changes made to FlashAttention and xFormers give a 2x speed improvement over a vanilla attention baseline (sliding window attention, Section 2). (2) 8x: on a 32k-token sequence the rolling buffer cache reduces cache memory usage by 8x without hurting model quality (Section 2). (3) More than 3x: on reasoning, comprehension and STEM reasoning (MMLU), Mistral 7B performs like a Llama 2 model more than three times its size (Size and Efficiency, Section 3). (4) 1.9x: on the knowledge benchmarks the equivalent-size compression rate is only 1.9x, likely because its limited parameter count restricts how much knowledge it can store.

Documents: mistral-7b-2023.pdf

## Evidence

- 2x speed improvement (sliding window attention, 16K sequence, W = 4096)
  - mistral-7b-2023.pdf, page 2
  - "In practice, for a sequence length of 16K and W = 4096, changes made to FlashAttention [11] and xFormers [18] yield a 2x speed improvement over a vanilla attention baseline."

- 8x cache memory reduction (rolling buffer cache)
  - mistral-7b-2023.pdf, page 2
  - "On a sequence length of 32k tokens, this reduces the cache memory usage by 8x, without impacting the model quality."

- more than 3x equivalent size on reasoning, comprehension and STEM (MMLU)
  - mistral-7b-2023.pdf, page 4
  - "When evaluated on reasoning, comprehension, and STEM reasoning (specifically MMLU), Mistral 7B mirrored performance that one might expect from a Llama 2 model with more than 3x its size."

- 1.9x on knowledge (the PDF has a curly apostrophe in 7B's, written here as ASCII)
  - mistral-7b-2023.pdf, page 4
  - "On the Knowledge benchmarks, Mistral 7B's performance achieves a lower compression rate of 1.9x, which is likely due to its limited parameter count that restricts the amount of knowledge it can store."

## Arithmetic and notes

A search of the whole extracted text for a number followed by "x", for "times" and for "-fold" finds exactly these four factors plus "preferred 5020 times, compared to 4143 times" (p4), which are vote counts, not factors. Figure 1's "k x W tokens" (written with a multiplication sign) is a formula, not a measured result.

## What makes it global

The factors are spread over Section 2 (p2, attention and cache) and Section 3 (p4, equivalent model sizes). A search for speed or memory finds the first two; the equivalent-size factors are phrased as model size and compression rate and are easily missed.
