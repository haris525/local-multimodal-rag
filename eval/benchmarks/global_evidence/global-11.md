# global-11

Question: How many APIs does the Gorilla paper say it collected from TorchHub (Torch Hub) and from TensorFlow Hub (TensorHub)? The paper gives different numbers in different places: list every count it states for these two hubs and where each appears, and say which pair adds up, with HuggingFace's 925, to the paper's total of 1,645 API calls.

Gold answer: TorchHub: 94 API calls in the introduction (p2) and in Figure 3 (p4), but 95 models in Section 3.1 (p4) and 95 APIs in Appendix 8.1 (p13). TensorFlow Hub: 696 API calls in the introduction (p2) and 696 APIs in Appendix 8.1 (p13); Section 3.1 (p3) says the v2 hub has 801 models, of which 626 remain after filtering, and Figure 3 (p4) also gives 626. Only 94 + 626 + 925 = 1,645 matches the stated total (95 + 626 + 925 = 1,646 and 94 + 696 + 925 = 1,715).

Documents: gorilla-llm-api-calls-2023.pdf

## Evidence

- introduction: 94 and 696
  - gorilla-llm-api-calls-2023.pdf, page 2
  - "We exhaustively include every API call in TorchHub (94 API calls) and TensorHub (696 API calls);"

- Section 3.1: HuggingFace 925
  - gorilla-llm-api-calls-2023.pdf, page 3
  - "Post filtering, we got a total of 925 models from HuggingFace."

- Section 3.1: TensorFlow Hub v2 801, then 626
  - gorilla-llm-api-calls-2023.pdf, page 3
  - "The latest version (v2) has 801 models in total, and we process all of them. Post filtering out models, whose mode cards had little to no information, we are left with 626 models."

- Figure 3: 1,645 = 94 + 626 + 925
  - gorilla-llm-api-calls-2023.pdf, page 4
  - "Dataset curation: 1,645 API calls. 94 from Torch Hub (exhaustive), 626 from TensorFlow Hub v2 (exhaustive) and 925 from HuggingFace (Top 20 in each domain)."

- Section 3.1 continues on page 4: 95 from Torch Hub
  - gorilla-llm-api-calls-2023.pdf, page 4
  - "TensorFlow Hub, we get 95 models from Torch Hub. We then converted the model cards for each of these 1,645 API calls into a json object"

- Appendix 8.1: 95, 696, 925
  - gorilla-llm-api-calls-2023.pdf, page 13
  - "Torch Hub provides 95 APIs. The second domain, Tensor Hub, is more expansive with a total of 696 APIs. Finally, the most extensive of them all, HuggingFace, comprises 925 APIs."

## Arithmetic and notes

94 + 626 + 925 = 1,645; 95 + 626 + 925 = 1,646; 94 + 696 + 925 = 1,715; 95 + 696 + 925 = 1,716. A search of the whole extracted text for 94, 95, 696, 801, 626 and 925 finds no other dataset count (other hits are table scores). The sentence on page 4 starts on page 3 with "Similar to".

## What makes it global

The four passages are on pages 2, 3-4, 4 (figure) and 13 (appendix). A retrieval finds one or two of them and reports their numbers as the answer; the disagreement shows only when all four are read.
