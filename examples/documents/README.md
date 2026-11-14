# Sample documents

These are optional sample files you can upload to try the app: short research papers about language
models, retrieval-augmented generation (RAG) and agents, real public data as CSV and Excel files, two
images and one plain-text reference document. The app never adds them on its own. Nothing here
appears in your library unless you drag it onto *Add documents* yourself. Every file is either in the
public domain (works of the U.S. federal government, or text from 1787) or licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The tables below list each file's source,
license and the attribution its license asks for. All files were downloaded on 12 November 2026.

## How to use them

- Upload a few files at a time. The text file and the images are ready in seconds. The CSV files
  with thousands of rows get one search entry per row, so they can take a few minutes each. PDFs take
  longer, because the vision model reads every figure.
- Tick the documents you want to search (up to 10), then ask a question.
- Turn on *Deep search* for questions that span several documents or need a calculation. Questions
  about totals, counts and lookups in the CSV and Excel files are usually answered with SQL over the
  real rows.
- Each file below comes with 2 or 3 questions it can answer, plus the expected answer so you can
  check the result. Each section ends with a question that needs two or more files. The same questions,
  with answers checked against the files, are the benchmark set `eval/benchmarks/local_samples.jsonl`
  (docs/GETTING_STARTED.md, section 8).

```text
examples/documents/
  papers/models/        4 PDFs: transformers, an open-weight LLM, an embedding model, a vision-language model
  papers/rag/           3 PDFs: RAG best practices, RAG evaluation, RAG failure modes
  papers/agents-tools/  3 PDFs: ReAct, API calling, a survey of agent architectures
  data/csv/             4 CSVs: earthquakes, causes of death, disaster costs, GDP and internet use
  data/excel/           3 XLSX: retail sales (35 sheets), electricity generation (2 sheets), state populations
  images/               2 PNGs: a temperature chart and a nutrition label
  reference/            1 TXT: the U.S. Constitution
```

20 files, 15.3 MB in total. All papers are unmodified copies of the arXiv version named in the table.
Changes to data files are listed in the notes under each table.

## Papers: models (`papers/models/`)

| File | What it is | Source | License | Attribution | Size | Pages |
|---|---|---|---|---|---|---|
| `formal-algorithms-for-transformers-2022.pdf` | Pseudocode for every part of the transformer (attention, layer norm, encoder-decoder, BERT- and GPT-style models, training and sampling). Stands in for "Attention Is All You Need", which is not openly licensed. | <https://arxiv.org/abs/2207.09238v1> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Mary Phuong and Marcus Hutter (2022). *Formal Algorithms for Transformers*. arXiv:2207.09238v1. Licensed under CC BY 4.0. Unmodified. | 515 KB | 16 |
| `mistral-7b-2023.pdf` | Technical report for the open-weight Mistral 7B model: grouped-query attention, sliding window attention, rolling buffer cache, and benchmarks against Llama 2. | <https://arxiv.org/abs/2310.06825v1> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix and William El Sayed (2023). *Mistral 7B*. arXiv:2310.06825v1. Licensed under CC BY 4.0. Unmodified. | 3.75 MB | 9 |
| `bge-m3-embedding-2024.pdf` | The BGE-M3 embedding model: dense, sparse and multi-vector retrieval from one model, 100+ languages, inputs up to 8,192 tokens. | <https://arxiv.org/abs/2402.03216v5> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian and Zheng Liu (2024). *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216v5. Licensed under CC BY 4.0. Unmodified. | 755 KB | 18 |
| `llava-1-5-improved-baselines-2023.pdf` | LLaVA-1.5, a vision-language model: a CLIP image encoder joined to an LLM by a small MLP, with an ablation table of each design choice. | <https://arxiv.org/abs/2310.03744v2> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Haotian Liu, Chunyuan Li, Yuheng Li and Yong Jae Lee (2023). *Improved Baselines with Visual Instruction Tuning*. arXiv:2310.03744v2. Licensed under CC BY 4.0. Unmodified. | 2.22 MB | 15 |

**Questions to try**

`formal-algorithms-for-transformers-2022.pdf`
- What is the main architectural difference between BERT and GPT/Gopher according to the paper?
  *(GPT and Gopher use unidirectional (causal) attention instead of bidirectional attention, and apply
  layer norms in a slightly different order.)*
- Which algorithm gives the pseudocode for a decoder-only transformer, and which models does the paper
  name as examples? *(Algorithm 10, DTransformer. Examples: GPT-2, GPT-3 and Gopher.)*
- What is RMSnorm? *(A simpler, cheaper version of layer normalisation that sets the mean m and the
  offset beta to 0.)*

`mistral-7b-2023.pdf`
- What are Mistral 7B's window size, context length, number of layers and number of KV heads?
  *(Window 4,096, context length 8,192, 32 layers, 8 KV heads; vocabulary 32,000.)*
- How far can sliding window attention see, and how much memory does the rolling buffer cache save?
  *(About 131K tokens in theory with W = 4,096; the cache uses 8x less memory on a 32k-token sequence.)*
- In the benchmark table, what MMLU score does Mistral 7B get compared with Llama 2 13B?
  *(60.1% vs 55.6%.)*

`bge-m3-embedding-2024.pdf`
- What does the "M3" stand for? *(Multi-Linguality (100+ languages), Multi-Functionality (dense,
  sparse and multi-vector retrieval) and Multi-Granularity (inputs up to 8,192 tokens).)*
- On MIRACL, what average nDCG@10 does M3 reach with dense retrieval alone and with all three methods
  combined? *(69.2 dense, 71.5 combined. BM25 reaches 31.9.)*
- What is self-knowledge distillation in this paper? *(The relevance scores of the three retrieval
  methods are combined into a teacher signal that is used to train each method.)*

`llava-1-5-improved-baselines-2023.pdf`
- Which two simple changes to LLaVA give the stronger baseline? *(A CLIP-ViT-L-336px image encoder
  with an MLP projection, and academic VQA data with response formatting prompts.)*
- How much data and compute does the final 13B model need? *(1.2M publicly available training
  examples, and about 1 day on a single node with 8 A100 GPUs.)*
- What prompt is appended to VQA questions that need short answers? *("Answer the question using a
  single word or phrase.")*

**Cross-document question:** *Mistral 7B and BGE-M3 both handle long inputs. What is the longest input
each one supports, and what technique does each use?* (Mistral: 8,192-token context with a 4,096-token
sliding window, about 131K tokens of theoretical attention span. BGE-M3: inputs up to 8,192 tokens,
plus MCLS, which inserts several CLS tokens into long documents.)

## Papers: RAG (`papers/rag/`)

| File | What it is | Source | License | Attribution | Size | Pages |
|---|---|---|---|---|---|---|
| `rag-best-practices-2024.pdf` | Experiments on every stage of a RAG pipeline (query classification, chunk size, embedding model, hybrid search, HyDE, reranking, repacking, summarization), ending in two recommended recipes. | <https://arxiv.org/abs/2407.01219v1> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng Yin, Changze Lv, Xiaoqing Zheng and Xuanjing Huang (2024). *Searching for Best Practices in Retrieval-Augmented Generation*. arXiv:2407.01219v1. Licensed under CC BY 4.0. Unmodified. | 914 KB | 22 |
| `ragas-rag-evaluation-2023.pdf` | Ragas: reference-free metrics for RAG (faithfulness, answer relevance, context relevance), tested against human judgements on the WikiEval dataset. | <https://arxiv.org/abs/2309.15217v2> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Shahul Es, Jithin James, Luis Espinosa-Anke and Steven Schockaert (2023). *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217v2. Licensed under CC BY 4.0. Unmodified. | 232 KB | 8 |
| `rag-seven-failure-points-2024.pdf` | An experience report from three real RAG projects, listing seven ways RAG systems fail and the lessons learned. | <https://arxiv.org/abs/2401.05856v1> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly and Mohamed Abdelrazek (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System*. arXiv:2401.05856v1. Licensed under CC BY 4.0. Unmodified. | 571 KB | 6 |

**Questions to try**

`rag-best-practices-2024.pdf`
- Which pipeline does the paper recommend for the best performance? *(Query classification, "Hybrid
  with HyDE" retrieval, monoT5 reranking, Reverse repacking and Recomp summarization; average score
  0.483.)*
- In the chunk size experiment, which chunk size had the highest average faithfulness? *(512 tokens,
  97.59. 256 tokens had the highest relevancy, 97.78.)*
- Which alpha worked best for hybrid search, and what does it weight? *(alpha = 0.3. It weights the sparse
  score: S = alpha * S_sparse + S_dense.)*

`ragas-rag-evaluation-2023.pdf`
- Which three quality dimensions does Ragas measure? *(Faithfulness, answer relevance and context
  relevance.)*
- How is the faithfulness score computed? *(An LLM splits the answer into statements and checks each
  one against the context. F = supported statements / total statements.)*
- How often did Ragas agree with human annotators on WikiEval? *(Faithfulness 0.95, answer relevance
  0.78, context relevance 0.70, all better than the GPT Score baseline of 0.72, 0.52 and 0.63.)*

`rag-seven-failure-points-2024.pdf`
- List the seven failure points. *(Missing content, missed the top-ranked documents, not in context,
  not extracted, wrong format, incorrect specificity, incomplete.)*
- What are the paper's two main lessons? *(A RAG system can only be validated while it runs in
  operation, and its robustness evolves over time rather than being designed in at the start.)*
- How large was the BioASQ case study? *(Section 4.3 and Table 1 say 4,017 open-access documents and
  1,000 questions; the introduction and the conclusion say 15,000 documents and 1,000 questions.
  Either answer, with a citation, is right.)*

**Cross-document question:** *Ragas and the best-practices paper both measure faithfulness. How does
each define it, and which of the seven failure points would a low faithfulness score point to?*
(Ragas: the share of answer statements that the context supports. Best practices: whether the response
matches the retrieved texts instead of being hallucinated. A low score suggests the model answered
from outside the context, for example FP1 "missing content", where the documents hold no answer but
the system replies anyway.)

## Papers: agents and tools (`papers/agents-tools/`)

| File | What it is | Source | License | Attribution | Size | Pages |
|---|---|---|---|---|---|---|
| `react-reasoning-and-acting-2022.pdf` | ReAct: the model interleaves reasoning ("thoughts") with tool actions such as Wikipedia search. Tested on HotpotQA, FEVER, ALFWorld and WebShop. 10 pages of paper plus a long appendix. | <https://arxiv.org/abs/2210.03629v3> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan and Yuan Cao (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629v3. Licensed under CC BY 4.0. Unmodified. | 634 KB | 33 |
| `gorilla-llm-api-calls-2023.pdf` | Gorilla: LLaMA fine-tuned to write correct API calls, with a document retriever, plus the APIBench benchmark (HuggingFace, TorchHub, TensorFlow Hub). | <https://arxiv.org/abs/2305.15334v1> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Shishir G. Patil, Tianjun Zhang, Xin Wang and Joseph E. Gonzalez (2023). *Gorilla: Large Language Model Connected with Massive APIs*. arXiv:2305.15334v1. Licensed under CC BY 4.0. Unmodified. | 1.25 MB | 18 |
| `ai-agent-architectures-survey-2024.pdf` | A survey of single-agent and multi-agent architectures for reasoning, planning and tool calling (ReAct, RAISE, Reflexion, AutoGPT+P, LATS, and others). | <https://arxiv.org/abs/2404.11584v1> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Tula Masterman, Sandi Besen, Mason Sawtell and Alex Chao (2024). *The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey*. arXiv:2404.11584v1. Licensed under CC BY 4.0. Unmodified. | 1.66 MB | 13 |

**Questions to try**

`react-reasoning-and-acting-2022.pdf`
- Which three actions does ReAct's Wikipedia API offer? *(search[entity], lookup[string] and
  finish[answer].)*
- By how much does ReAct beat imitation and reinforcement learning methods on ALFWorld and WebShop?
  *(By 34 and 10 points of absolute success rate, using only one or two in-context examples.)*
- In Table 1 (PaLM-540B), which method scores best on HotpotQA and which on FEVER? *(HotpotQA:
  ReAct falling back to CoT-SC, 35.1 exact match. FEVER: CoT-SC falling back to ReAct, 64.6
  accuracy.)*

`gorilla-llm-api-calls-2023.pdf`
- How many API calls does APIBench contain, and from which hubs? *(1,645: 94 from TorchHub, 626 from
  TensorFlow Hub and 925 from HuggingFace, turned into 16,450 instruction-API pairs.)*
- How does the paper check whether a generated API call is correct? *(AST sub-tree matching: the
  generated code is parsed into a syntax tree and matched against the reference API calls. This also
  detects hallucinated APIs.)*
- Which base model is Gorilla fine-tuned from? *(LLaMA-7B, with retriever-aware training.)*

`ai-agent-architectures-survey-2024.pdf`
- What is the difference between vertical and horizontal multi-agent architectures? *(Vertical: one
  lead agent, and the other agents report to it. Horizontal: all agents are equals in one shared
  discussion and can volunteer for tasks.)*
- What does RAISE add to ReAct? *(Memory: a scratchpad for short-term memory and a dataset of similar
  past examples for long-term memory.)*
- Which four tools can AutoGPT+P choose from? *(Plan Tool, Partial Plan Tool, Suggest Alternative Tool
  and Explore Tool.)*

**Cross-document question:** *How do ReAct and Gorilla teach a model to use tools, and how does the
survey classify ReAct?* (ReAct mainly prompts a frozen model with a few examples of interleaved
thoughts and actions; the paper also fine-tunes smaller PaLM models on 3,000 ReAct trajectories.
Gorilla fine-tunes LLaMA-7B on 16,450 instruction-API pairs. The survey lists ReAct as a single-agent
method.)

## Data: CSV (`data/csv/`)

| File | What it is | Source | License | Attribution | Size | Rows |
|---|---|---|---|---|---|---|
| `usgs-earthquakes-m5-plus-2025.csv` | Every magnitude 5.0+ event worldwide in 2025 from the USGS earthquake catalog: time, latitude, longitude, depth, magnitude, place name, status and error estimates (22 columns). | [USGS FDSN event query](https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2025-01-01&endtime=2026-01-01&minmagnitude=5&orderby=time-asc) | Public domain ([USGS policy](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)) | Not required. Suggested credit: U.S. Geological Survey, Earthquake Hazards Program, ANSS Comprehensive Earthquake Catalog (ComCat), downloaded 12 November 2026. | 406 KB | 2,130 |
| `cdc-leading-causes-of-death-us-1999-2017.csv` | Deaths and age-adjusted death rates (per 100,000) for the 10 leading causes of death in the U.S., by state and year, 1999-2017. | <https://data.cdc.gov/NCHS/NCHS-Leading-Causes-of-Death-United-States/bi63-dtpu> | Public domain ([U.S. Government Works](https://www.usa.gov/government-works), as stated on data.cdc.gov) | Not required. Suggested credit: National Center for Health Statistics, *NCHS - Leading Causes of Death: United States*, data.cdc.gov. | 843 KB | 10,868 |
| `noaa-us-billion-dollar-disasters-1980-2024.csv` | Every U.S. weather and climate disaster from 1980 to 2024 that cost at least $1 billion: name, type, start and end dates, cost and deaths. | <https://www.ncei.noaa.gov/access/billions/> (file: [events-US-1980-2024.csv](https://www.ncei.noaa.gov/access/billions/events-US-1980-2024.csv)) | Public domain ([U.S. Government Works](https://www.usa.gov/government-works)) | Not required. NCEI's suggested citation: NOAA National Centers for Environmental Information (NCEI) U.S. Billion-Dollar Weather and Climate Disasters (2025). https://www.ncei.noaa.gov/access/billions/, DOI: 10.25921/stkw-7w73. | 36 KB | 403 |
| `world-bank-gdp-per-capita-internet-use-2000-2024.csv` | GDP per capita (current US$) and internet users (% of population) for 217 economies, 2000-2024, with each economy's World Bank region and income group. | World Bank API: [NY.GDP.PCAP.CD](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD), [IT.NET.USER.ZS](https://data.worldbank.org/indicator/IT.NET.USER.ZS), [country list](https://api.worldbank.org/v2/country?format=json&per_page=400) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([World Bank dataset terms](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets)) | The World Bank: World Development Indicators: GDP per capita (current US$) (NY.GDP.PCAP.CD) and Individuals using the Internet (% of population) (IT.NET.USER.ZS); country classifications from the World Bank API. Licensed under CC BY 4.0. Modified: combined and filtered as described in the notes below. | 392 KB | 5,292 |

Notes:

- **USGS** and **CDC** files are unmodified downloads. The USGS file includes one event of type
  `landslide`; every other row is an earthquake.
- **NOAA**: the original file starts with two title lines. They were removed so that the first line
  is the header, and their information was moved into the header: costs are in millions of U.S.
  dollars, and the CPI-adjusted cost is in 2024 dollars. Nothing else was changed. The file covers
  1980-2024.
- **World Bank**: built from the World Bank API (data last updated 8 October 2026 for GDP and
  11 November 2026 for internet use). Changes: the two indicators were joined with the World Bank
  country list (region and income group), regional and income aggregates were removed, years were
  limited to 2000-2024, rows with neither value were dropped, GDP was rounded to cents and internet use
  to 4 decimals. Both indicators are marked "CC BY-4.0" in the World Bank's indicator metadata.
  Countries use World Bank names, for example "Russian Federation" and "Egypt, Arab Rep.".

**Questions to try**

`usgs-earthquakes-m5-plus-2025.csv`
- What was the strongest earthquake in 2025, and where and when did it happen? *(Magnitude 8.8,
  Kamchatka Peninsula, Russia, 29 July 2025 at 23:24 UTC, 35 km deep.)*
- How many events of magnitude 6.0 or more, and of 7.0 or more, are there? *(145 and 16, out of 2,130
  events.)*
- Which month had the most M5+ events? *(July 2025, with 413, mostly aftershocks of the Kamchatka
  earthquake.)*

`cdc-leading-causes-of-death-us-1999-2017.csv`
- What were the three leading causes of death in the United States in 2017, with death counts?
  *(Heart disease 647,457, cancer 599,108, unintentional injuries 169,936. All causes: 2,813,503.)*
- Which state had the highest age-adjusted death rate from unintentional injuries in 2017? *(West
  Virginia, 100.3 per 100,000, followed by Ohio at 75.1 and Kentucky at 72.9.)*
- How did the U.S. age-adjusted death rate for heart disease change from 1999 to 2017? *(From 266.5
  to 165.0 per 100,000. Deaths fell from 725,192 to 647,457.)*

`noaa-us-billion-dollar-disasters-1980-2024.csv`
- Which disaster had the highest CPI-adjusted cost? *(Hurricane Katrina, August 2005: $201.3 billion
  in 2024 dollars, 1,833 deaths.)*
- How many billion-dollar disasters began in 2024, and what did they cost in total? *(27 events,
  about $182.7 billion, 568 deaths.)*
- Which disaster type occurs most often? *(Severe storm: 203 of the 403 events.)*

`world-bank-gdp-per-capita-internet-use-2000-2024.csv`
- Which economy had the highest GDP per capita in 2024? *(Monaco, $288,002, then Liechtenstein
  $220,167 and Bermuda $142,250.)*
- How did internet use in India change between 2013 and 2024? *(From 12.3% to 64.9% of the
  population.)*
- What was the average share of internet users by income group in 2023? *(Low income 21.9%,
  lower middle income 55.3%, upper middle income 78.4%, high income 91.0%. These are simple averages
  over the economies that have data.)*

**Cross-document question:** *Which country had the most magnitude 5+ earthquakes in 2025, and what
were its GDP per capita and internet use in 2024?* (Russia, with about 500 events, mostly from the
Kamchatka sequence. The World Bank file lists it as "Russian Federation": $14,961.90 and 94.4%.)
For a second one, try: *In 2017, how many deaths did billion-dollar disasters cause, compared with U.S.
deaths from unintentional injuries?* (NOAA: 3,280 deaths from 19 disasters, 2,981 of them from
Hurricane Maria. CDC: 169,936.)

## Data: Excel (`data/excel/`)

| File | What it is | Source | License | Attribution | Size | Sheets / rows |
|---|---|---|---|---|---|---|
| `census-monthly-retail-sales-1992-2026.xlsx` | Monthly U.S. retail and food services sales by kind of business (NAICS code), in millions of dollars, seasonally adjusted and not adjusted. One sheet per year. | <https://www.census.gov/retail/mrts/www/mrtssales92-present.xlsx> (Monthly Retail Trade Survey) | Public domain ([U.S. Government Works](https://www.usa.gov/government-works)) | Not required. Suggested credit: U.S. Census Bureau, Monthly Retail Trade Survey, *Estimates of Monthly Retail and Food Services Sales by Kind of Business*, downloaded 12 November 2026. | 441 KB | 35 sheets (2026 back to 1992), about 120 rows each |
| `eia-us-electricity-generation-by-source-1949-2026.xlsx` | U.S. electricity net generation by source (coal, natural gas, nuclear, hydro, wind, solar and more), in million kWh. From the Monthly Energy Review, October 2026, Table 7.2a. | <https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T07.02A> | Public domain ([EIA copyright and reuse](https://www.eia.gov/about/copyrights_reuse.php)) | Not required. Suggested credit: U.S. Energy Information Administration, *Monthly Energy Review*, October 2026, Table 7.2a. | 88 KB | 2 sheets: "Monthly Data" (Jan 1973-Jul 2026, 643 rows) and "Annual Data" (1949-2025, 77 rows) |
| `census-state-population-estimates-2020-2025.xlsx` | Resident population of the U.S., its four regions, the 50 states, DC and Puerto Rico: the April 2020 base and July 1 estimates for 2020-2025 (Vintage 2025, table NST-EST2025-POP). | <https://www2.census.gov/programs-surveys/popest/tables/2020-2025/state/totals/NST-EST2025-POP.xlsx> | Public domain ([U.S. Government Works](https://www.usa.gov/government-works)) | Not required. Census Bureau's suggested citation: Annual Estimates of the Resident Population for the United States, Regions, States, District of Columbia, and Puerto Rico: April 1, 2020 to July 1, 2025 (NST-EST2025-POP). Source: U.S. Census Bureau, Population Division. Release Date: January 2026. | 17 KB | 1 sheet, 57 geographic rows |

All three workbooks are unmodified downloads. The retail workbook has 35 sheets, so it takes the
longest of the three to process. The two Census files are U.S. federal government works, which are
not protected by copyright in the United States (17 U.S.C. section 105).

**Questions to try**

`census-monthly-retail-sales-1992-2026.xlsx`
- What were total U.S. retail and food services sales, not seasonally adjusted, in December 2025?
  *($780,707 million, the highest month of 2025.)*
- What were total 2025 sales for electronic shopping and mail-order houses, and how does that compare
  with 2019? *($1,278,486 million in 2025 against $639,449 million in 2019, about twice as much.)*
- What were total retail and food services sales for all of 2025? *($8,350,933 million, about
  $8.35 trillion, not adjusted.)*

`eia-us-electricity-generation-by-source-1949-2026.xlsx`
- How much electricity did the U.S. generate in 2025, and what shares came from natural gas and from
  nuclear? *(4,429,502 million kWh, about 4,430 TWh. Natural gas 40.8%, nuclear 17.7%.)*
- In which year did natural gas first generate more electricity than coal? *(2016: 1,379,271 against
  1,239,149 million kWh.)*
- How much did solar generation grow from 2024 to 2025? *(From 219,834 to 295,671 million kWh, up
  34.5%.)*

`census-state-population-estimates-2020-2025.xlsx`
- What was the population of Texas on July 1, 2025, and how much did it grow from 2024? *(31,709,821,
  up 391,243, the largest gain of any state.)*
- Which state grew fastest in percentage terms from July 2024 to July 2025? *(South Carolina, +1.46%,
  just ahead of Idaho at +1.44%.)*
- What was the total U.S. population on July 1, 2025, and which region is the largest? *(341,784,857.
  The South, with 133,833,983.)*

**Cross-document question:** *About how much electricity was generated, and how much was spent in
retail stores and restaurants, per U.S. resident in 2025?* (About 12,960 kWh per person
(4,429,502 million kWh / 341,784,857 people) and about $24,430 per person ($8,350,933 million /
341,784,857).)

## Images and reference text (`images/`, `reference/`)

| File | What it is | Source | License | Attribution | Size | Pages / rows |
|---|---|---|---|---|---|---|
| `images/nasa-giss-global-temperature-chart.png` | Line chart of the global annual mean surface temperature anomaly, 1880-2025, relative to 1951-1980, with a smoothed trend and an uncertainty band (1130 by 600 pixels). | <https://data.giss.nasa.gov/gistemp/graphs_v4/> ([image](https://data.giss.nasa.gov/gistemp/graphs_v4/graph_data/Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.png)) | Public domain (NASA work; see [NASA image guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/) and the "Citation and Image Credits" section of [GISTEMP](https://data.giss.nasa.gov/gistemp/)) | Credit requested by GISS: NASA's Goddard Institute for Space Studies (short form: NASA GISS/GISTEMP). | 125 KB | 1 image |
| `images/fda-nutrition-facts-label-2016.png` | The FDA's 2016 example Nutrition Facts label: servings, calories, nutrients and % Daily Values (752 by 1444 pixels). Good for testing text extraction from an image. | <https://commons.wikimedia.org/wiki/File:FDA_Nutrition_Facts_Label_2016.png> (original by the U.S. Food and Drug Administration) | Public domain ([PD-USGov-FDA](https://commons.wikimedia.org/wiki/Template:PD-USGov-FDA); [U.S. Government Works](https://www.usa.gov/government-works)) | Not required. Suggested credit: U.S. Food and Drug Administration, via Wikimedia Commons. | 397 KB | 1 image |
| `reference/us-constitution-1787.txt` | The text of the U.S. Constitution as signed in 1787: Preamble, Articles I-VII and the signers. The amendments are not included. | Project Gutenberg eBook #5, <https://www.gutenberg.org/ebooks/5> | Public domain (written in 1787) | Not required. Text from Project Gutenberg eBook #5. | 27 KB | 593 lines |

Notes: the chart is the version GISS published on 12 November 2026 (data through 2025). The
Constitution file contains only the Constitution text from the Project Gutenberg edition. The Project
Gutenberg header, license, footer and transcriber's notes were removed, and line endings were changed
to LF. The text keeps that edition's original spelling, for example "defence".

**Questions to try**

`images/nasa-giss-global-temperature-chart.png`
- Which year has the highest annual mean on the chart, and roughly how warm was it? *(2024, about
  +1.28 C above the 1951-1980 average. 2025 is about +1.18 C.)*
- What baseline period does the chart use, and what does the red line show? *(1951-1980. The red line
  is a LOWESS smoothing of the annual means.)*

`images/fda-nutrition-facts-label-2016.png`
- How many calories are in one serving, and how many servings are in the container? *(230 calories,
  8 servings of 2/3 cup (55 g).)*
- How much sodium and added sugar does one serving contain, with % Daily Value? *(Sodium 160 mg, 7%.
  Added sugars 10 g, 20%.)*

`reference/us-constitution-1787.txt`
- What are the minimum ages for a Representative, a Senator and the President? *(25, 30 and 35 years.
  Representatives must also have been citizens for 7 years, Senators for 9.)*
- How many states had to ratify the Constitution for it to take effect? *(Nine, under Article VII.)*
- What vote does each house need to pass a bill over the President's veto? *(Two thirds.)*

**Cross-document question:** *The NASA chart shows warming since 1980. Does the NOAA disasters CSV
show more billion-dollar disasters per decade over the same period?* (Yes: 33 in the 1980s, 57 in
the 1990s, 67 in the 2000s, 131 in the 2010s and 115 in 2020-2024 alone. Costs are CPI-adjusted, but
the counts do not account for growth in population or property value.)

## Why some well-known papers are missing

Only files with a license that clearly allows redistribution are included. Several famous papers are
on arXiv only under arXiv's default license, which allows arXiv itself to distribute them but nobody
else. They were replaced with openly licensed papers on the same topic:

- "Attention Is All You Need" (1706.03762) was replaced by *Formal Algorithms for Transformers*.
- The original RAG paper by Lewis et al. (2005.11401) and the RAG survey by Gao et al. (2312.10997)
  were replaced by *Searching for Best Practices in RAG*.
- Toolformer (2302.04761) was replaced by *Gorilla*.
