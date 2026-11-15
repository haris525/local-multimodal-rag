# Analysis tools and measured performance

The Deep search agent uses a ReAct loop: choose a tool, inspect its result, and choose the next step.
It already has hybrid search, literal `grep`, passage reads, table listing/descriptions, column profiles,
read-only DuckDB SQL, and optional `run_python`. These serve different jobs. Replacing every operation
with a dataframe would bypass useful retrieval and database optimizations.

`run_python` is available to every Deep search agent when `RAG_DEEP_PYTHON=on` (or explicitly requested
with the `request` setting). It runs in a separate Linux container, with selected-document inputs,
no credentials, no document-volume mounts, and one run at a time. The default installation leaves it off;
[Mac instructions](MACOS.md#enable-python-analysis-for-deep-search) include the enable/build/check steps.

To execute an analysis in the app, select the data document, turn on **Deep search**, and ask explicitly:
"Run Python with pandas on the selected CSV. Group by region and show the computed count, mean and
standard deviation for each numeric column." Look for a `run_python` tool step and evidence from its
output. Standard search and **Read all** do not expose Python execution; they can generate example
code without running it. Deep search makes the tool available, but the model still chooses which
tools to call, so a code block or an "expected output" paragraph alone does not establish execution.

## Input files inside Docker

The sandbox container is created by Compose; each Python run gets a fresh working directory inside
that container. The app transfers only the selected documents' extracted tables and text into its
`data/` directory. These are per-run files, not paths on the operator's Mac or Windows machine.
Original uploads such as CSV and XLSX are represented by their extracted DuckDB tables, so reading
the original upload filename with `pandas.read_csv()` or `pandas.read_excel()` fails there.

The tool description supplies the exact input paths before the model writes code. The file
`data/manifest.json` supplies their table names and schemas. Use its `tables_file` value to open a
database with `duckdb.connect(path, read_only=True)` and convert a filtered or aggregated query to
pandas with `.df()`. A document without tables has a null `tables_file`; text passages, when
available, are at its `text_file`. Do not construct paths from document IDs or upload names.
Files and Python variables are removed after each run; the same selected inputs are supplied again
for subsequent calls. This behavior ships with the repository and needs no machine-specific mounts.

## Workload guidance

| Need | First tool to consider |
|---|---|
| Find relevant passages by meaning/keywords | Hybrid `search` |
| Find an exact code, identifier, or phrase in passages | `grep`, then `read` for surrounding evidence |
| Learn a table's schema, mixed types, names, or totals | `list_tables`, `describe_table`, `profile_column`; `peek` for ambiguous headers |
| Filter/count/sum/group/join tables within one document database | Read-only `sql`; return aggregated results |
| Combine different document databases or do statistical analysis | `run_python`, using DuckDB to reduce data before materializing results |
| Build a dataframe transformation pipeline | Polars expressions/lazy scans where the input format supports them |
| Feed a pandas-based library or manipulate a small result | pandas; use vectorized operations |

The Python tool description now advises early filtering/column selection, DuckDB for table operations,
Polars where useful, and vectorized work. Libraries remain available for the agent to choose from;
there is no hard-coded rule forcing every Python calculation into one engine.
[Polars documents its lazy predicate/projection pushdown](https://docs.pola.rs/user-guide/lazy/optimizations/).
[pandas' performance guide recommends removing Python loops and using vectorization](https://pandas.pydata.org/docs/user_guide/enhancingperf.html).

## Measurements on this Mac, November 14, 2026

Apple M5 Pro / 64 GB unified memory; Docker Desktop Linux ARM64. The real sandbox was capped at
**2 GiB and two CPUs**, with two library threads. Installed versions: DuckDB 2.0.0, Polars 2.1.0,
pandas 3.1.1, PyArrow 26.0.1. The benchmark uses the actual rag-to-sandbox request path and synthetic
data, with no model or private documents.

Each engine/workload/size runs in a fresh child process. Its three measured queries read identical
Parquet files, selecting the required columns; pandas also uses the Parquet filter API for the
selective filter. Polars uses lazy scans. Measurements include input reading and materializing the
small result, exclude imports and data generation, and use a warm filesystem cache. Integer results
must match across all three engines. Sizes: 1,000, 100,000, and 1,000,000 rows; four workloads each.
All **36 engine/workload/size comparisons** returned matching results.

Median query time for **1,000,000 rows**, milliseconds:

| Workload | DuckDB | Polars | pandas |
|---|---:|---:|---:|
| Selective numeric filter + count/sum | 6.98 | 7.21 | 13.28 |
| Literal text match + count/sum | 2.13 | 1.61 | 41.26 |
| Grouped integer aggregation | 5.28 | 6.27 | 20.99 |
| Many-to-one join + grouped aggregation | 8.07 | 15.27 | 62.57 |

Linux process peak RSS for the join workload: DuckDB **120 MiB**, Polars **166 MiB**, pandas **275 MiB**.
This includes the child's imports and temporary allocations; it is not the entire container's memory.
Imports alone took about 0.19-0.20 s / 0.22-0.24 s / 0.33-0.35 s, respectively, in these runs.
The [raw results](benchmarks/analysis-mac-2026-11-14.json) include every timing, process duration,
peak RSS, package version, and checked output.

These results support retaining DuckDB as the normal table-query path, using Polars for suitable
Python transformations, and keeping pandas for compatibility and small outputs. They do **not** prove
one engine always wins. Data is synthetic, mostly low-cardinality, narrow, and free of nulls; it does
not cover arbitrary joins, regex patterns, wide spreadsheets, messy dates, or larger-than-memory data.
Parquet results do not directly measure opening the repository's existing DuckDB document files.
The model's reasoning, retrieval network calls, input copying, Python startup, and evidence handling
can outweigh a few milliseconds saved in a query. Semantic search/reranking needs its own quality and
latency evaluation. Recheck representative user documents before making broader routing changes.

## Reproduce

With the Mac stack and Python sandbox running:

```sh
uv run --python 3.12 scripts/benchmark_analysis.py --macos --output /tmp/analysis-benchmark.json
```

The normal NVIDIA/Linux stack uses the same command without `--macos`. Use `--rows` and `--repeats`
to vary the synthetic load within the helper's bounds. The benchmark never reads user documents and
leaves no benchmark dataset in the sandbox after the run.
