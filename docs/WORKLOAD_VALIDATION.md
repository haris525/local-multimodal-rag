# Large documents and memory validation

`scripts/validate_workload.py` exercises an already running application with synthetic data. It creates
its own temporary ordinary account through the server's operator interface, then deletes that account
and its documents through the normal purge workflow. It does not use an existing user's credentials,
documents, chats or memories. Run it on the machine hosting Docker; it needs operator access there.
Inference can compete with your chats while the test runs.

Start the application using [Getting started](GETTING_STARTED.md), or [the Mac guide](MACOS.md).
From the repository folder, run:

```sh
uv run --python 3.12 --with httpx --with openpyxl --with psutil \
  scripts/validate_workload.py --macos --output /tmp/rag-workload.json
```

For the Linux/Windows Compose deployment, omit `--macos`. In PowerShell put the command on one line
and choose an output path such as `$env:TEMP/rag-workload.json`. The runner needs Python packages,
Docker and a running application; it does not download model weights. An existing ingest development
environment also has the required packages. The report is JSON; open it in a text editor. A successful
run prints `Synthetic workload passed; temporary account and documents removed.`

The defaults generate:

| Fixture | Checks |
|---|---|
| 250,000-row CSV | Exact full-table row count and integer sums through the real signed RAG → SQL client, including rows beyond the 20,000-row retrieval indexing cap |
| 50,000-row UTF-16 CSV | Semicolon delimiter, Unicode, quoted multiline fields, decimal commas and blank values; successful ingestion and subsequent reads |
| 12,000-row workbook | Merged title, notes, repeated headers, leading-zero IDs, dates, currency formats, missing values and uncached formulas; successful ingestion and warnings recorded |
| 100-page text PDF | Repeated headers and table-like text, plus a searchable marker on the final page |
| Approximately 1 MiB text document | Repeated prose, Unicode and a searchable end-of-file marker |
| Deliberately truncated PDF | A clear failed document rather than a worker crash |
| Concurrent reindex/read | Search results contain a single published generation throughout reindexing; old and new SQL snapshots remain readable afterward |
| Repeated 100-page PDF | Reindex after the large-sheet workload; final-page retrieval and the earlier generation remain available |
| Repeated reads | At least 30 cycles of searches and chunk paging; minimum total workload duration 10 minutes |

Change size with `--rows`, `--pages`, `--minutes`; `--document-timeout` defaults to 1,800 seconds per
document/reindex. Ten minutes is a minimum including ingestion, not an overall timeout. Large files
can take substantially longer. Each query uses grep for predictable retrieval without an additional
language-model answer. Ingestion still runs the real parser, tokenizer, summary model and embeddings.
Add `--python` when the optional sandbox is enabled to download both old and new table generations and
check the same full-table aggregates with pandas in real isolated Python runs.

The runner waits five seconds between samples of this Compose project's container working sets, host
available memory and swap. On a Mac it also samples native inference process RSS using the documented
local listener ports (18080/18083), or recorded background process groups. `native_rss_observed: false`
means the native process RSS could not be collected.
Metal allocations are not all represented by process RSS; Docker limits do not cap native inference.
The report contains per-service peaks, a time series by test stage, service health, restart counts,
out-of-memory flags and cleanup status. Tokens, passwords, personal paths, original document text and
account identifiers are excluded from the report. Review every report before publishing it.
On cgroup v2, parser/sandbox child OOM counters are checked too; `child_oom_kills: null` means that
counter was unavailable, so Docker's main-process flag is the remaining OOM check.
Docker calls add sampling overhead; use each sample's actual `elapsed_s` for timing. Sampled peaks can
miss brief spikes. OOM counters are compared with the baseline, so a historical sandbox containment
test does not count as a new OOM during this workload.

The runner exits unsuccessfully for incorrect aggregates, mixed search generations, missing final
content, unexpected parsing outcomes, monitoring failures, an unhealthy/restarted service or incomplete
cleanup. A normal interruption runs cleanup. If the process is killed forcibly or Docker stops, it may
leave its `validation-…` account; use the app's admin Users page to delete that account and its data.
Never bulk-delete other accounts to clean up a test.

The worker embeds and upserts one batch at a time; its vector memory no longer grows with every row.
PDF stage queues hold at most eight items. An idle parser above `INGEST_PARSER_CACHE_MB` (default
1,024 MiB) is recycled after its result reaches the worker, before embedding or the next file. Set it
to `0` to recycle after every parse; lower cache budgets trade model reload time for memory headroom.
This is a cache threshold, not a limit on memory needed during one parse; the container cap still applies.

This is finite synthetic coverage. It does not establish memory leak freedom, multi-user capacity,
scanned-PDF/OCR quality, chart-heavy document performance, or accuracy for every spreadsheet layout.
The messy fixtures' successful ingestion alone does not prove every cell was interpreted correctly.
The companion regression in `services/ingest/tests/test_workload_fixtures.py` checks exact typed values,
leading-zero IDs, Unicode, multiline cells, dates, null counts and formula warnings on the same layouts,
including a repeated header.

Reindex generations consume disk until document/account deletion. The test's cleanup removes its
generations; a production library that repeatedly reindexes needs disk monitoring. Earlier published
versions are deliberately retained to keep in-flight questions consistent.
