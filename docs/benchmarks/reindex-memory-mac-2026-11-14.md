# Reindex and memory validation — 2026-11-14

**Passed:** a 14.7-minute synthetic workload on Apple Silicon with 64 GiB host memory and a 16 GiB
Docker VM. Ingest-worker retained its 4 GiB cap; inference ran natively using the models/settings in
[the Mac guide](../MACOS.md). This was a shared development machine, not a dedicated benchmark host.
Raw measurements: [JSON report](reindex-memory-mac-2026-11-14.json).

| Check | Observed result |
|---|---|
| 250,000-row CSV | Exact SQL count and sums passed; 20,002 retrieval chunks; 159.6 s initial indexing |
| 50,000-row UTF-16 CSV | Semicolon/Unicode/multiline/decimal-comma ingestion passed; 147.5 s |
| 12,000-row workbook | Repeated headers, formatting, blanks and uncached formulas handled; 117.5 s; normalized table includes its total row |
| 100-page PDF | Final-page marker retrieved; 99.4 s initially, 84.4 s on repeat |
| Approximately 1 MiB text | End-of-file marker retrieved; 39.2 s |
| Truncated PDF | Expected failed document in 3.0 s; worker stayed healthy |
| Search during CSV reindex | 68 searches: 67 on the previous generation, then one on the new generation; no mixed generations |
| Old/new SQL and pandas | Both generations remained readable; exact aggregates passed through signed SQL and real isolated Python executions |
| Repeated reads | 30 cycles / 300 requests; search-plus-page median 57 ms, maximum 113 ms |
| Cleanup | Temporary account and all its documents removed |

Sampled peak Docker working sets:

| Service | Peak MiB | Configured cap |
|---|---:|---:|
| ingest-worker | 3,593.2 | 4 GiB |
| ingest-api | 141.3 | 1 GiB |
| rag | 330.4 | 2 GiB |
| qdrant | 536.7 | 2 GiB |
| sql | 113.8 | 1 GiB |
| sandbox | 170.1 | 2 GiB |

All services finished healthy, with unchanged restart counts and no new OOM events. The worker's child
OOM counter stayed at zero. Sandbox entered the run with one historical child OOM and its corresponding
Docker flag already set; its counter stayed at one. There were 127 samples and no monitoring errors.
Worker working set stayed at about 885.6 MiB through repeated reads and the final cooldown. Its baseline
was about 222.4 MiB; retained allocator/runtime memory therefore remained, rather than returning to the
startup baseline. The steady interval is short and does not establish long-term leak freedom.

Before the memory changes, a repeated large-file sequence hit 4,096 MiB and killed the PDF parser.
The fix streams embedding/upsert batches, caps PDF stage queues, and recycles parsers with large idle
caches. The successful run repeated the PDF after processing/reindexing large spreadsheets. PDF peaks
still approach the 4 GiB cap: larger/scanned/image-heavy files need separate validation and potentially
different resource budgets.

The same messy layouts also passed exact-value parser regressions for leading-zero IDs, decimal commas,
Unicode/multiline cells, dates, null counts and formula warnings. A separate live old/new CSV snapshot
check selected all four retrievers and preserved each generation; the image arm had no stored image
vectors for that CSV. Unit tests additionally delay stale file/Qdrant writers and overlap deletion with
re-upload, scenarios this live workload does not forcibly inject.

The report's native inference RSS used recorded process groups. Foreground listener discovery in the
current runner was checked separately. RSS omits some Metal allocations, and sampled Docker working
sets can miss short peaks. Native model allocations sit outside Docker's service caps. Disk retention,
OCR/chart quality, multi-user capacity and arbitrary user documents are not certified by this run.

Reproduce using [Workload validation](../WORKLOAD_VALIDATION.md), with `--macos --python`. Reindex
generations remain until document deletion; monitor disk usage in a persistent library.
