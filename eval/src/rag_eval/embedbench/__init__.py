"""Embedding-model retrieval benchmark (rag-eval embed-bench).

Compares text embedders on the same passages and questions, retrieval only (no LLM): dense search alone, dense
plus BM25 fused with RRF, and that fusion reranked, scored with recall@k against gold passages. core.py holds the
pure parts (question pools, BM25, fusion, scoring, summaries) and runs without torch; gpu.py loads the models;
run.py is the command; export_chunks.py runs inside the ingest-worker container to export the app's chunks of
one account. Results and how to rerun: docs/benchmarks/embedding-models-2026-11-15.md.
"""
