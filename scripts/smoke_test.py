"""End-to-end check through the gateway: ingest every sample type, then ask one question per type
with all samples selected, and require the expected fact plus a citation of the right document.
The uploaded samples are deleted afterwards (also on failure), so the library is left as it was.

    RAG_TOKEN=rag_pat_... uv run --with httpx scripts/smoke_test.py [--samples samples] [--url http://localhost:8000]
        [--deep] [--keep] [--token TOKEN]

The token is an API token (user menu > API tokens); the documents are uploaded as its user.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx

# The chat stream reader shared with eval and the protocol rag follows (libs/common, standard library only).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libs" / "common" / "src"))
from rag_common.chat_events import read_chat  # noqa: E402

CASES = [  # (document, question, regex the answer must contain, expected planner route or None)
    ("smoke-aurora_report.pdf", "What is the Aurora project budget for 2026?", r"4\.2", None),
    ("smoke-aurora_report.pdf", "Which region had the highest third-quarter revenue, and how much?", r"South.*340|340.*South", None),
    ("smoke-aurora_report.pdf", "What was customer satisfaction in 2025 according to the survey chart?", r"85", None),
    ("smoke-aurora_report.pdf", "Within how many days must warranty claims be filed?", r"45", None),
    ("smoke-handbook.txt", "How many unused vacation days can employees carry over?", r"\b5\b|five", None),
    ("smoke-runbook.md", "Who must approve a rollback and how long does it take?", r"on-call.*12|12.*on-call", None),
    ("smoke-orders.xlsx", "What is the total sale amount for orders in the North region?", r"35,?802", "sql"),
    ("smoke-orders.xlsx", "Which North region orders mention battery overheating?", r"ORD-0008.*ORD-0044|ORD-0044.*ORD-0008", "sql"),
    ("smoke-orders.xlsx", "Which orders from the North region had shipping problems?", r"late delivery|packaging", "filter"),
    ("smoke-inventory.csv", "How many units of INV-2025-0042 are in stock?", r"137", None),
    ("smoke-server_room_sign.png", "What is the temperature limit in the server room?", r"27", None),
]


def upload(client: httpx.Client, folder: Path) -> tuple[dict[str, str], list[str]]:
    """Upload the samples; returns ({doc_id: name} of the uploaded versions, upload errors)."""
    files = [("files", (p.name, p.read_bytes())) for p in sorted(folder.iterdir()) if p.is_file()]
    ids, errors = {}, []
    for result in client.post("/api/documents", files=files).raise_for_status().json():
        if result["error"]:
            errors.append(f"{result['name']}: {result['error']}")
        else:
            ids[result["document"]["doc_id"]] = result["name"]
    return ids, errors


def wait_ready(client: httpx.Client, ids: dict[str, str], budget_s: int) -> dict[str, dict]:
    """Wait for exactly these versions (by doc_id) to finish indexing."""
    deadline = time.time() + budget_s
    while time.time() < deadline:
        docs = {d["doc_id"]: d for d in client.get("/api/documents").raise_for_status().json() if d["doc_id"] in ids}
        if len(docs) == len(ids) and all(d["status"] in ("ready", "failed") for d in docs.values()):
            failed = [d["name"] for d in docs.values() if d["status"] == "failed"]
            if failed:
                sys.exit(f"ingestion failed: {failed}")
            return {d["name"]: d for d in docs.values()}
        time.sleep(3)
    sys.exit(f"ingestion did not finish within {budget_s} s")


def ask(client: httpx.Client, question: str, doc_ids: list[str], mode: str) -> dict:
    """One answer; error is set for an error event, a stream cut before done, or events out of order."""
    out = {"answer": "", "sources": [], "citations": [], "route": "", "error": None, "timings": {}}
    body = {"question": question, "doc_ids": doc_ids, "mode": mode}
    with client.stream("POST", "/api/chat", json=body, timeout=600) as response:
        if response.status_code >= 400:
            return {**out, "error": response.read().decode()}
        reply = read_chat(response.iter_lines())
    return {"answer": reply.answer, "sources": reply.sources, "citations": reply.citations, "route": reply.route or "",
            "error": reply.failure, "timings": reply.timings}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--deep", action="store_true", help="use deep search instead of single-shot")
    parser.add_argument("--ingest-budget", type=int, default=900)
    parser.add_argument("--keep", action="store_true", help="leave the uploaded samples in the library")
    parser.add_argument("--token", default=os.getenv("RAG_TOKEN"), help="API token (default: $RAG_TOKEN)")
    args = parser.parse_args()
    if not args.token:
        sys.exit("set RAG_TOKEN or pass --token: an API token from the user menu (API tokens)")
    client = httpx.Client(base_url=args.url, timeout=120, headers={"Authorization": f"Bearer {args.token}"})
    ids, errors = upload(client, Path(args.samples))
    try:
        if errors:
            sys.exit(f"upload failed: {errors}")
        docs = wait_ready(client, ids, args.ingest_budget)
        doc_ids = [d["doc_id"] for d in docs.values() if d["status"] == "ready"][:10]
        failures = 0
        for doc_name, question, pattern, route in CASES:
            started = time.time()
            r = ask(client, question, doc_ids, "deep" if args.deep else "single")
            cited_docs = {r["sources"][n - 1]["doc_name"] for n in r["citations"] if 0 < n <= len(r["sources"])}
            problems = []
            if r["error"]:
                problems.append(f"error: {r['error']}")
            if not re.search(pattern, r["answer"], re.I | re.S):
                problems.append(f"answer lacks /{pattern}/")
            if doc_name not in cited_docs:
                problems.append(f"no citation of {doc_name} (cited {sorted(cited_docs) or 'nothing'})")
            if route and not args.deep and r["route"] != route:
                problems.append(f"route {r['route']!r}, expected {route!r}")
            failures += bool(problems)
            status = "FAIL" if problems else "ok"
            print(f"[{status}] {question}\n       route={r['route']} {time.time() - started:.1f}s  "
                  f"answer: {r['answer'].strip()[:220].replace(chr(10), ' ')}")
            for p in problems:
                print(f"       - {p}")
        print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
        sys.exit(1 if failures else 0)
    finally:
        if not args.keep:
            for doc_id, name in ids.items():
                try:
                    client.delete(f"/api/documents/{doc_id}").raise_for_status()
                except httpx.HTTPError as exc:
                    print(f"warning: could not delete sample {name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
