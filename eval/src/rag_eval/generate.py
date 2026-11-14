"""Bootstrap a golden set from the indexed chunks with the local LLM. Review it before trusting it."""

import json
import random
import re
import sys
from pathlib import Path

from rag_eval.client import Stack

KINDS = ("text", "table", "row", "image", "sheet_summary")
MIN_TOKENS = 15
MAX_DOCS_PER_QUERY = 10

_QUESTION = """You are writing evaluation questions for a document search system.
Passage from "{doc}" ({kind}{where}):
\"\"\"
{text}
\"\"\"
Write ONE question that a real user could ask and that this passage alone answers, and the answer.
Paraphrase: do not copy any run of more than three consecutive words from the passage, except names,
codes and numbers. The question must make sense without seeing the passage.
Return JSON: {{"question": "...", "answer": "..."}}"""

_UNANSWERABLE = """You are writing evaluation questions for a document search system.
The user's library contains these documents: {docs}.
Here is one passage from them:
\"\"\"
{text}
\"\"\"
Write ONE plausible question on the same topic whose answer is NOT in this passage and is unlikely to
appear anywhere in such documents (for example a detail, date or figure the passage does not give).
Return JSON: {{"question": "..."}}"""


def _copies_passage(question: str, text: str, run: int = 5) -> bool:
    words = re.findall(r"\w+", question.lower())
    body = " ".join(re.findall(r"\w+", text.lower()))
    return any(" ".join(words[i : i + run]) in body for i in range(len(words) - run + 1))


def _label(chunk: dict) -> dict:
    keys = ("doc_id", "page", "page_end", "sheet", "row_start", "row_end", "asset")
    return {"chunk_id": chunk["id"], "text_sha": chunk["text_sha"], **{k: chunk.get(k) for k in keys}}


def _where(chunk: dict) -> str:
    if chunk.get("sheet"):
        return f', sheet "{chunk["sheet"]}"'
    return f", page {chunk['page']}" if chunk.get("page") else ""


def generate(stack: Stack, out: Path, n: int, unanswerable: float, doc_ids: list[str] | None, seed: int) -> int:
    rng = random.Random(seed)
    docs = [d for d in stack.documents() if d["status"] == "ready" and (not doc_ids or d["doc_id"] in doc_ids)]
    if not docs:
        raise SystemExit("no ready documents to sample from")
    names = {d["doc_id"]: d["name"] for d in docs}
    by_kind: dict[str, list[dict]] = {k: [] for k in KINDS}
    for doc in docs:
        for chunk in stack.chunks(doc["doc_id"]):
            if chunk["n_tokens"] >= MIN_TOKENS and chunk["kind"] in by_kind:
                by_kind[chunk["kind"]].append(chunk)
    present = [k for k in KINDS if by_kind[k]]
    if not present:
        raise SystemExit(f"no chunks of at least {MIN_TOKENS} tokens to write questions from")
    n_unans = round(n * unanswerable)
    failures: list[str] = []
    quota = {k: (n - n_unans) // len(present) + (1 if i < (n - n_unans) % len(present) else 0) for i, k in enumerate(present)}

    def scope(doc_id: str) -> list[str]:
        others = [d for d in names if d != doc_id]
        return [doc_id, *rng.sample(others, min(len(others), MAX_DOCS_PER_QUERY - 1))]

    items: list[dict] = []
    for kind in present:
        pool = by_kind[kind][:]
        rng.shuffle(pool)
        made = 0
        for chunk in pool:
            if made >= quota[kind]:
                break
            prompt = _QUESTION.format(doc=names[chunk["doc_id"]], kind=kind, where=_where(chunk), text=chunk["text"][:4000])
            try:
                qa = stack.llm_json(prompt, temperature=0.7)
            except Exception as exc:  # one bad generation is skipped, and reported below
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            question, answer = str(qa.get("question", "")).strip(), str(qa.get("answer", "")).strip()
            if not question or not answer or _copies_passage(question, chunk["text"]):
                continue
            items.append({"id": f"q{len(items) + 1:03d}", "question": question, "reference_answer": answer,
                          "doc_ids": scope(chunk["doc_id"]), "answerable": True, "kind": kind,
                          "relevant": [_label(chunk)]})
            made += 1

    every = [c for k in present for c in by_kind[k]]
    for _ in range(n_unans):
        chunk = rng.choice(every)
        try:
            q = stack.llm_json(_UNANSWERABLE.format(docs=", ".join(names.values()), text=chunk["text"][:3000]), temperature=0.7)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            continue
        if q.get("question"):
            items.append({"id": f"q{len(items) + 1:03d}", "question": q["question"].strip(), "reference_answer": "",
                          "doc_ids": scope(chunk["doc_id"]), "answerable": False, "kind": "unanswerable", "relevant": []})

    if failures:
        print(f"{len(failures)} generation calls failed and were skipped (last: {failures[-1][:200]})", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)
