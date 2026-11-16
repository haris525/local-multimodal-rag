"""Candidate gold chunks for the easy tier, for a person to check (the labels file is the checked result).

    python -m rag_eval.embedbench.label <chunks.jsonl> [--top 5] > candidates.txt

For each text question of the local set, the chunks of its selected documents are ranked by how many of the
question's facts and numbers they contain; each candidate is printed with its text_sha and an excerpt around the
first match. A chunk is gold if it states the answer or a fact the answer needs; chunks stating the same fact go
in one group (alternatives). Write the checked labels to eval/benchmarks/local_passages.jsonl as
{"id", "doc", "text_sha", "group", "quote"}.
"""

import argparse
import json
import re
from pathlib import Path

from rag_eval.embedbench.core import text_question
from rag_eval.paths import LOCAL_SET


def _number_forms(value: float, label: str = "") -> list[str]:
    forms = {f"{value:g}"}
    if float(value).is_integer():
        n = int(value)
        forms |= {str(n), f"{n:,}"}
        if n >= 1000 and n % 1000 == 0:
            forms.add(f"{n // 1000}K")
        if n >= 100_000:
            forms.add(f"{n / 1e6:g}M")
    forms |= set(re.findall(r"\d[\d,.]*\s?[KMkm]?\b", label))
    return sorted(forms, key=len, reverse=True)


def patterns(sample: dict) -> list[tuple[str, re.Pattern]]:
    """(label, pattern) for each fact and number of a question."""
    out = []
    for fact in sample.get("facts", []):
        if isinstance(fact, str):
            out.append((fact, re.compile(re.escape(fact), re.IGNORECASE)))
        elif "regex" in fact:
            out.append((fact.get("label", fact["regex"]), re.compile(fact["regex"], re.IGNORECASE)))
        elif "any" in fact:
            out.append((" / ".join(fact["any"]), re.compile("|".join(rf"\b{re.escape(a)}\b" for a in fact["any"]), re.IGNORECASE)))
    for num in sample.get("numbers", []):
        values = [n["value"] for n in num["any"]] if "any" in num else [num["value"]]
        forms = [f for v in values for f in _number_forms(v, num.get("label", ""))]
        out.append((num.get("label", str(values)), re.compile("|".join(rf"(?<![\d.]){re.escape(f)}(?![\d])" for f in forms), re.IGNORECASE)))
    return out


def candidates(sample: dict, chunks: list[dict], top: int = 5) -> list[tuple[int, dict, list[str], int]]:
    pats = patterns(sample)
    ranked = []
    for c in chunks:
        if c["doc_name"] not in sample["docs"]:
            continue
        found = [(label, m.start()) for label, p in pats if (m := p.search(c["search_text"]))]
        if found:
            ranked.append((len(found), c, [label for label, _ in found], min(pos for _, pos in found)))
    ranked.sort(key=lambda r: (-r[0], r[1]["doc_name"], r[1]["chunk_index"]))
    per_doc: dict[str, int] = {}
    out = []
    for r in ranked:  # at least the best candidates of every selected document
        doc = r[1]["doc_name"]
        if per_doc.get(doc, 0) < top:
            per_doc[doc] = per_doc.get(doc, 0) + 1
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m rag_eval.embedbench.label")
    p.add_argument("chunks", type=Path)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args(argv)
    chunks = [json.loads(line) for line in args.chunks.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in LOCAL_SET.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        if not text_question(s):
            continue
        print(f"=== {s['id']}  {s['question']}\n    reference: {s['reference']}")
        for n, c, labels, pos in candidates(s, chunks, args.top):
            text = c["search_text"].replace("\n", " ")
            start = max(0, pos - 120)
            print(f"  [{n}] {c['doc_name']}#{c['chunk_index']} {c['kind']} sha={c['text_sha'][:16]} matched={labels}\n"
                  f"      ...{text[start:start + 420]}...")
        print()


if __name__ == "__main__":
    main()
