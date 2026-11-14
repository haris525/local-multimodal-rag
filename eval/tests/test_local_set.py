"""The committed benchmark set (eval/benchmarks/local_samples.jsonl): it loads, every file it selects exists, every
reference answer passes its own checks, and the wrong answers seen in live runs fail them."""

import re
from collections import Counter
from pathlib import Path

import pytest

from rag_eval.bench import TAGS, load_items, score

ROOT = Path(__file__).resolve().parents[2]
SET = ROOT / "eval" / "benchmarks" / "local_samples.jsonl"
ITEMS = {it.id: it for it in load_items(SET)}


def check(item_id: str, answer: str) -> tuple[bool, list[str]]:
    it = ITEMS[item_id]
    s, why = score(it, answer, [{"n": 1, "doc_id": it.expected_docs[0]}], it.expected_docs)
    return bool(s["correct"]), why


def test_the_set_is_ascii_and_covers_every_readme_question_and_the_hard_cases():
    SET.read_text(encoding="ascii")
    readme = (ROOT / "examples" / "documents" / "README.md").read_text(encoding="utf-8")
    asked = re.findall(r"^- ((?:(?!\n- |\n\n).)+?[?.])\s+\*\(", readme, re.M | re.S)  # one bullet, maybe wrapped
    asked += re.findall(r"question:\*\* \*(.+?\?)\*", readme, re.S)
    asked += re.findall(r"For a second one, try: \*(.+?\?)\*", readme, re.S)
    questions = {" ".join(it.question.split()) for it in ITEMS.values()}
    missing = [q for q in (" ".join(a.split()).replace("\u03b1", "alpha") for a in asked) if q not in questions]
    assert len(asked) == 65 and missing == []
    tags = Counter(t for it in ITEMS.values() for t in it.tags)
    assert set(tags) <= set(TAGS) and tags["hard-case"] == 5 and tags["cross-document"] == 9


def test_every_selected_file_exists():
    samples = {p.name for p in (ROOT / "examples" / "documents").rglob("*") if p.is_file()}
    smoke = set(re.findall(r'"(smoke-[\w.-]+)"', (ROOT / "scripts" / "make_samples.py").read_text(encoding="utf-8")))
    assert {"smoke-aurora_report.pdf", "smoke-orders.xlsx"} <= smoke
    for it in ITEMS.values():
        assert set(it.docs) <= samples | smoke, it.id


@pytest.mark.parametrize("item_id", sorted(ITEMS))
def test_each_reference_answer_passes_its_own_checks(item_id):
    ok, why = check(item_id, ITEMS[item_id].reference + " [1]")
    assert ok, why


def test_known_wrong_answers_fail():
    wrong = {
        "img-cross-nasa-noaa": [
            "33 in the 1980s, 57, 67 and 131 in the 2010s; the 2020s: 11.5 per decade.",
            "33, 57, 67, 131 and 115 in the 2020s, which likely represents a full decade's worth of data.",
            "Per decade: 33, 57, 67, 131 and 115.",  # counts right, the partial decade not flagged
        ],
        "img-nasa-1": ["2023, at about +1.35 C.", "2024, at about +1.35 C."],
        "xlsx-pop-2": ["Idaho, +1.44%.", "South Carolina grew 1.4%."],
        "smoke-q3-report": ["South, with 2735.62."],
        "smoke-q3-conflict": ["South leads with 340 thousand dollars in the report."],
        "rag-failure-3": ["The study used 2,000 documents and 1,000 questions."],
        "text-const-1": ["Twenty five and thirty five years."],
    }
    for item_id, answers in wrong.items():
        for answer in answers:
            ok, why = check(item_id, answer)
            assert not ok, (item_id, answer, why)
    assert check("img-cross-nasa-noaa", "33, 57, 67, 131, and 115 for 2020-2024 so far.")[0]
    assert check("img-nasa-1", "2024 is the warmest, about 1.3 degrees above the baseline.")[0]
