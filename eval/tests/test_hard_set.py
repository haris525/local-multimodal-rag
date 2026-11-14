"""The hard set (eval/benchmarks/hard_samples.jsonl): it loads, it is ASCII, every file it selects exists, every
question has its evidence, every reference answer passes its own checks, and typical wrong answers fail them."""

import re
from collections import Counter
from pathlib import Path

import pytest
from rag_eval.bench import TAGS, load_items, score
from rag_eval.paths import HARD_SET

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "eval" / "benchmarks" / "hard_evidence"
ITEMS = {it.id: it for it in load_items(HARD_SET)}


def check(item_id: str, answer: str) -> tuple[bool, list[str]]:
    it = ITEMS[item_id]
    s, why = score(it, answer, [{"n": 1, "doc_id": it.expected_docs[0]}], it.expected_docs)
    return bool(s["correct"]), why


def test_the_set_is_ascii_hard_and_mostly_cross_document():
    HARD_SET.read_text(encoding="ascii")
    tags = Counter(t for it in ITEMS.values() for t in it.tags)
    assert len(ITEMS) == 33 and tags["hard"] == 33 and set(tags) <= set(TAGS)
    assert tags["cross-document"] >= 20 and tags["multi-step"] >= 25
    assert tags["distractors"] == 6 and tags["false-premise"] == 4


def test_every_selected_file_exists_and_distractors_are_not_expected():
    samples = {p.name for p in (ROOT / "examples" / "documents").rglob("*") if p.is_file()}
    for it in ITEMS.values():
        assert set(it.docs) <= samples, it.id
        assert ("distractors" in it.tags) == (set(it.expected_docs) < set(it.docs)), it.id


def test_every_question_has_its_evidence():
    """Data questions: a script that recomputes the gold values from the files. Paper questions: the page and the
    exact quote for each fact and number."""
    for item_id in ITEMS:
        kind = "py" if item_id.startswith("hard-data-") else "md"
        path = EVIDENCE / f"{item_id}.{kind}"
        assert path.is_file(), item_id
        text = path.read_text(encoding="ascii")
        cited = r'\.(pdf, page \d+|txt \(plain text, no pages\))\n\s*- "'  # the source, then the quote
        assert "examples" in text if kind == "py" else re.search(cited, text), item_id


@pytest.mark.parametrize("item_id", sorted(ITEMS))
def test_each_reference_answer_passes_its_own_checks(item_id):
    ok, why = check(item_id, ITEMS[item_id].reference + " [1]")
    assert ok, why


def test_typical_wrong_answers_fail():
    wrong = {
        "hard-data-1": ["Russia, with a GDP per capita of $14,961.90; its M8.8 event was on 29 July 2025."],
        "hard-data-3": ["2023, with 27 events; wind and solar made up 14.0% of generation."],  # the tie missed
        "hard-data-6": ["Mississippi, 23.0; its population fell 7,104 (-0.24%)."],
        "hard-data-13": ["Coal fell from 2,157 to 1,918 kWh per resident, a drop of about 239 kWh."],
        "hard-data-14": ["The file lists 1,833 deaths for Katrina; the CDC recorded 63,001 deaths in 2005."],
        "hard-data-16": ["Retail sales grew 36.6%, generation 10.5% and population 2.92%."],  # the April 2020 base
        "hard-paper-8": ["Both use text-davinci-003: 30.8 exact match on HotpotQA and 78.4% on ALFWorld."],
        "hard-paper-14": ["Ragas found high faithfulness and answer relevance for the BioASQ answers."],
        "hard-paper-15": ["Every State has two Senators and at least one Representative, so the minimum is 3 "
                          "electors; Virginia had ten Representatives, so 12."],  # the elector rule not stated
    }
    for item_id, answers in wrong.items():
        for answer in answers:
            ok, why = check(item_id, answer)
            assert not ok, (item_id, answer, why)


def test_correct_answers_in_other_words_pass():
    assert check("hard-data-10", "Apr. 2020 was lowest at $402,157 million, 22.0% below February; June 2020 was "
                                 "the first month above February.")[0]
    assert check("hard-data-12", "Wind first beat coal in April 2023, and solar first beat hydro in June 2024.")[0]
    assert check("hard-data-3", "A tie: 2023 and 2024 had 27 events each; wind and solar were 14.0% and 15.6%.")[0]
    for rule in ("adding the whole number of Senators to which the State is entitled in the Congress to the number "
                 "of Representatives to which it is entitled",
                 "adding the whole number of Senators to the whole number of Representatives",
                 "equal to the sum of its Representatives and its two Senators"):
        answer = f"Electors are found by {rule}. The minimum is 3 electors; Virginia would have had 12."
        assert check("hard-paper-15", answer)[0], answer
