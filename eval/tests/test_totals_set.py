"""The totals set (eval/benchmarks/totals_samples.jsonl): questions where a total or subtotal row changes the answer,
if it is counted with the rows it sums or if it is left out where the answer needs it. It loads, it is ASCII, every
file it selects exists, every question has its recompute script, every reference answer passes its own checks, the
typical answer of the trap fails, and correct answers in other words pass."""

from collections import Counter
from pathlib import Path

import pytest
from rag_eval.bench import TAGS, load_items, score
from rag_eval.paths import NAMED_SETS, TOTALS_SET

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "eval" / "benchmarks" / "totals_evidence"
ITEMS = {it.id: it for it in load_items(TOTALS_SET)}


def check(item_id: str, answer: str) -> tuple[bool, list[str]]:
    it = ITEMS[item_id]
    s, why = score(it, answer, [{"n": 1, "doc_id": it.expected_docs[0]}], it.expected_docs)
    return bool(s["correct"]), why


def test_the_set_is_named_ascii_and_mixes_single_and_cross_document():
    assert NAMED_SETS["totals"] == TOTALS_SET and TOTALS_SET.name == "totals_samples.jsonl"
    TOTALS_SET.read_text(encoding="ascii")
    assert b"\r" not in TOTALS_SET.read_bytes()
    tags = Counter(t for it in ITEMS.values() for t in it.tags)
    assert len(ITEMS) == 12 and sorted(ITEMS) == sorted(f"totals-{n}" for n in range(1, 13))
    assert tags["totals"] == 12 and set(tags) <= set(TAGS)
    assert tags["cross-document"] == 4 and tags["single-document"] == 8 and tags["distractors"] == 3


def test_every_selected_file_exists_and_distractors_are_not_expected():
    samples = {p.name for p in (ROOT / "examples" / "documents").rglob("*") if p.is_file()}
    for it in ITEMS.values():
        assert set(it.docs) <= samples, it.id
        assert ("distractors" in it.tags) == (set(it.expected_docs) < set(it.docs)), it.id
        assert ("cross-document" in it.tags) == (len(it.expected_docs) > 1), it.id


def test_every_question_has_its_evidence_script():
    """A script that recomputes every gold value from the files and prints what the naive computation gives."""
    for item_id in ITEMS:
        path = EVIDENCE / f"{item_id}.py"
        assert path.is_file(), item_id
        text = path.read_text(encoding="ascii")
        assert b"\r" not in path.read_bytes(), item_id
        assert '"examples" / "documents" / "data"' in text and "naive" in text, item_id
    assert sorted(p.stem for p in EVIDENCE.glob("*.py")) == sorted(ITEMS)


@pytest.mark.parametrize("item_id", sorted(ITEMS))
def test_each_reference_answer_passes_its_own_checks(item_id):
    ok, why = check(item_id, ITEMS[item_id].reference + " [1]")
    assert ok, why


TRAPPED = {  # one per question: the answer the trap gives (a total counted as a row, or a needed total left out)
    "totals-1": "All causes had the most deaths in Florida in 2017, 203,636, which is 157,196 more than heart "
                "disease (46,440).",
    "totals-2": "The United States (121,404), California (16,238), Texas (9,545), Florida (6,980) and Ohio (5,117): "
                "159,284 deaths combined.",
    "totals-3": "The ten causes account for all deaths in the file: 2,081,531 deaths, 100% of the total, so no deaths "
                "were from other causes.",
    "totals-4": "15 states had more than 10 million residents on July 1, 2025, with 868,407,472 residents, 254.1% of "
                "the U.S. population.",
    "totals-5": "The United States gained the most, 10,268,744 residents; its 2017 age-adjusted death rate from all "
                "causes was 731.9 per 100,000.",
    "totals-6": "Retail and food services sales, total was the highest at $8,350,933 million in 2025, 100% of retail "
                "and food services sales.",
    "totals-7": "Summing the kinds of business gives $21,104,922 million in 2025, about $61,749 per resident.",
    "totals-8": "Adding the matching rows (442, 443 and 442,443) gives $426,630 million in 2025, up 1.9% from "
                "$418,618 million in 2024.",
    "totals-9": "The United States (341,784,857), the South (133,833,983) and the West (80,146,154); the CDC file has "
                "only the United States, with 2,813,503 deaths from all causes in 2017.",
    "totals-10": "Total generation (including sources not shown) increased the most, by 120,867.3 million kWh "
                 "(2.8%).",
    "totals-11": "The 2017 disasters caused 3,280 deaths; summed over the states, unintentional injuries killed "
                 "339,872 people in 2017, about 103.6 times as many.",
    "totals-12": "The table has no region column, so the regions cannot be compared from the states; the state that "
                 "gained the most was Texas, up 2,560,323.",
}


def test_every_question_has_a_trapped_answer():
    assert sorted(TRAPPED) == sorted(ITEMS)


@pytest.mark.parametrize("item_id", sorted(TRAPPED))
def test_the_trapped_answer_fails(item_id):
    ok, why = check(item_id, TRAPPED[item_id] + " [1]")
    assert not ok, why


def test_other_trapped_answers_fail():
    wrong = [
        ("totals-2", ("California, Texas, Florida and Ohio had the most, 37,880 deaths combined (the United States "
                      "row is first with 121,404).")),
        ("totals-3", ("The ten causes plus the all-causes row come to 4,895,034 deaths, 174.0% of the 2,813,503 "
                      "deaths, so there were no deaths from other causes.")),
        ("totals-3", ("Summing every row for 2017 gives 4,163,062 deaths from the ten causes out of 5,627,006, 74.0%, "
                      "so 1,463,944 deaths were from other causes.")),  # the United States row summed too
        ("totals-4", ("10 states had more than 10 million residents: 184,837,758 people, 26.9% of the 686,754,549 "
                      "residents in the table.")),  # the regions summed with the states as the denominator
        ("totals-5", "The South gained the most, 7,552,378 residents; the CDC file has no row for it."),
        ("totals-6", ("Retail sales and food services excl gasoline stations, $7,717,274 million, 92.4% of the "
                      "total.")),  # only rows with "total" in the label dropped
        ("totals-6", ("Motor vehicle and parts dealers, $1,624,608 million, 7.7% of the $21,104,922 million summed "
                      "over every kind of business.")),
        # the total row named as the answer, with the right one only as a fallback
        ("totals-6", ("Highest: Retail and food services sales, total, $8,350,933 million, 100% of the total. If "
                      "aggregates are left out, motor vehicle and parts dealers: $1,624,608 million, 19.45%.")),
        ("totals-6", ("Retail and food services sales, total: 8,350,933 (100.00%, it is the total itself); the "
                      "largest single kind is motor vehicle and parts dealers, 1,624,608, 19.45%.")),
        ("totals-7", "$43,215,429 million in 2025, about $126,440 per resident."),
        ("totals-7", ("$8,350,933 million in 2025, about $12,160 per resident (686,754,549 residents over the rows of "
                      "the population table).")),
        ("totals-8", ("The furniture and electronics rows add up to $560,635 million in 2025, up 1.7% from $551,317 "
                      "million in 2024.")),
        ("totals-10", "Solar rose the most, by 75,837.5 million kWh (34.5%)."),
        ("totals-12", ("The United States gained 10,268,744 and the South 7,552,378, so the United States gained the "
                       "most: 100% of the national gain.")),
    ]
    for item_id, answer in wrong:
        ok, why = check(item_id, answer + " [1]")
        assert not ok, (item_id, answer, why)


def test_correct_answers_in_other_words_pass():
    right = [
        ("totals-1", ("Heart disease was the leading cause in Florida in 2017 with **46,440** deaths, ahead of cancer "
                      "at 45,131, a gap of 1,309.")),
        ("totals-1", "Heart disease: 46440 deaths. Margin over cancer: 1309."),
        ("totals-2", "California 16,238; Texas 9,545; Florida 6,980; Ohio 5,117; Michigan 4,428. Total: 42,308."),
        ("totals-2", ("The top five were California, Texas, Florida, Ohio and Michigan, with 42308 Alzheimer's deaths "
                      "in total (the national row of 121,404 is not a state).")),
        ("totals-3", "About 74% (2,081,531 out of 2,813,503 deaths); 731,972 deaths came from causes outside the ten."),
        ("totals-3", ("Together the ten causes made up **73.98%** of 2017 deaths, leaving about 732,000 from other "
                      "causes.")),
        ("totals-4", ("Ten states (CA, TX, FL, NY, PA, IL, OH, GA, NC, MI) topped 10 million; together they held "
                      "about 54.08% of the 341.8 million U.S. residents.")),
        ("totals-4", "Count: **10**. Their share of the U.S. population: 54.1% (184.8 million of 341.8 million)."),
        ("totals-4", "There were 10 such states in total, home to 54.1% of Americans."),
        ("totals-5", ("Texas: +2.56 million people (29.1M to 31.7M). Texas's 2017 all-cause age-adjusted death rate: "
                      "735.7 per 100k.")),
        ("totals-5", ("Texas added 2,560,323 residents, about a quarter of the national gain of 10,268,744; in 2017 "
                      "its age-adjusted all-cause death rate was 735.7, above the U.S. rate of 731.9.")),
        ("totals-6", "Motor vehicle and parts dealers: $1,624,608 million (19.45% of retail and food services sales)."),
        ("totals-6", "Auto dealers (NAICS 441) at 1,624.6 billion dollars, roughly 19.5% of the 8.35 trillion total."),
        ("totals-6", ("Motor vehicle and parts dealers led with about $1.62 trillion, 19.5% of all retail and food "
                      "services sales.")),
        ("totals-6", ("Motor vehicle and parts dealers (441): $1,624,608 million, 19.45% of the $8,350,933 million "
                      "total row, which is not a kind of business and is left out of the ranking.")),
        ("totals-7", "Total: $8.35 trillion. Per person: roughly $24,430."),
        ("totals-7", "Sales were 8,350,933 million dollars, or $24,433.30 for each of the 341,784,857 residents."),
        ("totals-8", ("Together $213.3 billion in 2025 ($127.3 billion + $86.0 billion), a 1.91% increase over 2024's "
                      "$209.3 billion.")),
        ("totals-8", "Combined: 213,315 (millions of dollars), up 1.9% from 209,309."),
        ("totals-9", ("The three largest states are California, Texas and Florida; in 2017 they recorded 268,189 + "
                      "198,106 + 203,636 = 669,931 deaths from all causes.")),
        ("totals-9", "California, Texas and Florida: **669,931** deaths in 2017 combined."),
        ("totals-10", "Coal, up roughly 85.0 billion kWh (about 13%), from 652.2 to 737.2 billion kWh."),
        ("totals-10", ("Coal generation grew by 84,995 million kilowatthours, or 13.03%; total generation rose "
                       "120,867 million kWh.")),
        ("totals-11", ("3,280 deaths from the 19 disasters; 169,936 unintentional-injury deaths; roughly 52 times as "
                       "many.")),
        ("totals-11", ("Disasters: 3280 deaths. Unintentional injuries: 169936 (the United States row). Ratio: about "
                       "51.8.")),
        ("totals-12", "**South**: +7.55 million residents, about 73.55% of the U.S. increase of 10.27 million."),
        ("totals-12", "The South gained 7,552,378 people, 73.5% of the 10,268,744 national gain."),
        # a share rounded to a whole percent is right too: the questions ask for no decimals
        ("totals-4", "10 states, home to about 54% of U.S. residents."),
        ("totals-6", "Motor vehicle and parts dealers: $1,624,608 million, about 19% of retail and food services sales."),
        ("totals-8", "$213,315 million combined, up about 2% from 2024."),
        ("totals-12", "The South: +7,552,378, roughly 74% of the national gain."),
    ]
    for item_id, answer in right:
        ok, why = check(item_id, answer + " [1]")
        assert ok, (item_id, answer, why)
