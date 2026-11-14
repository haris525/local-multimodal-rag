"""The forms set (eval/benchmarks/forms_samples.jsonl): questions where names are written in more than one form. It
loads, it is ASCII, every file it selects exists, every question has its recompute script, every reference answer
passes its own checks, the typical answer of a name filter that misses the data's form fails, and correct answers in
other words pass."""

from collections import Counter
from pathlib import Path

import pytest
from rag_eval.bench import TAGS, load_items, score
from rag_eval.paths import FORMS_SET, NAMED_SETS

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "eval" / "benchmarks" / "forms_evidence"
ITEMS = {it.id: it for it in load_items(FORMS_SET)}


def check(item_id: str, answer: str) -> tuple[bool, list[str]]:
    it = ITEMS[item_id]
    s, why = score(it, answer, [{"n": 1, "doc_id": it.expected_docs[0]}], it.expected_docs)
    return bool(s["correct"]), why


def test_the_set_is_named_ascii_and_mostly_cross_document():
    assert NAMED_SETS["forms"] == FORMS_SET and FORMS_SET.name == "forms_samples.jsonl"
    FORMS_SET.read_text(encoding="ascii")
    assert b"\r" not in FORMS_SET.read_bytes()
    tags = Counter(t for it in ITEMS.values() for t in it.tags)
    assert len(ITEMS) == 12 and sorted(ITEMS) == sorted(f"forms-{n}" for n in range(1, 13))
    assert tags["name-forms"] == 12 and set(tags) <= set(TAGS)
    assert tags["cross-document"] == 8 and tags["single-document"] == 4 and tags["distractors"] == 3


def test_every_selected_file_exists_and_distractors_are_not_expected():
    samples = {p.name for p in (ROOT / "examples" / "documents").rglob("*") if p.is_file()}
    for it in ITEMS.values():
        assert set(it.docs) <= samples, it.id
        assert ("distractors" in it.tags) == (set(it.expected_docs) < set(it.docs)), it.id
        assert ("cross-document" in it.tags) == (len(it.expected_docs) > 1), it.id


def test_every_question_has_its_evidence_script():
    """A script that recomputes every gold value from the files and prints what the naive filter gives."""
    for item_id in ITEMS:
        path = EVIDENCE / f"{item_id}.py"
        assert path.is_file(), item_id
        text = path.read_text(encoding="ascii")
        assert '"examples" / "documents" / "data"' in text and "naive" in text, item_id
    assert sorted(p.stem for p in EVIDENCE.glob("*.py")) == sorted(ITEMS)


@pytest.mark.parametrize("item_id", sorted(ITEMS))
def test_each_reference_answer_passes_its_own_checks(item_id):
    ok, why = check(item_id, ITEMS[item_id].reference + " [1]")
    assert ok, why


TRAPPED = {  # one per question: the answer a filter on the question's form of the name gives
    "forms-1": "India had more, 61 events against 24 for Peru; India's GDP per capita in 2024 was $2,591.99.",
    "forms-2": "One event, magnitude 5.9 in the State of Yap on 13 October 2025. GDP per capita was $4,251.69 and "
               "internet use 39.5%.",
    "forms-3": "Three M5+ events were located in Dominica (77 and 102 km NNE of Punta Cana and 38 km SE of Boca de "
               "Yuma). Dominica's GDP per capita in 2024 was $10,405.28 and internet use 82.0%.",
    "forms-4": "Together they had 16,911,601 residents on July 1, 2025 and 141,781 deaths from all causes in 2017.",
    "forms-5": "Washington's 2017 heart disease death rate was 138.8 per 100,000, and its population grew by 293,501, "
               "or 3.81%.",
    "forms-6": "The CDC file has no chronic lower respiratory diseases or cerebrovascular diseases category, so this "
               "cannot be answered from it.",
    "forms-7": "Grocery store sales were $1,732,518 million in 2024, about $5,096 per resident.",
    "forms-8": "General merchandise stores sold $918,402 million in 2024, 12.9% of retail sales.",
    "forms-9": "Combined they cost $115,740.8 million and caused 186 deaths; Ida was costlier, about 8.4 times Fran.",
    "forms-10": "Four disasters: the 1999 and 2010 tornadoes and the 2017 and 2019 Arkansas floods, $14,534.8 million "
                "and 83 deaths. Kansas grew by 39,234.",
    "forms-11": "12 events name China. China first exceeded $10,000 in 2000, at $25,756.77, with internet use at "
                "27.8%.",
    "forms-12": "Palau ($18,169.08) and the Marshall Islands ($7,590.28); Nauru is not in the file.",
}


def test_every_question_has_a_trapped_answer():
    assert sorted(TRAPPED) == sorted(ITEMS)


@pytest.mark.parametrize("item_id", sorted(TRAPPED))
def test_the_trapped_answer_fails(item_id):
    ok, why = check(item_id, TRAPPED[item_id] + " [1]")
    assert not ok, why


def test_other_trapped_answers_fail():
    wrong = {
        "forms-2": "7 events near Fais and Colonia, largest M5.4; GDP per capita $4,251.69, internet 39.5%.",
        "forms-3": "The file lists 3 events in Dominica, all near Punta Cana; GDP per capita $10,640.47, internet "
                   "use 86.5%.",  # both World Bank rows averaged
        "forms-6": "Influenza and pneumonia never exceeded stroke: 0 years. In 2017 there were 55,672 influenza and "
                   "pneumonia deaths and 146,383 stroke deaths.",
        "forms-8": "Department stores sold $93,016 million in 2024, 1.3% of retail sales.",  # 4522 and 45221 summed
        "forms-9": "Fran, Frances, Tropical Storm Frances, Francine, Ida and Idalia: $117,091.8 million and 188 "
                   "deaths; Ida about 8.4 times Fran.",
        "forms-10": "The 1999 and 2010 tornadoes: $8,657.3 million and 58 deaths. Kansas and Arkansas together grew by "
                    "142,495.",
        "forms-11": "11 events name China; China first exceeded $10,000 in 2000 (Hong Kong SAR, China, $25,756.77).",
    }
    for item_id, answer in wrong.items():
        ok, why = check(item_id, answer + " [1]")
        assert not ok, (item_id, answer, why)


def test_correct_answers_in_other_words_pass():
    right = {
        "forms-1": "Peru, with 24 M5+ events to India's 22 (20 plus two in the Nicobar Islands). Peru's 2024 GDP per "
                   "capita: 8,526.3 US dollars.",
        "forms-2": "Eight events (seven written 'Micronesia', one 'Federated States of Micronesia'), the strongest "
                   "M5.9. GDP per capita $4,252 and internet use 39.52% in 2024.",
        "forms-3": "There were zero M5+ events in Dominica: the three matches are off the Dominican Republic. "
                   "Dominica: GDP per capita $10,405 and internet use 82% in 2024.",
        "forms-4": "About 15.15 million people (Virginia plus Maryland) and 118,505 deaths in 2017.",
        "forms-5": "The District of Columbia: 189.8 heart disease deaths per 100,000 in 2017; population up 0.6% "
                   "(689,544 to 693,645).",
        "forms-6": "Ten years, 2008 through 2017, starting in 2008. 2017: CLRD 160,201 deaths, stroke 146,383.",
        "forms-7": "$885.7 billion at grocery stores in 2024, roughly $2,605 for each of the 340.0 million residents.",
        "forms-8": "GAFO sales were $1.53 trillion in 2024, about 21.57% of retail sales excluding food services.",
        "forms-9": "Ida (2021): $84.6 billion, 96 dead; Fran (1996): $10.0 billion, 37 dead. Together about $94.6 "
                   "billion and 133 deaths, Ida roughly 8.45x costlier.",
        "forms-10": "Only the May 1999 and May 2010 tornado outbreaks name Kansas: $8.66 billion and 58 deaths in all. "
                    "Kansas added 39,234 residents.",
        "forms-11": "Eleven events; China passed $10,000 per person in 2018 ($10,085.66), with 65.36% of people "
                    "online.",
        "forms-12": "Palau is first at 18,169.08 USD, then Nauru (listed as Naoero) at 14,048.16 USD.",
    }
    for item_id, answer in right.items():
        ok, why = check(item_id, answer + " [1]")
        assert ok, (item_id, answer, why)
