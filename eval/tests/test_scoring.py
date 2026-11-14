import pytest

from rag_eval.scoring import (
    citation_check,
    citations_in,
    contains_answer,
    exact_match,
    facts_match,
    loose,
    normalize_answer,
    numbers_in,
    numbers_match,
    strip_citations,
    token_f1,
)


def test_squad_normalization_drops_case_punctuation_articles_and_spacing():
    assert normalize_answer("The  Eiffel-Tower, in Paris!") == "eiffeltower in paris"
    assert normalize_answer("An apple a day") == "apple day"


def test_citation_markers_are_removed_before_em_and_f1():
    assert strip_citations("South Carolina [1].") == "South Carolina."
    assert strip_citations("Monaco [2, 3] and Bermuda [4; 5]") == "Monaco and Bermuda"
    assert exact_match("South Carolina [1].", ["South Carolina"]) == 1.0
    assert citations_in("a [2] b [1, 2] c [3;4]") == [2, 1, 3, 4]
    assert citations_in("search[entity]") == []  # a bracket that holds no number is not a citation


def test_exact_match_and_f1_take_the_best_gold_answer():
    golds = ["Barack Obama", "Obama"]
    assert exact_match("obama", golds) == 1.0
    assert exact_match("President Obama", golds) == 0.0
    assert token_f1("President Obama", golds) == pytest.approx(2 * 0.5 * 1.0 / 1.5)  # vs "Obama": p 1/2, r 1
    assert token_f1("Barack Obama", golds) == 1.0
    assert token_f1("nothing alike", golds) == 0.0
    assert token_f1("anything", []) == 0.0


def test_yes_no_golds_score_only_on_a_clear_answer():
    assert token_f1("yes", ["no"]) == 0.0 and token_f1("yes", ["yes"]) == 1.0
    assert contains_answer("Yes, both films were directed by him [1].", ["yes"]) == 1.0
    assert contains_answer("There is no doubt: yes, they were.", ["no"]) == 0.0  # "no" inside a sentence


def test_contains_needs_whole_words():
    assert contains_answer("The capital is Paris [2].", ["Paris"]) == 1.0
    assert contains_answer("Parisian food", ["Paris"]) == 0.0


def test_numbers_with_separators_units_signs_and_no_citation_numbers():
    found = numbers_in("Window 4,096, context 8192 [3], about 131K tokens, $201.3 billion, -0.55% and 1980-2024.")
    assert 4096 in found and 8192 in found and 131 in found and 131_000 in found
    assert 201.3 in found and pytest.approx(201.3e9) in found
    assert -0.55 in found and 2024 in found and -2024 not in found
    assert 3 not in found  # the citation marker
    assert numbers_in("8 KV heads, 160mg, 35 km") == [8, 160, 35]  # no unit taken from KV, mg or km
    assert numbers_in("v4 and M5 events") == []


def test_numbers_match_with_tolerance_and_alternatives():
    ok, missing = numbers_match("South Carolina grew 1.456% [1].", [{"value": 1.46, "tol": 0.01}])
    assert ok and missing == []
    ok, missing = numbers_match("Katrina cost $201,297.5 million.",
                                [{"any": [{"value": 201.3e9, "rel": 0.002}, {"value": 201.3, "tol": 0.05}]}])
    assert ok
    ok, missing = numbers_match("33, 57, 67 and 131", [{"value": v} for v in (33, 57, 67, 131, 115)])
    assert not ok and missing == ["115"]
    assert numbers_match("about 1.28 C", [{"value": 1.29, "tol": 0.03, "label": "~1.29 C"}]) == (True, [])
    assert numbers_match("about 1.35 C", [{"value": 1.29, "tol": 0.03, "label": "~1.29 C"}]) == (False, ["~1.29 C"])


def test_facts_phrases_alternatives_regexes_and_forbidden_answers():
    answer = "It uses a CLIP ViT-L/336px encoder [1] and an MLP; the 2020s are partial (2020-2024) [2]."
    assert loose("CLIP-ViT-L-336px") == loose("CLIP ViT L 336px")
    ok, failed = facts_match(answer, ["CLIP-ViT-L 336px", {"any": ["two-layer MLP", "MLP"]},
                                      {"regex": r"partial|incomplete"}])
    assert ok and failed == []
    ok, failed = facts_match(answer, ["Q-Former"], forbidden=[r"\b11\.5\b"])
    assert not ok and failed == ["missing: Q-Former"]
    ok, failed = facts_match("11.5 per decade in the 2020s", [], forbidden=[r"\b11\.5\b"])
    assert not ok and failed == [r"forbidden: /\b11\.5\b/"]
    assert facts_match("alpha and beta", [{"all": ["alpha", "beta"]}]) == (True, [])
    with pytest.raises(ValueError):
        facts_match("x", [{"nope": 1}])


def test_facts_match_whole_words_only():
    assert facts_match("nineteen states", ["nine"])[0] is False
    assert facts_match("nine states", ["nine"])[0] is True


SOURCES = [{"n": 1, "doc_id": "a", "doc_name": "a.pdf"}, {"n": 2, "doc_id": "b", "doc_name": "b.csv"},
           {"n": 4, "doc_id": "c", "doc_name": "c.txt"}]


def test_citation_check_resolves_and_counts_expected_documents():
    good = citation_check("A [1] and B [2].", SOURCES, ["a", "b"])
    assert good["ok"] and good["all_expected"] and good["cited_docs"] == ["a", "b"]
    partial = citation_check("A [1].", SOURCES, ["a", "b"])
    assert partial["ok"] and not partial["all_expected"]
    stray = citation_check("A [1] and C [4].", SOURCES, ["a"])
    assert not stray["ok"] and stray["outside"] == ["c"] and stray["cites_expected"]
    broken = citation_check("A [1] and [3].", SOURCES, ["a"])
    assert not broken["ok"] and broken["unresolved"] == [3]
    wrong = citation_check("C [4].", SOURCES, ["a"])
    assert not wrong["ok"] and not wrong["cites_expected"]
    none = citation_check("No markers.", SOURCES, ["a"])
    assert not none["ok"] and none["cited"] == []
    by_name = citation_check("B [2].", SOURCES, ["b.csv"], key="doc_name")
    assert by_name["ok"]


def test_citing_a_python_run_is_neither_expected_nor_outside():
    sources = SOURCES + [{"n": 5, "doc_id": "", "doc_name": "Python", "kind": "python"}]
    both = citation_check("A [1], computed [5].", sources, ["a"])
    assert both["ok"] and both["computed"] == [5] and both["outside"] == [] and both["cited_docs"] == ["a"]
    alone = citation_check("Computed [5].", sources, ["a"])
    assert not alone["ok"] and not alone["cites_expected"] and alone["resolve"]
    stray = citation_check("C [4] and [5].", sources, ["a"])
    assert not stray["ok"] and stray["outside"] == ["c"]
