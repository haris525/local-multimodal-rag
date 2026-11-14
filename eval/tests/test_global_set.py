"""The global set (eval/benchmarks/global_samples.jsonl): whole-document questions for the exhaustive answer mode. It
loads, it is ASCII, every file it selects exists, every question has its evidence and every quote is in its source,
every reference answer passes its own checks, partial answers fail them, and correct answers in other words pass."""

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest
from rag_eval.bench import TAGS, load_items, score
from rag_eval.paths import NAMED_SETS

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "examples" / "documents"
EVIDENCE = ROOT / "eval" / "benchmarks" / "global_evidence"
GLOBAL_SET = NAMED_SETS["global"]
ITEMS = {it.id: it for it in load_items(GLOBAL_SET)}
KINDS = ("enumeration", "coverage", "counting")
# An evidence bullet: the source line, then the quote on the next line.
SOURCE = re.compile(r"^\s*- (?P<pdf>\S+\.pdf), page (?P<page>\d+)\s*$"
                    r"|^\s*- (?P<txt>\S+\.txt) \(plain text, no pages\)\s*$")
QUOTE = re.compile(r'^\s*- "(?P<quote>.*)"\s*$')
# Quotes are written ASCII: the PDF text after NFKC (ligatures), with curly quotes and dashes folded and whitespace
# collapsed. A line-break hyphen stays as extracted ("specifi- cally"), as in the hard set.
FOLD = str.maketrans({0x2018: "'", 0x2019: "'", 0x201A: "'", 0x2032: "'", 0x201C: '"', 0x201D: '"', 0x201E: '"',
                      0x2013: "-", 0x2014: "-", 0x2212: "-", 0x00A0: " "})  # curly quotes, dashes, no-break space


def check(item_id: str, answer: str) -> tuple[bool, list[str]]:
    it = ITEMS[item_id]
    s, why = score(it, answer, [{"n": 1, "doc_id": it.expected_docs[0]}], it.expected_docs)
    return bool(s["correct"]), why


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).translate(FOLD).split())


def doc_path(name: str) -> Path:
    hits = [p for p in DOCS.rglob(name) if p.is_file()]
    assert len(hits) == 1, name
    return hits[0]


def evidence(item_id: str) -> list[tuple[str, int, str]]:
    """(file name, page or 0 for the txt file, quote) for every evidence bullet of a question."""
    lines = (EVIDENCE / f"{item_id}.md").read_text(encoding="ascii").splitlines()
    out = []
    for i, line in enumerate(lines):
        m = SOURCE.match(line)
        if m:
            q = QUOTE.match(lines[i + 1]) if i + 1 < len(lines) else None
            assert q, (item_id, i + 2)
            out.append((m.group("pdf") or m.group("txt"), int(m.group("page") or 0), q.group("quote")))
    return out


def test_the_set_is_ascii_global_and_covers_each_kind():
    raw = GLOBAL_SET.read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw
    assert 12 <= len(ITEMS) <= 20
    assert sorted(ITEMS, key=lambda i: int(i.split("-")[1])) == [f"global-{n}" for n in range(1, len(ITEMS) + 1)]
    tags = Counter(t for it in ITEMS.values() for t in it.tags)
    assert tags["global"] == len(ITEMS) and set(tags) <= set(TAGS)
    assert all(tags[k] >= 3 for k in KINDS), tags
    for it in ITEMS.values():
        assert set(it.tags) & set(KINDS), it.id
        assert ("cross-document" in it.tags) == (len(it.docs) > 1), it.id
        assert ("single-document" in it.tags) == (len(it.docs) == 1), it.id
        assert it.correct_by == "checks" and it.answers == [it.reference], it.id


def test_every_selected_file_exists_and_is_a_paper_or_the_constitution():
    samples = {p.name for p in DOCS.rglob("*") if p.is_file()}
    for it in ITEMS.values():
        assert set(it.docs) <= samples, it.id
        assert 1 <= len(it.docs) <= 4 and set(it.expected_docs) == set(it.docs), it.id
        assert all(d.endswith((".pdf", ".txt")) for d in it.docs), it.id  # no spreadsheets or images


def test_every_question_has_its_evidence():
    """The page and the exact quote for each fact and number, in the hard set's format."""
    for item_id, it in ITEMS.items():
        path = EVIDENCE / f"{item_id}.md"
        assert path.is_file(), item_id
        raw = path.read_bytes()
        text = raw.decode("ascii")
        assert b"\r" not in raw, item_id
        assert text.startswith(f"# {item_id}\n"), item_id
        for heading in ("## Evidence", "## Arithmetic and notes", "## What makes it global"):
            assert f"\n{heading}\n" in text, (item_id, heading)
        cited = r'\.(pdf, page \d+|txt \(plain text, no pages\))\n\s*- "'  # the source, then the quote
        assert re.search(cited, text), item_id
        sources = {name for name, _, _ in evidence(item_id)}
        assert sources and sources <= set(it.docs), item_id
        assert set(it.expected_docs) <= sources, item_id  # every expected document has evidence


def test_constitution_quotes_are_in_the_text():
    text = norm(doc_path("us-constitution-1787.txt").read_text(encoding="utf-8"))
    quotes = [(i, q) for i in ITEMS for name, _, q in evidence(i) if name.endswith(".txt")]
    assert quotes
    missing = [(i, q) for i, q in quotes if norm(q) not in text]
    assert not missing, missing


def test_pdf_quotes_are_on_their_pages():
    """Needs pypdf, which is not an eval dependency: run with `uv run --project eval --with pypdf python -m pytest`."""
    pypdf = pytest.importorskip("pypdf")

    @lru_cache(maxsize=None)
    def pages(name: str) -> list[str]:
        return [norm(p.extract_text() or "") for p in pypdf.PdfReader(str(doc_path(name))).pages]

    quotes = [(i, name, page, q) for i in ITEMS for name, page, q in evidence(i) if name.endswith(".pdf")]
    assert quotes
    missing = [(i, name, page, q) for i, name, page, q in quotes
               if not 1 <= page <= len(pages(name)) or norm(q) not in pages(name)[page - 1]]
    assert not missing, missing


@pytest.mark.parametrize("item_id", sorted(ITEMS))
def test_each_reference_answer_passes_its_own_checks(item_id):
    ok, why = check(item_id, ITEMS[item_id].reference + " [1]")
    assert ok, why


def test_partial_answers_fail():
    """Realistic answers that miss one or two items of the list (or count wrong) must not count as correct."""
    partial = {
        "global-1": ["A Representative must be at least 25 and seven years a citizen, a Senator at least 30 and nine "
                     "years a citizen, and the President at least 35."],
        "global-2": ["The Vice President is President of the Senate and votes only to break a tie; the Senate chooses "
                     "a president pro tempore when he is absent, and he takes over the President's powers if the "
                     "President dies or is removed."],
        "global-3": ["Two thirds is needed to convict on impeachment, to override a veto, to approve treaties, to "
                     "expel a member and to propose amendments: five times."],
        "global-4": ["Treason is defined in Article III, Section 3: levying war against the United States or adhering "
                     "to its enemies, giving them aid and comfort; conviction needs two witnesses to the same overt "
                     "act or a confession in open court."],
        "global-5": ["Article I has ten Sections, Article II has four and Article III has three.",
                     "Article I has 10 Sections, Article II has 4, Article III has four Sections and Article IV three; "
                     "21 in all, and Articles V to VII have none."],
        "global-6": ["The BioASQ experiment used 4017 documents and 1000 questions (Section 4.3, Table 1)."],
        "global-7": ["Sliding window attention gives a 2x speed-up and the rolling buffer cache cuts cache memory by "
                     "8x."],
        "global-8": ["ReAct can repeat the same thoughts and actions; RAISE struggles with complex logic; Reflexion "
                     "can get stuck in local minima; AutoGPT+P's tool selection is unreliable."],
        "global-9": ["BM25 is the sparse baseline on TREC DL19/20 and the sparse half of Hybrid Search."],
        "global-10": ["HotpotQA is a MultiHop QA dataset in the comprehensive evaluation, with 500 sub-sampled entries "
                      "scored by token F1 and EM, and it is one of three datasets in the summarization comparison."],
        "global-11": ["Section 3.1 gives 95 models from Torch Hub and 626 of 801 from TensorFlow Hub v2; with "
                      "HuggingFace's 925 that is 1,645."],
        "global-12": ["On MIRACL and MKQA, Dense+Sparse uses w1 = 1, w2 = 0.3, w3 = 0 and All uses w1 = 1, w2 = 0.3, "
                      "w3 = 1; training uses 1, 0.3, 1."],
        "global-13": ["M3-Embedding supports more than 100 languages; MIRACL covers 18 languages, MKQA 25 and the "
                      "unsupervised data 194."],
        "global-14": ["HotpotQA uses 6 exemplars and FEVER 3; ALFWorld uses 6 prompts per task type and WebShop two "
                      "examples.",
                      "HotpotQA 6, FEVER 3, ALFWorld 3 and WebShop 1."],
        "global-15": ["Ragas: two annotators agreed about 95% (faithfulness, context relevance) and 90% (answer "
                      "relevance). Mistral: on llmboxing.com Mistral 7B was preferred 5020 times against 4143 for "
                      "Llama 2 13B."],
        "global-16": ["Gorilla fine-tunes LLaMA-7B on 16,450 self-instruct instruction-API pairs; the best-practices "
                      "paper fine-tunes Llama-2-7B with LoRA on QA datasets; Mistral fine-tunes Mistral 7B into "
                      "Mistral 7B Instruct on public instruction datasets from Hugging Face."],
        "global-17": ["BGE-M3 builds its dense index with Faiss and its sparse index with Lucene and evaluates MIRACL "
                      "with Pyserini; the best-practices paper builds a BM25 index with Lucene on MS MARCO and a Faiss "
                      "Flat dense index, using Pyserini."],
    }
    assert set(partial) == set(ITEMS)
    for item_id, answers in partial.items():
        for answer in answers:
            ok, why = check(item_id, answer + " [1]")
            assert not ok, (item_id, answer, why)


def test_correct_answers_in_other_words_pass():
    other_words = {
        "global-1": "Representatives: age 25+, U.S. citizen for 7 years, must live in the state they represent. "
                    "Senators: 30+, citizen for 9 years, inhabitant of their state. President: natural-born citizen, "
                    "at least 35, 14 years resident in the US.",
        "global-3": "It appears nine times: conviction in impeachment trials (2/3 of senators present); expelling a "
                    "member; passing a bill over a veto in each house (two occurrences); repassing orders and "
                    "resolutions; the quorum of states when the House picks the President; treaties; Congress "
                    "proposing amendments; and state legislatures applying for a convention.",
        "global-5": "Only Articles I-IV have sections: Article I has 10, Article II has 4, Article III has 3, Article "
                    "IV has 4, for a total of twenty-one. Articles V, VI and VII have none.",
        "global-6": "Two figures: 15,000 documents (introduction and conclusion) and 4,017 documents (Section 4.3 and "
                    "Table 1); 1,000 questions in both.",
        "global-7": "Speed doubles with SWA at 16K tokens (2x), the rolling buffer cache needs 8 times less memory at "
                    "32k tokens, Mistral 7B matches a Llama 2 more than three times its size on reasoning and MMLU, "
                    "and only 1.9x on knowledge tasks.",
        "global-11": "TorchHub: 94 (intro, Fig. 3) vs 95 (Sec. 3.1, appendix 8.1). TensorFlow Hub: 696 (intro, "
                     "appendix) vs 801 models reduced to 626 (Sec. 3.1, Fig. 3). 94 + 626 + 925 = 1,645.",
        "global-14": "Six exemplars for HotpotQA, three for FEVER, two per prompt for ALFWorld (from three annotated "
                     "per task type) and a single one-shot example for WebShop.",
        "global-16": "**Gorilla**: base model LLaMA-7B, trained on APIBench (16,450 instruction-API pairs) to write API "
                     "calls. **Epochs:** 5. **Learning rate:** 2e-5. **Best practices**: the Llama-2-7B generator, "
                     "fine-tuned with LoRA; | epochs | 3 |, learning rate 5e-5; the query classifier is "
                     "BERT-base-multilingual-cased, learning rate 1e-5. **Mistral**: Mistral 7B - Instruct, trained on "
                     "publicly available instruction datasets from Hugging Face.",
    }
    for item_id, answer in other_words.items():
        ok, why = check(item_id, answer + " [1]")
        assert ok, (item_id, why)
