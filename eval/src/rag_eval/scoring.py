"""Model-free answer metrics: no judge model, so two runs score the same answer the same way.

- SQuAD v1.1 normalization, exact match and token F1, each the best over several gold answers (an answer
  and its aliases), plus `contains`: a gold answer appears as whole words in the reply, which suits the
  sentence-long replies a chat app gives.
- Numbers: every expected number appears in the reply, within a tolerance, also when the reply writes
  it with a unit word ("201.3 billion", "131K", "4,096").
- Required facts: every listed fact (a phrase, one of several phrasings, or a regular expression) is
  present, and no forbidden pattern (a known wrong answer) is.
- Citations: every [n] resolves to a source, cited sources come from the expected documents, and at
  least one citation points at an expected document.

Citation markers are removed before every text metric, so a cited answer is not penalized for them.
"""

import re
import string
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

# The marker rag writes and parses (services/rag/src/rag/prompts.py, _CITATION): [3], [1, 2], [4; 5].
CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def citations_in(text: str) -> list[int]:
    """The source numbers an answer cites, in first-seen order, each once."""
    seen: dict[int, None] = {}
    for group in CITATION.findall(text or ""):
        for part in re.split(r"\s*[,;]\s*", group):
            seen.setdefault(int(part), None)
    return list(seen)


def strip_citations(text: str) -> str:
    """The answer without its [n] markers (and the space before each)."""
    return re.sub(r"\s*" + CITATION.pattern, "", text or "")


# --- SQuAD exact match and F1 -------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """SQuAD v1.1: lower case, remove punctuation, remove the articles a/an/the, collapse whitespace."""
    text = "".join(ch for ch in (text or "").lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _golds(golds: str | Iterable[str]) -> list[str]:
    items = [golds] if isinstance(golds, str) else list(golds)
    return [g for g in items if g is not None and str(g).strip()]


def exact_match(prediction: str, golds: str | Iterable[str]) -> float:
    """1.0 when the normalized prediction equals a normalized gold answer (the best over golds)."""
    pred = normalize_answer(strip_citations(prediction))
    return max((float(pred == normalize_answer(g)) for g in _golds(golds)), default=0.0)


def _f1(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    # HotpotQA's official script: a yes/no/noanswer gold (or prediction) scores only on an exact match.
    if (pred in ("yes", "no", "noanswer") or gold in ("yes", "no", "noanswer")) and pred != gold:
        return 0.0
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: str | Iterable[str]) -> float:
    """SQuAD token F1 between the prediction and the best-matching gold answer."""
    pred = normalize_answer(strip_citations(prediction))
    return max((_f1(pred, normalize_answer(g)) for g in _golds(golds)), default=0.0)


def contains_answer(prediction: str, golds: str | Iterable[str]) -> float:
    """1.0 when a normalized gold answer appears as whole words in the normalized prediction. A yes/no
    gold must be the reply's first word instead (a long reply says "no" for many reasons)."""
    pred = normalize_answer(strip_citations(prediction))
    words = pred.split()
    for gold in _golds(golds):
        g = normalize_answer(gold)
        if not g:
            continue
        if g in ("yes", "no"):
            if words[:1] == [g]:
                return 1.0
        elif f" {g} " in f" {pred} ":
            return 1.0
    return 0.0


# --- numbers --------------------------------------------------------------------------------------

_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "mio": 1e6, "million": 1e6, "millions": 1e6,
          "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9, "t": 1e12, "tn": 1e12, "trillion": 1e12,
          "trillions": 1e12}
_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"  # 4,096 / 8192 / 1.28 (not v4, M5 or 1.2.3's tail)
    r"(?:\s?(thousand|millions?|billions?|trillions?|mn|mio|bn|tn|[kKmMbBtT])\b)?"
)
_MINUS = ("-", "\u2212", "\u2013")  # hyphen, minus sign, en dash


def numbers_in(text: str) -> list[float]:
    """Every number in the text, plus its value scaled by a unit word after it ("1.2M" gives 1.2 and
    1,200,000). Citation markers are removed first, so [3] is not a 3. A leading minus counts only at
    the start of a word: in "1980-2024" the 2024 stays positive."""
    text = unicodedata.normalize("NFKC", strip_citations(text))
    out: list[float] = []
    for m in _NUMBER.finditer(text):
        value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        start = m.start()
        if start and text[start - 1] in _MINUS and (start == 1 or not text[start - 2].isalnum()):
            value = -value
        out.append(value)
        unit = (m.group(3) or "").lower()
        if unit in _SCALE:
            out.append(value * _SCALE[unit])
    return out


def _number_ok(spec: dict, found: Sequence[float]) -> bool:
    if "any" in spec:
        return any(_number_ok(alt, found) for alt in spec["any"])
    value = float(spec["value"])
    room = max(float(spec.get("tol", 0)), float(spec.get("rel", 0)) * abs(value)) + 1e-9
    return any(abs(x - value) <= room for x in found)


def _label(spec: dict) -> str:
    if "label" in spec:
        return str(spec["label"])
    if "any" in spec:
        return " | ".join(_label(alt) for alt in spec["any"])
    return f"{spec['value']:g}"


def numbers_match(prediction: str, specs: Sequence[dict]) -> tuple[bool, list[str]]:
    """(all expected numbers present, labels of the missing ones). A spec is {"value", "tol" (absolute),
    "rel" (relative), "label"} or {"any": [spec, ...]} for numbers that may be written several ways
    (201.3 billion or 201,297.5 million)."""
    found = numbers_in(prediction)
    missing = [_label(s) for s in specs if not _number_ok(s, found)]
    return not missing, missing


# --- required facts -------------------------------------------------------------------------------


def loose(text: str) -> str:
    """Lenient form for fact matching: Unicode folded, lower case, every non-alphanumeric character a
    space, no articles. "CLIP-ViT-L-336px" and "CLIP ViT L 336px" are the same; so are "2/3 cup" and
    "2 3 cup"."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"[^\w]+|_", " ", text)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _fact_ok(fact: str | dict, plain: str, folded: str) -> bool:
    if isinstance(fact, str):
        want = loose(fact)
        return bool(want) and f" {want} " in f" {folded} "
    if "any" in fact:
        return any(_fact_ok(alt, plain, folded) for alt in fact["any"])
    if "all" in fact:
        return all(_fact_ok(part, plain, folded) for part in fact["all"])
    if "regex" in fact:
        return re.search(fact["regex"], plain, re.I | re.S) is not None
    raise ValueError(f"unknown fact form: {fact!r}")


def fact_label(fact: str | dict) -> str:
    if isinstance(fact, str):
        return fact
    if "label" in fact:
        return str(fact["label"])
    if "any" in fact:
        return " | ".join(fact_label(f) for f in fact["any"])
    if "all" in fact:
        return " & ".join(fact_label(f) for f in fact["all"])
    return f"/{fact['regex']}/"


def facts_match(prediction: str, facts: Sequence[str | dict], forbidden: Sequence[str] = ()) -> tuple[bool, list[str]]:
    """(every fact present and nothing forbidden, what failed). A fact is a phrase (matched on whole
    words in the lenient form), {"any": [...]}, {"all": [...]} or {"regex": "..."} (case-insensitive,
    on the answer without citation markers). `forbidden` are regular expressions of known wrong answers."""
    plain = strip_citations(unicodedata.normalize("NFKC", prediction or ""))
    folded = loose(plain)
    failed = [f"missing: {fact_label(f)}" for f in facts if not _fact_ok(f, plain, folded)]
    failed += [f"forbidden: /{p}/" for p in forbidden if re.search(p, plain, re.I | re.S)]
    return not failed, failed


# --- citations ------------------------------------------------------------------------------------


def citation_check(answer: str, sources: Sequence[dict], expected: Iterable[str], key: str = "doc_id") -> dict:
    """Check the answer's [n] markers against the sources it was given.

    resolve: every cited number is a source; only_expected: every cited source belongs to an expected
    document; cites_expected: at least one citation points at an expected document; all_expected: each
    expected document is cited at least once (cross-document questions); ok: cited at all, and
    resolve, only_expected and cites_expected hold. `key` names the source field that identifies a
    document (doc_id, or doc_name). A Python run (a source of kind "python") is a computed result, not a
    document: citing it resolves, but it neither counts as an expected document nor falls outside them."""
    cited = citations_in(answer)
    by_n = {s.get("n"): s for s in sources}
    unresolved = [n for n in cited if n not in by_n]
    computed = [n for n in cited if n in by_n and by_n[n].get("kind") == "python"]
    docs = [str(by_n[n].get(key)) for n in cited if n in by_n and n not in computed]
    want = {str(x) for x in expected}
    outside = sorted({d for d in docs if want and d not in want})
    cites_expected = bool(set(docs) & want) if want else bool(docs)
    out = {
        "cited": cited,
        "unresolved": unresolved,
        "computed": computed,
        "cited_docs": sorted(set(docs)),
        "outside": outside,
        "resolve": not unresolved,
        "only_expected": not outside,
        "cites_expected": cites_expected,
        "all_expected": want <= set(docs) if want else bool(docs),
    }
    out["ok"] = bool(cited) and out["resolve"] and out["only_expected"] and cites_expected
    return out
