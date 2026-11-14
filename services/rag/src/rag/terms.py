"""Pick the parts of a question that must match exactly: the input to the grep retriever."""

import re

_QUOTED = re.compile(r"\"([^\"]{2,100})\"|\u201c([^\u201d]{2,100})\u201d")
_TOKEN = re.compile(r"[\w$\u20ac\u00a3%][\w.,$\u20ac\u00a3%/:-]*[\w%]|[\w$]")
_HAS_DIGIT = re.compile(r"\d")
_HAS_ALPHA = re.compile(r"[A-Za-z]")
_ACRONYM = re.compile(r"^[A-Z]{2,10}s?$")
_CODE_LIKE = re.compile(r"^(?:[a-z]+[A-Z]\w*|\w+_\w+|\w+\.\w+\.?\w*)$")  # camelCase, snake_case, dotted
MAX_TERMS = 8


def _is_exact_term(token: str, shouting: bool) -> bool:
    if _HAS_DIGIT.search(token):
        if _HAS_ALPHA.search(token):
            return True  # INV-2025-0042, Q3, v2.1
        digits = re.sub(r"\D", "", token)
        return len(digits) >= 3 or bool(re.search(r"[$\u20ac\u00a3%.,]", token))  # 4.2, $90, 2026, 137
    if not shouting and _ACRONYM.match(token):
        return True
    return bool(_CODE_LIKE.match(token))


def exact_terms(question: str) -> list[str]:
    """Quoted phrases, identifiers and codes, numbers with 3+ digits or units, acronyms,
    and code-like names. Ordinary words are left to BM25 and dense retrieval."""
    terms: list[str] = [next(g for g in m.groups() if g).strip() for m in _QUOTED.finditer(question)]
    rest = _QUOTED.sub(" ", question)
    shouting = rest.isupper()
    for token in _TOKEN.findall(rest):
        token = token.strip(".,:;/-")
        if token and _is_exact_term(token, shouting):
            terms.append(token)
    unique: list[str] = []
    for term in terms:
        if term.lower() not in {u.lower() for u in unique}:
            unique.append(term)
    return unique[:MAX_TERMS]
