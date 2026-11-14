"""Token counts for prompt budgets.

One llama-server slot holds slot_tokens (32,768: LLM_CTX 65,536 over 2 parallel slots), and every prompt plus
its output room must fit in it. Qwen writes each digit as its own token, so SQL results and raw sheet cells
take 1.6-2.3 times the tokens a chars/3 estimate gives them: with that estimate a research prompt counted as
13,500 tokens was really 32,000.

count_tokens(text) uses the model family's tokenizer.json when it is available (RAG_TOKENIZER_FILE, the same file
the ingest worker counts chunks with; `configure` loads it at startup), times RAG_TOKENIZER_FACTOR: llama-server's
own tokenizer for Qwen3.8 counts up to 3.4% more than that file on English and numbers (measured with its
/tokenize endpoint), so the default factor 1.05 keeps every count at or above the model's. Without it, the estimate counts every
digit, punctuation mark, symbol and space before a number as one token, every non-ASCII character as two,
every word as one token per 3.5 letters, every run of capitals ("ABW", "SP.POP.TOTL") as one per 1.5 letters,
every run of random-looking mixed case ("aZx9Qm" in an ID) as one per 1.3 letters, and every line break and run
of spaces as one, then adds 5%: 1.07-1.3x llama-server's own count on numbers, tables, codes, IDs and JSON,
about 1.6x on English, German or French prose and 2.4x on Chinese (tests/fixtures/served_tokens.json holds the
texts and the served counts).
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

log = logging.getLogger("rag.tokens")

_LETTERS = re.compile(r"[A-Za-z]+")
_SEGMENTS = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z]?[a-z]+|[A-Z]")  # "GDPgrowth" -> GDP, growth
ESTIMATE_MARGIN = 1.05
_DIGITS = re.compile(r"[0-9]")
_SPACE_DIGIT = re.compile(r" (?=[0-9])")  # a space before a number is a token of its own
_BREAKS = re.compile(r"\n+|  +|\t+")
_OTHER = re.compile(r"[^A-Za-z0-9\s]")
MESSAGE_TOKENS = 8  # chat template markers around one message (<|im_start|>role ... <|im_end|>)
TOOLS_PREAMBLE_TOKENS = 200  # the chat template's text around the tool schemas

_tokenizer = None
_factor = 1.0


def configure(tokenizer_file: str | None, factor: float = 1.05) -> bool:
    """Count with this tokenizer.json (times `factor`) from now on; False (and the estimate) when it cannot be
    loaded."""
    global _tokenizer, _factor
    _tokenizer, _factor = None, max(1.0, factor)
    if not tokenizer_file:
        return False
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(Path(tokenizer_file)))
        tok.no_truncation()
        tok.no_padding()
    except Exception as exc:  # noqa: BLE001 - a missing file or package means the estimate, which is logged
        log.warning("tokenizer not loaded; prompt budgets use the conservative estimate",
                    extra={"fields": {"file": tokenizer_file, "error": f"{type(exc).__name__}: {exc}"[:200]}})
        return False
    _tokenizer = tok
    return True


def exact() -> bool:
    return _tokenizer is not None


def _letter_tokens(run: str) -> int:
    """A run of letters: a word takes one token per 3.5 letters; capitals ("ABW", "TOTL") one per 1.5, and a run
    whose case keeps flipping (an ID or key) one per 1.3, which is how Qwen's tokenizer splits them."""
    if run.islower() or (run[0].isupper() and run[1:].islower()):
        return math.ceil(len(run) / 3.5)
    flips = sum(a.isupper() != b.isupper() for a, b in zip(run, run[1:], strict=False))
    if flips >= 3 and flips >= len(run) / 4:
        return math.ceil(len(run) / 1.3)
    return sum(math.ceil(len(seg) / (1.5 if seg.isupper() else 3.5)) for seg in _SEGMENTS.findall(run))


def estimate(text: str) -> int:
    """The conservative estimate (see the module docstring)."""
    letters = sum(_letter_tokens(run) for run in _LETTERS.findall(text))
    non_ascii = sum(1 for ch in text if ord(ch) > 127)  # counted once in _OTHER; rare characters take 2-3
    raw = (letters + len(_DIGITS.findall(text)) + len(_SPACE_DIGIT.findall(text)) + len(_OTHER.findall(text))
           + non_ascii + len(_BREAKS.findall(text)) + 1)
    return math.ceil(raw * ESTIMATE_MARGIN)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _tokenizer is not None:
        return math.ceil(len(_tokenizer.encode(text, add_special_tokens=False).ids) * _factor)
    return estimate(text)


def truncate_tokens(text: str, limit: int) -> str:
    """The longest start of `text` within `limit` tokens."""
    if limit <= 0:
        return ""
    if count_tokens(text) <= limit:
        return text
    if _tokenizer is not None:
        offsets = _tokenizer.encode(text, add_special_tokens=False).offsets
        return text[:offsets[max(0, int(limit / _factor) - 1)][1]]
    lo, hi = 0, len(text)
    while lo < hi:  # the estimate grows with the length, so a binary search finds the cut
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def message_tokens(messages: list) -> int:
    """A chat prompt's size: langchain messages (content and tool calls) or OpenAI-style dicts (text parts;
    images are counted by the caller, who knows the per-image cost)."""
    total = 0
    for m in messages:
        total += MESSAGE_TOKENS
        if isinstance(m, dict):
            content = m.get("content") or ""
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            total += sum(count_tokens(p.get("text", "")) for p in parts if p.get("type") == "text")
            continue
        content = getattr(m, "content", "")
        total += count_tokens(content if isinstance(content, str) else json.dumps(content))
        for call in getattr(m, "tool_calls", None) or []:
            total += count_tokens(json.dumps({"name": call.get("name"), "arguments": call.get("args")})) + 4
    return total


def image_count(messages: list[dict]) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")
