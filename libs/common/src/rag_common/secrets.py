"""Credential detection and redaction for everything that is STORED: thread messages and titles, summaries,
memories and the audit log. The model still sees the user's original words (it runs locally and the
words may matter to the question); a credential is just never written down or remembered."""

import re
from pathlib import Path

REDACTED = "[redacted]"

_STRONG = r"password|passwd|pwd|passcode|api[ _-]?key|secret[ _-]?key|access[ _-]?token|credentials?"
_WEAK = r"pin|token|secret"  # common words ("the pin is bent"): only an explicit "pin: 1234" counts
_VALUE = r"(\"[^\"]+\"|'[^']+'|\S+)"

# "password is hunter2", "pwd=abc123", "api key: abc" keep the label and drop the value
_LABELLED = re.compile(rf"\b({_STRONG})(\s*(?:\bis\b|\bwas\b|=|:)\s*){_VALUE}", re.I)
_LABELLED_WEAK = re.compile(rf"\b({_WEAK})(\s*[=:]\s*){_VALUE}", re.I)
# "password hunter2": a bare label followed by something that looks like a value, not a word ("password reset")
_BARE = re.compile(rf"\b({_STRONG})(\s+)(?=\S*(?:\d|[^\w\s.,;!?]|(?-i:[a-z][A-Z])))(\S+)", re.I)
# scheme://user:password@host
_URL_CREDENTIALS = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s/:@]+:)([^\s/@]+)(@)", re.I)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_PROVIDER = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|xox[abpr]-[A-Za-z0-9-]{10,})")
# random keys: one unbroken run of 24+ letters and digits mixing all three classes
_RUN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])")


def _random_looking(run: str) -> bool:
    """Random strings switch between lower, upper and digit at ~60% of positions; words and CamelCase
    identifiers ("getUserAccountBalance2024") switch far less, hex hashes have no upper case."""
    body = run.rstrip("=")
    classes = ["d" if c.isdigit() else "u" if c.isupper() else "l" if c.islower() else "s" for c in body]
    if not {"d", "u", "l"} <= set(classes):
        return False
    switches = sum(a != b for a, b in zip(classes, classes[1:]))
    return switches >= 0.35 * len(body)


def redact_secrets(text: str) -> str:
    text = _URL_CREDENTIALS.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(3)}", text)
    text = _JWT.sub(REDACTED, text)
    text = _PROVIDER.sub(REDACTED, text)
    for pattern in (_LABELLED, _LABELLED_WEAK, _BARE):
        text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return _RUN.sub(lambda m: REDACTED if _random_looking(m.group(0)) else m.group(0), text)


def looks_secret(text: str) -> bool:
    """True when the text mentions or contains a credential; such text is never remembered."""
    return redact_secrets(text) != text or bool(
        re.search(r"\b(password|passcode|api[ _-]?key|credential|access[ _-]?token)\b", text, re.I))


def read_secret(path: Path | None) -> str | None:
    """Contents of a secret file (Docker secrets live in /run/secrets), whitespace stripped; None without a path.
    A missing or empty file is an error, so a misconfigured key fails at startup instead of being skipped."""
    if path is None:
        return None
    value = Path(path).read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"secret file {path} is empty")
    return value
