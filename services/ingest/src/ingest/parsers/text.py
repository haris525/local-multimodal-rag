"""Plain text and Markdown -> paragraph elements with a heading path."""

import re

from charset_normalizer import from_bytes

from ingest.elements import Element, ParsedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+\u2022]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(```|~~~)")


def decode(data: bytes) -> str:
    match = from_bytes(data).best()
    text = str(match) if match else data.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def _flush(lines: list[str], section: str, out: list[Element]) -> None:
    """Join hard-wrapped lines of one paragraph; keep list items and code on their own lines."""
    if not lines:
        return
    keep_breaks = any(_LIST_ITEM.match(l) for l in lines) or _FENCE.match(lines[0])
    text = "\n".join(lines) if keep_breaks else " ".join(l.strip() for l in lines)
    if text.strip():
        out.append(Element(kind="text", text=text.strip(), section=section))
    lines.clear()


def parse_text(data: bytes, markdown: bool) -> ParsedDocument:
    elements: list[Element] = []
    headings: list[str] = []
    section, buffer, in_code = "", [], False
    for line in decode(data).split("\n"):
        if _FENCE.match(line):
            if not in_code:
                _flush(buffer, section, elements)
            buffer.append(line)
            in_code = not in_code
            if not in_code:
                _flush(buffer, section, elements)
            continue
        if in_code:
            buffer.append(line)
            continue
        heading = _HEADING.match(line) if markdown else None
        if heading:
            _flush(buffer, section, elements)
            level = len(heading.group(1))
            headings = headings[: level - 1] + [heading.group(2).strip()]
            section = " > ".join(headings)
        elif not line.strip():
            _flush(buffer, section, elements)
        else:
            buffer.append(line)
    _flush(buffer, section, elements)
    doc = ParsedDocument(elements=elements)
    if not elements:
        doc.warnings.append("No text found")
    return doc
