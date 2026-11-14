"""Parsers, one per file type. `parse_file` is the single entry point and runs in the isolated child
process, so every format (not only PDF) gets the timeout, the memory isolation and crash detection."""

from pathlib import Path

from ingest.elements import ParsedDocument


def parse_file(path: str, ext: str, name: str, artifacts: str | None, threads: int) -> ParsedDocument:
    """Read and parse one stored upload. Imports stay inside, so the child loads only what it needs."""
    if ext == ".pdf":
        from ingest.parsers.pdf import parse_pdf

        return parse_pdf(path, artifacts, threads)
    if ext == ".xlsx":
        from ingest.parsers.sheets import read_xlsx

        return read_xlsx(Path(path))
    data = Path(path).read_bytes()
    if ext == ".csv":
        from ingest.parsers.sheets import read_csv

        return read_csv(data, name)
    if ext in (".txt", ".md"):
        from ingest.parsers.text import parse_text

        return parse_text(data, markdown=ext == ".md")
    from ingest.parsers.images import parse_image

    return parse_image(data)
