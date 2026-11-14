"""Reading the layout corpus: each file's tables as the layout reader and as the simple reader see them.

`tables_of(path)` gives [{sheet, columns, rows, types, reader}] from ingest.parsers.sheets (the layout reader,
with its fallback); `simple_of(path)` gives [{sheet, columns, rows}] from the simple reader on the same
cells. Types are the SQL types write_tables would give (BIGINT, DOUBLE, DATE, TIMESTAMP, VARCHAR).

    uv run python tests/fixtures/layouts/corpus.py --refresh    # from services/ingest: rewrites expected.json

Refreshing records what the code does now: review the diff of expected.json line by line before committing
it, since the corpus test then holds the code to it.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ingest.parsers.sheets import csv_grid, read_csv, read_xlsx, workbook_grids
from ingest.parsers.simple import simple_read
from ingest.tables import coerce_column

HERE = Path(__file__).parent
SAMPLES = Path(__file__).resolve().parents[5] / "examples" / "documents" / "data"
EXPECTED = HERE / "expected.json"
SIMPLE_RIGHT = HERE / "simple_right.txt"  # hand-labelled: never written by refresh()


def _sql_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    if str(series.dtype) == "Int64":
        return "BIGINT"
    if str(series.dtype) == "Float64":
        return "DOUBLE"
    values = series.dropna()
    if len(values) and all(isinstance(v, date) for v in values):
        return "DATE"
    return "VARCHAR"


def tables_of(path: Path) -> list[dict]:
    doc = read_xlsx(path) if path.suffix == ".xlsx" else read_csv(path.read_bytes(), path.name)
    out = []
    for s in doc.sheets:
        types = {str(c): _sql_type(coerce_column(s.frame.iloc[:, i].reset_index(drop=True), s.headers.get(str(c), str(c))))
                 for i, c in enumerate(s.frame.columns)}
        out.append({"sheet": s.name, "columns": [str(c) for c in s.frame.columns], "rows": len(s.frame), "types": types,
                    "reader": s.layout.get("reader", "layout")})
    return out


def simple_of(path: Path) -> list[dict]:
    if path.suffix == ".xlsx":
        sheets = workbook_grids(path)
    else:
        sheets = [(path.stem, csv_grid(path.read_bytes()), [])]
    out = []
    for title, grid, merges in sheets:
        frame, _, _ = simple_read(grid, merges, 1)
        if not frame.empty:
            out.append({"sheet": title, "columns": [str(c) for c in frame.columns], "rows": len(frame)})
    return out


def corpus_files() -> list[Path]:
    return sorted(p for p in HERE.iterdir() if p.suffix in (".csv", ".xlsx"))


def sample_files() -> list[Path]:
    return sorted(p for p in SAMPLES.rglob("*") if p.suffix in (".csv", ".xlsx")) if SAMPLES.is_dir() else []


def _same(a: list[dict], b: list[dict]) -> bool:
    return [(t["columns"], t["rows"]) for t in a] == [(t["columns"], t["rows"]) for t in b]


def simple_right() -> list[str]:
    """The hand-labelled files the simple reader reads right (simple_right.txt, one name per line, # comments)."""
    lines = SIMPLE_RIGHT.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def refresh() -> None:
    """Rewrite expected.json from what the code reads now (the parity list, simple_right.txt, is labelled by hand
    and never written here)."""
    corpus = {p.name: tables_of(p) for p in corpus_files()}
    expected = {"corpus": corpus, "samples": {p.relative_to(SAMPLES).as_posix(): tables_of(p) for p in sample_files()}}
    EXPECTED.write_text(json.dumps(expected, indent=1, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(expected['corpus'])} corpus files, {len(expected['samples'])} samples -> {EXPECTED.name}")
    differ = [n for n in simple_right() if not _same(corpus[n], simple_of(HERE / n))]
    if differ:
        print(f"the layout reader differs from the simple reader on {len(differ)} hand-labelled files: {', '.join(differ)}")


if __name__ == "__main__" and "--refresh" in sys.argv:
    refresh()
