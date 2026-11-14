"""The simple sheet reader of the main branch, kept as the layout reader's safety net.

Merged ranges are filled first; the header is the first row (within the first 30) with at least two filled
cells and at least 60% of the widest row's filled count; a horizontally merged header row over a row of text
labels makes a two-row header; every non-blank row below is data. It gets plain tables right and the
census title row wrong, which is why parsers/layout.py exists. layout.read_table returns this reading of a
table instead of its own when its own is not trustworthy (a header that holds numbers or placeholders, data
rows absorbed into the header, rows that do not fit the column types), with a warning naming why.
"""

from __future__ import annotations

import pandas as pd

HEADER_SCAN_ROWS = 30


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _filled(grid: list[list[object]], merges) -> tuple[list[list[object]], set[int]]:
    """A copy of the grid with each merged range's top-left value in every cell, and the rows holding a
    horizontal merge (candidate group-label rows of a two-row header)."""
    out = [list(r) for r in grid]
    group_rows = set()
    for m in merges:
        if m.r0 >= len(out):
            continue
        value = out[m.r0][m.c0] if m.c0 < len(out[m.r0]) else None
        for r in range(m.r0, min(m.r1, len(out) - 1) + 1):
            for c in range(m.c0, m.c1 + 1):
                if c < len(out[r]):
                    out[r][c] = value
        if m.c1 > m.c0 and m.r0 == m.r1:
            group_rows.add(m.r0)
    return out, group_rows


def header_row(grid: list[list[object]]) -> int:
    """The first row (within the scan window) with at least two filled cells and at least 60% of the widest
    row's filled count."""
    counts = [sum(not _blank(v) for v in row) for row in grid[:HEADER_SCAN_ROWS]]
    widest = max(counts, default=0)
    for i, n in enumerate(counts):
        if n >= 2 and n >= 0.6 * widest:
            return i
    return 0


def _is_label_row(row: list[object]) -> bool:
    filled = [v for v in row if not _blank(v)]
    return bool(filled) and len(filled) == len(row) and all(isinstance(v, str) for v in filled)


def _column_names(top: list[object], bottom: list[object] | None) -> list[str]:
    """The simple reader's names, made unique the way layout._unique does it (a plain "_2" suffix could collide:
    "A, A, A_2")."""
    names: list[str] = []
    for i, value in enumerate(top):
        name = "" if _blank(value) else str(value).strip()
        if bottom is not None and not _blank(bottom[i]):
            below = str(bottom[i]).strip()
            name = below if not name or below.lower().startswith(name.lower()) else f"{name} {below}"
        names.append(" ".join(name.split()) or f"column_{i + 1}")
    own, used, out = set(names), set(), []
    for n in names:
        candidate, k = n, 1
        while candidate in used or (candidate != n and candidate in own):
            k += 1
            candidate = f"{n}_{k}"
        used.add(candidate)
        out.append(candidate)
    return out


def simple_read(rows: list[list[object]], merges=(), first_row: int = 1) -> tuple[pd.DataFrame, list[int], list[str]]:
    """(frame indexed by source row number, header rows as source row numbers, preamble lines)."""
    if not rows:
        return pd.DataFrame(), [], []
    width = max(len(r) for r in rows)
    grid, group_rows = _filled([list(r) + [None] * (width - len(r)) for r in rows], merges)
    h = header_row(grid)
    preamble = [" ".join(str(v).strip() for v in row if not _blank(v)) for row in grid[:h] if any(not _blank(v) for v in row)]
    bottom = None
    if h in group_rows and h + 1 < len(grid) and _is_label_row(grid[h + 1]):
        bottom = grid[h + 1]
    names = _column_names(grid[h], bottom)
    data_start = h + (2 if bottom is not None else 1)
    data, index = [], []
    for offset, row in enumerate(grid[data_start:]):
        if any(not _blank(v) for v in row):
            data.append(row)
            index.append(first_row + data_start + offset)
    frame = pd.DataFrame(data, columns=names, index=pd.Index(index, name="row"))
    keep = [c for c in frame.columns if not frame[c].map(_blank).all() or not c.startswith("column_")]
    header = [first_row + h] + ([first_row + h + 1] if bottom is not None else [])
    return frame[keep], header, preamble
