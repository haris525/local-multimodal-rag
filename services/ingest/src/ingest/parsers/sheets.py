"""Spreadsheets -> frames indexed by source row number, with their title, notes, units and layout.

parsers/layout.py finds where the table sits (title and preamble rows, a header of one to four rows,
merged cells, units, section labels, footnotes); this module reads the cells and reports formulas whose
cached value is missing.
"""

import csv
import io
import re
from pathlib import Path

import openpyxl
from charset_normalizer import from_bytes

from rag_common.numbers import MINUS, decimal_comma

from ingest.elements import ParsedDocument, Sheet
from ingest.parsers.layout import RAW_CELL_CHARS, Merge, TableLayout, read_tables


def _sheet(name: str, table: TableLayout, blank_formulas: int = 0) -> Sheet:
    return Sheet(
        name=name,
        frame=table.frame,
        preamble="\n".join(table.preamble),
        blank_formulas=blank_formulas,
        title=table.title,
        notes="\n".join(table.notes),
        headers=table.headers,
        units=table.units,
        layout=table.layout,
        raw=table.raw,
    )


def _label(name: str, k: int, taken: set[str]) -> str:
    """The name of a sheet's k-th table: "<sheet> (k)", or "<sheet> (table k)" when the workbook already has a
    sheet of that name (Excel names a copied sheet "Sales (2)"), never a name already in use."""
    if k == 1:
        return name
    for candidate in (f"{name} ({k})", f"{name} (table {k})"):
        if candidate not in taken:
            return candidate
    n = 2
    while f"{name} (table {k}, {n})" in taken:
        n += 1
    return f"{name} (table {k}, {n})"


def _add(doc: ParsedDocument, name: str, tables: list[TableLayout], blank_formulas: int = 0,
         notes: list[tuple[str, str]] | None = None, taken: set[str] | None = None) -> None:
    """One Sheet per table; a second table on the same sheet is named "<sheet> (2)" (or "<sheet> (table 2)" when
    a sheet of the workbook has that name). Rows read as section labels or notes inside the data, columns
    without a header, and extra tables go to `notes` as (sheet, text) pairs (merged across sheets by _merged),
    or straight to the document's warnings."""
    out: list[tuple[str, str]] = []
    kept = [t for t in tables if not t.frame.empty]
    taken = taken if taken is not None else set()
    taken |= {s.name for s in doc.sheets}
    for k, table in enumerate(kept, start=1):
        label = _label(name, k, taken)
        taken.add(label)
        doc.sheets.append(_sheet(label, table, blank_formulas if k == 1 else 0))
        out += [(label, w) for w in table.warnings]
    if len(kept) > 1:
        starts = ", ".join(str(t.layout["data_rows"][0]) for t in kept if t.layout.get("data_rows"))
        out.append((name, f"{len(kept)} tables found (data from rows {starts}), read separately"))
    if notes is None:
        doc.warnings.extend(_merged(out))
    else:
        notes.extend(out)


def _merged(notes: list[tuple[str, str]]) -> list[str]:
    """One warning per text: a workbook of 35 yearly sheets laid out alike gets one line, not 35."""
    by_text: dict[str, list[str]] = {}
    for sheet, text in notes:
        by_text.setdefault(text, []).append(sheet)
    out = []
    for text, sheets in by_text.items():
        if len(sheets) == 1:
            out.append(f'Sheet "{sheets[0]}": {text}')
        else:
            out.append(f'Sheets "{sheets[0]}" to "{sheets[-1]}" ({len(sheets)} sheets): {text}')
    return out


def _number_format(ws):
    """A cell's number format ("0.0%", "General", ...): percent-formatted fractions are named in the profile."""
    def number_format(r: int, c: int) -> str:
        return ws.cell(row=r + 1, column=c + 1).number_format or ""
    return number_format


def _style(ws):
    """A cell's style key (bold, indent): tells a bold or indented section label from a data row whose other
    cells are empty. Fills are left out: banded rows fill every other row whatever it holds."""
    def style(r: int, c: int) -> tuple:
        cell = ws.cell(row=r + 1, column=c + 1)
        indent = int(cell.alignment.indent or 0) if cell.alignment is not None else 0
        return bool(cell.font is not None and cell.font.b), indent
    return style


def read_xlsx(path: Path) -> ParsedDocument:
    values = openpyxl.load_workbook(path, data_only=True)
    formulas = openpyxl.load_workbook(path, data_only=False, read_only=True)
    doc = ParsedDocument()
    notes: list[tuple[str, str]] = []
    taken = {ws.title for ws in values.worksheets}  # labels of extra tables never take a sheet's name
    for ws in values.worksheets:
        if ws.sheet_state != "visible":
            continue
        # Anchored at A1, so grid positions are Excel positions and merged ranges need no offset.
        grid = [[cell.value for cell in row] for row in ws.iter_rows(min_row=1, min_col=1)]
        if not grid:
            continue
        merges = [Merge(m.min_row - 1, m.max_row - 1, m.min_col - 1, m.max_col - 1) for m in ws.merged_cells.ranges]
        blank_formulas = sum(
            1
            for r, row in enumerate(formulas[ws.title].iter_rows(min_row=1, min_col=1, values_only=True))
            for c, v in enumerate(row)
            if isinstance(v, str) and v.startswith("=") and r < len(grid) and c < len(grid[r]) and grid[r][c] is None
        )
        tables = read_tables(grid, merges, first_row=1, style=_style(ws), number_format=_number_format(ws))
        if all(t.frame.empty for t in tables):
            continue
        _add(doc, ws.title, tables, blank_formulas, notes, taken)
        if blank_formulas:
            doc.warnings.append(
                f'Sheet "{ws.title}": {blank_formulas} formula cells have no saved value '
                "(open and save the file in Excel to store results)"
            )
    formulas.close()
    doc.warnings.extend(_merged(notes))
    if not doc.sheets:
        doc.warnings.append("No non-empty sheets found")
    return doc


_SIGN = r"[+\-" + MINUS + r"]?"
# In a file whose fields are separated by ";", a column's decimal mark is decided from its own cells: "1,5",
# "1.234,56" or "1.234.567" mean a decimal comma (dots group thousands); a dot followed by one or two or by four
# or more digits, "0.125" or "1,234.5" mean a decimal point; "1.250" alone could be either.
_COMMA_MARK = re.compile(r"^" + _SIGN + r"(\d{1,3}(\.\d{3})+|\d+),\d+\s*%?$|^" + _SIGN + r"\d{1,3}(\.\d{3}){2,}\s*%?$")
_POINT_MARK = re.compile(r"^" + _SIGN + r"(\d{1,3}(,\d{3})+\.\d+|\d+\.\d{1,2}|\d+\.\d{4,}|0\.\d+)\s*%?$")
_EITHER_MARK = re.compile(r"^" + _SIGN + r"[1-9]\d{0,2}\.\d{3}\s*%?$")
_THOUSANDS_COMMA = re.compile(r"^" + _SIGN + r"\d{1,3}(,\d{3})+$")  # "1,234", "12,500": thousands in a point file


def _decimal_marks(grid: list[list[str | None]]) -> tuple[list[list[str | None]], list[str]]:
    """The cells of a ";" file with decimal commas turned into points, column by column, and warnings. A column
    with decimal commas only is converted; one with decimal points only is kept; one with both is kept and
    named; one whose numbers could be either ("1.250") follows the rest of the file when the rest is clear, and
    is otherwise kept (the dot read as the decimal point) and named."""
    width = max((len(r) for r in grid), default=0)
    stats = []
    for c in range(width):
        vals = [r[c] for r in grid if c < len(r) and r[c]]
        stats.append((sum(bool(_COMMA_MARK.match(v)) for v in vals), sum(bool(_POINT_MARK.match(v)) for v in vals),
                      next((v for v in vals if _EITHER_MARK.match(v)), None),
                      sum(bool(_THOUSANDS_COMMA.match(v)) for v in vals)))
    comma_file = any(cm and not pt for cm, pt, _, _ in stats)
    point_file = any(pt and not cm for cm, pt, _, _ in stats)
    convert: set[int] = set()
    warnings: list[str] = []
    for c, (cm, pt, either, thousands) in enumerate(stats):
        name = next((r[c] for r in grid if c < len(r) and r[c]), "") or ""
        column = f'column {_column_letter(c)} ("{name[:40]}")'
        if cm and not pt and point_file and thousands == cm:  # every comma value is "1,234"-shaped
            warnings.append(f"{column}: the file writes decimals with a point, so commas such as "
                            f"{next(v for r in grid if c < len(r) and (v := r[c]) and _THOUSANDS_COMMA.match(v))} "
                            "were read as thousands separators (check the document)")
        elif cm and not pt and point_file and thousands:
            warnings.append(f"{column} has commas that group thousands and commas that mark decimals in a file that "
                            "writes decimals with a point: its numbers were kept as written")
        elif cm and not pt:
            convert.add(c)
        elif cm and pt:
            warnings.append(f"{column} mixes decimal commas and decimal points: its numbers were kept as written")
        elif either and not pt:
            if comma_file and not point_file:
                convert.add(c)
            elif not point_file:
                warnings.append(f"{column} has numbers such as {either} whose dot may group thousands or mark the "
                                "decimals: read with the dot as the decimal point (check the document)")
    cells = [[decimal_comma(v) if v and c in convert else v for c, v in enumerate(row)] for row in grid]
    return cells, warnings


def _column_letter(c: int) -> str:
    s, c = "", c + 1
    while c:
        c, rem = divmod(c - 1, 26)
        s = chr(65 + rem) + s
    return s


def csv_grid(data: bytes) -> list[list[str | None]]:
    """A CSV's cells to read (see csv_cells)."""
    return csv_cells(data)[0]


def csv_cells(data: bytes) -> tuple[list[list[str | None]], list[list[str | None]], list[str]]:
    """(the cells to read, the cells as written, warnings) of a CSV: encoding and dialect detected, blank cells
    as None. Bytes that are valid UTF-8 are read as UTF-8: on a short file the detector can take one Unicode
    minus for three cp1252 characters. In a ";" file a column's decimal commas become points when its own cells
    (or, for "1.250", the rest of the file) show the comma is the decimal mark; the cells as written stay for
    the raw table."""
    try:
        if b"\x00" in data:  # UTF-16 without a byte order mark is valid UTF-8 too
            raise UnicodeDecodeError("utf-8", data, 0, 1, "NUL bytes")
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        match = from_bytes(data).best()
        text = str(match) if match else data.decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:20_000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    grid = [list(row) for row in csv.reader(io.StringIO(text), dialect)]
    grid = [[None if not v.strip() else v.strip() for v in row] for row in grid]
    if dialect.delimiter != ";":
        return grid, grid, []
    cells, warnings = _decimal_marks(grid)
    return cells, grid, warnings


def _as_written(table: TableLayout, written: list[list[str | None]]) -> None:
    """Give a table's raw cells their text as written in the file (the reading may have changed decimal marks)."""
    start = int(table.layout.get("raw_first_row", 1)) - 1
    c0 = int(table.layout.get("first_col", 0) or 0)
    raw = []
    for i, row in enumerate(table.raw):
        source = written[start + i] if start + i < len(written) else []
        cells = [source[c0 + j] if c0 + j < len(source) else None for j in range(len(row))]
        raw.append([v[:RAW_CELL_CHARS] if v else None for v in cells])
    table.raw = raw


def workbook_grids(path: Path) -> list[tuple[str, list[list[object]], list[Merge]]]:
    """Each visible sheet's (title, cells, merged ranges), as read_xlsx sees them (for comparisons in tests)."""
    values = openpyxl.load_workbook(path, data_only=True)
    out = []
    for ws in values.worksheets:
        if ws.sheet_state != "visible":
            continue
        grid = [[cell.value for cell in row] for row in ws.iter_rows(min_row=1, min_col=1)]
        merges = [Merge(m.min_row - 1, m.max_row - 1, m.min_col - 1, m.max_col - 1) for m in ws.merged_cells.ranges]
        out.append((ws.title, grid, merges))
    return out


def read_csv(data: bytes, name: str) -> ParsedDocument:
    grid, written, marks = csv_cells(data)
    tables = read_tables(grid, [], first_row=1)
    if written is not grid:
        for t in tables:
            _as_written(t, written)
    doc = ParsedDocument()
    doc.warnings += [f'Sheet "{Path(name).stem}": {w}' for w in marks]
    if all(t.frame.empty for t in tables):
        doc.warnings.append("No rows found")
    else:
        _add(doc, Path(name).stem, tables)
    return doc
