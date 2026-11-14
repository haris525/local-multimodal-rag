"""Run model-written SQL safely: one SELECT, read-only file, no file or network access, capped
memory, threads, time, rows, cell length and response size."""

import os
import threading
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import duckdb

from rag_common.contracts import SqlResult

LIMITS = {
    "enable_external_access": False,  # no read_csv('/etc/...'), no httpfs, no extension installs
    "threads": 2,
    "lock_configuration": True,  # the query cannot SET its way out of the limits above
}
# A long text cell is cut; the model never sees more than its prompt budget anyway (SQL_MAX_CELL_CHARS).
MAX_CELL_CHARS = int(os.getenv("SQL_MAX_CELL_CHARS", "2000"))
# Rows past this many characters are dropped and the result is marked truncated (SQL_MAX_RESPONSE_CHARS).
MAX_RESPONSE_CHARS = int(os.getenv("SQL_MAX_RESPONSE_CHARS", "200000"))


class QueryTimeout(Exception):
    pass


def check_select(sql: str) -> None:
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError("only a single SELECT statement is allowed")


def _json(value: object) -> object:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):  # 12 significant digits hides binary noise from SUM/AVG
        return int(value) if value.is_integer() and abs(value) < 1e15 else float(f"{value:.12g}")
    if isinstance(value, Decimal):
        return _json(float(value))
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time() else value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _cell(value: object) -> object:
    value = _json(value)
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[: MAX_CELL_CHARS - 3] + "..."
    return value


def run(db_path: Path, sql: str, max_rows: int, timeout_s: float, memory_limit: str = "400MB") -> SqlResult:
    """`memory_limit` is per query: concurrent queries times this must fit the container's memory."""
    check_select(sql)
    with duckdb.connect(str(db_path), read_only=True, config={**LIMITS, "memory_limit": memory_limit}) as con:
        timer = threading.Timer(timeout_s, con.interrupt)
        timer.start()
        try:
            cursor = con.execute(sql)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchmany(max_rows + 1)
        except duckdb.InterruptException as exc:
            raise QueryTimeout(f"the query took longer than {timeout_s:g} s") from exc
        finally:
            timer.cancel()
    out: list[list[object]] = []
    size, truncated = 0, len(rows) > max_rows
    for row in rows[:max_rows]:
        cells = [_cell(v) for v in row]
        size += sum(len(str(c)) for c in cells)
        if size > MAX_RESPONSE_CHARS and out:
            truncated = True
            break
        out.append(cells)
    return SqlResult(columns=columns, rows=out, truncated=truncated)
