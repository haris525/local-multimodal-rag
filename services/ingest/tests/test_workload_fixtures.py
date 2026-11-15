"""Verify cell preservation on the same messy layouts used by the live memory workload."""

import sys
from pathlib import Path

import duckdb

from ingest.config import Settings
from ingest.parsers.sheets import read_csv, read_xlsx
from ingest.pipeline import Pipeline
from ingest.tables import write_tables

sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts"))
from validate_workload import fixtures, healthy_service  # noqa: E402


def test_workload_monitor_detects_child_ooms_even_when_the_parent_survives():
    baseline = {"health": "healthy", "restarts": 0, "oom_killed": True, "child_oom_kills": 1}
    assert healthy_service(baseline, dict(baseline))  # historical sandbox containment check
    assert not healthy_service(baseline, baseline | {"child_oom_kills": 2})
    assert not healthy_service(baseline, baseline | {"restarts": 1})
    assert not healthy_service(baseline, baseline | {"health": "unhealthy"})
    assert not healthy_service(baseline, baseline | {"child_oom_kills": None})


def test_live_workload_layouts_preserve_identifiers_numbers_unicode_dates_and_nulls(tmp_path):
    count = 5000  # includes a repeated workbook header and dozens of multiline/missing cells
    fixtures(tmp_path, count, 3)
    pipeline = Pipeline(Settings(data_dir=tmp_path, audit_url=None), None, None, None, None, None)
    csv = read_csv((tmp_path / "messy-unicode.csv").read_bytes(), "messy-unicode.csv")
    path = tmp_path / "csv.duckdb"
    schema, = write_tables(path, pipeline.table_sources(csv))
    with duckdb.connect(str(path), read_only=True) as con:
        result = con.sql(f'SELECT count(*), min(id), max(id), sum(CAST(amount*4 AS BIGINT)), count(note) '
                         f'FROM "{schema.table}"').fetchall()
        assert result == [(count, "00000001", f"{count:08}", sum(4 * (i % 900) + 1 for i in range(1, count + 1)), count // 101)]
        assert con.sql(f'SELECT region,note FROM "{schema.table}" WHERE id=\'00000101\'').fetchall() == [
            ("München", "quoted; value\nsecond line")]

    workbook = read_xlsx(tmp_path / "messy-workbook.xlsx")
    assert any("formula cells have no saved value" in warning for warning in workbook.warnings)
    path = tmp_path / "workbook.duckdb"
    schema, = write_tables(path, pipeline.table_sources(workbook))
    with duckdb.connect(str(path), read_only=True) as con:
        result = con.sql(f'SELECT count(*),min(order_id),max(order_id),sum(units),count(*) FILTER (WHERE amount IS NULL),'
                         f'count(*) FILTER (WHERE date IS NULL) FROM "{schema.table}" '
                         "WHERE regexp_matches(order_id,'^[0-9]{8}$')").fetchall()
        assert result == [(count, "00000001", f"{count:08}", sum(i % 20 for i in range(1, count + 1)), count // 113, 0)]
        assert con.sql(f'SELECT notes FROM "{schema.table}" WHERE order_id=\'00000097\'').fetchall() == [
            ("line one\nline two; naïve café",)]
