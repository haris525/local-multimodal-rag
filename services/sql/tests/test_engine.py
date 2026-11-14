import duckdb
import pytest

from sql_sandbox.engine import MAX_CELL_CHARS, QueryTimeout, run


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE sales AS SELECT * FROM (VALUES (1, 'North', 0.1), (2, 'North', 0.2), (3, 'South', 5.0)) t(_row, region, amount)")
    return path


def test_select_runs_with_row_cap_and_clean_floats(db):
    result = run(db, "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY region", 1, 5)
    assert result.columns == ["region", "total"]
    assert result.rows == [["North", 0.3]] and result.truncated  # 0.1 + 0.2 without binary noise


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "SELECT 1; SELECT 2",
    "COPY sales TO '/tmp/out.csv'",
])
def test_only_a_single_select_is_accepted(db, sql):
    with pytest.raises(ValueError):
        run(db, sql, 10, 5)


def test_file_access_from_sql_is_blocked(db, tmp_path):
    secret = tmp_path / "secret.csv"  # a file that really exists, so only the sandbox can stop the read
    secret.write_text("password\nhunter2\n")
    with pytest.raises(duckdb.PermissionException):
        run(db, f"SELECT * FROM read_csv('{secret.as_posix()}')", 10, 5)


def test_slow_queries_time_out_and_large_results_are_capped(db):
    with pytest.raises(QueryTimeout):
        run(db, "SELECT count(*) FROM range(100000) a, range(100000) b, range(100) c", 10, 0.3)
    assert len(run(db, "SELECT repeat('x', 5000) AS long", 10, 5).rows[0][0]) == MAX_CELL_CHARS
    big = run(db, "SELECT repeat('y', 1000) FROM range(500)", 500, 5)
    assert big.truncated and len(big.rows) < 500
