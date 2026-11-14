import duckdb
import pandas as pd

from ingest.tables import TableSource, coerce_column, sql_identifier, write_tables


def test_coerce_column_parses_currency_thousands_percent_and_negatives():
    values = coerce_column(pd.Series(["$1,250.50", "(30)", "12%", None]))
    assert values.tolist()[:3] == [1250.5, -30.0, 12.0] and pd.isna(values.iloc[3])


def test_exponents_are_numbers_only_beside_other_numbers():
    codes = coerce_column(pd.Series(["1E5", "2E10", "3E7"]), "Part")
    assert str(codes.dtype) == "string" and list(codes) == ["1E5", "2E10", "3E7"]
    p = coerce_column(pd.Series(["1.2e-05", "0.03", "3.4E-08", "2e-3"]), "p_value")
    assert str(p.dtype) == "Float64" and p.iloc[2] == 3.4e-08


def test_sql_identifier_is_safe_unique_and_avoids_reserved_words():
    taken: set[str] = set()
    assert [sql_identifier(n, taken) for n in ["Order ID", "order id", "select", "2025 Sales"]] == [
        "order_id", "order_id_2", "select_", "c_2025_sales"]


def test_write_tables_stores_row_numbers_and_profiles(tmp_path):
    frame = pd.DataFrame({"Region": ["North", "South"], "Amount": ["$10", "$32.5"]}, index=pd.Index([4, 5]))
    [schema] = write_tables(tmp_path / "t.duckdb", [TableSource("Sales", 'sheet "Sales"', "Sales", frame, True)])
    assert schema.table == "sales" and schema.row_filter
    assert [(c.sql_name, c.type) for c in schema.columns] == [("region", "text"), ("amount", "number")]
    assert schema.columns[0].profile == "2 distinct: 'North' (1), 'South' (1)"  # every value, with its count
    with duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True) as con:
        assert con.execute("SELECT _row, amount FROM sales ORDER BY _row").fetchall() == [(4, 10.0), (5, 32.5)]


def test_coerce_integers_dates_thousands_identifiers_and_missing_markers():
    from datetime import date

    assert str(coerce_column(pd.Series([1, 2, None])).dtype) == "Int64"  # whole numbers are BIGINT
    assert coerce_column(pd.Series(["4411,4412", "441", "4412"])).tolist() == ["4411,4412", "441", "4412"]  # not 44114412
    assert coerce_column(pd.Series(["1,234,567", "12"])).tolist() == [1234567, 12]
    assert coerce_column(pd.Series(["441", "442"]), "NAICS Code").tolist() == ["441", "442"]  # identifiers stay text
    days = coerce_column(pd.Series([19800410, 20241009]), "Begin Date")
    assert days.tolist() == [date(1980, 4, 10), date(2024, 10, 9)]
    assert coerce_column(pd.Series([19800410, 20241009]), "Amount").tolist() == [19800410, 20241009]
    marked = coerce_column(pd.Series(["Not Available", "(S)", "12.5", "3", "--"]))
    assert marked.tolist()[2:4] == [12.5, 3.0] and marked.isna().tolist() == [True, True, False, False, True]


def test_write_tables_keeps_header_unit_missing_count_and_raw_table(tmp_path):
    frame = pd.DataFrame({"Solar": ["Not Available", 5.2, 7.5]}, index=pd.Index([13, 14, 15]))
    src = TableSource("Annual", 'sheet "Annual"', "Annual", frame, True, title="Table 7.2a", notes="note",
                      headers={"Solar": "Electricity Net Generation From Solar, All Sectors"},
                      units={"Solar": "Million Kilowatthours"},
                      layout={"header_rows": [11, 12], "data_rows": [13, 15]}, raw=[["Year", "Solar"], ["2024", "5.2"]])
    [schema] = write_tables(tmp_path / "t.duckdb", [src])
    [col] = schema.columns
    assert (col.header, col.unit, col.missing, col.type) == (src.headers["Solar"], "Million Kilowatthours", 1, "number")
    assert "1 empty (1 marked not available)" in col.profile
    assert (schema.title, schema.notes, schema.layout.header_rows, schema.layout.raw_table) == (
        "Table 7.2a", "note", [11, 12], "annual__raw")
    with duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True) as con:
        assert con.execute("SELECT _row, c1, c2 FROM annual__raw ORDER BY _row").fetchall() == [
            (1, "Year", "Solar"), (2, "2024", "5.2")]
