"""The profile_column research tool, its value structure (rag.value_structure) and the column patterns in schema
cards (describe_table, the planner)."""

import asyncio
from types import SimpleNamespace

import pytest

from rag_common.contracts import ColumnSchema, DocStatus, DocType, DocumentInfo, SqlResult, TableSchema
from rag_common.http import ServiceError

from rag import notices, tokens
from rag import value_structure as vs
from rag.agent import _SYSTEM, SourceBook, ToolContext, _tools, profile_text
from rag.chat import ChatRequest
from rag.config import Settings
from rag.planner import schema_cards

BEGIN = ColumnSchema(name="Begin Date", sql_name="begin_date", type="date", profile="1980-04-10 to 2024-10-09",
                     pattern="a DATE (written as YYYYMMDD numbers in the file): compare with DATE '2024-01-01' or YEAR(col)")
START = ColumnSchema(name="Start", sql_name="start", type="number", profile="min 20231231, max 20240317",
                     pattern="dates written as YYYYMMDD numbers",
                     warning="dates stored as numbers: use STRPTIME(CAST(col AS VARCHAR), '%Y%m%d')")
KIND = ColumnSchema(name="Disaster", sql_name="disaster", type="text", profile="7 distinct")
NOAA = DocumentInfo(doc_id="a" * 32, name="noaa.csv", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
                    created_at="t", updated_at="t",
                    tables=[TableSchema(table="noaa", source='sheet "noaa"', rows=403, row_filter=True,
                                        columns=[BEGIN, START, KIND])])


class ProfileSql:
    def __init__(self, top: list[list] | None = None):
        self.queries: list[str] = []
        self.top = top if top is not None else [["Severe Storm", 203], ["Tropical Cyclone", 67], ["Flooding", 45]]

    async def query(self, doc_id, sql, max_rows=200):
        self.queries.append(sql)
        if sql.startswith("SELECT COUNT(*)"):
            return SqlResult(columns=["n", "filled", "d", "lo", "hi", "spread"],
                             rows=[[403, 400, 7, "Drought", "Wildfire", "6-16"]], truncated=False)
        return SqlResult(columns=["value", "n"], rows=self.top, truncated=False)


def run_tool(sql: ProfileSql, *args, s: Settings | None = None):
    ctx = ToolContext(SimpleNamespace(s=s or Settings(), sql=sql), ChatRequest(question="q", doc_ids=[NOAA.doc_id], mode="deep"),
                      [NOAA], SourceBook())
    profile = next(t for t in _tools(ctx) if t.name == "profile_column").coroutine
    return asyncio.run(profile(*args)), ctx


def test_profile_column_counts_values_and_reports_the_pattern_without_a_doc_id():
    sql = ProfileSql()
    (text, numbers), ctx = run_tool(sql, "noaa", "Disaster")  # the display name; no doc_id: one document has noaa
    assert numbers == [1] and ctx.book.sources[0].kind == "profile"
    assert "rows 403; empty 3 (0.7%)" in text and "distinct values 7" in text and "text length 6-16" in text
    assert "most frequent values (with counts): 'Severe Storm' 203, 'Tropical Cyclone' 67" in text
    assert sql.queries[0].endswith('FROM noaa') and 'COUNT(DISTINCT "disaster")' in sql.queries[0]
    assert 'GROUP BY 1 ORDER BY n DESC, hash(value) LIMIT 20' in sql.queries[1]


def test_profile_column_keeps_the_pattern_and_warning_found_at_ingest():
    sql = ProfileSql([["20240105", 1], ["20240317", 1]])
    (text, _), _ = run_tool(sql, "noaa", "start", NOAA.doc_id)
    assert "pattern: dates written as YYYYMMDD numbers" in text and "warning: dates stored as numbers" in text
    assert 'quantile_cont("start", [0.25, 0.5, 0.75])' in sql.queries[0]  # a number column gets quartiles


def test_profile_column_detects_a_pattern_in_fresh_values_when_ingest_had_none():
    col = ColumnSchema(name="Postal", sql_name="postal", type="text", profile="3 distinct")
    doc = NOAA.model_copy(update={"tables": [NOAA.tables[0].model_copy(update={"columns": [col]})]})
    ctx = ToolContext(SimpleNamespace(s=Settings(), sql=ProfileSql([["02134", 1], ["02139", 1], ["10001", 1]])),
                      ChatRequest(question="q", doc_ids=[doc.doc_id], mode="deep"), [doc], SourceBook())
    profile = next(t for t in _tools(ctx) if t.name == "profile_column").coroutine
    text, _ = asyncio.run(profile("noaa", "postal"))
    assert "pattern: codes with leading zeros" in text


def test_an_empty_column_name_matches_no_column():
    sql = ProfileSql()
    with pytest.raises(ValueError, match="has no column ''"):
        run_tool(sql, "noaa", "")
    assert sql.queries == []


def test_an_unknown_column_lists_the_real_ones_and_runs_no_sql():
    sql = ProfileSql()
    with pytest.raises(ValueError, match="has no column 'state'; its columns are: begin_date, start, disaster"):
        run_tool(sql, "noaa", "state")
    assert sql.queries == []


def test_the_profile_is_capped_in_tokens_and_identical_calls_run_once():
    long_values = [[f"value number {i} " + "x" * 80, 500 - i] for i in range(20)]
    sql = ProfileSql(long_values)
    s = Settings(profile_tokens=120)
    (text, _), ctx = run_tool(sql, "noaa", "disaster", s=s)
    assert tokens.count_tokens(text) <= 120 + tokens.count_tokens("[1] noaa.csv | profile of column disaster in table noaa\n") + 2
    profile = next(t for t in _tools(ctx) if t.name == "profile_column").coroutine
    again, _ = asyncio.run(profile("noaa", "disaster"))
    assert "already made this exact call" in again and len(sql.queries) == 2


def test_the_research_rules_point_to_profile_column():
    assert "profile_column: one column's counts" in _SYSTEM and "before filtering or grouping on it" in _SYSTEM


def test_cards_show_the_pattern_and_warning_and_fit_a_token_budget():
    card = schema_cards([NOAA], "when did they begin", budget_tokens=6000)
    assert "- begin_date (date): 1980-04-10 to 2024-10-09; pattern: a DATE (written as YYYYMMDD numbers in the file): " \
           "compare with DATE '2024-01-01'" in card
    assert "- start (number): min 20231231, max 20240317; pattern: dates written as YYYYMMDD numbers WARNING: dates " \
           "stored as numbers: use STRPTIME" in card
    many = [NOAA.model_copy(update={"doc_id": f"{i:032x}", "name": f"t{i}.csv"}) for i in range(60)]
    small = schema_cards(many, "", budget_tokens=1500)
    assert tokens.count_tokens(small) <= 1500 + 40 and "more tables (ask for them by name)" in small


# --- value structure: the last part of text values that mostly contain a separator ------------------------------

PLACE = ColumnSchema(name="Place", sql_name="place", type="text", profile="600 distinct")
MAG = ColumnSchema(name="Mag", sql_name="mag", type="number", profile="min 2, max 7")
QUAKES = DocumentInfo(doc_id="b" * 32, name="quakes.csv", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
                      created_at="t", updated_at="t",
                      tables=[TableSchema(table="quakes", source='sheet "quakes"', rows=1000, row_filter=True,
                                          columns=[PLACE, MAG])])
STATS = [1000, 1000, 600, "1 km E of A, Eastvale", "9 km W of Z, Westmark", "10-40"]
TOP = [[f"{i} km N of Town {i}, Westmark", 3] for i in range(20)]
PARTS = [["Westmark", 500, 6], ["Eastvale", 300, 6], ["Northport", 150, 6], ["EV", 40, 6], ["Southdale", 9, 6], ["NP", 1, 6]]


class PartsSql:
    """profile_column's statistics (with the separator counts after them), its most frequent values and the parts
    query; records every query and its row cap."""

    def __init__(self, seps=(980, 0, 0, 0), parts=PARTS, distinct=600, fail=False):
        self.queries: list[str] = []
        self.max_rows: list[int] = []
        self.seps, self.parts, self.distinct, self.fail = list(seps), parts, distinct, fail

    async def query(self, doc_id, sql, max_rows=200):
        self.queries.append(sql)
        self.max_rows.append(max_rows)
        if sql.startswith("SELECT COUNT(*)"):
            row = [*STATS[:2], self.distinct, *STATS[3:], *self.seps]
            return SqlResult(columns=[f"c{i}" for i in range(len(row))], rows=[row], truncated=False)
        if sql.startswith("WITH p AS"):
            if self.fail:
                raise ServiceError(400, "BinderException: no such function")
            return SqlResult(columns=["value", "n", "parts"], rows=self.parts, truncated=False)
        return SqlResult(columns=["value", "n"], rows=TOP, truncated=False)


def run_profile(sql, column: str = "place", s: Settings | None = None) -> tuple[str, list[dict]]:
    """profile_column on quakes.csv: (the profile without its source line, the notices)."""
    ctx = ToolContext(SimpleNamespace(s=s or Settings(), sql=sql),
                      ChatRequest(question="q", doc_ids=[QUAKES.doc_id], mode="deep"), [QUAKES], SourceBook())
    profile = next(t for t in _tools(ctx) if t.name == "profile_column").coroutine

    async def go():
        queue = notices.begin()
        text, _ = await profile("quakes", column)
        return text.split("\n", 1)[1], queue
    return asyncio.run(go())


def test_a_column_whose_values_mostly_hold_a_comma_lists_every_last_part_and_flags_the_rare_code():
    sql = PartsSql()
    text, queue = run_profile(sql)
    assert "COUNT(*) FILTER (WHERE contains(CAST(\"place\" AS VARCHAR), ','))" in sql.queries[0]
    assert len(sql.queries) == 3 and "trim(string_split(CAST(\"place\" AS VARCHAR), ',')[-1])" in sql.queries[2]
    assert "FROM quakes WHERE \"place\" IS NOT NULL" in sql.queries[2] and "LIMIT 200" in sql.queries[2]
    assert sql.max_rows[2] == 200  # RAG_PROFILE_PARTS_MAX and the code candidates past it
    base = profile_text(QUAKES.tables[0], PLACE, STATS, TOP, Settings())
    assert text == base + "\n" + "\n".join([  # what the profile gave before, unchanged, then the summary
        "last part after ',' (trimmed; 98% of 1000 values contain it, a value without it counts whole): 6 distinct, "
        "most rows first:",
        "'Westmark' 500, 'Eastvale' 300, 'Northport' 150, 'EV' 40, 'Southdale' 9, 'NP' 1",
        "possible abbreviations or codes: 'EV' 40, 'NP' 1"])  # the code in one row is kept and flagged
    assert queue == []


def test_the_separator_is_the_first_that_more_than_half_of_the_values_contain_comma_first():
    assert vs.pick([501, 900, 0, 0], 1000) == (",", 501)
    assert vs.pick([500, 900, 0, 0], 1000) == (";", 900)  # exactly half is not most
    assert vs.pick([0, 0, 0, 700], 1000) == (" - ", 700)
    assert vs.pick([0, 0, 0, 0], 1000) is None and vs.pick([5, 5, 5, 5], 0) is None and vs.pick([], 10) is None
    assert vs.pick(["South", None, "x", 1], 4) is None  # cells that are not counts count as none
    sql = PartsSql(seps=(100, 0, 1000, 0))
    text, _ = run_profile(sql)
    assert "trim(string_split(CAST(\"place\" AS VARCHAR), '|')[-1])" in sql.queries[2]
    assert "last part after '|' (trimmed; 100% of 1000 values contain it): 6 distinct" in text


@pytest.mark.parametrize("part", ["EV", "NP", "W.M.", "A.B.", "NNE", "N", "B12", "Q3", "SW1A", "A1B2C3"])
def test_short_upper_case_letters_or_letters_with_digits_look_like_codes(part):
    assert vs.looks_like_code(part)


@pytest.mark.parametrize("part", ["Ev", "Westmark", "km2", "12345", "", "St.", "Mt", "ABCD", "New York", "A-1",
                                  "ABCDEFG1", "Zz9"])
def test_words_numbers_and_longer_runs_do_not(part):
    assert not vs.looks_like_code(part)


def test_while_the_list_fits_every_part_is_listed_even_those_in_one_row():
    parts = [[f"Place {i}", 1, 150] for i in range(149)] + [["NV", 1, 150]]
    text, _ = run_profile(PartsSql(parts=parts))
    lines = text.splitlines()
    assert lines[-3].endswith(": 150 distinct, most rows first:")
    assert lines[-2] == ", ".join(f"'Place {i}' 1" for i in range(149)) + ", 'NV' 1"
    assert lines[-1] == "possible abbreviations or codes: 'NV' 1" and "more distinct" not in text


def test_past_the_cap_the_most_frequent_parts_are_listed_with_how_many_more_and_codes_from_every_part():
    ranked = [[f"Region {i:03d}", 1000 - i, 400] for i in range(150)]
    sql = PartsSql(parts=ranked + [["NV", 1, 400], ["Zz9", 1, 400]])  # code candidates past the cap; one is a code
    text, _ = run_profile(sql)
    lines = text.splitlines()
    assert lines[-4].endswith(": 400 distinct, most rows first:")
    assert lines[-3] == ", ".join(f"'Region {i:03d}' {1000 - i}" for i in range(vs.PARTS_TOP))
    assert lines[-2] == "... 370 more distinct last parts (over RAG_PROFILE_PARTS_MAX (150); SQL can count them)"
    assert lines[-1] == "possible abbreviations or codes: 'NV' 1"
    assert "QUALIFY row_number() OVER (ORDER BY n DESC, part) <= 150 OR (length(part) <= 12" in sql.queries[2]
    sql = PartsSql(parts=[[f"Region {i}", 9, 40] for i in range(40)])
    text, _ = run_profile(sql, s=Settings(profile_parts_max=30))  # the cap is a setting
    assert "LIMIT 80" in sql.queries[2] and "... 10 more distinct last parts (over RAG_PROFILE_PARTS_MAX (30)" in text


def test_columns_without_a_separator_in_most_values_or_of_other_types_or_listed_in_full_are_unchanged():
    sql = PartsSql(seps=(499, 100, 0, 0))
    text, _ = run_profile(sql)
    assert len(sql.queries) == 2 and text == profile_text(QUAKES.tables[0], PLACE, STATS, TOP, Settings())
    sql = PartsSql(distinct=20)  # the 20 most frequent values are every value: they show the forms already
    text, _ = run_profile(sql, s=Settings(totals_columns=0))  # (with the totals check, its query is the third)
    assert len(sql.queries) == 2 and "last part" not in text
    sql = PartsSql()
    run_profile(sql, "mag")
    assert len(sql.queries) == 2 and "contains(" not in sql.queries[0]  # a number column is not counted


def test_the_summary_fits_its_token_cap_dropping_the_rarest_parts_first_and_keeping_the_codes():
    s = Settings(profile_parts_tokens=300)
    parts = [[f"region name number {i:03d}", 500 - i, 150] for i in range(149)] + [["NP", 1, 150]]
    text, _ = run_profile(PartsSql(parts=parts), s=s)
    structure = text[text.index("last part after"):]
    assert tokens.count_tokens(structure) <= 300 and structure.endswith("possible abbreviations or codes: 'NP' 1")
    assert "'region name number 000' 500" in structure and "number 148" not in structure
    assert "more distinct last parts (cut to fit RAG_PROFILE_PARTS_TOKENS)" in structure
    assert tokens.count_tokens(text) <= s.profile_tokens + s.profile_parts_tokens + 1
    long = [[f"{'x' * 90} {i}", 9, 150] for i in range(150)]  # parts are shown cut at RAG_PEEK_CELL_CHARS
    assert repr("x" * 40) + " 9" in vs.parts_lines(",", 9, 9, long, Settings(), 2000)[1]
    for budget in (5, 40, 120, 600):
        assert tokens.count_tokens("\n".join(vs.parts_lines(",", 1000, 1000, parts, s, budget))) <= budget


def test_a_failed_parts_query_keeps_the_profile_says_so_and_is_a_notice():
    text, queue = run_profile(PartsSql(fail=True))
    assert text.startswith(profile_text(QUAKES.tables[0], PLACE, STATS, TOP, Settings()))
    assert text.splitlines()[-1] == ("value structure of place not available (the query failed: BinderException: "
                                     "no such function)")
    assert [n["code"] for n in queue] == ["profile_parts_failed"]
