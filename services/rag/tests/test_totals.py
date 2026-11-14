"""Total and subtotal rows (rag.totals): names in schema cards, what the sums and nested codes show in describe_table
and profile_column, and the research rule."""

import asyncio
from types import SimpleNamespace

from rag_common.contracts import ColumnSchema, DocStatus, DocType, DocumentInfo, SqlResult, TableSchema
from rag_common.http import ServiceError

from rag import notices
from rag import totals as tt
from rag.agent import _SYSTEM, SourceBook, ToolContext, _tools
from rag.chat import ChatRequest
from rag.config import Settings
from rag.planner import schema_cards

YEAR = ColumnSchema(name="Year", sql_name="year", type="number", profile="min 2000, max 2017", min="2000", max="2017")
STATE = ColumnSchema(name="State", sql_name="state", type="text", profile="12 distinct, e.g. 'Alaska'", distinct=12)
CAUSE = ColumnSchema(name="Cause", sql_name="cause_name", type="text", distinct=3,
                     profile="3 distinct: 'All causes' (20), \"Alzheimer's disease\" (20), 'Cancer' (20)")
DEATHS = ColumnSchema(name="Deaths", sql_name="deaths", type="number", profile="min 1, max 900")
RATE = ColumnSchema(name="Rate", sql_name="age_adjusted_death_rate", type="number", profile="min 1, max 9")
TABLE = TableSchema(table="deaths", source='sheet "deaths"', rows=240, row_filter=True,
                    columns=[YEAR, STATE, CAUSE, RATE, DEATHS])
DOC = DocumentInfo(doc_id="c" * 32, name="deaths.csv", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
                   created_at="t", updated_at="t", tables=[TABLE])

# a country of 4 regions and 12 states: the regions add up to it, and so do the states (each smaller than a region)
REGIONS = [["South", 400], ["West", 300], ["Midwest", 200], ["Northeast", 100]]
STATES = [[f"State {i}", v] for i, v in enumerate([85] * 11 + [65])]
LEVELS = [["Country", 1000], *REGIONS, *STATES]


def rows(pairs, n=1):
    return [[v, n, s] for v, s in pairs]


def test_names_that_say_total_and_the_ones_that_do_not():
    for value in ("Total", "All causes", "Retail sales, total", "Grand total", "Overall", "Subtotal: East",
                  "TOTALS", "all fuels (utility-scale)"):
        assert tt.names_total(value), value
    for value in ("All other general merchandise stores", "Totally Fun Inc.", "Ball bearings", "Small", "Cancer"):
        assert not tt.names_total(value), value
    total = ColumnSchema(name="Total", sql_name="total_renewable", type="number", profile="min 1, max 2")
    other = ColumnSchema(name="All other", sql_name="all_other", type="number", profile="min 1, max 2")
    assert tt.column_says_total(total) and not tt.column_says_total(other)


def test_cards_mark_listed_values_and_number_columns_that_say_total():
    assert tt.listed_values(CAUSE) == ["All causes", "Alzheimer's disease", "Cancer"]
    assert tt.card_note(CAUSE) == "possible totals by name: 'All causes'"
    top_only = ColumnSchema(name="Kind", sql_name="kind", type="text", profile="80 distinct, e.g. 'Food'",
                            top=[("Food", 3), ("Retail sales, total", 3)])
    assert tt.card_note(top_only) == "possible totals by name: 'Retail sales, total'"
    assert tt.card_note(STATE) == "" and tt.card_note(DEATHS) == ""
    card = schema_cards([DOC], budget_tokens=6000)
    assert "- cause_name (text) [Cause]: 3 distinct: 'All causes' (20)" in card and "; possible totals by name: 'All causes'" in card
    column = ColumnSchema(name="Total", sql_name="total", type="number", profile="min 1, max 9")
    wide = DOC.model_copy(update={"tables": [TABLE.model_copy(update={"columns": [YEAR, column]})]})
    assert "- total (number): min 1, max 9; possibly a total of other columns (its name says so)" in schema_cards([wide])


def test_measures_skip_years_dates_codes_and_ids_and_put_rates_last():
    code = ColumnSchema(name="Code", sql_name="naics_code", type="number", profile="x")
    when = ColumnSchema(name="Start", sql_name="start", type="number", profile="x", pattern="dates written as YYYYMMDD numbers")
    period = ColumnSchema(name="Period", sql_name="period", type="number", profile="x", min="1949", max="2026")
    events = ColumnSchema(name="Events", sql_name="number_of_events", type="number", profile="x")
    table = TABLE.model_copy(update={"columns": [YEAR, code, when, period, RATE, DEATHS, STATE, events]})
    assert [c.sql_name for c in tt.measures(table)] == ["deaths", "number_of_events", "age_adjusted_death_rate"]


def test_a_value_that_equals_the_sum_of_the_others_is_a_total():
    pairs = [["United States", 1000], ["Alaska", 400], ["Texas", 350], ["Ohio", 250]]
    assert tt.by_sums(rows(pairs), [DEATHS], 0.005) == "'United States' (its deaths equals the sum of all other values)"
    off = [["United States", 1003], ["Alaska", 400], ["Texas", 350], ["Ohio", 250]]  # 0.3%: rounding
    assert tt.by_sums(rows(off), [DEATHS], 0.005).startswith("'United States'")
    far = [["United States", 1020], ["Alaska", 400], ["Texas", 350], ["Ohio", 250]]  # 2%: not a total
    assert tt.by_sums(rows(far), [DEATHS], 0.005) == ""


def test_levels_of_parts_and_their_subtotals():
    found = tt.by_sums(rows(LEVELS), [DEATHS], 0.005)
    assert found == ("'Country' (the other values add up to 2 times its deaths, so it may total them in 2 levels); "
                     "'South', 'West', 'Midwest', 'Northeast' (the 4 largest after it add up to it: possibly subtotals)")


def test_equal_sums_negative_totals_and_too_few_values_are_not_totals():
    assert tt.by_sums(rows([["A", 3], ["B", 3], ["C", 3]]), [DEATHS], 0.005) == ""  # 3 = (3 + 3) / 2, but equal
    assert tt.by_sums(rows([["A", -6], ["B", -3], ["C", -3]]), [DEATHS], 0.005) == ""
    assert tt.by_sums(rows([["A", 6], ["B", 6]]), [DEATHS], 0.005) == ""
    # the first measure with a match wins: the rate sums say nothing, the deaths sums do
    mixed = [["US", 1, 9.0, 1000], ["A", 1, 8.0, 600], ["B", 1, 7.5, 400]]
    assert tt.by_sums(mixed, [RATE, DEATHS], 0.005) == "'US' (its deaths equals the sum of all other values)"


def test_nested_codes():
    found = tt.by_codes(["441", "4411", "4412", "722", "7225", "2016", "2017", "Food"])
    assert found == "'441' begins '4411', '4412' (a code that begins longer codes may total them)"
    assert tt.by_codes(["2016", "2017", "2018"]) == "" and tt.by_codes(["A1", "A10", "A11"]) == ""


class TotalsSql:
    """The sums query (value, n, SUM per measure), profile_column's statistics and most frequent values; records
    every query."""

    def __init__(self, sums=None, top=None, distinct=12, fail=False):
        self.queries: list[str] = []
        self.sums = sums if sums is not None else [[v, 20, s, 1.0] for v, s in LEVELS[:5]]  # deaths, then the rate
        self.top, self.distinct, self.fail = top or [["Alaska", 20], ["Ohio", 20]], distinct, fail

    async def query(self, doc_id, sql, max_rows=200):
        self.queries.append(sql)
        if "IS NOT NULL GROUP BY 1" in sql:  # the sums query
            if self.fail:
                raise ServiceError(400, "BinderException: no such column")
            return SqlResult(columns=["value", "n", "a", "b"], rows=self.sums, truncated=False)
        if sql.startswith("SELECT COUNT(*)"):
            row = [240, 240, self.distinct, "a", "z", "1-20", 0, 0, 0, 0]
            return SqlResult(columns=[f"c{i}" for i in range(len(row))], rows=[row], truncated=False)
        return SqlResult(columns=["value", "n"], rows=self.top, truncated=False)


def tools(sql, s=None, docs=(DOC,)):
    ctx = ToolContext(SimpleNamespace(s=s or Settings(), sql=sql), ChatRequest(question="q", doc_ids=[d.doc_id for d in docs],
                      mode="deep"), list(docs), SourceBook())
    return {t.name: t.coroutine for t in _tools(ctx)}


def run(coro):
    async def go():
        queue = notices.begin()
        return await coro, queue
    return asyncio.run(go())


def test_describe_table_adds_what_the_sums_show_once():
    sql = TotalsSql()
    t = tools(sql)
    (text, _), queue = run(t["describe_table"](DOC.doc_id, "deaths"))
    assert text.endswith("\n".join([
        "possible totals in column state: 'Country' (its deaths equals the sum of all other values)",
        "possible totals in column cause_name: 'Country' (its deaths equals the sum of all other values)",
        tt.HINT]))
    sums = [q for q in sql.queries if "AS value" in q]
    assert sums[0] == ('SELECT CAST("state" AS VARCHAR) AS value, COUNT(*) AS n, SUM("deaths"), '
                       'SUM("age_adjusted_death_rate") FROM deaths WHERE "state" IS NOT NULL GROUP BY 1 '
                       "ORDER BY n DESC, value LIMIT 501")
    run(t["describe_table"](DOC.doc_id, "deaths"))
    assert len(sql.queries) == 2 and queue == []  # a repeated describe_table runs its queries once


def test_the_sums_check_can_be_turned_off_and_a_failed_query_is_a_line_and_a_notice():
    sql = TotalsSql()
    (text, _), _ = run(tools(sql, Settings(totals_columns=0))["describe_table"](DOC.doc_id, "deaths"))
    assert sql.queries == [] and "possible totals by name: 'All causes'" in text  # the card still marks names
    sql = TotalsSql(fail=True)
    (text, _), queue = run(tools(sql)["describe_table"](DOC.doc_id, "deaths"))
    assert "total rows in column state not checked (the query failed: BinderException: no such column)" in text
    assert [n["code"] for n in queue] == ["totals_check_failed", "totals_check_failed"]


def test_profile_column_marks_fresh_values_that_say_total_and_what_the_sums_show():
    sql = TotalsSql(top=[["All causes", 20], ["Cancer", 20]], distinct=3)
    (text, _), _ = run(tools(sql)["profile_column"]("deaths", "cause_name"))
    assert "possible totals by name: 'All causes'" in text
    assert "possible totals in column cause_name: 'Country'" in text and text.endswith(tt.HINT)
    sql = TotalsSql(distinct=900)  # more distinct values than RAG_TOTALS_LABELS_MAX: no sums query
    (text, _), _ = run(tools(sql)["profile_column"]("deaths", "state"))
    assert not any("IS NOT NULL GROUP BY 1" in q for q in sql.queries) and "possible totals" not in text
    total = ColumnSchema(name="Total", sql_name="total", type="number", profile="min 1, max 9")
    wide = DOC.model_copy(update={"tables": [TABLE.model_copy(update={"columns": [YEAR, total]})]})
    (text, _), _ = run(tools(TotalsSql(), docs=(wide,))["profile_column"]("deaths", "total"))
    assert "possibly a total of other columns (its name says so)" in text


def test_the_research_rules_hold_the_totals_rule():
    assert "- Totals: a table can mix detail rows with rows that total them" in _SYSTEM
    assert "compare its value with the SUM\n  of the rows it would cover" in _SYSTEM
