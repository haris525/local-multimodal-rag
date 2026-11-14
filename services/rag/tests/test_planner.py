import asyncio

from rag_common.contracts import ColumnSchema, DocStatus, DocType, DocumentInfo, SqlResult, TableSchema

from rag.clients import ServiceError
from rag.planner import Planner

DOC = DocumentInfo(doc_id="d" * 32, name="orders.xlsx", doc_type=DocType.SPREADSHEET, size=1, status=DocStatus.READY,
                   tables=[TableSchema(table="orders", source='sheet "Orders"', sheet="Orders", rows=3, row_filter=True,
                                       columns=[ColumnSchema(name="Region", sql_name="region", type="text", profile="North, South")])],
                   created_at="t", updated_at="t")


class FakeLLM:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return {"content": self.answers.pop(0)}


class FakeSql:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)

    async def query(self, doc_id, sql, max_rows):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def plan(llm: FakeLLM, sql: FakeSql):
    return asyncio.run(Planner(llm, sql, sql_budget=1500).plan("Which North orders were late?", [DOC]))


def test_planner_falls_back_to_search_instead_of_failing_the_turn():
    route = '{"route": "filter", "doc_id": "%s", "table": "orders", "where": "region = \'North\'"}' % DOC.doc_id
    # a timeout (408) is not repaired by the model; the turn goes on as a search
    timed_out = plan(FakeLLM(route), FakeSql(ServiceError(408, "query timed out")))
    # a repaired query that selects text instead of row numbers
    repaired = plan(FakeLLM(route, '{"sql": "SELECT region FROM orders"}'),
                    FakeSql(ServiceError(400, "bad column"), SqlResult(columns=["region"], rows=[["North"]], truncated=False)))
    assert (timed_out.route, timed_out.detail) == ("search", "filter failed")
    assert (repaired.route, repaired.detail) == ("search", "filter failed")
