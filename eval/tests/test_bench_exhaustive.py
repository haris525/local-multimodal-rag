"""Exhaustive mode in the benchmark: --mode exhaustive reaches the chat request and the report, and --set global names
the global set (eval/benchmarks/global_samples.jsonl). Uses the stub gateway of test_bench_cli."""

import json

from rag_eval import cli
from rag_eval.paths import GLOBAL_SET, NAMED_SETS
from test_bench_cli import SET, gateway  # noqa: F401 - the stub gateway fixture


def test_the_global_set_is_a_named_set():
    assert NAMED_SETS["global"] == GLOBAL_SET and GLOBAL_SET.name == "global_samples.jsonl"
    assert "global" in cli.parser().parse_args(["bench", "--set", "global", "--mode", "exhaustive"]).set


def test_bench_sends_mode_exhaustive_and_records_it(gateway, tmp_path):  # noqa: F811 - the fixture
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "pop.xlsx").write_bytes(b"PK\x03\x04 population")
    (docs / "report.txt").write_text("The budget is 4.2 million dollars.", encoding="utf-8")
    bench_set = tmp_path / "set.jsonl"
    bench_set.write_text("".join(json.dumps(x) + "\n" for x in SET), encoding="utf-8")
    reports = tmp_path / "reports"
    common = ["--set", str(bench_set), "--url", gateway.url]
    cli.main(["bench-setup", *common, "--docs", str(docs), "--samples", str(tmp_path / "none")])
    cli.main(["bench", *common, "--repeats", "1", "--mode", "exhaustive", "--variant", "exhaustive",
              "--reports", str(reports)])
    assert [c["mode"] for c in gateway.chats] == ["exhaustive", "exhaustive"]
    assert all("agent" not in c and "deep_think" not in c for c in gateway.chats)  # rag refuses them in this mode
    report = json.loads(next(reports.glob("bench-exhaustive-*.json")).read_text(encoding="utf-8"))
    assert report["meta"]["mode"] == "exhaustive" and report["meta"]["variant"] == "exhaustive"
