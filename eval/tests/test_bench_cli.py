"""The benchmark commands end to end against a stub gateway on a real local socket: setup uploads and waits, bench
asks stateless questions (no thread_id; the agent field only when given), writes the reports, and compare reads
two of them. The stub answers correctly for agent=deepagents and wrongly for agent=langgraph on one question."""

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rag_eval import cli
from rag_eval.client import Gateway

TOKEN = "rag_pat_stub"
ENV = {"git_commit": "abc123", "rag_image": "sha256:" + "f" * 64, "model": "qwen3.8-27b", "llm_ctx": "65536",
       "rag_settings": {"RAG_LLM_MODEL": "qwen3.8-27b"}}


class Stub:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.chats: list[dict] = []
        self.auth: set[str] = set()
        self.deleted: list[str] = []

    def answer(self, body: dict) -> tuple[str, list[dict]]:
        docs = [self.docs[d] for d in body["doc_ids"]]
        sources = [{"n": i, "doc_id": d["doc_id"], "doc_name": d["name"], "text": "..."} for i, d in enumerate(docs, 1)]
        q = body["question"]
        if "fastest" in q:
            text = "Idaho grew 1.44% [1]." if body.get("agent") == "langgraph" else "South Carolina grew 1.456% [1]."
        elif "budget" in q:
            text = "The budget is 4.2 million dollars [1]."
        else:  # the public question: the supporting paragraph is the second file
            text = "Yes, both were American [2]."
        return text, sources


def handler_for(stub: Stub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, status: int, payload) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            stub.auth.add(self.headers.get("Authorization", ""))
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                self._json(401, {"detail": "sign in required"})
                return False
            return True

        def do_GET(self):
            if self._authorized() and self.path == "/api/documents":
                listing = list(stub.docs.values())
                self._json(200, listing)
                for d in stub.docs.values():  # processed by the next look
                    d["status"] = "ready"

        def do_DELETE(self):
            if self._authorized():
                doc_id = self.path.rsplit("/", 1)[1]
                stub.docs.pop(doc_id)
                stub.deleted.append(doc_id)
                self.send_response(204)
                self.end_headers()

        def do_POST(self):
            if not self._authorized():
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/api/documents":
                name = re.search(rb'filename="([^"]+)"', body).group(1).decode()
                doc_id = hashlib.sha256(body).hexdigest()[:32]
                doc = {"doc_id": doc_id, "name": name, "status": "processing", "updated_at": "t"}
                stub.docs[doc_id] = doc
                self._json(200, [{"name": name, "document": doc, "error": None}])
            elif self.path == "/api/chat":
                req = json.loads(body)
                stub.chats.append(req)
                text, sources = stub.answer(req)
                events = []
                if req["mode"] == "deep":
                    events += [("step", {"tool": "plan", "args": {}, "sources": None, "error": None}),
                               ("step", {"tool": "search", "args": {"q": "x"}, "sources": [1], "error": None}),
                               ("step", {"tool": "sql", "args": {"q": "bad"}, "sources": [], "error": "Binder Error"})]
                events += [("sources", {"route": req["mode"], "sources": sources}), ("token", {"text": text}),
                           ("done", {"citations": [1], "invalid_citations": [], "finish_reason": "stop",
                                     "usage": {"prompt_tokens": 500, "completion_tokens": 40},
                                     "timings": {"tool_calls": 2} if req["mode"] == "deep" else {}})]
                data = "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    return Handler


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    stub = Stub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(stub))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("RAG_TOKEN", TOKEN)
    monkeypatch.delenv("EVAL_URL", raising=False)
    monkeypatch.setattr(Gateway, "poll_s", 0.01)
    monkeypatch.setattr(cli, "DOCMAP", tmp_path / "bench-docmap.json")
    monkeypatch.setattr(cli.environment, "detect", lambda project, app_commit=None: dict(ENV, app_commit=app_commit))
    stub.url = f"http://127.0.0.1:{server.server_address[1]}/api"
    yield stub
    server.shutdown()


SET = [{"id": "pop", "question": "Which state grew fastest?", "docs": ["pop.xlsx"], "tags": ["table-sql", "numeric"],
        "answers": ["South Carolina"], "numbers": [{"value": 1.46, "tol": 0.01}], "facts": ["South Carolina"]},
       {"id": "budget", "question": "What is the budget?", "docs": ["report.txt", "pop.xlsx"],
        "expected_docs": ["report.txt"], "tags": ["single-document"], "numbers": [{"value": 4.2e6, "rel": 0.01}]}]


def test_setup_bench_compare_and_clean_end_to_end(gateway, tmp_path, capsys):
    docs = tmp_path / "docs"
    (docs / "data").mkdir(parents=True)
    (docs / "data" / "pop.xlsx").write_bytes(b"PK\x03\x04 population")
    (docs / "report.txt").write_text("The budget is 4.2 million dollars.", encoding="utf-8")
    bench_set = tmp_path / "set.jsonl"
    bench_set.write_text("".join(json.dumps(x) + "\n" for x in SET), encoding="utf-8")
    reports = tmp_path / "reports"
    common = ["--set", str(bench_set), "--url", gateway.url]

    cli.main(["bench-setup", *common, "--docs", str(docs), "--samples", str(tmp_path / "none")])
    docmap = json.loads(cli.DOCMAP.read_text(encoding="utf-8"))
    assert sorted(docmap) == ["pop.xlsx", "report.txt"] and set(docmap.values()) == set(gateway.docs)
    cli.main(["bench-setup", *common, "--docs", str(docs)])  # ready documents are reused, not uploaded again
    assert len(gateway.docs) == 2

    for agent in ("deepagents", "langgraph"):
        cli.main(["bench", *common, "--repeats", "2", "--agent", agent, "--reports", str(reports)])
    cli.main(["bench", *common, "--repeats", "1", "--mode", "single", "--variant", "plain", "--reports", str(reports)])
    assert len(gateway.chats) == 2 * 2 * 2 + 2
    assert all("thread_id" not in c for c in gateway.chats)  # stateless: no chat is created
    assert [c.get("agent") for c in gateway.chats] == ["deepagents"] * 4 + ["langgraph"] * 4 + [None] * 2
    assert gateway.auth == {f"Bearer {TOKEN}"}

    a, b, plain = (json.loads(next(reports.glob(f"bench-{v}-*.json")).read_text(encoding="utf-8"))
                   for v in ("deepagents", "langgraph", "plain"))
    assert a["meta"]["agent"] == "deepagents" and a["meta"]["repeats"] == 2 and a["meta"]["mode"] == "deep"
    assert a["summary"]["overall"]["correct"][0] == 1.0 and b["summary"]["overall"]["correct"][0] == 0.5
    run = a["runs"][0]
    assert (run["tool_calls"], run["tool_errors"], run["tools"]) == (2, 1, ["search", "sql"])
    assert run["usage"] == {"prompt_tokens": 500, "completion_tokens": 40} and run["finish_reason"] == "stop"
    assert run["scores"]["citations_ok"] == 1.0 and run["cited_docs"] == ["pop.xlsx"]
    assert plain["meta"]["agent"] is None and plain["runs"][0]["tool_calls"] == 0
    assert sorted(a["meta"]["documents"]) == ["pop.xlsx", "report.txt"]
    assert len((reports / f"bench-deepagents-{a['meta']['stamp']}.runs.jsonl").read_text().splitlines()) == 4

    capsys.readouterr()
    cli.main(["compare", str(next(reports.glob("bench-deepagents-*.json"))),
              str(next(reports.glob("bench-langgraph-*.json"))), "--reports", str(reports)])
    out = capsys.readouterr().out
    assert "A wins 1, loses 0, ties 1" in out and "deepagents (A) vs langgraph (B)" in out
    assert next(reports.glob("compare-deepagents-vs-langgraph-*.md")).read_text(encoding="ascii")

    cli.main(["bench-clean", *common])
    assert len(gateway.docs) == 2 and "dry run" in capsys.readouterr().out
    cli.main(["bench-clean", *common, "--yes"])
    assert gateway.docs == {} and len(gateway.deleted) == 2


def test_bench_refuses_documents_that_are_not_set_up(gateway, tmp_path):
    bench_set = tmp_path / "set.jsonl"
    bench_set.write_text(json.dumps(SET[0]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="pop.xlsx"):
        cli.main(["bench", "--set", str(bench_set), "--url", gateway.url])
    assert gateway.chats == []


def test_a_public_set_ingests_its_paragraphs_then_asks(gateway, tmp_path):
    hotpot = [{"_id": "abc", "question": "Were Scott Derrickson and Ed Wood of the same nationality?", "answer": "yes",
               "type": "comparison", "supporting_facts": [["Ed Wood", 0]],
               "context": [["Scott Derrickson", [" An American director."]], ["Ed Wood", [" An American filmmaker."]]]}]
    data = tmp_path / "hotpot.json"
    data.write_text(json.dumps(hotpot), encoding="utf-8")
    reports = tmp_path / "reports"
    cli.main(["bench", "--set", "hotpotqa", "--data", str(data), "--url", gateway.url, "--repeats", "1",
              "--reports", str(reports)])
    assert sorted(d["name"] for d in gateway.docs.values()) == [
        f"bench-hotpotqa-{hashlib.sha1(b'abc').hexdigest()[:10]}-01.txt",
        f"bench-hotpotqa-{hashlib.sha1(b'abc').hexdigest()[:10]}-02.txt"]
    assert gateway.chats[0]["mode"] == "deep" and len(gateway.chats[0]["doc_ids"]) == 2
    report = json.loads(next(reports.glob("bench-default-*.json")).read_text(encoding="utf-8"))
    scores = report["runs"][0]["scores"]
    assert scores["contains"] == 1.0 and scores["correct"] == 1.0 and scores["citations_ok"] == 1.0
    assert report["meta"]["corpus"] == "per-case" and report["meta"]["limit"] == 30
    cli.main(["bench-clean", "--set", "hotpotqa", "--url", gateway.url, "--yes"])
    assert gateway.docs == {}


def _local_set(tmp_path, gateway) -> str:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "pop.xlsx").write_bytes(b"PK\x03\x04 population")
    (docs / "report.txt").write_text("The budget is 4.2 million dollars.", encoding="utf-8")
    bench_set = tmp_path / "set.jsonl"
    bench_set.write_text("".join(json.dumps(x) + "\n" for x in SET), encoding="utf-8")
    cli.main(["bench-setup", "--set", str(bench_set), "--url", gateway.url, "--docs", str(docs)])
    return str(bench_set)


def test_deep_think_agent_values_repeats_default_and_environment_reach_the_report(gateway, tmp_path, monkeypatch, capsys):
    bench_set = _local_set(tmp_path, gateway)
    monkeypatch.setenv("EVAL_REPEATS", "2")
    reports = tmp_path / "reports"
    cli.main(["bench", "--set", bench_set, "--url", gateway.url, "--agent", "deepagents_stock", "--deep-think",
              "--reports", str(reports), "--app-commit", "feedbee"])
    assert [(c["agent"], c["deep_think"], c["think"]) for c in gateway.chats] == [("deepagents_stock", True, False)] * 4
    report = json.loads(next(reports.glob("bench-deepagents_stock-*.json")).read_text(encoding="utf-8"))
    meta = report["meta"]
    assert meta["deep_think"] is True and meta["repeats"] == 2 and meta["agent"] == "deepagents_stock"
    assert meta["environment"]["rag_image"] == ENV["rag_image"] and meta["environment"]["app_commit"] == "feedbee"
    md = next(reports.glob("bench-deepagents_stock-*.md")).read_text(encoding="utf-8")
    assert "deep think" in md and "model qwen3.8-27b" in md and "LLM_CTX 65536" in md
    err = capsys.readouterr().err
    assert "[4/4] repeat 2/2, question 2/2 budget: PASS" in err and "pass rate 4/4 (100%)" in err
    monkeypatch.setenv("EVAL_REPEATS", "zero")
    with pytest.raises(SystemExit, match="EVAL_REPEATS"):
        cli.main(["bench", "--set", bench_set, "--url", gateway.url])


def test_fanout_and_calls_variants_send_their_fields_and_reach_the_reports(gateway, tmp_path):
    bench_set = _local_set(tmp_path, gateway)
    reports = tmp_path / "reports"
    cli.main(["bench", "--set", bench_set, "--variants", "deepagents+fanout+calls20,deepagents,deepagents+think+calls5",
              "--repeats", "1", "--url", gateway.url, "--reports", str(reports)])
    sent = sorted(((c.get("fanout"), c.get("research_calls"), c.get("deep_think")) for c in gateway.chats
                   if c["question"] == gateway.chats[0]["question"]), key=repr)
    assert sent == sorted([(None, 5, True), (None, None, None), (True, 20, None)], key=repr)  # only fields it sets
    assert all(c["agent"] == "deepagents" and c["think"] is False for c in gateway.chats)
    per_variant = json.loads(next(reports.glob("bench-deepagents-fanout-calls20-*.json")).read_text(encoding="utf-8"))
    assert per_variant["meta"]["fanout"] is True and per_variant["meta"]["research_calls"] == 20
    md = next(reports.glob("bench-deepagents-fanout-calls20-*.md")).read_text(encoding="utf-8")
    assert "fanout: sent; research_calls: 20;" in md
    combined = json.loads(next(reports.glob("bench-combined-*.json")).read_text(encoding="utf-8"))
    assert [(v["label"], v["fanout"], v["research_calls"]) for v in combined["variants"]] == [
        ("deepagents+fanout+calls20", True, 20), ("deepagents", False, None), ("deepagents+think+calls5", False, 5)]
    table = next(reports.glob("bench-combined-*.md")).read_text(encoding="ascii")
    assert "| deep think | fan-out | research calls |" in table
    assert "| deepagents+fanout+calls20 | deepagents | off | on | 20 |" in table
    assert "| deepagents | deepagents | off | off | rag setting |" in table


def test_the_python_variant_sends_python_and_reaches_the_reports(gateway, tmp_path):
    bench_set = _local_set(tmp_path, gateway)
    reports = tmp_path / "reports"
    cli.main(["bench", "--set", bench_set, "--variants", "deepagents+python,deepagents", "--repeats", "1",
              "--url", gateway.url, "--reports", str(reports)])
    assert sorted(repr(c.get("python")) for c in gateway.chats if c["question"] == gateway.chats[0]["question"]) == [
        "None", "True"]  # sent only by the variant that sets it
    per_variant = json.loads(next(reports.glob("bench-deepagents-python-*.json")).read_text(encoding="utf-8"))
    assert per_variant["meta"]["python"] is True
    assert "python: sent;" in next(reports.glob("bench-deepagents-python-*.md")).read_text(encoding="utf-8")
    table = next(reports.glob("bench-combined-*.md")).read_text(encoding="ascii")
    assert "| research calls | python |" in table and "| deepagents+python | deepagents | off | off | rag setting | on |" in table


def test_variants_run_interleaved_with_identical_requests_and_a_combined_verdict(gateway, tmp_path, monkeypatch):
    bench_set = _local_set(tmp_path, gateway)
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    (public_dir / "hotpot_dev_distractor_v1.json").write_text(json.dumps([
        {"_id": "abc", "question": "Were Scott Derrickson and Ed Wood of the same nationality?", "answer": "yes",
         "type": "comparison", "supporting_facts": [["Ed Wood", 0]],
         "context": [["Scott Derrickson", [" An American director."]], ["Ed Wood", [" An American filmmaker."]]]}]),
        encoding="utf-8")
    monkeypatch.setenv("EVAL_PUBLIC_DIR", str(public_dir))
    reports = tmp_path / "reports"
    with pytest.raises(SystemExit, match="drop --think"):
        cli.main(["bench", "--variants", "a,b", "--think", "--url", gateway.url])
    with pytest.raises(SystemExit, match="drop --agent"):
        cli.main(["bench", "--variants", "a,b", "--agent", "a", "--url", gateway.url])
    cli.main(["bench", "--set", f"{bench_set},hotpotqa", "--variants", "deepagents,langgraph,deepagents_stock+think",
              "--repeats", "2", "--url", gateway.url, "--reports", str(reports)])

    chats = gateway.chats
    assert len(chats) == 3 * 2 * 3  # 3 questions (2 local, 1 public) x 2 repeats x 3 variants
    blocks = [chats[k:k + 3] for k in range(0, len(chats), 3)]
    for block in blocks:  # one question at a time, every variant, identical apart from agent and deep_think
        assert len({(c["question"], tuple(c["doc_ids"]), c["mode"], c["think"]) for c in block}) == 1
        assert sorted((c["agent"], c.get("deep_think", False)) for c in block) == [
            ("deepagents", False), ("deepagents_stock", True), ("langgraph", False)]
        assert block[0]["think"] is False and all("thread_id" not in c for c in block)
    assert len({tuple(c["agent"] for c in block) for block in blocks}) > 1  # the variant order is shuffled

    names = sorted(p.name.rsplit("-", 2)[0] for p in reports.glob("bench-*.json"))
    assert names == ["bench-combined", "bench-deepagents", "bench-deepagents_stock-think", "bench-langgraph"]
    combined = json.loads(next(reports.glob("bench-combined-*.json")).read_text(encoding="utf-8"))
    assert set(combined["verdicts"]) == {"all questions", "local set", "held-out public sets"}
    assert combined["verdicts"]["held-out public sets"]["questions"] == 1
    assert combined["meta"]["environment"]["model"] == "qwen3.8-27b" and combined["meta"]["schedule_seed"] == 7
    local = combined["verdicts"]["local set"]
    assert local["means"]["langgraph"] == 0.5 and local["means"]["deepagents"] == 1.0
    per_variant = json.loads(next(reports.glob("bench-langgraph-*.json")).read_text(encoding="utf-8"))
    assert per_variant["meta"]["agent"] == "langgraph" and per_variant["meta"]["deep_think"] is False
    assert all(r["variant"] == "langgraph" for r in per_variant["runs"]) and len(per_variant["runs"]) == 6
    md = next(reports.glob("bench-combined-*.md")).read_text(encoding="ascii")
    assert "**Verdict, local set (2 questions):**" in md and "interleaved" in md and "no LLM judge" in md
    assert "| hotpotqa | 1 |" in md and "| local | 2 |" in md
