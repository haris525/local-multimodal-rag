"""Figure descriptions stop when the vision model keeps failing or takes too long (each failure can cost two
vision timeouts), and the document summary is a short plain paragraph whose cut version is kept and marked."""

import asyncio
import json
from types import SimpleNamespace

import httpx

import ingest.pipeline as pipeline_module
from ingest.config import Settings
from ingest.describe import VisionSettings, summarize
from ingest.elements import Element, ParsedDocument
from ingest.pipeline import Pipeline

CFG = VisionSettings(model="m", retry_backoff_s=0.0)
DOC = SimpleNamespace(doc_id="d" * 32, name="report.pdf", doc_type=SimpleNamespace(value="pdf"))


def pipe(tmp_path, **overrides) -> Pipeline:
    s = Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None, **overrides)
    p = Pipeline(s, None, None, None, SimpleNamespace(count=lambda t: len(t.split())), None)
    p.step = lambda doc, text: None
    return p


def figures(n: int) -> ParsedDocument:
    return ParsedDocument(elements=[Element(kind="image", image=bytes([i]) * 8, page=i + 1) for i in range(n)])


def scripted(outcomes: list[bool], calls: list[int]):
    async def describe_image(client, cfg, png, doc_name, caption, section, standalone, max_tokens, cache=None):
        calls.append(png[0])
        if not outcomes.pop(0):
            raise httpx.ReadTimeout("vision timed out")
        return "A chart.", False
    return describe_image


def test_figure_descriptions_stop_after_three_failures_in_a_row(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(pipeline_module, "describe_image", scripted([False] * 10, calls))
    parsed = figures(10)
    asyncio.run(pipe(tmp_path).describe(DOC, parsed))
    assert len(calls) == 3
    [stopped] = [w for w in parsed.warnings if w.startswith("figure descriptions stopped")]
    assert "(3 figures in a row failed): 7 of 10 figures were not described" in stopped
    assert "INGEST_FIGURE_FAILURE_LIMIT" in stopped
    assert any(w.startswith("3 of 10 figures could not be described") for w in parsed.warnings)


def test_a_success_resets_the_count_of_failures_in_a_row(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(pipeline_module, "describe_image", scripted([False, False, True] * 3, calls))
    parsed = figures(9)
    asyncio.run(pipe(tmp_path).describe(DOC, parsed))
    assert len(calls) == 9 and not any(w.startswith("figure descriptions stopped") for w in parsed.warnings)


def test_figure_descriptions_stop_at_their_deadline(tmp_path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(pipeline_module, "describe_image", scripted([True] * 5, calls))
    clock = iter([0.0, 10.0, 4000.0, 4000.0, 4000.0, 4000.0])
    monkeypatch.setattr(pipeline_module, "_now", lambda: next(clock))
    parsed = figures(5)
    asyncio.run(pipe(tmp_path, figure_deadline_s=3600).describe(DOC, parsed))
    assert len(calls) == 1
    assert any("(describing took longer than 3600 s): 4 of 5 figures" in w for w in parsed.warnings)


def vision(replies: list[tuple[str, str]]):
    sent: list[dict] = []

    def handler(request):
        sent.append(json.loads(request.content))
        content, finish = replies.pop(0)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})

    return httpx.AsyncClient(base_url="http://llm", transport=httpx.MockTransport(handler)), sent


def test_the_summary_rules_are_a_system_message_and_a_cut_summary_is_retried_at_2_5x_up_to_3000():
    client, sent = vision([("One sentence. Two sentences. Three sen", "length"), ("One. Two. Three. Fo", "length")])
    text, cut = asyncio.run(summarize(client, CFG, "ragas.pdf", "pdf", "material", 1200, 3000))
    assert sent[0]["messages"][0]["role"] == "system" and "at most 150 words" in sent[0]["messages"][0]["content"]
    assert "Never write a" in sent[0]["messages"][0]["content"] and "<material>" in sent[0]["messages"][1]["content"]
    assert [b["max_tokens"] for b in sent] == [1200, 3000]
    assert cut and text == "One. Two. Three."  # kept, ending at its last full sentence


def test_a_cut_summary_is_kept_and_marked(tmp_path, monkeypatch):
    async def summarize_cut(*args, cache=None):
        return "The report covers revenue by region.", True

    monkeypatch.setattr(pipeline_module, "summarize", summarize_cut)
    warnings: list[str] = []
    chunk = asyncio.run(pipe(tmp_path).summary_chunk(DOC, [], [], warnings))
    assert chunk.text == "The report covers revenue by region.\n(This summary was cut at its length limit.)"
    assert warnings == ["document summary reached its token limit even after a retry and was kept incomplete "
                        "(INGEST_SUMMARY_TOKENS, INGEST_SUMMARY_RETRY_TOKENS)"]
    assert Settings(data_dir=tmp_path).summary_tokens == 1200 and Settings(data_dir=tmp_path).summary_retry_tokens == 3000
