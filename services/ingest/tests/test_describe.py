"""Figure descriptions: two passes for charts at temperature 0, values marked approximate, one retry with more
room for a cut reply and one after a backoff for a busy model; embeddings retry a full GPU queue once."""

import asyncio
import io
import json

import httpx
from PIL import Image

from ingest.describe import APPROXIMATE, DATA_HEADING, VisionSettings, describe_image, summarize
from ingest.indexer import Indexer

CFG = VisionSettings(model="m", retry_backoff_s=0.0)


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return buf.getvalue()


def vision(replies: list[tuple[int, str, str]]) -> tuple[httpx.AsyncClient, list[dict]]:
    sent: list[dict] = []

    def handler(request):
        sent.append(json.loads(request.content))
        status, content, finish = replies.pop(0)
        if status != 200:
            return httpx.Response(status, json={"detail": content})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})

    return httpx.AsyncClient(base_url="http://llm", transport=httpx.MockTransport(handler)), sent


def prompt_of(body: dict) -> str:
    return body["messages"][0]["content"][1]["text"]


def test_a_chart_is_described_with_its_extracted_data_at_temperature_0():
    table = "x-axis: 1880 to 2025\n| x | y |\n|---|---|\n| 2024 | 1.28 |\n| 2025 | 1.18 |"
    client, sent = vision([(200, table, "stop"), (200, "A line chart. Highest point: 2024, about 1.28.", "stop")])
    text, cut = asyncio.run(describe_image(client, CFG, png(), "chart.png", "", "", True, 4000))
    assert not cut and [b["temperature"] for b in sent] == [0.0, 0.0]
    assert prompt_of(sent[0]).startswith('This image is from the document "chart.png". If it is not a chart')
    assert "Data points read from this chart in a first, careful pass" in prompt_of(sent[1]) and table in prompt_of(sent[1])
    assert text.startswith(APPROXIMATE) and "Highest point: 2024" in text
    assert text.endswith(f"{DATA_HEADING}\n{table}")


def test_a_non_chart_gets_one_short_extra_call_and_no_data_table():
    client, sent = vision([(200, "NOT_A_CHART", "stop"), (200, "Nutrition Facts. 230 calories.", "stop")])
    text, _ = asyncio.run(describe_image(client, CFG, png(), "label.png", "", "", True, 4000))
    assert "Data points read" not in prompt_of(sent[1]) and DATA_HEADING not in text and "230 calories" in text
    off = VisionSettings(model="m", chart_extraction=False, retry_backoff_s=0.0)
    client, sent = vision([(200, "One call only.", "stop")])
    assert "One call only." in asyncio.run(describe_image(client, off, png(), "x.png", "", "", True, 4000))[0]
    assert len(sent) == 1


def test_a_cut_description_is_retried_with_more_room_and_a_busy_model_after_a_backoff():
    client, sent = vision([(200, "NOT_A_CHART", "stop"), (503, "busy", ""), (200, "Half", "length"),
                           (200, "Whole description.", "stop")])
    text, cut = asyncio.run(describe_image(client, CFG, png(), "x.png", "", "", True, 4000))
    assert not cut and "Whole description." in text
    assert [b["max_tokens"] for b in sent] == [2000, 4000, 4000, 6000]


def test_a_cut_extraction_table_is_marked_and_a_failed_extraction_describes_in_one_pass():
    table = "x-axis: 1880 to 2025\n| x | y |\n|---|---|\n| 2024 | 1.28 |\n| 20"
    client, sent = vision([(200, table, "length"), (200, table, "length"), (200, "A line chart.", "stop")])
    text, _ = asyncio.run(describe_image(client, CFG, png(), "chart.png", "", "", True, 4000))
    assert "| 2024 | 1.28 |" in text and "| 20\n" not in text + "\n" and "the table was cut at its length limit" in text
    client, sent = vision([(500, "boom", ""), (500, "boom", ""), (200, "A plain description.", "stop")])
    text, cut = asyncio.run(describe_image(client, CFG, png(), "x.png", "", "", True, 4000))
    assert "A plain description." in text and DATA_HEADING not in text and not cut


def test_a_summary_still_cut_after_its_retry_says_so():
    client, sent = vision([(200, "A partial", "length"), (200, "A longer partial", "length")])
    text, cut = asyncio.run(summarize(client, CFG, "doc.pdf", "pdf", "material", 600))
    assert cut and text == "A longer partial" and [b["max_tokens"] for b in sent] == [600, 1500]  # 2.5x


def test_embedding_retries_a_full_gpu_queue_once():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={"detail": "queue full"})
        return httpx.Response(200, json={"vectors": [[0.5]], "dim": 1})

    models = httpx.AsyncClient(base_url="http://models", transport=httpx.MockTransport(handler))
    indexer = Indexer(None, models, 32, 2, 128, retry_backoff_s=0.0)
    assert asyncio.run(indexer.embed_texts(["a"])) == [[0.5]] and len(calls) == 2
