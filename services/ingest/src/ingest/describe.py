"""Turn images into searchable text with the vision LLM (OpenAI-compatible chat API).

Charts are described in two passes (measured on the NASA temperature chart: the one-pass description named
the wrong year for the peak in 8 of 8 tries, two passes got it right in 6 of 8 and 3 of 3 at temperature 0):
1. extraction: the axes' tick labels and a table of data points per series (first point, the last 15, the
   3 highest and 3 lowest), or NOT_A_CHART for anything else (one short reply for photos and labels);
2. description: the usual description, with the extracted table given as the values to use.
Both run at temperature 0, so the same image is described the same way each time. The stored text says
that values read from a chart are approximate, and keeps the extracted table under its own heading.

A reply cut at its token limit is asked again once with more room; a busy, failed or timed-out call is
retried once after a backoff (rag_common.retry). Both are logged.

With a JobCache (llmcache.py), every call first looks for the reply in the LLM cache, keyed on the exact request
and the call's ModelCall below, and stores a whole reply afterwards; a cut, failed, empty or refused reply is
never stored.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from PIL import Image

from rag_common.logging import log_fields
from rag_common.retry import retry_once

if TYPE_CHECKING:
    from ingest.llmcache import JobCache

log = logging.getLogger("ingest.describe")

APPROXIMATE = "Automatic description of the image. Values read from a chart are approximate."
DATA_HEADING = "Data read from the chart (approximate):"


@dataclass(frozen=True)
class ModelCall:
    """One kind of model call. name: how the LLM cache knows it, apart from the log wording (purpose); the two
    passes of a chart's description are two calls. version: the prompt version, part of every cache key. The
    request (prompt text, budget, sampling) is in the key too, so editing a prompt already misses; bump the
    version also for a change the request does not show, such as a fix after which earlier replies are wrong."""

    name: str
    version: int
    purpose: str


CHART_EXTRACTION = ModelCall("chart_extraction", 1, "chart data extraction")  # pass 1 of a chart
IMAGE_DESCRIPTION = ModelCall("image_description", 1, "image description")  # pass 2, or the only pass
DOCUMENT_SUMMARY = ModelCall("document_summary", 1, "document summary")

_PROMPT = """You are preparing {what} from the document "{doc}" for a search index.
{context}Write a factual description that lets someone find and use this {what} without seeing it:
- Start with one sentence saying what it shows.
- Transcribe all visible text verbatim, including labels, numbers and units.
- If it contains a table, reproduce it as a Markdown table.
- If it is a chart, give the chart type, the axes with their tick labels, and the series. Read each value
  against the axis: find the two tick labels around the point and interpolate between them. When a chart
  has more than about 20 data points, do not list every point; give the trend and these key values: the
  first point, the last five points one by one (each with its x value), the highest and the lowest point.
  Find the highest and lowest by comparing the heights of the markers; do not assume the latest point is
  the highest.
- Values read from a chart are estimates: write "about 1.28" and say "between A and B" or "unclear" when
  neighbouring points are hard to tell apart.
Do not guess at anything you cannot read. Answer in plain text and Markdown only."""

_EXTRACT = """This image is from the document "{doc}". If it is not a chart or plot, reply exactly NOT_A_CHART.
If it is a chart, read its data points. First write the axis ranges and tick labels. Then, for each series,
write a Markdown table with columns x and y. For a series with more than 30 points, list only: the first
point, every point among the last 15 x values, the 3 highest and the 3 lowest points. Read each y value by
interpolating between the two nearest y tick labels and give 2 decimals. Compare neighbouring markers
carefully: which is higher? Put "~" before a value you are unsure of. Output only the axis line and tables."""

_WITH_DATA = """

Data points read from this chart in a first, careful pass (use these values; they are estimates):
{table}"""

# The format rules are a system message and repeated after the material: placed only after 13,000 characters of
# a paper, they were ignored and the model wrote a 900-1,150-token Markdown report (measured on the Ragas PDF).
_SUMMARY_SYSTEM = """You write short abstracts of documents for a search index. An abstract is one plain paragraph
of 3 to 6 sentences, at most 150 words: what the document is, its purpose, the main topics it covers, and what
data or tables it contains. Then one line that starts with "Topics:" and lists up to 10 keywords. Never write a
title, headings, bullet points, bold text or other Markdown, and nothing else."""

_SUMMARY = """Material from the document "{name}" ({kind}):
<material>
{material}
</material>

Write the abstract now: one plain paragraph of at most 150 words, then the "Topics:" line."""


@dataclass(frozen=True)
class VisionSettings:
    model: str
    temperature: float = 0.0
    max_side: int = 1536
    chart_extraction: bool = True
    extraction_tokens: int = 2000
    retry_max_tokens: int = 8000
    retry_backoff_s: float = 5.0


async def _complete(client: httpx.AsyncClient, cfg: VisionSettings, content, max_tokens: int, call: ModelCall,
                    temperature: float | None = None, system: str = "", retry_factor: float = 1.5,
                    retry_max_tokens: int | None = None, cache: "JobCache | None" = None) -> tuple[str, bool]:
    """(text, cut): one retry with retry_factor x the room for a cut reply (at most retry_max_tokens, by default
    cfg.retry_max_tokens), one after a backoff for a failed call. With a cache, a stored reply for the same
    request is returned without a call (never cut: only whole replies are stored), and the reply is offered to
    the cache afterwards; a call that fails is logged as a skip and raised as before."""
    ceiling = cfg.retry_max_tokens if retry_max_tokens is None else retry_max_tokens
    purpose = call.purpose

    def body(room: int) -> dict:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": room,
            "temperature": cfg.temperature if temperature is None else temperature,
            "top_p": 0.8,
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    key = ""
    if cache is not None:
        key = cache.key(call.name, call.version, body(max_tokens), {"factor": retry_factor, "max_tokens": ceiling})
        if (stored := await cache.get(call.name, key)) is not None:
            return stored, False

    async def once(room: int) -> dict:
        request = body(room)

        async def post():
            response = await client.post("/v1/chat/completions", json=request)
            response.raise_for_status()
            return response.json()["choices"][0]

        return await retry_once(post, what=purpose, backoff_s=cfg.retry_backoff_s, log=log)

    try:
        choice = await once(max_tokens)
        if choice.get("finish_reason") == "length" and max_tokens < ceiling:
            room = min(ceiling, int(max_tokens * retry_factor))
            log_fields(log, f"{purpose}: reply cut at its token limit; retrying once with more room", logging.WARNING,
                       max_tokens=max_tokens, retry_tokens=room)
            choice = await once(room)
        text = (choice["message"].get("content") or "").strip()
    except Exception:
        if cache is not None:
            cache.skip(call.name, "failed", key)
        raise
    finish = choice.get("finish_reason")
    cut = finish == "length"
    if cut:
        log_fields(log, f"{purpose}: reply still cut at its token limit", logging.ERROR)
    if cache is not None:
        await cache.offer(call.name, key, text, finish)
    return text, cut


async def summarize(client: httpx.AsyncClient, cfg: VisionSettings, name: str, kind: str, material: str,
                    max_tokens: int = 1200, retry_max_tokens: int = 3000,
                    cache: "JobCache | None" = None) -> tuple[str, bool]:
    """A short abstract of the whole document, indexed as its own chunk so overview questions ("what is this
    file about?") have something to match. A cut reply is asked again once with 2.5x the room (at most
    retry_max_tokens). Returns (text, cut); a summary still cut is kept, ending at its last full sentence."""
    content = _SUMMARY.format(name=name, kind=kind, material=material)
    text, cut = await _complete(client, cfg, content, max_tokens, DOCUMENT_SUMMARY, temperature=0.2,
                                system=_SUMMARY_SYSTEM, retry_factor=2.5, retry_max_tokens=retry_max_tokens,
                                cache=cache)
    if cut and "." in text:
        text = text[:text.rfind(".") + 1]
    return text, cut


def _data_uri(png: bytes, max_side: int) -> str:
    with Image.open(io.BytesIO(png)) as image:
        image.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def describe_image(
    client: httpx.AsyncClient,
    cfg: VisionSettings,
    png: bytes,
    doc_name: str,
    caption: str,
    section: str,
    standalone: bool,
    max_tokens: int,
    cache: "JobCache | None" = None,
) -> tuple[str, bool]:
    """Return (description, truncated). With chart extraction on, a chart's data table comes first. Each pass is
    looked up in the cache on its own; the second pass's request holds the table the first one read."""
    image = {"type": "image_url", "image_url": {"url": _data_uri(png, cfg.max_side)}}
    context = "".join(f"{label}: {value}\n" for label, value in (("Caption", caption), ("Section", section)) if value)
    what = "this image" if standalone else "a figure"
    prompt = _PROMPT.format(what=what, doc=doc_name, context=context)
    table = ""
    if cfg.chart_extraction:
        try:
            extracted, extraction_cut = await _complete(
                client, cfg, [image, {"type": "text", "text": _EXTRACT.format(doc=doc_name)}], cfg.extraction_tokens,
                CHART_EXTRACTION, cache=cache)
        except Exception as exc:  # noqa: BLE001 - the one-pass description still describes the image
            log_fields(log, "chart data extraction failed after a retry; describing in one pass", logging.WARNING,
                       error=f"{type(exc).__name__}: {exc}"[:300])
            extracted, extraction_cut = "", False
        if extracted and not extracted.startswith("NOT_A_CHART"):
            if extraction_cut:  # a table cut mid-row would be read as the whole series: drop its last line, mark it
                extracted = "\n".join(extracted.splitlines()[:-1]) + "\n(the table was cut at its length limit: values may be missing)"
            table = extracted
            prompt += _WITH_DATA.format(table=table)
    text, cut = await _complete(client, cfg, [image, {"type": "text", "text": prompt}], max_tokens, IMAGE_DESCRIPTION,
                                cache=cache)
    parts = [APPROXIMATE, text] + ([f"{DATA_HEADING}\n{table}"] if table else [])
    return "\n\n".join(p for p in parts if p), cut
