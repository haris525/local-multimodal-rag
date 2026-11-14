"""HTTP access to the running stack: the gateway's API with an API token (documents, chunks, search, chat), and
llm directly for generation and judging. Results are scoped to the token's user: golden sets belong to the user
who owns the documents.

Gateway is the benchmark's client: documents (upload, wait, delete) and stateless chats, with the token in
RAG_TOKEN. It never sends a thread_id, so a benchmark run leaves no chats behind."""

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from rag_common.chat_events import ChatReply, read_chat


@dataclass
class ChatOutcome:
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    route: str = ""
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    error: str | None = None
    timings: dict = field(default_factory=dict)


class Stack:
    def __init__(self):
        token = os.getenv("EVAL_TOKEN", "").strip()
        if not token:
            raise SystemExit("set EVAL_TOKEN to an API token (user menu > API tokens)")
        base = os.getenv("EVAL_URL", "http://gateway:8000/api")
        self.api = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {token}"}, timeout=900)
        self.llm = httpx.Client(base_url=os.getenv("EVAL_LLM_URL", "http://llm:8080"), timeout=900)
        self.model = os.getenv("EVAL_LLM_MODEL", "qwen3.8-27b")

    def documents(self) -> list[dict]:
        return self.api.get("/documents").raise_for_status().json()

    def chunks(self, doc_id: str) -> list[dict]:
        out, offset = [], 0
        while True:
            page = self.api.get(f"/documents/{doc_id}/chunks", params={"offset": offset, "limit": 1000})
            batch = page.raise_for_status().json()
            out += batch
            if len(batch) < 1000:
                return out
            offset += 1000

    def search(self, question: str, doc_ids: list[str], **flags) -> dict:
        body = {"question": question, "doc_ids": doc_ids, **flags}
        return self.api.post("/search", json=body).raise_for_status().json()

    def chat(self, question: str, doc_ids: list[str], mode: str = "single") -> ChatOutcome:
        """One answer. error is set for an HTTP error, an error event, a stream cut before its done event, or
        events out of the protocol's order (rag_common.chat_events), so a partial answer is never scored as
        a complete one."""
        out = ChatOutcome()
        body = {"question": question, "doc_ids": doc_ids, "mode": mode}
        with self.api.stream("POST", "/chat", json=body) as response:
            if response.status_code >= 400:
                out.error = response.read().decode(errors="replace")
                return out
            reply = read_chat(response.iter_lines())
        return ChatOutcome(answer=reply.answer, sources=reply.sources, route=reply.route or "",
                           citations=reply.citations, invalid_citations=reply.invalid_citations, error=reply.failure,
                           timings=reply.timings)

    def llm_json(self, prompt: str, max_tokens: int = 800, temperature: float = 0.0) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        content = self.llm.post("/v1/chat/completions", json=body).raise_for_status().json()["choices"][0]["message"]["content"]
        return json.loads(content)


@dataclass
class Turn:
    """One stateless chat turn as the benchmark records it. reply is None when the request itself failed
    (HTTP error before the stream, or a connection error); error says why the answer is not complete."""

    reply: ChatReply | None
    status: int
    error: str | None
    latency_s: float


DEFAULT_URL = "http://localhost:8000/api"


class Gateway:
    """The gateway's API as the benchmark account: the token from RAG_TOKEN (or EVAL_TOKEN, which the eval container
    gets), the base URL from EVAL_URL (the container's http://gateway:8000/api) or http://localhost:8000/api on the
    host."""

    poll_s = 5.0  # seconds between status checks while documents are processed

    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 900):
        token = (token or os.getenv("RAG_TOKEN") or os.getenv("EVAL_TOKEN") or "").strip()
        if not token:
            raise SystemExit("set RAG_TOKEN to an API token of the benchmark account (user menu > API tokens)")
        base = (url or os.getenv("EVAL_URL") or DEFAULT_URL).rstrip("/")
        self.api = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)

    def documents(self) -> list[dict]:
        return self.api.get("/documents").raise_for_status().json()

    def upload(self, name: str, data: bytes) -> dict:
        """Upload one file; returns its document. Identical bytes from the same user return the existing one."""
        result = self.api.post("/documents", files=[("files", (name, data))]).raise_for_status().json()[0]
        if result.get("error"):
            raise RuntimeError(f"upload of {name} refused: {result['error']}")
        return result["document"]

    def delete(self, doc_id: str) -> None:
        self.api.delete(f"/documents/{doc_id}").raise_for_status()

    def wait_ready(self, doc_ids: list[str], budget_s: float,
                   progress: Callable[[str], None] | None = None) -> dict[str, dict]:
        """Wait until these documents are ready or failed; returns {doc_id: document}. SystemExit when the
        budget runs out."""
        deadline, last = time.monotonic() + budget_s, ""
        want = set(doc_ids)
        while True:
            docs = {d["doc_id"]: d for d in self.documents() if d["doc_id"] in want}
            done = [d for d in docs.values() if d["status"] in ("ready", "failed")]
            note = f"{len(done)}/{len(want)} documents finished"
            if progress and note != last:
                progress(note)
                last = note
            if len(docs) == len(want) and len(done) == len(want):
                return docs
            if time.monotonic() > deadline:
                pending = sorted(d["name"] for d in docs.values() if d not in done)
                missing = sorted(want - set(docs))
                raise SystemExit(f"documents not ready after {budget_s:.0f} s: {pending[:10]} (missing {missing[:5]})")
            time.sleep(self.poll_s)

    def ensure(self, files: dict[str, Callable[[], bytes]], budget_s: float, refresh: bool = False,
               progress: Callable[[str], None] | None = None) -> dict[str, str]:
        """Make every named file a ready document: reuse a ready (or still processing) document of the same name,
        upload the rest one at a time, then wait. Returns {name: doc_id}; SystemExit if any failed."""
        say = progress or (lambda msg: print(msg, file=sys.stderr, flush=True))
        current = {d["name"]: d for d in self.documents() if d["status"] in ("ready", "queued", "processing")}
        ids: dict[str, str] = {}
        for name, read in files.items():
            if name in current and not refresh:
                ids[name] = current[name]["doc_id"]
                continue
            say(f"upload {name}")
            ids[name] = self.upload(name, read())["doc_id"]
        docs = self.wait_ready(list(ids.values()), budget_s, progress=say)
        failed = sorted(f"{d['name']}: {d.get('error') or 'failed'}" for d in docs.values() if d["status"] == "failed")
        if failed:
            raise SystemExit("ingestion failed:\n  " + "\n  ".join(failed))
        return ids

    def ask(self, question: str, doc_ids: list[str], mode: str = "deep", think: bool = False,
            agent: str | None = None, deep_think: bool = False, fanout: bool = False,
            research_calls: int | None = None, python: bool = False) -> Turn:
        """One stateless answer (no thread_id). `agent` selects a deep-search arm the rag registers (langgraph,
        deepagents, deepagents_stock or an optional one, passed through as given) and `deep_think` turns on thinking in deep search's
        plan and verify steps (`think` is the answer's thinking). `fanout` runs researchers on reserved budgets and
        `research_calls` sets deep search's research budget; `python` offers the run_python tool. Each is sent only
        when set; a rag that does not know a field ignores it."""
        body: dict = {"question": question, "doc_ids": doc_ids, "mode": mode, "think": think}
        if agent:
            body["agent"] = agent
        if deep_think:
            body["deep_think"] = True
        if fanout:
            body["fanout"] = True
        if research_calls is not None:
            body["research_calls"] = research_calls
        if python:
            body["python"] = True
        started = time.perf_counter()
        try:
            with self.api.stream("POST", "/chat", json=body) as response:
                if response.status_code >= 400:
                    text = response.read().decode(errors="replace")
                    return Turn(None, response.status_code, f"HTTP {response.status_code}: {text[:300]}",
                                time.perf_counter() - started)
                reply = read_chat(response.iter_lines())
        except httpx.HTTPError as exc:
            return Turn(None, 0, f"{type(exc).__name__}: {exc}"[:300], time.perf_counter() - started)
        return Turn(reply, 200, reply.failure, time.perf_counter() - started)
