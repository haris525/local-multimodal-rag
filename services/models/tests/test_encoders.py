"""The query prompt follows the request's instruction (a search task); documents never get one. No model is
loaded: the text embedder is a stand-in that records the prompt it was given."""

import numpy as np
from fastapi.testclient import TestClient

from models_svc import api
from models_svc.config import Settings
from models_svc.encoders import Encoders, query_prompt
from models_svc.gate import GpuGate


class Recorder:
    def __init__(self):
        self.prompts = []

    def encode(self, texts, prompt, batch_size, normalize_embeddings):
        self.prompts.append(prompt)
        return np.ones((len(texts), 2), dtype=np.float32)


def encoders() -> Encoders:
    enc = Encoders.__new__(Encoders)  # no model files: only embed_text is used
    enc.s, enc.text = Settings(), Recorder()
    return enc


def test_query_prompt_uses_the_instruction_and_an_empty_one_means_the_bare_question():
    assert query_prompt("Find clauses") == "Instruct: Find clauses\nQuery:"
    assert query_prompt("") == ""


def test_queries_use_the_requests_instruction_else_the_default_and_documents_none():
    enc = encoders()
    enc.embed_text(["q"], "query")
    enc.embed_text(["q"], "query", "Given a multi-hop question, retrieve documents that can help answer the question")
    enc.embed_text(["q"], "query", "")
    enc.embed_text(["chunk"], "document")
    assert enc.text.prompts == [
        "Instruct: Given a question about the user's documents, retrieve passages with the facts needed to answer it\nQuery:",
        "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery:",
        "",
        "",
    ]


def test_the_embed_route_passes_the_instruction_and_refuses_a_bad_one():
    enc = encoders()
    api.state.update(encoders=enc, gate=GpuGate(4))
    try:
        client = TestClient(api.app)  # no lifespan: the stand-in above is the loaded model
        ok = client.post("/v1/embed/text", json={"texts": ["q"], "role": "query", "instruction": "  Find clauses "})
        assert ok.status_code == 200 and ok.json()["dim"] == 2
        assert client.post("/v1/embed/text", json={"texts": ["q"], "role": "query", "instruction": ""}).status_code == 200
        assert enc.text.prompts == ["Instruct: Find clauses\nQuery:", ""]
        for body in ({"texts": ["q"], "role": "query", "instruction": "two\nlines"},
                     {"texts": ["q"], "role": "query", "instruction": "x" * 301},
                     {"texts": ["chunk"], "role": "document", "instruction": "Find clauses"},
                     {"texts": ["q"], "role": "query", "query_instruction": "Find clauses"}):  # a renamed field
            r = client.post("/v1/embed/text", json=body)
            assert r.status_code == 422 and "Find clauses" not in r.text and "lines" not in r.text  # no echo
        assert len(enc.text.prompts) == 2  # nothing refused reached the model
    finally:
        api.state.clear()


def test_the_embed_route_answers_with_the_instruction_it_applied():
    api.state.update(encoders=encoders(), gate=GpuGate(4))
    try:
        client = TestClient(api.app)

        def applied(body):
            return client.post("/v1/embed/text", json=body).json().get("instruction", "absent")

        assert applied({"texts": ["q"], "role": "query", "instruction": " Find clauses "}) == "Find clauses"
        assert applied({"texts": ["q"], "role": "query", "instruction": ""}) == ""  # none: the bare question
        assert applied({"texts": ["q"], "role": "query"}) == "Given a question about the user's documents, retrieve passages with the facts needed to answer it"
        assert applied({"texts": ["chunk"], "role": "document"}) == "absent"
    finally:
        api.state.clear()
