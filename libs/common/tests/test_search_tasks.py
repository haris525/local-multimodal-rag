"""Search task text: one check for the instruction in rag, threads and models, and the custom task contract."""

import json
import re

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_common.contracts import (
    MAX_INSTRUCTION,
    SEARCH_TASK,
    EmbedTextRequest,
    SearchTaskWrite,
    instruction_text,
    one_line,
)
from rag_common.service import create_app


def test_an_instruction_is_trimmed_and_put_in_nfc():
    assert instruction_text("  Given a contract, retrieve its clauses  ") == "Given a contract, retrieve its clauses"
    assert instruction_text("Cafe\u0301 menus") == "Caf\u00e9 menus"  # e + combining accent -> one character
    assert instruction_text("   ") == ""  # empty: no instruction, the bare question


@pytest.mark.parametrize("bad", ["two\nlines", "tab\there", "nul\x00", "bell\x07", "line\u2028separator",
                                 "para\u2029separator", "lone \ud800 surrogate", "del\x7f"])
def test_an_instruction_must_be_one_line_without_control_characters(bad):
    with pytest.raises(ValueError, match="single line"):
        instruction_text(bad)


def test_an_instruction_has_at_most_300_characters_after_trimming():
    assert len(instruction_text(" " + "x" * MAX_INSTRUCTION + " ")) == MAX_INSTRUCTION
    with pytest.raises(ValueError, match="at most 300 characters, not 301"):
        instruction_text("x" * (MAX_INSTRUCTION + 1))
    with pytest.raises(ValueError, match="the name must be at most 40"):
        one_line("n" * 41, "name", 40)


def test_embed_requests_take_an_instruction_for_queries_only():
    assert EmbedTextRequest(texts=["q"], role="query").instruction is None  # the models default
    assert EmbedTextRequest(texts=["q"], role="query", instruction="").instruction == ""  # the bare text
    assert EmbedTextRequest(texts=["q"], role="query", instruction=" Find clauses ").instruction == "Find clauses"
    with pytest.raises(ValidationError, match="role query only"):
        EmbedTextRequest(texts=["chunk"], role="document", instruction="Find clauses")
    with pytest.raises(ValidationError, match="single line"):
        EmbedTextRequest(texts=["q"], role="query", instruction="a\nb")
    with pytest.raises(ValidationError):
        EmbedTextRequest(texts=["q"], role="query", instruction="x" * (MAX_INSTRUCTION + 1))


def test_a_custom_task_needs_a_name_and_an_instruction_in_plain_limits():
    task = SearchTaskWrite(name=" Contracts ", instruction=" Given a question about an agreement, retrieve the clauses ")
    assert (task.name, task.instruction) == ("Contracts", "Given a question about an agreement, retrieve the clauses")
    for body, message in [({"name": " ", "instruction": "x"}, "name must not be empty"),
                          ({"name": "n" * 41, "instruction": "x"}, "name must be at most 40"),
                          ({"name": "a\tb", "instruction": "x"}, "name must be a single line"),
                          ({"name": "n", "instruction": ""}, "instruction must not be empty"),
                          ({"name": "n", "instruction": "y" * 301}, "instruction must be at most 300"),
                          ({"name": "n", "instruction": "a\r\nb"}, "instruction must be a single line")]:
        with pytest.raises(ValidationError, match=message):
            SearchTaskWrite(**body)


def test_search_task_ids_are_short_lowercase_names_or_custom_ids():
    for good in ("general", "multi-step", "exact", "0" * 32, "ab12" * 8):
        assert re.match(SEARCH_TASK, good)
    for bad in ("", "General", "-x", "a" * 41, "multi step", "../x"):
        assert not re.match(SEARCH_TASK, bad)


def refusing_app():
    app = create_app("test")

    @app.post("/embed")
    def embed(body: EmbedTextRequest) -> dict:
        return {"ok": True}

    return TestClient(app)


def test_a_refused_request_never_echoes_what_was_sent():
    client = refusing_app()
    secret = "Given a question, retrieve the clauses of contract 4471-B " + "x" * 300
    r = client.post("/embed", json={"texts": ["q"], "role": "query", "instruction": secret})
    assert r.status_code == 422 and "4471-B" not in r.text
    [error] = r.json()["detail"]
    assert set(error) == {"loc", "msg", "type"} and error["loc"] == ["body", "instruction"]
    assert error["msg"] == "Value error, the instruction must be at most 300 characters, not 358"
    r = client.post("/embed", json={"texts": ["q"], "role": "query", "extra": "4471-B"})
    assert r.status_code == 422 and "4471-B" not in r.text  # unknown fields are refused, without their value


def test_a_lone_surrogate_is_a_plain_422_not_a_500():
    client = refusing_app()
    body = json.dumps({"texts": ["q"], "role": "query", "instruction": "lone " + chr(0xD800) + " surrogate"})
    r = client.post("/embed", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["msg"] == "Input should be a valid string, unable to parse raw data as a unicode string"
    assert "surrogate" not in r.text


def test_a_task_name_needs_a_letter_or_digit_and_no_invisible_formatting():
    for name in ("...", "- -", chr(0x200B) + "Contracts", "Con" + chr(0x200D) + "tracts", "Contracts" + chr(0x202E),
                 chr(0xFEFF) + "Contracts", "Contracts" + chr(0xE0041)):  # zero-width, bidi, BOM, tag
        with pytest.raises(ValidationError, match="letter or digit|invisible formatting"):
            SearchTaskWrite(name=name, instruction="Find clauses")
    assert SearchTaskWrite(name="Q3 (2026)", instruction="Find clauses").name == "Q3 (2026)"
    instruction = "Find clauses" + chr(0x200D) + " here"  # instructions keep the one-line rule only
    assert SearchTaskWrite(name="Contracts", instruction=instruction).instruction == instruction
