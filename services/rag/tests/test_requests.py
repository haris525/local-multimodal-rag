import pytest
from pydantic import ValidationError

from rag.api import SearchRequest
from rag.chat import ChatRequest

DOC = "0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize("bad", ["../../tables/x", "ABCDEF" * 6, "short"])
def test_chat_and_search_accept_only_document_ids(bad):
    assert ChatRequest(question="q", doc_ids=[DOC]).doc_ids == [DOC]
    assert SearchRequest(question="q", doc_ids=[DOC]).doc_ids == [DOC]
    with pytest.raises(ValidationError):
        ChatRequest(question="q", doc_ids=[DOC, bad])
    with pytest.raises(ValidationError):
        SearchRequest(question="q", doc_ids=[bad])
