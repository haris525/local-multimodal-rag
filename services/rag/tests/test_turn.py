import asyncio
import json
from types import SimpleNamespace

from rag_common.contracts import MemoryItem

from rag.chat import ChatRequest
from rag.config import get_settings
from rag.conversation import Conversation
from rag.turn import respond

T = "a" * 32
NOTE = MemoryItem(memory_id="1" * 32, text="The user works in finance.", source="user", created_at="t")


class Router:
    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        return {"content": json.dumps({"intent": "memory_forget", "forget": [1],
                                       "question": "What does the report say about revenue?"})}


class Threads:
    def __init__(self):
        self.forgotten: list[str] = []

    async def forget(self, memory_id):
        self.forgotten.append(memory_id)


async def no_embeddings(texts):
    raise AssertionError("a forget needs no duplicate check")


def test_a_thread_turn_does_the_memory_part_then_routes_the_question():
    """Mixed message, no documents selected: the forget happens, is confirmed, and the question gets the
    "select a document" answer instead of a search."""
    threads = Threads()
    deps = SimpleNamespace(llm=Router(), threads=threads, s=get_settings(),
                           retrieval=SimpleNamespace(models=SimpleNamespace(embed_documents=no_embeddings)))
    req = ChatRequest(question="forget that I work in finance, and what does the report say about revenue?",
                      doc_ids=[], thread_id=T)

    async def run():
        return [e async for e in respond(deps, req, [], Conversation(T, memories=[NOTE]))]

    events = asyncio.run(run())
    kinds = [name for name, _ in events]
    text = "".join(data["text"] for name, data in events if name == "token")
    assert threads.forgotten == [NOTE.memory_id] and kinds[0] == "memory"
    assert dict(events)["sources"]["route"] == "no_documents"
    assert "Forgot: The user works in finance." in text and "Select at least one document" in text
