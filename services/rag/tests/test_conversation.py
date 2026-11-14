import asyncio
import json

from rag_common.contracts import MemoryItem, ThreadContext, ThreadMessage

from rag.conversation import Conversation, extract_memories, maintain, split_for_compaction
from rag.prompts import build_messages


def msgs(n: int, words: int = 10) -> list[ThreadMessage]:
    return [ThreadMessage(seq=i + 1, role="user" if i % 2 == 0 else "assistant", content="word " * words,
                          created_at="t") for i in range(n)]


def test_compaction_waits_for_the_trigger_then_folds_all_but_the_recent_turns():
    assert split_for_compaction(msgs(8), trigger_tokens=3000, trigger_messages=12, keep=6) == ([], msgs(8))
    fold, keep = split_for_compaction(msgs(14), trigger_tokens=3000, trigger_messages=12, keep=6)  # too many
    assert ([m.seq for m in fold], [m.seq for m in keep]) == (list(range(1, 9)), list(range(9, 15)))
    fold, _ = split_for_compaction(msgs(8, words=800), trigger_tokens=3000, trigger_messages=12, keep=6)  # too big
    assert [m.seq for m in fold] == [1, 2]


def test_compaction_folds_at_most_a_capped_number_of_the_oldest_turns_per_run():
    """A thread whose summary kept failing grows; one run folds at most max_fold messages within max_fold_tokens,
    the oldest first, and leaves the rest for the next run."""
    many = msgs(200, words=300)
    fold, rest = split_for_compaction(many, trigger_tokens=3000, trigger_messages=12, keep=6, max_fold=24,
                                      max_fold_tokens=12000)
    assert [m.seq for m in fold] == list(range(1, len(fold) + 1)) and 1 <= len(fold) <= 24
    assert sum(len(m.content[:2000]) for m in fold) // 3 <= 12000 + 2000 and rest[0].seq == len(fold) + 1
    fold, _ = split_for_compaction(msgs(100), trigger_tokens=3000, trigger_messages=12, keep=6, max_fold=24)
    assert len(fold) == 24


def test_conversation_block_strips_old_citations_and_sits_before_the_sources():
    conv = Conversation("t" * 32, "User compared Q3 revenue across regions.",
                        [{"role": "user", "content": "And the North?"},
                         {"role": "assistant", "content": "North had 120 k$ [1][2]."}])
    block = conv.prompt_block(n_turns=4)
    assert block.endswith("assistant: North had 120 k$.") and "[1]" not in block
    _system, user = build_messages("What about West?", [], {}, block, conv.memory_block())
    assert user["content"][0]["text"].startswith("Conversation so far") and "Summary of earlier" in user["content"][0]["text"]


class FakeLLM:
    """Summaries for plain calls, a memory JSON for json_mode calls; records every prompt."""

    def __init__(self, memories: list[str]):
        self.memories = memories
        self.prompts: list[str] = []

    async def complete(self, messages, max_tokens, temperature=0.0, json_mode=False, tools=None, purpose=""):
        self.prompts.append(messages[-1]["content"])
        if json_mode:
            return {"content": json.dumps({"memories": self.memories})}
        return {"content": "The user compared Q3 revenue across regions; South was highest."}


class FakeThreads:
    def __init__(self, messages: list[ThreadMessage]):
        self.messages = messages
        self.summary: tuple[str, int] | None = None
        self.saved: list[str] = []

    async def context(self, thread_id):
        return ThreadContext(summary="", summary_upto=0, messages=self.messages)

    async def set_summary(self, thread_id, summary, upto):
        self.summary = (summary, upto)

    async def memories(self):
        return [MemoryItem(memory_id="b" * 32, text="The user prefers tables.", source="user", created_at="t")]

    async def remember(self, text, thread_id, source="auto"):
        self.saved.append(text)


def test_maintain_writes_the_summary_and_saves_implicit_memories():
    """Runs the real background job end to end, so a missing prompt or name fails here, not silently."""
    threads, llm = FakeThreads(msgs(14)), FakeLLM(["The user is a regulatory engineer."])
    asyncio.run(maintain(threads, llm, "a" * 32, "As a regulatory engineer I need the Q3 numbers",
                         trigger_tokens=3000, trigger_messages=12, keep=6, auto_memory=True))
    assert threads.summary == ("The user compared Q3 revenue across regions; South was highest.", 8)
    assert threads.saved == ["The user is a regulatory engineer."]
    assert "Current summary:" in llm.prompts[0] and "Never remember" in llm.prompts[1]
    assert "- The user prefers tables." in llm.prompts[1]  # known memories are shown, so they are not re-saved


def test_extracted_memories_never_include_credentials():
    llm = FakeLLM(["The user's api key is sk-Abc123XYZ789qwerty", "The user works in legal."])
    assert asyncio.run(extract_memories(llm, "msg", [])) == ["The user works in legal."]
