"""Route each thread message before any retrieval, and carry out memory requests.

One JSON call reads the user's own message (never document text, so a document cannot write a memory)
plus the conversation, and decides:

    documents       a question about the selected documents (it may also carry facts to remember)
    history         a question about this conversation ("what did I ask earlier?")
    memory_save     only something to remember
    memory_recall   "what do you remember about me?", "what is my lucky number?" (about the user only)
    memory_forget   forget one or more saved items
    assistant       questions about the assistant itself, or small talk

Memory changes follow the Deep Agents memory guidelines: saving is the first action, the user is told
what changed, and credentials are never stored.

A word shared with a saved memory can pull the router towards memory_recall ("What is the beacon word for
this run?" with a memory mentioning a "run"). So the prompt says whether documents are selected, and
parse_intent downgrades memory_recall to documents when documents are selected and the message neither
refers to the user nor to memory.
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, get_args

from rag_common.chat_events import ChatEvent
from rag_common.contracts import MemoryItem
from rag_common.secrets import looks_secret

from rag.clients import LLMClient, ServiceError
from rag.conversation import MEMORY_RULES, Conversation, ThreadsClient
from rag.notices import notice

log = logging.getLogger("rag.intent")

Kind = Literal["documents", "history", "memory_save", "memory_recall", "memory_forget", "assistant"]
KINDS = get_args(Kind)
MAX_SAVES = 5
MAX_CHARS = 500

_ROUTER = """You route the user's latest message in a chat with an assistant that answers from the user's
selected documents and keeps a long-term memory of the user. The user may make typos.

Saved memories (number: text):
{memories}

Selected documents: {selection}

Conversation so far:
{conversation}

User's latest message:
\"\"\"{message}\"\"\"

Intents:
- documents: asks about the content of the documents (possibly also saying something to remember).
- history: asks about this conversation itself, e.g. "what did I ask earlier?", "do you remember what the
  report said about X?" (answered from the conversation above).
- memory_save: only asks to remember something, or only states lasting facts or preferences about the user.
- memory_recall: only when the message asks what you remember or know about the user themselves
  ("what do you remember about me?", "what do you know about me?") or asks for one of the user's own
  saved facts ("what is my lucky number?").
- memory_forget: asks to forget or delete something you remember.
- assistant: asks about the assistant itself (what it can do, whether it can remember), or small talk.
Words like "memory" or "remember" inside a document question do not make it a memory intent. A
factual question about content ("what is the beacon word for this run?", "what does the report say
about X?") is documents, even when one of its words also appears in a saved memory, and especially
when documents are selected.

Memory rules: {rules}
Take memories only from the user's latest message, never from the conversation. One item per fact, as a
short third-person sentence, in the language the user wrote in. When a new fact contradicts a saved
memory, put it in "replace" with that memory's number instead of in "save". When the message repeats
something already saved (in any wording or language), put that memory's number in "known" instead.

Return JSON only:
{{"intent": "...",
  "save": ["new fact"],
  "replace": [{{"number": 1, "text": "updated fact"}}],
  "known": [3],
  "forget": [2],
  "forget_confirmed": false,
  "forget_all": false,
  "question": "the document question as a standalone query using the conversation, or empty"}}
"forget_confirmed" is true only when the user clearly named every item to forget (e.g. "forget both").
"forget_all" is true when the user asks to forget everything."""


# The user or memory, referred to in English or a few other languages: without any of these, a message is not
# asking what is remembered about the user. A miss only means a document search (whose prompt still carries
# the memories), so the list leans towards keeping the router's recall.
_ABOUT_USER = re.compile(
    r"\b(i|me|my|mine|myself|remember\w*|memor\w*|recall\w*|forg[eo]t\w*|saved"
    r"|ich|mich|mir|mein\w*|erinner\w*|je|moi|mon|ma|mes|souviens|rappelles|yo|mi|mis|recuerd\w*"
    r"|io|mio|mia|miei|mie|ricord\w*|eu|meu|minha|lembr\w*|ik|mij|mijn|herinner\w*"
    r"|jeg|meg|min|mitt|husk\w*|jag|mig)\b",
    re.IGNORECASE,
)


# Asking to keep something (an imperative, in English or a few other languages): with a credential in the message,
# the memory path's fixed reply refuses it (never a refusal the model writes itself). "What does note 3 say about
# API key rotation?" asks nothing to be kept.
_KEEP_VERB = (r"(remember|memori[sz]e|save|store|note\s+down|jot\s+down|write\s+down|keep\s+in\s+mind|don'?t\s+forget"
              r"|do\s+not\s+forget|merk\w*|erinner\w*|speicher\w*|notier\w*|retiens|souviens|enregistre\w*|recuerda\w*"
              r"|guarda\w*|anota\w*|ricorda\w*|salva\w*|memorizza\w*|lembr\w*|grava\w*|onthoud\w*|husk\w*|kom\s+ih\w*)")
_KEEP = re.compile(
    r"^\s*(?:(?:please|pls|hey|ok|okay|now|also)[\s,]+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|i\s+(?:want|need)\s+you\s+to\s+|i'?d\s+like\s+you\s+to\s+)?"
    + _KEEP_VERB + r"\b|(?:\b(?:and|also|then|please)\s+|[,;.!?]\s*(?:please\s+)?)" + _KEEP_VERB + r"\b"
    r"|^\s*(?:note|remember|reminder|fyi)\s*:",  # "my password is X, remember it"; "Note: my API key is ..."
    re.IGNORECASE,
)


def keep_request(message: str) -> bool:
    """True when the message asks to remember, save or note something (an imperative; see _KEEP)."""
    return bool(_KEEP.search(message))


_CLAUSES = re.compile(r"[,;.!?\n]+|\s+(?:and|but|then)\s+", re.IGNORECASE)
_ASKS = re.compile(
    r"^\s*(?:also\s+|now\s+)?(what|which|who|whom|whose|when|where|why|how|is|are|was|were|does|do|did|can|could|"
    r"would|will|should|list|show|summari[sz]e|find|compare|explain|tell)\b", re.IGNORECASE)


def asks_more(message: str) -> bool:
    """True when a clause other than the one asking to remember asks something ("remember my password is
    hunter2, and what does the report say about revenue?"); "Can you remember my password is hunter2?" does not."""
    return any(_ASKS.match(c) for c in _CLAUSES.split(message) if c.strip() and not _KEEP.search(c))


def about_user(message: str) -> bool:
    """True when the message refers to the user or to memory at all (see _ABOUT_USER)."""
    return bool(_ABOUT_USER.search(message))


@dataclass
class Intent:
    kind: Kind
    question: str = ""  # standalone document question
    memory_kind: str = ""  # memory_save | memory_forget as the router saw it, kept when a question turns it into documents
    save: list[str] = field(default_factory=list)
    replace: list[tuple[str, str]] = field(default_factory=list)  # (memory_id, new text)
    forget: list[str] = field(default_factory=list)  # memory ids
    forget_confirmed: bool = False
    forget_all: bool = False
    known: list[str] = field(default_factory=list)  # saved memories the message repeats
    refused_secret: bool = False
    too_long: int = 0


@dataclass
class MemoryOutcome:
    saved: list[MemoryItem] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    replaced: list[tuple[str, MemoryItem]] = field(default_factory=list)  # (old text, new item)
    forgotten: list[MemoryItem] = field(default_factory=list)
    ambiguous: list[MemoryItem] = field(default_factory=list)
    full: bool = False

    def events(self) -> list[tuple[ChatEvent, dict]]:
        """SSE `memory` events: {action: saved | forgotten, items: [{memory_id, text}]}."""
        out = []
        saved = self.saved + [new for _, new in self.replaced]
        if saved:
            out.append((ChatEvent.MEMORY, {"action": "saved",
                                           "items": [{"memory_id": m.memory_id, "text": m.text} for m in saved]}))
        if self.forgotten:
            out.append((ChatEvent.MEMORY, {"action": "forgotten",
                                           "items": [{"memory_id": m.memory_id, "text": m.text} for m in self.forgotten]}))
        return out


def _items(value) -> list:
    """A list field of the router's JSON: a lone value is a one-item list, anything else empty."""
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, int, dict)) and not isinstance(value, bool) else []


def _clean(texts: list, intent: Intent) -> list[str]:
    kept = []
    for t in texts:
        if not isinstance(t, str) or len(t.strip()) < 3:
            continue
        t = t.strip()
        if looks_secret(t):
            intent.refused_secret = True
        elif len(t) > MAX_CHARS:
            intent.too_long += 1
        else:
            kept.append(t)
    return kept[:MAX_SAVES]


def parse_intent(raw: dict, message: str, memories: list[MemoryItem], has_documents: bool = False) -> Intent:
    """Validate the router's JSON. Memory numbers outside the list are ignored; credentials and
    over-long items are refused; a memory request that also carries a question becomes a document turn;
    with documents selected, a recall that does not refer to the user or to memory is a document question."""
    kind = raw.get("intent") if raw.get("intent") in KINDS else "documents"
    if kind == "memory_recall" and has_documents and not about_user(message):
        kind = "documents"
    by_number = {i: m for i, m in enumerate(memories, 1)}

    def memory(number) -> MemoryItem | None:
        try:
            return by_number.get(int(number))
        except (TypeError, ValueError):
            return None

    question = raw.get("question")
    secret = keep_request(message) and looks_secret(message)
    asked = question.strip() if isinstance(question, str) else ""
    separate_question = bool(asked) and asked != message.strip() and not looks_secret(asked)
    if secret and not separate_question and asks_more(message):
        # the message also asks something: answer it, with the fixed refusal for the credential part
        kind, question = "documents", message
    elif secret and not separate_question:
        # A credential to remember, and nothing else asked: the memory path and its fixed refusal, whatever the
        # router said (it has answered "assistant", leaving the refusal to the model).
        kind = "memory_save"
    elif secret and kind not in ("documents", "history"):
        kind = "documents"  # a separate question that holds no credential is answered after the refusal
    intent = Intent(kind=kind, question=question.strip() if isinstance(question, str) else "")
    intent.memory_kind = "memory_save" if secret else (kind if kind in ("memory_save", "memory_forget") else "")
    intent.save = _clean(_items(raw.get("save")), intent)
    for entry in _items(raw.get("replace")):
        old = memory(entry.get("number")) if isinstance(entry, dict) else None
        new = _clean([entry.get("text")], intent) if isinstance(entry, dict) else []
        if old and new:
            intent.replace.append((old.memory_id, new[0]))
        elif new:
            intent.save += new
    intent.forget = list(dict.fromkeys(m.memory_id for m in map(memory, _items(raw.get("forget"))) if m))
    intent.known = list(dict.fromkeys(m.text for m in map(memory, _items(raw.get("known"))) if m))
    intent.forget_confirmed = raw.get("forget_confirmed") is True
    intent.forget_all = raw.get("forget_all") is True
    if secret or (intent.kind == "memory_save" and looks_secret(message)):
        intent.refused_secret = True  # the message held a credential: nothing from it that looks secret is kept
    if intent.memory_kind and intent.question:
        intent.kind = "documents"  # do the memory part first (memory_kind keeps it), then answer the question
    if intent.kind in ("documents", "history") and not intent.question:
        intent.question = message  # history falls back to a document search with it
    elif intent.kind not in ("documents", "history"):
        intent.question = ""
    return intent


async def route(llm: LLMClient, message: str, conv: Conversation, documents: int = 0, max_tokens: int = 600) -> Intent:
    """`documents`: how many documents the user has selected for this turn."""
    memories = "\n".join(f"{i}: {m.text}" for i, m in enumerate(conv.memories, 1)) or "(none)"
    selection = f"{documents} (questions about their content are documents)" if documents else "none"
    prompt = _ROUTER.format(memories=memories, selection=selection,
                            conversation=conv.prompt_block(6) or "(start of chat)", message=message[:4000],
                            rules=MEMORY_RULES)
    try:
        msg = await llm.complete([{"role": "user", "content": prompt}], max_tokens=max_tokens, json_mode=True,
                                 purpose="intent routing")
        raw = json.loads(msg.get("content") or "{}")
        return parse_intent(raw if isinstance(raw, dict) else {}, message, conv.memories, documents > 0)
    except (ValueError, TypeError, AttributeError, ServiceError) as exc:  # unusable answer: a plain document turn
        notice("routing_failed", "This message could not be classified (a memory request, a question about the "
               "chat, or a document question), so it was answered as a document question.", error=str(exc))
        return parse_intent({}, message, conv.memories, documents > 0)


Embed = Callable[[list[str]], Awaitable[list[list[float]]]]


async def find_repeat(text: str, memories: list[MemoryItem], embed: Embed | None, threshold: float) -> MemoryItem | None:
    """The saved memory that says the same thing in other words or another language, if any. Measured with
    the multilingual text embedder: real repeats scored 0.82-0.95, different facts at most 0.73."""
    if embed is None or not memories:
        return None
    try:
        vectors = await embed([text] + [m.text for m in memories])
    except ServiceError as exc:  # models busy or down: the store's exact-text check still applies
        log.warning("duplicate check skipped: %s", exc.detail)
        return None
    scores = [sum(a * b for a, b in zip(vectors[0], v, strict=True)) for v in vectors[1:]]
    best = max(range(len(scores)), key=scores.__getitem__)
    return memories[best] if scores[best] >= threshold else None


async def apply_memory(threads: ThreadsClient, intent: Intent, conv: Conversation, embed: Embed | None = None,
                       duplicate_cosine: float = 0.78) -> MemoryOutcome:
    """Carry out the memory part of a turn and keep `conv.memories` current for the answer prompt.
    Ambiguous forgets (several matches, not clearly named) and "forget everything" delete nothing."""
    out = MemoryOutcome()
    source = "user" if intent.memory_kind == "memory_save" else "auto"
    by_id = {m.memory_id: m for m in conv.memories}

    async def repeated(text: str, replacing: str | None = None) -> bool:
        others = [m for m in conv.memories + out.saved if m.memory_id != replacing]
        if repeat := await find_repeat(text, others, embed, duplicate_cosine):
            out.already.append(repeat.text)
        return repeat is not None

    async def save(text: str) -> MemoryItem | None:
        if await repeated(text):
            return None
        try:
            item = await threads.remember(text, conv.thread_id, source)
        except ServiceError as exc:
            if exc.status == 409:  # memory is full of items the user added
                out.full = True
                return None
            raise
        if item is None:
            out.already.append(text)
        return item

    for memory_id, text in intent.replace:  # rewritten in place: a rejected update keeps the old fact
        if await repeated(text, replacing=memory_id):
            continue
        try:
            item = await threads.replace(memory_id, text, source)
        except ServiceError as exc:
            if exc.status != 404:
                raise
            if item := await save(text):  # deleted meanwhile (e.g. in the Memory panel): save it as new
                out.saved.append(item)
            continue
        if item is None:
            out.already.append(text)
        else:
            out.replaced.append((by_id[memory_id].text, item))
    for text in intent.save:
        if item := await save(text):
            out.saved.append(item)
    targets = [by_id[i] for i in intent.forget if i in by_id]
    if intent.memory_kind == "memory_forget" and not intent.forget_all:
        if len(targets) > 1 and not intent.forget_confirmed:
            out.ambiguous = targets
        else:
            for item in targets:
                await threads.forget(item.memory_id)
                out.forgotten.append(item)
    gone = {m.memory_id for m in out.forgotten}
    updated = {new.memory_id: new for _, new in out.replaced}
    conv.memories = [updated.get(m.memory_id, m) for m in conv.memories if m.memory_id not in gone] + out.saved
    return out


def memory_reply(intent: Intent, out: MemoryOutcome, conv: Conversation) -> str:
    """The direct answer for memory turns. Never repeats refused text (it may hold a credential)."""
    lines: list[str] = []
    lines += [f"Saved: {m.text}" for m in out.saved]
    lines += [f"Already saved: {t}" for t in dict.fromkeys(out.already + intent.known)]
    lines += [f"Updated: {new.text} (replaces: {old})" for old, new in out.replaced]
    lines += [f"Forgot: {m.text}" for m in out.forgotten]
    if out.ambiguous:
        lines.append("Several memories match. Which should I forget?\n"
                     + "\n".join(f"{i}. {m.text}" for i, m in enumerate(out.ambiguous, 1)))
    if intent.forget_all:
        lines.append("I don't delete everything from chat. Open the Memory panel to review and delete items.")
    elif intent.memory_kind == "memory_forget" and not (out.forgotten or out.ambiguous):
        lines.append("I couldn't find that in my memory, so nothing was deleted.")
    if intent.refused_secret:
        lines.append("I don't store passwords, keys, tokens or other credentials, so I did not save that.")
    if intent.too_long:
        lines.append(f"That is too long to remember (the limit is {MAX_CHARS} characters). Please shorten it.")
    if out.full:
        lines.append("Memory is full. Delete some items in the Memory panel, then ask again.")
    if intent.kind == "memory_recall":
        if conv.memories:
            lines.append("Here is what I remember about you:\n" + "\n".join(f"- {m.text}" for m in conv.memories))
        else:
            lines.append('I have nothing saved about you yet. Tell me, for example, "Remember that I prefer short answers."')
    if intent.kind == "memory_save" and not lines:
        lines.append("I didn't find anything to remember in that message.")
    if out.saved or out.replaced:
        lines.append("You can review or delete it under Memory.")
    return "\n".join(lines)
