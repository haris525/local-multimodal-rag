"""Notices: what the user is told when something did not go as planned in their turn.

Every fallback, cut-off and failed retry in a chat turn calls notice(): it is logged (WARNING by default, with
the request ID) and, inside a chat request, queued for the stream. api._sse sends the queued notices as
`notice` events (rag_common.chat_events) before the next event, so they reach the UI in order, and saves them
with the answer. Outside a chat (the /v1/search endpoint, background work) a notice is only logged.

The queue is one list per request in a context variable; tasks started inside the request (LangGraph nodes)
copy the context and share the list.
"""

import logging
from contextvars import ContextVar

from rag_common.logging import log_fields

log = logging.getLogger("rag.notices")
_queue: ContextVar[list[dict] | None] = ContextVar("rag_notices", default=None)


def begin() -> list[dict]:
    """Start collecting notices for the current request; returns the list that collects them."""
    notices: list[dict] = []
    _queue.set(notices)
    return notices


def notice(code: str, message: str, level: int = logging.WARNING, **fields) -> None:
    """Log the event and tell the user. `message` is plain language for the user and may quote the question, so
    only the code and `fields` go to the log."""
    log_fields(log, f"notice {code}", level, code=code, **fields)
    queue = _queue.get()
    if queue is not None and not any(n["code"] == code and n["message"] == message for n in queue):
        queue.append({"code": code, "message": message, "sent": False})


def pending() -> list[dict]:
    """Notices not yet sent, marked sent ({code, message} each)."""
    queue = _queue.get() or []
    out = []
    for n in queue:
        if not n["sent"]:
            n["sent"] = True
            out.append({"code": n["code"], "message": n["message"]})
    return out


def collected() -> list[dict]:
    return [{"code": n["code"], "message": n["message"]} for n in (_queue.get() or [])]
