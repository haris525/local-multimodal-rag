"""Deep-search implementations ("arms") by name: the one place the rest of rag looks them up.

Core registers three: langgraph (rag.agent), deepagents (rag.agent_deepagents) and deepagents_stock
(rag.agent_deepagents_stock). Any package directly under rag/ is an optional arm: it is imported when the
registry first loads and registers itself with register(). A package that is not there is simply not
registered, so deleting an optional arm's folder removes the arm and nothing else (its tests live in a folder
of their own as well).

A chat request's `agent` and RAG_DEEP_AGENT must name a registered arm; anything else is refused with the
list of names (a 422 for a request, a failed start for the setting).

An arm declares at registration whether it supports fan-out (rag.fanout: researchers with reserved research
budgets). Core's deepagents does; langgraph and deepagents_stock do not. A request's `fanout: true` for an arm
without it is a 422, and RAG_DEEP_FANOUT with such an arm as RAG_DEEP_AGENT fails at start; both messages list
the arms that support it.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

CORE = ("langgraph", "deepagents", "deepagents_stock")
_ARMS: dict[str, ModuleType] = {}
_FANOUT: set[str] = set()  # the arms that support fan-out
_loaded = False


def register(name: str, module: ModuleType, *, fanout: bool = False) -> None:
    """Register `module` as the arm `name`; its deep_search(deps, req, docs, conv, question, timings) is looked
    up at call time (so a test can replace it). `fanout`: the arm runs researchers on reserved budgets when a
    request or RAG_DEEP_FANOUT asks (rag.fanout)."""
    if not callable(getattr(module, "deep_search", None)):
        raise TypeError(f"arm {name!r}: {module.__name__} has no deep_search")
    _ARMS[name] = module
    if fanout:
        _FANOUT.add(name)
    else:
        _FANOUT.discard(name)


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    import rag
    from rag import agent, agent_deepagents, agent_deepagents_stock

    register("langgraph", agent)
    register("deepagents", agent_deepagents, fanout=True)
    register("deepagents_stock", agent_deepagents_stock)
    for info in pkgutil.iter_modules(rag.__path__):
        if info.ispkg:  # an optional arm registers itself on import
            importlib.import_module(f"rag.{info.name}")


def names() -> list[str]:
    """The registered arms: the core three first, then optional ones in the order they registered (an optional arm
    imported directly, by its tests, registers before the core ones load)."""
    _load()
    return [n for n in CORE if n in _ARMS] + [n for n in _ARMS if n not in CORE]


def get(name: str) -> ModuleType:
    """The arm `name`; ValueError naming the registered arms for anything else."""
    _load()
    if name not in _ARMS:
        raise ValueError(unknown(name))
    return _ARMS[name]


def unknown(name: object) -> str:
    return f"unknown deep-search agent {name!r}; available: {', '.join(names())}"


def check(name: str | None) -> str | None:
    """`name` when it is registered (or None); ValueError otherwise (pydantic turns it into a 422)."""
    if name is not None and name not in names():
        raise ValueError(unknown(name))
    return name


def supports_fanout(name: str) -> bool:
    """Whether the registered arm `name` supports fan-out (False for an unknown name)."""
    _load()
    return name in _FANOUT


def fanout_names() -> list[str]:
    """The registered arms that support fan-out, in the order of names()."""
    return [n for n in names() if n in _FANOUT]


def no_fanout(name: str) -> str:
    return f"deep-search agent {name!r} does not support fan-out; arms that do: {', '.join(fanout_names())}"
