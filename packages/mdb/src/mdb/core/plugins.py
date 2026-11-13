"""Plug-in lookup through entry points: how the engine reaches code it does not ship.

Standard library only (``importlib.metadata``); nothing is imported until a name is asked for.
Two groups are read:

* ``mdb.lanes`` (``VERB_GROUP``): command verbs. Each entry point is a callable
  ``main(argv: list[str]) -> int`` that receives the verb itself followed by its arguments.
* ``mdb.lanes.exports`` (``EXPORT_GROUP``): attributes of the ``mdb`` package that the engine
  does not define (``name = "module:attribute"``), resolved lazily by ``mdb.__getattr__``.

The experiment lanes (distribution ``mdb-lanes``) declare both groups. Distributions are found
on ``sys.path`` like any other installed metadata, so a source tree that carries a
``*.dist-info`` directory next to its packages is found without being installed. The first
entry point of a name wins (``sys.path`` order); later duplicates are ignored.
"""
from __future__ import annotations

from functools import lru_cache

VERB_GROUP = "mdb.lanes"
EXPORT_GROUP = "mdb.lanes.exports"
PLUGIN_DIST = "mdb-lanes"


@lru_cache(maxsize=None)
def entries(group: str) -> dict:
    """``{name: EntryPoint}`` of one group (cached; ``clear()`` forgets it)."""
    from importlib.metadata import entry_points

    out = {}
    for entry in entry_points(group=group):
        out.setdefault(entry.name, entry)
    return out


def clear() -> None:
    entries.cache_clear()


def verbs() -> list[str]:
    return sorted(entries(VERB_GROUP))


def verb(name: str):
    """The callable registered for verb ``name``, or ``None`` when no plug-in provides it."""
    entry = entries(VERB_GROUP).get(name)
    return None if entry is None else entry.load()


def export(name: str):
    """``(found, value)`` for attribute ``name`` of the ``mdb`` package."""
    entry = entries(EXPORT_GROUP).get(name)
    return (False, None) if entry is None else (True, entry.load())


def installed() -> bool:
    """Whether any plug-in declares verbs or exports."""
    return bool(entries(VERB_GROUP) or entries(EXPORT_GROUP))
