"""Bound a list that grows with the model to a cap, and store the full list by pointer.

One mechanism for every record or result list that grows with the layers or heads (the forecast
identity chain, the explain terms, the what-if operands): the record keeps ``cap`` items chosen
by a stated rule plus a summary (``count``, ``listed``, a ``rest`` summary, a ``pointer``), and
the full list is a sha256-checked JSON blob, put in a project ``LocalStore`` (a tensor pointer
to uint8 bytes) or, with no store, held in memory per ``run`` (``load_json`` resolves both).
Torch-free: numpy is imported only when a blob is stored or read from a store.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any

MAX_ITEMS = 64        # the record's sequence bound (``core.pointers.MAX_INLINE_ITEMS``)
_HELD: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def json_bytes(obj: Any) -> bytes:
    """The canonical JSON bytes of ``obj`` (what the blob's sha256 is taken over)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def put_json(obj: Any, *, store: Any = None, run: Any = None,
             absent_reason: str | None = None) -> dict[str, Any]:
    """Store ``obj`` and return its pointer. With ``store``: ``{state: "stored", ref, bytes,
    sha256, tensor}`` (``tensor`` is the record-level tensor pointer). Without: ``{state:
    "absent", sha256, reason}``, and ``obj`` is held in memory on ``run`` when one is given."""
    raw = json_bytes(obj)
    digest = hashlib.sha256(raw).hexdigest()
    if store is not None:
        import numpy as np

        from mdb.core.pointers import tensor_pointer

        stored = store.put_tensor(np.frombuffer(raw, dtype=np.uint8).copy())
        return {"state": "stored", "ref": stored["ref"], "bytes": len(raw), "sha256": digest,
                "tensor": tensor_pointer(stored["ref"], sha256=stored["sha256"],
                                         shape=stored["shape"], dtype=stored["dtype"])}
    if run is not None:
        _HELD.setdefault(run, {})[digest] = raw     # the bytes, so a held copy cannot change
    return {"state": "absent", "sha256": digest,
            "reason": absent_reason or "no project store: the full list is held in memory on "
                                       "the run (bounded.load_json), not stored"}


def load_json(pointer: Mapping[str, Any], *, store: Any = None, run: Any = None,
              tensor: Mapping[str, Any] | None = None) -> Any:
    """The full object ``pointer`` (from ``put_json``) points at: from ``store`` (checked
    against ``pointer["sha256"]``; ``tensor`` overrides ``pointer["tensor"]``) or from the run's
    in-memory copy. ``ValueError`` if absent or changed."""
    if pointer["state"] == "stored":
        from mdb.core.pointers import resolve_tensor

        raw = resolve_tensor(tensor if tensor is not None else pointer["tensor"], store=store,
                             framework="numpy").tobytes()
        if hashlib.sha256(raw).hexdigest() != pointer["sha256"]:
            raise ValueError("the stored list differs from the pointer's sha256")
        return json.loads(raw)
    held = _HELD.get(run, {}).get(pointer["sha256"]) if run is not None else None
    if held is None:
        raise ValueError("the full list is not held: " + pointer.get("reason", "absent"))
    if hashlib.sha256(held).hexdigest() != pointer["sha256"]:
        raise ValueError("the held list differs from the pointer's sha256")
    return json.loads(held)


def keep_indices(items: Sequence[Any], cap: int,
                 rank: Callable[[Any], Any] | None = None) -> list[int]:
    """Indices (original order) of the ``cap`` items to keep: the ``cap`` smallest by ``rank``
    (ties by position), or the first ``cap`` with no ``rank``."""
    if rank is None:
        return list(range(min(cap, len(items))))
    order = sorted(range(len(items)), key=lambda i: (rank(items[i]), i))
    return sorted(order[:cap])


def bound_list(items: Sequence[Any], cap: int = MAX_ITEMS, *, rank: Callable[[Any], Any] | None = None,
               rest: Callable[[list[Any]], dict[str, Any]] | None = None, store: Any = None,
               run: Any = None, keep: str = "") -> tuple[list[Any], dict[str, Any] | None]:
    """``(kept, summary)``. Within ``cap`` the list is returned whole and ``summary`` is
    ``None`` (nothing changes, nothing is stored). Over it ``kept`` is the ``keep_indices``
    selection and ``summary`` is ``{count, listed, keep, rest: {count, **rest(rest_items)},
    pointer}``; ``keep`` states the rule in words."""
    if len(items) <= cap:
        return list(items), None
    chosen = keep_indices(items, cap, rank)
    taken = set(chosen)
    left = [x for i, x in enumerate(items) if i not in taken]
    summary = {"count": len(items), "listed": len(chosen),
               "keep": keep or ("the first " if rank is None else "the best ranked ") + str(cap),
               "rest": {"count": len(left), **(rest(left) if rest else {})},
               "pointer": put_json(list(items), store=store, run=run)}
    return [items[i] for i in chosen], summary
