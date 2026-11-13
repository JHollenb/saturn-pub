"""Producer/consumer edges of a port program, read from ``PortSpec.inputs`` (no text parsing).

``PortSpec.inputs`` names the earlier ports a port reads, plus two pseudo-nodes: ``"cache"``
(the key/value cache rows of earlier tokens) and ``"prev_layer"`` (the previous layer's
committed output; ``emb`` for layer 0). ``producers``/``consumers`` hold real ports only;
``pseudo`` lists a port's pseudo-node reads, which ``backtrace`` shows as named frames.

Across layers: ``L<i>.h`` is produced by ``L<i-1>.h_out`` (``emb`` for layer 0), ``emb`` by
``start``, ``fn`` by the last layer's ``h_out``, ``logits`` by ``fn``, ``sample`` by
``logits``.
"""

from __future__ import annotations

from typing import Any

PSEUDO = ("cache", "prev_layer")


def producers(program: Any) -> dict[str, tuple[str, ...]]:
    names = set(program.names)
    return {spec.name: tuple(i for i in spec.inputs if i in names) for spec in program.ports}


def pseudo(program: Any) -> dict[str, tuple[str, ...]]:
    return {spec.name: tuple(i for i in spec.inputs if i in PSEUDO) for spec in program.ports}


def consumers(program: Any) -> dict[str, tuple[str, ...]]:
    edges = producers(program)
    return {
        name: tuple(other for other in program.names if name in edges[other])
        for name in program.names
    }


def layering(program: Any, focus: str, *, root: str = "h") -> list[list[str]]:
    """Ancestors of ``focus`` back to ``root``, by longest distance from ``focus``.

    Level 0 is ``[focus]``; a port shared by two paths sits at its deepest level, so the
    layer input ``h`` comes last.
    """
    edges = producers(program)
    order = {name: i for i, name in enumerate(program.names)}
    depth = {focus: 0}
    for name in sorted(order, key=order.get, reverse=True):
        if name not in depth or name == root:
            continue
        for item in edges[name]:
            depth[item] = max(depth.get(item, 0), depth[name] + 1)
    levels: dict[int, list[str]] = {}
    for name, level in depth.items():
        levels.setdefault(level, []).append(name)
    return [sorted(levels[i], key=order.get) for i in sorted(levels)]


def chain_text(levels: list[list[str]]) -> str:
    return " <- ".join(",".join(level) for level in levels)
