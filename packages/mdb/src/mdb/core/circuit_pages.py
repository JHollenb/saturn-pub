"""Paging of a circuit's node tree: the record keeps the root and its first levels, and each group
that does not fit one page is a pointer to a child page (a sha256-checked JSON blob).

Torch-free (the recorded server reads pages with it). ``paginate`` works on node views (the JSON
a circuit record holds); ``find`` routes a node id to the page that holds it. A paged node keeps
its own numbers and gains ``paged``: its children WERE examined and are stored elsewhere, which
is not the same as ``unexamined`` (members dropped from a ``+N more`` group, never expanded).
Node ids are preorder numbers, so the ids below a paged node are the ``descendants`` that follow
its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .bounded import json_bytes
from .records import num

PAGE_SCHEMA = "mdb-circuit-page-v1"
# One page (the record's root page or a child page) holds at most this much canonical JSON and
# this many values. Measured at 40 layers (qwen3 tiny, top=8 depth=3): 371 nodes, 66 KiB, 3576
# values, ~178 bytes a node, so the record's own 64 KiB bound is what a large circuit hits first.
PAGE_BYTES = 32 * 1024
PAGE_VALUES = 1500


class PageError(ValueError):
    """A circuit page cannot be made, found or read."""


def _values(x: Any) -> int:
    if isinstance(x, dict):
        return 1 + sum(_values(v) for v in x.values())
    if isinstance(x, list):
        return 1 + sum(_values(v) for v in x)
    return 1


def fits(node: dict[str, Any]) -> bool:
    return len(json_bytes(node)) <= PAGE_BYTES and _values(node) <= PAGE_VALUES


def total(node: dict[str, Any]) -> int:
    """Nodes in ``node``'s subtree, itself included, counting the ones held in pages."""
    paged = node.get("paged")
    return (1 + sum(total(c) for c in node.get("children", ()))
            + (int(paged["descendants"]["value"]) if paged else 0))


def _inline(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Descendants of ``node`` that still hold their children inline."""
    out = []
    for child in node.get("children", ()):
        if child.get("children"):
            out.append(child)
        out += _inline(child)
    return out


def paginate(node: dict[str, Any], put: Callable[[dict[str, Any]], dict[str, Any]],
             meta: dict[str, Any]) -> None:
    """Make ``node`` fit one page: the descendant with the largest subtree has its children
    moved to a child page (``put`` stores it and returns its pointer) until it fits. A child page
    is fitted the same way, so no page is over ``PAGE_BYTES`` / ``PAGE_VALUES``."""
    while not fits(node):
        candidates = _inline(node)
        if not candidates:
            raise PageError("a circuit node does not fit one page even with every child "
                            "paged; lower top")
        child = max(candidates, key=lambda c: len(json_bytes(c)))
        below, kids = total(child) - 1, len(child["children"])
        subtree = dict(child)
        paginate(subtree, put, meta)
        pointer = put({"schema": PAGE_SCHEMA, **meta, "node": subtree})
        del child["children"]
        child["paged"] = {
            "state": "paged",
            "children": num(kids, "derived-exact"), "descendants": num(below, "derived-exact"),
            "pointer": pointer,
            "meaning": "the children of this node were examined; they are stored as a "
                       "sha256-checked page, not listed here (`circuit show ID node=ID`, or "
                       "the stored route with node=ID, reads it)"}


def find(node: dict[str, Any], wanted: str, load: Callable[[dict[str, Any]], dict[str, Any]]
         ) -> dict[str, Any] | None:
    """The page for node ``wanted`` below ``node`` (a node view of a root page or a child page):
    its child page when it is paged, else its inline subtree as an ``inline`` page. ``load``
    resolves a pointer to a checked page."""
    number = int(wanted[1:])
    for child in node.get("children", ()):
        first = int(child["id"][1:])
        if not first <= number < first + total(child):
            continue
        if child["id"] == wanted:
            if "paged" in child:
                return load(child["paged"]["pointer"])
            return {"schema": PAGE_SCHEMA, "state": "inline", "node": child}
        if "paged" in child:
            return find(load(child["paged"]["pointer"])["node"], wanted, load)
        return find(child, wanted, load)
    return None


def locate(root: dict[str, Any], count: int, wanted: Any,
           load: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """``find`` from the record's root, with the id checked (``n1..n<count>``)."""
    if not (isinstance(wanted, str) and wanted[:1] == "n" and wanted[1:].isdigit()
            and 1 <= int(wanted[1:]) <= count):
        raise PageError(f"no node {wanted!r} (ids n1..n{count})")
    if wanted == root["id"]:
        return {"schema": PAGE_SCHEMA, "state": "inline", "node": root}
    found = find(root, wanted, load)
    if found is None:
        raise PageError(f"no node {wanted!r} (ids n1..n{count})")
    return found
