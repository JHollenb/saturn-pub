"""Statistics over the results of one op run on many prompts (``each``). Pure Python.

``claims(result)`` flattens an op result into its labelled numbers: ``path -> (value,
authority)`` for every ``{"value": number, "authority": ...}`` mapping (``path`` is the dotted
key path, list items by index: ``reads.0.mass``), and ``verdicts(result)`` its verdict and
state fields (``path -> text`` for string leaves under the keys ``verdict`` and ``state``).

Name keys (review 2026-11-14 2.1): a list whose items are all mappings carrying a distinct
text ``symbol`` (``NAME_KEYS``) is walked by that name, not by position:
``bindings.carrier.mass``, never ``bindings.0.mass``. A symbol table's page order depends on
which slots bound on the item (``circuit_map.bind`` skips a slot that binds no row), so a
positional path would line up different symbols across items. Every other list keeps index
paths. ``positional=True`` restores the old paths.

``summarize_by(results, switch)`` is the ``switch`` group-by: the same statistics per value of
a label (a dotted path into each result, e.g. ``labels.family``), with ``n`` per group.

``summarize(results)`` takes the results of the items that ran (refused items are not passed)
and returns, for every path present in all of them:

* a numeric claim: ``n``, ``mean``, ``min``, ``max`` (float64, ``math.fsum`` for the mean) and
  ``agree`` (the most common sign and how many items have it). The statistics carry the
  inputs' authority when all ``n`` inputs share it; mixed inputs are summarised per label under
  ``by_authority``, never merged into one field.
* a verdict/state field: a count per value.

Paths present in only some results are counted (``partial``), not summarised. A statistic is
arithmetic on the item values; it says nothing about prompts outside the set.

``KEY_COLUMNS`` names, per verb, the few claims a one-line-per-item table shows; another verb
shows its first two claims and its first verdict. It is a view, not a dispatch rule.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

from mdb.core.records import num

VERDICT_KEYS = ("verdict", "state")
NAME_KEYS = ("symbol",)     # list items keyed by this field when every item has a distinct one

# verb -> alternatives, each (columns (name, path), verdict path, label path); the first
# alternative whose first column the result has is used (``print`` of a margin or a value)
KEY_COLUMNS: dict[str, tuple[tuple[tuple[tuple[str, str], ...], str | None, str | None], ...]] = {
    "whatif": (((("native", "measured.delta"), ("exact", "prediction.exact.value")),
                "effect.verdict", None),),
    "reads": (((("mass", "reads.0.mass"), ("effect", "reads.0.to_consumer"),
                ("causal", "reads.0.causal.delta"),
                ("resid sp", "reads.0.causal.effect.residual_spacings")),
               "reads.0.causal.verdict", "reads.0.name"),),
    "print": (((("margin", "margin.value"),), None, None),
              ((("mean", "stats.mean"), ("abs_max", "stats.abs_max")), None, None)),
    "explain": (((("margin", "explain.native_margin"), ("float64", "explain.float64_margin")),
                 None, None),),
    "compare": (((("A", "effects.A"), ("B", "effects.B"), ("A+B", "effects.A+B")),
                 None, None),),
}


def _is_claim(node: Any) -> bool:
    return (isinstance(node, dict) and "authority" in node and "value" in node
            and isinstance(node["value"], numbers.Real) and not isinstance(node["value"], bool))


def list_keys(node: Any) -> list[str] | None:
    """The name keys of a list (``NAME_KEYS``): one distinct text per item, or None when the
    list is walked by index (some item lacks the key, or two items share a name)."""
    if not isinstance(node, (list, tuple)) or not node:
        return None
    for field in NAME_KEYS:
        names = [item.get(field) if isinstance(item, dict) else None for item in node]
        if all(isinstance(n, str) and n and "." not in n for n in names) and \
                len(set(names)) == len(names):
            return names
    return None


def _items(node: Any, positional: bool) -> list[tuple[str, Any]]:
    keys = None if positional else list_keys(node)
    if keys is None:
        return [(str(i), item) for i, item in enumerate(node)]
    return list(zip(keys, node))


def claims(result: Any, *, positional: bool = False) -> dict[str, tuple[float, str]]:
    """Every labelled number of ``result``: dotted path -> (value, authority), in order. Lists
    of named items are walked by name (``NAME_KEYS``) unless ``positional``."""
    out: dict[str, tuple[float, str]] = {}

    def walk(node: Any, path: str) -> None:
        if _is_claim(node):
            out[path] = (float(node["value"]), str(node["authority"]))
            return
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for key, item in _items(node, positional):
                walk(item, f"{path}.{key}" if path else key)

    walk(result, "")
    return out


def verdicts(result: Any, *, positional: bool = False) -> dict[str, str]:
    """Every verdict/state text of ``result``: dotted path -> value, in order (named lists as
    in ``claims``)."""
    out: dict[str, str] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                here = f"{path}.{key}" if path else str(key)
                if key in VERDICT_KEYS and isinstance(item, str):
                    out[here] = item
                else:
                    walk(item, here)
        elif isinstance(node, (list, tuple)):
            for key, item in _items(node, positional):
                walk(item, f"{path}.{key}" if path else key)

    walk(result, "")
    return out


def _sign(value: float) -> str:
    return "+" if value > 0 else "-" if value < 0 else "0"


def _stats(values: list[float], authority: str) -> dict[str, Any]:
    signs: dict[str, int] = {}
    for value in values:
        signs[_sign(value)] = signs.get(_sign(value), 0) + 1
    sign = max(("+", "-", "0"), key=lambda s: (signs.get(s, 0), s == "+", s == "-"))
    return {"n": len(values), "mean": num(math.fsum(values) / len(values), authority),
            "min": num(min(values), authority), "max": num(max(values), authority),
            "agree": {"sign": sign, "count": signs.get(sign, 0)}}


def numeric_stat(path: str, items: list[tuple[float, str]]) -> dict[str, Any]:
    """The statistics of one path; per authority label when the inputs carry several."""
    labels = sorted({authority for _, authority in items})
    if len(labels) == 1:
        return {"path": path, **_stats([v for v, _ in items], labels[0])}
    return {"path": path, "n": len(items), "mixed": True, "by_authority": {
        label: _stats([v for v, a in items if a == label], label) for label in labels}}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Statistics over the paths every result has (see the module docstring)."""
    if not results:
        return {"numeric": [], "verdicts": [], "partial": 0, "n": 0}
    flat = [claims(result) for result in results]
    texts = [verdicts(result) for result in results]
    common = [path for path in flat[0] if all(path in other for other in flat[1:])]
    every = set().union(*flat)
    numeric = [numeric_stat(path, [found[path] for found in flat]) for path in common]
    shared = [path for path in texts[0] if all(path in other for other in texts[1:])]
    counted = []
    for path in shared:
        counts: dict[str, int] = {}
        for found in texts:
            counts[found[path]] = counts.get(found[path], 0) + 1
        counted.append({"path": path, "counts": [
            {"label": label, "count": count}
            for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]})
    every_text = set().union(*texts)
    return {"numeric": numeric, "verdicts": counted, "n": len(results),
            "partial": (len(every) - len(common)) + (len(every_text) - len(shared))}


def leaf(result: Any, path: str) -> Any:
    """The value at a dotted path (list items by index, or by name for a named list as in
    ``claims``); ``KeyError`` when absent. A labelled number gives its ``value``."""
    node = result
    for key in path.split("."):
        if isinstance(node, (list, tuple)):
            if key.lstrip("-").isdigit():
                node = node[int(key)]
            else:
                names = list_keys(node)
                if names is None or key not in names:
                    raise KeyError(key)
                node = node[names.index(key)]
        else:
            node = node[key]
    return node["value"] if _is_claim(node) else node


def _value_order(key: Any) -> tuple:
    """A total order over JSON leaves (numbers before texts before the rest), so a tie is
    broken by value, whatever the item order."""
    if isinstance(key, bool):
        return (1, int(key), "")
    if isinstance(key, numbers.Real):
        return (0, float(key), "")
    if isinstance(key, str):
        return (2, 0.0, key)
    return (3, 0.0, repr(key))


def modes(results: list[dict[str, Any]], paths: list[str], *,
          ties: str = "first") -> list[dict[str, Any]]:
    """The most common value of each dotted path over the results that have it (categorical
    fields: a layer, a KV head, a best token): ``mode``, its ``count``, ``n`` (results with the
    path) and every value's count. ``ties``: ``first`` breaks a tie by first appearance (depends
    on item order, AUDIT-B1); ``value`` by the smallest value (order-independent). A tie is
    always reported (``tie``: the tied values)."""
    if ties not in ("first", "value"):
        raise ValueError("ties is first or value")
    out = []
    for path in paths:
        counts: dict[Any, int] = {}
        order: list[Any] = []
        for result in results:
            try:
                value = leaf(result, path)
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            key = json_key(value)
            if key not in counts:
                order.append(key)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            out.append({"path": path, "n": 0, "mode": None, "count": 0, "counts": []})
            continue
        top = max(counts.values())
        tied = [k for k in order if counts[k] == top]
        if ties == "value":
            tied.sort(key=_value_order)
            mode = tied[0]
            listing = sorted(order, key=lambda k: (-counts[k], _value_order(k)))
        else:
            mode = tied[0]
            listing = sorted(order, key=lambda k: (-counts[k], order.index(k)))
        entry = {"path": path, "n": sum(counts.values()), "mode": mode, "count": counts[mode],
                 "counts": [{"value": k, "count": counts[k]} for k in listing]}
        if len(tied) > 1:
            entry["tie"] = {"values": tied, "broken_by": ties}
        out.append(entry)
    return out


def summarize_by(results: list[dict[str, Any]], switch: str) -> dict[str, Any]:
    """``summarize`` per value of the label at the dotted path ``switch`` (a group-by): groups
    in value order, each with its ``n``; results without the label are counted in
    ``unlabelled``. Statistics carry the inputs' labels, per group."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    missing = 0
    for result in results:
        try:
            value = leaf(result, switch)
        except (KeyError, IndexError, ValueError, TypeError):
            missing += 1
            continue
        groups.setdefault(json_key(value), []).append(result)
    return {"switch": switch, "unlabelled": missing,
            "groups": [{"value": key, "n": len(found), "summary": summarize(found)}
                       for key, found in sorted(groups.items(),
                                                key=lambda kv: _value_order(kv[0]))]}


def json_key(value: Any) -> Any:
    """A hashable form of a JSON leaf (lists become tuples)."""
    if isinstance(value, list):
        return tuple(json_key(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, json_key(v)) for k, v in value.items()))
    return value


def key_columns(verb: str, result: dict[str, Any]) -> dict[str, Any]:
    """The table cells of one item: ``columns`` (name -> labelled number, or the field's state
    and reason), ``verdict`` and ``label`` texts when the verb has them."""
    found = claims(result)
    texts = verdicts(result)
    choices = KEY_COLUMNS.get(verb, ())
    spec = next((c for c in choices if c[0][0][1] in found), choices[0] if choices else None)
    if spec is None:
        columns = tuple((path.rsplit(".", 1)[-1], path) for path in list(found)[:2])
        verdict = next((p for p in texts if p.endswith("verdict")), None)
        label = None
    else:
        columns, verdict, label = spec
    out: dict[str, Any] = {"columns": {}}
    for name, path in columns:
        if path in found:
            value, authority = found[path]
            out["columns"][name] = num(value, authority)
        else:
            out["columns"][name] = {"state": "unresolved", "reason": f"no number at {path}"}
    if verdict is not None and verdict in texts:
        out["verdict"] = texts[verdict]
    if label is not None:
        node: Any = result
        for part in label.split("."):
            node = (node[int(part)] if isinstance(node, list) and part.isdigit()
                    and int(part) < len(node) else
                    node.get(part) if isinstance(node, dict) else None)
        if isinstance(node, str):
            out["label"] = node
    out["paths"] = [path for _, path in columns]
    return out
