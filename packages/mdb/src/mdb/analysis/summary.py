"""Statistics over the results of one op run on many prompts (``each``). Pure Python.

``claims(result)`` flattens an op result into its labelled numbers: ``path -> (value,
authority)`` for every ``{"value": number, "authority": ...}`` mapping (``path`` is the dotted
key path, list items by index: ``reads.0.mass``), and ``verdicts(result)`` its verdict and
state fields (``path -> text`` for string leaves under the keys ``verdict`` and ``state``).

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


def claims(result: Any) -> dict[str, tuple[float, str]]:
    """Every labelled number of ``result``: dotted path -> (value, authority), in order."""
    out: dict[str, tuple[float, str]] = {}

    def walk(node: Any, path: str) -> None:
        if _is_claim(node):
            out[path] = (float(node["value"]), str(node["authority"]))
            return
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                walk(item, f"{path}.{i}" if path else str(i))

    walk(result, "")
    return out


def verdicts(result: Any) -> dict[str, str]:
    """Every verdict/state text of ``result``: dotted path -> value, in order."""
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
            for i, item in enumerate(node):
                walk(item, f"{path}.{i}" if path else str(i))

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
