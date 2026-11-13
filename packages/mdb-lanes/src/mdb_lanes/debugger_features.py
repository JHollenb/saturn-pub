"""Small, dependency-free helpers used by the interactive debugger.

The live debugger deliberately keeps these helpers boring.  They operate on
typed summaries and JSON metadata, never on raw pointers or arbitrary worker
Python.  Native adapters may return richer summaries, but the fallback paths
remain useful for fake workers and lightweight instrumented runtimes.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Mapping, Sequence
from typing import Any


class DebuggerFeatureError(ValueError):
    """A bounded debugger feature could not be evaluated."""


def _numbers(value: Any, *, limit: int = 100_000) -> list[float]:
    values: list[float] = []

    def visit(item: Any) -> None:
        if len(values) >= limit:
            return
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            if math.isfinite(float(item)):
                values.append(float(item))
            return
        detach = getattr(item, "detach", None)
        if callable(detach):
            try:
                detached = detach()
                cpu = getattr(detached, "cpu", None)
                converted = cpu() if callable(cpu) else detached
                tolist = getattr(converted, "tolist", None)
                if callable(tolist):
                    visit(tolist())
                    return
            except Exception:  # noqa: BLE001 - summaries are best-effort
                pass
        for method_name in ("tolist", "as_list"):
            method = getattr(item, method_name, None)
            if callable(method):
                try:
                    visit(method())
                    return
                except Exception:  # noqa: BLE001 - summaries are best-effort
                    pass
        if isinstance(item, Mapping):
            if "value" in item:
                visit(item["value"])
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)

    visit(value)
    return values


def value_stats(value: Any) -> dict[str, Any]:
    """Return compact numeric statistics for a JSON/native summary value."""

    numbers = _numbers(value)
    shape: list[int] = []
    cursor = value
    while isinstance(cursor, Sequence) and not isinstance(cursor, (str, bytes, bytearray)):
        shape.append(len(cursor))
        if not cursor:
            break
        cursor = cursor[0]
    result: dict[str, Any] = {
        "count": len(numbers),
        "shape": shape,
        "finite": True,
    }
    if numbers:
        result.update(
            {
                "min": min(numbers),
                "max": max(numbers),
                "mean": sum(numbers) / len(numbers),
                "norm": math.sqrt(sum(item * item for item in numbers)),
            }
        )
    return result


def attach_stats(result: Mapping[str, Any], *, enabled: bool = True) -> dict[str, Any]:
    """Attach compact statistics when an inspect result carries a value."""

    payload = dict(result)
    if enabled and "value" in payload and "stats" not in payload:
        payload["stats"] = value_stats(payload["value"])
    return payload


_PREDICATES: dict[str, Any] = {
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _scalar(value: Any) -> float | str | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    numbers = _numbers(value)
    if len(numbers) == 1:
        return numbers[0]
    return None


def _metric(value: Any, name: str) -> float | str | bool | None:
    if name in {"value", "raw", "eq", "ne"}:
        return _scalar(value)
    numbers = _numbers(value)
    if name == "norm":
        return math.sqrt(sum(item * item for item in numbers)) if numbers else None
    if name == "mean":
        return sum(numbers) / len(numbers) if numbers else None
    if name == "min":
        return min(numbers) if numbers else None
    if name == "max":
        return max(numbers) if numbers else None
    if name == "count":
        return len(numbers)
    return None


def evaluate_predicate(
    predicate: str | None, value: Any, metadata: Mapping[str, Any] | None = None
) -> bool:
    """Evaluate one tiny, explicit watch predicate.

    Accepted forms are ``norm > 10``, ``value == 3``, ``gt 10``,
    ``abs_delta_gt 0.2``, and ``regex TEXT`` for string values.  This is not a
    Python expression evaluator.  Boundary conditions over event metadata are
    handled separately by the worker's breakpoint matcher.
    """

    if predicate is None or not predicate.strip():
        return True
    expression = predicate.strip()
    if len(expression) > 512:
        raise DebuggerFeatureError("watch predicate is too long")
    if expression.startswith("regex "):
        target = _scalar(value)
        if not isinstance(target, str):
            return False
        try:
            return re.search(expression[6:].strip(), target) is not None
        except re.error as exc:
            raise DebuggerFeatureError(f"invalid watch regex: {exc}") from exc
    match = re.fullmatch(
        r"(?:(?P<metric>[A-Za-z_][A-Za-z0-9_]*)\s*)?"
        r"(?P<op>abs_delta_gt|rel_delta_gt|gt|lt|ge|le|eq|ne|>=|<=|==|!=|>|<)"
        r"\s*(?P<expected>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|true|false|null|'.*?'|\".*?\")",
        expression,
    )
    if match is None:
        raise DebuggerFeatureError(
            "watch predicate must look like 'norm > 10', 'gt 0', 'eq 3', or 'regex TEXT'"
        )
    metric = match.group("metric") or "value"
    op = match.group("op")
    raw_expected = match.group("expected")
    if raw_expected in {"true", "false"}:
        expected: Any = raw_expected == "true"
    elif raw_expected == "null":
        expected = None
    elif raw_expected[:1] in {"'", '"'}:
        expected = raw_expected[1:-1]
    else:
        expected = float(raw_expected)
    observed = _metric(value, metric)
    if observed is None:
        return False
    if op == "abs_delta_gt":
        previous = (metadata or {}).get("previous")
        previous_scalar = _metric(previous, metric)
        return previous_scalar is not None and abs(
            float(observed) - float(previous_scalar)
        ) > float(expected)
    if op == "rel_delta_gt":
        previous = (metadata or {}).get("previous")
        previous_scalar = _metric(previous, metric)
        if previous_scalar in {None, 0}:
            return False
        return abs(float(observed) - float(previous_scalar)) / abs(float(previous_scalar)) > float(
            expected
        )
    comparator = _PREDICATES[op]
    try:
        return bool(comparator(observed, expected))
    except (TypeError, ValueError) as exc:
        raise DebuggerFeatureError(f"watch comparison failed: {exc}") from exc


def search_capabilities(query: str, capability: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Search boundary, address, struct, and slot metadata by human text."""

    needle = str(query or "").strip().lower()
    if not needle:
        return []
    results: list[dict[str, Any]] = []

    for boundary in capability.get("boundary_catalog", ()):
        if needle in repr(boundary).lower():
            results.append({"kind": "boundary", **dict(boundary)})
    for address in capability.get("address_specs", ()):
        if needle in repr(address).lower():
            results.append({"kind": "address", **dict(address)})
    for struct in capability.get("structs", ()):
        if needle in repr(struct).lower():
            results.append({"kind": "struct", **dict(struct)})
    for slot_key in ("inspectable_slots", "writable_slots"):
        for slot in capability.get(slot_key, ()):
            if needle in str(slot).lower():
                results.append({"kind": slot_key[:-1], "slot": slot})
    return results


def template_tree(capability: Mapping[str, Any], prefix: str | None = None) -> list[dict[str, Any]]:
    """Return address templates and their fields as a compact logical tree."""

    needle = (prefix or "").lower()
    rows: list[dict[str, Any]] = []
    for address in capability.get("address_specs", ()):
        path = str(address.get("path_template") or "")
        if (
            needle
            and needle not in path.lower()
            and needle not in str(address.get("name", "")).lower()
        ):
            continue
        rows.append(
            {
                "name": address.get("name"),
                "kind": address.get("kind"),
                "path_template": path,
                "readable": bool(address.get("readable", True)),
                "writable": bool(address.get("writable", False)),
                "fields": address.get("fields", {}),
                "struct_name": address.get("struct_name"),
                "boundaries": address.get("boundaries", []),
                "invalidates": address.get("invalidates", []),
            }
        )
    declared = {str(row.get("path_template")) for row in rows}
    writable = {str(item) for item in capability.get("writable_slots", ())}
    for slot in capability.get("inspectable_slots", ()):
        slot = str(slot)
        if slot in declared or (needle and needle not in slot.lower()):
            continue
        rows.append(
            {
                "name": slot,
                "kind": "slot",
                "path_template": slot,
                "readable": True,
                "writable": slot in writable,
                "fields": {},
            }
        )
    return rows


def provenance_for_path(path: str, capability: Mapping[str, Any]) -> dict[str, Any]:
    """Build a conservative ``why/xref`` record from declared metadata."""

    matches: list[dict[str, Any]] = []
    for address in capability.get("address_specs", ()):
        template = str(address.get("path_template") or "")
        if path == template or path.startswith(template.split("{")[0]):
            matches.append(dict(address))
    return {
        "path": path,
        "matches": matches,
        "model_identity": capability.get("model_identity", {}),
        "family": capability.get("family"),
        "backend": capability.get("backend"),
        "consumer": capability.get("consumer"),
        "state_schema_id": capability.get("state_schema_id"),
        "claim_boundary": "declared debugger metadata; native consumer remains authoritative",
    }


__all__ = [
    "DebuggerFeatureError",
    "attach_stats",
    "evaluate_predicate",
    "provenance_for_path",
    "search_capabilities",
    "template_tree",
    "value_stats",
]
