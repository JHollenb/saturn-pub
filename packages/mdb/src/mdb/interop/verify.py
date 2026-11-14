"""Re-measure another tool's claim natively in mdb.

A claim is an edit in mdb's language (what the other tool did, translated through
``mdb.interop.names``), a consumer margin and the change the tool reported. ``verify`` runs
mdb's ``whatif`` for it (sealed prediction first, then the native batch-1 stepped measurement,
then the P20 verdict) and sets the claim beside it. The claim is ``supplied``; mdb's numbers
keep their own labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Claim:
    target: str            # mdb address of the edit, e.g. "L22.A[h6]" or "cache.L22.v[kv1,row=11]"
    value: Any             # a number or "ORIGIN:ADDR" (an import or a named run)
    op: str = "replace"
    consumer: str = ""     # "M[a,b]"
    claimed: float | None = None
    tool: str = ""
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _num(entry: Any) -> float | None:
    if isinstance(entry, dict):
        entry = entry.get("value", entry.get("v"))
    try:
        return None if entry is None else float(entry)
    except (TypeError, ValueError):
        return None


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    """The numbers of a ``whatif`` report that a comparison needs."""
    prediction = report.get("prediction") or {}
    measured = report.get("measured") or {}
    effect = report.get("effect") or {}

    def inner(entry: Any) -> float | None:
        return _num(entry.get("value") if isinstance(entry, dict) and "value" in entry
                    and isinstance(entry["value"], dict) else entry)

    return {
        "linear": inner(prediction.get("linear")),
        "exact": inner(prediction.get("exact")),
        "native": _num(measured.get("delta")),
        "base": _num(measured.get("base")),
        "edited": _num(measured.get("edited")),
        "verdict": effect.get("verdict"),
        "floor": _num(effect.get("floor")),
    }


def verify(dbg: Any, run: Any, claims: Sequence[Claim]) -> list[dict[str, Any]]:
    """mdb's ``whatif`` for each claim on ``run`` (left where it was); one row per claim:
    the claim (``supplied``), mdb's sealed linear/exact predictions, the native change, the
    verdict and ``claim_minus_native``."""
    from mdb.ops import dispatch
    from mdb.ops.errors import OpError

    ctx = dbg.ops(run)
    rows = []
    for claim in claims:
        row: dict[str, Any] = {"target": claim.target, "value": claim.value, "op": claim.op,
                               "consumer": claim.consumer, "tool": claim.tool,
                               "claimed": claim.claimed, "claimed_authority": "supplied",
                               "note": claim.note, **claim.extra}
        try:
            report = dispatch(ctx, "whatif", {"target": claim.target, "value": claim.value,
                                              "op": claim.op, "consumer": claim.consumer})
        except OpError as error:
            row.update(state="refused", reason=str(error))
            rows.append(row)
            continue
        found = summarise(report)
        row.update(found, state="resolved", report=report)
        if claim.claimed is not None and found["native"] is not None:
            row["claim_minus_native"] = claim.claimed - found["native"]
        rows.append(row)
    return rows
