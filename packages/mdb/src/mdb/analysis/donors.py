"""Donor counts on causal nulls (binary ground truth 2026-11-16, SCORES.md lessons 2 and 8).

On real binaries, where the answer key is exact, a single resample donor wrongly said "no effect"
on 41-71% of live cells (mean over the 32 single-donor choices), and among donors whose value
differed from the base's only 0-3.5% did (MEASURED, SCORES.md run 2(d)). Almost all single-donor
nulls were identity interventions: the donor carried the base's own value. A null therefore says
what it rests on:

* every ``no effect`` carries ``donors = {n, informative}``: how many donors were measured and in
  how many the patched value differed from the base's (bitwise). ``informative`` is ``None`` when
  the caller could not compare the two values (a supplied number, a vector consumer only);
* ``label`` is the text a reader sees: ``no effect (1 donor)`` for one donor, ``no effect (16
  donors, 12 informative)`` for several, ``no effect (1 donor, not informative: the donor value
  equals the base)`` for an identity patch. A bare ``no effect`` is never printed.

Multi-donor mode (``whatif``, ``compare``, ``liveness`` and ``reads`` take ``donors=``; ``sweep``
takes several origins): one native measurement per donor, then ``pooled``: ``effect`` when at least one
donor's verdict is ``effect`` (the x86 method's liveness rule: live if >= 1 donor changes y),
``no effect`` when every donor was measured and none had one, else ``unresolved``. ``n_effect``
and ``n_informative`` are reported beside it, so "1 of 16" and "16 of 16" read differently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .finite import MEASURED, SUPPLIED, num

EFFECT, NO_EFFECT, UNRESOLVED = "effect", "no effect", "unresolved"
RULE = ("pooled over donors: effect when >= 1 donor has an effect; no effect when every donor "
        "was measured and none had; else unresolved. informative = the donor's patched value "
        "differs (bitwise) from the base value at the patched site")


def _torch():
    import torch

    return torch


def informative(base: Any, donor: Any) -> bool | None:
    """Whether the donor's value differs from the base's (bitwise, in the stored dtype); None
    when either is missing or the shapes differ."""
    if base is None or donor is None:
        return None
    torch = _torch()
    base, donor = torch.as_tensor(base), torch.as_tensor(donor)
    if base.shape != donor.shape and base.numel() != donor.numel():
        return None
    return not torch.equal(base.reshape(-1).to(donor.dtype).cpu(), donor.reshape(-1).cpu())


def label(verdict: str, n: int = 1, k: int | None = None) -> str:
    """The text of a verdict with its donor count (module docstring); effects and unresolved
    verdicts of one donor are printed as they are."""
    if verdict != NO_EFFECT and n == 1:
        return verdict
    if n == 1:
        if k == 0:
            return f"{verdict} (1 donor, not informative: the donor value equals the base)"
        return f"{verdict} (1 donor)"
    shown = "informative not measured" if k is None else f"{k} informative"
    return f"{verdict} ({n} donors, {shown})"


def stamp(effect: dict[str, Any], n: int = 1, k: int | None = None) -> dict[str, Any]:
    """Attach ``donors`` and ``label`` to a verdict dict (``verdict.decide``'s, or any dict with
    a ``verdict``); returns it."""
    effect["donors"] = {"n": num(int(n), SUPPLIED),
                        "informative": (num(int(k), MEASURED) if k is not None else
                                        {"compared": False,
                                         "reason": "the donor value was not compared with the "
                                                   "base value"})}
    effect["label"] = label(effect["verdict"], n, k)
    return effect


def pooled(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-donor verdicts. Each row: ``verdict`` and ``informative`` (bool or None), and
    optionally ``path_diverged`` (bool). Returns the pooled verdict with its counts and label."""
    n = len(rows)
    verdicts = [r["verdict"] for r in rows]
    known = [r.get("informative") for r in rows]
    k = None if any(x is None for x in known) else sum(bool(x) for x in known)
    n_effect = sum(v == EFFECT for v in verdicts)
    if n_effect:
        verdict = EFFECT
    elif n and all(v == NO_EFFECT for v in verdicts):
        verdict = NO_EFFECT
    else:
        verdict = UNRESOLVED
    out = {"verdict": verdict, "n_donors": n, "n_informative": k, "n_effect": n_effect,
           "n_unresolved": sum(v not in (EFFECT, NO_EFFECT) for v in verdicts),
           "rule": RULE}
    if any("path_diverged" in r for r in rows):
        out["n_path_diverged"] = sum(bool(r.get("path_diverged")) for r in rows)
    out["label"] = (f"effect ({n_effect} of {n} donors, {k} informative)" if verdict == EFFECT
                    and n > 1 else label(verdict, n, k))
    return out


def _value(donor: Any) -> tuple[Any, Any]:
    """``(value, origin pointer)`` of a donor: a ``ValueRef`` (pool or run), a tensor or a
    number."""
    if hasattr(donor, "value") and hasattr(donor, "pointer"):
        return donor.value, dict(donor.pointer)
    return donor, None


def whatif(run: Any, address: str, donors: Sequence[Any], consumer: Any, *, op: str = "replace",
           mask: Any = None, selection: str | None = None, path: bool = False) -> dict[str, Any]:
    """``propagate.whatif`` once per donor value at ``address`` (``ValueRef`` from
    ``Pool.donors`` or ``Run.ref``, tensor or number); each verdict keeps its own P20 accounting
    and donor stamp; the pooled verdict says how many donors were informative. ``path=True``
    also runs ``paths.paired`` per donor (total and frozen-selection effects)."""
    from .propagate import Edit
    from .propagate import whatif as one

    rows, per = [], []
    for i, donor in enumerate(donors):
        value, origin = _value(donor)
        result = one(run, Edit(address, value, op, mask=mask, selection=selection,
                               source=f"donor {i}", origin=origin), consumer)
        effect = result.report["effect"]
        k = (effect.get("donors") or {}).get("informative", {})
        row = {"verdict": effect["verdict"], "informative": (bool(k["value"]) if "value" in k
                                                             else None)}
        entry = {"donor": i, "verdict": effect["verdict"], "label": effect.get("label"),
                 "informative": row["informative"], "delta": result.report["measured"]["delta"],
                 "record": result.report["record"], "origin": origin}
        if path:
            from .paths import paired

            report = paired(run, [(address, value, op, mask)], consumer)
            row["path_diverged"] = report["path"]["diverged"]
            entry["path"] = report
        rows.append(row)
        per.append(entry)
    return {"address": address, "consumer": getattr(consumer, "text", str(consumer)),
            **pooled(rows), "per_donor": per}


def compare(run: Any, sides: Callable[[Any], tuple[Any, Any]], donors: Sequence[Any],
            consumer: Any) -> dict[str, Any]:
    """``compare.compare`` once per donor: ``sides(donor)`` builds the two ``compare.Side``s for
    that donor. Each column (A, B, A+B) is pooled over donors."""
    from .compare import compare as one

    columns: dict[str, list[dict[str, Any]]] = {}
    per = []
    for i, donor in enumerate(donors):
        a, b = sides(donor)
        report = one(run, a, b, consumer)["report"]
        per.append({"donor": i, "verdicts": {k: v.get("label", v["verdict"])
                                             for k, v in report["verdicts"].items()}})
        for name, v in report["verdicts"].items():
            k = (v.get("donors") or {}).get("informative", {})
            columns.setdefault(name, []).append(
                {"verdict": v["verdict"], "informative": bool(k["value"]) if "value" in k
                 else None})
    return {"consumer": getattr(consumer, "text", str(consumer)),
            "columns": {name: pooled(rows) for name, rows in columns.items()}, "per_donor": per}


__all__ = ["EFFECT", "NO_EFFECT", "RULE", "UNRESOLVED", "compare", "informative", "label",
           "pooled", "stamp", "whatif"]
