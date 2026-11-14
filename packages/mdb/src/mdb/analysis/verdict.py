"""Effect verdicts from the P20 accounting, not from a worst-case rounding bound.

P20 (chapter 07, recursive error) and chapter 09 section 1 give the exact accounting of a native
change: ``native delta = exact delta + residual``, with ``exact delta`` the float64 identity chain
on the retained operands (``derived-exact``) and ``residual = native - exact`` the arithmetic
discrepancy (``measured``). A verdict reads only these measured and derived-exact numbers:

* ``floor = max(|residual|, 2 x spacing)``, ``spacing`` the gap between adjacent values of the
  consumer's native dtype at the magnitude of the consumer (both logits of a margin, before and
  after the edit; the logit; the largest element of a vector). A one-step difference is not
  stable across cache-build procedures, hence two steps.
* ``effect`` (resolved) when ``|native| > floor`` and ``exact`` has the sign of ``native`` with
  ``|exact| > |residual|``; ``no effect above the noise floor`` (resolved) when ``|native| <=
  floor`` and ``|exact| <= floor``; otherwise ``unresolved`` with the numbers.
* No exact chain (the identity is refused, or the verb measures natively only): ``native``
  against ``2 x spacing`` alone, said so in ``basis``.

A vector consumer is judged at its element of largest ``|native delta|`` (the residual there,
the floor from the largest ``|residual|`` over all elements). Every number is ``measured`` (or
``derived-exact`` for the exact delta) with proof P20; no number here is a ``bound``.
"""

from __future__ import annotations

import math
from typing import Any

from .finite import EXACT, MEASURED, SUPPLIED

EFFECT = "effect"
NO_EFFECT = "no effect above the noise floor"
UNRESOLVED = "unresolved"
CRITERION = ("effect when |native Δ| > floor and the exact Δ has its sign with |exact Δ| > "
             "|residual|; no effect when |native Δ| <= floor and |exact Δ| <= floor; floor = "
             "max(|residual|, 2 x dtype spacing at the consumer's magnitude); P20: native Δ = "
             "exact Δ + residual")
NATIVE_ONLY = ("effect when |native Δ| > floor = 2 x dtype spacing at the consumer's magnitude, "
               "else no effect above the noise floor (no exact chain: native only)")


def _torch():
    import torch

    return torch


def spacing(magnitude: float, dtype: Any) -> float:
    """Gap between adjacent ``dtype`` values at ``|magnitude|`` (one ulp; bf16 near 20: 0.125)."""
    info = _torch().finfo(dtype)
    size = max(abs(float(magnitude)), float(info.tiny))
    return 2.0 ** math.floor(math.log2(size)) * float(info.eps)


def magnitude(consumer: Any, *tensors: Any) -> float:
    """The consumer's magnitude over native tensors read at its stop: the two logits of a margin
    (``consumer.a``/``.b``), the logit (``.i``), else the largest element."""
    out = 0.0
    for tensor in tensors:
        flat = tensor.reshape(-1).detach().abs()
        if hasattr(consumer, "a") and hasattr(consumer, "b"):
            out = max(out, float(flat[consumer.a].float()), float(flat[consumer.b].float()))
        elif hasattr(consumer, "i"):
            out = max(out, float(flat[consumer.i].float()))
        else:
            out = max(out, float(flat.float().max()))
    return out


def _num(value: float, authority: str) -> dict[str, Any]:
    return {"value": float(value), "authority": authority, "proof": "P20"}


def decide(native: Any, exact: Any, residual: Any, size: float, dtype: Any,
           why_native_only: str | None = None) -> dict[str, Any]:
    """The verdict on one change. ``native``, ``exact``, ``residual``: float64 tensors (scalar or
    vector; ``exact``/``residual`` None when no exact chain exists, ``why_native_only`` saying
    why); ``size`` the consumer magnitude; ``dtype`` its native dtype."""
    torch = _torch()
    native = torch.as_tensor(native, dtype=torch.float64).reshape(-1)
    k = int(native.abs().argmax())
    n = float(native[k])
    step = spacing(size, dtype)
    out: dict[str, Any] = {"native": _num(n, MEASURED),
                           "spacing": _num(step, MEASURED),
                           "dtype": str(dtype).removeprefix("torch.")}
    if native.numel() > 1:
        out["index"] = k
    if exact is None or residual is None:
        floor = 2 * step
        out.update(floor=_num(floor, MEASURED), criterion=NATIVE_ONLY,
                   basis=f"native only: {why_native_only or 'no exact chain'}")
        out["verdict"] = EFFECT if abs(n) > floor else NO_EFFECT
        out["state"] = "resolved"
        return out
    exact = torch.as_tensor(exact, dtype=torch.float64).reshape(-1)
    residual = torch.as_tensor(residual, dtype=torch.float64).reshape(-1)
    e, r = float(exact[k]), float(residual[k])
    floor = max(float(residual.abs().max()), 2 * step)
    out.update(exact=_num(e, EXACT), residual=_num(r, MEASURED), floor=_num(floor, MEASURED),
               criterion=CRITERION, basis="P20 accounting")
    if abs(n) > floor:
        if e != 0 and (e > 0) == (n > 0) and abs(e) > abs(r):
            out.update(verdict=EFFECT, state="resolved")
        elif e == 0 or (e > 0) != (n > 0):
            out.update(verdict=UNRESOLVED, state="unresolved",
                       reason=f"native Δ {n:.5g} is above the floor {floor:.5g} but the exact "
                              f"Δ {e:.5g} has the other sign")
        else:
            out.update(verdict=UNRESOLVED, state="unresolved",
                       reason=f"native Δ {n:.5g} is above the floor {floor:.5g} but |exact Δ| "
                              f"{abs(e):.5g} <= |residual| {abs(r):.5g}")
    elif abs(e) <= floor:
        out.update(verdict=NO_EFFECT, state="resolved")
    else:
        out.update(verdict=UNRESOLVED, state="unresolved",
                   reason=f"native Δ {n:.5g} is within the floor {floor:.5g} but the exact Δ "
                          f"{e:.5g} is not")
    return out


CONFIRMED = "confirmed"
NOT_CONFIRMED = "not confirmed"
# The one supplied constant of `confirm`: the floor in spacings between the native value and the
# emulated prediction (`forecast.EmulatedRowModel`, the run's dtype rounding at every eager-HF
# rounding point). Above it `forecast.verify` names where emulation and native part (not
# confirmed, or unresolved for rounding drift); it does not widen.
FLOOR_SPACINGS = 1.0
FLOOR_SUPPORT = ("largest |native - emulated| in spacings of the consumer at every stop from the "
                 "sealing stop to the logits: 0 (bit-exact) on tiny qwen2/qwen3/llama/mistral bf16 "
                 "and fp32, 2-28 layers, 3 prompts (CPU), and on Qwen2.5-1.5B bf16 (28 layers) and "
                 "fp32 (2 layers), 3 prompts, sealed at L0.h and L14.h (CUDA)")
CONFIRM_CRITERION = ("confirmed when every element's |native - emulated| <= floor = 1 (supplied) x "
                     "dtype spacing at the magnitude of the consumer (a margin: the sum of its two "
                     "logits' spacings); above it, not confirmed unless every stage from where "
                     "emulation and native part to the target is the native step (then "
                     "unresolved: rounding drift); P20: residual = native - emulated")


def confirm(residual: Any, size: float, dtype: Any, *, partner: float | None = None) -> dict[str, Any]:
    """The verdict on one forecast: confirmed when every element's measured residual (native -
    emulated prediction; scalar or vector) is within ``floor = FLOOR_SPACINGS x step``, else not
    confirmed (``forecast.verify`` then says where emulation and native part).

    ``step`` is the spacing of ``dtype`` at ``size``, the consumer's magnitude (a margin, the
    difference of two rounded logits: ``size`` and ``partner``, the step is the sum of the two
    spacings). A non-floating ``dtype`` (token ids) has no spacing: the floor is 0. ``residual``
    is the largest ``|.|``; ``residual_spacings`` (``measured``) says how close to the floor it
    came."""
    torch = _torch()
    gap = torch.as_tensor(residual, dtype=torch.float64).abs().reshape(-1)
    floating = dtype is not None and torch.empty(0, dtype=dtype).is_floating_point()
    step = 0.0
    if floating:
        step = spacing(size, dtype) + (0.0 if partner is None else spacing(partner, dtype))
    floor = FLOOR_SPACINGS * step
    worst = float(gap.max())
    out = {"verdict": CONFIRMED if worst <= floor else NOT_CONFIRMED, "state": "resolved",
           "residual": _num(worst, MEASURED), "spacing": _num(step, MEASURED),
           "floor": _num(floor, SUPPLIED),
           "floor_spacings": {**_num(FLOOR_SPACINGS, SUPPLIED), "support": FLOOR_SUPPORT},
           "criterion": CONFIRM_CRITERION, "dtype": str(dtype).removeprefix("torch.")}
    if step:
        out["residual_spacings"] = _num(worst / step, MEASURED)
    if gap.numel() > 1:
        out["index"] = int(gap.argmax())
    return out


__all__ = ["CONFIRMED", "CONFIRM_CRITERION", "CRITERION", "EFFECT", "NATIVE_ONLY", "NOT_CONFIRMED",
           "NO_EFFECT", "UNRESOLVED", "FLOOR_SPACINGS", "confirm", "decide", "magnitude", "spacing"]
