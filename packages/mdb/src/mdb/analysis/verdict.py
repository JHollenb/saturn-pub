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

from .finite import BOUND, EXACT, MEASURED, SUPPLIED

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
# The one supplied constant of `confirm`: the coefficient of the random-walk count. Measured on tiny
# qwen2/qwen3 (bf16 and fp32; 2, 6, 12, 28 layers; vector ports, margins, 12264 correct forecasts):
# the largest residual / (sqrt(n) x spacing) was 0.86 (bf16 vector), 2.18 (fp32 vector), 1.91 (bf16
# margin), 1.80 (fp32 margin), n the rounded steps between seal and target.
COEFFICIENT = 3.0
COEFFICIENT_SUPPORT = ("max residual/(sqrt(n) x spacing) over 12264 correct tiny-model forecasts: "
                       "vector 0.86 bf16, 2.18 fp32; margin 1.91 bf16, 1.80 fp32")
CONFIRM_CRITERION = ("confirmed when every element's |native - predicted| <= min(floor, its sealed "
                     "worst-case bound where finite); floor = 3 x sqrt(n) x dtype spacing at the "
                     "magnitude of the consumer (a margin: the sum of its two logits' spacings), "
                     "n the rounded steps between the sealed operands and the target, 3 supplied; "
                     "not confirmed otherwise; P20: residual = native - predicted")


def confirm(residual: Any, size: float, dtype: Any, *, count: int = 1, partner: float | None = None,
            limit: Any = None) -> dict[str, Any]:
    """The verdict on one forecast: confirmed when every element's measured residual (native -
    predicted; scalar or vector) is within ``min(floor, limit_i)``.

    ``floor = COEFFICIENT x sqrt(count) x step``: ``count`` the rounded operations between the
    sealed operands and the target (their errors add as a random walk), ``step`` the spacing of
    ``dtype`` at ``size``, the consumer's magnitude (a margin, the difference of two rounded
    logits: ``size`` and ``partner``, the step is the sum of the two spacings). ``limit``, the
    sealed per-element worst-case bound (finite entries only; ``derived``, premises on the
    forecast), can only tighten the floor at its own element, never loosen it. A non-floating
    ``dtype`` (token ids) has no spacing: the floor is 0. Always ``resolved``. ``residual`` is the
    largest ``|.|``; ``residual_spacings`` (``measured``) says how close to the floor it came."""
    torch = _torch()
    gap = torch.as_tensor(residual, dtype=torch.float64).abs().reshape(-1)
    floating = dtype is not None and torch.empty(0, dtype=dtype).is_floating_point()
    step = 0.0
    if floating:
        step = spacing(size, dtype) + (0.0 if partner is None else spacing(partner, dtype))
    steps = COEFFICIENT * math.sqrt(max(int(count), 1))
    floor = steps * step
    bound = torch.full_like(gap, floor)
    if limit is not None:
        lim = torch.as_tensor(limit, dtype=torch.float64).reshape(-1)
        bound = torch.where(torch.isfinite(lim), torch.minimum(lim, bound), bound)
    worst = float(gap.max())
    k = int((gap - bound).argmax())
    confirmed = bool((gap <= bound).all())
    out = {"verdict": CONFIRMED if confirmed else NOT_CONFIRMED, "state": "resolved",
           "residual": _num(worst, MEASURED), "spacing": _num(step, MEASURED),
           "floor": _num(floor, SUPPLIED), "floor_spacings": _num(steps, SUPPLIED),
           "rounded_steps": _num(max(int(count), 1), EXACT),
           "coefficient": {"value": COEFFICIENT, "authority": SUPPLIED, "support": COEFFICIENT_SUPPORT},
           "criterion": CONFIRM_CRITERION, "dtype": str(dtype).removeprefix("torch.")}
    if step:
        out["residual_spacings"] = _num(worst / step, MEASURED)
    if gap.numel() > 1:
        out["index"] = k
    if float(bound[k]) < floor:
        out["element_limit"] = {"value": float(bound[k]), "authority": BOUND, "proof": "P20",
                                "index": k, "basis": "the sealed worst-case bound at this element"}
    return out


__all__ = ["CONFIRMED", "CONFIRM_CRITERION", "CRITERION", "EFFECT", "NATIVE_ONLY", "NOT_CONFIRMED",
           "NO_EFFECT", "UNRESOLVED", "COEFFICIENT", "confirm", "decide", "magnitude", "spacing"]
