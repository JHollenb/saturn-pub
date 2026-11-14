"""Effect verdicts from the P20 accounting against an emulated prediction, not a noise floor.

P20 (chapter 07, recursive error) and chapter 09 section 1 give the exact accounting of a native
change: ``native delta = exact delta + rounding``, with ``exact delta`` the float64 identity chain
on the retained operands (``derived-exact``). The emulated chains (``forecast.EmulatedRowModel``:
the run's dtype rounding at every point where eager HF rounds, the backend's own kernels) split the
rounding into ``rounding effect = emulated delta - exact delta`` (``derived-exact`` under the
emulation premises) and ``residual = native delta - emulated delta`` (``measured``), which is 0
when the emulation holds (measured bit-exact on tiny models and Qwen2.5-1.5B, M42/M51). A
verdict reads:

* ``floor = FLOOR_SPACINGS x spacing`` (``FLOOR_SPACINGS`` = 1, ``supplied``, the one constant
  ``confirm`` uses too), ``spacing`` the gap between adjacent values of the consumer's dtype at
  the consumer's magnitude (both logits of a margin, before and after the edit; the logit; the
  largest element of a vector).
* ``effect`` (resolved) when every element's ``|native - emulated|`` is within the floor, the
  native delta is not zero and the emulated delta has its sign; ``no effect`` (resolved) when
  they agree and both are zero: the dtype shows no change at the consumer. Within the floor but
  with different signs (or one of them zero) ``unresolved``: a floor of one spacing cannot decide
  a change no bigger than itself (M59). ``size_spacings`` is ``|native delta|`` in spacings and
  ``rests_on`` says what an effect rests on. The exact delta is reported either way, as a fact
  beside the verdict (``note``: an exact change the dtype cannot show; a native change that is
  the dtype's rounding).
* beyond the floor: what the caller found where emulation and native part (``parting``: ``not
  confirmed`` when a stage does not reproduce its native value from its native operands within
  its identity's bound, else ``unresolved``), or ``unresolved`` with the numbers.
* No emulated chain (the prediction is refused, or the verb measures natively only, as
  ``compare``): ``effect`` when the edited and base native values differ, ``no effect`` when they
  are equal; ``basis`` says so.

A vector consumer is reported at its element of largest ``|native delta|``.
"""

from __future__ import annotations

import math
from typing import Any

from .finite import EXACT, MEASURED, SUPPLIED

EFFECT = "effect"
NO_EFFECT = "no effect"
UNRESOLVED = "unresolved"
CRITERION = ("effect when native Δ and emulated Δ agree within the floor (1 x dtype spacing at "
             "the consumer, supplied) and on the sign, and native Δ != 0; no effect when they "
             "agree and both are 0; within the floor but not on the sign unresolved; beyond the "
             "floor not confirmed or unresolved by where emulation and native part; P20: native "
             "Δ = exact Δ + rounding effect (emulated - exact) + residual (native - emulated)")
NATIVE_ONLY = ("effect when the edited and base native values differ (native Δ != 0), no effect "
               "when they are equal (no prediction: native only)")


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


def decide(native: Any, emulated: Any, size: float, dtype: Any, *, exact: Any = None,
           why_native_only: str | None = None, parting: dict[str, Any] | None = None
           ) -> dict[str, Any]:
    """The verdict on one change. ``native`` (measured), ``emulated`` (the emulated chains'
    delta, None when there is none, ``why_native_only`` saying why) and ``exact`` (the float64
    chain's delta, optional, reported): float64 tensors, scalar or vector; ``size`` the
    consumer's magnitude; ``dtype`` its native dtype; ``parting``: the caller's finding where
    emulation and native part (``verdict``, ``state``, ``reason`` and its details), used when
    they disagree beyond the floor."""
    torch = _torch()
    native = torch.as_tensor(native, dtype=torch.float64).reshape(-1)
    k = int(native.abs().argmax())
    n = float(native[k])
    step = spacing(size, dtype)
    floor = FLOOR_SPACINGS * step
    out: dict[str, Any] = {"native": _num(n, MEASURED), "spacing": _num(step, MEASURED),
                           "floor": _num(floor, SUPPLIED),
                           "floor_spacings": _num(FLOOR_SPACINGS, SUPPLIED),
                           "dtype": str(dtype).removeprefix("torch.")}
    if native.numel() > 1:
        out["index"] = k
    changed = bool((native != 0).any())
    if exact is not None:
        exact = torch.as_tensor(exact, dtype=torch.float64).reshape(-1)
        out["exact"] = _num(float(exact[k]), EXACT)
    if emulated is None:
        out.update(criterion=NATIVE_ONLY,
                   basis=f"native only: {why_native_only or 'no emulated chain'}",
                   verdict=EFFECT if changed else NO_EFFECT, state="resolved")
        return out
    emulated = torch.as_tensor(emulated, dtype=torch.float64).reshape(-1)
    residual = native - emulated
    worst = float(residual.abs().max())
    out.update(emulated=_num(float(emulated[k]), EXACT), residual=_num(float(residual[k]), MEASURED),
               residual_spacings=_num(worst / step, MEASURED), criterion=CRITERION,
               basis="P20: native against the emulated delta")
    if exact is not None:
        out["rounding_effect"] = _num(float(emulated[k] - exact[k]), EXACT)
    out["size_spacings"] = _num(abs(n) / step, MEASURED)
    if worst > floor:
        found = parting or {"verdict": UNRESOLVED, "state": "unresolved",
                            "reason": f"native and emulated Δ part by {worst / step:.3g} "
                                      "spacings"}
        out.update(found)
        return out
    m = float(emulated[k])
    if (n == 0) != (m == 0) or n * m < 0:
        # within the floor, but the two disagree on whether (or which way) the consumer moved:
        # the floor cannot decide a change no bigger than itself (M59)
        out.update(verdict=UNRESOLVED, state="unresolved",
                   reason=f"native Δ {n:.5g} and emulated Δ {m:.5g} agree within the floor "
                          f"({worst / step:.2g} spacings) but not on the sign of the change")
        return out
    out["state"] = "resolved"
    rests = (f"rests on native = emulated within {worst / step:.2g} spacings at a change of "
             f"{abs(n) / step:.2g} spacings")
    if changed:
        out["verdict"] = EFFECT
        out["rests_on"] = rests
        if exact is not None:
            e = float(exact[k])
            if e == 0 or (e > 0) != (n > 0):
                out["note"] = (f"the native change is the dtype's rounding: exact Δ {e:.5g}; "
                               + rests)
    else:
        out["verdict"] = NO_EFFECT
        if exact is not None:
            x = float(exact[int(exact.abs().argmax())])
            out["note"] = (f"no change at {out['dtype']} resolution (spacing {step:g}); exact Δ "
                           f"{x:.5g}" + (", which the dtype cannot show" if x != 0 else ""))
    return out


CONFIRMED = "confirmed"
NOT_CONFIRMED = "not confirmed"
# The one supplied constant of `confirm` and `decide`: the floor in spacings between the native value and the
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
