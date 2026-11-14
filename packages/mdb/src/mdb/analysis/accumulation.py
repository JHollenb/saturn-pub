"""The accumulation precision a matmul ``bound`` may assume, from the run's numeric scope.

Every ``bound`` on a native matmul (``forecast``/``propagate`` affine maps, scores and value
mixes; ``reads`` value-mix and head-write checks; ``circuits`` down_proj remainders;
``explain``'s head bound) uses the Higham sum bound ``|fl(sum) - sum| <= gamma_n sum |terms|``,
``gamma_n = n u / (1 - n u)``, with ``u`` the unit roundoff of the precision the partial sums are
kept in. For a 16-bit model that is float32 only when the backend keeps them in float32:

* CUDA with ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` (or the fp16
  twin) True, the torch default: cuBLAS may round partial sums (split-K) to the 16-bit dtype. The
  split count is not observable, so the premise is the worst case, every partial sum rounded to
  the model dtype: ``u = 2^-8`` (bf16) or ``2^-11`` (fp16). ``gamma_n`` then exists only for
  ``n u < 1`` (n < 256 terms in bf16); a longer sum has no bound and the bound is refused.
  Unless the run's scope holds a measurement (``numeric_scope["accumulation"]``, made when the
  adapter was built by ``saturn_pub.numerics.measure_accumulation``: native linear stages
  recomputed at batch 1 and compared with float32-accumulate-then-round). When it supports
  fp32-then-round the premise is that, ``u = 2^-24``, labelled measured, and ``gamma`` widens by
  the measured deviation (``max_deviation_milli``/1000, in ``eps x sum|terms|``) where it is not zero. When it does not
  support it, the per-partial-sum premise above stands and its refusal carries the measurement.
* CUDA float32 with TF32 allowed (``tf32_matmul`` True or ``float32_matmul_precision`` not
  ``highest``): products of TF32-rounded inputs, which the float32 premise does not cover: refused.
* Otherwise (CPU; CUDA with the flag False): float32 accumulation, ``u = 2^-24``.

``scope_flags(owner)`` is the one place the flags are read: the run's recorded numeric scope
(``adapter.execution["numeric_scope"]``, bound when the adapter was built and checked on every
native call), falling back to the live torch global when the scope does not record a field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

U32 = 2.0 ** -24
PREMISE_FAILS = "premise fails: "
FP32_PREMISE = "matmul accumulates in float32 in any order (gamma_n bound)"
FIELDS = ("bf16_reduced_precision_reduction", "fp16_reduced_precision_reduction",
          "tf32_matmul", "float32_matmul_precision")


def _torch():
    import torch

    return torch


def _live(field: str) -> Any:
    torch = _torch()
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if field == "float32_matmul_precision":
        return torch.get_float32_matmul_precision()
    name = {"tf32_matmul": "allow_tf32"}.get(field, f"allow_{field}")
    try:
        return getattr(matmul, name)
    except (AttributeError, RuntimeError):
        return "absent"


def scope_flags(owner: Any) -> dict[str, tuple[Any, str]]:
    """``{field: (value, source)}`` for the numeric fields a matmul bound depends on, plus
    ``device``. ``owner`` is a ``Run`` or an adapter. Source ``numeric scope`` when the run's
    recorded scope holds the field, else ``torch global`` (the live setting)."""
    adapter = getattr(owner, "adapter", owner)
    execution = getattr(adapter, "execution", None) or {}
    scope = execution.get("numeric_scope") or {}
    out: dict[str, tuple[Any, str]] = {}
    for field in FIELDS:
        if field in scope:
            out[field] = (scope[field], "numeric scope")
        else:
            out[field] = (_live(field), "torch global")
    device = (execution.get("device") or {}).get("type")
    if device is None:
        device = getattr(getattr(adapter, "device", None), "type", "cpu")
    out["device"] = (device, "execution" if execution.get("device") else "adapter")
    return out


def execution_scope(owner: Any) -> dict[str, Any]:
    """The run's recorded numeric scope (empty when it holds none)."""
    adapter = getattr(owner, "adapter", owner)
    return dict((getattr(adapter, "execution", None) or {}).get("numeric_scope") or {})


def _off(value: Any) -> bool:
    """A flag recorded as off: False, or a list/tuple of Falses (some torch builds return a
    tuple). Anything else, ``"absent"`` included, is taken as possibly on."""
    if isinstance(value, (list, tuple)):
        return bool(value) and all(item is False for item in value)
    return value is False


@dataclass(frozen=True)
class Accumulation:
    """``u``: unit roundoff of the partial sums a matmul may keep (None: no premise holds, see
    ``failure``); ``premise``: the stated accumulation premise of every bound built on it."""

    u: float | None
    premise: str
    failure: str | None = None
    reduced: bool = False
    deviation: float = 0.0       # measured |native - fp32-then-round| / (eps sum|terms|)
    eps: float = 0.0             # relative ulp of the output dtype (machine epsilon)

    def gamma(self, n: int) -> float | None:
        """``gamma_n`` (Higham) of an ``n``-term sum at this precision; None when it does not
        exist (``n u >= 1``) or no premise holds."""
        if self.u is None or n * self.u >= 1.0:
            return None
        base = n * self.u / (1.0 - n * self.u)
        if self.deviation:
            # native = fp32-then-round + at most d eps sum|terms| (measured), so it folds into
            # gamma as d eps (times 1 + gamma: the measured sum|terms| is the float32 one)
            base += self.deviation * self.eps * (1.0 + base)
        return base

    def refusal(self, n: int) -> str:
        """Why an ``n``-term sum has no bound under this accumulation."""
        if self.failure is not None:
            return self.failure
        return (f"{PREMISE_FAILS}{self.premise}; an {n}-term sum at u = {self.u:.3g} has no "
                f"gamma_n bound (n u = {n * self.u:.3g} >= 1)")


def accumulation(owner: Any, dtype: Any = None) -> Accumulation:
    """The accumulation premise of a native matmul of ``owner`` (a ``Run`` or an adapter) with
    16-bit or float32 inputs of ``dtype`` (default: the model dtype)."""
    torch = _torch()
    adapter = getattr(owner, "adapter", owner)
    dtype = getattr(adapter, "dtype", None) if dtype is None else dtype
    flags = scope_flags(owner)
    device = flags["device"][0]
    if device != "cuda":
        return Accumulation(U32, FP32_PREMISE)
    if dtype in (torch.bfloat16, torch.float16):
        name = "bf16" if dtype == torch.bfloat16 else "fp16"
        field = f"{name}_reduced_precision_reduction"
        value, source = flags[field]
        if _off(value):
            return Accumulation(U32, f"{FP32_PREMISE}: CUDA allow_{field} = False ({source})")
        eps = float(torch.finfo(dtype).eps)
        u = eps / 2.0
        measured = (execution_scope(owner).get("accumulation") or {})
        milli = int(measured.get("max_deviation_milli") or 0)
        if measured.get("dtype") == str(dtype):
            summary = (f"{measured.get('stages')} stages, batch {measured.get('batch')}, "
                       f"max deviation {milli / 1000:g} eps sum|terms| "
                       f"(ulp-index max {measured.get('max_ulp_index')}); by type "
                       f"{ {k: v / 1000 for k, v in (measured.get('by_type_milli') or {}).items()} }")
            if measured.get("premise") == "fp32-then-round":
                return Accumulation(
                    U32, f"matmul = fp32 accumulate then round once to {name}, measured on this "
                         f"device ({summary}); CUDA allow_{field} = {value!r} ({source}); "
                         f"measured on linear stages with synthetic inputs, premise for the "
                         f"rest", deviation=milli / 1000.0, eps=eps)
            note = (f" (measurement does not support fp32-then-round: {summary})")
        else:
            note = ""
        state = "True" if value is True else f"{value!r} (not False)"
        return Accumulation(u, f"CUDA allow_{field} = {state} ({source}): cuBLAS may round "
                               f"partial sums to {name}, so every partial sum is taken as rounded "
                               f"to {name} (u = {u:.3g}), any order{note}", reduced=True)
    if dtype == torch.float32:
        tf32, source = flags["tf32_matmul"]
        precision, psource = flags["float32_matmul_precision"]
        if not _off(tf32) or precision != "highest":
            return Accumulation(None, "float32 matmul", failure=(
                f"{PREMISE_FAILS}CUDA float32 matmul may use TF32 (tf32_matmul = {tf32!r} "
                f"({source}), float32_matmul_precision = {precision!r} ({psource})): products "
                "of TF32-rounded inputs are outside the float32 accumulation premise"))
    return Accumulation(U32, FP32_PREMISE)


def bound_state(reason: str | None) -> str:
    """``refused`` for a bound whose premise fails, else ``unresolved``."""
    return "refused" if reason is not None and reason.startswith(PREMISE_FAILS) else "unresolved"


__all__ = ["FP32_PREMISE", "execution_scope", "PREMISE_FAILS", "U32", "Accumulation", "accumulation", "bound_state",
           "scope_flags"]
