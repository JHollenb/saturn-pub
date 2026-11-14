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

    def gamma(self, n: int) -> float | None:
        """``gamma_n`` (Higham) of an ``n``-term sum at this precision; None when it does not
        exist (``n u >= 1``) or no premise holds."""
        if self.u is None or n * self.u >= 1.0:
            return None
        return n * self.u / (1.0 - n * self.u)

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
        u = float(torch.finfo(dtype).eps) / 2.0
        state = "True" if value is True else f"{value!r} (not False)"
        return Accumulation(u, f"CUDA allow_{field} = {state} ({source}): cuBLAS may round "
                               f"partial sums to {name}, so every partial sum is taken as rounded "
                               f"to {name} (u = {u:.3g}), any order", reduced=True)
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


__all__ = ["FP32_PREMISE", "PREMISE_FAILS", "U32", "Accumulation", "accumulation", "bound_state",
           "scope_flags"]
