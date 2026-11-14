"""The live numerical environment an adapter binds into ``execution["numeric_scope"]``.

Every field here is a process-wide torch setting that can change native results bit for bit, so
a StateCut fingerprint binds it (through the adapter's ``execution``) and ``validate_execution``
refuses, naming the field, when the live value drifts from the bound one.

``bf16_reduced_precision_reduction`` / ``fp16_reduced_precision_reduction`` are
``torch.backends.cuda.matmul.allow_{bf16,fp16}_reduced_precision_reduction``: when true (the
torch default) cuBLAS may accumulate bf16/fp16 matmuls in reduced precision (split-K partial
sums rounded to the input dtype), so a ``bound`` that assumes float32 accumulation does not
hold. On a torch build without the attribute the field is the string ``"absent"``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REDUCED_PRECISION_FIELDS = (
    "bf16_reduced_precision_reduction",
    "fp16_reduced_precision_reduction",
)


def _flag(value: Any) -> Any:
    """A JSON-stable rendering of a backend flag (some torch builds return a tuple)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (tuple, list)):
        return [_flag(item) for item in value]
    if isinstance(value, int):
        return bool(value)
    return str(value)


def _matmul_flag(name: str) -> Any:
    import torch

    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    try:
        value = getattr(matmul, name)
    except (AttributeError, RuntimeError):
        return "absent"
    return _flag(value)


def live_numeric_environment() -> dict[str, Any]:
    """The drift-checked fields of a numeric scope, read from the live torch globals."""
    import torch

    return {
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "bf16_reduced_precision_reduction": _matmul_flag(
            "allow_bf16_reduced_precision_reduction"
        ),
        "fp16_reduced_precision_reduction": _matmul_flag(
            "allow_fp16_reduced_precision_reduction"
        ),
    }


def numeric_scope(**constants: Any) -> dict[str, Any]:
    """An adapter's ``numeric_scope``: its declared constants plus the live environment."""
    return {**constants, **live_numeric_environment()}


def scope_difference(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """``name: recorded R, current C`` for every field whose value differs (sorted by name)."""
    names = sorted(set(recorded) | set(current))
    return [
        f"{name}: recorded {recorded.get(name, 'missing')!r}, current {current.get(name, 'missing')!r}"
        for name in names
        if recorded.get(name, "missing") != current.get(name, "missing")
    ]


def check_numeric_environment(expected: Mapping[str, Any]) -> None:
    """Refuse when a live field differs from ``expected`` (the bound scope), naming the field."""
    live = live_numeric_environment()
    differing = scope_difference(
        {name: expected.get(name, "missing") for name in live}, live
    )
    if differing:
        raise ValueError(
            "live numerical environment drifted from the execution ABI ("
            + "; ".join(differing)
            + "); rewrap"
        )


MEASURE_ROWS = (1, 16)          # token rows per stage: a decode step and a short prefill
MEASURE_LAYERS = 3              # first, middle, last
MEASURE_CHUNK = 4096            # output rows of the float32 reference cast at once
SUPPORTED_SCALED = 2.0          # largest deviation (eps x sum|terms| units) the premise accepts


def _ulp_index(value: Any) -> Any:
    """Monotone integer index of 16-bit floats (adjacent representable values differ by 1)."""
    import torch

    bits = value.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _stage_deviation(module: Any, rows: int, generator: Any) -> tuple[int, float]:
    """``(ulps, scaled)`` of ``module(x)`` natively against the float32 accumulate-then-round
    reference, on ``rows`` standard-normal rows (batch 1): the largest ulp-index distance
    (informational: near a cancellation it is huge for a tiny absolute gap) and the largest
    ``|native - ref| / (eps S)`` with ``S = sum |terms|`` (+ ``|bias|``), the unit the rounding
    bounds are written in. ``x`` is cast to float32 once, the weight one block of output rows
    at a time (nothing wider than a block is kept)."""
    import torch
    import torch.nn.functional as F

    weight, bias = module.weight, module.bias
    eps = float(torch.finfo(weight.dtype).eps)
    tiny = float(torch.finfo(torch.float32).tiny)
    x = torch.randn(1, rows, weight.shape[1], generator=generator).to(
        device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        native = module(x)
        wide = x.float()
        magnitude = wide.abs()
        ulps, scaled = 0, 0.0
        for start in range(0, int(weight.shape[0]), MEASURE_CHUNK):
            stop = start + MEASURE_CHUNK
            block, bblock = weight[start:stop].float(), (
                None if bias is None else bias[start:stop].float())
            got = native[..., start:stop]
            if not bool(torch.isfinite(got.float()).all()):
                return 1 << 20, float("inf")
            ref = F.linear(wide, block, bblock).to(weight.dtype)
            size = F.linear(magnitude, block.abs(), None if bblock is None else bblock.abs())
            gap = (got.float() - ref.float()).abs()
            scaled = max(scaled, float((gap / (eps * size.clamp_min(tiny))).max()))
            ulps = max(ulps, int((_ulp_index(got) - _ulp_index(ref)).abs().max()))
            del block, bblock, ref, size, gap
    return ulps, scaled


def measure_accumulation(layer_modules: Any, head: Any = None) -> dict[str, Any]:
    """Whether native 16-bit linear stages equal "accumulate in float32, round once", measured
    on this device at batch 1: the first, middle and last layer's ``nn.Linear`` of each type, plus
    ``head``, each recomputed on ``MEASURE_ROWS`` token rows and compared in ulps. Inputs are
    seeded standard-normal rows (not the run's activations). Supported when the largest
    deviation (``max_deviation_milli`` / 1000, in ``eps x sum|terms|``) is at most
    ``SUPPORTED_SCALED``; it is the premise's measured widening."""
    import torch

    generator = torch.Generator().manual_seed(0)
    count = len(layer_modules)
    picks = sorted({0, count // 2, count - 1})[:MEASURE_LAYERS]
    stages: list[Any] = []
    for index in picks:
        seen: set[str] = set()
        for name, module in layer_modules[index].named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if isinstance(module, torch.nn.Linear) and leaf not in seen:
                seen.add(leaf)
                stages.append((leaf, module))
    if isinstance(head, torch.nn.Linear):
        stages.append(("lm_head", head))
    by_type: dict[str, float] = {}
    by_ulp: dict[str, int] = {}
    for leaf, module in stages:
        for rows in MEASURE_ROWS:
            ulps, scaled = _stage_deviation(module, rows, generator)
            by_type[leaf] = max(by_type.get(leaf, 0.0), scaled)
            by_ulp[leaf] = max(by_ulp.get(leaf, 0), ulps)
    by_milli = {k: math.ceil(v * 1000.0) if math.isfinite(v) else 1 << 30 for k, v in by_type.items()}
    worst = max(by_milli.values(), default=0)
    first = stages[0][1] if stages else None
    return {
        "premise": "fp32-then-round" if stages and worst <= 1000 * SUPPORTED_SCALED else "unsupported",
        "device": str(first.weight.device.type) if first is not None else "none",
        "dtype": str(first.weight.dtype) if first is not None else "none",
        "batch": 1, "rows": list(MEASURE_ROWS), "stages": len(stages) * len(MEASURE_ROWS),
        "unit": "milli eps x sum|terms| (1000 |native - fp32-then-round| / (eps sum|terms|), "
                "rounded up; integers because a numeric scope holds no bare floats)",
        "max_deviation_milli": worst, "by_type_milli": dict(sorted(by_milli.items())),
        "max_ulp_index": max(by_ulp.values(), default=0), "by_type_ulp_index": dict(sorted(by_ulp.items())),
    }


def needs_accumulation_measurement(device_type: str, dtype: Any) -> bool:
    """A 16-bit CUDA run whose reduced-precision-reduction flag is not off: the one case where
    the accumulation precision of native matmuls is not already fixed by the scope."""
    import torch

    if device_type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
        return False
    name = "bf16" if dtype == torch.bfloat16 else "fp16"
    value = _matmul_flag(f"allow_{name}_reduced_precision_reduction")
    off = value is False or (isinstance(value, list) and bool(value) and all(
        item is False for item in value))
    return not off


__all__ = [
    "MEASURE_ROWS",
    "SUPPORTED_SCALED",
    "measure_accumulation",
    "needs_accumulation_measurement",
    "REDUCED_PRECISION_FIELDS",
    "check_numeric_environment",
    "live_numeric_environment",
    "numeric_scope",
    "scope_difference",
]
