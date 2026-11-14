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


__all__ = [
    "REDUCED_PRECISION_FIELDS",
    "check_numeric_environment",
    "live_numeric_environment",
    "numeric_scope",
    "scope_difference",
]
