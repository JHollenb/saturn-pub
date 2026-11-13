"""Centered second-order scalar forecasts for a smooth Qwen read.

This module reuses the native-zero-anchored path from
:mod:`mdb.read_weight_taylor`, but expands it around a caller-selected dose.
The raw centered polynomial can miss the native zero-dose anchor; that residual
is retained explicitly.  The optional baseline-corrected evaluator subtracts
the polynomial's value at zero and adds the native baseline back.

Compilation uses PyTorch.  The returned program contains only Python scalar
lists and remains usable by framework-free viewers and collectors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import read_weight_taylor as _taylor
from .read_weight_transport import ReadWeightTransportError, SmoothQwenReadTransport, _torch

_CENTERED_SCHEMA = "mdb-smooth-qwen-centered-taylor-program-v1"
_EVALUATION_SCHEMA = "mdb-smooth-qwen-centered-taylor-evaluation-v1"
_METHODS = {
    "centered",
    "baseline-corrected",
    "baseline_corrected",
    "stale",
    "wrong-sign",
    "wrong_sign",
    "wrong-curvature",
    "wrong_curvature",
}


class PreparedCenteredPath:
    """Callable, once-prepared native-anchored smooth path.

    Preparing a path copies the baseline tape and computes its zero-dose smooth
    Q/K anchor once.  Calling the object evaluates selected log weights at one
    dose without rebuilding that preparation.  :meth:`to_python_matrix` is a
    convenience for benchmarks that need the same CPU float readout as a
    compiled program.
    """

    def __init__(self, transport: SmoothQwenReadTransport, config: Mapping[str, Any]) -> None:
        self.transport = transport
        self.config = config

    def __call__(self, dose: Any):
        torch, _ = _torch()
        with torch.inference_mode(False):
            return _taylor._state(self.transport, self.config, dose)[2]

    def to_python_matrix(self, dose: Any) -> list[list[float]]:
        """Evaluate one dose and copy selected values to ordinary Python floats."""

        return _taylor._to_matrix(self(dose))

    @property
    def metadata(self) -> dict[str, Any]:
        """Return non-tensor identity and geometry needed by a compiler."""

        config = self.config
        return {
            "selected_rows": list(config["selected_rows"]),
            "query_heads": list(range(config["first_head"], config["last_head"])),
            "kv_head": config["kv_head"],
            "source_row": config["source_row"],
            "current_row": config["rows"] - 1,
            "scaling": config["scaling"],
        }


def prepare_centered_path(
    transport: SmoothQwenReadTransport,
    q0_native: Any,
    k_full_native: Any,
    native_score0: Any,
    write_delta: Any,
    *,
    kv_head: int,
    source_row: int,
    selected_rows: Sequence[int] | None = None,
    scaling: float | None = None,
) -> PreparedCenteredPath:
    """Validate a baseline tape once and return a reusable smooth path."""

    config = _taylor._prepare(
        transport,
        q0_native,
        k_full_native,
        native_score0,
        write_delta,
        kv_head=kv_head,
        source_row=source_row,
        selected_rows=selected_rows,
        scaling=scaling,
    )
    return PreparedCenteredPath(transport, config)


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReadWeightTransportError(f"{label} must be finite")
    return float(value)


def _matrices_from_program(program: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    labels = (
        "logp_native0",
        "g_center",
        "c1",
        "c2",
        "centered_zero_logp",
        "zero_centered_residual",
    )
    matrices = {label: _taylor._as_float_matrix(program.get(label), label=label) for label in labels}
    reference = matrices["logp_native0"]
    if any(not _taylor._same_matrix_shape(reference, matrix) for matrix in matrices.values()):
        raise ReadWeightTransportError("centered program matrices have inconsistent shapes")
    return matrices


def _poly(
    g_center: list[list[float]],
    c1: list[list[float]],
    c2: list[list[float]],
    *,
    point: float,
    center: float,
    curvature_sign: float = 1.0,
) -> list[list[float]]:
    offset = point - center
    return _taylor._matrix_add_scaled(
        g_center,
        c1,
        c2,
        offset,
        curvature_sign * 0.5 * offset * offset,
    )


def compile_centered_path(
    prepared_path: PreparedCenteredPath,
    *,
    center: float = 0.5,
) -> dict[str, Any]:
    """Compile ``g(center)``, ``g'(center)`` and ``g''(center)`` once prepared.

    The underlying path remains anchored to native Q/K and native baseline
    scores at dose zero.  Moving the Taylor expansion point does not replace
    that anchor with an edited or smooth midpoint.
    """

    if not isinstance(prepared_path, PreparedCenteredPath):
        raise ReadWeightTransportError("prepared_path must come from prepare_centered_path")
    center = _finite_real(center, label="center")
    torch, _ = _torch()
    config = prepared_path.config
    center_tensor = torch.tensor(center, device=config["device"], dtype=torch.float32)

    def state(value):
        return prepared_path(value)

    def first_derivative(value):
        _, tangent = torch.func.jvp(state, (value,), (torch.ones_like(value),))
        return tangent

    with torch.inference_mode(False):
        g_center_tensor, c1_tensor = torch.func.jvp(
            state, (center_tensor,), (torch.ones_like(center_tensor),)
        )
        _, c2_tensor = torch.func.jvp(
            first_derivative, (center_tensor,), (torch.ones_like(center_tensor),)
        )
        native_logits = config["native_scores"][:, config["first_head"]:config["last_head"], 0, :]
        logp_native0_tensor = torch.log_softmax(native_logits, dim=-1).index_select(
            -1, config["selected_index"]
        )

    g_center = _taylor._to_matrix(g_center_tensor)
    c1 = _taylor._to_matrix(c1_tensor)
    c2 = _taylor._to_matrix(c2_tensor)
    logp_native0 = _taylor._to_matrix(logp_native0_tensor)
    centered_zero_logp = _poly(g_center, c1, c2, point=0.0, center=center)
    zero_centered_residual = _taylor._matrix_subtract(centered_zero_logp, logp_native0)
    return {
        "schema": _CENTERED_SCHEMA,
        "logp_native0": logp_native0,
        "g_center": g_center,
        "c1": c1,
        "c2": c2,
        "center": center,
        "centered_zero_logp": centered_zero_logp,
        "zero_centered_residual": zero_centered_residual,
        **prepared_path.metadata,
        "cost": {
            "smooth_dense_linears": int(prepared_path.transport.logical_linears),
            "nested_jvp_passes": 3,
            "native_forward_calls": 0,
            "selected_rows": len(config["selected_rows"]),
            "selected_logweights": len(config["selected_rows"])
            * (config["last_head"] - config["first_head"]),
            "native_baseline_scores": "supplied-and-retained",
        },
    }


def compile_at_center(
    transport: SmoothQwenReadTransport,
    q0_native: Any,
    k_full_native: Any,
    native_score0: Any,
    write_delta: Any,
    *,
    center: float = 0.5,
    kv_head: int,
    source_row: int,
    selected_rows: Sequence[int] | None = None,
    scaling: float | None = None,
) -> dict[str, Any]:
    """Prepare a baseline path and compile its local centered polynomial."""

    prepared_path = prepare_centered_path(
        transport,
        q0_native,
        k_full_native,
        native_score0,
        write_delta,
        kv_head=kv_head,
        source_row=source_row,
        selected_rows=selected_rows,
        scaling=scaling,
    )
    return compile_centered_path(prepared_path, center=center)


def evaluate_centered_program(
    program: Mapping[str, Any], dose: float, *, method: str = "centered"
) -> dict[str, Any]:
    """Evaluate a centered program with Python arithmetic only.

    ``centered`` returns the raw Taylor polynomial. ``baseline-corrected``
    subtracts the same polynomial at zero and adds the native zero-dose anchor,
    giving an exact zero-dose delta while retaining the raw residual separately.
    ``wrong-sign`` evaluates the same polynomial at ``-dose``.
    """

    if not isinstance(program, Mapping) or program.get("schema") != _CENTERED_SCHEMA:
        raise ReadWeightTransportError("program has an unsupported centered Taylor schema")
    if method not in _METHODS:
        raise ReadWeightTransportError(f"unknown centered Taylor method {method!r}")
    dose = _finite_real(dose, label="dose")
    center = _finite_real(program.get("center"), label="program center")
    matrices = _matrices_from_program(program)
    native0 = matrices["logp_native0"]
    g_center = matrices["g_center"]
    c1 = matrices["c1"]
    c2 = matrices["c2"]
    method = {
        "baseline_corrected": "baseline-corrected",
        "wrong_sign": "wrong-sign",
        "wrong_curvature": "wrong-curvature",
    }.get(method, method)

    centered_zero = _poly(g_center, c1, c2, point=0.0, center=center)
    retained_zero = _taylor._matrix_subtract(centered_zero, native0)
    declared_zero = matrices["zero_centered_residual"]
    if retained_zero != declared_zero or centered_zero != matrices["centered_zero_logp"]:
        raise ReadWeightTransportError("centered zero-dose diagnostic disagrees with program coefficients")

    if method == "centered":
        predicted = _poly(g_center, c1, c2, point=dose, center=center)
        zero_prediction = centered_zero
    elif method == "baseline-corrected":
        raw_at_dose = _poly(g_center, c1, c2, point=dose, center=center)
        if dose == 0.0:
            predicted = [row[:] for row in native0]
        else:
            difference = _taylor._matrix_subtract(raw_at_dose, centered_zero)
            predicted = [
                [base + delta for base, delta in zip(base_row, delta_row)]
                for base_row, delta_row in zip(native0, difference)
            ]
        zero_prediction = [row[:] for row in native0]
    elif method == "stale":
        predicted = [row[:] for row in native0]
        zero_prediction = [row[:] for row in native0]
    elif method == "wrong-sign":
        predicted = _poly(g_center, c1, c2, point=-dose, center=center)
        zero_prediction = centered_zero
    else:  # wrong-curvature
        predicted = _poly(g_center, c1, c2, point=dose, center=center, curvature_sign=-1.0)
        zero_prediction = _poly(g_center, c1, c2, point=0.0, center=center, curvature_sign=-1.0)

    return {
        "schema": _EVALUATION_SCHEMA,
        "method": method,
        "dose": dose,
        "pred_logpB": predicted,
        "logpB": predicted,
        "delta_logpB": _taylor._matrix_subtract(predicted, native0),
        "pB": [[math.exp(value) for value in row] for row in predicted],
        "centered_zero_residual": retained_zero,
        "predicted_zero_logpB": zero_prediction,
        "zero_delta_logpB": _taylor._matrix_subtract(zero_prediction, native0),
    }


__all__ = [
    "PreparedCenteredPath",
    "ReadWeightTransportError",
    "SmoothQwenReadTransport",
    "compile_at_center",
    "compile_centered_path",
    "evaluate_centered_program",
    "prepare_centered_path",
]
