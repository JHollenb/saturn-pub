"""Second-order scalar forecasts for a smooth Qwen read.

The first-order producer in :mod:`mdb.read_weight_transport` compiles a query
and current-key tangent.  This module adds the smallest second-order receipt
around that producer.  It evaluates the smooth shadow at
``baseline_attention + lambda * write_delta`` and anchors the resulting
displacement to the *native* query and current key.  Thus the native BF16
baseline score is preserved at ``lambda == 0`` while the smooth path supplies
the local derivatives.

Compilation uses PyTorch and performs the dense shadow work.  The returned
serving program contains only ordinary Python floats, so
``evaluate_taylor_program`` is usable by framework-free collectors and
viewers without importing PyTorch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .read_weight_transport import (
    ReadWeightTransportError,
    SmoothQwenReadTransport,
    _canonical_k,
    _canonical_q,
    _canonical_scores,
    _float_input,
    _plain_int,
    _torch,
    compile_score_program,
)

_TAYLOR_SCHEMA = "mdb-smooth-qwen-taylor-program-v1"
_MAX_SELECTED_ROWS = 6
_METHODS = {
    "order2",
    "order1",
    "second-order",
    "second_order",
    "linear-logweight",
    "stale",
    "routing-only",
    "wrong-curvature",
    "wrongcurvature",
    "wrong-sign",
    "wrongsign",
    "wrong_sign",
}


def _ordinary_float(value: Any, *, device: Any, label: str):
    """Clone a tensor out of inference mode before it enters a JVP."""

    torch, _ = _torch()
    if not isinstance(value, torch.Tensor):
        raise ReadWeightTransportError(f"{label} must be a tensor")
    with torch.inference_mode(False):
        return value.detach().to(device=device, dtype=torch.float32).clone()


def _scalar_lambda(value: Any, *, device: Any):
    torch, _ = _torch()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ReadWeightTransportError("lambda must be scalar")
        return value.to(device=device, dtype=torch.float32)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ReadWeightTransportError("lambda must be finite")
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def _selected_rows(value: Sequence[int] | None, *, source_row: int, rows: int) -> tuple[int, ...]:
    if value is None:
        result = (source_row,)
    else:
        if isinstance(value, (str, bytes)):
            raise ReadWeightTransportError("selected_rows must be a sequence of integers")
        try:
            result = tuple(_plain_int(item, "selected row") for item in value)
        except TypeError as exc:
            raise ReadWeightTransportError("selected_rows must be a sequence of integers") from exc
    if not result:
        raise ReadWeightTransportError("selected_rows must not be empty")
    if len(result) > _MAX_SELECTED_ROWS:
        raise ReadWeightTransportError(f"selected_rows may contain at most {_MAX_SELECTED_ROWS} rows")
    if len(set(result)) != len(result):
        raise ReadWeightTransportError("selected_rows must not contain duplicates")
    if any(row >= rows for row in result):
        raise ReadWeightTransportError("selected row is outside the native key bank")
    return result


def _as_float_matrix(value: Any, *, label: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ReadWeightTransportError(f"program {label} must be a non-empty matrix")
    matrix: list[list[float]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ReadWeightTransportError(f"program {label} must be a rectangular matrix")
        converted = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ReadWeightTransportError(f"program {label} contains a non-finite scalar")
            converted.append(float(item))
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            raise ReadWeightTransportError(f"program {label} must be rectangular")
        matrix.append(converted)
    return matrix


def _matrix_subtract(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[a - b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]


def _same_matrix_shape(left: list[list[float]], right: list[list[float]]) -> bool:
    return len(left) == len(right) and all(len(a) == len(b) for a, b in zip(left, right))


def _matrix_add_scaled(
    base: list[list[float]], first: list[list[float]], second: list[list[float]], dose: float, curvature: float
) -> list[list[float]]:
    return [
        [base_value + dose * first_value + curvature * second_value
         for base_value, first_value, second_value in zip(base_row, first_row, second_row)]
        for base_row, first_row, second_row in zip(base, first, second)
    ]


def _prepare(
    transport: SmoothQwenReadTransport,
    q0_native: Any,
    k_full_native: Any,
    native_score0: Any,
    write_delta: Any,
    *,
    kv_head: int,
    source_row: int,
    selected_rows: Sequence[int] | None,
    scaling: float | None,
) -> dict[str, Any]:
    """Validate and clone the native anchor once per compilation."""

    torch, _ = _torch()
    if not isinstance(transport, SmoothQwenReadTransport):
        raise ReadWeightTransportError("transport must be SmoothQwenReadTransport")
    device = transport.device
    q0 = _canonical_q(_ordinary_float(q0_native, device=device, label="q0_native"), label="q0_native")
    k_full = _canonical_k(
        _ordinary_float(k_full_native, device=device, label="k_full_native"), label="k_full_native"
    )
    scores = _canonical_scores(
        _ordinary_float(native_score0, device=device, label="native_score0"), label="native_score0"
    )
    direction = _float_input(write_delta, device=device, label="write_delta")
    with torch.inference_mode(False):
        direction = direction.detach().clone()
    if direction.shape != transport.baseline_attention.shape:
        raise ReadWeightTransportError("write_delta must match transport baseline_attention")
    if q0.shape[0] != 1 or k_full.shape[0] != 1 or scores.shape[0] != 1:
        raise ReadWeightTransportError("Taylor compilation supports batch size one")
    if q0.shape[1] != scores.shape[1] or scores.shape[-1] != k_full.shape[-2]:
        raise ReadWeightTransportError("native q/k/score geometry is inconsistent")
    if q0.shape[-1] != k_full.shape[-1]:
        raise ReadWeightTransportError("native q/k head dimensions differ")
    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        if scaling is None:
            scaling = float(transport._target_geometry[-1])
        else:
            raise ReadWeightTransportError("scaling must be a finite positive number")
    if not math.isfinite(float(scaling)) or float(scaling) <= 0:
        raise ReadWeightTransportError("scaling must be a finite positive number")
    kv_head = _plain_int(kv_head, "kv_head")
    source_row = _plain_int(source_row, "source_row")
    kv_heads = int(k_full.shape[1])
    rows = int(k_full.shape[-2])
    if kv_head >= kv_heads:
        raise ReadWeightTransportError("kv_head is outside k_full_native")
    if source_row >= rows:
        raise ReadWeightTransportError("source_row is outside k_full_native")
    query_heads = int(q0.shape[1])
    if query_heads % kv_heads:
        raise ReadWeightTransportError("query heads are not divisible by KV heads")
    repeats = query_heads // kv_heads
    first_head = kv_head * repeats
    last_head = first_head + repeats
    selected = _selected_rows(selected_rows, source_row=source_row, rows=rows)
    selected_index = torch.tensor(selected, device=device, dtype=torch.long)
    with torch.inference_mode(False):
        baseline_attention = transport.baseline_attention.detach().clone()
        shadow_q0, shadow_k0 = transport.shadow(baseline_attention)
    return {
        "device": device,
        "q0": q0,
        "k_full": k_full,
        "native_scores": scores,
        "direction": direction,
        "baseline_attention": baseline_attention,
        "shadow_q0": shadow_q0.detach().clone(),
        "shadow_k0": shadow_k0.detach().clone(),
        "kv_head": kv_head,
        "source_row": source_row,
        "selected_rows": selected,
        "selected_index": selected_index,
        "first_head": first_head,
        "last_head": last_head,
        "scaling": float(scaling),
        "rows": rows,
    }


def _state(transport: SmoothQwenReadTransport, config: Mapping[str, Any], lambda_value: Any):
    """Return anchored q, current k, and selected native-baseline logweights."""

    torch, _ = _torch()
    lam = _scalar_lambda(lambda_value, device=config["device"])
    smooth_q, smooth_k = transport.shadow(
        config["baseline_attention"] + lam * config["direction"]
    )
    q = config["q0"] + (smooth_q - config["shadow_q0"])
    current_k = config["k_full"][:, :, -1:, :] + (smooth_k - config["shadow_k0"])
    first_head, last_head = config["first_head"], config["last_head"]
    q_group = q[:, first_head:last_head, 0, :]
    key_group = config["k_full"][:, config["kv_head"], :, :]
    key_group = torch.cat((key_group[..., :-1, :], current_k[:, config["kv_head"], :, :]), dim=-2)
    native_key_group = config["k_full"][:, config["kv_head"], :, :]
    smooth_scores = torch.einsum("bhd,btd->bht", q_group, key_group) * config["scaling"]
    native_scores = torch.einsum(
        "bhd,btd->bht", config["q0"][:, first_head:last_head, 0, :], native_key_group
    ) * config["scaling"]
    anchored_scores = config["native_scores"][:, first_head:last_head, 0, :] + (
        smooth_scores - native_scores
    )
    logp = torch.log_softmax(anchored_scores, dim=-1)
    selected_logp = logp.index_select(-1, config["selected_index"])
    return q, current_k, selected_logp


def smooth_native_logweights(
    transport: SmoothQwenReadTransport,
    q0_native: Any,
    k_full_native: Any,
    native_score0: Any,
    write_delta: Any,
    lambda_value: float,
    *,
    kv_head: int,
    source_row: int,
    selected_rows: Sequence[int] | None = None,
    scaling: float | None = None,
):
    """Evaluate the anchored smooth selected logweights at one dose."""

    config = _prepare(
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
    with _torch()[0].inference_mode(False):
        return _state(transport, config, lambda_value)[2]


def _to_matrix(tensor: Any) -> list[list[float]]:
    values = tensor.detach().to(device="cpu", dtype=_torch()[0].float32).tolist()
    return [[float(item) for item in row] for row in values[0]]


def compile_taylor_program(
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
    score_program: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a scalar second-order logweight program.

    ``c1`` and ``c2`` are the first and second derivatives of selected native
    logweights with respect to ``lambda`` at zero.  ``c2_route`` uses only the
    first-order Q/K score transport (the current-row Q/K cross term and the
    softmax normalizer); ``c2 - c2_route`` is the centered upstream smooth
    q/k-curvature diagnostic.
    """

    torch, _ = _torch()
    config = _prepare(
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
    zero = torch.zeros((), device=config["device"], dtype=torch.float32)

    def state(value):
        return _state(transport, config, value)

    def state_tangent(value):
        _, tangent = torch.func.jvp(state, (value,), (torch.ones_like(value),))
        return tangent

    with torch.inference_mode(False):
        primal, first = torch.func.jvp(state, (zero,), (torch.ones_like(zero),))
        _, second = torch.func.jvp(state_tangent, (zero,), (torch.ones_like(zero),))
    _, _, logp0 = primal
    # ``state`` returns (q, current_k, selected_logp).  Keep the two tangent
    # values named explicitly so the score-program comparator remains clear.
    q_dot = first[0]
    k_dot = first[1]
    c1 = first[2]
    c2 = second[2]

    if score_program is None:
        score_program = compile_score_program(
            config["q0"],
            config["k_full"],
            config["native_scores"],
            q_dot,
            k_dot,
            kv_head=config["kv_head"],
            source_row=config["source_row"],
            scaling=config["scaling"],
        )
    score_logp0 = score_program["logp0"]
    score_a = score_program["a"]
    score_b = torch.zeros_like(score_a)
    score_b[..., score_program["current_row"]] = score_program["b"]
    probability = score_logp0.exp()
    expectation_a = (probability * score_a).sum(dim=-1, keepdim=True)
    expectation_b = (probability * score_b).sum(dim=-1, keepdim=True)
    variance_a = (probability * (score_a - expectation_a).square()).sum(dim=-1, keepdim=True)
    route_c2 = 2.0 * (score_b - expectation_b) - variance_a
    selected_index = config["selected_index"]
    selected_c1_score = (score_a - expectation_a).index_select(-1, selected_index)
    selected_route_c2 = route_c2.index_select(-1, selected_index)

    return {
        "schema": _TAYLOR_SCHEMA,
        "logp0": _to_matrix(logp0),
        "c1": _to_matrix(c1),
        "c2": _to_matrix(c2),
        "c1_score": _to_matrix(selected_c1_score),
        "c2_route": _to_matrix(selected_route_c2),
        "c2_upstream": _to_matrix(c2 - selected_route_c2),
        "selected_rows": list(config["selected_rows"]),
        "query_heads": list(range(config["first_head"], config["last_head"])),
        "kv_head": config["kv_head"],
        "source_row": config["source_row"],
        "current_row": config["rows"] - 1,
        "scaling": config["scaling"],
        "cost": {
            "smooth_dense_linears": int(transport.logical_linears),
            "nested_jvp_passes": 3,
            "native_forward_calls": 0,
            "selected_rows": len(config["selected_rows"]),
            "selected_logweights": len(config["selected_rows"])
            * (config["last_head"] - config["first_head"]),
        },
    }


def evaluate_taylor_program(
    program: Mapping[str, Any], dose: float, *, method: str = "order2"
) -> dict[str, Any]:
    """Evaluate a Taylor receipt using only Python scalar arithmetic."""

    if not isinstance(program, Mapping) or program.get("schema") != _TAYLOR_SCHEMA:
        raise ReadWeightTransportError("program has an unsupported Taylor schema")
    if method not in _METHODS:
        raise ReadWeightTransportError(f"unknown Taylor method {method!r}")
    if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(float(dose)):
        raise ReadWeightTransportError("dose must be finite")
    dose = float(dose)
    logp0 = _as_float_matrix(program.get("logp0"), label="logp0")
    c1 = _as_float_matrix(program.get("c1"), label="c1")
    c2 = _as_float_matrix(program.get("c2"), label="c2")
    if not (len(logp0) == len(c1) == len(c2)) or any(
        not (len(a) == len(b) == len(c)) for a, b, c in zip(logp0, c1, c2)
    ):
        raise ReadWeightTransportError("Taylor coefficient matrices have inconsistent shapes")
    canonical_method = {
        "second-order": "order2",
        "second_order": "order2",
        "linear-logweight": "order1",
        "wrongcurvature": "wrong-curvature",
        "wrongsign": "wrong-sign",
        "wrong_sign": "wrong-sign",
    }.get(method, method)
    if canonical_method == "stale":
        predicted = [row[:] for row in logp0]
    else:
        sign = -1.0 if canonical_method == "wrong-sign" else 1.0
        curvature = -0.5 if canonical_method == "wrong-curvature" else 0.5
        second = _as_float_matrix(
            program.get("c2_route" if canonical_method == "routing-only" else "c2"),
            label="c2_route" if canonical_method == "routing-only" else "c2",
        )
        if not _same_matrix_shape(second, logp0):
            raise ReadWeightTransportError("Taylor curvature matrix has inconsistent shape")
        if canonical_method == "order1":
            curvature = 0.0
        predicted = _matrix_add_scaled(logp0, c1, second, sign * dose, curvature * dose * dose)
    delta = _matrix_subtract(predicted, logp0)
    return {
        "schema": "mdb-smooth-qwen-taylor-evaluation-v1",
        "method": method,
        "dose": dose,
        "pred_logpB": predicted,
        "logpB": predicted,
        "delta_logpB": delta,
        "pB": [[math.exp(value) for value in row] for row in predicted],
    }


__all__ = [
    "ReadWeightTransportError",
    "SmoothQwenReadTransport",
    "compile_taylor_program",
    "evaluate_taylor_program",
    "smooth_native_logweights",
]
