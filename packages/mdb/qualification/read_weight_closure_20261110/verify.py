"""Offline audit for the prospective midpoint/native-profile closure assay.

The saved report and Bench/CAS trees are audited without loading a model or
re-running the trained derivative compiler. Scalar forecasts are evaluated
again from their sealed Python coefficients. Native errors, parity, and
continuation outcomes remain observations; only custody, typed data, cut
lineage, and span-install mechanics are required for a valid audit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from mdb.read_program import FrozenReadProgram

_BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "read_weight_transport_20261107"
    / "verify.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location(
    "mdb_read_weight_transport_audit_for_closure", _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load fixed local audit helpers from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)


REPORT_SCHEMA = "mdb-read-weight-closure-report-v1"
VERIFY_SCHEMA = "mdb-read-weight-closure-verification-v1"
FORECAST_SCHEMA = "mdb-read-weight-closure-forecast-v1"
DOSES = (0.0, -0.25, 0.2, 0.6, 1.0)
METHODS = (
    "midpoint",
    "baseline-corrected",
    "zero-centered",
    "stale",
    "wrong-sign",
    "wrong-curvature",
)
EXTENDED_WEIGHT_METRICS = {
    "smooth_remainder_rmse",
    "smooth_native_profile_rmse",
    "mse_cross_term",
    "fp64_total_rmse",
    "fp64_identity_max_abs",
    "fp64_mse_identity_residual",
}
CONSUMERS = (
    "midpoint",
    "baseline-corrected",
    "zero-centered",
    "stale",
    "functional-native",
)
CASE_SPLITS = {
    "nora-ethan": "development",
    "ivy-ivy": "development",
    "clara-leo": "fresh",
    "maya-maya": "fresh",
}
NUM_LAYERS = 28


def _source_dependencies() -> dict[str, dict[str, str]]:
    return {
        "read_weight_transport_verify.py": {
            "path": "qualification/read_weight_transport_20261107/verify.py",
            "sha256": sha256(_BASE_PATH.read_bytes()).hexdigest(),
        }
    }


# The shared verifier contains only offline Bench, CAS, cut, receipt, and
# tensor-custody helpers. The compiler/evaluator are deliberately not imported.
_finite = _BASE._finite
_dict = _BASE._dict
_list = _BASE._list
_ref = _BASE._ref
_sha = _BASE._sha
_int = _BASE._int
_load_case = _BASE._load_case
_bench_read_only = _BASE._bench_read_only
_payload = _BASE._payload
_audit_payloads = _BASE._audit_payloads
_hydrate_tree = _BASE._hydrate_tree
_artifact = _BASE._artifact
_tensor_equal = _BASE._tensor_equal
_tensor_close = _BASE._tensor_close
_error = _BASE._error
_compare = _BASE._compare
_load_cut = _BASE._load_cut
_span = _BASE._span
_check_sites = _BASE._check_sites
_ancestral = _BASE._ancestral
_verify_acts = _BASE._verify_acts
_capture = _BASE._capture
_signature = _BASE._signature
_required_capture = _BASE._required_capture
_future = _BASE._future
_verify_receipt = _BASE._verify_receipt


def _same_number(actual: Any, expected: float, *, label: str,
                 rtol: float = 2e-5, atol: float = 2e-6) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), float(expected), rel_tol=rtol, abs_tol=atol
    ):
        raise ValueError(f"{label} differs from the sealed scalar law")


def _matrix(value: Any, *, label: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty matrix")
    rows: list[list[float]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"{label} must be rectangular")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"{label} must be rectangular")
        numeric: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{label} contains a non-numeric value")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"{label} contains a non-finite value")
            numeric.append(number)
        rows.append(numeric)
    return rows


def _matrix_sub(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [a - b for a, b in zip(arow, brow, strict=True)]
        for arow, brow in zip(left, right, strict=True)
    ]


def _poly(program: Mapping[str, Any], dose: float, *, curvature_sign: float = 1.0):
    center = float(program["center"])
    offset = float(dose) - center
    g = _matrix(program["g_center"], label="program g_center")
    c1 = _matrix(program["c1"], label="program c1")
    c2 = _matrix(program["c2"], label="program c2")
    if not (len(g) == len(c1) == len(c2)) or any(
        not (len(g[i]) == len(c1[i]) == len(c2[i])) for i in range(len(g))
    ):
        raise ValueError("centered coefficient geometry changed")
    return [
        [
            gv + offset * first + curvature_sign * 0.5 * offset * offset * second
            for gv, first, second in zip(grow, frow, srow, strict=True)
        ]
        for grow, frow, srow in zip(g, c1, c2, strict=True)
    ]


def _flat(value: Any, *, label: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    result = []
    for item in value:
        if isinstance(item, (list, tuple)):
            result.extend(_flat(item, label=label))
        else:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{label} contains a non-numeric value")
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(f"{label} contains a non-finite value")
            result.append(number)
    return result


def _close_flat(actual: Any, expected: Any, *, label: str,
                rtol: float = 2e-5, atol: float = 2e-6) -> None:
    left, right = _flat(actual, label=label), _flat(expected, label=label)
    if len(left) != len(right):
        raise ValueError(f"{label} geometry changed")
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        _same_number(a, b, label=f"{label}[{index}]", rtol=rtol, atol=atol)


def _program(program: Mapping[str, Any], *, center: float,
             sites: Mapping[str, Any], q0, k0) -> dict[str, Any]:
    program = _dict(program, "centered scalar program")
    if program.get("schema") != "mdb-smooth-qwen-centered-taylor-program-v1":
        raise ValueError("centered scalar program schema changed")
    _same_number(program.get("center"), center, label="program center", rtol=0, atol=1e-12)
    bsite = sites["B"]
    if (program.get("kv_head"), program.get("source_row")) != (
        bsite["kv_head"], bsite["source_row"]
    ):
        raise ValueError("centered program reader binding changed")
    if program.get("selected_rows") != [bsite["source_row"]]:
        raise ValueError("centered program selected-row binding changed")
    groups = int(q0.shape[1]) // int(k0.shape[1])
    first_head = bsite["kv_head"] * groups
    expected_heads = list(range(first_head, first_head + groups))
    if program.get("query_heads") != expected_heads:
        raise ValueError("centered program query-head binding changed")
    if program.get("current_row") != int(k0.shape[2]) - 1:
        raise ValueError("centered program current-row binding changed")
    cost = _dict(program.get("cost"), "centered program cost")
    if cost.get("native_forward_calls") != 0:
        raise ValueError("centered program declares a native module forward")

    names = (
        "logp_native0",
        "g_center",
        "c1",
        "c2",
        "centered_zero_logp",
        "zero_centered_residual",
    )
    matrices = {name: _matrix(program.get(name), label=f"program {name}") for name in names}
    shape = (len(matrices["logp_native0"]), len(matrices["logp_native0"][0]))
    if any(
        (len(matrix), len(matrix[0])) != shape
        or any(len(row) != shape[1] for row in matrix)
        for matrix in matrices.values()
    ):
        raise ValueError("centered program matrix geometry changed")
    zero_at_center = _poly(program, 0.0)
    _close_flat(
        matrices["centered_zero_logp"],
        zero_at_center,
        label="centered zero-dose polynomial",
    )
    residual = _matrix_sub(zero_at_center, matrices["logp_native0"])
    _close_flat(
        matrices["zero_centered_residual"],
        residual,
        label="centered zero-dose residual",
    )
    return {**program, "_matrices": matrices}


def _scalar_prediction(programs, dose: float, method: str) -> dict[str, list[list[float]]]:
    midpoint = programs["midpoint"]
    zero = programs["zero"]
    native0 = midpoint["_matrices"]["logp_native0"]
    if method == "midpoint":
        predicted = _poly(midpoint, dose)
    elif method == "baseline-corrected":
        raw = _poly(midpoint, dose)
        raw0 = _poly(midpoint, 0.0)
        predicted = [
            [base + value - zero_value
             for base, value, zero_value in zip(brow, row, zrow, strict=True)]
            for brow, row, zrow in zip(native0, raw, raw0, strict=True)
        ]
    elif method == "zero-centered":
        predicted = _poly(zero, dose)
    elif method == "stale":
        predicted = [row[:] for row in native0]
    elif method == "wrong-sign":
        predicted = _poly(midpoint, -dose)
    elif method == "wrong-curvature":
        predicted = _poly(midpoint, dose, curvature_sign=-1.0)
    else:  # pragma: no cover - internal constant
        raise ValueError(f"unknown scalar method {method!r}")
    delta = _matrix_sub(predicted, native0)
    probability = [[math.exp(value) for value in row] for row in predicted]
    return {
        "pred_logpB": predicted,
        "logpB": predicted,
        "delta_logpB": delta,
        "pB": probability,
    }


def _method_forecasts(record: Mapping[str, Any], vm_artifacts, sites, arms):
    q0 = _capture(arms["00"], "q22", vm_artifacts)[0]
    k0 = _capture(arms["00"], "k22", vm_artifacts)[0]
    midpoint = _program(
        _dict(record.get("midpoint_program"), "midpoint program"),
        center=0.5,
        sites=sites,
        q0=q0,
        k0=k0,
    )
    zero = _program(
        _dict(record.get("zero_program"), "zero-centered program"),
        center=0.0,
        sites=sites,
        q0=q0,
        k0=k0,
    )
    _close_flat(
        midpoint["_matrices"]["logp_native0"],
        zero["_matrices"]["logp_native0"],
        label="program native-zero anchor",
    )
    forecast_rows = _dict(record.get("forecasts"), "scalar forecast rows")
    if set(forecast_rows) != {str(dose) for dose in DOSES}:
        raise ValueError("scalar forecast dose set changed")
    programs = {"midpoint": midpoint, "zero": zero}
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for dose in DOSES:
        dose_ref = _dict(forecast_rows[str(dose)], f"forecast dose {dose}")
        hydrated = _dict(_hydrate_tree(dose_ref, vm_artifacts), f"forecast dose {dose}")
        if set(hydrated) != set(METHODS):
            raise ValueError(f"scalar forecast method set changed at dose {dose}")
        expected[str(dose)] = {}
        for method in METHODS:
            supplied = _dict(hydrated[method], f"forecast {dose}/{method}")
            if supplied.get("method") != method:
                raise ValueError(f"forecast method label changed at {dose}/{method}")
            _same_number(supplied.get("dose"), dose, label=f"forecast dose {dose}/{method}")
            formula = _scalar_prediction(programs, dose, method)
            for key, values in formula.items():
                _close_flat(
                    supplied.get(key),
                    values,
                    label=f"forecast {dose}/{method}/{key}",
                )
            expected[str(dose)][method] = supplied
    return programs, expected


def _weight_metrics(prediction: Mapping[str, Any], true_logp, baseline_logp):
    pred_delta = prediction["delta_logpB"].float().reshape(-1)
    predicted_probability = prediction["pB"].float().reshape(-1)
    true_logp = true_logp.float().reshape(-1)
    delta = true_logp - baseline_logp.float().reshape(-1)
    residual = pred_delta - delta
    return {
        "log_change_rmse": float(residual.square().mean().sqrt()),
        "log_change_max_abs": float(residual.abs().max()),
        "true_log_change_rms": float(delta.square().mean().sqrt()),
        "probability_error": _error(predicted_probability, true_logp.exp()),
        "predicted_probability_in_range": bool(
            ((prediction["pB"] >= 0) & (prediction["pB"] <= 1)).all()
        ),
        "head_rows": [
            {
                "head_offset": index,
                "true_delta_logp": float(actual),
                "predicted_delta_logp": float(estimated),
                "error": float(estimated - actual),
            }
            for index, (actual, estimated) in enumerate(
                zip(delta, pred_delta, strict=True)
            )
        ],
    }


def _compare_measurement(actual: Any, expected: Any, *, label: str) -> None:
    """Compare saved measurements while treating backend exactness as diagnostic."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} fields changed")
        for key, expected_value in expected.items():
            if key == "exact":
                if type(actual.get(key)) is not bool:
                    raise ValueError(f"{label}.exact is malformed")
            else:
                _compare_measurement(actual.get(key), expected_value, label=f"{label}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length changed")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare_measurement(left, right, label=f"{label}[{index}]")
    elif isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise ValueError(f"{label} changed")
    elif isinstance(expected, (int, float)):
        _same_number(actual, float(expected), label=label, rtol=1e-4, atol=3e-6)
    elif actual != expected:
        raise ValueError(f"{label} changed")


def _verify_source_and_doses(bundle, arms, record, vm_artifacts, programs, predictions):
    import torch

    sites = bundle["sites"]
    zero = _capture(arms["00"], "attention19", vm_artifacts)[0]
    source_rows = _dict(
        _hydrate_tree(_dict(record.get("source_attention"), "source attention"), vm_artifacts),
        "source attention rows",
    )
    if set(source_rows) != {str(dose) for dose in DOSES}:
        raise ValueError("source-attention dose set changed")
    write = _payload(record.get("write_ref"), vm_artifacts, "source write")[0]
    _tensor_close(source_rows["0.0"], zero, label="native-zero source attention")
    _tensor_close(
        source_rows["1.0"].float() - source_rows["0.0"].float(),
        write.float(),
        label="source write relation",
    )

    q0 = _capture(arms["00"], "q22", vm_artifacts)[0]
    k0 = _capture(arms["00"], "k22", vm_artifacts)[0]
    scores0 = _capture(arms["00"], "scores22", vm_artifacts)[0]
    if q0.shape[1] % k0.shape[1]:
        raise ValueError("native query/KV head geometry is inconsistent")
    groups = q0.shape[1] // k0.shape[1]
    first_head = sites["B"]["kv_head"] * groups
    selected = slice(first_head, first_head + groups)
    row = sites["B"]["source_row"]
    baseline_logp = torch.log_softmax(scores0.float(), -1)[0, selected, 0, row]

    rows = _list(bundle.get("dose_rows"), "dose rows")
    if [float(_dict(item, "dose row").get("dose")) for item in rows] != list(DOSES):
        raise ValueError("dose row set changed")
    dose_summary = []
    for item, dose in zip(rows, DOSES, strict=True):
        item = _dict(item, f"dose row {dose}")
        arm_label = "00" if dose == 0.0 else f"A-{dose}"
        native = _capture(arms[arm_label], "scores22", vm_artifacts)[0]
        native_logp = torch.log_softmax(native.float(), -1)[0, selected, 0, row]
        supplied_methods = _dict(item.get("methods"), f"dose {dose} methods")
        if set(supplied_methods) != set(METHODS):
            raise ValueError(f"dose {dose} method set changed")
        for method in METHODS:
            metrics = _weight_metrics(
                predictions[str(dose)][method], native_logp, baseline_logp
            )
            saved_metrics = _dict(
                supplied_methods[method], f"dose {dose}/{method} saved metrics"
            )
            if set(saved_metrics) != set(metrics) | EXTENDED_WEIGHT_METRICS:
                raise ValueError(f"dose {dose}/{method} measurement fields changed")
            _compare_measurement(
                {key: saved_metrics[key] for key in metrics},
                metrics,
                label=f"dose {dose}/{method}",
            )

        smooth, _ = _payload(
            item.get("smooth_logp_ref"), vm_artifacts, f"dose {dose} smooth logp"
        )
        for method in METHODS:
            pred = predictions[str(dose)][method]["logpB"].float().reshape(-1).double()
            smooth_flat = smooth.float().reshape(-1).double()
            native_flat = native_logp.double()
            if pred.numel() != smooth_flat.numel() or pred.numel() != native_flat.numel():
                raise ValueError(f"dose {dose}/{method} FP64 comparison geometry changed")
            remainder = pred - smooth_flat
            profile = smooth_flat - native_flat
            total = pred - native_flat
            fp64 = {
                "smooth_remainder_rmse": float(remainder.square().mean().sqrt()),
                "smooth_native_profile_rmse": float(profile.square().mean().sqrt()),
                "mse_cross_term": float(2.0 * (remainder * profile).mean()),
                "fp64_total_rmse": float(total.square().mean().sqrt()),
                "fp64_identity_max_abs": float((total - remainder - profile).abs().max()),
                "fp64_mse_identity_residual": float(
                    total.square().mean()
                    - (
                        remainder.square().mean()
                        + profile.square().mean()
                        + 2.0 * (remainder * profile).mean()
                    )
                ),
            }
            for key, value in fp64.items():
                _same_number(
                    supplied_methods[method].get(key),
                    value,
                    label=f"dose {dose}/{method}/{key}",
                    rtol=1e-5,
                    # Inputs to this decomposition were captured as FP32;
                    # allow a few FP32 ulps for backend log-softmax replay.
                    atol=5e-7,
                )

        source_actual = _capture(arms[arm_label], "attention19", vm_artifacts)[0]
        _compare(
            item.get("local_source_error"),
            _error(source_rows[str(dose)], source_actual),
            label=f"dose {dose} local source error",
        )
        if item.get("native_capture") != arms[arm_label].get("capture"):
            raise ValueError(f"dose {dose} native capture binding changed")

        profile_tape = _dict(
            _dict(record.get("native_predictions"), "native predictions")[str(dose)],
            f"native prediction {dose}",
        )
        native_capture = _dict(arms[arm_label].get("capture"), "native capture")
        profile_errors = {}
        for key, reference in profile_tape.items():
            if (
                isinstance(reference, dict)
                and reference.get("schema") == "saturn-vm-payload-descriptor-v1"
                and key in native_capture
            ):
                predicted_tensor, _ = _payload(reference, vm_artifacts, f"profile {dose}/{key}")
                actual_tensor, _ = _payload(native_capture[key], vm_artifacts, f"native {dose}/{key}")
                if tuple(predicted_tensor.shape) == tuple(actual_tensor.shape):
                    profile_errors[key] = _error(predicted_tensor, actual_tensor)
        _compare(
            item.get("functional_profile_errors"),
            profile_errors,
            label=f"dose {dose} functional profile errors",
        )
        dense_logp = torch.log_softmax(
            _payload(profile_tape.get("scores22"), vm_artifacts, f"profile {dose} scores22")[0].float(),
            -1,
        )[0, selected, 0, row]
        dense_metric = float((dense_logp - native_logp).square().mean().sqrt())
        _same_number(
            item.get("functional_logp_rmse"),
            dense_metric,
            label=f"dose {dose} functional log-weight RMSE",
            rtol=1e-4,
            atol=3e-6,
        )
        dose_summary.append(
            {
                "dose": dose,
                "methods": {
                    method: supplied_methods[method] for method in METHODS
                },
                "functional_logp_rmse": dense_metric,
                "local_source_error": item["local_source_error"],
            }
        )
    return dose_summary


def _verify_arm(
    arm: Mapping[str, Any],
    *,
    label: str,
    parent_ref: str,
    parent: Mapping[str, Any],
    artifacts,
    vm_artifacts,
    manifest,
    contract: str,
    expects_span: bool,
) -> dict[str, Any]:
    arm = _dict(arm, f"arm {label}")
    if arm.get("name") != label or arm.get("parent_cut") != parent_ref:
        raise ValueError(f"arm {label} parent/name changed")
    trajectory = _list(arm.get("trajectory"), f"arm {label} trajectory")
    expected_count = 1 if label in {"A--0.25", "A-0.2", "A-0.6"} else 3
    if len(trajectory) != expected_count:
        raise ValueError(f"arm {label} native trajectory length changed")
    previous_ref, previous = parent_ref, parent
    for index, value in enumerate(trajectory):
        row = _dict(value, f"arm {label} step")
        if row.get("offset") != index:
            raise ValueError(f"arm {label} trajectory offset changed")
        token = _int(row.get("consumed_token_id"), f"arm {label} consumed token")
        if row.get("authority") != ("caller-supplied" if index == 0 else "native-greedy"):
            raise ValueError(f"arm {label} token authority changed")
        cut_ref = _ref(row.get("cut"), f"arm {label} cut")
        cut = _load_cut(cut_ref, artifacts, manifest, contract=contract)
        committed_parent = cut.get("parent_cut")
        if index == 0:
            staged_ref = _ref(committed_parent, f"arm {label} staged cut")
            staged = _load_cut(staged_ref, artifacts, manifest, contract=contract)
            if (
                staged.get("pending") != token
                or staged.get("token_ids") != previous.get("token_ids")
                or staged.get("label") != "supplied-input-embedding"
                or not _ancestral(staged_ref, previous_ref, artifacts, manifest, contract)
            ):
                raise ValueError(f"arm {label} supplied-input staging changed")
        elif committed_parent != previous_ref:
            raise ValueError(f"arm {label} step parent changed")
        if cut.get("token_ids") != list(previous.get("token_ids", ())) + [token]:
            raise ValueError(f"arm {label} token history changed")
        _verify_receipt(row.get("receipt_ref"), artifacts, vm_artifacts, row=row)
        logits, payload = _payload(cut.get("logits"), vm_artifacts, f"arm {label} logits")
        if row.get("logits_sha256") != payload.sha256 or cut.get("pending") != row.get("next_token_id"):
            raise ValueError(f"arm {label} logits/readout binding changed")
        import torch

        if int(torch.argmax(logits.float()[0]).item()) != row.get("next_token_id"):
            raise ValueError(f"arm {label} next-token readout changed")
        previous_ref, previous = cut_ref, cut
    if arm.get("end_cut") != previous_ref or not _sha(arm.get("closure_sha256")):
        raise ValueError(f"arm {label} endpoint/closure digest changed")

    replacements = _list(arm.get("span_replacements"), f"arm {label} replacements")
    expected_replacements = 4 if expects_span else 0
    if len(replacements) != expected_replacements:
        raise ValueError(f"arm {label} layer replacement count changed")
    if expects_span:
        first = trajectory[0]
        for layer, replacement in zip(range(19, 23), replacements, strict=True):
            replacement = _dict(replacement, f"arm {label} replacement")
            if (
                replacement.get("layer") != layer
                or replacement.get("position") != len(parent.get("token_ids", ()))
                or replacement.get("token_id") != first["consumed_token_id"]
                or replacement.get("next_layer") != layer + 1
            ):
                raise ValueError(f"arm {label} replaced-layer cursor changed")

    replaced = 4 if expects_span else 0
    native = _int(arm.get("native_layer_transitions"), f"arm {label} native transitions")
    arm_replaced = _int(arm.get("replaced_layer_transitions"), f"arm {label} replaced transitions")
    if arm_replaced != replaced or native != len(trajectory) * NUM_LAYERS - replaced:
        raise ValueError(f"arm {label} layer-transition accounting changed")
    return arm


def _verify_bundle(bundle: Mapping[str, Any], case_root: Path, artifacts, vm_artifacts,
                   manifest, program: FrozenReadProgram, *, answer_span,
                   events: list[Mapping[str, Any]]) -> dict[str, Any]:
    bundle = _dict(bundle, "receiver bundle")
    parent_ref = _ref(bundle.get("parent_cut"), "bundle parent")
    parent = _load_cut(parent_ref, artifacts, manifest)
    contract = parent.get("numerical_contract")
    if not _sha(contract):
        raise ValueError("bundle cut numerical contract is malformed")
    sites = _BASE._check_sites(
        _dict(bundle.get("sites"), "bundle sites"), program, parent, answer_span
    )
    arms = _dict(bundle.get("arms"), "bundle arms")
    expected_arms = {
        "00",
        "sham",
        "reinstall",
        "functional-zero",
        "A--0.25",
        "A-0.2",
        "A-0.6",
        "A-1.0",
        *CONSUMERS,
        "AB",
    }
    if set(arms) != expected_arms:
        raise ValueError("closure arm set changed")
    for label, arm in arms.items():
        _verify_arm(
            arm,
            label=label,
            parent_ref=parent_ref,
            parent=parent,
            artifacts=artifacts,
            vm_artifacts=vm_artifacts,
            manifest=manifest,
            contract=contract,
            expects_span=label == "functional-zero" or label in CONSUMERS,
        )
    _verify_acts(arms, sites, parent_ref, artifacts, vm_artifacts)
    _required_capture(arms["00"], vm_artifacts)
    _required_capture(arms["AB"], vm_artifacts)

    # Exact unedited controls are mechanics observations recorded by the
    # worker. Their parity outcomes are reported, not made into trend gates.
    for control in ("sham", "reinstall"):
        control_exact = (
            _signature(arms[control]) == _signature(arms["00"])
            and arms[control].get("closure_sha256") == arms["00"].get("closure_sha256")
        )
        if control_exact is False:
            raise ValueError(f"{control} closure control differs from its saved declaration")

    forecast_ref = _ref(bundle.get("forecast_ref"), "closure forecast")
    record = _artifact(forecast_ref, artifacts, "closure forecast")
    if record.get("schema") != FORECAST_SCHEMA or record != bundle.get("forecast"):
        raise ValueError("forecast artifact/report binding changed")
    if record.get("parent_cut") != parent_ref or record.get("sites") != bundle.get("sites"):
        raise ValueError("forecast parent/site binding changed")
    visibility = str(record.get("visibility", ""))
    if "native00 tape" not in visibility or "no edited receiver" not in visibility:
        raise ValueError("forecast visibility partition changed")
    if record.get("native_module_entries") != 0:
        raise ValueError("native module entry was recorded during forecast")
    if record.get("baseline_capture") != arms["00"].get("capture"):
        raise ValueError("forecast baseline capture binding changed")
    payload_count = _audit_payloads(record, vm_artifacts)
    hydrated_spans = _dict(
        _hydrate_tree(_dict(record.get("consumer_spans"), "consumer spans"), vm_artifacts),
        "consumer spans",
    )
    if set(hydrated_spans) != set(CONSUMERS):
        raise ValueError("consumer span method set changed")
    zero_span = _dict(
        _hydrate_tree(_dict(record.get("zero_span"), "zero span"), vm_artifacts),
        "zero span",
    )
    programs, predictions = _method_forecasts(record, vm_artifacts, sites, arms)
    dose_summary = _verify_source_and_doses(
        bundle, arms, record, vm_artifacts, programs, predictions
    )

    tape_summary = {}
    for method in CONSUMERS:
        arm = arms[method]
        span = hydrated_spans[method]
        binding = _dict(span.get("binding"), f"span {method} binding")
        tape = _dict(span.get("tape"), f"span {method} tape")
        first_token = arm["trajectory"][0]["consumed_token_id"]
        if (
            binding.get("parent_cut") != parent_ref
            or binding.get("token_id") != first_token
            or binding.get("sites") != bundle.get("sites")
            or binding.get("length") != len(parent.get("token_ids", ()))
            or binding.get("layers") != [19, 20, 21, 22]
        ):
            raise ValueError(f"consumer span {method} parent/token/site binding changed")
        if arm["trajectory"][0]["authority"] != "caller-supplied":
            raise ValueError(f"consumer span {method} does not start on the sealed input token")
        for layer in range(19, 23):
            for key, installed_key in (
                (f"input{layer}", f"consumed_input{layer}"),
                (f"exit{layer}", f"installed_exit{layer}"),
                (f"current_k{layer}", f"installed_current_k{layer}"),
                (f"current_v{layer}", f"installed_current_v{layer}"),
            ):
                if key not in tape:
                    raise ValueError(f"consumer span {method} lacks {key}")
                predicted = tape[key]
                actual, _ = _capture(arm, installed_key, vm_artifacts)
                _tensor_equal_check(actual, predicted, label=f"{method}/{installed_key}")

            # The registered full cache is the actual branch bank after cache
            # assignment. Its history remains the edited branch's own history;
            # its final row is supplied by the prospective sealed span.
            for component in ("k", "v"):
                bank, _ = _capture(
                    arm, f"installed_{component}{layer}", vm_artifacts
                )
                installed_current, _ = _capture(
                    arm,
                    f"installed_current_{component}{layer}",
                    vm_artifacts,
                )
                predicted_current = tape[f"current_{component}{layer}"]
                _tensor_equal_check(
                    bank[:, :, -1:, :],
                    installed_current,
                    label=f"{method}/installed {component}{layer} final row",
                )
                _tensor_equal_check(
                    installed_current,
                    predicted_current,
                    label=f"{method}/sealed current {component}{layer}",
                )

            expected_history_arm = "00" if method == "functional-zero" else "AB"
            installed_bank, _ = _capture(
                arm, f"installed_{'k'}{layer}", vm_artifacts
            )
            reference_bank, _ = _capture(
                arms[expected_history_arm], f"k{layer}", vm_artifacts
            )
            if tuple(installed_bank.shape) != tuple(reference_bank.shape):
                raise ValueError(f"{method} installed key-bank shape changed at L{layer}")
            _tensor_equal_check(
                installed_bank[:, :, :-1, :],
                reference_bank[:, :, :-1, :],
                label=f"{method} preserved key history L{layer}",
            )
            installed_values, _ = _capture(
                arm, f"installed_v{layer}", vm_artifacts
            )
            reference_values, _ = _capture(
                arms[expected_history_arm], f"v{layer}", vm_artifacts
            )
            if tuple(installed_values.shape) != tuple(reference_values.shape):
                raise ValueError(f"{method} installed value-bank shape changed at L{layer}")
            _tensor_equal_check(
                installed_values[:, :, :-1, :],
                reference_values[:, :, :-1, :],
                label=f"{method} preserved value history L{layer}",
            )

        target_errors = {}
        for name, native_name in (
            ("attention", "attention22"),
            ("mlp", "mlp22"),
            ("exit", "exit22"),
            ("input", "input22"),
        ):
            predicted = tape.get(f"{native_name}")
            if predicted is None:
                raise ValueError(f"consumer span {method} lacks {native_name}")
            actual, _ = _capture(arms["AB"], native_name, vm_artifacts)
            target_errors[name] = _error(predicted, actual)
        local_row = _dict(bundle.get("local_errors"), "local errors").get(method)
        local_row = _dict(local_row, f"local errors {method}")
        for name, error_value in target_errors.items():
            _compare(local_row.get(name), error_value, label=f"local {method}/{name}")
        _compare(
            local_row.get("injected_exit_exact"),
            _tensor_equal(
                tape["exit22"],
                _capture(arm, "installed_exit22", vm_artifacts)[0],
            ),
            label=f"local {method}/injected_exit_exact",
        )
        _compare(
            local_row.get("injected_current_k_exact"),
            _tensor_equal(
                tape["current_k22"],
                _capture(arm, "installed_current_k22", vm_artifacts)[0],
            ),
            label=f"local {method}/injected_current_k_exact",
        )
        tape_summary[method] = {
            "binding": binding,
            "installed_layers": [19, 20, 21, 22],
            "local_errors": target_errors,
        }

    zero_arm = arms["functional-zero"]
    zero_binding = _dict(zero_span.get("binding"), "zero span binding")
    zero_tape = _dict(zero_span.get("tape"), "zero span tape")
    if (
        zero_binding.get("parent_cut") != parent_ref
        or zero_binding.get("token_id") != zero_arm["trajectory"][0]["consumed_token_id"]
        or zero_binding.get("sites") != bundle.get("sites")
        or zero_binding.get("length") != len(parent.get("token_ids", ()))
        or zero_binding.get("layers") != [19, 20, 21, 22]
    ):
        raise ValueError("functional-zero span binding changed")
    for layer in range(19, 23):
        for key, installed_key in (
            (f"input{layer}", f"consumed_input{layer}"),
            (f"exit{layer}", f"installed_exit{layer}"),
            (f"current_k{layer}", f"installed_current_k{layer}"),
            (f"current_v{layer}", f"installed_current_v{layer}"),
        ):
            _tensor_equal_check(
                _capture(zero_arm, installed_key, vm_artifacts)[0],
                zero_tape[key],
                label=f"functional-zero/{installed_key}",
            )

    # These exact parity fields describe observed behavior only. A mismatch is
    # retained in the summary rather than used as a scientific rejection.
    zero_closure = (
        _signature(zero_arm) == _signature(arms["00"])
        and zero_arm.get("closure_sha256") == arms["00"].get("closure_sha256")
    )
    if bundle.get("functional_zero_closure_exact") is not zero_closure:
        raise ValueError("functional-zero closure observation changed")
    full_closure = (
        arms["functional-native"].get("closure_sha256") == arms["AB"].get("closure_sha256")
    )
    full_signature = _signature(arms["functional-native"]) == _signature(arms["AB"])
    if bundle.get("functional_AB_native_closure_exact") is not full_closure:
        raise ValueError("functional-native cache parity observation changed")
    if bundle.get("functional_AB_full_logit_signature_exact") is not full_signature:
        raise ValueError("functional-native logit parity observation changed")
    if bundle.get("native_A_assisted") is not False:
        raise ValueError("consumer forecast declared native-A output assistance")

    future = {}
    supplied_future = _dict(bundle.get("future_comparisons"), "future comparisons")
    for method in CONSUMERS:
        expected = _future(arms[method], arms["AB"], artifacts, vm_artifacts)
        _compare(supplied_future.get(method), expected, label=f"future {method}")
        future[method] = expected

    event = _verify_bundle_event(bundle, events)
    if event["physical_steps"] != record.get("physical_steps_at_seal"):
        raise ValueError("forecast seal physical-step count changed")
    return {
        "parent_cut": parent_ref,
        "sites": sites,
        "forecast_ref": forecast_ref,
        "payload_count": payload_count,
        "scalar_predictions": {
            "dose_set": list(DOSES),
            "methods": list(METHODS),
            "midpoint_center": 0.5,
            "zero_center": 0.0,
        },
        "dose_rows": dose_summary,
        "span_installations": tape_summary,
        "functional_zero_closure_exact": zero_closure,
        "functional_AB_native_closure_exact": full_closure,
        "functional_AB_full_logit_signature_exact": full_signature,
        "future_comparisons": future,
        "prediction_event": event,
        "derivative_compiler_independently_rerun": False,
    }


def _tensor_equal_check(actual, expected, *, label: str) -> None:
    if hasattr(expected, "schema") and isinstance(expected, dict):
        raise ValueError(f"{label} unexpectedly contains an unhydrated payload descriptor")
    if not _tensor_equal(actual, expected):
        raise ValueError(f"{label} differs from the sealed/installed tensor")


def _verify_bundle_event(bundle: Mapping[str, Any], events: list[Mapping[str, Any]]):
    forecast_ref = _ref(bundle.get("forecast_ref"), "forecast event reference")
    seals = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("kind") == "all-predictions-sealed"
        and event.get("ref") == forecast_ref
    ]
    starts = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("kind") == "native-AB-started"
        and event.get("forecast_ref") == forecast_ref
    ]
    edited = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("kind") == "edited-receiver-started"
        and event.get("forecast_ref") == forecast_ref
    ]
    if len(seals) != 1 or len(starts) != 1:
        raise ValueError("prediction seal or native AB start event is missing/duplicated")
    if {float(event.get("dose")) for _, event in edited} != set(DOSES[1:]):
        raise ValueError("edited receiver start event dose set changed")
    seal_index, seal = seals[0]
    start_index, start = starts[0]
    if not (seal_index < start_index and all(seal_index < index for index, _ in edited)):
        raise ValueError("edited receiver event preceded the saved forecast seal")
    physical = [event.get("physical_steps") for event in events]
    if any(type(step) is not int or step < 0 for step in physical):
        raise ValueError("prediction event physical-step count is malformed")
    if physical != sorted(physical):
        raise ValueError("prediction event physical-step order changed")
    return {
        "status": "valid",
        "seal_index": seal_index,
        "native_AB_index": start_index,
        "physical_steps": seal.get("physical_steps"),
        "native_AB_steps": start.get("physical_steps"),
        "edited_receiver_events": len(edited),
    }


def _verify_case(case: Mapping[str, Any], case_root: str | Path) -> dict[str, Any]:
    """Audit one saved report case; provided for root-side integration checks."""
    _finite(case)
    case = _dict(case, "case report")
    case_root = Path(case_root)
    bench_summary = _bench_read_only(case_root)
    artifacts, vm_artifacts, manifest_ref, manifest = _load_case(case_root)
    if case.get("model_identity") != manifest.get("identity"):
        raise ValueError("case model identity differs from the saved Bench manifest")
    if manifest.get("attention_backend") != "native-eager":
        raise ValueError("case attention backend changed")
    if case.get("semantic_cot_qualified") is not False:
        raise ValueError("case asserted an unsupported semantic-CoT claim")
    case_decl = _dict(case.get("case"), "case declaration")
    case_id = case_decl.get("id")
    if case_id not in CASE_SPLITS or case_decl.get("split") != CASE_SPLITS[case_id]:
        raise ValueError("case identity or split changed")
    program = FrozenReadProgram.standard()
    declared_program = FrozenReadProgram.from_dict(case.get("program"))
    if declared_program.to_dict() != program.to_dict():
        raise ValueError("case read-program declaration changed")

    first = _dict(case.get("first"), "first receiver bundle")
    first_parent_ref = _ref(first.get("parent_cut"), "first parent")
    first_parent = _load_cut(first_parent_ref, artifacts, manifest)
    first_span = _span(
        case.get("initial_answer_span"),
        "initial generated-answer span",
        len(first_parent["token_ids"]),
    )
    events = _list(case.get("prediction_events"), "prediction events")
    event_rows = [_dict(value, "prediction event") for value in events]
    first_summary = _verify_bundle(
        first,
        case_root,
        artifacts,
        vm_artifacts,
        manifest,
        program,
        answer_span=first_span,
        events=event_rows,
    )

    second = _dict(case.get("second_candidate_world"), "second receiver bundle")
    source_cut = _ref(case.get("second_world_source_cut"), "second world source cut")
    midpoint_arm = first["arms"].get("midpoint")
    if not isinstance(midpoint_arm, dict) or source_cut != midpoint_arm.get("end_cut"):
        raise ValueError("second receiver is not rooted in the first midpoint branch end cut")
    second_parent_ref = _ref(second.get("parent_cut"), "second parent")
    contract = first_parent.get("numerical_contract")
    if not _ancestral(second_parent_ref, source_cut, artifacts, manifest, contract):
        raise ValueError("second receiver parent is not descended from the midpoint branch")
    second_parent = _load_cut(second_parent_ref, artifacts, manifest, contract=contract)
    second_span = _span(
        case.get("second_answer_span"),
        "second generated-answer span",
        len(second_parent["token_ids"]),
    )
    second_summary = _verify_bundle(
        second,
        case_root,
        artifacts,
        vm_artifacts,
        manifest,
        program,
        answer_span=second_span,
        events=event_rows,
    )
    if bundle_events_unique(event_rows) != 2:
        raise ValueError("case must contain one sealed forecast event pair per receiver")

    steps = _int(case.get("physical_token_steps"), "case physical token steps")
    total_arms = sum(len(_list(arm.get("trajectory"), "arm trajectory"))
                     for bundle in (first, second)
                     for arm in _dict(bundle.get("arms"), "bundle arms").values())
    initial_native = _list(case.get("initial_native"), "initial native transitions")
    if len(initial_native) != 2:
        raise ValueError("case must retain its two native initial-answer transitions")
    # The final prompt token is stepped immediately before those two answer
    # tokens; prefill work is recorded separately and is not a token step.
    initial_last_cut_ref = _ref(initial_native[-1].get("cut"), "last initial native cut")
    initial_last_cut = _load_cut(initial_last_cut_ref, artifacts, manifest)
    if not _ancestral(
        first_parent_ref, initial_last_cut_ref, artifacts, manifest,
        initial_last_cut.get("numerical_contract"),
    ):
        raise ValueError("first receiver parent is not descended from initial generation")
    source_cut_state = _load_cut(source_cut, artifacts, manifest, contract=contract)
    first_append_steps = len(first_parent.get("token_ids", ())) - len(
        initial_last_cut.get("token_ids", ())
    )
    second_append_steps = len(second_parent.get("token_ids", ())) - len(
        source_cut_state.get("token_ids", ())
    )
    if first_append_steps < 0 or second_append_steps < 0:
        raise ValueError("follow-up append cut history shrank")
    # Prefill processes every prompt token except the final one. The final
    # prompt token, two generated answer tokens, follow-up prefixes, and each
    # arm trajectory are counted by Runner.step.
    expected_steps = (
        total_arms
        + len(initial_native)
        + 1
        + first_append_steps
        + second_append_steps
    )
    replaced = _int(case.get("replaced_layer_transitions"), "case replaced transitions")
    expected_replaced = sum(
        4
        for bundle in (first, second)
        for label in _dict(bundle.get("arms"), "bundle arms")
        if label == "functional-zero" or label in CONSUMERS
    )
    if replaced != expected_replaced:
        raise ValueError("case replaced-layer total differs from span events")
    observed_native = sum(
        _int(arm.get("native_layer_transitions"), "arm native transitions")
        for bundle in (first, second)
        for arm in _dict(bundle.get("arms"), "bundle arms").values()
    )
    if case.get("observed_native_layer_transitions_in_arms") != observed_native:
        raise ValueError("case observed native-layer counter differs from arm counters")
    expected_native = steps * NUM_LAYERS - replaced
    expected_arm_native = total_arms * NUM_LAYERS - replaced
    if case.get("native_layer_transitions") != expected_native or observed_native != expected_arm_native:
        raise ValueError("case physical layer accounting differs from trajectory/replacement rows")
    if steps != expected_steps:
        raise ValueError("case physical token steps differ from arm and initial trajectories")

    return {
        "case_id": case_id,
        "split": case_decl["split"],
        "mechanics_status": "valid",
        "instrument_status": "saved scalar laws and payloads audited",
        "terminal_status": "not-assessed",
        "bench": bench_summary,
        "manifest_ref": manifest_ref,
        "first": first_summary,
        "second": second_summary,
        "physical_token_steps": steps,
        "native_layer_transitions": expected_native,
        "replaced_layer_transitions": replaced,
        "limitations": [
            "the trained centered derivative compiler was not independently rerun",
            "native parity, score error, and continuation differences are retained observations",
            "semantic CoT and economical full-query claims are not assessed",
        ],
    }


def bundle_events_unique(events: list[Mapping[str, Any]]) -> int:
    return sum(event.get("kind") == "all-predictions-sealed" for event in events)


def _case_root(results_root: Path, case_id: str) -> Path:
    direct = results_root / case_id
    if (direct / "selected.json").is_file():
        return direct
    matches = [
        path.parent
        for path in results_root.rglob("selected.json")
        if path.parent.name == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot locate unique session tree for {case_id!r}")
    return matches[0]


def _verify_output_manifest(results_root: Path, custody: Mapping[str, Any]) -> dict[str, Any]:
    manifest_name = custody.get("manifest")
    if not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name:
        raise ValueError("output manifest path is unsafe")
    manifest_path = results_root / manifest_name
    if not manifest_path.is_file():
        raise ValueError("output manifest is missing")
    raw = manifest_path.read_bytes()
    if sha256(raw).hexdigest() != custody.get("manifest_sha256"):
        raise ValueError("output manifest digest differs from report custody")
    manifest = _dict(json.loads(raw), "output manifest")
    if manifest.get("schema") not in {
        "mdb-read-weight-closure-output-manifest-v1",
        "mdb-read-composition-output-manifest-v1",
        # The worker reuses the frozen support writer whose inventory schema
        # remains the original circuit-discovery name.
        "mdb-circuit-discovery-output-manifest-v1",
    }:
        raise ValueError("output manifest schema changed")
    entries = _list(manifest.get("files"), "output manifest files")
    declared: set[str] = set()
    total = 0
    for entry_value in entries:
        entry = _dict(entry_value, "output manifest entry")
        if set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("output manifest entry fields changed")
        relative = Path(_ref(entry.get("path"), "manifest path"))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in declared:
            raise ValueError("output manifest contains unsafe/duplicate paths")
        path = results_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or type(entry.get("bytes")) is not int
            or path.stat().st_size != entry["bytes"]
            or not _sha(entry.get("sha256"))
            or sha256(path.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"output file custody differs: {relative}")
        declared.add(relative.as_posix())
        total += entry["bytes"]
    if total != manifest.get("expanded_bytes") or len(entries) != manifest.get("file_count"):
        raise ValueError("output manifest totals changed")
    actual = {
        path.relative_to(results_root).as_posix()
        for path in results_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != declared | {manifest_name}:
        raise ValueError("output tree has unexpected or missing files")
    if custody.get("file_count") != manifest.get("file_count"):
        raise ValueError("custody file count differs from output manifest")
    if custody.get("expanded_bytes") != manifest.get("expanded_bytes"):
        raise ValueError("custody expanded byte count differs from output manifest")
    if custody.get("pretrained_weight_copies") is not False or manifest.get("pretrained_weight_copies") is not False:
        raise ValueError("result tree unexpectedly contains pretrained weight copies")
    return {
        "manifest_sha256": custody["manifest_sha256"],
        "file_count": len(entries),
        "expanded_bytes": total,
        "pretrained_weight_copies": False,
    }


def verify_report(report: Mapping[str, Any], results_root: str | Path) -> dict[str, Any]:
    _finite(report)
    report = _dict(report, "report")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("read-weight-closure report schema changed")
    if (
        report.get("native_A_output_assistance") is not False
        or report.get("economical_full_query_qualified") is not False
        or report.get("semantic_cot_qualified") is not False
        or report.get("receiver_count") != 8
    ):
        raise ValueError("report scope or bounded claim declarations changed")
    request_sha = _ref(report.get("request_sha256"), "report request digest")
    if not _sha(request_sha):
        raise ValueError("report request digest is malformed")
    source_hashes = _dict(report.get("source_hashes"), "report source hashes")
    if not source_hashes or any(not isinstance(name, str) or not name or not _sha(value)
                                for name, value in source_hashes.items()):
        raise ValueError("report source hash map is malformed")
    if not isinstance(report.get("job_id"), (str, int)) or not str(report.get("job_id")):
        raise ValueError("report job identity is malformed")
    results_root = Path(results_root)
    cases = _list(report.get("cases"), "report cases")
    if len(cases) != 4:
        raise ValueError("closure report must contain four cases")
    seen: set[str] = set()
    checked = []
    for value in cases:
        row = _dict(value, "report case")
        case_id = _dict(row.get("case"), "case declaration").get("id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError("case identity is missing or duplicated")
        seen.add(case_id)
        checked.append(_verify_case(row, _case_root(results_root, case_id)))
    if seen != set(CASE_SPLITS):
        raise ValueError("report case identities differ from the frozen assay")
    split_counts = {
        split: sum(item["split"] == split for item in checked)
        for split in ("development", "fresh")
    }
    if split_counts != {"development": 2, "fresh": 2}:
        raise ValueError("report development/fresh split allocation changed")
    steps = sum(item["physical_token_steps"] for item in checked)
    native_layers = sum(item["native_layer_transitions"] for item in checked)
    replaced_layers = sum(item["replaced_layer_transitions"] for item in checked)
    if report.get("physical_token_steps") != steps:
        raise ValueError("report physical token total differs from case rows")
    if report.get("native_layer_transitions") != native_layers:
        raise ValueError("report native layer total differs from case rows")
    if report.get("replaced_layer_transitions") != replaced_layers:
        raise ValueError("report replaced layer total differs from case rows")
    if report.get("prefill_calls") != 4:
        raise ValueError("report prefill count differs from the frozen specimen set")

    custody = _dict(report.get("artifact_custody"), "artifact custody")
    if custody.get("schema") != "mdb-read-weight-closure-custody-v1":
        raise ValueError("artifact custody schema changed")
    manifest_summary = _verify_output_manifest(results_root, custody)

    source_dependencies = _source_dependencies()
    dependency_digest = sha256(
        json.dumps(source_dependencies, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": VERIFY_SCHEMA,
        "report_schema": REPORT_SCHEMA,
        "request_sha256": request_sha,
        "source_hashes": source_hashes,
        "source_dependencies": source_dependencies,
        "source_dependency_sha256": dependency_digest,
        "mechanics_status": "valid",
        "instrument_status": "saved scalar laws and payloads audited",
        "trend_status": "all retained rows summarized",
        "terminal_status": "not-assessed",
        "derivative_compiler_independently_rerun": False,
        "case_count": len(checked),
        "receiver_count": report["receiver_count"],
        "split_counts": split_counts,
        "physical_token_steps": steps,
        "native_layer_transitions": native_layers,
        "replaced_layer_transitions": replaced_layers,
        "custody": manifest_summary,
        "cases": checked,
    }


def verify(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    report_path = root / "report.json" if root.is_dir() else root
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results_root = report_path.parent / "results"
    result = verify_report(report, results_root)

    request_path = report_path.parent / "request.json"
    if request_path.is_file():
        request = _dict(json.loads(request_path.read_text(encoding="utf-8")), "staged request")
        body = {key: value for key, value in request.items() if key != "request_sha256"}
        request_identity = sha256(
            json.dumps(body, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if (
            request.get("request_sha256") != report.get("request_sha256")
            or request_identity != request.get("request_sha256")
            or request.get("source_hashes") != report.get("source_hashes")
        ):
            raise ValueError("saved report/request/source bindings differ")
        result["request_binding"] = {
            "status": "valid",
            "request_sha256": request_identity,
            "source_hash_count": len(request.get("source_hashes", {})),
        }

    archive_path = report_path.parent / str(report.get("artifact_custody", {}).get("archive", ""))
    if archive_path.is_file():
        archive_digest = sha256(archive_path.read_bytes()).hexdigest()
        custody = report["artifact_custody"]
        if (
            archive_path.stat().st_size != custody.get("archive_bytes")
            or archive_digest != custody.get("archive_sha256")
        ):
            raise ValueError("collected archive differs from report custody")
        result["archive_custody"] = {
            "archive_sha256": archive_digest,
            "archive_bytes": archive_path.stat().st_size,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="collected run directory or report.json")
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
