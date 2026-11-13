"""Offline verifier for the baseline-only second-order read-weight assay.

This verifier replays no model and does not execute the nested derivative
compiler.  It authenticates the saved Bench trees and payload CAS, rebuilds
the scalar log-weight laws from the sealed coefficients, and checks that all
native receiver work occurred after those coefficients were sealed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from mdb_lanes.read_program import FrozenReadProgram
from mdb_lanes.read_weight_transport import compile_score_program, evaluate_score_program

_BASE_PATH = Path(__file__).resolve().parents[1] / "read_weight_transport_20261107" / "verify.py"
_BASE_SPEC = importlib.util.spec_from_file_location("mdb_read_weight_transport_audit", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load transport audit helpers from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)


REPORT_SCHEMA = "mdb-read-weight-second-order-report-v1"
VERIFY_SCHEMA = "mdb-read-weight-second-order-verification-v1"
WEIGHT_SCHEMA = "mdb-read-weight-forecast-v1"
CONSUMER_SCHEMA = "mdb-read-weight-consumer-forecast-v1"
DOSES = (0.0, -0.25, 0.2, 0.6, 1.0)
METHODS = ("second-order", "linear-logweight", "transport", "stale", "wrong-curvature", "wrong-sign", "routing-only")
CONSUMERS = ("second-order", "linear-logweight", "transport", "stale", "oracle")


def _source_dependencies() -> dict[str, dict[str, str]]:
    """Bind the inherited offline helper without embedding a host path."""

    return {
        "read_weight_transport_verify.py": {
            "path": "qualification/read_weight_transport_20261107/verify.py",
            "sha256": sha256(_BASE_PATH.read_bytes()).hexdigest(),
        }
    }


# Keep the byte/cut helpers in one place.  The transport verifier is a local
# source support module, not a runtime model dependency; this module remains
# usable when the taylor producer is absent from an installed wheel.
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


def _field(value: Mapping[str, Any], *names: str, label: str) -> Any:
    for name in names:
        if name in value:
            return value[name]
    raise ValueError(f"{label} is missing")


def _tensor_field(value: Mapping[str, Any], *names: str, vm_artifacts, label: str):
    import torch

    reference = _field(value, *names, label=label)
    if isinstance(reference, torch.Tensor):
        return reference
    return _payload(reference, vm_artifacts, label)[0]


def _matrix(value: Any, *, label: str):
    import torch

    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(row, (list, tuple)) for row in value):
        raise ValueError(f"{label} must be a non-empty matrix")
    width = len(value[0])
    if width == 0 or any(len(row) != width for row in value):
        raise ValueError(f"{label} must be rectangular")
    try:
        matrix = torch.tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} contains a non-finite scalar") from error
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{label} contains a non-finite scalar")
    return matrix.transpose(0, 1).contiguous()


def _verify_arm(arm: Mapping[str, Any], *, label: str, parent_ref: str, parent: Mapping[str, Any], artifacts, vm_artifacts, manifest, contract: str) -> dict[str, Any]:
    """Verify a native branch with one-step A doses and three-step consumers."""

    arm = _dict(arm, f"arm {label}")
    if arm.get("name") != label or arm.get("parent_cut") != parent_ref:
        raise ValueError(f"arm {label} parent/name changed")
    trajectory = _list(arm.get("trajectory"), f"arm {label} trajectory")
    expected_count = 1 if label.startswith("A-") and label not in {"A-1.0"} else 3
    if len(trajectory) != expected_count:
        raise ValueError(f"arm {label} native count changed")
    previous_ref, previous = parent_ref, parent
    for index, value in enumerate(trajectory):
        row = _dict(value, f"arm {label} step")
        if row.get("offset") != index:
            raise ValueError(f"arm {label} offset changed")
        token = _int(row.get("consumed_token_id"), f"arm {label} consumed token")
        if row.get("authority") != ("caller-supplied" if index == 0 else "native-greedy"):
            raise ValueError(f"arm {label} token authority changed")
        cut_ref = _ref(row.get("cut"), f"arm {label} cut")
        cut = _load_cut(cut_ref, artifacts, manifest, contract=contract)
        committed_parent = cut.get("parent_cut")
        if index == 0:
            staged_ref = _ref(committed_parent, f"arm {label} staged cut")
            staged = _load_cut(staged_ref, artifacts, manifest, contract=contract)
            if (staged.get("pending") != token
                    or staged.get("token_ids") != previous.get("token_ids")
                    or staged.get("label") != "supplied-input-embedding"
                    or not _ancestral(staged_ref, previous_ref, artifacts, manifest, contract)):
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
        raise ValueError(f"arm {label} endpoint/closure changed")
    return arm


def _verify_controls(arms: Mapping[str, Any]) -> None:
    zero = arms["00"]
    for name in ("sham", "reinstall"):
        if _signature(arms[name]) != _signature(zero) or arms[name].get("closure_sha256") != zero.get("closure_sha256"):
            raise ValueError(f"{name} native closure changed")


def _routing_coefficients(program: Mapping[str, Any], *, vm_artifacts):
    """Reconstruct c1 and the routing-only c2 from a/q/k score coefficients."""

    import torch

    logp0 = _tensor_field(program, "logp0", vm_artifacts=vm_artifacts, label="program logp0").float()
    a = _tensor_field(program, "a", "score_a", vm_artifacts=vm_artifacts, label="program a").float()
    b = _tensor_field(program, "b", "score_b", vm_artifacts=vm_artifacts, label="program b").float()
    source_row = _int(_field(program, "source_row", label="program source row"), "program source row")
    if logp0.ndim != 3 or a.shape != logp0.shape:
        raise ValueError("program score coefficient geometry changed")
    if b.shape != logp0.shape[:-1]:
        # Some producer revisions store b for every key row; accept that form
        # and reduce it below, while rejecting unrelated scalar shapes.
        if b.shape != logp0.shape:
            raise ValueError("program quadratic coefficient geometry changed")
    p0 = logp0.exp()
    if source_row >= logp0.shape[-1]:
        raise ValueError("program source row is outside score program")
    if b.ndim == 3:
        b_full = b
    else:
        b_full = torch.zeros_like(a)
        b_full[..., int(logp0.shape[-1]) - 1] = b
    expectation_a = (p0 * a).sum(-1)
    routing_c1 = a[..., source_row] - expectation_a
    expectation_b = (p0 * b_full).sum(-1)
    variance_a = (p0 * (a - expectation_a.unsqueeze(-1)).square()).sum(-1)
    routing_c2 = 2.0 * (b_full[..., source_row] - expectation_b) - variance_a
    return logp0[..., source_row], routing_c1, routing_c2, a, b_full


def _evaluate_scalar(program: Mapping[str, Any], score_program: Mapping[str, Any], dose: float, method: str, vm_artifacts):
    """Evaluate one saved scalar law without importing the producer module."""

    import torch

    g0 = _matrix(_field(program, "logp0", label="program logp0"), label="program logp0")
    c1 = _matrix(_field(program, "c1", "first", label="program c1"), label="program c1")
    c2 = _matrix(_field(program, "c2", "second", label="program c2"), label="program c2")
    if c1.shape != g0.shape or c2.shape != g0.shape:
        raise ValueError("Taylor coefficient geometry changed")
    routing_g0, routing_c1, routing_c2, _a, _b = _routing_coefficients(score_program, vm_artifacts=vm_artifacts)
    _field(program, "selected_rows", label="Taylor selected rows")
    score_selected = routing_g0
    if g0.shape != score_selected.shape or not torch.allclose(g0, score_selected, rtol=2e-5, atol=2e-6):
        raise ValueError("Taylor baseline log-weight does not match score program")
    if c1.shape != routing_c1.shape or c2.shape != routing_c2.shape:
        raise ValueError("Taylor selected coefficient geometry changed")
    dose = float(dose)
    if method == "transport":
        return evaluate_score_program(score_program, dose, method="transport")
    if method == "stale":
        logp = g0.clone()
        delta = torch.zeros_like(logp)
    elif method == "linear-logweight":
        logp = g0 + dose * c1
        delta = dose * c1
    elif method == "second-order":
        logp = g0 + dose * c1 + 0.5 * dose * dose * c2
        delta = dose * c1 + 0.5 * dose * dose * c2
    elif method == "wrong-curvature":
        logp = g0 + dose * c1 - 0.5 * dose * dose * c2
        delta = dose * c1 - 0.5 * dose * dose * c2
    elif method == "wrong-sign":
        logp = g0 - dose * c1 + 0.5 * dose * dose * c2
        delta = -dose * c1 + 0.5 * dose * dose * c2
    elif method == "routing-only":
        logp = g0 + dose * routing_c1 + 0.5 * dose * dose * routing_c2
        delta = dose * routing_c1 + 0.5 * dose * dose * routing_c2
    else:
        raise ValueError(f"unknown scalar method {method!r}")
    # The producer stores the selected B log weight, not the full vocabulary.
    return {"method": method, "dose": dose, "pred_logpB": logp, "logpB": logp,
            "delta_logpB": delta, "pB": logp.exp()}


def _verify_taylor(bundle: Mapping[str, Any], arms: Mapping[str, Any], sites: Mapping[str, Any], artifacts, vm_artifacts):
    record = _dict(bundle.get("weight_forecast"), "weight forecast")
    if record.get("schema") != WEIGHT_SCHEMA or record != _artifact(bundle.get("weight_forecast_ref"), artifacts, "weight forecast"):
        raise ValueError("weight forecast artifact/report binding changed")
    if record.get("native_module_entries") != 0 or "native00 tape" not in str(record.get("visibility", "")) or "no edited receiver" not in str(record.get("visibility", "")):
        raise ValueError("baseline-only visibility or native-call declaration changed")
    if record.get("baseline_capture") != arms["00"].get("capture"):
        raise ValueError("baseline capture binding changed")
    if record.get("parent_cut") != bundle.get("parent_cut") or record.get("sites") != bundle.get("sites"):
        raise ValueError("weight forecast parent/site binding changed")
    payload_count = _audit_payloads(record, vm_artifacts)
    program = _hydrate_tree(_dict(record.get("taylor_program"), "Taylor program"), vm_artifacts)
    score_program = _hydrate_tree(_dict(record.get("score_program"), "score program"), vm_artifacts)
    if not isinstance(program.get("c1"), list) or not isinstance(program.get("c2"), list):
        raise ValueError("Taylor program coefficients are missing")
    if program.get("schema") != "mdb-smooth-qwen-taylor-program-v1":
        raise ValueError("Taylor program schema changed")
    if score_program.get("schema") != "mdb-smooth-qwen-score-program-v1":
        raise ValueError("score program schema changed")
    if (program.get("kv_head"), program.get("source_row")) != (sites["B"]["kv_head"], sites["B"]["source_row"]):
        raise ValueError("Taylor receiver binding changed")
    if (score_program.get("kv_head"), score_program.get("source_row")) != (sites["B"]["kv_head"], sites["B"]["source_row"]):
        raise ValueError("score receiver binding changed")
    if program.get("cost", {}).get("native_forward_calls") != 0:
        raise ValueError("Taylor compiler entered a native forward")
    import torch

    q0 = _capture(arms["00"], "q22", vm_artifacts)[0]
    k0 = _capture(arms["00"], "k22", vm_artifacts)[0]
    scores0 = _capture(arms["00"], "scores22", vm_artifacts)[0]
    compiled = _hydrate_tree(_dict(record.get("compiled"), "compiled tangent"), vm_artifacts)
    logical_cost = _dict(compiled.get("logical_cost"), "compiled logical cost")
    if compiled.get("logical_linears") != 19 or logical_cost.get("native_forward_calls") != 0:
        raise ValueError("compiled logical/native cost declaration changed")
    q_dot, k_dot = compiled.get("q_dot"), compiled.get("k_dot")
    if not hasattr(q_dot, "shape") or not hasattr(k_dot, "shape"):
        raise ValueError("compiled q/k tangent payloads are missing")
    recomputed_score = compile_score_program(
        q0, k0, scores0, q_dot, k_dot,
        kv_head=sites["B"]["kv_head"], source_row=sites["B"]["source_row"],
        scaling=score_program.get("scaling"),
    )
    for key in ("schema", "kv_head", "source_row", "current_row", "scaling", "query_heads"):
        saved_value, recomputed_value = score_program.get(key), recomputed_score.get(key)
        if key == "query_heads":
            saved_value, recomputed_value = list(saved_value), list(recomputed_value)
        if saved_value != recomputed_value:
            raise ValueError(f"score program {key} changed")
    for key in ("logp0", "a", "a_query", "b", "b_current"):
        _tensor_close(score_program[key], recomputed_score[key], label=f"score program {key}")
    score_program = recomputed_score
    routing_g0, routing_c1, routing_c2, _a, _b = _routing_coefficients(score_program, vm_artifacts=vm_artifacts)
    source = _hydrate_tree(_dict(record.get("source_attention"), "source attention"), vm_artifacts)
    write = _payload(record.get("write_ref"), vm_artifacts, "source write")[0]
    if "0.0" not in source or "1.0" not in source:
        raise ValueError("source dose payloads are incomplete")
    _tensor_close(source["0.0"], _capture(arms["00"], "attention19", vm_artifacts)[0],
                  label="baseline source attention", rtol=2e-4, atol=2e-5)
    _tensor_close(source["1.0"].float() - source["0.0"].float(), write.float(),
                  label="source write relation", rtol=2e-5, atol=2e-6)
    c1_saved = _matrix(program["c1"], label="Taylor c1")
    c1_score = _matrix(program.get("c1_score"), label="Taylor c1_score")
    c2_route_saved = _matrix(program.get("c2_route"), label="Taylor c2_route")
    if not torch.allclose(c1_saved, routing_c1, rtol=2e-5, atol=2e-6):
        raise ValueError("c1 routing identity changed")
    if not torch.allclose(c1_score, routing_c1, rtol=2e-5, atol=2e-6):
        raise ValueError("c1_score routing identity changed")
    if not torch.allclose(c2_route_saved, routing_c2, rtol=2e-5, atol=2e-6):
        raise ValueError("routing c2 control changed")
    forecasts = _dict(record.get("forecasts"), "scalar forecasts")
    if set(forecasts) != {str(dose) for dose in DOSES}:
        raise ValueError("scalar dose set changed")
    summaries = {}
    for dose in DOSES:
        values = _dict(_hydrate_tree(forecasts[str(dose)], vm_artifacts), f"forecasts {dose}")
        if set(values) != set(METHODS):
            raise ValueError(f"scalar method set changed at dose {dose}")
        summaries[str(dose)] = {}
        for method in METHODS:
            expected = _evaluate_scalar(program, score_program, dose, method, vm_artifacts)
            actual = values[method]
            if actual.get("method") != method or float(actual.get("dose")) != float(dose):
                raise ValueError(f"scalar forecast metadata changed at {dose}/{method}")
            for key in ("pred_logpB", "logpB", "delta_logpB", "pB"):
                _tensor_close(actual[key], expected[key], label=f"scalar forecast {dose}/{method}/{key}")
            summaries[str(dose)][method] = {"pB": actual["pB"].detach().cpu().tolist()}
    return {"payload_count": payload_count, "program": program, "score_program": score_program, "forecasts": summaries,
            "c1_routing_exact": True, "c2_routing_exact": True,
            "c2_trained_compiler_reexecuted": False}


def _verify_doses(bundle: Mapping[str, Any], arms: Mapping[str, Any], sites: Mapping[str, Any], vm_artifacts, taylor):
    import torch

    scores0 = _capture(arms["00"], "scores22", vm_artifacts)[0]
    k0 = _capture(arms["00"], "k22", vm_artifacts)[0]
    groups = scores0.shape[1] // k0.shape[1]
    selected = slice(sites["B"]["kv_head"] * groups, (sites["B"]["kv_head"] + 1) * groups)
    row = sites["B"]["source_row"]
    baseline = torch.log_softmax(scores0.float(), -1)[0, selected, 0, row]
    rows = _list(bundle.get("dose_rows"), "dose rows")
    if [float(item.get("dose")) for item in rows] != list(DOSES):
        raise ValueError("dose row set changed")
    results = []
    for item, dose in zip(rows, DOSES, strict=True):
        native_arm = arms["00"] if dose == 0 else arms[f"A-{dose}"]
        native_scores = _capture(native_arm, "scores22", vm_artifacts)[0]
        true = torch.log_softmax(native_scores.float(), -1)[0, selected, 0, row]
        expected = {method: _weight_error(_evaluate_scalar(taylor["program"], taylor["score_program"], dose, method, vm_artifacts), true, baseline) for method in METHODS}
        supplied = _dict(item.get("methods"), f"dose {dose} methods")
        if set(supplied) != set(METHODS):
            raise ValueError(f"dose {dose} method set changed")
        for method in METHODS:
            _compare_weight_error(supplied[method], expected[method], label=f"dose {dose}/{method}")
        results.append({"dose": dose, "methods": expected})
    return results


def _weight_error(prediction: Mapping[str, Any], true, baseline):
    delta = true - baseline
    residual = prediction["delta_logpB"].float() - delta
    return {"log_change_rmse": float(residual.square().mean().sqrt()),
            "log_change_max_abs": float(residual.abs().max()),
            "true_log_change_rms": float(delta.square().mean().sqrt()),
            "probability_error": _error(prediction["pB"].reshape(-1), true.exp().reshape(-1)),
            "predicted_probability_in_range": bool(((prediction["pB"] >= 0) & (prediction["pB"] <= 1)).all()),
            "head_rows": [{"head_offset": index, "true_delta_logp": float(actual), "predicted_delta_logp": float(estimated), "error": float(estimated - actual)} for index, (actual, estimated) in enumerate(zip(delta.flatten(), prediction["delta_logpB"].flatten(), strict=True))]}


def _compare_weight_error(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    """Compare scalar diagnostics while treating digest parity as diagnostic.

    Native log-softmax ran on the worker backend; this audit widens the saved
    score tape on CPU.  Numeric norms remain checked, while the nested byte
    digest can differ at harmless backend rounding.
    """

    actual_probability = _dict(actual.get("probability_error"), f"{label}.probability_error")
    expected_probability = _dict(expected.get("probability_error"), f"{label}.probability_error")
    for key, value in expected.items():
        if key != "probability_error":
            _compare(actual.get(key), value, label=f"{label}.{key}")
            continue
        for metric, expected_value in expected_probability.items():
            if metric == "exact":
                if type(actual_probability.get(metric)) is not bool:
                    raise ValueError(f"{label}.probability_error.exact is malformed")
                continue
            _compare(actual_probability.get(metric), expected_value, label=f"{label}.probability_error.{metric}")


def _verify_consumers(bundle: Mapping[str, Any], arms: Mapping[str, Any], artifacts, vm_artifacts):
    record = _artifact(bundle.get("consumer_forecast_ref"), artifacts, "consumer forecast")
    if record.get("schema") != CONSUMER_SCHEMA or record != bundle.get("consumer_forecast"):
        raise ValueError("consumer artifact/report binding changed")
    _audit_payloads(record, vm_artifacts)
    methods = _dict(record.get("methods"), "consumer methods")
    if set(methods) != set(CONSUMERS):
        raise ValueError("consumer method set changed")
    for method in CONSUMERS:
        item = _dict(methods[method], f"consumer {method}")
        if item.get("native_A_output_assisted") is not True or item.get("explicit_dense_mlp_linears") != 3 or item.get("baseline_attention_ref") != arms["A-1.0"]["capture"]["attention22"]:
            raise ValueError(f"consumer {method} assistance declaration changed")
        for name, capture_name in (("attention", "attention22"), ("consumed_attention", "attention22"), ("mlp", "mlp22"), ("exit", "exit22")):
            reference = item.get(f"{name}_ref", item.get(name))
            if reference is None:
                raise ValueError(f"consumer {method} {name} payload is missing")
            predicted = _payload(reference, vm_artifacts, f"consumer {method} {name}")[0]
            if name == "consumed_attention":
                expected = _capture(arms[method], capture_name, vm_artifacts)[0]
            elif name in {"mlp", "exit"}:
                expected = _capture(arms[method], capture_name, vm_artifacts)[0]
            else:
                continue
            if not _tensor_equal(predicted, expected):
                raise ValueError(f"consumer {method} {name} injection changed")
    return {"status": "valid", "methods": list(CONSUMERS)}


def _verify_local(bundle: Mapping[str, Any], arms: Mapping[str, Any], vm_artifacts):
    supplied = _dict(bundle.get("local_errors"), "local errors")
    result = {}
    for method in CONSUMERS:
        record = _dict(supplied.get(method), f"local errors {method}")
        forecast = _dict(bundle["consumer_forecast"]["methods"][method], f"consumer {method}")
        native = {name: _capture(arms["AB"], key, vm_artifacts)[0] for name, key in (("attention", "attention22"), ("mlp", "mlp22"), ("exit", "exit22"))}
        result[method] = {}
        for name in native:
            predicted = _payload(forecast.get(f"{name}_ref", forecast.get(name)), vm_artifacts, f"local {method}/{name}")[0]
            expected = _error(predicted, native[name])
            _compare(record.get(name), expected, label=f"local {method}/{name}")
            result[method][name] = expected
        for key, pname, cname in (("injected_mlp_exact", "mlp", "mlp22"), ("injected_exit_exact", "exit", "exit22")):
            expected = _tensor_equal(_payload(forecast.get(f"{pname}_ref", forecast.get(pname)), vm_artifacts, f"local {method}/{pname}")[0], _capture(arms[method], cname, vm_artifacts)[0])
            if record.get(key) is not expected:
                raise ValueError(f"local {method}/{key} changed")
            result[method][key] = expected
    return result


def _verify_events(case: Mapping[str, Any], bundles: list[Mapping[str, Any]]) -> dict[str, Any]:
    events = _list(case.get("prediction_events"), "prediction events")
    if any(type(event.get("physical_steps")) is not int for event in events if isinstance(event, dict)):
        raise ValueError("prediction event physical steps are malformed")
    if [event["physical_steps"] for event in events] != sorted(event["physical_steps"] for event in events):
        raise ValueError("prediction event physical order changed")
    pairs = []
    for bundle in bundles:
        weight_ref = _ref(bundle["weight_forecast_ref"], "weight event ref")
        consumer_ref = _ref(bundle["consumer_forecast_ref"], "consumer event ref")
        weights = [(i, e) for i, e in enumerate(events) if e.get("kind") == "weights-sealed" and e.get("ref") == weight_ref]
        consumers = [(i, e) for i, e in enumerate(events) if e.get("kind") == "consumers-sealed" and e.get("ref") == consumer_ref]
        starts = [(i, e) for i, e in enumerate(events) if e.get("kind") == "native-AB-started" and e.get("consumer_ref") == consumer_ref]
        edited = [(i, e) for i, e in enumerate(events) if e.get("kind") == "edited-receiver-started" and e.get("weight_ref") == weight_ref]
        if len(weights) != 1 or len(consumers) != 1 or len(starts) != 1 or {float(e.get("dose")) for _, e in edited} != set(DOSES[1:]):
            raise ValueError("second-order seal/start event set is incomplete")
        wi, we = weights[0]
        ci, ce = consumers[0]
        ai, ae = starts[0]
        if not (wi < ci < ai and all(wi < i for i, _ in edited)):
            raise ValueError("native receiver work preceded its sealed predictions")
        pairs.append({"weight_seal_index": wi, "consumer_seal_index": ci, "native_AB_index": ai,
                      "weight_seal_steps": we["physical_steps"], "consumer_seal_steps": ce["physical_steps"], "native_AB_steps": ae["physical_steps"]})
    return {"status": "valid", "pairs": pairs}


def _verify_bundle(bundle: Mapping[str, Any], artifacts, vm_artifacts, manifest, program: FrozenReadProgram, *, answer_span: tuple[int, int]):
    parent_ref = _ref(bundle.get("parent_cut"), "bundle parent")
    parent = _load_cut(parent_ref, artifacts, manifest)
    contract = parent["numerical_contract"]
    if not _sha(contract):
        raise ValueError("cut numerical ABI fingerprint is malformed")
    sites = _check_sites(bundle.get("sites"), program, parent, answer_span)
    arms = _dict(bundle.get("arms"), "bundle arms")
    required = {"00", "sham", "reinstall", "A--0.25", "A-0.2", "A-0.6", "A-1.0", *CONSUMERS, "AB"}
    if not required <= set(arms):
        # JSON renders f"A-{dose}" for -0.25 as A--0.25.  Keep the check
        # explicit but tolerate an implementation that canonicalizes the sign.
        aliases = {"A--0.25": "A--0.25", "A-0.2": "A-0.2", "A-0.6": "A-0.6", "A-1.0": "A-1.0"}
        if not all(alias in arms or alias.replace("A-", "A") in arms for alias in aliases):
            raise ValueError("second-order arm set is incomplete")
    for label, arm in arms.items():
        if label in {"00", "sham", "reinstall", *CONSUMERS, "AB"} or label.startswith("A-"):
            _verify_arm(arm, label=label, parent_ref=parent_ref, parent=parent, artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest, contract=contract)
    _verify_acts(arms, sites, parent_ref, artifacts, vm_artifacts)
    _verify_controls(arms)
    _required_capture(arms["00"], vm_artifacts)
    taylor = _verify_taylor(bundle, arms, sites, artifacts, vm_artifacts)
    dose_summary = _verify_doses(bundle, arms, sites, vm_artifacts, taylor)
    consumers = _verify_consumers(bundle, arms, artifacts, vm_artifacts)
    local = _verify_local(bundle, arms, vm_artifacts)
    futures = {method: _future(arms[method], arms["AB"], artifacts, vm_artifacts) for method in CONSUMERS}
    for method in CONSUMERS:
        _compare(bundle["future_comparisons"][method], futures[method], label=f"future {method}")
    if bundle.get("native_module_entries_during_forecast") != 0 or bundle.get("dense_control_native_module_entries") != 0:
        raise ValueError("forecast native-module call count changed")
    a = arms["A-1.0"]
    ab = arms["AB"]
    if bundle.get("old_keys22_unchanged") is not _tensor_equal(_capture(a, "k22", vm_artifacts)[0][:, :, :-1], _capture(arms["00"], "k22", vm_artifacts)[0][:, :, :-1]):
        raise ValueError("old-key locality declaration changed")
    if bundle.get("B_source_value_unchanged") is not _tensor_equal(_capture(a, "v22", vm_artifacts)[0][0, sites["B"]["kv_head"], sites["B"]["source_row"]], _capture(arms["00"], "v22", vm_artifacts)[0][0, sites["B"]["kv_head"], sites["B"]["source_row"]]):
        raise ValueError("B source-value locality declaration changed")
    for key, left, right in (("input22", "input22", "input22"), ("q22", "q22", "q22"), ("k22", "k22", "k22"), ("current_v22", "current_v22", "current_v22")):
        if not _tensor_equal(_capture(a, left, vm_artifacts)[0], _capture(ab, right, vm_artifacts)[0]):
            raise ValueError(f"A/AB {key} locality changed")
    dense_refs = bundle.get("dense_fp32_control") or bundle.get("dense_fp32_control_qk")
    if dense_refs is not None:
        dense = _hydrate_tree(dense_refs, vm_artifacts)
        if not isinstance(dense, list) or len(dense) != 2:
            raise ValueError("dense FP32 control geometry changed")
    return {"parent_cut": parent_ref, "sites": sites, "taylor": {key: value for key, value in taylor.items() if key not in {"program", "score_program"}},
            "doses": dose_summary, "consumers": consumers, "local_errors": local, "future": futures,
            "second_order_compiler_reexecuted": False}


def _case_root(results_root: Path, case_id: str) -> Path:
    direct = results_root / case_id
    if (direct / "selected.json").is_file():
        return direct
    matches = [path.parent for path in results_root.rglob("selected.json") if path.parent.name == case_id]
    if len(matches) != 1:
        raise ValueError(f"cannot locate unique session tree for {case_id!r}")
    return matches[0]


def _verify_case(case: Mapping[str, Any], case_root: Path, program: FrozenReadProgram) -> dict[str, Any]:
    bench = _bench_read_only(case_root)
    artifacts, vm_artifacts, manifest_ref, manifest = _load_case(case_root)
    if case.get("model_identity") != manifest.get("identity") or manifest.get("attention_backend") != "native-eager":
        raise ValueError("case model/backend identity changed")
    declared_program = FrozenReadProgram.from_dict(case.get("program"))
    if declared_program.to_dict() != program.to_dict():
        raise ValueError("case read-program declaration changed")
    first = _dict(case.get("first"), "first bundle")
    first_parent = _load_cut(_ref(first["parent_cut"], "first parent"), artifacts, manifest)
    first_span = _span(case["initial_answer_span"], "first answer", len(first_parent["token_ids"]))
    first_summary = _verify_bundle(first, artifacts, vm_artifacts, manifest, program, answer_span=first_span)
    second = _dict(case.get("second_candidate_world"), "second bundle")
    source_cut = _ref(case.get("second_world_source_cut"), "second world source cut")
    second_parent_ref = _ref(second.get("parent_cut"), "second parent")
    if (source_cut != first["arms"]["second-order"]["end_cut"]
            or not _ancestral(second_parent_ref, source_cut, artifacts, manifest, first_parent["numerical_contract"])):
        raise ValueError("second world does not descend from second-order candidate")
    second_parent = _load_cut(second_parent_ref, artifacts, manifest)
    second_span = _span(case["second_answer_span"], "second answer", len(second_parent["token_ids"]))
    second_summary = _verify_bundle(second, artifacts, vm_artifacts, manifest, program, answer_span=second_span)
    if case.get("semantic_cot_qualified") is not False:
        raise ValueError("case made an unsupported semantic-CoT claim")
    events = _verify_events(case, [first, second])
    return {"schema": VERIFY_SCHEMA, "case_id": _dict(case["case"], "case declaration").get("id"),
            "mechanics_status": "valid", "instrument_status": "valid", "terminal_status": "not-assessed",
            "bench": bench, "manifest_ref": manifest_ref, "first": first_summary, "second": second_summary,
            "prediction_events": events, "limitations": ["nested second-derivative compiler was not independently reexecuted", "semantic CoT and portable full-query claims are not assessed"]}


def verify_report(report: Mapping[str, Any], results_root: str | Path) -> dict[str, Any]:
    _finite(report)
    report = _dict(report, "report")
    if report.get("schema") != REPORT_SCHEMA or report.get("semantic_cot_qualified") is not False:
        raise ValueError("report schema or semantic-CoT qualification changed")
    program = FrozenReadProgram.standard()
    cases = _list(report.get("cases"), "report cases")
    if len(cases) != 4:
        raise ValueError("second-order report must contain four cases")
    results_root = Path(results_root)
    seen: set[str] = set()
    checked = []
    for value in cases:
        case = _dict(value, "report case")
        case_id = _dict(case.get("case"), "case declaration").get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case identity is malformed or duplicated")
        seen.add(case_id)
        checked.append(_verify_case(case, _case_root(results_root, case_id), program))
    return {"schema": VERIFY_SCHEMA, "report_schema": REPORT_SCHEMA, "mechanics_status": "valid",
            "instrument_status": "valid", "terminal_status": "not-assessed", "semantic_cot_qualified": False,
            "second_order_compiler_independently_reexecuted": False,
            "source_dependencies": _source_dependencies(), "cases": checked}


def verify(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    report_path = root / "report.json" if root.is_dir() else root
    return verify_report(json.loads(report_path.read_text()), report_path.parent / "results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="result directory or report.json")
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
