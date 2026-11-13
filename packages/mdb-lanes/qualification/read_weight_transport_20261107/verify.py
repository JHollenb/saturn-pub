"""Offline custody and mechanics audit for the read-weight transport assay.

The worker intentionally separates a baseline-only score forecast from the
later native receiver.  This verifier checks that separation from the saved
session trees.  It does not load the model, retrieve parameters, run a JVP, or
turn a forecast error or answer change into a scientific gate.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from mdb_lanes.bench_circuit_discovery_verify import (
    _close_number,
    _cut,
    _tensor_digest,
    _verify_receipt,
)
from mdb_lanes.bench_hidden_memory import verify_write_act
from mdb_lanes.bench_verify import verify as verify_bench
from mdb_lanes.circuit_runtime import DirectoryArtifactStore
from mdb_lanes.model_os_contracts import content_digest
from mdb_lanes.read_program import FrozenReadProgram
from mdb_lanes.read_weight_transport import compile_score_program, evaluate_score_program

REPORT_SCHEMA = "mdb-read-weight-transport-report-v1"
VERIFY_SCHEMA = "mdb-read-weight-transport-verification-v1"
WEIGHT_SCHEMA = "mdb-read-weight-forecast-v1"
CONSUMER_SCHEMA = "mdb-read-weight-consumer-forecast-v1"
DOSES = (0.0, 0.25, 0.5, 1.0)
METHODS = ("transport", "linear-logweight", "stale", "wrong-sign", "query-only")
CONSUMERS = ("transport", "linear-logweight", "stale", "oracle")
_PAYLOAD_SCHEMA = "saturn-vm-payload-descriptor-v1"


def _finite(value: Any, path: str = "value") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def _dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _ref(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} reference is missing")
    return value


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} is malformed")
    return value


def _load_case(case_root: Path):
    artifacts = DirectoryArtifactStore(case_root / "artifacts")
    vm_artifacts = DirectoryArtifactStore(case_root / "vm" / "artifacts")
    selected = json.loads((case_root / "selected.json").read_text())
    manifest_ref = _ref(selected.get("manifest"), "session manifest")
    manifest = _dict(artifacts.get(manifest_ref), "session manifest")
    return artifacts, vm_artifacts, manifest_ref, manifest


def _bench_read_only(case_root: Path) -> dict[str, Any]:
    """Run Bench's public verifier on a disposable copy.

    A failed public verification may publish a diagnostic byte-audit.  The
    recovered result tree remains immutable by putting that possible write in
    a temporary copy.
    """

    with tempfile.TemporaryDirectory(prefix="mdb-read-weight-verify-") as directory:
        copy_root = Path(directory) / case_root.name
        shutil.copytree(case_root, copy_root, symlinks=False)
        return verify_bench(copy_root)


def _payload(reference: Any, vm_artifacts, label: str):
    """Read one tensor through PayloadStore.audit, never hydrate."""

    import torch

    from mdb_lanes.vm_payload_cache import PayloadRef, PayloadStore

    try:
        payload = PayloadRef.from_dict(reference)
        store = PayloadStore(vm_artifacts.root.parent / "radix" / "cas", read_only=True)
        raw = store.audit(payload).payload
        if sha256(raw).hexdigest() != payload.sha256:
            raise ValueError("payload digest mismatch")
        dtype = getattr(torch, payload.dtype.removeprefix("torch."))
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(payload.shape)).clone()
    except Exception as error:
        raise ValueError(f"{label} payload is malformed") from error
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{label} payload is non-finite")
    return tensor, payload


def _audit_payloads(value: Any, vm_artifacts, seen: set[str] | None = None) -> int:
    """Audit every payload descriptor reachable from a JSON artifact."""

    if seen is None:
        seen = set()
    if isinstance(value, dict):
        if value.get("schema") == _PAYLOAD_SCHEMA:
            payload, _ = _payload(value, vm_artifacts, "artifact payload")
            del payload
            digest = value.get("sha256")
            if isinstance(digest, str):
                seen.add(digest)
            return 1
        return sum(_audit_payloads(child, vm_artifacts, seen) for child in value.values())
    if isinstance(value, list):
        return sum(_audit_payloads(child, vm_artifacts, seen) for child in value)
    return 0


def _hydrate_tree(value: Any, vm_artifacts):
    if isinstance(value, dict):
        if value.get("schema") == _PAYLOAD_SCHEMA:
            return _payload(value, vm_artifacts, "forecast payload")[0]
        return {key: _hydrate_tree(child, vm_artifacts) for key, child in value.items()}
    if isinstance(value, list):
        return [_hydrate_tree(child, vm_artifacts) for child in value]
    return value


def _artifact(reference: Any, artifacts, label: str) -> dict[str, Any]:
    reference = _ref(reference, label)
    if not reference.startswith("cas://sha256/"):
        raise ValueError(f"{label} is not a CAS artifact")
    artifact = _dict(artifacts.get(reference), label)
    if content_digest(artifact) != reference.removeprefix("cas://sha256/"):
        raise ValueError(f"{label} digest binding changed")
    return artifact


def _tensor_equal(left, right) -> bool:
    import torch

    return (
        tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and bool(torch.equal(left, right))
    )


def _tensor_close(left, right, *, label: str, rtol: float = 2e-5, atol: float = 2e-6) -> None:
    import torch

    if tuple(left.shape) != tuple(right.shape) or not bool(torch.allclose(left.float(), right.float(), rtol=rtol, atol=atol)):
        raise ValueError(f"{label} tensor changed")


def _error(actual, expected) -> dict[str, Any]:
    delta = actual.float() - expected.float()
    norm = float(expected.float().norm())
    return {
        "l2": float(delta.norm()),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm()) / norm if norm else None,
        "exact": _tensor_digest(actual) == _tensor_digest(expected),
    }


def _compare(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} fields changed")
        for key in expected:
            _compare(actual[key], expected[key], label=f"{label}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length changed")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare(left, right, label=f"{label}[{index}]")
    elif isinstance(actual, float) or isinstance(expected, float):
        _close_number(actual, expected, label=label, relative_tolerance=1e-4)
    elif actual != expected:
        raise ValueError(f"{label} changed")


def _load_cut(reference: str, artifacts, manifest, *, contract: str | None = None):
    return _cut(reference, artifacts, manifest, numerical_contract=contract)


def _span(value: Any, label: str, limit: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} span is malformed")
    start, stop = (_int(value[0], f"{label} start"), _int(value[1], f"{label} stop"))
    if start >= stop or stop > limit:
        raise ValueError(f"{label} span is out of bounds")
    return start, stop


def _check_sites(sites: Mapping[str, Any], program: FrozenReadProgram, parent: Mapping[str, Any], answer_span: tuple[int, int]) -> dict[str, Any]:
    if not isinstance(sites, Mapping) or not {"A", "B"} <= set(sites):
        raise ValueError("resolved A/B sites are incomplete")
    tokens = parent.get("token_ids")
    if not isinstance(tokens, list):
        raise ValueError("parent token history is missing")
    result = {}
    spans = {}
    for name in ("A", "B"):
        site = _dict(sites[name], f"site {name}")
        port = program.port(name)
        for key, expected in (("port", name), ("layer", port.layer), ("kv_head", port.kv_head),
                              ("role", port.row_role), ("provenance", port.provenance)):
            if site.get(key) != expected:
                raise ValueError(f"site {name} {key} binding changed")
        row = _int(site.get("row", site.get("source_row")), f"site {name} row")
        if site.get("source_row", row) != row or row >= len(tokens) or site.get("source_token_id") != tokens[row]:
            raise ValueError(f"site {name} source binding changed")
        lineage = _dict(site.get("lineage"), f"site {name} lineage")
        if lineage.get("provenance") != port.provenance or lineage.get("role") != port.row_role:
            raise ValueError(f"site {name} lineage role changed")
        span = _span(lineage.get("span"), f"site {name} lineage", len(tokens))
        if name == "A":
            if span != answer_span or row != span[0] or lineage.get("source") != "preceding_answer_span" or lineage.get("occurrence") != "first-generated":
                raise ValueError("A source lineage changed")
        else:
            if lineage.get("source") != "followup_span" or lineage.get("occurrence") != "last-copula-in-span":
                raise ValueError("B source lineage changed")
            copula = lineage.get("copula_token_id")
            if type(copula) is not int or tokens[row] != copula or not (span[0] <= row < span[1]):
                raise ValueError("B copula binding changed")
            if row != max(index for index in range(span[0], span[1]) if tokens[index] == copula):
                raise ValueError("B is not the last in-span copula")
        result[name] = site
        spans[name] = span
    if spans["A"][1] > spans["B"][0]:
        raise ValueError("A/B spans overlap or are reversed")
    if "wrong" in sites:
        wrong = _dict(sites["wrong"], "wrong-row control")
        if wrong.get("role") != "wrong-row-control":
            raise ValueError("wrong-row declaration changed")
    return result


def _ancestral(cut_ref: str, target_ref: str, artifacts, manifest, contract: str) -> bool:
    seen: set[str] = set()
    current = cut_ref
    while current and current not in seen:
        if current == target_ref:
            return True
        seen.add(current)
        cut = _load_cut(current, artifacts, manifest, contract=contract)
        current = cut.get("parent_cut")
    return False


def _verify_arm(arm: Mapping[str, Any], *, label: str, parent_ref: str, parent: Mapping[str, Any], artifacts, vm_artifacts, manifest, contract: str) -> dict[str, Any]:
    arm = _dict(arm, f"arm {label}")
    if arm.get("name") != label or arm.get("parent_cut") != parent_ref:
        raise ValueError(f"arm {label} parent/name changed")
    trajectory = _list(arm.get("trajectory"), f"arm {label} trajectory")
    expected_count = 1 if label in {"A-0.25", "A-0.5"} else 3
    if len(trajectory) != expected_count:
        raise ValueError(f"arm {label} native count changed")
    previous_ref, previous = parent_ref, parent
    for index, value in enumerate(trajectory):
        row = _dict(value, f"arm {label} step")
        if row.get("offset") != index:
            raise ValueError(f"arm {label} offset changed")
        token = _int(row.get("consumed_token_id"), f"arm {label} consumed token")
        expected_authority = "caller-supplied" if index == 0 else "native-greedy"
        if row.get("authority") != expected_authority:
            raise ValueError(f"arm {label} token authority changed")
        cut_ref = _ref(row.get("cut"), f"arm {label} cut")
        cut = _load_cut(cut_ref, artifacts, manifest, contract=contract)
        committed_parent = cut.get("parent_cut")
        if index == 0:
            staged_ref = _ref(committed_parent, f"arm {label} staged cut")
            staged = _load_cut(staged_ref, artifacts, manifest, contract=contract)
            if staged.get("pending") != token or staged.get("token_ids") != previous.get("token_ids") or staged.get("label") != "supplied-input-embedding" or not _ancestral(staged_ref, previous_ref, artifacts, manifest, contract):
                raise ValueError(f"arm {label} supplied-input staging changed")
        elif committed_parent != previous_ref:
            raise ValueError(f"arm {label} step parent changed")
        expected_tokens = list(previous.get("token_ids", ())) + [token]
        if cut.get("token_ids") != expected_tokens:
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


def _verify_acts(arms: Mapping[str, Any], sites: Mapping[str, Any], parent_ref: str, artifacts, vm_artifacts) -> None:
    for label, value in arms.items():
        arm = _dict(value, f"arm {label}")
        previous = parent_ref
        for item_value in _list(arm.get("acts", []), f"arm {label} Acts"):
            item = _dict(item_value, f"arm {label} Act item")
            port = item.get("port")
            if port not in {"A", "B"}:
                raise ValueError(f"arm {label} unknown Act port")
            act = _dict(item.get("act"), f"arm {label} Act")
            verify_write_act(act, artifacts, vm_artifacts)
            if act.get("parent_cut") != previous or act.get("donor_cut") != previous:
                raise ValueError(f"arm {label} Act ancestry changed")
            site = sites[port]
            selector = {"layers": [site["layer"]], "components": ["value"],
                        "row_start": site["source_row"], "row_stop": site["source_row"] + 1}
            if act.get("selector") != selector or act.get("mode") != "replace":
                raise ValueError(f"arm {label} {port} Act selector changed")
            previous = _ref(act.get("cut_id"), f"arm {label} Act cut")


def _capture(arm: Mapping[str, Any], name: str, vm_artifacts, *, required: bool = True):
    captures = _dict(arm.get("capture", {}), f"arm {arm.get('name')} capture")
    reference = captures.get(name)
    if reference is None and not required:
        return None
    return _payload(reference, vm_artifacts, f"arm {arm.get('name')} {name}")


def _signature(arm: Mapping[str, Any]):
    return [(row["consumed_token_id"], row["logits_sha256"]) for row in arm["trajectory"]]


def _verify_controls(arms: Mapping[str, Any]) -> None:
    zero = arms["00"]
    for name in ("sham", "reinstall"):
        if _signature(arms[name]) != _signature(zero) or arms[name].get("closure_sha256") != zero.get("closure_sha256"):
            raise ValueError(f"{name} native closure changed")


def _required_capture(arm: Mapping[str, Any], vm_artifacts) -> None:
    # These fields establish the full unedited-00 visibility partition and the
    # receiver input/key history used by the score program.
    required = ("input19", "attention19", "q19", "k19", "v19", "weights19",
                "input20", "q20", "k20", "v20", "weights20",
                "input21", "q21", "k21", "v21", "weights21",
                "input22", "q22", "k22", "v22", "scores22", "weights22",
                "attention22", "current_v22", "cos22", "sin22")
    for name in required:
        _capture(arm, name, vm_artifacts)


def _verify_weight(bundle: Mapping[str, Any], arms: Mapping[str, Any], sites: Mapping[str, Any], artifacts, vm_artifacts, manifest):
    ref = _ref(bundle.get("weight_forecast_ref"), "weight forecast")
    record = _artifact(ref, artifacts, "weight forecast")
    if record.get("schema") != WEIGHT_SCHEMA or record != bundle.get("weight_forecast"):
        raise ValueError("weight forecast artifact/report binding changed")
    if record.get("parent_cut") != bundle.get("parent_cut") or record.get("sites") != bundle.get("sites"):
        raise ValueError("weight forecast parent/site binding changed")
    if record.get("native_module_entries") != 0:
        raise ValueError("native module entered during forecast")
    visibility = str(record.get("visibility", ""))
    if "native00 tape" not in visibility or "no edited receiver" not in visibility:
        raise ValueError("weight visibility partition changed")
    if record.get("baseline_capture") != arms["00"].get("capture"):
        raise ValueError("weight baseline capture binding changed")
    payload_count = _audit_payloads(record, vm_artifacts)
    baseline = _dict(record.get("baseline_capture"), "baseline capture")
    q0 = _capture(arms["00"], "q22", vm_artifacts)[0]
    k0 = _capture(arms["00"], "k22", vm_artifacts)[0]
    scores0 = _capture(arms["00"], "scores22", vm_artifacts)[0]
    compiled = _hydrate_tree(_dict(record.get("compiled"), "compiled tangent"), vm_artifacts)
    if compiled.get("logical_linears") != 19 or compiled.get("logical_cost", {}).get("native_forward_calls") != 0:
        raise ValueError("compiled logical/native cost declaration changed")
    q_dot, k_dot = compiled.get("q_dot"), compiled.get("k_dot")
    if not hasattr(q_dot, "shape") or not hasattr(k_dot, "shape"):
        raise ValueError("compiled q/k tangent payloads are missing")
    score_saved = _hydrate_tree(_dict(record.get("score_program"), "score program"), vm_artifacts)
    scaling = score_saved.get("scaling")
    recomputed = compile_score_program(q0, k0, scores0, q_dot, k_dot,
                                       kv_head=sites["B"]["kv_head"], source_row=sites["B"]["source_row"], scaling=scaling)
    for key in ("schema", "kv_head", "source_row", "current_row", "scaling", "query_heads"):
        saved_value = score_saved.get(key)
        recomputed_value = recomputed.get(key)
        if key == "query_heads":
            saved_value, recomputed_value = list(saved_value), list(recomputed_value)
        if saved_value != recomputed_value:
            raise ValueError(f"score program {key} changed")
    for key in ("logp0", "a", "a_query", "b", "b_current"):
        _tensor_close(score_saved[key], recomputed[key], label=f"score program {key}")
    forecasts = _dict(record.get("forecasts"), "score forecasts")
    if set(forecasts) != {str(dose) for dose in DOSES}:
        raise ValueError("score forecast dose set changed")
    forecast_summary = {}
    for dose in DOSES:
        supplied_methods = _dict(_hydrate_tree(forecasts[str(dose)], vm_artifacts), f"forecast dose {dose}")
        if set(supplied_methods) != set(METHODS):
            raise ValueError(f"forecast methods changed at dose {dose}")
        forecast_summary[str(dose)] = {}
        for method in METHODS:
            expected = evaluate_score_program(recomputed, dose, method=method)
            actual = supplied_methods[method]
            for key in ("method", "dose"):
                if actual.get(key) != expected.get(key):
                    raise ValueError(f"forecast {dose}/{method} {key} changed")
            for key in ("pred_logpB", "logpB", "delta_logpB", "pB"):
                _tensor_close(actual[key], expected[key], label=f"forecast {dose}/{method}/{key}")
            forecast_summary[str(dose)][method] = {"pB": actual["pB"].detach().cpu().tolist()}
    source = _hydrate_tree(_dict(record.get("source_attention"), "source attention"), vm_artifacts)
    write = _payload(record.get("write_ref"), vm_artifacts, "source write")[0]
    if "0.0" not in source or "1.0" not in source:
        raise ValueError("source dose payloads are incomplete")
    _tensor_close(source["0.0"], _capture(arms["00"], "attention19", vm_artifacts)[0], label="baseline source attention", rtol=2e-4, atol=2e-5)
    _tensor_close(source["1.0"].float() - source["0.0"].float(), write.float(), label="source write relation", rtol=2e-5, atol=2e-6)
    return {"artifact_ref": ref, "payload_count": payload_count, "score_program": recomputed, "forecasts": forecast_summary, "baseline_capture_fields": len(baseline)}


def _weight_row(prediction: Mapping[str, Any], true_logp, baseline_logp) -> dict[str, Any]:
    delta = true_logp - baseline_logp
    residual = prediction["delta_logpB"].float() - delta
    return {"log_change_rmse": float(residual.square().mean().sqrt()),
            "log_change_max_abs": float(residual.abs().max()),
            "true_log_change_rms": float(delta.square().mean().sqrt()),
            "probability_error": _error(prediction["pB"].reshape(-1), true_logp.exp().reshape(-1)),
            "predicted_probability_in_range": bool(((prediction["pB"] >= 0) & (prediction["pB"] <= 1)).all()),
            "head_rows": [{"head_offset": index, "true_delta_logp": float(actual),
                           "predicted_delta_logp": float(estimated), "error": float(estimated - actual)}
                          for index, (actual, estimated) in enumerate(zip(delta.flatten(), prediction["delta_logpB"].flatten(), strict=True))]}


def _verify_doses(bundle: Mapping[str, Any], arms: Mapping[str, Any], sites: Mapping[str, Any], vm_artifacts, weight):
    import torch

    q0 = _capture(arms["00"], "q22", vm_artifacts)[0]
    scores0 = _capture(arms["00"], "scores22", vm_artifacts)[0]
    groups = q0.shape[1] // _capture(arms["00"], "k22", vm_artifacts)[0].shape[1]
    first = sites["B"]["kv_head"] * groups
    selected = slice(first, first + groups)
    row = sites["B"]["source_row"]
    baseline_logp = torch.log_softmax(scores0.float(), -1)[0, selected, 0, row]
    forecasts = weight["score_program"]
    supplied_rows = _list(bundle.get("dose_rows"), "dose rows")
    if [float(item.get("dose")) for item in supplied_rows] != list(DOSES):
        raise ValueError("dose row set changed")
    summary = []
    for item, dose in zip(supplied_rows, DOSES, strict=True):
        native_arm = arms["00"] if dose == 0 else arms[f"A-{dose}"]
        native = _capture(native_arm, "scores22", vm_artifacts)[0]
        true_logp = torch.log_softmax(native.float(), -1)[0, selected, 0, row]
        expected_methods = {method: _weight_row(evaluate_score_program(forecasts, dose, method=method), true_logp, baseline_logp) for method in METHODS}
        supplied_methods = _dict(item.get("methods"), f"dose {dose} methods")
        if set(supplied_methods) != set(METHODS):
            raise ValueError(f"dose {dose} methods changed")
        for method in METHODS:
            _compare(supplied_methods[method], expected_methods[method], label=f"dose {dose}/{method}")
        source0 = _payload(bundle["weight_forecast"]["source_attention"]["0.0"], vm_artifacts, "dose baseline source")[0]
        source_dose = _payload(bundle["weight_forecast"]["source_attention"][str(dose)], vm_artifacts, f"dose {dose} source")[0]
        actual_attention = _capture(native_arm, "attention19", vm_artifacts)[0]
        expected_earlier = _error(source_dose, actual_attention)
        expected_scaled = _error(source0.float() + dose * _payload(bundle["weight_forecast"]["write_ref"], vm_artifacts, "dose source write")[0].float(), actual_attention)
        _compare(item.get("earlier_write_error"), expected_earlier, label=f"dose {dose} earlier write")
        _compare(item.get("scaled_source_write_error"), expected_scaled, label=f"dose {dose} scaled write")
        summary.append({"dose": dose, "methods": expected_methods})
    return summary


def _verify_consumers(bundle: Mapping[str, Any], arms: Mapping[str, Any], artifacts, vm_artifacts):
    ref = _ref(bundle.get("consumer_forecast_ref"), "consumer forecast")
    record = _artifact(ref, artifacts, "consumer forecast")
    if record.get("schema") != CONSUMER_SCHEMA or record != bundle.get("consumer_forecast"):
        raise ValueError("consumer forecast artifact/report binding changed")
    if record.get("weight_forecast_ref") != bundle.get("weight_forecast_ref"):
        raise ValueError("consumer/weight seal binding changed")
    _audit_payloads(record, vm_artifacts)
    methods = _dict(record.get("methods"), "consumer methods")
    if set(methods) != set(CONSUMERS):
        raise ValueError("consumer method set changed")
    summaries = {}
    baseline_ref = arms["A-1.0"]["capture"]["attention22"]
    for method in CONSUMERS:
        item = _dict(methods[method], f"consumer {method}")
        if item.get("native_A_output_assisted") is not True or item.get("baseline_attention_ref") != baseline_ref or item.get("explicit_dense_mlp_linears") != 3:
            raise ValueError(f"consumer {method} native-assistance declaration changed")
        predicted = {name: _payload(item.get(f"{name}_ref", item.get(name)), vm_artifacts, f"consumer {method} {name}")[0] for name in ("attention", "consumed_attention", "mlp", "exit")}
        injected = {name: _capture(arms[method], key, vm_artifacts)[0] for name, key in (("attention", "attention22"), ("mlp", "mlp22"), ("exit", "exit22"))}
        if not _tensor_equal(predicted["consumed_attention"], injected["attention"]) or not _tensor_equal(predicted["mlp"], injected["mlp"]) or not _tensor_equal(predicted["exit"], injected["exit"]):
            raise ValueError(f"consumer {method} injected payload changed")
        summaries[method] = {"attention_cast_exact": True, "mlp_exact": True, "exit_exact": True}
    return summaries


def _verify_local(bundle: Mapping[str, Any], arms: Mapping[str, Any], vm_artifacts):
    supplied = _dict(bundle.get("local_errors"), "local errors")
    result = {}
    for method in CONSUMERS:
        record = _dict(supplied.get(method), f"local errors {method}")
        forecast = _dict(bundle["consumer_forecast"]["methods"][method], f"consumer {method}")
        native = {name: _capture(arms["AB"], key, vm_artifacts)[0] for name, key in (("attention", "attention22"), ("mlp", "mlp22"), ("exit", "exit22"))}
        predicted = {name: _payload(forecast.get(f"{name}_ref", forecast.get(name)), vm_artifacts, f"consumer {method} {name}")[0] for name in native}
        result[method] = {}
        for name in native:
            expected = _error(predicted[name], native[name])
            _compare(record.get(name), expected, label=f"local {method}/{name}")
            result[method][name] = expected
        for key, left, right in (("injected_mlp_exact", "mlp", "mlp22"), ("injected_exit_exact", "exit", "exit22")):
            expected = _tensor_equal(predicted[left], _capture(arms[method], right, vm_artifacts)[0])
            if record.get(key) is not expected:
                raise ValueError(f"local {method}/{key} changed")
            result[method][key] = expected
    return result


def _future(candidate: Mapping[str, Any], reference: Mapping[str, Any], artifacts, vm_artifacts):
    left_rows, right_rows = _list(candidate["trajectory"], "candidate trajectory"), _list(reference["trajectory"], "reference trajectory")
    if len(left_rows) != len(right_rows):
        raise ValueError("future trajectory lengths differ")
    rows = []
    for left, right in zip(left_rows, right_rows, strict=True):
        left_cut = _dict(artifacts.get(_ref(left["cut"], "candidate cut")), "candidate cut")
        right_cut = _dict(artifacts.get(_ref(right["cut"], "reference cut")), "reference cut")
        left_logits = _payload(left_cut["logits"], vm_artifacts, "candidate logits")[0]
        right_logits = _payload(right_cut["logits"], vm_artifacts, "reference logits")[0]
        rows.append({"offset": left["offset"], "same_consumed": left["consumed_token_id"] == right["consumed_token_id"],
                     "same_next": left["next_token_id"] == right["next_token_id"],
                     "target_margin_change": left["target_margin"] - right["target_margin"], **_error(left_logits, right_logits)})
    return rows


def _verify_future(bundle: Mapping[str, Any], arms: Mapping[str, Any], artifacts, vm_artifacts):
    supplied = _dict(bundle.get("future_comparisons"), "future comparisons")
    result = {}
    for method in CONSUMERS:
        expected = _future(arms[method], arms["AB"], artifacts, vm_artifacts)
        _compare(supplied.get(method), expected, label=f"future {method}")
        result[method] = expected
    return result


def _verify_events(case: Mapping[str, Any], bundles: list[Mapping[str, Any]]) -> dict[str, Any]:
    events = _list(case.get("prediction_events"), "prediction events")
    positions = []
    for bundle in bundles:
        weight_ref = _ref(bundle["weight_forecast_ref"], "weight event ref")
        consumer_ref = _ref(bundle["consumer_forecast_ref"], "consumer event ref")
        weight = [(i, e) for i, e in enumerate(events) if isinstance(e, dict) and e.get("kind") == "weights-sealed" and e.get("ref") == weight_ref]
        consumer = [(i, e) for i, e in enumerate(events) if isinstance(e, dict) and e.get("kind") == "consumers-sealed" and e.get("ref") == consumer_ref]
        starts = [(i, e) for i, e in enumerate(events) if isinstance(e, dict) and e.get("kind") == "native-AB-started" and e.get("consumer_ref") == consumer_ref]
        edited = [(i, e) for i, e in enumerate(events) if isinstance(e, dict) and e.get("kind") == "edited-receiver-started" and e.get("weight_ref") == weight_ref]
        if len(weight) != 1 or len(consumer) != 1 or len(starts) != 1 or {e.get("dose") for _, e in edited} != {0.25, 0.5, 1.0}:
            raise ValueError("weight/consumer seal event set is incomplete")
        wi, we = weight[0]
        ci, ce = consumer[0]
        ai, ae = starts[0]
        if not (wi < ci < ai and all(wi < i for i, _ in edited)):
            raise ValueError("receiver work was recorded before its weight seal")
        physical = [e.get("physical_steps") for e in events]
        if any(type(step) is not int or step < 0 for step in physical) or physical != sorted(physical):
            raise ValueError("prediction event physical-step order changed")
        positions.append({"weight_seal_index": wi, "consumer_seal_index": ci, "native_AB_index": ai,
                          "weight_seal_steps": we["physical_steps"], "consumer_seal_steps": ce["physical_steps"],
                          "native_AB_steps": ae["physical_steps"]})
    return {"status": "valid", "pairs": positions}


def _verify_bundle(bundle: Mapping[str, Any], case_root: Path, artifacts, vm_artifacts, manifest, program: FrozenReadProgram, *, answer_span: tuple[int, int]):
    parent_ref = _ref(bundle.get("parent_cut"), "bundle parent")
    parent = _load_cut(parent_ref, artifacts, manifest)
    contract = parent["numerical_contract"]
    if not _sha(contract):
        raise ValueError("cut numerical ABI fingerprint is malformed")
    sites = _check_sites(bundle.get("sites"), program, parent, answer_span)
    arms = _dict(bundle.get("arms"), "bundle arms")
    required = {"00", "sham", "reinstall", "A-0.25", "A-0.5", "A-1.0", "transport", "linear-logweight", "stale", "oracle", "AB"}
    if not required <= set(arms):
        raise ValueError("bundle arm set is incomplete")
    for label, arm in arms.items():
        _verify_arm(arm, label=label, parent_ref=parent_ref, parent=parent, artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest, contract=contract)
    _verify_acts(arms, sites, parent_ref, artifacts, vm_artifacts)
    _verify_controls(arms)
    _required_capture(arms["00"], vm_artifacts)
    weight = _verify_weight(bundle, arms, sites, artifacts, vm_artifacts, manifest)
    dose_summary = _verify_doses(bundle, arms, sites, vm_artifacts, weight)
    consumer_summary = _verify_consumers(bundle, arms, artifacts, vm_artifacts)
    local_summary = _verify_local(bundle, arms, vm_artifacts)
    future_summary = _verify_future(bundle, arms, artifacts, vm_artifacts)
    a_q = _capture(arms["A-1.0"], "q22", vm_artifacts)[0]
    a_k = _capture(arms["A-1.0"], "k22", vm_artifacts)[0]
    ab_q = _capture(arms["AB"], "q22", vm_artifacts)[0]
    ab_k = _capture(arms["AB"], "k22", vm_artifacts)[0]
    a_input = _capture(arms["A-1.0"], "input22", vm_artifacts)[0]
    ab_input = _capture(arms["AB"], "input22", vm_artifacts)[0]
    a_v = _capture(arms["A-1.0"], "current_v22", vm_artifacts)[0]
    ab_v = _capture(arms["AB"], "current_v22", vm_artifacts)[0]
    exact_route = _tensor_equal(a_input, ab_input) and _tensor_equal(a_q, ab_q) and _tensor_equal(a_k, ab_k) and _tensor_equal(a_v, ab_v)
    if ("A_vs_AB_input22_exact" in bundle and bundle.get("A_vs_AB_input22_exact") is not _tensor_equal(a_input, ab_input)) or ("A_vs_AB_query_keys_current_value_exact" in bundle and bundle.get("A_vs_AB_query_keys_current_value_exact") is not exact_route):
        raise ValueError("A/AB receiver locality declaration changed")
    if bundle.get("old_keys22_unchanged") is not _tensor_equal(a_k[:, :, :-1], _capture(arms["00"], "k22", vm_artifacts)[0][:, :, :-1]):
        raise ValueError("old-key locality declaration changed")
    if bundle.get("B_source_value_unchanged") is not _tensor_equal(_capture(arms["A-1.0"], "v22", vm_artifacts)[0][0, sites["B"]["kv_head"], sites["B"]["source_row"]], _capture(arms["00"], "v22", vm_artifacts)[0][0, sites["B"]["kv_head"], sites["B"]["source_row"]]):
        raise ValueError("B source-value locality declaration changed")
    for key, expected in (("q_transport_error", _error(_capture(arms["00"], "q22", vm_artifacts)[0].float() + _hydrate_tree(bundle["weight_forecast"]["compiled"], vm_artifacts)["q_dot"], a_q)),
                          ("k_transport_error", _error(_capture(arms["00"], "k22", vm_artifacts)[0][:, :, -1:].float() + _hydrate_tree(bundle["weight_forecast"]["compiled"], vm_artifacts)["k_dot"], a_k[:, :, -1:]))):
        _compare(bundle.get(key), expected, label=key)
    dense_refs = bundle.get("dense_fp32_control") or bundle.get("dense_fp32_control_qk")
    dense = _hydrate_tree(dense_refs, vm_artifacts)
    if not isinstance(dense, list) or len(dense) != 2 or bundle.get("dense_control_native_module_entries") != 0:
        raise ValueError("post-seal dense control declaration changed")
    dense_expected = {"q": _error(dense[0], a_q), "k": _error(dense[1], a_k[:, :, -1:])}
    _compare(bundle.get("dense_fp32_control_errors"), dense_expected, label="dense FP32 control")
    weight_summary = {key: value for key, value in weight.items() if key != "score_program"}
    return {"parent_cut": parent_ref, "sites": sites, "weight": weight_summary, "doses": dose_summary,
            "consumers": consumer_summary, "local_errors": local_summary, "future": future_summary,
            "receiver_value_only": exact_route, "native_module_entries_during_forecast": bundle.get("native_module_entries_during_forecast")}


def _case_root(results_root: Path, case_id: str) -> Path:
    direct = results_root / case_id
    if (direct / "selected.json").is_file():
        return direct
    matches = [path.parent for path in results_root.rglob("selected.json") if path.parent.name == case_id]
    if len(matches) != 1:
        raise ValueError(f"cannot locate unique session tree for {case_id!r}")
    return matches[0]


def _verify_case(case: Mapping[str, Any], case_root: Path, program: FrozenReadProgram) -> dict[str, Any]:
    bench_summary = _bench_read_only(case_root)
    artifacts, vm_artifacts, manifest_ref, manifest = _load_case(case_root)
    if case.get("model_identity") != manifest.get("identity") or manifest.get("attention_backend") != "native-eager":
        raise ValueError("case model/backend identity changed")
    declared_contract = case.get("numerical_contract")
    if declared_contract is not None and declared_contract != manifest.get("capabilities", {}).get("numerical_contract"):
        raise ValueError("case numerical profile changed")
    FrozenReadProgram.from_dict(case.get("program"))
    first = _dict(case.get("first"), "first bundle")
    first_parent = _load_cut(_ref(first.get("parent_cut"), "first parent"), artifacts, manifest)
    answer_span = _span(case.get("initial_answer_span"), "initial answer", len(first_parent["token_ids"]))
    first_summary = _verify_bundle(first, case_root, artifacts, vm_artifacts, manifest, program, answer_span=answer_span)
    second = case.get("second_candidate_world")
    second_summary = None
    if second is not None:
        second = _dict(second, "second bundle")
        source_cut = _ref(case.get("second_world_source_cut"), "second world source cut")
        if source_cut != first["arms"]["transport"]["end_cut"] or not _ancestral(_ref(second["parent_cut"], "second parent"), source_cut, artifacts, manifest, first_parent["numerical_contract"]):
            raise ValueError("second bundle is not descended from transport world")
        second_parent = _load_cut(_ref(second["parent_cut"], "second parent"), artifacts, manifest)
        second_span = _span(case.get("second_answer_span"), "second answer", len(second_parent["token_ids"]))
        second_summary = _verify_bundle(second, case_root, artifacts, vm_artifacts, manifest, program, answer_span=second_span)
    if case.get("semantic_cot_qualified") is not False:
        raise ValueError("case made an unsupported semantic-CoT claim")
    event_summary = _verify_events(case, [first] + ([] if second is None else [second]))
    return {"schema": VERIFY_SCHEMA, "case_id": _dict(case.get("case"), "case declaration").get("id"),
            "mechanics_status": "valid", "instrument_status": "valid", "terminal_status": "not-assessed",
            "bench": bench_summary, "manifest_ref": manifest_ref, "first": first_summary,
            "second": second_summary, "prediction_events": event_summary,
            "limitations": ["derivative compiler was not independently rerun", "semantic CoT and portable full-query claims are not assessed"]}


def verify_report(report: Mapping[str, Any], results_root: str | Path) -> dict[str, Any]:
    _finite(report)
    report = _dict(report, "report")
    if report.get("schema") != REPORT_SCHEMA or report.get("semantic_cot_qualified") is not False:
        raise ValueError("report schema or semantic-CoT qualification changed")
    program = FrozenReadProgram.standard()
    cases = _list(report.get("cases"), "report cases")
    if len(cases) != 4:
        raise ValueError("transport report must contain four fresh cases")
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
            "instrument_status": "valid", "terminal_status": "not-assessed",
            "semantic_cot_qualified": False, "derivative_compiler_independently_rerun": False,
            "cases": checked}


def verify(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    report_path = root / "report.json" if root.is_dir() else root
    report = json.loads(report_path.read_text())
    return verify_report(report, report_path.parent / "results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="result directory or report.json")
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
