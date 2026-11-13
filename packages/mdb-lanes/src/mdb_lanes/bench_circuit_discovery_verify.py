"""Offline custody verification for native Qwen circuit-discovery reports.

This verifier deliberately checks the frozen execution record rather than
re-running a model.  Route scores and intervention magnitudes are observations
whose exact provenance is checked; neither is promoted to a semantic-CoT or
effect-size gate.
"""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Any

from .model_os_contracts import content_digest

SCHEMA = "saturn-bench-circuit-discovery-v1"
_CUT_SCHEMA = "saturn-open-bench-cut-v1"
_RECEIPT_SCHEMA = "saturn-model-os-execution-receipt-v1"


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _finite(value: Any, path: str = "value") -> None:
    """Reject non-finite JSON numbers without importing a tensor framework."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"circuit discovery non-finite value at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def _close_number(actual, expected, *, label, relative_tolerance=1e-5):
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        raise ValueError(f"{label} is not numeric")
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError(f"{label} is non-finite")
    # Norms and max errors may be reduced on a different device during
    # offline verification.  Tensor digests and scalar readouts remain exact;
    # these derived summaries use a bounded numerical tolerance only.
    if not math.isclose(float(actual), float(expected), rel_tol=relative_tolerance, abs_tol=1e-6):
        raise ValueError(f"{label} changed")


def _close_structure(actual, expected, *, label):
    if isinstance(actual, float) or isinstance(expected, float):
        _close_number(actual, expected, label=label)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} fields changed")
        for key in expected:
            if key == "logit_l2":
                # CUDA and CPU FP32 norm reductions over the trained Qwen
                # 151,936-entry vocabulary differed by at most 4.307e-5
                # relative on the frozen 2026-11-06 assay. Use 100 ppm for
                # this derived summary only. Payload hashes, token/margin
                # readouts, max error, and all other fields stay unchanged.
                _close_number(actual[key], expected[key], label=f"{label}.{key}",
                              relative_tolerance=1e-4)
            else:
                _close_structure(actual[key], expected[key], label=f"{label}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length changed")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _close_structure(left, right, label=f"{label}[{index}]")
    elif actual != expected:
        raise ValueError(f"{label} changed")


def _get(store, reference: str, fallback=None):
    try:
        return store.get(reference)
    except Exception:
        if fallback is None:
            raise
        return fallback.get(reference)


def _cut(reference: str, artifacts, manifest, *, numerical_contract=None):
    if not isinstance(reference, str) or not reference:
        raise ValueError("circuit discovery cut reference is missing")
    cut = artifacts.get(reference)
    if not isinstance(cut, dict) or cut.get("schema") != _CUT_SCHEMA:
        raise ValueError("circuit discovery references a non-bench cut")
    if cut.get("identity") != manifest["identity"] or cut.get("lane") != manifest["lane"]:
        raise ValueError("circuit discovery cut model/lane identity differs from manifest")
    if numerical_contract is not None and cut.get("numerical_contract") != numerical_contract:
        raise ValueError("circuit discovery cut numerical contract differs from report")
    tokens = cut.get("token_ids")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
        or type(cut.get("pending")) is not int
        or cut["pending"] < 0
        or type(cut.get("prompt_length")) is not int
        or not 0 < cut["prompt_length"] <= len(tokens)
    ):
        raise ValueError("circuit discovery cut token accounting is malformed")
    return cut


def _plan(spec):
    plan = spec.get("plan")
    if not isinstance(plan, list) or not 1 <= len(plan) <= 16:
        raise ValueError("circuit discovery plan is malformed")
    flat = []
    for segment in plan:
        if not isinstance(segment, dict):
            raise ValueError("circuit discovery plan segment is malformed")
        if segment.get("kind") == "generate" and set(segment) == {"kind", "count"}:
            count = segment["count"]
            if type(count) is not int or not 1 <= count <= 32:
                raise ValueError("circuit discovery generation count is out of bounds")
            flat.extend({"kind": "generate"} for _ in range(count))
        elif segment.get("kind") == "input" and set(segment) == {"kind", "token_ids"}:
            tokens = segment["token_ids"]
            if (
                not isinstance(tokens, list)
                or not 1 <= len(tokens) <= 32
                or any(type(token) is not int or token < 0 for token in tokens)
            ):
                raise ValueError("circuit discovery input token IDs are out of bounds")
            flat.extend({"kind": "input", "token_id": token} for token in tokens)
        else:
            raise ValueError("circuit discovery plan segment has unknown fields")
    if not 1 <= len(flat) <= 32:
        raise ValueError("circuit discovery plan length is out of bounds")
    return flat


def _verify_spec(spec, supplied_digest):
    if not isinstance(spec, dict):
        raise ValueError("circuit discovery spec is not an object")
    try:
        digest = content_digest(spec)
    except Exception as error:
        raise ValueError("circuit discovery spec is not finite JSON") from error
    if digest != supplied_digest:
        raise ValueError("circuit discovery spec digest mismatch")
    name = spec.get("name", "read-circuit")
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ValueError("circuit discovery name is out of bounds")
    flat = _plan(spec)
    offsets = spec.get("receiver_offsets", [0])
    if (
        not isinstance(offsets, list)
        or not 1 <= len(offsets) <= 8
        or any(type(offset) is not int or not 0 <= offset < len(flat) for offset in offsets)
        or len(set(offsets)) != len(offsets)
    ):
        raise ValueError("circuit discovery receiver offsets are malformed")
    layers = spec.get("layers")
    if (
        not isinstance(layers, list)
        or not layers
        or len(layers) > 128
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("circuit discovery layers are malformed")
    candidates = spec.get("candidates_per_receiver", 1)
    if type(candidates) is not int or not 1 <= candidates <= 4:
        raise ValueError("circuit discovery candidate bound is malformed")
    scales = spec.get("scales", [0.0, 0.5])
    if (
        not isinstance(scales, list)
        or not 1 <= len(scales) <= 4
        or any(type(scale) not in {int, float} or not math.isfinite(float(scale)) or not 0 <= scale <= 2
               for scale in scales)
        or len(set(scales)) != len(scales)
    ):
        raise ValueError("circuit discovery scales are malformed")
    horizon = spec.get("continuation_tokens", len(flat))
    if type(horizon) is not int or not 1 <= horizon <= 32:
        raise ValueError("circuit discovery continuation bound is malformed")
    top_k = spec.get("top_k", 5)
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("circuit discovery top-k bound is malformed")
    for key, upper in (("max_native_operations", 100000), ("max_capture_bytes", 1 << 30)):
        if key in spec and (type(spec[key]) is not int or not 1 <= spec[key] <= upper):
            raise ValueError(f"circuit discovery {key} is malformed")
    observable = spec.get("observable")
    if observable is not None:
        if (
            not isinstance(observable, dict)
            or set(observable) != {"kind", "token_id", "against_token_id"}
            or observable.get("kind") != "token_margin"
            or any(type(observable.get(key)) is not int or observable[key] < 0
                   for key in ("token_id", "against_token_id"))
            or observable["token_id"] == observable["against_token_id"]
        ):
            raise ValueError("circuit discovery observable is malformed")
    return flat, sorted(offsets), sorted(layers), [float(scale) for scale in scales]


def _verify_source(instrument, artifacts):
    if not isinstance(instrument, dict):
        raise ValueError("circuit discovery instrument identity is missing")
    required = {"module", "source_sha256", "source_ref"}
    if required - set(instrument) or not _sha(instrument["source_sha256"]):
        raise ValueError("circuit discovery instrument identity is malformed")
    source = artifacts.get(instrument["source_ref"])
    if (
        not isinstance(source, dict)
        or source.get("schema") != "saturn-instrument-source-v1"
        or source.get("module") != instrument["module"]
        or source.get("source_sha256") != instrument["source_sha256"]
        or not isinstance(source.get("source_text"), str)
        or sha256(source["source_text"].encode("utf-8")).hexdigest() != instrument["source_sha256"]
    ):
        raise ValueError("circuit discovery instrument source custody mismatch")


def _payload_tensor(reference, vm_artifacts, *, label):
    """Hydrate one sealed tensor only when a report actually needs it."""

    if vm_artifacts is None:
        raise ValueError(f"{label} requires native VM payload closure")
    import torch

    from .vm_payload_cache import PayloadRef, PayloadStore

    try:
        payload = PayloadRef.from_dict(reference)
    except Exception as error:
        raise ValueError(f"{label} tensor reference is malformed") from error
    payloads = PayloadStore(vm_artifacts.root.parent / "radix/cas")
    raw = payloads.hydrate(payload, native_consumer="circuit-discovery-verification").payload
    if sha256(raw).hexdigest() != payload.sha256:
        raise ValueError(f"{label} tensor payload digest mismatch")
    try:
        dtype = getattr(torch, payload.dtype.removeprefix("torch."))
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(payload.shape)).clone()
    except Exception as error:
        raise ValueError(f"{label} tensor payload geometry is malformed") from error
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{label} tensor payload is non-finite")
    return tensor, payload


def _tensor_digest(tensor) -> str:
    import torch

    return sha256(
        tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _verify_receipt(reference, artifacts, vm_artifacts, *, row):
    if not isinstance(reference, str) or not reference:
        raise ValueError("circuit discovery native receipt reference is missing")
    receipt = _get(artifacts, reference, vm_artifacts)
    if not isinstance(receipt, dict):
        raise ValueError("circuit discovery native receipt is malformed")
    digest = receipt.get("content_sha256")
    if digest is not None:
        body = dict(receipt)
        body.pop("content_sha256", None)
        if digest != content_digest(body):
            raise ValueError("circuit discovery native receipt digest mismatch")
    if receipt.get("schema") != _RECEIPT_SCHEMA or receipt.get("decision") != "commit":
        raise ValueError("circuit discovery native step did not commit")
    continuation = receipt.get("continuation")
    if not isinstance(continuation, dict):
        raise ValueError("circuit discovery native receipt continuation is missing")
    if (
        continuation.get("next_token_id") != row["next_token_id"]
        or continuation.get("logits_sha256") != row["logits_sha256"]
    ):
        raise ValueError("circuit discovery native receipt/readout binding changed")


def _read_logits(cut_id, cut, artifacts, vm_artifacts, cache):
    if cut_id not in cache:
        tensor, payload = _payload_tensor(cut.get("logits"), vm_artifacts, label=f"cut {cut_id} logits")
        cache[cut_id] = (tensor, payload)
    return cache[cut_id]


def _verify_readout(row, cut_id, cut, spec, artifacts, vm_artifacts, cache):
    import torch

    tensor, payload = _read_logits(cut_id, cut, artifacts, vm_artifacts, cache)
    if tuple(tensor.shape) != tuple(payload.shape) or tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError("circuit discovery logits geometry is malformed")
    if row.get("logits_sha256") != payload.sha256:
        raise ValueError("circuit discovery readout/logits payload binding changed")
    expected_next = int(torch.argmax(tensor.float()[0]).item())
    if row.get("next_token_id") != expected_next or cut.get("pending") != expected_next:
        raise ValueError("circuit discovery next-token readout binding changed")
    observable = spec.get("observable")
    expected_margin = None
    if observable is not None:
        expected_margin = float(
            tensor.float()[0, observable["token_id"]]
            - tensor.float()[0, observable["against_token_id"]]
        )
    if row.get("margin") != expected_margin:
        raise ValueError("circuit discovery observable margin changed")


def _verify_closure(arm, endpoint, *, label):
    closure = arm.get("closure") if isinstance(arm, dict) else None
    if not isinstance(closure, dict) or not _sha(closure.get("sha256")):
        raise ValueError(f"{label} closure is missing")
    body = dict(closure)
    digest = body.pop("sha256")
    if content_digest(body) != digest:
        raise ValueError(f"{label} closure digest mismatch")
    if closure.get("token_ids") != endpoint.get("token_ids"):
        raise ValueError(f"{label} closure token history mismatch")
    logits = closure.get("logits")
    if not isinstance(logits, dict) or logits.get("sha256") != endpoint.get("logits", {}).get("sha256"):
        raise ValueError(f"{label} closure logits binding mismatch")
    memory = closure.get("memory")
    if not isinstance(memory, dict) or not _sha(memory.get("sha256")):
        raise ValueError(f"{label} memory closure is missing")
    memory_digest = memory["sha256"]
    if content_digest({"addresses": memory.get("addresses"), "cache_flags": memory.get("cache_flags")}) != memory_digest:
        raise ValueError(f"{label} memory closure digest mismatch")


def _verify_trajectory(arm, *, label, start, expected_parent, flat, spec, artifacts,
                       vm_artifacts, manifest, numerical_contract, cache):
    if not isinstance(arm, dict) or not isinstance(arm.get("trajectory"), list):
        raise ValueError(f"{label} trajectory is missing")
    trajectory = arm["trajectory"]
    if not trajectory:
        raise ValueError(f"{label} trajectory is empty")
    previous_ref = expected_parent
    previous = _cut(expected_parent, artifacts, manifest, numerical_contract=numerical_contract)
    for local, row in enumerate(trajectory):
        if not isinstance(row, dict):
            raise ValueError(f"{label} step is malformed")
        offset = start + local
        if row.get("offset") != offset or row.get("parent_cut") != previous_ref:
            raise ValueError(f"{label} step parent/offset binding changed")
        if offset >= len(flat):
            raise ValueError(f"{label} step exceeds declared plan")
        instruction = flat[offset]
        authority = "caller-supplied" if instruction["kind"] == "input" else "native-greedy"
        if row.get("input_authority") != authority:
            raise ValueError(f"{label} input authority changed")
        token = row.get("consumed_token_id")
        if type(token) is not int or token < 0:
            raise ValueError(f"{label} consumed token is malformed")
        if instruction["kind"] == "input" and token != instruction["token_id"]:
            raise ValueError(f"{label} caller-supplied token changed")
        cut_id = row.get("cut_id")
        cut = _cut(cut_id, artifacts, manifest, numerical_contract=numerical_contract)
        committed_parent = cut.get("parent_cut")
        if instruction["kind"] == "input":
            # Bench.stage_input publishes a provisional embedding cut before
            # step() commits the supplied token.  The trajectory intentionally
            # keeps its semantic parent at the prior committed boundary, so
            # verify that hidden intermediate cut rather than pretending the
            # final token cut links directly to the prior boundary.
            staged = _cut(committed_parent, artifacts, manifest,
                          numerical_contract=numerical_contract)
            if (
                staged.get("parent_cut") != previous_ref
                or staged.get("token_ids") != previous["token_ids"]
                or staged.get("pending") != token
                or staged.get("label") != "supplied-input-embedding"
            ):
                raise ValueError(f"{label} supplied-input staging parent changed")
        elif committed_parent != previous_ref:
            raise ValueError(f"{label} committed cut parent changed")
        expected_tokens = list(previous["token_ids"]) + [token]
        if cut.get("token_ids") != expected_tokens:
            raise ValueError(f"{label} committed token history changed")
        if row.get("history_sha256") != content_digest(expected_tokens):
            raise ValueError(f"{label} history digest changed")
        if row.get("consumed_token_id") != cut["token_ids"][-1]:
            raise ValueError(f"{label} consumed token/cut binding changed")
        _verify_receipt(row.get("receipt_ref"), artifacts, vm_artifacts, row=row)
        _verify_readout(row, cut_id, cut, spec, artifacts, vm_artifacts, cache)
        previous_ref, previous = cut_id, cut
    _verify_closure(arm, previous, label=label)
    return previous


def _expected_compare(actual, reference, *, artifacts, vm_artifacts, cache):
    if len(actual["trajectory"]) != len(reference["trajectory"]):
        raise ValueError("circuit discovery paired trajectories have different lengths")
    rows = []
    for left, right in zip(actual["trajectory"], reference["trajectory"], strict=True):
        left_tensor, _ = _read_logits(left["cut_id"], artifacts.get(left["cut_id"]), artifacts, vm_artifacts, cache)
        right_tensor, _ = _read_logits(right["cut_id"], artifacts.get(right["cut_id"]), artifacts, vm_artifacts, cache)
        difference = left_tensor.float() - right_tensor.float()
        rows.append({
            "offset": left["offset"],
            "same_consumed_token": left["consumed_token_id"] == right["consumed_token_id"],
            "same_history": left["history_sha256"] == right["history_sha256"],
            "same_logits": left["logits_sha256"] == right["logits_sha256"],
            "same_next_token": left["next_token_id"] == right["next_token_id"],
            "logit_l2": float(difference.norm()),
            "logit_max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
            "signed_margin_effect": left["margin"] - right["margin"]
            if left["margin"] is not None else None,
        })
    return {
        "alignment": "plan-offset; generated tokens remain branch-owned",
        "steps": rows,
        "closure_exact": actual["closure"]["sha256"] == reference["closure"]["sha256"],
        "same_consumed_tokens": [row["consumed_token_id"] for row in actual["trajectory"]]
        == [row["consumed_token_id"] for row in reference["trajectory"]],
    }


def _expected_graph(receivers, tests):
    nodes, edges = [], []
    payloads = {}
    for observation in receivers:
        receiver = f"receiver:{observation['offset']}"
        nodes.append({"id": receiver, "kind": "native-receiver",
                      "parent_cut": observation["parent_cut"], "offset": observation["offset"],
                      "token_id": observation["receiver_token_id"]})
        for candidate in observation["candidates"]:
            nodes.append({"id": candidate["id"], "kind": "versioned-memory-value", **candidate})
            edges.append({"source": candidate["id"], "target": receiver, "kind": "observed-native-read",
                          "read_score": candidate["read_score"], "causal_dependency_tested": False})
            payload = (candidate["layer"], candidate["kv_head"], candidate["source_row"],
                       candidate["value_sha256"])
            previous = payloads.get(payload)
            if previous is not None:
                edges.append({"source": previous, "target": candidate["id"],
                              "kind": "same-payload-observation", "value_sha256": candidate["value_sha256"],
                              "causal_dependency_tested": False})
            payloads[payload] = candidate["id"]
    for test in tests:
        target = f"future:{test['id']}"
        nodes.append({"id": target, "kind": "native-future",
                      "cut_id": test["candidate"]["trajectory"][-1]["cut_id"]})
        edges.append({"source": test["candidate_id"], "target": target,
                      "kind": "measured-intervention-response", "test_id": test["id"],
                      "scale": test["scale"], "effect": test["effect"],
                      "interpretation": "conditional response at this receiver; no uniqueness or semantic role claim"})
    for left, right in zip(receivers, receivers[1:]):
        edges.append({"source": f"receiver:{left['offset']}", "target": f"receiver:{right['offset']}",
                      "kind": "native-execution-order", "causal_dependency_tested": False})
    return {"nodes": nodes, "edges": edges, "semantic_cot_qualified": False,
            "composition_status": "occurrence-graph; edges do not certify a composed minimal circuit"}


def _verify_local_prediction(test, candidate, observation, artifacts, vm_artifacts):
    local = test.get("local_prediction")
    required = {"prediction_ref", "actual_read", "query_and_keys_exact", "error_l2",
                "error_max_abs", "relative_l2", "interpretation"}
    if not isinstance(local, dict) or required - set(local) or local["query_and_keys_exact"] is not True:
        raise ValueError("circuit discovery local prediction is incomplete")
    if local["interpretation"] != "native rounding and contraction error; diagnostic only":
        raise ValueError("circuit discovery local prediction interpretation changed")
    prediction = artifacts.get(local["prediction_ref"])
    if (
        not isinstance(prediction, dict)
        or prediction.get("schema") != "saturn-circuit-read-write-prediction-v1"
        or prediction.get("candidate_id") != candidate["id"]
        or prediction.get("scale") != test["scale"]
        or prediction.get("scope") != "first receiver attention output; FP32 decomposition with native rounding residual"
    ):
        raise ValueError("circuit discovery local prediction artifact binding changed")
    source_read = next(
        row for row in observation["coverage"] if row["layer"] == candidate["layer"]
    )
    if (
        prediction.get("baseline_output_ref") != source_read.get("attention_output_ref")
        or prediction.get("contribution_ref") != candidate.get("contribution_ref")
    ):
        raise ValueError("circuit discovery local prediction operand binding changed")
    actual_read = local["actual_read"]
    if (
        not isinstance(actual_read, dict)
        or actual_read.get("offset") != observation["offset"]
        or actual_read.get("parent_cut") != test["edited_parent_cut"]
        or actual_read.get("layer") != candidate["layer"]
        or actual_read.get("row_start") != candidate["source_row"]
        or actual_read.get("row_stop") != candidate["source_row"] + 1
        or actual_read.get("query_sha256") != source_read.get("query_sha256")
        or actual_read.get("key_history_sha256") != source_read.get("key_history_sha256")
        or actual_read.get("reconstruction_status") != "diagnostic; does not replace native attention"
    ):
        raise ValueError("circuit discovery local prediction route binding changed")
    if not _sha(actual_read.get("attention_output_sha256")):
        raise ValueError("circuit discovery local prediction output digest is malformed")

    baseline, _ = _payload_tensor(
        prediction.get("baseline_output_ref"), vm_artifacts, label="prediction baseline output"
    )
    contribution, _ = _payload_tensor(
        prediction.get("contribution_ref"), vm_artifacts, label="prediction contribution"
    )
    predicted, predicted_payload = _payload_tensor(
        prediction.get("predicted_output_ref"), vm_artifacts, label="prediction output"
    )
    actual, actual_payload = _payload_tensor(
        actual_read.get("attention_output_ref"), vm_artifacts, label="actual attention output"
    )
    if actual_payload.sha256 != actual_read["attention_output_sha256"]:
        raise ValueError("circuit discovery actual attention-output digest changed")
    import torch

    try:
        expected = baseline.float() + (float(test["scale"]) - 1.0) * contribution.float()
    except RuntimeError as error:
        raise ValueError("circuit discovery local prediction tensor geometry changed") from error
    if tuple(predicted.shape) != tuple(expected.shape) or tuple(actual.shape) != tuple(expected.shape):
        raise ValueError("circuit discovery local prediction tensor geometry changed")
    if not bool(torch.allclose(predicted.float(), expected.float(), rtol=1e-5, atol=1e-6)):
        raise ValueError("circuit discovery saved local prediction differs from operands")
    error = actual.float() - predicted.float()
    _close_number(local["error_l2"], float(error.norm()), label="circuit discovery local prediction error_l2")
    _close_number(local["error_max_abs"], float(error.abs().max()), label="circuit discovery local prediction error_max_abs")
    relative = float(error.norm() / actual.float().norm()) if float(actual.float().norm()) else None
    if relative is None:
        if local["relative_l2"] is not None:
            raise ValueError("circuit discovery local prediction relative error changed")
    else:
        _close_number(local["relative_l2"], relative, label="circuit discovery local prediction relative_l2")
    if predicted_payload.sha256 != prediction["predicted_output_ref"].get("sha256"):
        raise ValueError("circuit discovery prediction output digest changed")


def _verify_head_locality(test, candidate, observation, artifacts, vm_artifacts, cache):
    if test.get("head_locality_exact") is not True:
        raise ValueError("circuit discovery head-locality control is not exact")
    locality = test.get("head_locality")
    required = {"selected_head", "before_ref", "after_ref", "untouched_heads_before_sha256",
                "untouched_heads_after_sha256", "selected_before_sha256", "selected_after_sha256"}
    if not isinstance(locality, dict) or required - set(locality):
        raise ValueError("circuit discovery head-locality payload is incomplete")
    act = test["act"]
    geometry = act.get("geometry", {})
    selected_head = locality["selected_head"]
    kv_heads, head_dim = geometry.get("kv_heads"), geometry.get("head_dim")
    if (
        type(selected_head) is not int
        or type(kv_heads) is not int
        or type(head_dim) is not int
        or not 0 <= selected_head < kv_heads
        or candidate.get("kv_head") != selected_head
        or head_dim <= 0
    ):
        raise ValueError("circuit discovery selected head geometry is malformed")
    before, before_payload = _payload_tensor(locality["before_ref"], vm_artifacts, label="head-locality before")
    after, after_payload = _payload_tensor(locality["after_ref"], vm_artifacts, label="head-locality after")
    expected_shape = (1, kv_heads, 1, head_dim)
    if tuple(before.shape) != expected_shape or tuple(after.shape) != expected_shape:
        raise ValueError("circuit discovery head-locality row geometry changed")
    before_digest, after_digest = _tensor_digest(before), _tensor_digest(after)
    if (
        locality["selected_before_sha256"] != _tensor_digest(before[:, selected_head:selected_head + 1])
        or locality["selected_after_sha256"] != _tensor_digest(after[:, selected_head:selected_head + 1])
    ):
        raise ValueError("circuit discovery selected-head payload digest changed")
    mask = [head for head in range(kv_heads) if head != selected_head]
    if (
        locality["untouched_heads_before_sha256"] != _tensor_digest(before[:, mask])
        or locality["untouched_heads_after_sha256"] != _tensor_digest(after[:, mask])
        or not bool((before[:, mask] == after[:, mask]).all())
    ):
        raise ValueError("circuit discovery unselected head locality changed")
    if candidate["value_sha256"] != _tensor_digest(before[:, candidate["kv_head"]:candidate["kv_head"] + 1]):
        raise ValueError("circuit discovery source value provenance changed")
    if before_payload.sha256 != before_digest or after_payload.sha256 != after_digest:
        raise ValueError("circuit discovery head-locality tensor digest binding changed")
    from .bench_hidden_memory import logical_address

    address = logical_address(candidate["layer"], "value", candidate["source_row"])
    if act.get("selected_payload", {}).get(address) != after_digest:
        raise ValueError("circuit discovery Act selected payload differs from after row")
    scale = float(test["scale"])
    expected = before.clone()
    expected[:, selected_head:selected_head + 1] = (
        before[:, selected_head:selected_head + 1] * scale
    ).to(before.dtype)
    if not bool((expected == after).all()):
        raise ValueError("circuit discovery selected-head dose differs from saved payload")
    if not _sha(locality["untouched_heads_before_sha256"]) or not _sha(locality["untouched_heads_after_sha256"]):
        raise ValueError("circuit discovery untouched-head digest is malformed")


def _verify_report(value, artifacts, manifest, vm_artifacts):
    _finite(value)
    if value.get("model_identity") != manifest["identity"] or value.get("lane") != manifest["lane"]:
        raise ValueError("circuit discovery report model/lane identity differs from manifest")
    if manifest.get("session_id") is not None and value.get("session_id") != manifest["session_id"]:
        raise ValueError("circuit discovery report session identity differs from manifest")
    if value.get("semantic_cot_qualified") is not False:
        raise ValueError("circuit discovery cannot qualify semantic CoT")
    if value.get("mechanics_status") != "valid" or value.get("instrument_status") != "native-consumers-executed":
        raise ValueError("circuit discovery mechanics/instrument status is not valid")
    if value.get("selected_parent_restored_exact") is not True or value.get("sibling_selections_preserved") is not True:
        raise ValueError("circuit discovery selected-parent restoration is not closed")
    native_contract = value.get("numerical_contract")
    if not isinstance(native_contract, str) or not native_contract:
        raise ValueError("circuit discovery numerical contract is missing")
    capabilities = manifest.get("capabilities")
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("numerical_contract") != native_contract
        or manifest.get("attention_backend") != "native-eager"
    ):
        raise ValueError("circuit discovery native numerical profile differs from manifest")
    flat, offsets, layers, scales = _verify_spec(value.get("spec"), value.get("spec_sha256"))
    _verify_source(value.get("instrument_identity"), artifacts)
    parent_ref = value.get("parent_cut")
    parent = _cut(parent_ref, artifacts, manifest)
    cut_contract = parent.get("numerical_contract")
    if not _sha(cut_contract):
        raise ValueError("circuit discovery parent cut ABI fingerprint is malformed")
    cache = {}

    baseline = value.get("baseline")
    _verify_trajectory(baseline, label="baseline", start=0, expected_parent=parent_ref, flat=flat,
                       spec=value["spec"], artifacts=artifacts, vm_artifacts=vm_artifacts,
                       manifest=manifest, numerical_contract=cut_contract, cache=cache)
    if len(baseline["trajectory"]) != len(flat):
        raise ValueError("circuit discovery baseline does not cover the declared plan")
    control = value.get("observer_control")
    if not isinstance(control, dict) or "future" not in control or "effect" not in control:
        raise ValueError("circuit discovery observer control is incomplete")
    future = control["future"]
    _verify_trajectory(future, label="observer-control", start=0, expected_parent=parent_ref, flat=flat,
                       spec=value["spec"], artifacts=artifacts, vm_artifacts=vm_artifacts,
                       manifest=manifest, numerical_contract=cut_contract, cache=cache)
    if len(future["trajectory"]) != len(flat):
        raise ValueError("circuit discovery observer control does not cover the declared plan")
    expected_control = _expected_compare(future, baseline, artifacts=artifacts, vm_artifacts=vm_artifacts, cache=cache)
    _close_structure(control["effect"], expected_control, label="circuit discovery observer-control effect")
    if not expected_control["closure_exact"] or any(not row["same_logits"] for row in expected_control["steps"]):
        raise ValueError("circuit discovery passive observer parity is not exact")

    receivers = value.get("receivers")
    tests = value.get("tests")
    if not isinstance(receivers, list) or not isinstance(tests, list):
        raise ValueError("circuit discovery receivers/tests are incomplete")
    if [row.get("offset") for row in receivers] != offsets:
        raise ValueError("circuit discovery receiver offsets changed")
    by_candidate = {}
    observations = {}
    for observation in receivers:
        if not isinstance(observation, dict) or observation["offset"] in observations:
            raise ValueError("circuit discovery receiver is malformed or duplicated")
        offset = observation["offset"]
        if offset not in offsets or observation.get("parent_cut") != baseline["trajectory"][offset]["parent_cut"]:
            raise ValueError("circuit discovery receiver parent binding changed")
        if observation.get("receiver_token_id") != baseline["trajectory"][offset]["consumed_token_id"]:
            raise ValueError("circuit discovery receiver token binding changed")
        observations[offset] = observation
        coverage = observation.get("coverage")
        if not isinstance(coverage, list) or len(coverage) != len(layers):
            raise ValueError("circuit discovery receiver coverage is incomplete")
        seen_layers = set()
        receiver_parent = _cut(observation["parent_cut"], artifacts, manifest, numerical_contract=cut_contract)
        row_count = len(receiver_parent["token_ids"])
        expected_start = value["spec"].get("row_start", max(0, row_count - 256))
        expected_stop = value["spec"].get("row_stop", row_count)
        if (
            type(expected_start) is not int or type(expected_stop) is not int
            or not 0 <= expected_start < expected_stop <= row_count
            or expected_stop - expected_start > 256
        ):
            raise ValueError("circuit discovery receiver source-row window is out of bounds")
        for item in coverage:
            if not isinstance(item, dict) or item.get("layer") in seen_layers:
                raise ValueError("circuit discovery receiver coverage is malformed")
            seen_layers.add(item.get("layer"))
            if (
                item.get("offset") != offset
                or item.get("parent_cut") != observation["parent_cut"]
                or item.get("layer") not in layers
                or item.get("row_start") != expected_start
                or item.get("row_stop") != expected_stop
                or item.get("receiver_position") != row_count
                or item.get("softmax_context_rows") != row_count + 1
                or item.get("current_uncommitted_row_excluded_from_candidates") is not True
                or not _sha(item.get("query_sha256"))
                or not _sha(item.get("key_history_sha256"))
                or item.get("backend") != "native-eager"
                or item.get("reconstruction_status") != "diagnostic; does not replace native attention"
            ):
                raise ValueError("circuit discovery receiver coverage binding changed")
            for key in ("attention_reconstruction_l2", "attention_reconstruction_max_abs"):
                if key not in item or not isinstance(item[key], (int, float)) or not math.isfinite(float(item[key])) or item[key] < 0:
                    raise ValueError("circuit discovery attention reconstruction diagnostic is malformed")
            relative = item.get("attention_reconstruction_relative_l2")
            if relative is not None and (not isinstance(relative, (int, float)) or not math.isfinite(float(relative)) or relative < 0):
                raise ValueError("circuit discovery relative reconstruction diagnostic is malformed")
            output_ref = item.get("attention_output_ref")
            if not isinstance(output_ref, dict) or not _sha(item.get("attention_output_sha256")):
                raise ValueError("circuit discovery native attention-output payload is missing")
            output, output_payload = _payload_tensor(output_ref, vm_artifacts, label="native attention output")
            if output_payload.sha256 != item["attention_output_sha256"]:
                raise ValueError("circuit discovery native attention-output digest changed")
            if output.numel() == 0:
                raise ValueError("circuit discovery native attention-output payload is empty")
        if seen_layers != set(layers):
            raise ValueError("circuit discovery receiver coverage omitted a layer")
        candidates = observation.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= value["spec"].get("candidates_per_receiver", 1):
            raise ValueError("circuit discovery candidate bound changed")
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("circuit discovery candidate is malformed")
            required = {"id", "offset", "parent_cut", "layer", "kv_head", "source_row", "source_token_id",
                        "value_sha256", "key_sha256", "read_score", "group_write_l2", "query_heads",
                        "vocabulary", "contribution_ref", "reinstall"}
            if required - set(candidate) or "_write" in candidate:
                raise ValueError("circuit discovery candidate fields are incomplete")
            key = (candidate["layer"], candidate["kv_head"], candidate["source_row"])
            if key in seen:
                raise ValueError("circuit discovery candidate is duplicated")
            seen.add(key)
            expected_id = "carrier:" + content_digest({"parent_cut": observation["parent_cut"], "layer": candidate["layer"],
                                                           "kv_head": candidate["kv_head"], "source_row": candidate["source_row"]})[:24]
            if candidate["id"] != expected_id or candidate["offset"] != offset or candidate["parent_cut"] != observation["parent_cut"]:
                raise ValueError("circuit discovery candidate identity changed")
            if candidate["layer"] not in layers or type(candidate["kv_head"]) is not int or candidate["kv_head"] < 0:
                raise ValueError("circuit discovery candidate address is malformed")
            if type(candidate["source_row"]) is not int or not 0 <= candidate["source_row"] < row_count:
                raise ValueError("circuit discovery candidate source row is out of bounds")
            if (candidate["source_token_id"] != receiver_parent["token_ids"][candidate["source_row"]]
                    or not _sha(candidate["value_sha256"]) or not _sha(candidate["key_sha256"])):
                raise ValueError("circuit discovery candidate source binding changed")
            contribution, contribution_payload = _payload_tensor(
                candidate["contribution_ref"], vm_artifacts, label="circuit contribution"
            )
            if contribution.numel() == 0 or not contribution_payload.sha256:
                raise ValueError("circuit discovery contribution payload is empty")
            for key in ("read_score", "group_write_l2"):
                if not isinstance(candidate[key], (int, float)) or not math.isfinite(float(candidate[key])) or candidate[key] < 0:
                    raise ValueError("circuit discovery candidate score is malformed")
            heads = candidate["query_heads"]
            if not isinstance(heads, list) or not heads:
                raise ValueError("circuit discovery candidate query-head list is missing")
            head_ids = set()
            for head in heads:
                if (not isinstance(head, dict) or type(head.get("head")) is not int
                        or head["head"] in head_ids or not _sha(head.get("query_sha256"))):
                    raise ValueError("circuit discovery query-head binding is malformed")
                head_ids.add(head["head"])
                if not 0 <= head["attention"] <= 1 or not math.isfinite(float(head["attention"])):
                    raise ValueError("circuit discovery attention weight is malformed")
                if not math.isfinite(float(head["weighted_write_l2"])) or head["weighted_write_l2"] < 0:
                    raise ValueError("circuit discovery weighted write is malformed")
            vocabulary = candidate["vocabulary"]
            if not isinstance(vocabulary, dict) or vocabulary.get("semantic_cot_qualified") is not False:
                raise ValueError("circuit discovery vocabulary readout qualified semantic CoT")
            tokens = vocabulary.get("tokens")
            if not isinstance(tokens, list) or len(tokens) > value["spec"].get("top_k", 5):
                raise ValueError("circuit discovery vocabulary candidate bound changed")
            for token in tokens:
                if not isinstance(token, dict) or type(token.get("token_id")) is not int or token["token_id"] < 0 or not math.isfinite(float(token["score"])):
                    raise ValueError("circuit discovery vocabulary readout is malformed")
            by_candidate[candidate["id"]] = (observation, candidate)

    expected_tests = {(candidate_id, scale) for candidate_id in by_candidate for scale in scales}
    seen_tests = set()
    for test in tests:
        if not isinstance(test, dict):
            raise ValueError("circuit discovery test is malformed")
        required = {"id", "candidate_id", "receiver_offset", "scale", "act", "edited_parent_cut",
                    "head_locality_exact", "head_locality", "local_prediction", "reference", "candidate", "effect"}
        if required - set(test):
            raise ValueError("circuit discovery test fields are incomplete")
        key = (test["candidate_id"], float(test["scale"]))
        if key not in expected_tests or key in seen_tests:
            raise ValueError("circuit discovery test candidate/dose binding changed")
        seen_tests.add(key)
        observation, candidate = by_candidate[test["candidate_id"]]
        if test["receiver_offset"] != observation["offset"]:
            raise ValueError("circuit discovery test receiver parent changed")
        expected_id = "test:" + content_digest({"candidate": test["candidate_id"], "scale": test["scale"]})[:24]
        if test["id"] != expected_id or test["scale"] not in scales:
            raise ValueError("circuit discovery test identity/dose changed")
        act = test["act"]
        from .bench_hidden_memory import verify_write_act

        verify_write_act(act, artifacts, vm_artifacts)
        selector = act.get("selector")
        if (
            act.get("mode") != "replace"
            or act.get("parent_cut") != observation["parent_cut"]
            or act.get("donor_cut") != observation["parent_cut"]
            or selector != {"layers": [candidate["layer"]], "row_start": candidate["source_row"],
                            "row_stop": candidate["source_row"] + 1, "components": ["value"]}
            or test["edited_parent_cut"] != act.get("cut_id")
        ):
            raise ValueError("circuit discovery head-write selector/parent changed")
        _verify_head_locality(test, candidate, observation, artifacts, vm_artifacts, cache)
        _verify_local_prediction(test, candidate, observation, artifacts, vm_artifacts)
        reference = test["reference"]
        candidate_arm = test["candidate"]
        reference_rows = baseline["trajectory"][observation["offset"]:observation["offset"] + len(reference.get("trajectory", []))]
        if reference.get("trajectory") != reference_rows:
            raise ValueError("circuit discovery reference arm is not the baseline suffix")
        _verify_trajectory(reference, label=f"reference {test['id']}", start=observation["offset"],
                           expected_parent=observation["parent_cut"], flat=flat, spec=value["spec"],
                           artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest,
                           numerical_contract=cut_contract, cache=cache)
        _verify_trajectory(candidate_arm, label=f"candidate {test['id']}", start=observation["offset"],
                           expected_parent=act["cut_id"], flat=flat, spec=value["spec"],
                           artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest,
                           numerical_contract=cut_contract, cache=cache)
        expected_effect = _expected_compare(candidate_arm, reference, artifacts=artifacts,
                                            vm_artifacts=vm_artifacts, cache=cache)
        _close_structure(test["effect"], expected_effect, label="circuit discovery intervention effect")
    if seen_tests != expected_tests:
        raise ValueError("circuit discovery omitted a declared candidate dose")

    for candidate_id, (observation, candidate) in by_candidate.items():
        reinstall = candidate["reinstall"]
        if not isinstance(reinstall, dict) or not {"act", "future", "effect"} <= set(reinstall):
            raise ValueError("circuit discovery reinstall control is incomplete")
        reinstall_act = reinstall["act"]
        from .bench_hidden_memory import verify_write_act

        verify_write_act(reinstall_act, artifacts, vm_artifacts)
        last_test = next(test for test in tests if test["candidate_id"] == candidate_id and test["scale"] == scales[-1])
        if (
            reinstall_act.get("mode") != "copy"
            or reinstall_act.get("parent_cut") != last_test["act"]["cut_id"]
            or reinstall_act.get("donor_cut") != observation["parent_cut"]
            or reinstall_act.get("selector") != last_test["act"]["selector"]
        ):
            raise ValueError("circuit discovery reinstall Act binding changed")
        reference = last_test["reference"]
        _verify_trajectory(reinstall["future"], label=f"reinstall {candidate_id}", start=observation["offset"],
                           expected_parent=reinstall_act["cut_id"], flat=flat, spec=value["spec"],
                           artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest,
                           numerical_contract=cut_contract, cache=cache)
        expected_effect = _expected_compare(reinstall["future"], reference, artifacts=artifacts,
                                            vm_artifacts=vm_artifacts, cache=cache)
        _close_structure(reinstall["effect"], expected_effect, label="circuit discovery reinstall effect")
        if not expected_effect["closure_exact"] or any(
            not row["same_logits"] for row in expected_effect["steps"]
        ):
            raise ValueError("circuit discovery reinstall closure changed")

    graph = value.get("graph")
    if graph != _expected_graph(receivers, tests):
        raise ValueError("circuit discovery graph is not derived from frozen observations")
    if graph.get("semantic_cot_qualified") is not False:
        raise ValueError("circuit discovery graph qualified semantic CoT")
    return True


def verify_artifact(value, artifacts, manifest, vm_artifacts=None):
    """Verify a frozen native circuit-discovery report without model execution."""

    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return False
    return _verify_report(value, artifacts, manifest, vm_artifacts)


__all__ = ["SCHEMA", "verify_artifact"]
