"""Dependency-light reference closure for Jacobian bench sidecars."""

from __future__ import annotations

import hashlib
import json


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _seal(value):
    if value.get("fingerprint") != _digest({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError("Jacobian sidecar metadata seal differs")


def _tensor(reference, identity):
    if (
        reference["sha256"] != identity["sha256"]
        or reference["shape"] != identity["shape"]
        or reference["dtype"] != identity["dtype"]
    ):
        raise ValueError("Jacobian tensor reference differs from declared identity")


def _receipt_seal(receipt):
    body = dict(receipt)
    declared = body.pop("content_sha256", None)
    if declared != _digest(body):
        raise ValueError("Jacobian native Act receipt seal differs")
    if (
        receipt.get("schema") != "saturn-model-os-execution-receipt-v1"
        or receipt.get("kernel_schema") != "saturn-model-os-kernel-v1"
        or receipt.get("consumer_id") != "bench.qwen.residual-install"
        or receipt.get("metadata", {}).get("terminal_claim") is not False
    ):
        raise ValueError("Jacobian native Act receipt contract differs")


def _readout_closure(value, source, pin):
    references = value.get("readout_refs")
    expected = {
        "transported",
        "selected_token_ids",
        "selected_raw_scores",
        "selected_normalized_scores",
    }
    if not isinstance(references, dict) or set(references) != expected:
        raise ValueError("Jacobian readout tensor inventory differs")
    observation = value["observation"]
    activation = source["activation"]
    transported = references["transported"]
    selected_ids = references["selected_token_ids"]
    selected_raw = references["selected_raw_scores"]
    selected_normalized = references["selected_normalized_scores"]
    ranked = observation.get("ranked_token_count")
    candidates = observation.get("candidate_token_count")
    top_k = value.get("options", {}).get("top_k")
    selected_shape = [*activation["shape"][:-1], ranked]
    if (
        not isinstance(ranked, int)
        or isinstance(ranked, bool)
        or ranked < 1
        or not isinstance(candidates, int)
        or isinstance(candidates, bool)
        or candidates < ranked
        or not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
        or ranked != min(top_k, candidates)
        or transported.get("shape") != activation["shape"]
        or transported.get("dtype") != pin["profile"]["dtype"]
        or selected_ids.get("shape") != selected_shape
        or selected_ids.get("dtype") != "torch.int64"
        or selected_raw.get("shape") != selected_shape
        or selected_normalized.get("shape") != selected_shape
        or selected_raw.get("dtype") != activation["dtype"]
        or selected_normalized.get("dtype") != activation["dtype"]
        or observation.get("readout_policy", {}).get("top_k") != top_k
        or observation.get("readout_policy", {}).get("token_subset")
        != value.get("options", {}).get("token_subset")
    ):
        raise ValueError("Jacobian readout geometry or ranking contract differs")


def _native_receipt(receipt, *, event, preview, candidate, source, manifest):
    _receipt_seal(receipt)
    continuation = receipt.get("continuation", {})
    boundary = continuation.get("boundary", {})
    backend = receipt.get("backend", {})
    verification = receipt.get("verification", {})
    if (
        continuation.get("accepted") is not True
        or continuation.get("consumer_id") != receipt["consumer_id"]
        or continuation.get("execution_mode") != "native-residual-install"
        or continuation.get("horizon") != 0
        or continuation.get("payload") != candidate["candidate_ref"]
        or boundary.get("parent_cut") != event["parent"]
        or boundary.get("session_id") != manifest["session_id"]
        or boundary.get("branch") != source["branch"]
        or boundary.get("pending_token") != source["cursor"]["pending_token"]
        or boundary.get("next_layer") != source["cursor"]["next_layer"]
        or event.get("act_id") != preview.get("act_id")
        or event.get("parent") != preview.get("parent_cut")
    ):
        raise ValueError("Jacobian candidate native Act linkage differs")
    if event.get("kind") == "residual-act":
        if (
            receipt.get("decision") != "commit"
            or receipt.get("status") != "committed"
            or verification.get("accepted") is not True
            or backend.get("status") != "committed"
            or backend.get("execution_mode") != "native-residual-install"
            or backend.get("output_visible") is not True
        ):
            raise ValueError("Jacobian candidate commit receipt differs")
    elif event.get("kind") == "act-rejected":
        if (
            receipt.get("decision") != "restore"
            or receipt.get("status") != "restored"
            or verification.get("accepted") is not False
            or backend.get("status") != "restored"
            or "execution_mode" in backend
            or backend.get("output_visible") is not False
        ):
            raise ValueError("Jacobian candidate rejection receipt differs")
    else:  # pragma: no cover - caller filters terminal kinds
        raise ValueError("unsupported Jacobian terminal event")


def _committed_cut(artifacts, manifest, *, event, source, candidate):
    matches = []
    source_cut = artifacts.get(source["cut_ref"])
    for reference, record in manifest["cuts"].items():
        if record.get("label") != "act-committed":
            continue
        cut = artifacts.get(reference)
        execution = cut.get("execution")
        if (
            cut.get("parent_cut") == event["parent"]
            and cut.get("branch") == source["branch"]
            and cut.get("identity") == manifest["identity"]
            and cut.get("lane") == manifest["lane"]
            and cut.get("numerical_contract") == source["numerical_contract"]
            and cut.get("token_ids") == source_cut["token_ids"]
            and cut.get("prompt_length") == source_cut["prompt_length"]
            and cut.get("pending") == source_cut["pending"]
            and cut.get("intervention_lineage")
            and cut["intervention_lineage"][-1] == event["act_id"]
            and execution is not None
            and execution.get("next_layer") == source["completed_layer_boundary"]
        ):
            try:
                _tensor(execution["hidden"], candidate["candidate_ref"])
            except (KeyError, TypeError, ValueError):
                continue
            matches.append(reference)
    if not matches:
        raise ValueError("Jacobian committed candidate lacks a native committed cut")


def verify_artifact(value, artifacts, manifest):
    if not isinstance(value, dict) or value.get("schema") not in {
        "mdb-bench-jacobian-pin-v1",
        "mdb-bench-jacobian-read-v1",
        "mdb-bench-jacobian-candidate-v1",
    }:
        return False
    schema = value["schema"]
    if schema == "mdb-bench-jacobian-pin-v1":
        lens = value["lens_manifest"]
        _seal(lens)
        if value["model_identity"] != manifest["identity"]:
            raise ValueError("Jacobian pin belongs to a different model")
        profile = value["profile"]
        if (
            profile["bench_lane"] != manifest["lane"]
            or profile["attention_backend"] != manifest["attention_backend"]
            or profile["model_abi_sha256"] != _digest(manifest["identity"])
        ):
            raise ValueError("Jacobian pin numerical profile differs")
        _tensor(value["transport_ref"], lens["transport"])
        if profile["dtype"] != lens["transport"]["dtype"]:
            raise ValueError("Jacobian pin dtype differs from transport")
        if (
            profile["application_lane"]
            not in lens["execution_compatibility"]["permitted_application_lanes"]
        ):
            raise ValueError("Jacobian pin lane is not permitted")
        return True
    pin = artifacts.get(value["pin_ref"])
    if pin.get("schema") != "mdb-bench-jacobian-pin-v1":
        raise ValueError("Jacobian sidecar lacks a lens pin")
    verify_artifact(pin, artifacts, manifest)
    source = value["source"]
    cut = artifacts.get(source["cut_ref"])
    execution = cut["execution"]
    if (
        cut["identity"] != manifest["identity"]
        or cut["branch"] != source["branch"]
        or source["model_abi_sha256"] != pin["profile"]["model_abi_sha256"]
        or cut["numerical_contract"] != source["numerical_contract"]
        or source["numerical_contract"] != pin["profile"]["numerical_contract"]
        or source["session_id"] != manifest["session_id"]
        or execution is None
        or execution["next_layer"] != source["completed_layer_boundary"]
        or source["completed_layer_boundary"]
        != pin["lens_manifest"]["coordinate_map"]["completed_layer_boundary"]
        or len(cut["token_ids"]) != source["token_position"]
        or _digest(cut["token_ids"]) != source["input_tokens_sha256"]
        or cut["intervention_lineage"] != source["intervention_lineage"]
        or cut["pending"] != source["cursor"]["pending_token"]
        or source["cursor"]["next_layer"] != execution["next_layer"]
    ):
        raise ValueError("Jacobian source cut binding differs")
    _tensor(execution["hidden"], source["activation"])
    binding = {
        "model_abi_sha256": pin["profile"]["model_abi_sha256"],
        "numerical_contract": pin["profile"]["numerical_contract"],
        "session_id": manifest["session_id"],
    }
    if schema == "mdb-bench-jacobian-read-v1":
        _tensor(value["source_ref"], source["activation"])
        observation = value["observation"]
        _seal(observation)
        if (
            observation["source"] != source
            or observation["lens"]["fingerprint"] != pin["lens_manifest"]["fingerprint"]
            or observation["execution_binding"] != binding
            or observation["readout_policy"]["application_lane"]
            != pin["profile"]["application_lane"]
        ):
            raise ValueError("Jacobian observation source/lens differs")
        _readout_closure(value, source, pin)
        return True
    observed = artifacts.get(value["observation_ref"])
    if observed.get("schema") != "mdb-bench-jacobian-read-v1":
        raise ValueError("Jacobian candidate lacks an observation")
    verify_artifact(observed, artifacts, manifest)
    write = value["write"]
    _seal(write)
    attempt_id = value.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 32
        or any(character not in "0123456789abcdef" for character in attempt_id)
    ):
        raise ValueError("Jacobian candidate attempt identity differs")
    if (
        write["source"] != source
        or observed["source"] != source
        or write["lens"]["fingerprint"] != pin["lens_manifest"]["fingerprint"]
        or write["lens"]["application_lane"] != pin["profile"]["application_lane"]
        or write["execution_binding"] != binding
    ):
        raise ValueError("Jacobian candidate source/lens differs")
    _tensor(value["candidate_ref"], write["candidate"])
    _tensor(value["delta_ref"], write["delta"])
    candidate_ref = artifacts.expected_ref(_digest(value))
    previews = [
        event
        for event in manifest["events"]
        if event.get("kind") == "jacobian-preview"
        and event.get("jacobian_candidate_ref") == candidate_ref
    ]
    terminals = [
        event
        for event in manifest["events"]
        if event.get("kind") in {"residual-act", "act-rejected"}
        and event.get("jacobian_candidate_ref") == candidate_ref
    ]
    # A retained candidate pack from a failed atomic preview is intentionally
    # outside the selected event closure. Only selected candidates have event
    # cardinality and native terminal obligations.
    if not previews:
        if terminals:
            raise ValueError("Jacobian terminal event lacks a selected preview")
        return True
    if len(previews) != 1 or len(terminals) > 1:
        raise ValueError("Jacobian candidate event cardinality differs")
    preview = previews[0]
    expected_act = _digest(
        {
            "parent": preview["parent_cut"],
            "path": "residual",
            "mode": "replace",
            "payload": value["candidate_ref"]["sha256"],
        }
    )
    if (
        preview.get("act_id") != expected_act
        or preview.get("observation_ref") != value["observation_ref"]
    ):
        raise ValueError("Jacobian preview binding differs")
    parent = artifacts.get(preview["parent_cut"])
    if parent.get("parent_cut") != source["cut_ref"]:
        raise ValueError("Jacobian candidate native parent differs")
    if terminals:
        terminal = terminals[0]
        if (
            terminal.get("act_id") != preview["act_id"]
            or terminal.get("parent") != preview["parent_cut"]
        ):
            raise ValueError("Jacobian terminal event differs from preview")
        native = artifacts.get(terminal["receipt_ref"])
        _native_receipt(
            native,
            event=terminal,
            preview=preview,
            candidate=value,
            source=source,
            manifest=manifest,
        )
        if terminal["kind"] == "residual-act":
            _committed_cut(
                artifacts,
                manifest,
                event=terminal,
                source=source,
                candidate=value,
            )
    return True
