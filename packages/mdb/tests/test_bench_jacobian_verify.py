"""Dependency-light closure tests for Jacobian Bench sidecars."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from mdb.bench_jacobian_verify import verify_artifact


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sealed(value):
    return {**value, "fingerprint": _digest(value)}


def _tensor(character, shape, dtype="torch.float32"):
    return {"sha256": character * 64, "shape": list(shape), "dtype": dtype}


class _Artifacts:
    def __init__(self):
        self.values = {}

    @staticmethod
    def expected_ref(digest):
        return f"cas://sha256/{digest}"

    def put(self, value):
        reference = self.expected_ref(_digest(value))
        self.values[reference] = deepcopy(value)
        return reference

    def get(self, reference):
        return deepcopy(self.values[reference])


def _fixture(*, terminal="commit"):
    artifacts = _Artifacts()
    identity = {
        "model_id": "fixture/qwen",
        "revision": "r1",
        "weights_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "dtype": "torch.float32",
    }
    abi = _digest(identity)
    numerical = "4" * 64
    lens = _sealed(
        {
            "schema": "saturn-jacobian-lens-v1",
            "transport": _tensor("5", (2, 2)),
            "execution_compatibility": {
                "permitted_application_lanes": ["cpu-float32-eager"]
            },
            "coordinate_map": {"completed_layer_boundary": 1},
        }
    )
    profile = {
        "application_lane": "cpu-float32-eager",
        "model_abi_sha256": abi,
        "numerical_contract": numerical,
        "bench_lane": "native",
        "attention_backend": "native-eager",
        "dtype": "torch.float32",
    }
    pin = {
        "schema": "mdb-bench-jacobian-pin-v1",
        "lens_manifest": lens,
        "transport_ref": lens["transport"],
        "profile": profile,
        "model_identity": identity,
        "semantic_writer_qualified": False,
    }
    pin_ref = artifacts.put(pin)
    activation = _tensor("6", (1, 1, 2))
    cursor = {"token": 0, "phase": "layers", "next_layer": 1, "pending_token": 3}
    source_cut = {
        "identity": identity,
        "lane": "native",
        "numerical_contract": numerical,
        "branch": "main",
        "token_ids": [1, 2],
        "prompt_length": 2,
        "pending": 3,
        "parent_cut": None,
        "intervention_lineage": [],
        "execution": {"next_layer": 1, "hidden": activation},
    }
    source_cut_ref = artifacts.put(source_cut)
    source = {
        "model_abi_sha256": abi,
        "numerical_contract": numerical,
        "session_id": "bench",
        "cut_ref": source_cut_ref,
        "branch": "main",
        "token_position": 2,
        "completed_layer_boundary": 1,
        "activation": activation,
        "input_tokens_sha256": _digest([1, 2]),
        "intervention_lineage": [],
        "cursor": cursor,
    }
    binding = {
        "model_abi_sha256": abi,
        "numerical_contract": numerical,
        "session_id": "bench",
    }
    observation = _sealed(
        {
            "schema": "saturn-j-workspace-observation-v1",
            "source": source,
            "lens": {"fingerprint": lens["fingerprint"]},
            "execution_binding": binding,
            "readout_policy": {
                "application_lane": profile["application_lane"],
                "token_subset": [7, 9],
                "top_k": 1,
            },
            "ranked_token_count": 1,
            "candidate_token_count": 2,
        }
    )
    read = {
        "schema": "mdb-bench-jacobian-read-v1",
        "pin_ref": pin_ref,
        "source": source,
        "source_ref": activation,
        "observation": observation,
        "readout_refs": {
            "transported": _tensor("7", (1, 1, 2)),
            "selected_token_ids": _tensor("8", (1, 1, 1), "torch.int64"),
            "selected_raw_scores": _tensor("9", (1, 1, 1)),
            "selected_normalized_scores": _tensor("a", (1, 1, 1)),
        },
        "options": {"token_subset": [7, 9], "top_k": 1, "logical_address": None},
        "model_identity": identity,
        "native_mutation": False,
        "semantic_writer_qualified": False,
    }
    read_ref = artifacts.put(read)
    candidate_tensor = _tensor("b", (1, 1, 2))
    delta_tensor = _tensor("c", (1, 1, 2))
    write = _sealed(
        {
            "schema": "saturn-j-prepared-write-v1",
            "source": source,
            "lens": {
                "fingerprint": lens["fingerprint"],
                "application_lane": profile["application_lane"],
            },
            "execution_binding": binding,
            "candidate": candidate_tensor,
            "delta": delta_tensor,
            "consumer": {"name": "fixture", "horizon": 1},
        }
    )
    candidate = {
        "schema": "mdb-bench-jacobian-candidate-v1",
        "attempt_id": "e" * 32,
        "observation_ref": read_ref,
        "pin_ref": pin_ref,
        "source": source,
        "write": write,
        "candidate_ref": candidate_tensor,
        "delta_ref": delta_tensor,
        "semantic_writer_qualified": False,
    }
    candidate_ref = artifacts.put(candidate)
    act_parent = {**source_cut, "parent_cut": source_cut_ref, "label": "act-parent"}
    parent_ref = artifacts.put(act_parent)
    act_id = _digest(
        {
            "parent": parent_ref,
            "path": "residual",
            "mode": "replace",
            "payload": candidate_tensor["sha256"],
        }
    )
    preview = {
        "kind": "jacobian-preview",
        "act_id": act_id,
        "jacobian_candidate_ref": candidate_ref,
        "observation_ref": read_ref,
        "parent_cut": parent_ref,
    }
    verification = {"accepted": terminal == "commit"}
    backend = (
        {
            "status": "committed",
            "current_handle": "candidate",
            "state_fingerprint": "state",
            "output_visible": True,
            "execution_mode": "native-residual-install",
        }
        if terminal == "commit"
        else {
            "status": "restored",
            "restored_to": "parent",
            "reason": "verifier rejected candidate",
            "output_visible": False,
        }
    )
    receipt_body = {
        "schema": "saturn-model-os-execution-receipt-v1",
        "kernel_schema": "saturn-model-os-kernel-v1",
        "status": "committed" if terminal == "commit" else "restored",
        "decision": "commit" if terminal == "commit" else "restore",
        "package_fingerprint": "package",
        "model_abi_fingerprint": "abi",
        "state_schema_fingerprint": "schema",
        "program_fingerprint": "program",
        "plan_fingerprint": "plan",
        "parent_handle": "parent",
        "child_handle": "candidate",
        "consumer_id": "bench.qwen.residual-install",
        "continuation": {
            "accepted": True,
            "consumer_id": "bench.qwen.residual-install",
            "execution_mode": "native-residual-install",
            "horizon": 0,
            "boundary": {
                "session_id": "bench",
                "branch": "main",
                "parent_cut": parent_ref,
                **cursor,
            },
            "payload": candidate_tensor,
        },
        "verification": verification,
        "backend": backend,
        "metadata": {"terminal_claim": False},
    }
    receipt = {**receipt_body, "content_sha256": _digest(receipt_body)}
    receipt_ref = artifacts.put(receipt)
    terminal_event = {
        "kind": "residual-act" if terminal == "commit" else "act-rejected",
        "act_id": act_id,
        "parent": parent_ref,
        "receipt_ref": receipt_ref,
        "jacobian_candidate_ref": candidate_ref,
    }
    cuts = {
        source_cut_ref: {"label": "jacobian-source"},
        parent_ref: {"label": "act-parent"},
    }
    if terminal == "commit":
        committed = {
            **source_cut,
            "parent_cut": parent_ref,
            "label": "act-committed",
            "intervention_lineage": [act_id],
            "execution": {"next_layer": 1, "hidden": candidate_tensor},
        }
        committed_ref = artifacts.put(committed)
        cuts[committed_ref] = {"label": "act-committed"}
    manifest = {
        "identity": identity,
        "session_id": "bench",
        "lane": "native",
        "attention_backend": "native-eager",
        "events": [preview, terminal_event],
        "cuts": cuts,
    }
    return artifacts, manifest, candidate_ref, read_ref


def test_commit_closes_native_receipt_and_committed_hidden_state():
    artifacts, manifest, candidate_ref, _ = _fixture()
    assert verify_artifact(artifacts.get(candidate_ref), artifacts, manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("content_sha256", "0" * 64, "receipt seal"),
        ("consumer_id", "other.consumer", "receipt seal"),
    ],
)
def test_commit_rejects_tampered_native_receipt(field, value, message):
    artifacts, manifest, candidate_ref, _ = _fixture()
    event = manifest["events"][-1]
    receipt = artifacts.get(event["receipt_ref"])
    receipt[field] = value
    artifacts.values[event["receipt_ref"]] = receipt
    with pytest.raises(ValueError, match=message):
        verify_artifact(artifacts.get(candidate_ref), artifacts, manifest)


def test_commit_requires_candidate_bytes_in_committed_cut():
    artifacts, manifest, candidate_ref, _ = _fixture()
    committed_ref = next(
        reference for reference, row in manifest["cuts"].items() if row["label"] == "act-committed"
    )
    committed = artifacts.get(committed_ref)
    committed["execution"]["hidden"] = _tensor("d", (1, 1, 2))
    artifacts.values[committed_ref] = committed
    with pytest.raises(ValueError, match="native committed cut"):
        verify_artifact(artifacts.get(candidate_ref), artifacts, manifest)


def test_rejection_receipt_is_terminal_without_a_committed_cut():
    artifacts, manifest, candidate_ref, _ = _fixture(terminal="restore")
    assert verify_artifact(artifacts.get(candidate_ref), artifacts, manifest)
    manifest["events"].append(deepcopy(manifest["events"][-1]))
    with pytest.raises(ValueError, match="event cardinality"):
        verify_artifact(artifacts.get(candidate_ref), artifacts, manifest)


def test_retained_unselected_candidate_has_no_event_obligation():
    artifacts, manifest, candidate_ref, _ = _fixture()
    manifest["events"] = []
    assert verify_artifact(artifacts.get(candidate_ref), artifacts, manifest)


def test_readout_inventory_and_rank_are_verified():
    artifacts, manifest, _, read_ref = _fixture()
    read = artifacts.get(read_ref)
    read["readout_refs"].pop("selected_raw_scores")
    with pytest.raises(ValueError, match="inventory"):
        verify_artifact(read, artifacts, manifest)
    read = artifacts.get(read_ref)
    read["observation"]["ranked_token_count"] = 2
    read["observation"] = _sealed(
        {key: value for key, value in read["observation"].items() if key != "fingerprint"}
    )
    with pytest.raises(ValueError, match="geometry or ranking"):
        verify_artifact(read, artifacts, manifest)
