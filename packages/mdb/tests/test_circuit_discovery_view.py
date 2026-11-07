from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mdb.circuit_discovery_view import DISCOVERY_SCHEMA, NATIVE_CONSUMER, typed_events
from mdb.temporal_causal_viewer import TemporalViewerError, normalize_evidence


def _trajectory(parent: str, cut: str, logits: str):
    return {
        "trajectory": [
            {
                "offset": 0,
                "parent_cut": parent,
                "cut_id": cut,
                "input_authority": "native-greedy",
                "consumed_token_id": 11,
                "logits_sha256": logits,
                "next_token_id": 12,
                "margin": 0.5,
            }
        ],
        "closure": {"sha256": f"closure:{cut}"},
    }


def _report():
    parent = "cas://sha256/parent"
    candidate_a = {
        "id": "carrier:a",
        "offset": 0,
        "parent_cut": parent,
        "layer": 7,
        "kv_head": 2,
        "source_row": 4,
        "source_token_id": 99,
        "value_sha256": "value-shared",
        "key_sha256": "key-a",
        "read_score": 3.25,
        "group_write_l2": 1.5,
    }
    candidate_b = {**candidate_a, "id": "carrier:b", "offset": 2, "key_sha256": "key-b"}
    good_test = {
        "id": "test:good",
        "candidate_id": candidate_a["id"],
        "receiver_offset": 0,
        "scale": 0.5,
        "act": {"parent_cut": parent, "receipt_ref": "cas://sha256/act-good"},
        "edited_parent_cut": "cas://sha256/edited-good",
        "head_locality_exact": True,
        "reference": _trajectory(parent, "cas://sha256/ref", "ref-logits"),
        "candidate": _trajectory("cas://sha256/edited-good", "cas://sha256/candidate", "candidate-logits"),
        "effect": {
            "alignment": "plan-offset; generated tokens remain branch-owned",
            "closure_exact": False,
            "steps": [
                {
                    "offset": 1,
                    "same_consumed_token": True,
                    "same_history": False,
                    "same_logits": False,
                    "same_next_token": True,
                    "logit_l2": 1.25,
                    "logit_max_abs": 0.75,
                    "signed_margin_effect": -0.25,
                }
            ],
        },
    }
    unknown_test = {
        **good_test,
        "id": "test:unknown-margin",
        "scale": 1.0,
        "act": {"parent_cut": parent, "receipt_ref": "cas://sha256/act-unknown"},
        "effect": {
            "alignment": "plan-offset; generated tokens remain branch-owned",
            "steps": [{"offset": 2, "logit_l2": 2.5, "logit_max_abs": 1.0, "signed_margin_effect": None}],
        },
    }
    return {
        "schema": DISCOVERY_SCHEMA,
        "name": "read-circuit",
        "session_id": "session-1",
        "model_identity": {"model_id": "qwen/tiny", "revision": "r1"},
        "numerical_contract": {"dtype": "bfloat16"},
        "parent_cut": parent,
        "spec": {"name": "read-circuit"},
        "spec_sha256": "spec-digest",
        "input_authority": {"input_segments": "caller-supplied"},
        "mechanics_status": "valid",
        "instrument_status": "native-consumers-executed",
        "selected_parent_restored_exact": True,
        "sibling_selections_preserved": True,
        "receivers": [
            {
                "offset": 0,
                "parent_cut": parent,
                "receiver_token_id": 11,
                "coverage": [{"offset": 0, "layer": 7, "attention_reconstruction_l2": 0.01}],
                "candidates": [candidate_a],
            },
            {
                "offset": 2,
                "parent_cut": "cas://sha256/parent-2",
                "receiver_token_id": 13,
                "coverage": [],
                "candidates": [candidate_b],
            },
        ],
        "tests": [good_test, unknown_test],
        "graph": {
            "nodes": [
                {"id": "receiver:0", "kind": "native-receiver", "offset": 0},
                {"id": "receiver:2", "kind": "native-receiver", "offset": 2},
                {"id": "carrier:a", "kind": "versioned-memory-value", "layer": 7},
                {"id": "carrier:b", "kind": "versioned-memory-value", "layer": 7},
            ],
            "edges": [
                {
                    "source": "carrier:a",
                    "target": "receiver:0",
                    "kind": "observed-native-read",
                    "causal_dependency_tested": True,
                },
                {
                    "source": "carrier:b",
                    "target": "receiver:2",
                    "kind": "observed-native-read",
                    "causal_dependency_tested": True,
                },
                {
                    "source": "carrier:a",
                    "target": "carrier:b",
                    "kind": "same-payload-observation",
                    "value_sha256": "value-shared",
                    "causal_dependency_tested": False,
                },
                {
                    "source": "receiver:0",
                    "target": "receiver:2",
                    "kind": "native-execution-order",
                    "causal_dependency_tested": False,
                },
            ],
        },
    }


def test_typed_events_separates_native_reads_and_intervention_responses():
    payload = typed_events(_report())
    assert payload["schema"] == "mdb-dual-time-causal-events-v1"
    assert payload["binding"]["input_binding"]["native_consumer"] == NATIVE_CONSUMER

    candidate = next(event for event in payload["events"] if event["event_kind"] == "observed-native-read")
    assert candidate["evidence_class"] == "observation"
    assert candidate["token_time"] == 0
    assert candidate["layer_depth"] == 7
    assert candidate["effect"]["read_score"] == 3.25

    response = next(
        event
        for event in payload["events"]
        if event["event_kind"] == "intervention-response" and event["event_id"].startswith("intervention-response:test:good")
    )
    assert response["evidence_class"] == "causal_evidence"
    assert response["token_time"] == 1
    assert response["layer_depth"] == "?"
    assert response["future_offset"] == 1
    assert response["native_consumer"] == NATIVE_CONSUMER


def test_unknown_margin_preserves_logit_geometry_as_observation():
    payload = typed_events(_report())
    response = next(
        event
        for event in payload["events"]
        if event["event_kind"] == "intervention-response" and "unknown-margin" in event["event_id"]
    )
    assert response["evidence_class"] == "observation"
    assert response["effect"]["signed_effect"] is None
    assert response["effect"]["logit_l2"] == 2.5
    assert response["layer_depth"] == "?"
    assert normalize_evidence([payload])["events"]


def test_graph_equality_and_execution_order_remain_observations():
    payload = typed_events(_report())
    same_payload = next(event for event in payload["events"] if event["event_kind"] == "same-payload-observation")
    order = next(event for event in payload["events"] if event["event_kind"] == "native-execution-order")
    assert same_payload["evidence_class"] == "observation"
    assert same_payload["payload"]["causal_dependency_tested"] is False
    assert order["evidence_class"] == "observation"
    assert order["payload"]["causal_dependency_tested"] is False
    assert order["future_offset"] == 2


def test_invalid_schema_and_nonfinite_values_are_rejected():
    bad = _report()
    bad["schema"] = "other-v1"
    with pytest.raises(TemporalViewerError, match="schema"):
        typed_events(bad)
    bad = _report()
    bad["tests"][0]["effect"]["steps"][0]["logit_l2"] = float("nan")
    with pytest.raises(TemporalViewerError, match="finite JSON"):
        typed_events(bad)


def test_cli_renders_report_without_framework_imports(tmp_path):
    report_path = tmp_path / "report.json"
    output = tmp_path / "view.html"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "mdb.circuit_discovery_view", str(report_path), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    summary = json.loads(result.stdout)
    assert summary["events"] >= 5
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mdb.circuit_discovery_view; assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    assert probe.returncode == 0
