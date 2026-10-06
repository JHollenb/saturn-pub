from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mdb.temporal_causal_viewer import (
    TemporalViewerError,
    normalize_evidence,
    render_html,
)


def _measurement():
    base = {
        "schema": "saturn-bench-circuit-measurement-v1",
        "model_identity": {"model_id": "tiny/test", "revision": "abc"},
        "numerical_contract": {"dtype": "float32"},
        "parent_cut": "cas://sha256/parent",
        "spec_sha256": "spec-digest",
        "spec": {
            "name": "path",
            "contrasts": [{"name": "ablate", "mode": "scale", "value": 0.0}],
            "observable": {"kind": "token_margin", "token_id": 7, "against_token_id": 9},
        },
        "instrument_status": "native-consumer-executed",
        "rows": [],
    }

    def row(name, role, signed, *, site, act=None):
        return {
            "name": name,
            "role": role,
            "parent_cut": base["parent_cut"],
            "site": site,
            "site_cut": f"cas://sha256/site-{site}",
            "replay_cut": f"cas://sha256/replay-{name}",
            "replay_stage": "site",
            "evidence_ref": f"cas://sha256/evidence-{name}",
            "instrument_status": "native-consumer-executed",
            "source_address": "residual",
            "act": act,
            "first": {"value": 1.5, "closure": {"cursor": {"token": 4}}},
            "future": {"value": 1.25},
            "metrics": {
                "memory": {"addresses": {"kv.layer.0": {"sha256": "memory-digest"}}},
                "first_effect": {"signed_effect": signed / 2},
                "future_effect": {"signed_effect": signed},
            },
        }

    base["rows"] = [
        row("native", "reference", 0.0, site=0),
        row(
            "ablate",
            "candidate",
            -0.75,
            site=1,
            act={"act_id": "a1", "receipt_ref": "cas://sha256/receipt"},
        ),
    ]
    return base


def _trace():
    return {
        "schema": "saturn-thott-circuit-thought-v1",
        "circuit_id": "trace-1",
        "model_identity": {"model_id": "tiny/test", "revision": "abc"},
        "specimen": {"parent_cut": "cas://sha256/parent"},
        "content_sha256": "trace-digest",
        "nodes": [
            {
                "node_id": "producer",
                "role": "producer",
                "address": "residual.layer.1",
                "layer": 1,
                "position": None,
                "component": "residual",
                "metadata": {"logical_address": "answer-source"},
            }
        ],
        "edges": [],
        "interventions": [
            {
                "name": "half-dose",
                "node_id": "producer",
                "branch": "half",
                "dose": 0.5,
                "local_effect": {"signed_effect": -0.1},
                "consumer_effect": {"signed_effect": -0.2},
                "behavior_effect": {},
                "status": "observed_intervention",
            }
        ],
        "events": [
            {
                "sequence": 0,
                "event_key": "e0",
                "node_id": "producer",
                "stage": "write",
                "phase": "decode",
                "execution_step": 5,
                "summary": {"available": True, "sample_l2": 2.0},
                "metadata": {"positions": [5]},
            }
        ],
    }


def test_normalizes_same_parent_measurement_and_trace_without_semantic_inference():
    bundle = normalize_evidence([_measurement(), _trace()], sources=["report", "trace"])
    assert bundle["schema"] == "mdb-dual-time-causal-view-v1"
    assert bundle["binding"]["model_identity"]["model_id"] == "tiny/test"
    assert [row["arm_kind"] for row in bundle["branches"]] == ["native", "ablation"]
    assert all(row["same_parent"] for row in bundle["branches"])
    ablation = bundle["branches"][1]
    assert ablation["effect"]["future_signed"] == -0.75
    assert ablation["dose"] == {"value": 0.0, "mode": "scale", "source": "measurement_spec"}
    assert ablation["evidence_class"] == "causal_evidence"
    assert bundle["branches"][0]["evidence_class"] == "observation"
    trace_event = next(event for event in bundle["events"] if event["event_id"].startswith("thought:"))
    assert trace_event["event_kind"] == "commit"
    assert trace_event["evidence_class"] == "observation"
    assert trace_event["logical_address"] == "residual.layer.1"
    trace_intervention = next(
        event for event in bundle["events"] if event["event_id"].startswith("thought-intervention:")
    )
    assert trace_intervention["dose"]["value"] == 0.5
    assert trace_intervention["effect"]["future_signed"] == -0.2
    assert trace_intervention["evidence_class"] == "observation"
    missing = next(
        cell
        for cell in bundle["cells"]
        if cell["token_time"] == 4 and cell["layer_depth"] == 0
    )
    assert missing["status"] == "observed"  # native measurement occupies this coordinate
    assert any(cell["status"] == "missing" for cell in bundle["cells"])


def test_unknown_coordinates_are_explicit():
    value = _trace()
    value["nodes"][0]["layer"] = None
    value["nodes"][0]["position"] = None
    value["events"][0]["execution_step"] = -1
    value["events"][0]["metadata"] = {}
    bundle = normalize_evidence([value])
    assert bundle["axes"]["layer_depth"] == ["?"]
    assert bundle["events"][0]["layer_depth"] == "?"


def test_rejects_parent_or_model_binding_mismatch():
    bad = _measurement()
    bad["rows"][1]["parent_cut"] = "cas://sha256/other"
    with pytest.raises(TemporalViewerError, match="parent_cut"):
        normalize_evidence([bad])
    trace = _trace()
    trace["model_identity"] = {"model_id": "different"}
    with pytest.raises(TemporalViewerError, match="model identities"):
        normalize_evidence([_measurement(), trace])


def test_typed_events_require_actual_causal_fields():
    payload = {
        "schema": "mdb-dual-time-causal-events-v1",
        "binding": {"model_identity": {"model_id": "m"}, "parent_cut": "cut"},
        "branches": [{"branch_id": "swap", "role": "swap", "arm_kind": "swap"}],
        "events": [
            {
                "token_time": 2,
                "layer_depth": 3,
                "event_kind": "read",
                "branch_id": "swap",
                "evidence_class": "causal_evidence",
                "same_parent": True,
                "native_consumer_executed": True,
                "effect": {"signed_effect": 0.25},
            }
        ],
    }
    bundle = normalize_evidence([payload])
    assert bundle["branches"][0]["arm_kind"] == "swap"
    assert bundle["events"][0]["evidence_class"] == "causal_evidence"
    del payload["events"][0]["native_consumer_executed"]
    with pytest.raises(TemporalViewerError, match="causal_evidence requires"):
        normalize_evidence([payload])


def test_pass_diff_adapter_preserves_positions_and_receipts_as_observations():
    payload = {
        "schema": "saturn-pass-causal-state-diff-v1",
        "report_id": "diff-1",
        "pass_transition": {"from_pass": 1, "to_pass": 2},
        "binding": {
            "model_identity": {"model_id": "m"},
            "common_parent_statecut_id": "cut-parent",
            "identical_followup_query": True,
            "native_consumer": "logits",
        },
        "input_authority": {
            "positions": [
                {
                    "logical_position": 7,
                    "physical_position": 3,
                    "role": "appended_question",
                    "input_authority": "caller-supplied",
                }
            ]
        },
        "state": {
            "old_rows": {
                "rows": [
                    {
                        "logical_position": 8,
                        "parent_physical_position": 4,
                        "child_physical_position": 5,
                        "role": "immutable_context",
                        "per_layer": {"0": {"key_sha256": "key-digest"}},
                    }
                ]
            }
        },
        "native_readout": {"raw_margin": -0.5, "decoded_output": "x"},
        "receipt_links": {"interventions": ["cas://sha256/act"]},
        "status": {"causal_verdict": "same-parent-change"},
    }
    bundle = normalize_evidence([payload])
    row = next(event for event in bundle["events"] if event["event_kind"] == "memory-old-row")
    assert row["token_time"] == 8
    assert row["layer_depth"] == "?"
    assert row["physical_address"]["parent"] == 4
    assert row["evidence_class"] == "observation"
    assert any(ref["value"] == "cas://sha256/act" for ref in row["refs"])
    position = next(event for event in bundle["events"] if event["event_kind"] == "position-binding")
    assert position["token_time"] == 7
    assert position["node_role"] == "appended_question"


def test_html_escapes_embedded_json_and_cli_writes_standalone_file(tmp_path):
    report = _measurement()
    report["spec"]["name"] = "</script><img src=x onerror=alert(1)>"
    rendered = render_html(normalize_evidence([report]), title="<viewer>")
    assert "</script><img" not in rendered
    assert "\\u003c/script\\u003e" in rendered
    assert "<title>&lt;viewer&gt;</title>" in rendered

    report_path = tmp_path / "report.json"
    output = tmp_path / "viewer.html"
    report_path.write_text(json.dumps(_measurement()))
    result = subprocess.run(
        [sys.executable, "-m", "mdb.temporal_causal_viewer", str(report_path), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        },
    )
    assert json.loads(result.stdout)["events"] == 2
    assert output.read_text().startswith("<!doctype html>")
