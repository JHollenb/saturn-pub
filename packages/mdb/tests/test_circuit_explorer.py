from __future__ import annotations

import json

import pytest
from saturn_pub import Receipt

from mdb.circuit_explorer import (
    CircuitCatalog,
    CircuitCatalogError,
    TraceCircuitAdapter,
)
from mdb.forward_symbols import FORWARD_SYMBOLS_SCHEMA, ForwardSymbolTable, seal_trace


def _trace():
    body = {
        "schema": FORWARD_SYMBOLS_SCHEMA,
        "model_identity": {"model_id": "toy", "weights_sha256": "a" * 64},
        "execution": {"backend": "cpu", "dtype": "float32"},
        "input_identity": {"input_sha256": "b" * 64},
        "values": {
            "v000": {
                "shape": [2], "dtype": "torch.float32", "sha256": "1" * 64,
                "origin": "input", "producer": None, "preview": [1.0, 2.0],
            },
            "v001": {
                "shape": [2], "dtype": "torch.float32", "sha256": "2" * 64,
                "origin": "parameter", "producer": None, "preview": [0.5, -0.5],
            },
            "v002": {
                "shape": [2], "dtype": "torch.float32", "sha256": "3" * 64,
                "origin": "operation-output", "producer": 0, "preview": [1.5, 1.5],
            },
            "v003": {
                "shape": [], "dtype": "torch.int64", "sha256": "4" * 64,
                "origin": "operation-output", "producer": 1, "preview": [0],
            },
        },
        "events": [
            {
                "sequence": 0, "operator": "aten.add.Tensor", "module_path": "projection",
                "inputs": ["v000", "v001"], "outputs": ["v002"],
                "arguments": {"self": {"$value": "v000"}, "other": {"$value": "v001"}},
                "result": {"$value": "v002"},
                "clock": {"forward_call": 0, "operation": 0},
            },
            {
                "sequence": 1, "operator": "aten.argmax.default", "module_path": "readout",
                "inputs": ["v002"], "outputs": ["v003"],
                "arguments": {"self": {"$value": "v002"}, "dim": 0},
                "result": {"$value": "v003"},
                "clock": {"forward_call": 0, "operation": 1},
            },
        ],
        "outputs": {"class_id": {"$value": "v003"}},
        "capture_status": "complete",
        "semantic_labels_supplied": False,
        "metadata": {"capture": "fixture"},
    }
    return seal_trace(body)


def _catalog(tmp_path):
    trace = _trace()
    table = ForwardSymbolTable(trace)
    adapter = TraceCircuitAdapter.from_table(table)
    record = adapter.to_record(
        circuit_id="circuit://toy/readout",
        route_candidates=[{
            "route_id": "readout",
            "source_values": ["v000", "v001"],
            "target_values": ["v003"],
            "roles": {"source": "hidden", "consumer": "class-readout"},
        }],
        evidence=[{"kind": "native-trace", "note": "fixture"}],
        supported_interventions=[{
            "intervention_id": "replace-hidden",
            "value_ids": ["v002"],
            "dose": 1.0,
        }],
        predictions=[{
            "prediction_id": "pred-0",
            "route_id": "readout",
            "output_digest": "c" * 64,
            "metadata": {"source": "fixture predictor"},
        }],
        replay_handle={"backend": "fixture"},
    )
    catalog = CircuitCatalog(tmp_path / "circuits.json")
    catalog.register(record)
    catalog.attach_trace(trace)
    return catalog, trace, record


def test_trace_adapter_separates_native_structure_from_supplied_roles(tmp_path):
    catalog, trace, record = _catalog(tmp_path)
    body = record.to_dict()
    assert body["trace"]["fingerprint"] == trace["fingerprint"]
    assert body["statuses"]["structure"] == "observed-native-trace"
    assert body["statuses"]["roles"] == "supplied-unverified"
    assert body["routes"][0]["event_sequences"] == [0, 1]
    assert {edge["source"] for edge in body["source_edges"]} == {"v000", "v001", "v002"}
    assert catalog.list()[0]["route_count"] == 1


def test_plan_prediction_and_measured_outcome_persist(tmp_path):
    catalog, _, _ = _catalog(tmp_path)
    plan = catalog.plan(
        "circuit://toy/readout", route_id="readout", intervention="replace-hidden"
    )
    receipt = Receipt.make(operation="fixture.branch", status="observed")
    outcome = catalog.record_outcome(plan, receipt=receipt, output={"class_id": 0})
    assert outcome.to_dict()["receipt_digest"] == receipt.fingerprint
    assert outcome.to_dict()["output_digest"]
    assert outcome.to_dict()["prediction_status"] == "diverged"

    reopened = CircuitCatalog.load(tmp_path / "circuits.json")
    assert reopened.show("circuit://toy/readout")["outcomes"][0]["outcome_id"] == outcome.outcome_id


def test_prediction_can_bind_to_an_explicit_output_projection(tmp_path):
    trace = _trace()
    record = TraceCircuitAdapter.from_trace(trace).to_record(
        circuit_id="circuit://toy/scoped",
        route_candidates=[{"route_id": "readout", "target_values": ["v003"]}],
        predictions=[{
            "prediction_id": "pred-scoped",
            "route_id": "readout",
            "output_keys": ["class_id"],
            "output": {"class_id": 0},
        }],
    )
    catalog = CircuitCatalog(tmp_path / "scoped.json")
    catalog.register(record)
    plan = catalog.plan("circuit://toy/scoped", route_id="readout")
    outcome = catalog.record_outcome(
        plan,
        receipt=Receipt.make(operation="fixture.scoped", status="observed"),
        output={"class_id": 0, "unpredicted_auxiliary": 99},
    )
    assert outcome.to_dict()["prediction_status"] == "matched"
    assert outcome.to_dict()["prediction_observed_digest"] == record.to_dict()["predictions"][0]["output_digest"]


def test_stale_context_and_plan_mutation_are_refused(tmp_path):
    catalog, _, record = _catalog(tmp_path)
    stale = {"trace_fingerprint": "d" * 64}
    with pytest.raises(CircuitCatalogError, match="stale context"):
        catalog.plan("circuit://toy/readout", route_id="readout", context=stale)
    plan = catalog.plan("circuit://toy/readout", route_id="readout")
    receipt = Receipt.make(operation="fixture.branch", status="observed")
    catalog.record_outcome(plan, receipt=receipt, output={"class_id": 0})
    with pytest.raises(CircuitCatalogError, match="cannot change a plan"):
        catalog.plan(
            "circuit://toy/readout", route_id="readout", branch_id="different",
            plan_id=plan.plan_id,
        )
    with pytest.raises(CircuitCatalogError, match="cannot change a plan"):
        catalog.record_outcome(plan, receipt=Receipt.make(operation="other"), output={"class_id": 1})
    assert record.fingerprint == catalog.show("circuit://toy/readout")["fingerprint"]


def test_registered_backend_dispatches_and_binds_actual_result(tmp_path):
    catalog, trace, _ = _catalog(tmp_path)
    seen = {}

    def backend(*, plan, record, context, trace, **kwargs):
        seen.update({"plan": plan["plan_id"], "trace": trace["fingerprint"], "context": context})
        return {
            "receipt": Receipt.make(operation="fixture.backend", plan=plan["plan_id"]),
            "output": {"class_id": 0},
        }

    def verify_fixture(*, plan, record, context, trace, result, kwargs):
        del record, context, trace, kwargs
        assert result["receipt"].to_dict()["operation"] == "fixture.backend"
        assert result["receipt"].to_dict()["plan"] == plan["plan_id"]
        assert result["output"] == {"class_id": 0}

    catalog.register_backend("fixture", backend, verifier=verify_fixture)
    plan = catalog.plan("circuit://toy/readout", route_id="readout")
    outcome = catalog.execute(plan, trace=trace)
    assert seen["plan"] == plan.plan_id
    assert seen["trace"] == trace["fingerprint"]
    assert outcome.to_dict()["status"] == "measured"
    assert outcome.to_dict()["receipt"]["fingerprint"] == outcome.to_dict()["receipt_digest"]


def test_catalog_cli_list_and_show(tmp_path, capsys):
    catalog, _, _ = _catalog(tmp_path)
    catalog.save()
    from mdb.circuit_explorer import main

    assert main(["list", "--catalog", str(tmp_path / "circuits.json")]) == 0
    assert "circuit://toy/readout" in capsys.readouterr().out
    assert main(["show", "circuit://toy/readout", "--catalog", str(tmp_path / "circuits.json")]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["routes"][0]["route_id"] == "readout"
