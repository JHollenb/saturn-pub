"""Producer-to-native-symbol joins stay sealed, scoped, and read-only."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from saturn_pub import StateAddress
from saturn_pub.symbols import SymbolBinding, SymbolTable
from saturn_pub.values import digest
from test_saturn_pub_adapter import opened

from mdb.bench_cot import CONFIG_SCHEMA, OBSERVATION_SCHEMA
from mdb.hyperion import HyperionObservation, compile_dynamic_circuit
from mdb.semantic_binding import SemanticBinding, SemanticLocation
from mdb.spectral_address import (
    CanonicalSymbolKey,
    SpectralFrame,
    SpectralInstanceKey,
    SpectralLogicalAddress,
    SpectralRoleTransformSpec,
)
from mdb.symbol_lifecycle import ClockWitness, SymbolUse
from mdb.symbol_producers import (
    ISA_SOURCE_CANDIDATE_SCHEMA,
    SYMBOL_SOURCE_CANDIDATE_SCHEMA,
    SymbolProducerError,
    join_symbol_producers,
    seal_isa_source_candidate,
)
from mdb.symbol_sources import make_vocabulary, nominate_scores
from mdb.symbol_trace import trace_symbol, verify_symbol_trace
from mdb.vm_cache import VMCacheExecutionBundle


def _native_binding(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    source = adapter.checkpoint("producer-source")["cut_id"]
    binding = SymbolBinding.bind(
        adapter.session,
        "counter.value",
        (StateAddress("value"),),
        context="producer-context",
        context_slots=(),
        evidence=(f"cas://sha256/{source}",),
        claim="Supplied native counter location; producer labels remain descriptive.",
    )
    adapter.symbols = SymbolTable((binding,))
    adapter.symbol_context = binding.context
    adapter._persist()
    return project, adapter, owner, debug, binding, source


def _hyperion_report():
    rows = [
        HyperionObservation(
            "discovery-a", "cell", "a", "discovery", "candidate", ("counter.value",),
            ((0, 1.0),), 1.0, "counter-native", "fixture://a",
        ),
        HyperionObservation(
            "discovery-b", "cell", "b", "discovery", "candidate", ("counter.value",),
            ((1, 1.0),), 0.5, "counter-native", "fixture://b",
        ),
        HyperionObservation(
            "control", "cell", "noop", "control", "noop", ("counter.value",),
            (), 0.0, "counter-native", "fixture://noop",
        ),
    ]
    return compile_dynamic_circuit(
        rows,
        symbol_id="counter.native",
        logical_symbol="counter.value",
        model_id="public-counter-v1",
        consumer_id="counter-native",
        time_axis="transition",
        address_space="native.counter",
        residual_definition="declared native counter response",
        source_artifacts=(),
        context={"binding_context": "producer-context"},
    )


def _cot_artifacts(source, identity):
    activation = "a" * 64
    row = {
        "schema": OBSERVATION_SCHEMA,
        "config_ref": "config-ref",
        "observation_ref": "read-ref",
        "source_cut": source,
        "lens_ref": "pin-ref",
        "session_id": "experiment",
        "branch": "main",
        "model_identity": identity,
        "token_position": 1,
        "completed_layer_boundary": 1,
        "activation_sha256": activation,
        "candidates": [],
        "interpretation": "decoded-candidates",
        "causal_dependency_tested": False,
        "native_mutation": False,
        "semantic_cot_qualified": False,
    }
    config = {
        "schema": CONFIG_SCHEMA,
        "model_identity": identity,
        "session_id": "experiment",
        "enabled": True,
        "lens_refs": ["pin-ref"],
        "boundaries": {"1": "pin-ref"},
        "options": {"top_k": 1, "token_subset": None, "logical_address": None},
    }
    pin = {
        "schema": "mdb-bench-jacobian-pin-v1",
        "model_identity": identity,
        "lens_manifest": {"coordinate_map": {"completed_layer_boundary": 1}},
    }
    read = {
        "schema": "mdb-bench-jacobian-read-v1",
        "source": {
            "cut_ref": source,
            "token_position": 1,
            "completed_layer_boundary": 1,
            "activation": {"sha256": activation},
        },
        "pin_ref": "pin-ref",
    }
    return row, {
        "config-ref": config,
        "read-ref": read,
        "pin-ref": pin,
        source: {"execution": {"hidden": {"sha256": activation}}},
    }


def test_hyperion_report_is_joined_without_semantic_promotion_or_payload_load(tmp_path, monkeypatch):
    project, _, owner, _, binding, source = _native_binding(tmp_path)
    monkeypatch.setattr(project.store, "load", lambda *args, **kwargs: pytest.fail("payload load"))
    report = _hyperion_report()
    saved = join_symbol_producers(
        project,
        "experiment",
        binding,
        {"producer": report, "source_cut": source, "binding_context": binding.context},
    )
    record = saved["lifecycle"]["definitions"][0]
    joined = record["metadata"]["producer_join"]
    assert joined["producer_schema"] == report["schema"]
    assert joined["raw_record"] == report
    assert record["validity_status"] == "unknown"
    assert record["authority"]["causal_effect"] == "not-established"
    assert saved["lifecycle"]["metadata"]["semantic_equivalence"] == "not-established"
    assert saved["joins"]["native_payload_reads"] == 0
    owner.close()


def test_semantic_binding_status_is_retained_as_provenance_only(tmp_path):
    project, _, owner, _, binding, source = _native_binding(tmp_path)
    semantic = SemanticBinding(
        symbol=binding.symbol,
        semantic_type="object.attribute",
        manifest_fingerprint="a" * 64,
        context={
            "model_id": "public-counter-v1",
            "model_fingerprint": "c" * 64,
            "conditioner_id": "counter",
            "conditioner_family": "counter",
            "tokenizer_fingerprint": "d" * 64,
            "input_fingerprint": "e" * 64,
            "state_schema_id": "test-counter-state-v1",
            "seed": 0,
        },
        locations=(
            SemanticLocation(
                location_id="counter.value.0",
                kind="custom",
                boundary="transition",
                indices=(0,),
                carrier="native",
                support={"address": "value"},
                resolver="fixture-resolver",
                native_consumer="counter-native",
                confidence=0.9,
            ),
        ),
        confidence=0.9,
        status="validated",
        evidence_refs=("manifest://fixture",),
    )
    saved = join_symbol_producers(
        project, "experiment", binding,
        {"producer": semantic.to_dict(), "source_cut": source},
    )
    record = saved["lifecycle"]["definitions"][0]
    assert record["metadata"]["producer_join"]["producer_status"] == "validated"
    assert record["validity_status"] == "unknown"
    assert saved["lifecycle"]["statuses"]["terminal"] == "not-assessed"
    owner.close()


def test_native_observation_report_is_artifact_checked_before_join(tmp_path):
    project, _, owner, _, binding, source = _native_binding(tmp_path)
    observation = {
        "schema": "saturn-bench-observation-v1",
        "session_id": "experiment",
        "model_identity": "public-counter-v1",
        "execution": binding.execution,
        "parent_cut": source,
        "site_cut": source,
        "baseline_cut": source,
        "act": {"cut_id": source, "operation": "noop"},
        "arm": {"name": "noop", "mode": "scale", "value": 1.0},
        "site": 0,
        "target_effect": 0.0,
        "max_abs_logit_delta": 0.0,
        "mechanics_status": "valid",
        "terminal_status": "not-assessed",
    }
    saved = join_symbol_producers(
        project,
        "experiment",
        binding,
        {"producer": observation, "source_cut": source, "producer_ref": "native-ref"},
        producer_artifacts={"native-ref": observation},
    )
    joined = saved["lifecycle"]["definitions"][0]["metadata"]["producer_join"]
    assert joined["verification"]["verification"] == "artifact-verified"
    assert joined["raw_record"] == observation

    tampered = copy.deepcopy(observation)
    tampered["target_effect"] = 1.0
    with pytest.raises(SymbolProducerError, match="artifact bytes differ"):
        join_symbol_producers(
            project,
            "experiment",
            binding,
            {"producer": tampered, "source_cut": source, "producer_ref": "native-ref"},
            producer_artifacts={"native-ref": observation},
        )
    owner.close()


def test_cot_observation_uses_real_verifier_and_keeps_false_flags(tmp_path):
    project, adapter, owner, _, binding, source = _native_binding(tmp_path)
    identity = adapter.session.adapter.model_identity
    row, artifacts = _cot_artifacts(source, identity)
    saved = join_symbol_producers(
        project,
        "experiment",
        binding,
        {"producer": row, "source_cut": source},
        producer_artifacts=artifacts,
        producer_manifest={"identity": identity, "session_id": "experiment"},
    )
    definition = saved["lifecycle"]["definitions"][0]
    assert definition["cot_refs"][0]["semantic_cot_qualified"] is False
    assert definition["metadata"]["producer_join"]["verification"]["verification"] == "artifact-verified"
    tampered = copy.deepcopy(row)
    tampered["semantic_cot_qualified"] = True
    with pytest.raises(SymbolProducerError, match="cannot be promoted"):
        join_symbol_producers(
            project, "experiment", binding,
            {"producer": tampered, "source_cut": source},
            producer_artifacts=artifacts,
            producer_manifest={"identity": identity, "session_id": "experiment"},
        )
    owner.close()


def test_isa_candidate_and_source_candidate_require_selected_exact_binding(tmp_path):
    project, adapter, owner, _, binding, source = _native_binding(tmp_path)
    isa = seal_isa_source_candidate({
        "candidate_id": "isa-counter-0",
        "source_cut": source,
        "source_kind": "isa",
        "selected_symbol": binding.symbol,
        "status": "candidate",
        "proposals": [{"label": binding.symbol, "score": 0.5}],
    })
    saved = join_symbol_producers(project, "experiment", binding, isa)
    assert saved["lifecycle"]["definitions"][0]["metadata"]["producer_join"]["producer_schema"] == ISA_SOURCE_CANDIDATE_SCHEMA

    vocabulary = make_vocabulary(
        ["value", "other"], namespace="counter", evidence_refs=["fixture://isa-vocabulary"]
    )
    source_candidate = nominate_scores(
        [0.5, 0.0],
        vocabulary,
        model_identity="public-counter-v1",
        execution=binding.execution,
        source_cut=source,
        context=binding.context,
        evidence_refs=["fixture://isa"],
        producer_id="fixture-score-reader",
        top_k=1,
    )
    assert source_candidate["schema"] == SYMBOL_SOURCE_CANDIDATE_SCHEMA
    joined = join_symbol_producers(project, "experiment", binding, source_candidate)
    assert joined["lifecycle"]["metadata"]["producer_records"][0]["schema"] == SYMBOL_SOURCE_CANDIDATE_SCHEMA
    assert joined["lifecycle"]["statuses"]["instrument"] == "valid"
    assert (
        joined["lifecycle"]["definitions"][0]["metadata"]["producer_join"]["verification"][
            "source_trust"
        ]
        == "producer-supplied source; native witness verification is separate"
    )
    wrong = copy.deepcopy(source_candidate)
    wrong["selected_symbol"] = "other.symbol"
    wrong["fingerprint"] = digest({key: value for key, value in wrong.items() if key != "fingerprint"})
    with pytest.raises(SymbolProducerError, match="recomputation"):
        join_symbol_producers(project, "experiment", binding, wrong)
    wrong_context = copy.deepcopy(source_candidate)
    wrong_context["context"] = "other-context"
    wrong_context["fingerprint"] = digest(
        {key: value for key, value in wrong_context.items() if key != "fingerprint"}
    )
    with pytest.raises(SymbolProducerError, match="context"):
        join_symbol_producers(project, "experiment", binding, wrong_context)
    owner.close()


def test_source_candidate_resigning_cannot_change_winner(tmp_path):
    project, adapter, owner, _, binding, source = _native_binding(tmp_path)
    vocabulary = make_vocabulary(
        ["value", "other"], namespace="counter", evidence_refs=["fixture://isa-vocabulary"]
    )
    candidate = nominate_scores(
        [0.5, 0.0],
        vocabulary,
        model_identity="public-counter-v1",
        execution=binding.execution,
        source_cut=source,
        context=binding.context,
        evidence_refs=["fixture://isa"],
        producer_id="fixture-score-reader",
        top_k=1,
    )
    wrong_binding = SymbolBinding.bind(
        adapter.session,
        "counter.other",
        (StateAddress("value"),),
        context=binding.context,
        context_slots=(),
        evidence=(f"cas://sha256/{source}",),
        claim="Separate supplied native location for hostile winner test.",
    )
    adapter.symbols = SymbolTable((binding, wrong_binding))
    adapter._persist()
    resigned = copy.deepcopy(candidate)
    resigned["selected_label"] = "other"
    resigned["selected_symbol"] = wrong_binding.symbol
    resigned["fingerprint"] = digest(
        {key: value for key, value in resigned.items() if key != "fingerprint"}
    )
    with pytest.raises(SymbolProducerError, match="recomputation"):
        join_symbol_producers(project, "experiment", wrong_binding, resigned)
    owner.close()


def test_typed_clock_spectral_and_vm_envelope_passes_native_join_checks(tmp_path):
    project, _, owner, _, binding, source = _native_binding(tmp_path)
    native_cut = project.cut_descriptor("experiment", source)
    clock = ClockWitness(
        clock_id="producer-execution-clock",
        kind="execution-point",
        domain="counter.native",
        reading=native_cut["execution_point"],
        provenance=(source,),
        status="observed",
        metadata={
            "source_cut": source,
            "model_execution": binding.execution,
            "consumer": native_cut["surface"]["consumer"],
            "reset": "native-boundary",
        },
    )
    spectral = SpectralLogicalAddress(
        CanonicalSymbolKey(
            binding.symbol, (0.1, 0.2), "fixture", "property", "producer-book"
        ),
        SpectralInstanceKey.deterministic("producer-scene", "counter", 2),
        SpectralRoleTransformSpec(
            "carrier", (0.0, 0.0), 1, 0, "property", "property"
        ),
        SpectralFrame("producer-frame"),
    )
    spectral_payload = dataclasses.asdict(spectral)
    spectral_payload["schema"] = "saturn-spectral-address-abi-v1"
    spectral_payload["fingerprint"] = spectral.fingerprint
    bundle = VMCacheExecutionBundle.create(
        run_id="producer-vm",
        parent_identity={"session": "experiment"},
        graph_fingerprint="a" * 64,
        model_identity={"id": binding.model_identity},
        input_identity={"source_cut": source, "context": binding.context},
        abi_identity={"execution": binding.execution},
        plane_references={
            plane: f"cas://sha256/{'b' * 64}"
            for plane in ("source", "address", "operator", "renderer")
        },
    )
    report = _hyperion_report()
    saved = join_symbol_producers(
        project,
        "experiment",
        binding,
        {
            "producer": report,
            "source_cut": source,
            "binding_context": binding.context,
            "clock": clock.to_dict(),
            "spectral_address": spectral_payload,
            "vm_bundle": bundle.to_dict(),
        },
    )
    definition = saved["lifecycle"]["definitions"][0]
    assert definition["clock"]["fingerprint"] == clock.fingerprint
    assert definition["spectral_address"]["fingerprint"] == spectral.fingerprint
    assert definition["vm_bundle"]["fingerprint"] == bundle.fingerprint

    wrong_clock = copy.deepcopy(clock.to_dict())
    wrong_clock["metadata"]["source_cut"] = "f" * 64
    wrong_clock["fingerprint"] = digest(
        {key: value for key, value in wrong_clock.items() if key != "fingerprint"}
    )
    with pytest.raises(SymbolProducerError, match="another source cut"):
        join_symbol_producers(
            project,
            "experiment",
            binding,
            {
                "producer": report,
                "source_cut": source,
                "clock": wrong_clock,
            },
        )
    wrong_bundle = VMCacheExecutionBundle.create(
        run_id="producer-vm-wrong-source",
        parent_identity={"session": "experiment"},
        graph_fingerprint="a" * 64,
        model_identity={"id": binding.model_identity},
        input_identity={"source_cut": "f" * 64, "context": binding.context},
        abi_identity={"execution": binding.execution},
        plane_references={
            plane: f"cas://sha256/{'c' * 64}"
            for plane in ("source", "address", "operator", "renderer")
        },
    )
    with pytest.raises(SymbolProducerError, match="VM bundle differs"):
        join_symbol_producers(
            project,
            "experiment",
            binding,
            {
                "producer": report,
                "source_cut": source,
                "vm_bundle": wrong_bundle.to_dict(),
            },
        )
    owner.close()


def test_bench_cut_model_context_and_tampered_producer_are_rejected(tmp_path):
    project, adapter, owner, _, binding, source = _native_binding(tmp_path)
    report = _hyperion_report()
    bad = copy.deepcopy(report)
    bad["symbol"]["logical_symbol"] = "other"
    with pytest.raises(SymbolProducerError, match="seal"):
        join_symbol_producers(project, "experiment", binding, {"producer": bad, "source_cut": source})
    with pytest.raises(SymbolProducerError, match="Bench cuts"):
        join_symbol_producers(
            project, "experiment", binding,
            {"producer": report, "source_cut": "f" * 64},
        )
    with pytest.raises(SymbolProducerError, match="model_id"):
        join_symbol_producers(
            project, "experiment", binding,
            {"producer": report, "source_cut": source, "model_identity": "wrong-model"},
        )
    with pytest.raises(SymbolProducerError, match="already registered"):
        other = SymbolBinding.bind(
            adapter.session, "counter.value", (StateAddress("value"),),
            context=binding.context, context_slots=(), evidence=("fixture://other",), claim="other",
        )
        join_symbol_producers(project, "experiment", other, {"producer": report, "source_cut": source})
    owner.close()


def test_supplied_native_use_keeps_receipt_closure_and_rejects_wrong_receipt(tmp_path):
    project, _, owner, debug, binding, source = _native_binding(tmp_path)
    debug.step()
    result = debug.checkpoint("producer-result")["cut_id"]
    receipt = next(
        item for item in project.receipts("experiment")
        if item.to_dict().get("result") == result and item.to_dict().get("operation") == "continue"
    )
    use = SymbolUse(
        "producer-use", binding.to_dict()["fingerprint"], "experiment", "counter-native",
        source_cut=source, parent_cut=source, result_cut=result, branch="main",
        execution=binding.execution, receipt=receipt.fingerprint, locations=binding.locations,
        signed_observations=({"delta": 1},), evidence_refs=(f"cas://sha256/{receipt.fingerprint}",),
        status="observed",
    )
    saved = join_symbol_producers(project, "experiment", binding, {"producer": _hyperion_report(), "source_cut": source}, uses=(use,))
    assert saved["joins"]["uses"][0]["receipt_closure"] == "native-parent-result-matched"
    bad = copy.copy(use)
    bad = SymbolUse(
        "bad-use", binding.to_dict()["fingerprint"], "experiment", "counter-native",
        source_cut=source, parent_cut=source, result_cut=result, branch="main",
        execution=binding.execution, receipt="f" * 64, locations=binding.locations,
        signed_observations=({"delta": 1},), evidence_refs=("fixture://bad",), status="observed",
    )
    with pytest.raises(SymbolProducerError, match="not retained"):
        join_symbol_producers(project, "experiment", binding, {"producer": _hyperion_report(), "source_cut": source}, uses=(bad,))
    owner.close()


def test_symbol_producers_cli_preserves_original_record_in_trace(tmp_path, capsys):
    project, _, owner, _, binding, source = _native_binding(tmp_path)
    report = _hyperion_report()
    envelope = {"producer": report, "source_cut": source, "binding_context": binding.context}
    producer_path = tmp_path / "producer.json"
    producer_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    from mdb.symbol_producers import main

    assert main([
        str(project.root),
        "--binding", binding.to_dict()["fingerprint"],
        str(producer_path),
    ]) == 0
    cli_result = json.loads(capsys.readouterr().out)
    lifecycle_id = cli_result["lifecycle_id"]
    assert cli_result["lifecycle"]["metadata"]["producer_records"][0]["schema"] == report["schema"]

    trace = trace_symbol(project, "experiment", binding.symbol)
    assert verify_symbol_trace(trace)["passed"]
    occurrence = trace["occurrences"][0]
    assert occurrence["metadata"]["producer_join"]["raw_record"] == report
    assert occurrence["metadata"]["producer_join"]["producer_schema"] == report["schema"]
    assert lifecycle_id in occurrence["lifecycle_ids"]
    owner.close()
