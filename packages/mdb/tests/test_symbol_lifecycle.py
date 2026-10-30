"""Framework-free tests for the sealed symbol lifecycle sidecar."""

from __future__ import annotations

import copy

import pytest
from saturn_pub import StateAddress
from saturn_pub.symbols import SymbolBinding
from saturn_pub.values import digest

from mdb.spectral_address import (
    CanonicalSymbolKey,
    SpectralFrame,
    SpectralInstanceKey,
    SpectralLogicalAddress,
    SpectralRoleTransformSpec,
)
from mdb.symbol_lifecycle import (
    ClockWitness,
    SymbolLifecycle,
    SymbolLifecycleError,
    SymbolOccurrence,
    SymbolUse,
    verify_symbol_lifecycle,
)
from mdb.vm_cache import VMCacheExecutionBundle


def _binding() -> SymbolBinding:
    return SymbolBinding(
        symbol="object.color",
        model_identity="fixture-model-v1",
        execution=digest({"family": "qwen", "dtype": "fp32"}),
        context="prompt-1",
        context_slots=(("token_ids", digest({"kind": "list", "items": []})),),
        locations=(StateAddress("hidden", (0,), "carrier", "point-a", "state-v1"),),
        evidence=("receipt://binding",),
        claim="Supplied context-qualified debug location.",
        status="observed",
        confidence=0.7,
    )


def _spectral_address() -> SpectralLogicalAddress:
    return SpectralLogicalAddress(
        CanonicalSymbolKey("object.color", (0.1, 0.2), "fixture", "property", "book"),
        SpectralInstanceKey.deterministic("scene-1", "object-1", 2),
        SpectralRoleTransformSpec("carrier", (0.0, 0.0), 1, 0, "property", "property"),
        SpectralFrame("layer-1"),
    )


def _vm_bundle() -> VMCacheExecutionBundle:
    digest_value = "a" * 64
    return VMCacheExecutionBundle.create(
        run_id="run-1",
        parent_identity={"session": "session-1"},
        graph_fingerprint=digest_value,
        model_identity={"model": "fixture-model-v1"},
        input_identity={"input": "prompt-1"},
        abi_identity={"abi": "vm-v1"},
        plane_references={
            plane: f"cas://sha256/{digest_value}"
            for plane in ("source", "address", "operator", "renderer")
        },
    )


def test_clock_keeps_unknown_local_age_unknown_and_roundtrips():
    clock = ClockWitness(
        clock_id="qwen-token-7",
        kind="token-position",
        domain="qwen.decoder",
        reading=7,
        provenance=("cut://source",),
        status="unknown",
        validity_status="unknown",
        lifetime={"start": "cut://source", "end": None},
    )
    assert clock.status == "unknown"
    assert ClockWitness.from_dict(clock.to_dict()) == clock
    with pytest.raises(SymbolLifecycleError, match="validity evidence"):
        ClockWitness(
            "clock", "layer", "qwen", 1, status="observed", validity_status="supported"
        )
    with pytest.raises(SymbolLifecycleError, match="source provenance"):
        ClockWitness("clock", "layer", "qwen", 1, status="unknown")


def test_lifecycle_joins_binding_occurrence_use_and_external_refs(tmp_path):
    binding = _binding()
    clock = ClockWitness(
        "layer-1",
        "execution-point",
        "qwen.decoder",
        {"layer": 1},
        provenance=("receipt://step",),
        status="observed",
        validity_status="supported",
        validity_support=("receipt://step",),
    )
    occurrence = SymbolOccurrence(
        occurrence_id="occurrence-1",
        binding_fingerprint=binding.to_dict()["fingerprint"],
        binding=binding,
        session_id="session-1",
        source_cut="cut://source",
        result_cut="cut://result",
        branch="main",
        execution=binding.execution,
        receipt="receipt://step",
        locations=binding.locations,
        coalition=("carrier", "pooled"),
        location_sets={"definition": binding.locations},
        clock=clock,
        spectral_address=_spectral_address(),
        vm_bundle=_vm_bundle(),
        cot_refs=(
            {
                "schema": "mdb-cot-register-observation-v1",
                "source_cut": "cut://source",
                "completed_layer_boundary": 1,
                "activation_sha256": "b" * 64,
                "semantic_cot_qualified": False,
                "causal_dependency_tested": False,
            },
        ),
        definition_refs=("receipt://binding",),
        evidence_refs=("receipt://step",),
    )
    use = SymbolUse(
        use_id="use-1",
        binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1",
        consumer="native.decoder.readout",
        source_cut="cut://source",
        parent_cut="cut://source",
        result_cut="cut://result",
        branch="main",
        execution=binding.execution,
        receipt="receipt://consumer",
        locations=binding.locations,
        clock=clock,
        signed_observations=(
            {"observation": "logit_delta", "value": -0.2, "direction": "signed"},
        ),
        evidence_refs=("receipt://consumer",),
    )
    lifecycle = SymbolLifecycle.define(binding)
    lifecycle = lifecycle.add_occurrence(occurrence).add_use(use)
    lifecycle = lifecycle.invalidate(
        scope="occurrence:occurrence-1",
        reason="input context changed",
        evidence_refs=("cut://changed",),
        clock=clock,
    )

    payload = lifecycle.to_dict()
    assert payload["authority"] == {"write": "native-only", "causal_effect": "not-established"}
    assert payload["statuses"] == {
        "mechanics": "unknown", "instrument": "unknown", "trend": "unknown", "terminal": "unknown"
    }
    restored = SymbolLifecycle.from_dict(payload)
    assert restored.fingerprint == lifecycle.fingerprint
    assert verify_symbol_lifecycle(payload)["passed"] is True
    lifecycle.save(tmp_path / "lifecycle.json")
    assert SymbolLifecycle.load(tmp_path / "lifecycle.json").fingerprint == lifecycle.fingerprint


def test_lifecycle_rejects_identity_drift_and_tampering():
    binding = _binding()
    lifecycle = SymbolLifecycle.define(binding)
    wrong = SymbolOccurrence(
        occurrence_id="wrong",
        binding_fingerprint=digest("different-binding"),
        session_id="session-1",
        source_cut="cut://source",
        locations=binding.locations,
    )
    with pytest.raises(SymbolLifecycleError, match="another binding"):
        lifecycle.add_occurrence(wrong)
    payload = lifecycle.to_dict()
    tampered = copy.deepcopy(payload)
    tampered["claim_boundary"] = "changed"
    with pytest.raises(SymbolLifecycleError, match="fingerprint"):
        SymbolLifecycle.from_dict(tampered)
    with pytest.raises(SymbolLifecycleError, match="unknown occurrence"):
        lifecycle.invalidate(scope="occurrence:missing", reason="stale")


def test_nested_payloads_are_frozen_and_to_dict_is_a_deep_copy():
    binding = _binding()
    nested_metadata = {"nested": {"items": [1, {"value": "before"}]}}
    clock = ClockWitness(
        "qwen-token-7", "token-position", "qwen.decoder", {"token": [7]},
        provenance=("cut://source",), metadata=nested_metadata,
    )
    occurrence = SymbolOccurrence(
        occurrence_id="occurrence-freeze",
        binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1", source_cut="cut://source", locations=binding.locations,
        clock=clock, cot_refs=(nested_metadata,), metadata=nested_metadata,
        evidence_refs=("receipt://occurrence",),
    )
    use = SymbolUse(
        use_id="use-freeze", binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1", consumer="native.decoder", locations=binding.locations,
        receipt="receipt://use", status="observed", signed_observations=(nested_metadata,),
        metadata=nested_metadata,
    )
    lifecycle = (
        SymbolLifecycle.define(binding, metadata=nested_metadata)
        .add_occurrence(occurrence)
        .add_use(use)
    )
    before = lifecycle.to_dict()
    fingerprint = lifecycle.fingerprint

    nested_metadata["nested"]["items"].append("caller-mutation")
    exported = lifecycle.to_dict()
    exported["metadata"]["nested"]["items"].append("serialized-mutation")
    exported["definitions"][0]["metadata"]["nested"]["items"][1]["value"] = "changed"
    exported["definitions"][0]["clock"]["reading"]["token"].append(99)
    exported["definitions"][0]["cot_refs"][0]["nested"]["items"].append(100)
    exported["uses"][0]["signed_observations"][0]["nested"]["items"].append(101)
    exported["uses"][0]["metadata"]["nested"]["items"].append(102)

    assert lifecycle.fingerprint == fingerprint
    assert lifecycle.to_dict() == before


def test_metadata_budget_rejects_depth_count_string_total_and_tensor_like():
    deep = cursor = {}
    for index in range(14):
        cursor["next"] = {}
        cursor = cursor["next"]
    with pytest.raises(SymbolLifecycleError, match="nesting"):
        ClockWitness("deep", "unknown", "test", metadata=deep)
    with pytest.raises(SymbolLifecycleError, match="too many metadata items"):
        ClockWitness("wide", "unknown", "test", metadata={"items": list(range(513))})
    with pytest.raises(SymbolLifecycleError, match="too large"):
        ClockWitness("long", "unknown", "test", metadata={"text": "x" * (16 * 1024 + 1)})

    oversized = {f"item-{index}": "x" * (16 * 1024) for index in range(20)}
    with pytest.raises(SymbolLifecycleError, match="bounded metadata budget"):
        ClockWitness("large", "unknown", "test", metadata=oversized)

    class TensorLike:
        shape = (1,)

    with pytest.raises(SymbolLifecycleError, match="tensor-like"):
        ClockWitness("tensor", "unknown", "test", metadata={"value": TensorLike()})


def test_occurrence_and_use_locations_must_be_root_binding_subset():
    binding = _binding()
    outside = StateAddress("outside", (0,), "carrier", binding.locations[0].execution_point,
                           binding.locations[0].state_schema)
    occurrence = SymbolOccurrence(
        occurrence_id="outside-occurrence", binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1", source_cut="cut://source", locations=(outside,),
        receipt="receipt://occurrence", status="observed",
    )
    lifecycle = SymbolLifecycle.define(binding)
    with pytest.raises(SymbolLifecycleError, match="root binding locations"):
        lifecycle.add_occurrence(occurrence)

    use = SymbolUse(
        use_id="outside-use", binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1", consumer="native.decoder", locations=(outside,),
        receipt="receipt://use", status="observed",
    )
    with pytest.raises(SymbolLifecycleError, match="root binding locations"):
        lifecycle.add_use(use)


def test_observed_and_validated_rows_need_evidence_and_defaults_are_unknown():
    binding = _binding()
    occurrence_kwargs = dict(
        occurrence_id="evidence-occurrence", binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1", source_cut="cut://source", locations=binding.locations,
    )
    assert SymbolOccurrence(**occurrence_kwargs).status == "unknown"
    with pytest.raises(SymbolLifecycleError, match="receipt or evidence"):
        SymbolOccurrence(**occurrence_kwargs, status="observed")
    with pytest.raises(SymbolLifecycleError, match="receipt or evidence"):
        SymbolUse(
            use_id="evidence-use", binding_fingerprint=binding.to_dict()["fingerprint"],
            session_id="session-1", consumer="native.decoder", status="validated",
        )
    lifecycle = SymbolLifecycle.define(
        binding,
        mechanics_status="mixed",
        instrument_status="directional",
        trend_status="observations-only",
        terminal_status="not-established",
    )
    assert lifecycle.to_dict()["statuses"] == {
        "mechanics": "mixed", "instrument": "directional",
        "trend": "observations-only", "terminal": "not-established",
    }


def test_bare_refs_are_explicitly_unjoined_and_unsealed_bundles_rejected():
    binding = _binding()
    occurrence_kwargs = dict(
        occurrence_id="ref-occurrence", binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1", source_cut="cut://source", locations=binding.locations,
    )
    occurrence = SymbolOccurrence(
        **occurrence_kwargs, spectral_address="ar://spectral/address-1",
        vm_bundle="cas://vm-cache-bundle/sha256/" + "b" * 64,
    )
    payload = occurrence.to_dict()
    assert payload["spectral_address"] == {
        "reference": "ar://spectral/address-1", "joined": False, "verification": "unverified",
    }
    assert payload["vm_bundle"] == {
        "reference": "cas://vm-cache-bundle/sha256/" + "b" * 64,
        "joined": False, "verification": "unverified",
    }

    unsealed = _vm_bundle().to_dict()
    unsealed.pop("fingerprint")
    unsealed.pop("content_sha256")
    with pytest.raises(SymbolLifecycleError, match="original seal"):
        SymbolOccurrence(**occurrence_kwargs, vm_bundle=unsealed)
    extra = _vm_bundle().to_dict()
    extra["untrusted"] = "ignored-by-vm-parser"
    with pytest.raises(SymbolLifecycleError, match="unsupported fields"):
        SymbolOccurrence(**occurrence_kwargs, vm_bundle=extra)


def test_typed_spectral_seal_and_clock_contracts_are_checked():
    binding = _binding()
    base = dict(
        occurrence_id="spectral-occurrence", binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="session-1", source_cut="cut://source", locations=binding.locations,
    )
    spectral = SymbolOccurrence(**base, spectral_address=_spectral_address()).to_dict()
    tampered_spectral = copy.deepcopy(spectral["spectral_address"])
    tampered_spectral["canonical_key"]["symbol_id"] = "other.symbol"
    with pytest.raises(SymbolLifecycleError, match="spectral address fingerprint"):
        SymbolOccurrence(**base, spectral_address=tampered_spectral)

    with pytest.raises(SymbolLifecycleError, match="explicit 'unknown'"):
        ClockWitness("bad-kind", "mamba-age", "mamba", 1, provenance=("cut",))
    with pytest.raises(SymbolLifecycleError, match="requires source/model/consumer/reset"):
        ClockWitness(
            "local", "local-dt", "mamba", [0.1, 0.2], provenance=("cut",),
            status="observed",
        )
    supplied = ClockWitness(
        "local-supplied", "local-dt", "mamba", [0.1, 0.2], provenance=("cut",),
        status="supplied",
    )
    assert supplied.status == "supplied"
    assert supplied.validity_status == "unknown"
    local = ClockWitness(
        "local", "local-dt", "mamba", [0.1, 0.2], provenance=("cut",),
        status="observed", metadata={
            "source_cut": "cut://source", "model_execution": "execution-v1",
            "consumer": "mamba.reader", "reset": "reset-v1",
        },
    )
    assert local.status == "observed"
    with pytest.raises(SymbolLifecycleError, match="cannot be an observed symbol clock"):
        ClockWitness("host", "wall-clock", "host", 1, provenance=("receipt",), status="observed")
