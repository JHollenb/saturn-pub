"""Native source support, reverse lookup, and durable evidence remain distinct."""

import json
from dataclasses import replace

import pytest
from saturn_pub import StateAddress
from saturn_pub.symbols import SymbolBinding, SymbolTable
from test_saturn_pub_adapter import opened

from mdb.cli import main
from mdb.project import Project
from mdb.spectral_address import (
    CanonicalSymbolKey,
    SpectralFrame,
    SpectralInstanceKey,
    SpectralLogicalAddress,
    SpectralQuery,
    SpectralRoleTransformSpec,
)
from mdb.symbol_lifecycle import ClockWitness, SymbolLifecycle, SymbolOccurrence, SymbolUse
from mdb.vm_cache import VMCacheExecutionBundle


def recorded(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    source = adapter.checkpoint()["cut_id"]
    binding = SymbolBinding.bind(adapter.session, "counter.value", (StateAddress("value"),),
        context="counter", context_slots=(), evidence=("cas://sha256/" + source,),
        claim="Supplied counter field for the native increment consumer.")
    adapter.symbols = SymbolTable((binding,))
    adapter.symbol_context = "counter"
    source = adapter.checkpoint()["cut_id"]
    debug.step()
    result = adapter.checkpoint()["cut_id"]
    receipt = next(r for r in project.receipts("experiment")
                   if r.to_dict().get("result") == result and r.to_dict()["operation"] == "continue")
    clock = ClockWitness("counter-native-cursor", "execution-point", "counter/main",
        project.cut_descriptor("experiment", source)["execution_point"],
        provenance=("cas://sha256/" + source,), status="observed",
        metadata={"source_cut": source})
    occurrence = SymbolOccurrence.for_binding(binding, occurrence_id="definition-1",
        session_id="experiment", source_cut=source, locations=binding.locations,
        branch="main", execution=binding.execution, clock=clock,
        evidence_refs=("cas://sha256/" + source,))
    use = SymbolUse("use-1", binding.to_dict()["fingerprint"], "experiment", "counter-native",
        source_cut=source, parent_cut=source, result_cut=result, branch="main",
        receipt=receipt.fingerprint, execution=binding.execution,
        evidence_refs=("cas://sha256/" + receipt.fingerprint,),
        signed_observations=({"observable": "value-change", "value": 1, "n": 1},))
    lifecycle = SymbolLifecycle.define(binding, definitions=(occurrence,), uses=(use,))
    return project, adapter, owner, lifecycle, source, result


def test_native_join_reverse_lookup_and_reopen_do_not_hydrate_or_select(tmp_path, monkeypatch):
    project, _, owner, lifecycle, source, result = recorded(tmp_path)
    monkeypatch.setattr(project.store, "load", lambda *a, **k: pytest.fail("tensor hydration"))
    saved = project.save_symbol_lifecycle("experiment", lifecycle)
    assert saved["lifecycle_id"] == lifecycle.fingerprint
    assert saved["joins"]["native_payload_reads"] == 0
    assert saved["joins"]["definitions"][0]["slice_id"]
    assert saved["joins"]["uses"][0]["support"] == "recorded-consumer-association"
    found = project.symbols_at("experiment", "value", cut_id=source, selector=())
    assert [r["symbol"] for r in found["matches"]] == ["counter.value"]
    assert found["matches"][0]["clock"]["domain"] == "counter/main"
    assert not project.symbols_at("experiment", "tick", cut_id=source)["matches"]
    assert not project.symbols_at("experiment", "value", cut_id=result)["matches"]
    assert project.session("experiment")["selected"] == result
    assert project.symbol_lifecycles("experiment", branch="sibling") == []
    assert len(project.symbol_lifecycles("experiment", symbol="counter.value")) == 1
    owner.close()
    reopened = Project(project.root)
    assert reopened.symbol_lifecycle("experiment", lifecycle.fingerprint) == saved
    assert reopened.save_symbol_lifecycle("experiment", lifecycle) == saved


def test_use_execution_clock_points_to_retained_result_cut(tmp_path):
    project, _, owner, lifecycle, source, result = recorded(tmp_path)
    try:
        result_point = project.cut_descriptor("experiment", result)["execution_point"]
        result_ref = "cas://sha256/" + result
        result_clock = ClockWitness(
            "counter-native-use-result",
            "execution-point",
            "counter/main",
            result_point,
            provenance=(result_ref,),
            validity_status="supported",
            validity_support=(result_ref,),
            lifetime={"kind": "bounded-suffix", "reset": "common-parent"},
            status="observed",
            metadata={
                "source_cut": result,
                "model_execution": lifecycle.binding.execution,
                "consumer": lifecycle.uses[0].consumer,
                "reset": "common-parent",
            },
        )
        corrected_use = replace(lifecycle.uses[0], clock=result_clock)
        corrected = replace(lifecycle, uses=(corrected_use,))
        saved = project.save_symbol_lifecycle("experiment", corrected)
        assert corrected_use.clock.metadata["source_cut"] == result
        assert corrected_use.clock.reading == result_point
        assert saved["joins"]["uses"][0]["support"] == "recorded-consumer-association"

        # The historical driver bug used source_cut in the clock metadata while
        # recording result_point.  The native join must reject that mismatch.
        bad_clock = replace(result_clock, metadata={**result_clock.metadata, "source_cut": source})
        with pytest.raises(ValueError, match="clock reading"):
            project.save_symbol_lifecycle(
                "experiment", replace(lifecycle, uses=(replace(corrected_use, clock=bad_clock),))
            )
    finally:
        owner.close()


def test_clock_host_time_and_historical_invalidation_do_not_rewrite_source(tmp_path, monkeypatch):
    project, _, owner, lifecycle, source, _ = recorded(tmp_path)
    original = project.save_symbol_lifecycle("experiment", lifecycle)
    invalidated = lifecycle.invalidate(scope="occurrence:definition-1", reason="supplied overwrite policy",
        evidence_refs=("policy://counter-overwrite",),
        clock=replace(lifecycle.definitions[0].clock, domain="counter/other-bank",
                      reading={"local_age": 4096}, status="supplied"))
    later = project.save_symbol_lifecycle("experiment", invalidated)
    assert later["lifecycle_id"] != original["lifecycle_id"]
    monkeypatch.setattr("time.time", lambda: 10 ** 15)
    assert project.symbol_lifecycle("experiment", lifecycle.fingerprint) == original
    # An invalidation observation is retained; it neither deletes old evidence nor
    # turns another bank's local age into an automatic expiry policy.
    matches = project.symbols_at("experiment", "value", cut_id=source)["matches"]
    assert len(matches) == 2
    assert matches[0]["clock"]["domain"] == "counter/main"
    assert matches[1]["invalidations"][0]["clock"]["domain"] == "counter/other-bank"
    owner.close()


def test_source_context_and_session_cannot_be_forged(tmp_path):
    project, adapter, owner, lifecycle, source, result = recorded(tmp_path)
    wrong_session = replace(lifecycle.definitions[0], session_id="other")
    with pytest.raises(ValueError, match="session"):
        project.save_symbol_lifecycle("experiment", replace(lifecycle, definitions=(wrong_session,)))
    wrong_clock = replace(lifecycle.definitions[0].clock, metadata={"source_cut": result})
    with pytest.raises(ValueError, match="clock witness"):
        project.save_symbol_lifecycle("experiment", replace(lifecycle,
            definitions=(replace(lifecycle.definitions[0], clock=wrong_clock),)))
    forged_reading = replace(lifecycle.definitions[0].clock, reading={"logical_step": 999})
    with pytest.raises(ValueError, match="clock reading"):
        project.save_symbol_lifecycle("experiment", replace(lifecycle,
            definitions=(replace(lifecycle.definitions[0], clock=forged_reading),)))
    with pytest.raises(ValueError, match="clock mismatch"):
        project.save_symbol_lifecycle("experiment", replace(lifecycle,
            definitions=(replace(lifecycle.definitions[0], source_cut=result, clock=None),)))
    adapter.restore(source)
    assert project.symbol_lifecycles("experiment") == []
    owner.close()


def test_invalidation_native_clock_is_verified_without_inferring_expiry(tmp_path):
    project, _, owner, lifecycle, source, result = recorded(tmp_path)
    try:
        clock = replace(lifecycle.definitions[0].clock,
                        reading=project.cut_descriptor("experiment", result)["execution_point"],
                        metadata={"source_cut": result})
        later = lifecycle.invalidate(scope="occurrence:definition-1", reason="later cursor observed",
                                     clock=clock)
        saved = project.save_symbol_lifecycle("experiment", later)
        assert result in saved["joins"]["cuts"]
        forged = replace(clock, metadata={"source_cut": source})
        with pytest.raises(ValueError, match="invalidation clock reading"):
            project.save_symbol_lifecycle("experiment", lifecycle.invalidate(
                scope="occurrence:definition-1", reason="forged cursor", clock=forged))
        malformed = replace(clock, metadata={"source_cut": 42})
        with pytest.raises(ValueError, match="reference must be a string"):
            project.save_symbol_lifecycle("experiment", lifecycle.invalidate(
                scope="occurrence:definition-1", reason="malformed cursor", clock=malformed))
    finally:
        owner.close()


def test_observed_coordinate_clock_and_use_need_native_closure(tmp_path):
    project, _, owner, lifecycle, source, _ = recorded(tmp_path)
    try:
        point = project.cut_descriptor("experiment", source)["execution_point"]
        token = replace(lifecycle.definitions[0].clock, kind="token-position",
                        reading=point["logical_step"],
                        metadata={"source_cut": source, "native_point_field": "logical_step"})
        project.save_symbol_lifecycle("experiment", replace(lifecycle,
            definitions=(replace(lifecycle.definitions[0], clock=token),)))
        with pytest.raises(ValueError, match="exact native point field"):
            project.save_symbol_lifecycle("experiment", replace(lifecycle,
                definitions=(replace(lifecycle.definitions[0], clock=replace(token, reading=999)),)))
        incomplete = replace(lifecycle.uses[0], parent_cut=None, result_cut=None, status="observed")
        with pytest.raises(ValueError, match="parent/result closure"):
            project.save_symbol_lifecycle("experiment", replace(lifecycle, uses=(incomplete,)))
        unknown = replace(incomplete, status="unknown")
        saved = project.save_symbol_lifecycle("experiment", replace(lifecycle, uses=(unknown,)))
        assert saved["joins"]["uses"][0]["consumer_support"] == "unverified-label"
        assert saved["joins"]["uses"][0]["receipt_closure"] == "incomplete"
    finally:
        owner.close()


def test_consumer_receipt_and_join_index_are_checked(tmp_path):
    project, _, owner, lifecycle, source, _ = recorded(tmp_path)
    wrong = replace(lifecycle.uses[0], result_cut=source)
    with pytest.raises(ValueError, match="result cut"):
        project.save_symbol_lifecycle("experiment", replace(lifecycle, uses=(wrong,)))
    unknown = replace(lifecycle.uses[0], receipt="a" * 64)
    with pytest.raises(ValueError, match="receipt is not retained"):
        project.save_symbol_lifecycle("experiment", replace(lifecycle, uses=(unknown,)))
    saved = project.save_symbol_lifecycle("experiment", lifecycle)
    with project._db() as db:
        row = db.execute("SELECT joins FROM symbol_lifecycles").fetchone()
        body = json.loads(row[0])
        body["definitions"][0]["support"] = "semantic-truth"
        db.execute("UPDATE symbol_lifecycles SET joins=?", (json.dumps(body),))
    with pytest.raises(ValueError, match="definition join"):
        project.symbol_lifecycle("experiment", saved["lifecycle_id"])
    owner.close()


def test_native_receipt_and_vm_context_cannot_cross_specimens(tmp_path, monkeypatch):
    project, _, owner, lifecycle, source, _ = recorded(tmp_path)
    try:
        native = next(r for r in project.receipts("experiment")
                      if r.fingerprint == lifecycle.uses[0].receipt)
        body = native.to_dict()
        body.pop("fingerprint")
        forged = type(native).make(**{**body, "model_identity": "other-model"})
        with monkeypatch.context() as patch:
            patch.setattr(project, "receipts", lambda sid: [forged])
            wrong_use = replace(lifecycle.uses[0], receipt=forged.fingerprint)
            with pytest.raises(ValueError, match="another model or execution"):
                project.save_symbol_lifecycle("experiment", replace(lifecycle, uses=(wrong_use,)))

        def bundled(context):
            return VMCacheExecutionBundle.create(
                run_id="counter", parent_identity={"source_cut": source},
                graph_fingerprint="a" * 64,
                model_identity={"id": lifecycle.binding.model_identity},
                input_identity={"source_cut": source, "context": context},
                abi_identity={"execution": lifecycle.binding.execution},
                plane_references={plane: "cas://sha256/" + "a" * 64
                                  for plane in ("source", "address", "operator", "renderer")})

        occurrence = replace(lifecycle.definitions[0], vm_bundle=bundled("other-context"))
        with pytest.raises(ValueError, match="VM bundle"):
            project.save_symbol_lifecycle("experiment", replace(lifecycle, definitions=(occurrence,)))
        occurrence = replace(occurrence, vm_bundle=bundled(lifecycle.binding.context))
        result = project.save_symbol_lifecycle("experiment", replace(lifecycle, definitions=(occurrence,)))
        assert result["joins"]["definitions"][0]["vm_support"] == "native-source-metadata-matched"
    finally:
        owner.close()


def test_lifecycle_cli_keeps_forward_and_reverse_results_sealed(tmp_path, capsys):
    project, _, owner, lifecycle, source, _ = recorded(tmp_path)
    artifact = tmp_path / "lifecycle.json"
    lifecycle.save(artifact)
    assert main(["symbols", str(project.root), "add", str(artifact)]) == 0
    assert json.loads(capsys.readouterr().out)["lifecycle_id"] == lifecycle.fingerprint
    assert main(["symbols", str(project.root), "at", "value", "--cut", source]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["matches"][0]["binding_fingerprint"] == lifecycle.binding_fingerprint
    owner.close()


def test_exact_spectral_lookup_uses_native_scope_and_abstains_on_wrong_instance(tmp_path):
    project, _, owner, lifecycle, source, _ = recorded(tmp_path)
    address = SpectralLogicalAddress(
        CanonicalSymbolKey(lifecycle.symbol, (0.1, 0.2), "debug", "counter", "fixture-book"),
        SpectralInstanceKey.deterministic("experiment", "main-counter", 2),
        SpectralRoleTransformSpec("carrier", (0.0, 0.0), 1, 0, "counter", "counter"),
        SpectralFrame(source))
    occurrence = replace(lifecycle.definitions[0], spectral_address=address)
    joined = replace(lifecycle, definitions=(occurrence,))
    project.save_symbol_lifecycle("experiment", joined)
    query = SpectralQuery("read-main", address.phases, "debug", "counter", source,
        exact_symbol_id=lifecycle.symbol, exact_scene_id="experiment",
        exact_instance_id="main-counter", exact_role_id="carrier")
    result = project.resolve_symbol_address("experiment", query, cut_id=source, branch="main")
    assert result["selected"]["occurrence"]["fingerprint"] == occurrence.fingerprint
    assert result["trace"]["selected_address_fingerprint"] == address.fingerprint
    assert not result["support"]["consumer_certified"]
    assert result["native_payload_reads"] == 0
    invalidated = joined.invalidate(scope="occurrence:definition-1", reason="supplied expiry",
                                    evidence_refs=("policy://expiry",))
    project.save_symbol_lifecycle("experiment", invalidated)
    historical = project.resolve_symbol_address("experiment", query, cut_id=source, branch="main")
    assert historical["selected"]["invalidations"][0]["reason"] == "supplied expiry"
    assert historical["selected"]["validity_policy"].startswith("recorded-evidence-only")
    wrong = project.resolve_symbol_address("experiment", replace(query, exact_instance_id="sibling"),
                                           cut_id=source, branch="main")
    assert wrong["selected"] is None and wrong["abstain_reason"] == "exact-identity-mismatch"
    wrong_role = project.resolve_symbol_address("experiment", replace(query, exact_role_id="other"),
                                                cut_id=source, branch="main")
    assert wrong_role["selected"] is None
    assert project.resolve_symbol_address("experiment", query, cut_id=source,
                                           branch="sibling")["selected"] is None
    with pytest.raises(ValueError, match="complete exact identity"):
        project.resolve_symbol_address("experiment", replace(query, exact_role_id=None), cut_id=source)
    owner.close()
