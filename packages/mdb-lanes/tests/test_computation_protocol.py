"""Finite protocol execution, quarantine identities and nonlinear graph structure."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys

import pytest

from mdb_lanes.computation_protocol import (
    SPEC_SCHEMA,
    STAGES,
    ProtocolError,
    compare_protocols,
    compile_protocol,
    digest,
    evaluate_rule,
    execute_program,
    freeze_rule,
    run_graph,
    specialize_spec,
    verify_plan,
)


def spec(kind="attention"):
    def port(stage):
        return {"shape": [1, "T", "D"], "dtype": "fixture", "stage": stage}

    if kind == "attention":
        ports = {name: port("actual consumed " + name) for name in ("q", "k", "v")}
        factors = [{"id": "route", "ports": ["q", "k"]}, {"id": "value", "ports": ["v"]}]
        nodes = [{"id": "result", "op": "sdpa", "inputs": ["q", "k", "v"]}]
    else:
        ports = {name: port("actual preactivation " + name) for name in ("G", "U")}
        factors = [{"id": "gate", "ports": ["G"]}, {"id": "value", "ports": ["U"]}]
        nodes = [{"id": "activated", "op": "silu", "inputs": ["G"]},
                 {"id": "result", "op": "multiply", "inputs": ["activated", "U"]}]
    return {"schema": SPEC_SCHEMA, "family_id": "arbitrary-" + kind, "model_identity": "checkpoint-fixture",
            "endpoint_adapter": "fixture-held-full-projection", "source_policy": "paired fixture only",
            "ports": ports, "factors": factors, "nodes": nodes, "output": "result",
            "max_contexts": 4, "receiver": {"R": "P", "text": "Q"},
            "clock": {"step": 0}, "continuation": {"consumer": "fixture only", "transitions": 101},
            "metric": {"observable": "signed final latent coordinate", "units": "dimensionless",
                       "reference": "natural P=0/Q=1", "scope": "fixture scope", "direction": "minimize"}}


def measurements(plan, scores, context="development"):
    return {"plan_sha256": plan["plan_sha256"], "model_identity": plan["specification"]["model_identity"],
            "metric": copy.deepcopy(plan["specification"]["metric"]), "provenance": {"fixture": True},
            "rows": [{"context": context, "action": action["id"], "value": scores.get(action["id"], 0.)}
                     for action in plan["actions"]]}


def test_compile_emits_nontrivial_candidates_and_distinguishes_two_baselines():
    plan = compile_protocol(spec())
    roles = {action["id"]: action for action in plan["actions"]}
    assert len(roles) == 6 and plan["candidate_menu"] == ["value", "route"]
    assert roles["recipient"]["source"] == "Q" and roles["recipient"]["role"] == "natural-reference"
    assert roles["baseline"]["role"] == "baseline" and roles["baseline"]["bindings"] == {"route": "Q", "value": "Q"}
    assert roles["both"]["role"] == "component-reference"
    assert plan["specification"]["receiver"]["R"] == "P"
    assert [row["operation"] for row in plan["program"]] == list(STAGES)


def test_registered_graph_traits_find_shared_payload_law_without_family_names():
    attention, mlp = (compile_protocol(spec(kind)) for kind in ("attention", "mlp"))
    for plan in (attention, mlp):
        assert plan["structure"]["factor_linearity"]["value"] == "linear"
        assert plan["structure"]["shared_pattern"] == "state-conditioned linear action on payload"
        assert plan["structure"]["finite_reassociation_permitted"] is False
    assert attention["structure"]["factor_linearity"]["route"] == "not-proven-affine"
    assert mlp["structure"]["factor_linearity"]["gate"] == "not-proven-affine"
    assert "token rows" in attention["structure"]["operation_mixing"]
    assert mlp["structure"]["operation_mixing"] == ["local channels"]
    compared = compare_protocols([attention, mlp])
    assert compared["cross_family_score_ranking"] is False


def test_geometry_is_generated_from_roles_graph_nodes_and_native_horizon():
    source = spec("mlp")
    source["continuation"].update(transformer_blocks=90, vae_decodes=1)
    geometry = compile_protocol(source)["execution_geometry"]
    assert geometry["per_context_per_lane"] == {
        "source_branches": 2, "primitive_endpoints": 4, "futures": 6, "suffix_transitions": 606,
        "VAE_decodes": 6, "common_parent_Acts": 5, "native_transformer_blocks": 542,
        "declared_graph_node_invocations": {"silu": 4, "multiply": 4},
    }
    assert geometry["aggregate_work_items_per_context_both_lanes"] == 36


def test_autoregressive_geometry_counts_own_token_forwards_without_image_or_act_units():
    source = spec("mlp")
    source["continuation"].update(execution_kind="autoregressive", transitions=6, vae_decodes=0)
    geometry = compile_protocol(source)["execution_geometry"]
    row = geometry["per_context_per_lane"]
    assert row["prefill_forwards"] == 6 and row["cached_forwards"] == 30
    assert row["physical_forwards"] == row["greedy_choices"] == 36
    assert row["VAE_decodes"] == row["common_parent_Acts"] == 0
    assert row["module_hook_interventions"] == 4
    assert geometry["aggregate_work_items_per_context_both_lanes"] == 72
    assert verify_plan(compile_protocol(source))
    source["continuation"]["vae_decodes"] = 1
    with pytest.raises(ProtocolError, match="no VAE"):
        compile_protocol(source)
    source["continuation"].update(execution_kind="unknown", vae_decodes=0)
    with pytest.raises(ProtocolError, match="execution kind"):
        compile_protocol(source)


def capture_spec():
    source = spec("mlp")
    source["requires_specialization"] = True
    for index, name in enumerate(("G", "U")):
        source["ports"][name].update(shape=["B", "T", "H_M"],
            capture={"port": "mlp_input", "chunk": {"axis": -1, "parts": 2, "index": index}})
    capture = {"schema": "mdb-operation-port-capture-metadata-v1", "model_identity": source["model_identity"],
        "clock": source["clock"], "provenance": {"source_record_sha256": "b"*64},
        "ports": {"mlp_input": {"shape": [1, 1536, 18432], "stride": [42467328, 27648, 1],
            "dtype": "fixture", "content_sha256": "a"*64}}}
    return source, capture


def test_specialization_uses_captured_shapes_and_exact_view_recipes_without_arrays(monkeypatch):
    source, capture = capture_spec()
    with pytest.raises(ProtocolError, match="specialization required"):
        compile_protocol(source)
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: pytest.fail("specializer opened a file"))
    concrete = specialize_spec(source, capture)
    plan = compile_protocol(concrete)
    assert plan["specification"]["ports"]["G"]["shape"] == [1, 1536, 9216]
    assert concrete["specialization"]["symbols"] == {"B": 1, "T": 1536, "H_M": 9216}
    assert concrete["specialization"]["views"]["U"]["relative_storage_offset_elements"] == 9216
    assert concrete["specialization"]["views"]["U"]["parent_content_sha256"] == "a"*64
    assert "child_content_sha256" not in concrete["specialization"]["views"]["U"]
    assert source["ports"]["U"]["shape"] == ["B", "T", "H_M"]


def test_specialization_rejects_wrong_checkpoint_clock_hash_and_slice_geometry():
    for mutate in (lambda c: c.update(model_identity="other"), lambda c: c.update(clock={"step": 1}),
                   lambda c: c["ports"]["mlp_input"].update(content_sha256="unbound"),
                   lambda c: c["ports"]["mlp_input"].update(shape=[1, 1536, 18433])):
        source, capture = capture_spec()
        mutate(capture)
        with pytest.raises(ProtocolError):
            specialize_spec(source, capture)


def test_previous_v2_plan_identity_is_preserved_without_rewriting_legacy_bytes():
    plan = compile_protocol(spec())
    plan.pop("execution_geometry")
    plan["schema"] = "mdb-computation-family-program-v2"
    plan.pop("plan_sha256")
    plan["plan_sha256"] = digest(plan)
    before = copy.deepcopy(plan)
    assert verify_plan(plan) == before and plan == before


def test_linearity_does_not_survive_same_factor_on_both_product_inputs():
    source = spec("mlp")
    source["nodes"] = [{"id": "result", "op": "multiply", "inputs": ["G", "U"]}]
    source["factors"] = [{"id": "joint", "ports": ["G", "U"]}]
    # Add a separate fixed-use factor so the candidate universe stays nontrivial.
    source["ports"]["b"] = {"shape": [1], "dtype": "fixture", "stage": "offset"}
    source["factors"].append({"id": "offset", "ports": ["b"]})
    source["nodes"].append({"id": "out", "op": "add", "inputs": ["result", "b"]})
    source["output"] = "out"
    plan = compile_protocol(source)
    assert plan["structure"]["factor_linearity"] == {"joint": "not-proven-affine", "offset": "affine"}


def test_additive_conditioning_is_affine_and_does_not_gain_linear_label():
    source = spec("mlp")
    source["nodes"][1]["op"] = "add"
    structure = compile_protocol(source)["structure"]
    assert structure["factor_linearity"]["value"] == "affine"
    assert structure["shared_pattern"] == "state-conditioned affine action on payload"


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["nodes"][0].update(inputs=["result", "k", "v"]), "cyclic"),
    (lambda s: s["nodes"][0].update(inputs=["missing", "k", "v"]), "unknown graph"),
    (lambda s: s["factors"][1].update(ports=["q"]), "aliases overlap"),
    (lambda s: s["nodes"][0].update(op="eval"), "unregistered"),
    (lambda s: s["nodes"][0].update(inputs=["q", "k"]), "arity"),
    (lambda s: s["ports"]["q"].update(dtype="torch.float16"), "dtype"),
    (lambda s: s["nodes"][0].update(attributes={"dropout_p": .5}), "SDPA profile"),
    (lambda s: s.update(max_contexts=True), "finite bound"),
])
def test_invalid_graphs_and_profiles_refuse_before_execution(mutate, match):
    source = spec()
    mutate(source)
    with pytest.raises(ProtocolError, match=match):
        compile_protocol(source)


def test_interpreter_binds_bundles_coherently_and_runs_actual_generated_nodes():
    plan = compile_protocol(spec("mlp"))
    sources = {"P": {"G": 2., "U": 3.}, "Q": {"G": -1., "U": 4.}}
    calls = []

    def silu(value):
        calls.append(("silu", value))
        return value / (1 + math.exp(-value))

    result = run_graph(plan, sources, {"gate": "P", "value": "Q"},
                       {"silu": silu, "multiply": lambda x, y: x*y})
    assert calls == [("silu", 2.)] and result == pytest.approx(8/(1+math.exp(-2)))
    with pytest.raises(ProtocolError, match="binding census"):
        run_graph(plan, sources, {"gate": "P"}, {})


def test_program_executes_all_stages_after_seal_and_retains_scientific_misses():
    plan, called = compile_protocol(spec()), []

    def handler(command):
        called.append(command["operation"])
        if command["operation"] == "seal":
            return {"seal_sha256": "a"*64, "native_calls_before_seal": 0}
        return {"mechanics": "complete", "scientific_status": "counterexample"}

    receipt = execute_program(plan, {name: handler for name in STAGES})
    assert called == list(STAGES) and len(receipt["commands"]) == 10
    assert receipt["commands"][-1]["result"]["scientific_status"] == "counterexample"


def test_program_rejects_missing_or_dirty_seal_before_any_native_callback():
    for seal in ({}, {"seal_sha256": "a"*64, "native_calls_before_seal": 1}):
        called = []

        def handler(command):
            called.append(command["operation"])
            return seal if command["operation"] == "seal" else {}

        with pytest.raises(ProtocolError, match="forecast release"):
            execute_program(compile_protocol(spec()), {name: handler for name in STAGES})
        assert not any(name.startswith("native_") for name in called)


def test_plan_mutation_rejects_before_model_callbacks():
    plan = compile_protocol(spec())
    plan["program"][0]["operation"] = "native_sources"
    with pytest.raises(ProtocolError, match="differs"):
        verify_plan(plan)


def test_freeze_ignores_even_better_references_and_captures_no_fresh_outcomes():
    plan = compile_protocol(spec())
    packet = measurements(plan, {"source": -1e8, "both": -1e9, "route": .2, "value": .8})
    rule = freeze_rule(plan, packet, ["fresh"])
    assert rule["selected_action"] == "route" and rule["fresh_outcomes_used"] is False
    assert set(rule["mean_scores"]) == {"route", "value"}
    with pytest.raises(ProtocolError, match="overlap"):
        freeze_rule(plan, packet, ["development"])


def test_counterexamples_and_undefined_axes_are_retained_per_context():
    plan = compile_protocol(spec())
    rule = freeze_rule(plan, measurements(plan, {"route": .2, "value": .8}), ["good", "miss", "undefined"])
    packet = measurements(plan, {"route": .1, "value": .9}, "good")
    packet["rows"] += measurements(plan, {"route": .9, "value": .1}, "miss")["rows"]
    packet["rows"] += measurements(plan, {"route": None, "value": None}, "undefined")["rows"]
    result = evaluate_rule(plan, rule, packet)
    assert result["counts"] == {"held": 1, "counterexample": 1, "not-applicable": 1}
    assert [row["signed_margin"] for row in result["outcomes"]] == [.8, -.8, None]


def context(identifier, seed):
    return {"id": identifier, "seed": seed, "panel": "fixture", "prompts": {"P": "a left of b", "Q": "a right of b"}}


def test_full_context_binding_prevents_label_or_reciprocal_reuse_and_outcome_rebinding():
    plan = compile_protocol(spec())
    packet = measurements(plan, {"route": .2, "value": .8})
    packet["contexts"] = [context("development", 17)]
    for reused in (context("renamed", 17), {**context("reciprocal", 17), "prompts": {"P": "a right of b", "Q": "a left of b"}}):
        with pytest.raises(ProtocolError, match="prompt/seed pair reused"):
            freeze_rule(plan, packet, [reused])
    rule = freeze_rule(plan, packet, [context("fresh", 19)])
    fresh = measurements(plan, {"route": .3, "value": .7}, "fresh")
    fresh["contexts"] = [context("fresh", 19)]
    assert evaluate_rule(plan, rule, fresh)["counts"]["held"] == 1
    fresh["contexts"][0]["seed"] = 23
    with pytest.raises(ProtocolError, match="prompt/seed binding"):
        evaluate_rule(plan, rule, fresh)


def test_full_context_mode_requires_matching_measurement_metadata_and_preserves_prompt_bytes():
    plan = compile_protocol(spec())
    packet = measurements(plan, {"route": .2, "value": .8})
    with pytest.raises(ProtocolError, match="full prompt/seed"):
        freeze_rule(plan, packet, [context("fresh", 19)])
    packet["contexts"] = [context("development", 17)]
    exact = context("fresh", 19)
    exact["prompts"]["P"] += "  "
    rule = freeze_rule(plan, packet, [exact])
    fresh = measurements(plan, {"route": .3, "value": .7}, "fresh")
    fresh["contexts"] = [context("fresh", 19)]
    with pytest.raises(ProtocolError, match="prompt/seed binding"):
        evaluate_rule(plan, rule, fresh)


def test_measurement_metric_identity_and_full_action_census_are_required():
    plan = compile_protocol(spec())
    for mutate in (lambda p: p.update(model_identity="other"),
                   lambda p: p["metric"].update(reference="component both"),
                   lambda p: p["rows"].pop()):
        packet = measurements(plan, {"route": .2, "value": .8})
        mutate(packet)
        with pytest.raises(ProtocolError):
            freeze_rule(plan, packet, ["fresh"])


def test_installed_module_and_cli_are_framework_free_and_copy_owned(tmp_path):
    source = spec("mlp")
    plan = compile_protocol(source)
    source["receiver"]["R"] = "Q"
    assert plan["specification"]["receiver"]["R"] == "P"
    path, output = tmp_path / "spec.json", tmp_path / "plan.json"
    path.write_text(json.dumps(spec("mlp")))
    subprocess.run([sys.executable, "-I", "-B", "-c",
                    "import sys; from mdb_lanes.computation_protocol import main; main(sys.argv[1:]); "
                    "assert not {'torch','numpy','diffusers','transformers'} & sys.modules.keys()",
                    "compile", str(path), "--out", str(output)], check=True)
    assert json.loads(output.read_text()) == plan
