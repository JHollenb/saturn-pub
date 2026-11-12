"""Framework-free hand-specified protocol grammar and historical bookkeeping."""
from __future__ import annotations

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

FILE = Path(__file__).parents[1] / "qualification/compile_computation_family_protocols.py"
C = runpy.run_path(str(FILE))


def rules():
    sequential = dict(component_arm="PQP", development=dict(
        mean_to_support=dict(QQP=42.59515097313334, QPQ=45.92291265586476, QPP=41.10400537274775,
            PQQ=45.777682569035264, PQP=6.722908837175881, PPQ=45.70283876537572,
            QPM=45.86530272420121, PQM=8.725305225931914),
        menu_and_tie_order=["QQP", "QPQ", "QPP", "PQQ", "PQP", "PPQ", "QPM", "PQM"],
        score="Minimum mean old-context full-native-RGB RMS distance to support PPP"))
    parallel = dict(selected_component="RA", development=dict(mean_scores=dict(
        R=.7032671640437135, A=.8258993922820572, RA=.07103524682812941),
        menu_and_tie_order=["R", "A", "RA"], rule="Minimum mean signed historical final-latent coordinate"))
    return sequential, parallel


def test_framework_free_definitions_and_cli_only_read_supplied_rules(tmp_path):
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not {'torch','numpy','diffusers','mdb','saturn_pub'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(FILE)], check=True)
    first, second = (tmp_path / name for name in ("sequential.json", "parallel.json"))
    for path, rule in zip((first, second), rules(), strict=True):
        path.write_text(json.dumps(rule))
    output = tmp_path / "comparison.json"
    subprocess.run([sys.executable, "-I", "-B", str(FILE), "--sequential-rule", str(first),
                    "--parallel-rule", str(second), "--output", str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["protocols"]["sequential-text"]["historical_selection"]["selected_action"] == "PQP"
    assert result["protocols"]["parallel-image"]["historical_selection"]["selected_action"] == "RA"
    assert result["protocols"]["sequential-text"]["historical_selection"]["source_rule"]["path"] == str(first)


def test_known_dag_ancestry_distinguishes_formation_from_held_port_construction():
    sequential, parallel = (C["template"](kind) for kind in C["KINDS"])
    path = C["ancestry_path"](sequential["supplied_dag"], "A", "M")
    assert path == ["A", "write_A", "U", "X_M", "M"]
    assert C["ancestry_path"](parallel["supplied_dag"], "A", "M") is None
    assert C["ancestry_path"](parallel["supplied_dag"], "M", "A") is None
    for protocol in (sequential, parallel):
        validation = C["validate_protocol"](protocol)
        for name, port in protocol["supplied_dag"].items():
            assert all(validation["topological_order"].index(parent) < validation["topological_order"].index(name) for parent in port["parents"])
    assert "does not recompute M" in sequential["holding_policy"]
    assert "M_Q" in parallel["recipient_constants"]
    assert sequential["held_endpoint_real_arithmetic"]["expression"] == "E = R + G_A A + G_M M"
    assert parallel["held_endpoint_real_arithmetic"]["expression"] == "E = R + G_Q [W_A A + W_M M_Q + b]"
    assert "never substitutes split GEMMs" in parallel["held_endpoint_real_arithmetic"]["projection_rule"]
    assert "actual finite operator" in sequential["finite_B_convention"]
    assert sequential["port_types"]["A"]["output_projection_included"] is True
    assert sequential["port_types"]["M"]["output_projection_included"] is True
    assert parallel["port_types"]["A"]["output_projection_included"] is False
    assert parallel["port_types"]["M"]["output_projection_included"] is False
    assert parallel["port_types"]["projected"]["input_width"] == "attention.inner_dim + attention.mlp_hidden_dim"


def test_cycles_unknown_dependencies_and_unknown_intervention_ports_reject():
    protocol = C["template"]("parallel-image")
    protocol["supplied_dag"]["X"]["parents"].append("T")
    with pytest.raises(ValueError, match="cyclic"):
        C["validate_protocol"](protocol)
    protocol = C["template"]("sequential-text")
    protocol["supplied_dag"]["A"]["parents"].append("absent")
    with pytest.raises(ValueError, match="unknown DAG port"):
        C["validate_protocol"](protocol)
    protocol = C["template"]("sequential-text")
    protocol["actions"][1]["bindings"]["unknown_port"] = "P"
    with pytest.raises(ValueError, match="unknown action port"):
        C["validate_protocol"](protocol)


def test_corner_menus_do_not_confuse_source_anchors_with_natural_references():
    seq = C["enumerate_held_corners"]("sequential-text")
    assert len(seq) == 8 and sum(row["role"] == "candidate" for row in seq) == 6
    assert {row["id"]: row["role"] for row in seq}["PPP"] == "source-copy-anchor"
    parallel = C["template"]("parallel-image")
    by_id = {row["id"]: row for row in parallel["actions"]}
    assert by_id["RA"]["role"] == "candidate" and by_id["RA"]["bindings"] == dict(R="P", A="P")
    assert by_id["P"]["role"] == "natural-reference" and not by_id["P"]["bindings"]
    sequential = C["template"]("sequential-text")
    own = {row["id"]: row for row in sequential["actions"]}
    assert own["QPM"]["bindings"]["M"] == own["PQM"]["bindings"]["M"] == "recomputed-own"
    assert own["PQP"]["bindings"]["M"] == "P"
    assert own["whole_P"]["role"] == "whole-text-reference"
    assert "Both retain Q image/nontext receiving closure" in sequential["reference_note"]
    assert "image-primed" in sequential["source_policy"] and "no image prime" in parallel["source_policy"]


def test_historical_selector_excludes_even_better_anchor_scores_and_rejects_relabeling():
    rule, _ = rules()
    protocol = C["template"]("sequential-text")
    packet = C["development_packet"](protocol, rule)
    packet["scores"].update(PPP=-1000., QQQ=-2000., whole_P=-3000.)
    selected = C["select_historical"](protocol, packet)
    assert selected["selected_action"] == "PQP" and set(selected["excluded_scores"]) == {"PPP", "QQQ", "whole_P"}
    hostile = copy.deepcopy(packet)
    hostile["menu_and_tie_order"].append("PPP")
    with pytest.raises(ValueError, match="anchor/reference"):
        C["select_historical"](protocol, hostile)
    for row in protocol["actions"]:
        if row["id"] == "PPP":
            row["role"] = "candidate"
    protocol["candidate_menu"].append("PPP")
    with pytest.raises(ValueError, match="anchor role"):
        C["select_historical"](protocol, hostile)
    protocol = C["template"]("sequential-text")
    protocol["actions"].append(dict(id="renamed_anchor", role="candidate", kind="held-port-corner", bindings=dict(R="P", A="P", M="P")))
    protocol["candidate_menu"].append("renamed_anchor")
    with pytest.raises(ValueError, match="anchor role"):
        C["validate_protocol"](protocol)


def test_metrics_units_scopes_and_observables_are_not_cross_ranked():
    seq_rule, parallel_rule = rules()
    seq, parallel = (C["template"](kind) for kind in C["KINDS"])
    packet = C["development_packet"](parallel, parallel_rule)
    with pytest.raises(ValueError, match="score domain"):
        C["select_historical"](seq, packet)
    packet = C["development_packet"](seq, seq_rule)
    packet["metric"]["units"] = "dimensionless"
    with pytest.raises(ValueError, match="score domain"):
        C["select_historical"](seq, packet)
    result = C["compile_protocols"](seq_rule, parallel_rule)
    comparison = result["comparison"]["selection_metrics"]
    assert comparison["comparable"] is False and {"observable", "units", "scope", "reference"} <= set(comparison["differences"])
    assert "not independently verified" in comparison["comparison_scope"]
    assert "overall_winner" not in result and "cross_family_scores" not in result


def test_declared_tie_order_mismatch_nonfinite_and_unknown_scores_are_handled():
    _, rule = rules()
    protocol = C["template"]("parallel-image")
    packet = C["development_packet"](protocol, rule)
    packet["scores"] = dict(R=1., A=1., RA=1.)
    packet["declared_selection"] = "R"
    assert C["select_historical"](protocol, packet)["selected_action"] == "R"
    packet["declared_selection"] = "RA"
    with pytest.raises(ValueError, match="declared historical selection"):
        C["select_historical"](protocol, packet)
    packet["scores"]["R"] = float("nan")
    with pytest.raises(ValueError):
        C["select_historical"](protocol, packet)
    packet = C["development_packet"](protocol, rule)
    packet["scores"]["fresh_outcome_candidate"] = -1e9
    with pytest.raises(ValueError, match="score/action census"):
        C["select_historical"](protocol, packet)


def test_compilation_is_copy_owned_stable_and_labels_supplied_grammar_limits():
    first, second = rules()
    before = copy.deepcopy((first, second))
    result = C["compile_protocols"](first, second)
    assert (first, second) == before
    assert result == C["compile_protocols"](first, second)
    assert result["comparison"]["overlap"]["automatic_topology_or_ontology_discovery"] is False
    assert "Hand-specified" in result["compiler_scope"]
    assert all(protocol["historical_selection"]["fresh_outcomes_used"] is False for protocol in result["protocols"].values())
    result["protocols"]["parallel-image"]["candidate_menu"].append("P")
    assert C["compile_protocols"](first, second)["protocols"]["parallel-image"]["candidate_menu"] == ["R", "A", "RA"]
