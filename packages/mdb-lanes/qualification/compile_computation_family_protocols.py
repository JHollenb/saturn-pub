"""Compile two hand-specified computation-family protocols from rule metadata.

This is experimental grammar and bookkeeping. The graphs, ports, candidate
menus, clocks and observables are supplied explicitly; no topology, semantic
ontology, model execution or fresh outcome is discovered here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import deque
from pathlib import Path

SCHEMA = "mdb-computation-family-protocols-v1"
KINDS = ("sequential-text", "parallel-image")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def copied(value):
    return json.loads(canonical(value))


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_dag(ports):
    require(isinstance(ports, dict) and ports and all(isinstance(name, str) and name for name in ports), "named DAG ports required")
    for name, port in ports.items():
        require(isinstance(port, dict) and isinstance(port.get("operation"), str)
                and isinstance(port.get("parents"), list) and all(isinstance(parent, str) for parent in port["parents"])
                and len(port["parents"]) == len(set(port["parents"])), "invalid DAG port: " + name)
        require(all(parent in ports for parent in port["parents"]), "unknown DAG port referenced by " + name)
        require(port["operation"] != "supplied" or not port["parents"], "supplied port cannot have parents")
    active, done, order, ancestors = set(), set(), [], {}

    def visit(name):
        require(name not in active, "cyclic computation DAG at " + name)
        if name in done:
            return
        active.add(name)
        inherited = set()
        for parent in ports[name]["parents"]:
            visit(parent)
            inherited.update(ancestors[parent])
            inherited.add(parent)
        active.remove(name)
        done.add(name)
        ancestors[name] = sorted(inherited)
        order.append(name)

    for name in ports:
        visit(name)
    return dict(topological_order=order, ancestors=ancestors)


def ancestry_path(ports, ancestor, descendant):
    validate_dag(ports)
    require(ancestor in ports and descendant in ports, "unknown ancestry port")
    queue = deque([[ancestor]])
    while queue:
        path = queue.popleft()
        if path[-1] == descendant:
            return path
        queue.extend(path + [name] for name, port in ports.items() if path[-1] in port["parents"])
    return None


def node(operation, *parents):
    return dict(operation=operation, parents=list(parents))


def enumerate_held_corners(kind):
    require(kind in KINDS, "unknown family template")
    names = ("R", "A", "M") if kind == "sequential-text" else ("R", "A")
    result = []
    parallel = {"QQ": "Q", "PQ": "R", "QP": "A", "PP": "RA"}
    for sources in itertools.product(("Q", "P"), repeat=len(names)):
        signature = "".join(sources)
        identifier = signature if kind == "sequential-text" else parallel[signature]
        role = "baseline" if all(source == "Q" for source in sources) else "candidate"
        if kind == "sequential-text" and signature == "PPP":
            role = "source-copy-anchor"
        result.append(dict(id=identifier, role=role, kind="held-port-corner", bindings=dict(zip(names, sources, strict=True))))
    return result


def template(kind):
    require(kind in KINDS, "unknown family template")
    if kind == "sequential-text":
        ports = dict(R=node("supplied"), gA=node("supplied"), gM=node("supplied"), context=node("supplied"), factors=node("supplied"),
            X_A=node("normalize", "R", "context"), A=node("attention", "X_A", "context", "factors"),
            write_A=node("gate_multiply", "gA", "A"), U=node("residual_add", "R", "write_A"),
            X_M=node("normalize", "U", "context"), M=node("mlp", "X_M", "factors"),
            write_M=node("gate_multiply", "gM", "M"), T=node("residual_add", "U", "write_M"))
        actions = enumerate_held_corners(kind) + [
            dict(id="QPM", role="candidate", kind="recomputed-mlp-crossing", bindings=dict(R="Q", A="P", M="recomputed-own")),
            dict(id="PQM", role="candidate", kind="recomputed-mlp-crossing", bindings=dict(R="P", A="Q", M="recomputed-own")),
            dict(id="whole_P", role="whole-text-reference", kind="whole-text-reference-in-Q-closure", bindings={})]
        menu = ["QQP", "QPQ", "QPP", "PQQ", "PQP", "PPQ", "QPM", "PQM"]
        metric = dict(observable="full-native-RGB RMS distance", units="RGB uint8 intensity RMS", reference="support PPP endpoint",
            scope="historical sequential text-support contexts; fixed Q receiver; step2/joint0;54-transition reset-policy future", direction="minimize")
        return dict(id=kind, supplied_dag=ports, intervention_ports=["R", "A", "M"], actions=actions, candidate_menu=menu,
            selection_metric=metric, clock=dict(step=2, phase="joint", index=0), receiving_boundary="step2/joint1",
            port_types=dict(
                R=dict(stage="text residual before sequential writer", width="model.hidden_width", representation="residual state"),
                A=dict(stage="attention returned contribution after separate text output projection", width="model.hidden_width",
                    representation="output-projected return; ungated until recipient gA multiplication", output_projection_included=True),
                M=dict(stage="MLP returned contribution after its separate output projection", width="model.hidden_width",
                    representation="output-projected return; ungated until recipient gM multiplication", output_projection_included=True)),
            stream="declared text rows S; Q image and Qout outside S", finite_equation="U=B(R+B(gA*A)); T=B(U+B(gM*M))",
            held_endpoint_real_arithmetic=dict(expression="E = R + G_A A + G_M M",
                fixed_constants=["G_A=diag(gA_Q)", "G_M=diag(gM_Q)", "selected text rows S", "Qout outside S and Q nontext closure"],
                scope="Affine endpoint map of held R/A/M ports only; source formation and downstream continuation are separate graphs",
                recomputed_candidates="QPM/PQM derive M from their own nonlinear U-to-MLP computation; they are not three independently held coordinates"),
            finite_B_convention="B denotes the qualified Torch2.13 CUDA BF16 operation/cast convention at each native result; it is an actual finite operator, with no ideal rounding guarantee asserted",
            holding_policy="Held M_P was formed from P's own preceding attention update; crossing A does not recompute M unless QPM/PQM explicitly requests it.",
            recipient_constants=["gA_Q", "gM_Q", "Q image/nontext state", "Qout text outside S"],
            continuation="54 own-state transitions; baseline conditioning reset at step3/project; original VAE and RGB",
            source_policy="Source-assisted/image-primed paired inputs under the frozen source policy; P is not an untouched natural history.",
            extra_controls=["complement_P", "wrong_address"],
            reference_note="PPP installs P postblock text only on S; whole_P installs all P postblock text rows. Both retain Q image/nontext receiving closure and the same baseline reset policy.")
    ports = dict(R=node("supplied"), g=node("supplied"), context=node("supplied"), factors=node("supplied"),
        X=node("normalize", "R", "context"), A=node("attention", "X", "context", "factors"), M=node("mlp", "X", "factors"),
        merge=node("concat", "A", "M"), projected=node("linear", "merge", "factors"),
        write=node("gate_multiply", "g", "projected"), T=node("residual_add", "R", "write"))
    actions = enumerate_held_corners(kind) + [dict(id="P", role="natural-reference", kind="separate-full-state-reference", bindings={})]
    metric = dict(observable="native final-latent signed P=0/Q=1 coordinate", units="dimensionless signed axis projection",
        reference="natural full P and Q futures", scope="historical parallel image-relation contexts; step0/single4; natural persistent conditioning",
        direction="minimize")
    return dict(id=kind, supplied_dag=ports, intervention_ports=["R", "A"], actions=actions, candidate_menu=["R", "A", "RA"],
        selection_metric=metric, clock=dict(step=0, phase="single", index=4), receiving_boundary="step0/single5",
        port_types=dict(
            R=dict(stage="combined text/image residual before parallel writer; image rows selected", width="model.hidden_width", representation="residual state"),
            A=dict(stage="flattened attention features before shared to_out", width="attention.inner_dim",
                representation="preprojection features; common recipient gate applied after shared projection", output_projection_included=False),
            M=dict(stage="SwiGLU features before shared to_out", width="attention.mlp_hidden_dim",
                representation="preprojection features; M_Q fixed in every R/A candidate", output_projection_included=False),
            projected=dict(stage="shared original to_out(concat(A,M)) return", input_width="attention.inner_dim + attention.mlp_hidden_dim",
                width="model.hidden_width", representation="output-projected return; recipient gate applied afterward")),
        stream="all image rows; Q postblock text and complete Q persistent state", finite_equation="T=B(R+B(g_Q*Linear_original_full(concat(A,M_Q))))",
        held_endpoint_real_arithmetic=dict(expression="E = R + G_Q [W_A A + W_M M_Q + b]",
            fixed_constants=["G_Q=diag(g_Q)", "original W_out=[W_A W_M] and bias b", "M_Q", "Q postblock text and complete Q nonimage state"],
            scope="Affine endpoint map of held R/A ports only; source formation and downstream continuation are separate graphs",
            projection_rule="W_A/W_M are structural column partitions for the real-arithmetic identity; execution retains one full concat-shaped original projection and never substitutes split GEMMs"),
        finite_B_convention="B denotes the qualified Torch2.13 CUDA BF16 operation/cast convention at each native result; the full original projection keeps its own finite accumulation profile, with no ideal rounding guarantee asserted",
        holding_policy="A and M form in parallel from X; every new R/A corner retains M_Q and g_Q.",
        recipient_constants=["M_Q", "g_Q", "Q postblock text", "Q conditioning and every nonimage state field"],
        continuation="101 own-state transitions; persistent native conditioning; original VAE and RGB",
        source_policy="Natural paired prompt prefixes with identical initial noise per pair; no image prime.",
        extra_controls=[], reference_note="RA is a fixed-Q candidate, not a source-copy anchor: natural P also has P MLP, text and conditioning.")


def validate_protocol(protocol):
    require(protocol["id"] in KINDS, "unknown family template")
    result = validate_dag(protocol["supplied_dag"])
    allowed = set(protocol["intervention_ports"])
    require(allowed <= set(protocol["supplied_dag"]), "unknown intervention port")
    actions = protocol["actions"]
    require(len({action["id"] for action in actions}) == len(actions), "duplicate action ID")
    held = {action["id"]: action for action in enumerate_held_corners(protocol["id"])}
    for action in actions:
        require(action["role"] in ("baseline", "candidate", "source-copy-anchor", "whole-text-reference", "natural-reference"), "unknown action role")
        require(set(action["bindings"]) <= allowed, "unknown action port")
        require(all(source in ("P", "Q", "recomputed-own") for source in action["bindings"].values()), "unknown source binding")
        require(action["kind"] in ("held-port-corner", "recomputed-mlp-crossing", "whole-text-reference-in-Q-closure", "separate-full-state-reference"), "unknown action kind")
        if action["kind"] == "held-port-corner":
            require(action["id"] in held and action == held[action["id"]], "held corner/anchor role or source binding differs")
        if action["role"] == "natural-reference":
            require(protocol["id"] == "parallel-image" and action["id"] == "P" and not action["bindings"]
                and action["kind"] == "separate-full-state-reference", "natural reference is not a candidate action")
        if action["kind"] == "whole-text-reference-in-Q-closure":
            require(protocol["id"] == "sequential-text" and action["id"] == "whole_P" and action["role"] == "whole-text-reference"
                and not action["bindings"], "whole-text reference must retain Q closure")
        if "recomputed-own" in action["bindings"].values():
            require(action["kind"] == "recomputed-mlp-crossing"
                and [name for name, source in action["bindings"].items() if source == "recomputed-own"] == ["M"]
                and protocol["id"] == "sequential-text" and action["id"] in ("QPM", "PQM"), "unsupported recomputed port")
    candidates = {action["id"] for action in actions if action["role"] == "candidate"}
    require(len(protocol["candidate_menu"]) == len(set(protocol["candidate_menu"]))
            and set(protocol["candidate_menu"]) == candidates, "candidate menu contains anchor/reference or omits a candidate")
    return result


def development_packet(protocol, rule, source=None):
    rule = copied(rule)
    development = rule["development"]
    sequential = protocol["id"] == "sequential-text"
    return dict(family_id=protocol["id"], metric=copied(protocol["selection_metric"]),
        menu_and_tie_order=development["menu_and_tie_order"],
        scores=development["mean_to_support" if sequential else "mean_scores"],
        declared_selection=rule["component_arm" if sequential else "selected_component"],
        source_rule_content_sha256=digest(rule), source_rule=None if source is None else copied(source),
        provenance=dict(development=development, score_authority="immutable supplied historical rule metadata; underlying result files not opened"))


def select_historical(protocol, packet):
    validate_protocol(protocol)
    packet = copied(packet)
    require(packet["family_id"] == protocol["id"] and packet["metric"] == protocol["selection_metric"], "score domain is not comparable to this protocol")
    menu = packet["menu_and_tie_order"]
    require(len(menu) == len(set(menu)) and set(menu) == set(protocol["candidate_menu"]), "historical menu contains anchor/reference or unknown candidate")
    actions = {action["id"]: action for action in protocol["actions"]}
    scores = packet["scores"]
    require(isinstance(scores, dict) and set(menu) <= set(scores) <= set(actions), "historical score/action census differs")
    require(all(type(value) in (int, float) and math.isfinite(value) for value in scores.values()), "finite numeric development scores required")
    selected = min(menu, key=lambda name: (scores[name], menu.index(name)))
    require(actions[selected]["role"] == "candidate", "anchors/references cannot be selected")
    require(packet["declared_selection"] == selected, "declared historical selection differs")
    return dict(selected_action=selected, selected_score=scores[selected], metric=packet["metric"],
        declared_menu_and_tie_order=menu, candidate_scores={name: scores[name] for name in menu},
        excluded_scores={name: dict(score=value, role=actions[name]["role"]) for name, value in scores.items() if name not in menu},
        source_rule=packet["source_rule"], source_rule_content_sha256=packet["source_rule_content_sha256"],
        basis="Minimum supplied historical mean within the declared candidate menu; declared order breaks exact ties",
        fresh_outcomes_used=False, cross_family_ranking=False, provenance=packet["provenance"])


def compare_metric_domains(left, right):
    require(isinstance(left, dict) and isinstance(right, dict), "metric metadata required")
    differences = {key: dict(left=left.get(key), right=right.get(key)) for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)}
    return dict(comparable=not differences, differences=differences,
        comparison_scope="Attached protocol metric declarations only; original measurement and score provenance are not independently verified",
        cross_family_scalar_ranking="not defined; retain separate observables, units, reference axes and scopes" if differences else "same declared metric domain only")


def compile_protocols(sequential_rule, parallel_rule, *, rule_sources=None):
    rules = dict(zip(KINDS, (sequential_rule, parallel_rule), strict=True))
    sources = {} if rule_sources is None else copied(rule_sources)
    require(set(sources) <= set(KINDS), "unknown rule-source family")
    protocols = {}
    for kind in KINDS:
        protocol = template(kind)
        protocol["dag_validation"] = validate_protocol(protocol)
        protocol["historical_selection"] = select_historical(protocol, development_packet(protocol, rules[kind], sources.get(kind)))
        protocol["protocol_sha256"] = digest(protocol)
        protocols[kind] = protocol
    sequential, parallel = (protocols[kind] for kind in KINDS)
    operations = [{port["operation"] for port in protocol["supplied_dag"].values()} for protocol in (sequential, parallel)]
    evidence = {kind: dict(A_to_M=ancestry_path(protocol["supplied_dag"], "A", "M"),
        R_to_A=ancestry_path(protocol["supplied_dag"], "R", "A")) for kind, protocol in protocols.items()}
    comparison = dict(overlap=dict(operations=sorted(operations[0] & operations[1]),
        experimental_grammar=["supplied source frontier", "original factors", "typed ports", "explicit candidate menu", "fixed recipient closure", "sealed own-state forecast", "native continuation", "saved independent audit"],
        automatic_topology_or_ontology_discovery=False),
        differences=dict(ancestry_evidence=evidence,
            dependency="Sequential A has a directed formation path to M; parallel A and M share X but neither is an ancestor of the other.",
            writer_equations={kind: protocol["finite_equation"] for kind, protocol in protocols.items()},
            typed_ports={kind: protocol["port_types"] for kind, protocol in protocols.items()},
            port_interpretation="Sequential A/M are separately output-projected returned contributions; parallel A/M are features before one shared projection. The same letters do not equate energies or winning coalitions across families.",
            width_scope="Symbolic widths name declared original model slots; this compiler does not independently inspect or verify model tensor shapes.",
            held_endpoint_real_arithmetic={kind: protocol["held_endpoint_real_arithmetic"] for kind, protocol in protocols.items()},
            finite_B_conventions={kind: protocol["finite_B_convention"] for kind, protocol in protocols.items()},
            recipient_closures={kind: protocol["recipient_constants"] for kind, protocol in protocols.items()},
            clocks={kind: protocol["clock"] for kind, protocol in protocols.items()},
            streams={kind: protocol["stream"] for kind, protocol in protocols.items()},
            source_policies={kind: protocol["source_policy"] for kind, protocol in protocols.items()},
            continuations={kind: protocol["continuation"] for kind, protocol in protocols.items()}),
        selection_metrics=compare_metric_domains(sequential["selection_metric"], parallel["selection_metric"]))
    result = dict(schema=SCHEMA, compiler_scope="Hand-specified experimental grammar and source-metadata bookkeeping",
        protocols=protocols, comparison=comparison,
        limits=["The supplied DAGs are architecture descriptions, not inferred topology or discovered semantic ontology.",
            "Held-port construction and source-port formation are distinct: frozen M does not become a fresh recomputation.",
            "Real-arithmetic affine endpoint identities do not assert affine source formation, finite BF16 additivity or affine downstream continuation.",
            "A/M label different typed stages in the two protocols; energies and winning coalitions are not directly comparable from the shared letters.",
            "Only listed candidates are selected; source-copy anchors, whole-text Q-closure references and natural references remain controls.",
            "Historical scores are accepted from immutable rule metadata; this compiler does not re-audit underlying outcomes.",
            "No fresh outcomes, model calls, parameter reads, job/result reads, cross-family scalar ranking, quality or speed claim.",
            "Finite rounding, applicability, source policy, typed closure and native continuation still require their existing execution/audit evidence."])
    result["comparison_sha256"] = digest(result)
    return copied(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-rule", type=Path, required=True)
    parser.add_argument("--parallel-rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = dict(zip(KINDS, (args.sequential_rule, args.parallel_rule), strict=True))
    raw = {kind: path.read_bytes() for kind, path in files.items()}
    result = compile_protocols(*(json.loads(raw[kind]) for kind in KINDS), rule_sources={kind:
        dict(path=str(path), sha256=hashlib.sha256(raw[kind]).hexdigest()) for kind, path in files.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), comparison_sha256=result["comparison_sha256"],
        selected={kind: result["protocols"][kind]["historical_selection"]["selected_action"] for kind in KINDS}), sort_keys=True))


if __name__ == "__main__":
    main()
