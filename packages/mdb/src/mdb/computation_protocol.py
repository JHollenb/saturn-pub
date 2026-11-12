"""Typed, finite computation-family experiments without model-framework imports.

Operation graphs and capture adapters are supplied. This module compiles their
factorials, analyses conditional linearity, orders forecast/native execution,
and retains the result of each frozen effect predicate. It does not infer a
semantic ontology or execute an unregistered model adapter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

SPEC_SCHEMA = "mdb-computation-family-spec-v2"
PLAN_SCHEMA = "mdb-computation-family-program-v3"
LEGACY_PLAN_SCHEMA = "mdb-computation-family-program-v2"
RULE_SCHEMA = "mdb-computation-family-rule-v3"
LEGACY_RULE_SCHEMA = "mdb-computation-family-rule-v2"
STAGES = (
    "forecast_sources", "forecast_endpoints", "forecast_continuations", "forecast_decodes",
    "seal", "native_sources", "native_endpoints", "native_continuations", "native_decodes",
    "audit",
)
# Properties concern ideal real arithmetic. Literal finite kernels remain the
# numerical authority; no algebraic reassociation is emitted by this compiler.
OPERATORS = {
    "sdpa": {"arity": 3, "rule": "attention", "mixes": "token rows",
             "description": "softmax(Q K^T / sqrt(d)) V; affine in V with Q/K fixed"},
    "silu": {"arity": 1, "rule": "nonlinear", "mixes": "local channels",
             "description": "elementwise SiLU of a supplied gate preactivation"},
    "multiply": {"arity": 2, "rule": "product", "mixes": "local channels",
                 "description": "literal elementwise product, separately linear in each operand"},
    "add": {"arity": 2, "rule": "affine", "mixes": "declared tensor coordinates",
            "description": "literal same-shape addition"},
    "linear": {"arity": 1, "rule": "affine", "mixes": "feature channels",
               "description": "one original full projection with its supplied fixed parameters"},
    "identity": {"arity": 1, "rule": "affine", "mixes": "none",
                 "description": "held operand; no upstream recomputation"},
}


class ProtocolError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _bounded_int(value, low, high, name):
    require(type(value) is int and low <= value <= high, name + " outside finite bound")


def _name(value):
    return isinstance(value, str) and bool(value) and len(value) <= 160


def _shape(value):
    return isinstance(value, list) and bool(value) and all(
        (type(size) is int and size > 0) or (_name(size) and size.isidentifier())
        for size in value
    )


def validate_spec(spec):
    require(isinstance(spec, dict) and spec.get("schema") == SPEC_SCHEMA, "unknown specification schema")
    for field in ("family_id", "model_identity", "endpoint_adapter", "source_policy"):
        require(_name(spec.get(field)), "missing " + field)
    ports = spec.get("ports")
    require(isinstance(ports, dict) and 1 <= len(ports) <= 64, "bounded named ports required")
    for name, port in ports.items():
        require(_name(name) and isinstance(port, dict) and _shape(port.get("shape")), "invalid typed port")
        require(port.get("dtype") in {"torch.bfloat16", "torch.float32", "float", "fixture"}
                and _name(port.get("stage")), "port dtype and capture stage required")
    factors = spec.get("factors")
    require(isinstance(factors, list) and 1 <= len(factors) <= 6, "one to six factors required")
    ids, touched = [], set()
    for factor in factors:
        require(isinstance(factor, dict) and _name(factor.get("id")), "factor id required")
        names = factor.get("ports")
        require(isinstance(names, list) and names and len(names) == len(set(names))
                and all(name in ports for name in names), "factor references unknown or duplicate ports")
        require(factor["id"] not in ids and not touched.intersection(names), "factor aliases overlap")
        require(factor["id"] not in {"recipient", "source", "baseline", "both"}, "reserved factor id")
        ids.append(factor["id"])
        touched.update(names)
    nodes = spec.get("nodes")
    require(isinstance(nodes, list) and 1 <= len(nodes) <= 64, "bounded operation graph required")
    by_id = {}
    for node in nodes:
        require(isinstance(node, dict) and _name(node.get("id")) and node["id"] not in ports
                and node["id"] not in by_id, "duplicate graph name")
        require(node.get("op") in OPERATORS, "unregistered operation")
        require(isinstance(node.get("inputs"), list)
                and len(node["inputs"]) == OPERATORS[node["op"]]["arity"], "operation arity differs")
        require(isinstance(node.get("attributes", {}), dict), "operation attributes must be data")
        if node["op"] == "sdpa":
            require(node.get("attributes", {}) == {}, "SDPA profile is fixed by the registered native adapter")
        elif node["op"] in {"silu", "multiply", "add", "identity"}:
            require(node.get("attributes", {}) == {}, "primitive does not accept attributes")
        by_id[node["id"]] = node
    order, active, seen = [], set(), set(ports)

    def visit(name):
        require(name in ports or name in by_id, "unknown graph input: " + str(name))
        require(name not in active, "cyclic operation graph")
        if name in seen:
            return
        active.add(name)
        for parent in by_id[name]["inputs"]:
            require(isinstance(parent, str), "graph input names must be strings")
            visit(parent)
        active.remove(name)
        seen.add(name)
        order.append(name)

    for name in by_id:
        visit(name)
    require(spec.get("output") in by_id, "graph output must be a formed value")
    ancestry = set()

    def collect(name):
        ancestry.add(name)
        for parent in by_id.get(name, {}).get("inputs", []):
            if parent not in ancestry:
                collect(parent)

    collect(spec["output"])
    require(set(ports) <= ancestry and set(by_id) <= ancestry, "unused graph port or operation")
    _bounded_int(spec.get("max_contexts"), 1, 128, "context count")
    continuation = spec.get("continuation")
    require(isinstance(continuation, dict) and _name(continuation.get("consumer")), "native consumer required")
    _bounded_int(continuation.get("transitions"), 1, 10000, "continuation")
    if "execution_kind" in continuation:
        require(continuation["execution_kind"] == "autoregressive", "unknown explicit execution kind")
        require(continuation.get("vae_decodes", 0) == 0, "autoregressive token futures have no VAE")
    require(isinstance(spec.get("receiver"), dict) and bool(spec["receiver"])
            and all(source in {"P", "Q"} for source in spec["receiver"].values()), "explicit receiving closure required")
    require(isinstance(spec.get("clock"), dict) and bool(spec["clock"]), "capture clock required")
    metric = spec.get("metric")
    require(isinstance(metric, dict) and all(_name(metric.get(key))
            for key in ("observable", "units", "reference", "scope")), "metric domain required")
    require(metric.get("direction") == "minimize", "only declared minimization is supported")
    return order


def conditional_structure(spec):
    """Infer per-factor ideal linearity from registered operations, not family IDs."""
    order = validate_spec(spec)
    factors = [item["id"] for item in spec["factors"]]
    owners = {port: factor["id"] for factor in spec["factors"] for port in factor["ports"]}
    states = {port: {factor: (1 if owners.get(port) == factor else 0) for factor in factors}
              for port in spec["ports"]}
    zeros = {port: {factor: owners.get(port) == factor for factor in factors}
             for port in spec["ports"]}
    nodes = {node["id"]: node for node in spec["nodes"]}
    for name in order:
        node = nodes[name]
        parents = [states[parent] for parent in node["inputs"]]
        result, zero = {}, {}
        for factor in factors:
            values = [parent[factor] for parent in parents]
            rule = OPERATORS[node["op"]]["rule"]
            parent_zeros = [zeros[parent][factor] for parent in node["inputs"]]
            if rule == "nonlinear":
                result[factor] = None if values[0] != 0 else 0
            elif rule == "attention":
                result[factor] = None if values[0] != 0 or values[1] != 0 else values[2]
            elif rule == "product":
                result[factor] = None if None in values else sum(values)
            else:
                result[factor] = None if None in values else max(values)
            if rule == "attention":
                zero[factor] = parent_zeros[2]
            elif rule == "product":
                zero[factor] = any(parent_zeros)
            elif node["op"] == "linear":
                zero[factor] = False  # The original fixed projection can have a bias.
            else:
                zero[factor] = all(parent_zeros)
        states[name] = result
        zeros[name] = zero
    degrees = states[spec["output"]]
    homogeneous = zeros[spec["output"]]
    payloads = [factor for factor, degree in degrees.items() if degree == 1]
    conditioned = bool(payloads) and any(degree is None for degree in degrees.values())
    kind = "linear" if payloads and all(homogeneous[factor] for factor in payloads) else "affine"
    return {
        "scope": "Ideal real arithmetic of supplied graph; all other factors and parameters held fixed",
        "factor_linearity": {factor: ("independent" if degree == 0 else
                                      "linear" if degree == 1 and homogeneous[factor] else
                                      "affine" if degree == 1 else "not-proven-affine")
                             for factor, degree in degrees.items()},
        "operation_mixing": sorted({OPERATORS[node["op"]]["mixes"] for node in spec["nodes"]}),
        "shared_pattern": "state-conditioned " + kind + " action on payload" if conditioned
            else "supplied graph retains its explicit dependencies",
        "finite_reassociation_permitted": False,
    }


def specialize_spec(spec, capture):
    """Resolve declared operand views from source-qualified capture metadata.

    No array or parameter is opened. A chunk view binds its parent content hash
    and slice recipe; its child content hash must be checked at execution.
    """
    spec = copy.deepcopy(spec)
    validate_spec(spec)
    require(isinstance(capture, dict) and capture.get("schema") == "mdb-operation-port-capture-metadata-v1",
            "qualified capture metadata required")
    require(capture.get("model_identity") == spec["model_identity"] and capture.get("clock") == spec["clock"],
            "capture checkpoint or clock differs")
    require(isinstance(capture.get("provenance"), dict) and capture["provenance"], "capture provenance required")
    symbols, views = {}, {}
    for name, port in spec["ports"].items():
        binding = port.get("capture")
        require(isinstance(binding, dict) and _name(binding.get("port")), "explicit captured port required")
        descriptor = capture.get("ports", {}).get(binding["port"])
        require(isinstance(descriptor, dict) and _shape(descriptor.get("shape"))
                and all(type(size) is int for size in descriptor["shape"]), "concrete captured dimensions required")
        require(descriptor.get("dtype") == port["dtype"], "captured operand dtype differs")
        shape, stride = list(descriptor["shape"]), descriptor.get("stride")
        require(isinstance(stride, list) and len(stride) == len(shape)
                and all(type(value) is int and value > 0 for value in stride), "captured stride required")
        content = descriptor.get("content_sha256")
        require(isinstance(content, str) and len(content) == 64 and all(c in "0123456789abcdef" for c in content),
                "captured logical content hash required")
        offset = 0
        if "chunk" in binding:
            chunk = binding["chunk"]
            require(isinstance(chunk, dict) and type(chunk.get("axis")) is int, "chunk axis required")
            axis = chunk["axis"]
            require(-len(shape) <= axis < len(shape), "chunk axis outside rank")
            axis %= len(shape)
            parts, index = chunk.get("parts"), chunk.get("index")
            _bounded_int(parts, 1, 64, "chunk count")
            _bounded_int(index, 0, parts-1, "chunk index")
            require(shape[axis] % parts == 0, "captured dimension is not divisible into chunks")
            shape[axis] //= parts
            offset = index * shape[axis] * stride[axis]
        require(len(port["shape"]) == len(shape), "captured operand rank differs")
        for declared, actual in zip(port["shape"], shape):
            if isinstance(declared, str):
                require(declared not in symbols or symbols[declared] == actual, "captured symbolic dimensions disagree")
                symbols[declared] = actual
            else:
                require(declared == actual, "captured operand dimension differs")
        port["shape"] = shape
        views[name] = {"captured_port": binding["port"], "parent_content_sha256": content,
                       "parent_shape": descriptor["shape"], "view_shape": shape, "view_stride": stride,
                       "relative_storage_offset_elements": offset,
                       "child_content_scope": "Parent hash and declared view only; child bytes checked by execution adapter"}
    spec["specialization"] = {"capture_metadata_sha256": digest(capture), "symbols": symbols,
                              "views": views, "provenance": capture["provenance"], "model_calls": 0}
    return spec


def execution_geometry(spec, actions):
    """Generate logical branch work from actions and the declared native horizon."""
    fixed = [action for action in actions if action["role"] != "natural-reference"]
    sources = {action["source"] for action in actions if action["role"] == "natural-reference"}
    nodes = {}
    for node in spec["nodes"]:
        nodes[node["op"]] = nodes.get(node["op"], 0) + len(fixed)
    if spec["continuation"].get("execution_kind") == "autoregressive":
        choices = spec["continuation"]["transitions"]
        forwards = len(actions) * choices
        return {"per_context_per_lane": {"source_branches": len(sources),
            "primitive_endpoints": len(fixed), "futures": len(actions),
            "prefill_forwards": len(actions), "cached_forwards": len(actions) * (choices - 1),
            "physical_forwards": forwards, "greedy_choices": forwards,
            "VAE_decodes": 0, "common_parent_Acts": 0,
            "module_hook_interventions": len(fixed),
            "declared_graph_node_invocations": nodes},
            "aggregate_work_items_per_context_both_lanes": 2 * forwards,
            "max_contexts": spec["max_contexts"],
            "scope": "Autoregressive adapter: one prefill and choices-minus-one cached forwards per own-state future; hook interventions are not public Acts",
            "source_unit": "Natural source ports are captured during their already-counted prefill forwards"}
    vae = spec["continuation"].get("vae_decodes", 1)
    _bounded_int(vae, 0, 16, "decoder calls")
    per_context = {"source_branches": len(sources), "primitive_endpoints": len(fixed),
        "futures": len(actions), "suffix_transitions": len(actions)*spec["continuation"]["transitions"],
        "VAE_decodes": len(actions)*vae,
        "common_parent_Acts": sum(not (action["role"] == "natural-reference" and action["source"] == "P") for action in actions),
        "declared_graph_node_invocations": nodes}
    if "transformer_blocks" in spec["continuation"]:
        blocks = spec["continuation"]["transformer_blocks"]
        _bounded_int(blocks, 0, spec["continuation"]["transitions"], "continuation transformer blocks")
        per_context["native_transformer_blocks"] = len(sources)+len(actions)*blocks
    return {"per_context_per_lane": per_context,
        "aggregate_work_items_per_context_both_lanes": 2*(len(sources)+len(fixed)+len(actions)+len(actions)*vae),
        "max_contexts": spec["max_contexts"],
        "scope": "Declared adapter units; graph-node callbacks are distinct from internal native module/kernel calls",
        "source_unit": "one declared source capture/formation per natural source"}


def compile_protocol(spec):
    spec = copy.deepcopy(spec)
    order = validate_spec(spec)
    if spec.get("requires_specialization"):
        require(isinstance(spec.get("specialization"), dict)
                and all(type(size) is int for port in spec["ports"].values() for size in port["shape"]),
                "capture specialization required before compilation")
    factors = [factor["id"] for factor in spec["factors"]]
    actions = [
        {"id": "recipient", "role": "natural-reference", "source": "Q", "bindings": {}},
        {"id": "source", "role": "natural-reference", "source": "P", "bindings": {}},
    ]
    for sources in itertools.product(("Q", "P"), repeat=len(factors)):
        chosen = [name for name, source in zip(factors, sources) if source == "P"]
        if not chosen:
            name, role = "baseline", "baseline"
        elif len(chosen) == len(factors):
            name, role = "both", "component-reference"
        else:
            name, role = "+".join(chosen), "candidate"
        actions.append({"id": name, "role": role, "bindings": dict(zip(factors, sources))})
    menu = [action["id"] for action in actions if action["role"] == "candidate"]
    require(menu, "at least two factors are needed for a nontrivial candidate menu")
    _bounded_int(len(actions) * spec["max_contexts"] * spec["continuation"]["transitions"],
                 1, 1_000_000, "total transition budget")
    commands = [{"operation": stage, "actions": [action["id"] for action in actions],
                 "requires_seal": stage.startswith("native_") or stage == "audit"}
                for stage in STAGES]
    plan = {"schema": PLAN_SCHEMA, "specification": spec, "graph_order": order,
            "actions": actions, "candidate_menu": menu, "program": commands,
            "structure": conditional_structure(spec),
            "execution_geometry": execution_geometry(spec, actions),
            "audit_obligations": ["checkpoint/input/source identity", "source and receiver closure",
                "actual primitive arguments and returns", "original finite operation geometry",
                "global forecast barrier before native targets", "common-parent Act and no-op replay",
                "complete retained array comparisons", "native continuation and original decoder",
                "frozen effects with signed margins and counterexamples"],
            "limits": ["Graph, ports and adapter are supplied, not discovered from model traces",
                "Exact finite forecasts pay original parameters and primitive kernels",
                "Structural linearity is conditional ideal algebra, not finite additivity or semantic meaning"]}
    plan["plan_sha256"] = digest(plan)
    return plan


def verify_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") in {PLAN_SCHEMA, LEGACY_PLAN_SCHEMA}, "unknown compiled plan")
    expected = compile_protocol(plan["specification"])
    if plan["schema"] == LEGACY_PLAN_SCHEMA:
        expected.pop("execution_geometry")
        expected["schema"] = LEGACY_PLAN_SCHEMA
        expected.pop("plan_sha256")
        expected["plan_sha256"] = digest(expected)
    require(plan == expected, "compiled plan differs from its specification")
    return plan


def run_graph(plan, sources, bindings, kernels, *, check=None):
    """Execute the generated graph using a caller's qualified primitive kernels.

    Tensor objects remain opaque here. A model adapter checks shape, dtype,
    layout and checkpoint contracts through ``check`` and the supplied kernels.
    """
    verify_plan(plan)
    spec = plan["specification"]
    names = {factor["id"] for factor in spec["factors"]}
    require(set(bindings) == names and all(source in {"P", "Q"} for source in bindings.values()),
            "factor binding census differs")
    owners = {port: factor["id"] for factor in spec["factors"] for port in factor["ports"]}
    values = {}
    for name, port in spec["ports"].items():
        source = bindings[owners[name]] if name in owners else "Q"
        require(source in sources and name in sources[source], "missing supplied operand: " + name)
        value = sources[source][name]
        if check:
            check(name, value, port)
        values[name] = value
    by_id = {node["id"]: node for node in spec["nodes"]}
    for name in plan["graph_order"]:
        node = by_id[name]
        require(node["op"] in kernels and callable(kernels[node["op"]]), "missing qualified kernel")
        values[name] = kernels[node["op"]](*(values[parent] for parent in node["inputs"]),
                                         **node.get("attributes", {}))
    return values[spec["output"]]


def execute_program(plan, handlers, *, record=None):
    """Run the generated phase order; native callbacks require a returned seal.

    The model adapter must enforce the same barrier at actual module entry.
    Callback results are evidence, including scientific misses; this controller
    never aggregates scientific pass/fail fields into an execution gate.
    """
    verify_plan(plan)
    require(set(handlers) == set(STAGES) and all(callable(value) for value in handlers.values()),
            "execution handler census differs")
    receipts, seal = [], None
    for number, command in enumerate(plan["program"]):
        if command["requires_seal"]:
            require(isinstance(seal, str) and len(seal) == 64, "native release lacks forecast seal")
        result = handlers[command["operation"]](copy.deepcopy(command))
        require(isinstance(result, dict), "execution callback must return a receipt")
        if command["operation"] == "seal":
            seal = result.get("seal_sha256")
            require(isinstance(seal, str) and len(seal) == 64
                    and result.get("native_calls_before_seal") == 0, "invalid forecast release receipt")
        row = {"number": number, "operation": command["operation"],
               "plan_sha256": plan["plan_sha256"], "result": result}
        canonical(row)
        receipts.append(row)
        if record:
            record(copy.deepcopy(row))
    return {"schema": "mdb-computation-family-program-receipt-v2", "plan_sha256": plan["plan_sha256"],
            "seal_sha256": seal, "commands": receipts}


def _measurements(plan, packet, *, expected_contexts=None):
    verify_plan(plan)
    require(isinstance(packet, dict) and packet.get("plan_sha256") == plan["plan_sha256"]
            and packet.get("model_identity") == plan["specification"]["model_identity"]
            and packet.get("metric") == plan["specification"]["metric"], "measurement identity or metric differs")
    require(isinstance(packet.get("provenance"), dict) and packet["provenance"], "measurement provenance required")
    rows = packet.get("rows")
    require(isinstance(rows, list) and rows, "measurement rows required")
    actions = {action["id"] for action in plan["actions"]}
    contexts = {}
    for row in rows:
        require(isinstance(row, dict) and _name(row.get("context")) and row.get("action") in actions,
                "measurement row identity differs")
        context = contexts.setdefault(row["context"], {})
        require(row["action"] not in context, "duplicate measurement")
        value = row.get("value")
        require(value is None or (type(value) in {int, float} and math.isfinite(value)), "nonfinite measurement")
        context[row["action"]] = value
    require(len(contexts) <= plan["specification"]["max_contexts"]
            and all(set(values) == actions for values in contexts.values()), "incomplete action census")
    if expected_contexts is not None:
        require(set(contexts) == set(expected_contexts), "frozen fresh-context census differs")
    if "contexts" in packet:
        declared = _context_rows(packet["contexts"])
        require(set(declared) == set(contexts), "measurement/context metadata census differs")
    return contexts


def _context_rows(rows):
    require(isinstance(rows, list) and rows, "full prompt/seed context metadata required")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and _name(row.get("id")) and row["id"] not in result,
                "unique full context ids required")
        seed, prompts = row.get("seed"), row.get("prompts")
        require(type(seed) is int and -(2**63) <= seed < 2**64 and isinstance(prompts, dict)
                and set(prompts) == {"P", "Q"} and all(_name(value) or (isinstance(value, str) and value)
                for value in prompts.values()), "exact prompt/seed metadata required")
        identity = {"seed": seed % (2**64), "prompts": dict(prompts)}
        pair = {"seed": identity["seed"], "prompts": sorted(prompts.values())}
        result[row["id"]] = {"context_sha256": digest(identity), "state_pair_sha256": digest(pair)}
    return result


def freeze_rule(plan, development, fresh_contexts):
    contexts = _measurements(plan, development)
    require(isinstance(fresh_contexts, list) and fresh_contexts, "named fresh contexts required")
    bound = isinstance(fresh_contexts[0], dict)
    fresh_metadata = _context_rows(fresh_contexts) if bound else {}
    names = list(fresh_metadata) if bound else fresh_contexts
    require(len(names) == len(set(names)) and all(_name(name) for name in names), "named fresh contexts required")
    require(not set(contexts).intersection(names), "development and fresh contexts overlap")
    require(len(names) <= plan["specification"]["max_contexts"], "fresh contexts exceed bound")
    development_metadata = _context_rows(development.get("contexts")) if bound else {}
    if bound:
        require(set(development_metadata) == set(contexts), "development context binding census differs")
        require(not {row["state_pair_sha256"] for row in development_metadata.values()}.intersection(
            row["state_pair_sha256"] for row in fresh_metadata.values()), "development prompt/seed pair reused under a fresh label")
    menu = plan["candidate_menu"]
    require(all(values[name] is not None for values in contexts.values() for name in menu),
            "undefined development axis cannot select a candidate")
    means = {name: sum(values[name] for values in contexts.values()) / len(contexts) for name in menu}
    selected = min(menu, key=lambda name: (means[name], menu.index(name)))
    predicates = [{"left": selected, "relation": "lt", "right": name} for name in menu if name != selected]
    rule = {"schema": RULE_SCHEMA, "plan_sha256": plan["plan_sha256"],
            "model_identity": plan["specification"]["model_identity"], "metric": plan["specification"]["metric"],
            "development_contexts": list(contexts), "development_sha256": digest(development),
            "development_provenance": development["provenance"], "mean_scores": means,
            "candidate_menu_and_tie_order": menu, "excluded_roles": ["baseline", "natural-reference", "component-reference"],
            "selected_action": selected, "fresh_contexts": list(names), "predicates": predicates,
            "context_identity_policy": "exact-prompt-seed" if bound else "context-names-only",
            "development_context_metadata": development_metadata, "fresh_context_metadata": fresh_metadata,
            "selection_rule": "minimum mean supplied development score among nontrivial candidates; declared menu breaks exact ties",
            "fresh_outcomes_used": False}
    rule["rule_sha256"] = digest(rule)
    return rule


def evaluate_rule(plan, rule, fresh):
    verify_plan(plan)
    require(rule.get("schema") in {RULE_SCHEMA, LEGACY_RULE_SCHEMA}
            and rule.get("plan_sha256") == plan["plan_sha256"], "rule/plan mismatch")
    require(rule.get("rule_sha256") == digest({key: value for key, value in rule.items() if key != "rule_sha256"}),
            "frozen rule digest differs")
    require(rule.get("model_identity") == plan["specification"]["model_identity"]
            and rule.get("metric") == plan["specification"]["metric"], "frozen rule identity differs")
    require(set(rule.get("candidate_menu_and_tie_order", [])) == set(plan["candidate_menu"])
            and rule.get("selected_action") in plan["candidate_menu"], "frozen candidate menu differs")
    expected_predicates = [{"left": rule["selected_action"], "relation": "lt", "right": name}
                           for name in plan["candidate_menu"] if name != rule["selected_action"]]
    require(rule.get("predicates") == expected_predicates, "frozen predicate census differs")
    contexts = _measurements(plan, fresh, expected_contexts=rule["fresh_contexts"])
    if rule.get("context_identity_policy") == "exact-prompt-seed":
        require(_context_rows(fresh.get("contexts")) == rule["fresh_context_metadata"], "frozen prompt/seed binding differs")
    require(not set(contexts).intersection(rule["development_contexts"]), "held-out context was used in development")
    outcomes = []
    for context, values in contexts.items():
        for predicate in rule["predicates"]:
            left, right = values[predicate["left"]], values[predicate["right"]]
            margin = None if left is None or right is None else right - left
            outcomes.append({"context": context, **predicate, "left_value": left, "right_value": right,
                             "signed_margin": margin, "status": "not-applicable" if margin is None else
                             "held" if margin > 0 else "counterexample"})
    return {"schema": "mdb-computation-family-effects-v2", "plan_sha256": plan["plan_sha256"],
            "rule_sha256": rule["rule_sha256"], "measurement_sha256": digest(fresh), "outcomes": outcomes,
            "counts": {status: sum(row["status"] == status for row in outcomes)
                       for status in ("held", "counterexample", "not-applicable")},
            "status_scope": "Frozen consumer effects only; numerical agreement and semantic interpretation are separate"}


def compare_protocols(plans):
    require(len(plans) >= 2, "two protocols required")
    for plan in plans:
        verify_plan(plan)
    return {"schema": "mdb-computation-family-comparison-v2", "families": [{
        "family_id": plan["specification"]["family_id"], "plan_sha256": plan["plan_sha256"],
        "structure": plan["structure"], "receiver": plan["specification"]["receiver"],
        "clock": plan["specification"]["clock"], "metric": plan["specification"]["metric"],
        "candidate_menu": plan["candidate_menu"], "operation_graph": plan["specification"]["nodes"],
    } for plan in plans], "shared_execution_program": list(STAGES),
        "shared_audit_obligations": sorted(set.intersection(*(set(plan["audit_obligations"]) for plan in plans))),
        "cross_family_score_ranking": False,
        "scope": "Supplied typed graphs and registered operation properties; no inferred semantic dictionary"}


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compile_cmd = sub.add_parser("compile")
    compile_cmd.add_argument("specification")
    compile_cmd.add_argument("--out", required=True)
    specialize = sub.add_parser("specialize")
    specialize.add_argument("specification")
    specialize.add_argument("capture")
    specialize.add_argument("--out", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("plan")
    freeze.add_argument("development")
    freeze.add_argument("--fresh-contexts", required=True)
    freeze.add_argument("--out", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("plan")
    evaluate.add_argument("rule")
    evaluate.add_argument("measurements")
    evaluate.add_argument("--out", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("plans", nargs="+")
    compare.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "compile":
        result = compile_protocol(_read(args.specification))
    elif args.command == "specialize":
        result = specialize_spec(_read(args.specification), _read(args.capture))
    elif args.command == "freeze":
        result = freeze_rule(_read(args.plan), _read(args.development), _read(args.fresh_contexts))
    elif args.command == "evaluate":
        result = evaluate_rule(_read(args.plan), _read(args.rule), _read(args.measurements))
    else:
        result = compare_protocols([_read(path) for path in args.plans])
    _write(args.out, result)
    print(json.dumps({"output": args.out, "schema": result["schema"], "sha256": digest(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
