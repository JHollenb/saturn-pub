"""Summarize authenticated compact family evidence without model/array reads.

Comparison counts describe dependent retained field pairs, not independent
scientific samples. Missing old-reader coverage stays unknown. The reporter
neither selects a candidate from fresh outcomes nor changes a frozen rule.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "mdb-flux-primitive-family-summary-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=nonfinite)


def data(value):
    return read(value) if isinstance(value, (str, Path)) else copy.deepcopy(value)


class Compact:
    """Verify local compact bytes against the supplied collection receipt.

    Remote member ledgers are metadata only. A compact directory without a
    collection receipt is usable with explicit unverified-custody status.
    """
    def __init__(self, path, kind):
        path = Path(path).resolve()
        self.root = path / "compact" if (path / "compact").is_dir() else path
        require(self.root.is_dir(), "compact directory absent")
        self.run = self.root.parent
        self.collection = read(self.run / "collection.json") if (self.run / "collection.json").is_file() else None
        self.kind, self.loaded = kind, {}
        actual = {}
        for item in self.root.rglob("*"):
            require(not item.is_symlink(), "compact contains a symbolic link")
            if not item.is_file():
                continue
            require(item.suffix not in {".npy", ".npyb", ".safetensors", ".whl"}, "compact reporter refuses tensor/weight payloads")
            name = item.relative_to(self.root).as_posix()
            actual[name] = dict(sha256=sha(item), bytes=item.stat().st_size)
        self.local = actual
        if self.collection is not None:
            receipt = self.collection
            require(receipt["kind"] == kind and receipt["inventory"]["files"] == actual, "compact collection bytes differ")
            custody = receipt.get("payload_custody")
            if custody is not None:
                require(custody.get("required") is True and all(custody[name]["sha256"] == custody["sealed"]["sha256"]
                    for name in ("declared", "executed")), "collection payload custody differs")

    def json(self, name, *, optional=False):
        require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe compact metadata path")
        if name not in self.local:
            require(optional, "compact metadata absent: " + name)
            return None
        require(name.endswith(".json"), "only JSON metadata can be decoded")
        self.loaded[name] = self.local[name]
        return read(self.root / name)

    def storage(self):
        receipt = self.collection
        inventory = {} if receipt is None else receipt["inventory"]
        return dict(compact_bytes=sum(row["bytes"] for row in self.local.values()), compact_files=len(self.local),
            raw_bytes=inventory.get("rawBytes"), raw_files=inventory.get("rawFileCount"),
            raw_retained_on_durable_volume=None if receipt is None else receipt.get("raw_retained_on_durable_volume"),
            raw_byte_scope="collection inventory rawBytes; publisher manifest convention retained",
            collection_path=None if receipt is None else str(self.run / "collection.json"),
            collection_sha256=None if receipt is None else sha(self.run / "collection.json"),
            custody_status="not-supplied" if receipt is None else "compact-and-receipt-verified",
            remote_root=inventory.get("root"))


def request_identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def expected_pairs(forecast, native, field):
    require(set(forecast) == set(native), "writer context census differs")
    result = {}
    for cid in native:
        left, right = forecast[cid][field], native[cid][field]
        require(set(left) == set(right), "writer table census differs")
        for name in right:
            if field == "sources":
                a, b = left[name]["ports"], right[name]["ports"]
                require(set(a) == set(b), "retained source port census differs")
                for port, record in b.items():
                    result[(cid, name, port)] = math.prod(record["shape"])
            else:
                result[(cid, name)] = math.prod(right[name]["shape"])
    return result


def comparison_account(expected, observed):
    """Keep row metrics and unknown mismatch counts rather than inventing zeros."""
    rows, misses, unknown = [], [], []
    elements = exact = numeric_changed = signed_zero = raw_elements = raw_bits = 0
    unknown_raw_elements = unknown_raw_bits = 0
    for key, size in expected.items():
        value = observed.get(key)
        label = list(key)
        if value is None:
            unknown.append(label)
            continue
        require(value.get("elements", size) == size, "comparison element census differs")
        require(type(value.get("raw_bit_exact")) is bool, "comparison raw-bit coverage absent")
        record = dict(address=label, metrics=value)
        rows.append(record)
        elements += size
        exact += int(value["raw_bit_exact"])
        numeric_changed += int(value.get("changed", 0))
        signed_zero += int(value.get("zero_sign_mismatches", 0))
        if not value["raw_bit_exact"]:
            misses.append(record)
        count = value.get("raw_bit_mismatch_elements", 0 if value["raw_bit_exact"] else None)
        bits = value.get("raw_bit_mismatch_bits", 0 if value["raw_bit_exact"] else None)
        if count is None:
            unknown_raw_elements += 1
        else:
            raw_elements += count
        if bits is None:
            unknown_raw_bits += 1
        else:
            raw_bits += bits
    extras = [list(key) for key in observed if key not in expected]
    return dict(status="complete" if not unknown and not extras else "incomplete", expected_array_pairs=len(expected),
        compared_array_pairs=len(rows), compared_elements=elements, raw_bit_exact_pairs=exact, raw_bit_miss_pairs=len(misses),
        numeric_changed_elements=numeric_changed, zero_sign_mismatches=signed_zero,
        known_raw_bit_mismatch_elements=raw_elements, known_raw_bit_mismatch_bits=raw_bits,
        miss_pairs_without_element_count=unknown_raw_elements, miss_pairs_without_bit_count=unknown_raw_bits,
        missing=unknown, unexpected=extras, rows=rows, raw_bit_misses=misses)


def geometry_account(plan, contexts, summary, native_contract, audit, index, program):
    actual = summary["counts"]
    require(actual == native_contract["counts"] == audit["native_counts"], "native/audit physical counters differ")
    size = len(contexts)
    geometry = plan.get("execution_geometry")
    result = dict(status="legacy-plan-without-generated-geometry" if geometry is None else "generated-geometry-compared",
        generated=geometry, actual_counts=actual, joins=[], missing_direct_counters=[],
        action_case_census=dict(expected=size*len(plan["actions"]), actual=len(index)),
        boundary_row_census=dict(expected=size*len(plan["actions"])*(plan["specification"]["continuation"]["transitions"]+1),
            actual=audit.get("boundary_counts", {}).get("paired_rows")),
        program_operations=[row["operation"] for row in program["commands"]], observed_native_phases=native_contract.get("observed", {}))
    if geometry is None:
        result["diagnostic"] = "No generated physical census was attached to this legacy plan; actual counters are preserved without inventing one."
        return result
    mappings = {
        "source_branches": ("functional_source_blocks", "native_source_blocks"),
        "primitive_endpoints": ("functional_primitive_calls", "native_primitive_calls", "functional_hybrid_projections", "native_hybrid_projections"),
        "suffix_transitions": ("functional_transitions", "native_target_transitions"),
        "VAE_decodes": ("functional_vae", "native_vae"),
        "common_parent_Acts": ("forecast_acts", "native_acts"),
        "native_transformer_blocks": ("native_blocks",),
    }
    for name, value in geometry["per_context_per_lane"].items():
        if name == "futures":
            result["joins"].append(dict(generated_field=name, expected=value*size, observed=len(index), equal=value*size == len(index)))
        elif name in mappings:
            for field in mappings[name]:
                observed = actual.get(field)
                result["joins"].append(dict(generated_field=name, actual_field=field, expected=value*size,
                    observed=observed, equal=None if observed is None else value*size == observed))
        else:
            result["missing_direct_counters"].append(dict(field=name, declared_per_context=value,
                diagnostic="Declared graph callbacks are distinct from internal native kernel/module calls; no generic node counter was emitted."))
    return result


def rule_account(plan, rule, audit, measurements):
    from mdb_lanes.computation_protocol import digest, evaluate_rule

    effect = audit.get("frozen_effect_evaluation")
    if effect is not None:
        require(effect["plan_sha256"] == plan["plan_sha256"] and (measurements is None
            or effect["measurement_sha256"] == digest(measurements)), "saved effect/measurement identity differs")
    if rule is None:
        return dict(status="saved-effect-evaluation" if effect is not None else "no-frozen-fresh-rule", evaluation=effect,
            counterexamples=[] if effect is None else [row for row in effect["outcomes"] if row["status"] == "counterexample"])
    require(rule.get("plan_sha256") == plan["plan_sha256"] and rule.get("rule_sha256")
        == digest({key: value for key, value in rule.items() if key != "rule_sha256"}), "frozen rule identity differs")
    if set(audit["contexts"]) != set(rule["fresh_contexts"]):
        return dict(status="not-the-frozen-fresh-cohort", rule=rule, evaluation=None, counterexamples=[],
            diagnostic="Development or partial cohort; no held-out predicate evaluated.")
    require(measurements is not None, "frozen fresh evaluation requires saved measurements")
    current = evaluate_rule(plan, rule, measurements)
    require(effect is None or current == effect, "saved/fresh rule evaluations differ")
    return dict(status="frozen-rule-evaluated", rule=rule, evaluation=current,
        counterexamples=[row for row in current["outcomes"] if row["status"] == "counterexample"])


def summarize(native_run, audit_run, plan, *, rule=None):
    from mdb_lanes.computation_protocol import verify_plan

    plan = verify_plan(data(plan))
    rule = None if rule is None else data(rule)
    native, saved = Compact(native_run, "native"), Compact(audit_run, "audit")
    summary, config, request, contract, program, index, runtime, provenance = (native.json(name) for name in
        ("summary.json", "plan.json", "request.json", "native-contract.json", "program.json", "state-index.json", "runtime-contract.json", "source-provenance.json"))
    audit = saved.json("audit.json")
    measurements = saved.json("measurements.json", optional=True)
    require(config["compiled_protocol"] == plan and summary["protocol_sha256"] == audit["protocol_sha256"] == plan["plan_sha256"], "compiled plan binding differs")
    require(request_identity(request) == request["request_sha256"] == summary["request_sha256"] == audit["native_request_sha256"]
        and summary["job_id"] == audit["native_job_id"], "native/audit request binding differs")
    require({key: value for key, value in config.items() if key not in {"job_id", "request_sha256", "output"}} == request["config"], "native request/config differs")
    if native.collection is not None:
        publication = native.collection["publication"]
        require(publication["job_id"] == summary["job_id"] and publication["request_sha256"] == summary["request_sha256"]
            and sha(native.root / "files.json") == publication["files_manifest_sha256"] == audit["native_manifest_sha256"], "native publication binding differs")
    if saved.collection is not None:
        publication = saved.collection["publication"]
        require(publication["source_job"] == summary["job_id"] and publication["audit_sha256"] == sha(saved.root / "audit.json"), "audit publication binding differs")
    actions = plan["actions"]
    by_id = {row["id"]: row for row in actions}
    contexts = {row["id"]: row for row in config["contexts"]}
    require(set(contexts) == set(audit["contexts"]) and config["arms"] == list(by_id), "context/action cohort differs")
    expected_cases = {cid + "/" + aid for cid in contexts for aid in by_id}
    require(set(index) == set(audit["cases"]) == expected_cases, "future case census differs")
    writers = {lane: native.json(lane + "-source-writers.json") for lane in ("forecast", "native")}
    source_expected = expected_pairs(writers["forecast"], writers["native"], "sources")
    source_observed = {(cid, role, port): value for cid, roles in audit["writer_checks"].items()
        for role, row in roles.items() for port, value in row["comparisons"].items()}
    endpoint_expected = expected_pairs(writers["forecast"], writers["native"], "endpoints")
    endpoint_observed = {(cid, aid): value for cid, rows in audit["endpoint_checks"].items() for aid, value in rows.items()}
    returns_expected = {name: {} for name in ("primitive_return", "projection_return")}
    returns_observed = {name: {} for name in returns_expected}
    for cid in contexts:
        for action in actions:
            aid = action["id"]
            if action["role"] == "natural-reference":
                continue
            for name in returns_expected:
                record = writers["native"][cid]["constructions"][aid]["ports"][name]
                returns_expected[name][cid, aid] = math.prod(record["shape"])
                value = audit.get("construction_return_comparisons", {}).get(cid, {}).get(aid, {}).get(name)
                if value is not None:
                    returns_observed[name][cid, aid] = value
    consumers = native.json("consumer-ports.json")
    consumer_expected, consumer_observed, future_expected, future_observed, decoder_inputs_expected, decoder_inputs_observed = {}, {}, {}, {}, {}, {}
    png_rows = []
    for key, case in audit["cases"].items():
        require(set(consumers["forecast"][key]["ports"]) == set(consumers["native"][key]["ports"]), "consumer retained census differs")
        for name, record in consumers["native"][key]["ports"].items():
            consumer_expected[key, name] = math.prod(record["shape"])
        for name, value in case["consumer_ports"].items():
            consumer_observed[key, name] = value
        continuation = audit["equation_reports"][key]["continuation"]["fields"]
        for name, value in continuation.items():
            future_expected[key, name] = math.prod(value["shape"])
        for name, value in case["decoder"].items():
            if name == "decoder_input":
                decoder_inputs_expected[key, name] = value["elements"]
                decoder_inputs_observed[key, name] = value
            else:
                future_observed[key, name] = value
        png_sources = {lane: audit.get("images", {}).get(key + "/" + lane) for lane in ("forecast", "native")}
        if all(row is not None for row in png_sources.values()):
            exact = png_sources["forecast"]["sha256"] == png_sources["native"]["sha256"]
            require(case.get("png_byte_exact") == exact, "PNG hashes/audit comparison differ")
        else:
            exact = case.get("png_byte_exact")
        png_rows.append(dict(case=key, byte_exact=exact, image_records=png_sources))
    comparisons = dict(source=comparison_account(source_expected, source_observed),
        primitive=comparison_account(returns_expected["primitive_return"], returns_observed["primitive_return"]),
        projection=comparison_account(returns_expected["projection_return"], returns_observed["projection_return"]),
        endpoint=comparison_account(endpoint_expected, endpoint_observed), consumer=comparison_account(consumer_expected, consumer_observed),
        future=comparison_account(future_expected, future_observed), decoder_input=comparison_account(decoder_inputs_expected, decoder_inputs_observed))
    png = dict(expected_pairs=len(index), compared_pairs=sum(type(row["byte_exact"]) is bool for row in png_rows),
        exact_pairs=sum(row["byte_exact"] is True for row in png_rows), misses=[row for row in png_rows if row["byte_exact"] is False],
        missing=[row["case"] for row in png_rows if type(row["byte_exact"]) is not bool], rows=png_rows)
    rows, interactions = [], {}
    for cid, fields in audit["observations"].items():
        require(cid in contexts, "observation outside declared context")
        interactions[cid] = dict(primitive=audit.get("native_primitive_interactions", {}).get(cid),
            endpoint={lane: audit["construction_checks"][cid][lane].get("mixed_remainder") for lane in ("forecast", "native")},
            continuation={name: field.get("mixed_remainder") for name, field in fields.items()})
        primitive_j = interactions[cid]["primitive"]
        if primitive_j is not None:
            record = primitive_j["array"]
            ledger = {} if saved.collection is None else saved.collection["inventory"].get("retained_remote_members", {})
            member = ledger.get(record["path"])
            require(member is None or all(member[key] == record[key] for key in ("bytes", "sha256")), "remote J ledger binding differs")
            interactions[cid]["primitive_remote_ledger_status"] = "not-supplied" if member is None else "hash-and-bytes-joined"
        else:
            interactions[cid]["primitive_remote_ledger_status"] = "not-retained-by-this-reader"
        for observable, field in fields.items():
            require(set(field["rows"]) == set(by_id), "signed-axis action census differs")
            for aid, value in field["rows"].items():
                rows.append(dict(context=cid, context_metadata=contexts[cid], action=aid, role=by_id[aid]["role"],
                    observable=observable, axis_status=field["status"], natural_axis=dict(source=field["source"], recipient=field["recipient"]),
                    metrics=value, component_axis_diagnostic=field.get("component_axis_diagnostic", {}).get("rows", {}).get(aid)))
    incomplete = [name for name, account in comparisons.items() if account["status"] != "complete"]
    effects = rule_account(plan, rule, audit, measurements)
    result = dict(schema=SCHEMA, family_id=plan["specification"]["family_id"], protocol_sha256=plan["plan_sha256"],
        model_identity=plan["specification"]["model_identity"], family=config["family"], contexts=config["contexts"],
        actions=[dict(**action, candidate_for_selection=action["id"] in plan["candidate_menu"],
            control_or_reference=action["role"] != "candidate") for action in actions], candidate_menu_and_tie_order=plan["candidate_menu"],
        scope=dict(source_policy=plan["specification"]["source_policy"], observed_native_inputs=True, raw_prompt_prediction=False,
            receiver=plan["specification"]["receiver"], clock=plan["specification"]["clock"], continuation=plan["specification"]["continuation"],
            metric=plan["specification"]["metric"], runtime=runtime,
            source_profiles={cid: {role: source.get("profile") for role, source in row["sources"].items()}
                for cid, row in writers["native"].items()},
            own_state_forecasts="Original-weight equations/shared primitives; no intermediate native refresh",
            off_axis_scope="Orthogonal displacement is descriptive collateral, not a semantic evaluator"),
        provenance=dict(native_job_id=summary["job_id"], native_request_sha256=summary["request_sha256"],
            native_manifest_sha256=audit["native_manifest_sha256"], forecast_seal=summary["forecast_seal"],
            transaction=audit["transaction"], source_transaction=audit.get("source_transaction"), source=provenance,
            audited_proof_counts={name: audit.get(name) for name in ("published_files_verified", "prefix_files_verified",
                "external_sealed_files", "unique_arrays_read", "repeated_array_elements_read")},
            source_prefix_physical_counts=audit.get("source_prefix_counts"),
            source_prefix_count_scope="Prefix identity/seal is audited; physical prefix counters remain unavailable when not exported in this compact audit. No count is inferred from a renamed context or selected subset.",
            source_hashes=request.get("source_hashes"), runtime_wheels=request.get("runtime_wheels"),
            native_loaded_metadata=native.loaded, audit_loaded_metadata=saved.loaded),
        storage=dict(native=native.storage(), audit=saved.storage()),
        comparisons=comparisons, png=png,
        comparison_totals=dict(expected_array_pairs=sum(row["expected_array_pairs"] for row in comparisons.values()),
            compared_array_pairs=sum(row["compared_array_pairs"] for row in comparisons.values()),
            compared_elements=sum(row["compared_elements"] for row in comparisons.values()),
            raw_bit_miss_pairs=sum(row["raw_bit_miss_pairs"] for row in comparisons.values()),
            zero_sign_mismatches=sum(row["zero_sign_mismatches"] for row in comparisons.values()),
            incomplete_groups=incomplete, dependent_field_pairs_are_not_independent_samples=True),
        geometry=geometry_account(plan, contexts, summary, contract, audit, index, program),
        per_context_action_rows=rows, mixed_interactions=interactions, frozen_effects=effects,
        statuses=dict(mechanics="compact identity/custody joins verified" if native.collection is not None and saved.collection is not None
                else "native/audit metadata identities joined; collection custody not fully supplied",
            native_mechanics=summary.get("mechanics_status"), audit_mechanics=audit.get("mechanics_status"),
            evidence="incomplete-comparison-coverage" if incomplete or png["missing"] else "all-declared-retained-pairs-accounted",
            evaluated_frozen_predicates=0 if effects["evaluation"] is None else len(effects["evaluation"]["outcomes"]),
            held_counterexamples=None if effects["evaluation"] is None else len(effects["counterexamples"]),
            terminal=audit.get("terminal_status", "unassessed")),
        boundary_counts=audit.get("boundary_counts"),
        limits=["Metadata-only report: no NPY, model, J or remote payload was opened; native/audit collection receipts are the custody authority.",
            "Old audits missing direct primitive/projection comparisons remain incomplete; exact endpoints cannot certify a hidden primitive nullspace.",
            "Signed coordinates, orthogonal error, raw bits, signed zeros and frozen consumer counterexamples remain separate observations.",
            "Primitive J includes all primitive token/features; endpoints replace image rows with held Q text and receiving closure.",
            "Shared kernels and original parameters are paid; no independent-kernel, speed, semantic or cross-family scalar-ranking claim."])
    return json.loads(canonical(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run", required=True, type=Path)
    parser.add_argument("--audit-run", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--rule", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.native_run, args.audit_run, args.plan, rule=args.rule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps(dict(output=str(args.out), sha256=sha(args.out), statuses=result["statuses"],
        comparison_totals=result["comparison_totals"]), sort_keys=True))
    return result


if __name__ == "__main__":
    main()
