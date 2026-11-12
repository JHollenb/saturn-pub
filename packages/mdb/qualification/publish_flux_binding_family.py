"""Import audited computation-family occurrences without extending applicability.

Runs entirely on saved metadata. Equation reports were computed against complete
arrays by the independent CPU reader. Route predicates are a separate prospectively
frozen result, never inferred from the exactness of the numerical reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def checked_compact(run, kind):
    collection = read(run / "collection.json")
    require(collection["schema"] == "mdb-family-authenticated-compact-v1" and collection["kind"] == kind, "authenticated compact required")
    root = run / "compact"
    for name, record in collection["inventory"]["files"].items():
        path = root / name
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "compact member differs: " + name)
    return root, collection


def publish(run, audit_run, output):
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog

    native, collection = checked_compact(run, "native")
    audited, audit_collection = checked_compact(audit_run, "audit")
    audit = read(audited / "audit.json")
    request, plan, summary, roots, writers, states, runtime = (read(native / name) for name in (
        "request.json", "plan.json", "summary.json", "source-roots.json", "native-source-writers.json", "state-index.json", "runtime-contract.json"))
    require(audit["native_request_sha256"] == request["request_sha256"] == collection["request_sha256"]
        and audit["native_job_id"] == summary["job_id"] == collection["job_id"]
        and audit_collection["publication"]["source_job"] == summary["job_id"], "audit/native identity differs")
    require(audit["mechanics_status"] == "verified" and audit["model_forwards"] == 0, "saved audit required")
    require(set(audit["equation_reports"]) == set(states), "complete original-array equation reports required")
    output.mkdir(parents=True, exist_ok=False)
    catalog, atlas = EquationCatalog(output / "catalog"), ComputationAtlas(output / "atlas")
    audit_ref = dict(path=str(audited / "audit.json"), sha256=sha(audited / "audit.json"))
    request_ref = dict(path=str(run / "request.json"), sha256=sha(run / "request.json"))
    execution = dict(backend="torch-original-primitive-cuda", runtime=runtime,
        geometry=dict(batch_size=1, text_tokens=512, image_tokens=1024, height=512, width=512, steps=4, dtype="bfloat16"))
    records, contexts, effects, continuations, counters = [], {}, {}, {}, {}
    route_rows, curated = [], []
    fresh = {row["id"]: row for row in plan.get("family_rule", {}).get("fresh_contexts", [])}
    for key, state in states.items():
        cid, arm = state["context"], state["arm"]
        root = roots[cid]
        model = root["inputs"]["Q"]["model_identity"]
        context = dict(model_identity=model, source_identity=request["source_hashes"], execution=execution,
            clock=dict(step=2, phase="joint", index=0), entry_cut=root["inputs"]["Q"]["cut"],
            paired_input=root["inputs"]["P"]["cut"], receiving_state=writers[cid]["sources"]["Q"]["state"]["cut"],
            support=root["support"], context_id=cid, arm=arm,
            horizon=dict(equation="Original joint0 writer and selected full-text endpoint", consumer="54 own-state transitions through step4 latent and original VAE/RGB; fixed step3 baseline conditioning reset"))
        evidence = dict(audit=audit_ref, request=request_ref, forecast_seal=summary["forecast_seal"],
            first_consumer=audit["cases"][key]["consumer_ports"], entry_act=audit["act_checks"][key],
            endpoint_construction=audit["construction_checks"][cid][arm], native_final=state["native_final"],
            applicability_scope="Exact recorded occurrence; family trend and fresh predicates are scored separately")
        record = catalog.record(law_id="flux.joint.sequential-text-write.bf16-reference.v1",
            route_id="flux.step2.joint0.text-support-to-image.v1", alias="computation-family/selected-joint-writer-occurrence",
            binding=dict(model_identity=model, execution=execution, source_identity=context["source_identity"], input_identity=context["entry_cut"],
                horizon=context["horizon"], frontier=dict(kind="architectural", source="P/Q joint0 INPUTS; original weights",
                    equation="U=B(R+B(gA*A)); T=B(U+B(gM*M)); selected rows installed in Qout; remainder Qout",
                    continuation="Original primitive suffix consumes its own predicted state; no native refresh")),
            report=audit["equation_reports"][key]["writer_exit"], conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()],
            tested_context=context, readers=["joint1.image-mlp", "latent.step4", "rgb.full-image"], evidence=evidence)
        identifier = record["fingerprint"]
        records.append(record); contexts[identifier] = context
        effect = audit["observations"][cid]["pixels"]["responses"][arm]
        effects[identifier] = [dict(consumer_id="rgb.full-image", contrast_id="arm-minus-Q-on-support-minus-Q-axis",
            value=effect["projection"], units="signed RGB contrast projection; not semantic percentage", provenance=[audit_ref])]
        if effect["projection"] is None:
            effects[identifier] = []
        continuations[identifier] = dict(status="sealed-equation-and-native-futures-observed", case=key,
            equation_report=audit["equation_reports"][key]["continuation"], provenance=[audit_ref])
        counters[identifier] = []
        predicates = audit["observations"][cid]["pixels"].get("frozen_route_predictions", {})
        for name, result in predicates.items():
            if type(result) is bool and not result:
                counters[identifier].append(dict(claim_id=name, status="counterexample", context=cid, provenance=[audit_ref], details=predicates))
        component = audit["observations"][cid]["pixels"].get("component_prediction")
        if component and component["closer_to_support_than_Q"] is False:
            counters[identifier].append(dict(claim_id="component-closer-to-support-than-Q", status="counterexample",
                context=cid, provenance=[audit_ref], details=component))
    for cid in roots:
        observed = audit["observations"][cid]["pixels"]
        predicates = {k: v for k, v in observed.get("frozen_route_predictions", {}).items() if type(v) is bool}
        row = dict(context=cid, panel=fresh.get(cid, {}).get("panel", "historical-development"),
            predictions=predicates, measurements=observed, component_prediction=observed.get("component_prediction"),
            zero_effect=observed.get("frozen_route_predictions", {}).get("whole_Q_rms") == 0,
            outcomes_known_when_rule_selected=plan["mode"] == "historical")
        route_rows.append(row)
        curated.append(dict(schema_version="known-computations.v1", id="flux-binding-family-" + plan["mode"] + "-" + cid,
            family_id="diffusion.support-addressed-joint-writer.v1", model="FLUX.2 Klein 4B distilled",
            model_identity=roots[cid]["inputs"]["Q"]["model_identity"], request_sha256=request["request_sha256"],
            producer_job_id=summary["job_id"], prompt=fresh.get(cid, {}).get("prompts"),
            coordinates=dict(step=2, phase="joint", index=0, support=roots[cid]["support"]),
            record_type="prospective-family-test" if fresh else "historical-component-development",
            computation="Sequential BF16 gated text attention/MLP writes followed by an original-weight own-state continuation",
            consumer="Joint1 image MLP, remaining denoiser, Euler scheduler and native VAE/RGB",
            applicability=dict(scope="Captured P/Q joint0-input family; qualified original profile; fixed source and reset policy", contexts=[cid]),
            results=[dict(endpoint_rows=[dict(case=key, **audit["cases"][key]) for key in states if states[key]["context"] == cid]), row],
            counterexamples=[name for name, value in predicates.items() if not value] +
                (["component-closer-to-support-than-Q"] if observed.get("component_prediction", {}).get("closer_to_support_than_Q") is False else []),
            evidence_status=dict(mechanics="verified saved custody/arrays", trend="See all separate frozen predicates", terminal="bounded per-predicate outcomes"),
            limitations=["Source-assisted native prefix and paired prompt direction; no raw-prompt or donor-free universal claim.",
                "Original parameter arithmetic and primitive kernels remain paid; no compact-model speed claim.",
                "Full-image numerical and route effects have no calibrated object-level semantics."]))
    imported = atlas.ingest(output / "catalog", [record["fingerprint"] for record in records],
        provenance=dict(run_id=summary["job_id"], artifacts=[audit_ref, request_ref]), signed_effects=effects,
        continuation=continuations, counterexamples=counters)
    queries = []
    for identifier, context in contexts.items():
        exact = atlas.find(applicable_to=context)
        wrong_clock = atlas.find(applicable_to={**context, "clock": dict(step=1, phase="joint", index=0)})
        wrong_receiver = atlas.find(applicable_to={**context, "receiving_state": "unobserved-receiver"})
        require(len(exact) == 1 and not wrong_clock and not wrong_receiver, "atlas exact/abstention query differs")
        queries.append(dict(equation_id=identifier, atlas_id=exact[0]["fingerprint"], exact_hits=1, changed_clock_hits=0, changed_receiver_hits=0))
    with (output / "computation-records-v1.jsonl").open("x") as stream:
        for row in curated:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result = dict(schema="mdb-binding-family-publication-v1", native_job=summary["job_id"], mode=plan["mode"],
        audit=audit_ref, request=request_ref, rule=plan.get("family_rule"), family_rule_sha256=plan.get("family_rule_file_sha256"),
        occurrences=len(records), atlas_ids=[row["fingerprint"] for row in imported], queries=queries,
        route_rows=route_rows, new_model_forwards=0,
        materializer_sha256=sha(__file__),
        numerical_misses={key: audit[key] for key in ("numerical_bit_misses", "writer_bit_misses", "endpoint_bit_misses", "recomputed_bit_misses")},
        scope="Prospective route tests remain separate from source-qualified exact numerical occurrences; applicability does not expand automatically")
    save(output / "family-summary.json", result)
    (output / "materializer.py").write_bytes(Path(__file__).read_bytes())
    return dict(occurrences=len(records), contexts=len(roots), exact_queries=len(queries), abstention_queries=2*len(queries), new_model_forwards=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_run", type=Path)
    parser.add_argument("audit_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.native_run.resolve(), args.audit_run.resolve(), args.output.resolve()), sort_keys=True))
