"""Index audited native writer equations, causal continuations and misses."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog, compare_equations


def publish(root, output):
    helper = runpy.run_path(str(Path(__file__).with_name("analyze_flux_context_transfer.py")))
    read, sha = helper["read"], helper["sha"]
    audit = read(root / "audit-v2.json")
    review = read(root / "image-review-v1.json")
    if audit["mechanics_status"] != "verified":
        raise ValueError("audited panel required")
    data = root / "collection-01/extracted/binding-minimal-result"
    request = read(data / "request.json")
    rows = read(data / "rows.json")
    receipts = read(data / "receipts.json")
    model = receipts[0]["body"]["model_identity"]
    execution = dict(backend=request["backend"], **next(r["body"]["execution"] for r in receipts if r["body"]["operation"] == "continue"))
    refs = [dict(path=str(p), sha256=sha(p)) for p in (
        root / "audit-v2.json", root / "image-review-v1.json", data / "request.json",
        root / "collection-01/archive-custody.json", data / "port-interventions.json")]
    output.mkdir(exist_ok=False, parents=True)
    catalog, atlas = EquationCatalog(output / "catalog"), ComputationAtlas(output / "atlas")
    ids, contexts, effects, continuations, counterexamples = [], {}, {}, {}, {}
    unique_entries = set()
    for row in rows:
        name, writer = row["id"], row["writer"]
        inspection = review["rows"][name]
        assert inspection["png_sha256"] == row["png_sha256"] == sha(data / row["png"])
        ports = {k: np.load(data / v["path"], allow_pickle=False) for k, v in writer["fields"].items()}
        incoming = helper["read_slot"](data / "store", writer["entry"], "image")
        actual = helper["read_slot"](data / "store", writer["exit"], "image")
        predicted = helper["writer_equation"](incoming, ports)
        report = compare_equations({"image.exit": predicted}, {"image.exit": actual}, max_elements=4_000_000)
        horizon = dict(equation="step2/joint1/image-exit", consumer="native continuation and VAE RGB")
        context = dict(model_identity=model, source_identity=request["source_hashes"], execution=execution,
            horizon=horizon, receiving_state=writer["entry"], clock=dict(step=2, joint=1),
            job_id=audit["job_id"], row=name, source_seed=109, relation="sphere right of cube",
            program={k: row[k] for k in ("edited", "restore", "port")},
            priming="donor image pulse at step1/joint4, base working text; common project2 parent")
        record = catalog.record(law_id="flux.joint.conditional-bf16-residual.v1",
            route_id="flux.working-text.image-attention.image-mlp.native-future.v1",
            alias="cmp.flux.context.j1." + name,
            binding=dict(model_identity=model, source_identity=request["source_hashes"], execution=execution,
                input_identity=writer["entry"], horizon=horizon,
                frontier=dict(kind="observed-native-ports", entry_cut=writer["entry"],
                    scope="Post-hoc conditional arithmetic; native attention/MLP operands, not independent forecasts")),
            report=report, conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()],
            tested_context=context, readers=["native.image.exit", "rgb.right-minus-left.chromaticity"],
            evidence=dict(artifacts=refs, observation=inspection, local=audit["writer_comparisons"][name],
                seal=read(data / "summary.json")["seal"],
                scope="Prospective categorical/local hypotheses, native edited computation and continuation. Equation itself known architectural arithmetic."))
        identifier = record["fingerprint"]
        ids.append(identifier)
        contexts[identifier] = context
        effects[identifier] = [dict(consumer_id="rgb.right-minus-left.chromaticity", contrast_id="branch-minus-control",
            units="right-minus-left mean (B-R)/(R+G+B+1), uint8 channel units",
            value=audit["outcomes"][name]["signed_effect"], provenance=refs)]
        continuations[identifier] = dict(status="native-RGB-observed", row_id=name,
            png_sha256=row["png_sha256"], provenance=refs,
            categorical_prediction_matched=inspection["categorical_prediction_matched"],
            final_baseline_exact=audit["writer_comparisons"][name]["final_payload_exact"])
        counterexamples[identifier] = [] if inspection["categorical_prediction_matched"] is True else [dict(
            claim_id="frozen-categorical-continuation", status="counterexample-or-ambiguous-retained",
            predicted=audit["outcomes"][name]["prediction"], observed=inspection["observation"], provenance=refs)]
        payload = read(data / "store/cuts" / (writer["entry"] + ".json"))["manifest"]["payload"]
        unique_entries.add(json.dumps(payload, sort_keys=True))
    atlas.ingest(output / "catalog", ids, provenance=dict(run_id=audit["job_id"], artifacts=refs),
                 signed_effects=effects, continuation=continuations, counterexamples=counterexamples)
    for identifier, context in contexts.items():
        assert len(atlas.find(applicable_to=context)) == 1
        assert not atlas.find(applicable_to={**context, "source_seed": 110})
        assert not atlas.find(applicable_to={**context, "relation": "sphere left of cube"})
        assert not atlas.find(applicable_to={**context, "receiving_state": "unknown"})
    result = dict(schema="mdb-context-transfer-atlas-publication-v1", equation_occurrences=len(ids),
        unique_joint1_payloads=len(unique_entries), selected_equation_ids=ids,
        exact_queries=len(ids), abstention_queries=3 * len(ids), new_model_calls=0,
        scope="Exact-context records; repeated states counted separately from unique payloads. Family generalization remains a hypothesis, not automatic induction.")
    with (output / "provenance.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.root.resolve(), args.output.resolve()), sort_keys=True))
