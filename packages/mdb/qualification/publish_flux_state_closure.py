"""Searchable full-state and own-input continuation records; no new model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

import numpy as np
from mdb.equation_comparison import EquationCatalog, compare_equations
from mdb.forecast_transaction import load_forecast_tree
from saturn_pub.store import LocalStore


def publish(root, collection, output):
    helper = runpy.run_path(str(Path(__file__).with_name("analyze_flux_attention_state_law.py")))
    read, sha, tree = [helper[n] for n in ("read", "sha", "tree")]
    audit = read(root / "audit-v1.json")
    assert audit["mechanics_status"] == "verified"
    data = collection / "extracted/binding-minimal-result"
    config, summary, rows = [read(data / n) for n in ("plan.json", "summary.json", "rows.json")]
    assert summary["request_sha256"] == audit["request_sha256"]
    tx = data / "transaction"
    predicted = {i:tree(load_forecast_tree(tx, read(tx / f"forecasts/joint{i}.json")["predictions"])) for i in config["blocks"]}
    store = LocalStore(data / "store")
    rowmap = {(r["branch"],r["case"],r["block"]):r for r in rows}
    output.mkdir(parents=True,exist_ok=False)
    catalog = EquationCatalog(output / "equations")
    evidence = [dict(path=str(p),sha256=sha(p)) for p in (root/"audit-v1.json",data/"transaction/seal.json",data/"native-source.json",data/"parameter-artifacts.json",data/"forks.json")]
    records,queries = [],[]
    for row in rows:
        name,index,branch = row["case"],row["block"],row["branch"]
        with np.load(data / row["archive"]["path"],allow_pickle=False) as z:
            values = {k:z[k] for k in z.files}
        after = store.load(row["result"]).payload
        values.update({s:after[s].float().numpy() for s in ("image","text")})
        report = compare_equations(predicted[index][name],values,max_elements=20_000_000)
        own_input = branch == "predicted-input"
        source = row["parent"] if own_input else rowmap[("native",name,1)]["parent"]
        frontier = dict(kind="sealed predicted complete joint1 exit" if own_input else "observed raw joint1-entry state",
            input_cut=source,seal=summary["seal"]["seal_sha256"])
        context = dict(model_identity=config["model"],checkpoint=config["checkpoint_sha256"],
            execution=dict(backend="cpu",device="cpu",dtype="bfloat16",threads=4,native_queries=1536,native_keys=1536),
            case=name,branch=branch,input_identity=source,frontier=frontier,
            clock=dict(step=2,phase="joint",entry_block=2 if own_input else 1,exit_block=index+1),
            horizon="one native joint2 on predicted joint1 state" if own_input else f"complete {index}-block image/text forecast from joint1 entry",
            image_shape=[1,1024,3072],text_shape=[1,512,3072])
        law = "law.flux.joint-state-chain.v1" if index == 2 and not own_input else "law.flux.joint-state.v1"
        alias = f"cmp.flux.state-closure.{branch}.{name}.j{index}"
        record = catalog.record(law_id=law,route_id="route.joint-image-text.attention.swiglu.residual.v1",alias=alias,
            binding=dict(model_identity=config["model"],execution=context["execution"],input_identity=source,
                frontier=frontier,horizon=context["horizon"],source_identity=audit["request_sha256"]),
            report=report,conditions=[dict(field=k,op="eq",value=v) for k,v in context.items()],
            tested_context=context,readers=["native.block:return.image[complete]","native.block:return.text[complete]"],
            evidence=dict(artifacts=evidence,native_receipt=row["receipt"]["fingerprint"],
                complete_predicted_closure=True,predicted_input_consumer=own_input,
                new_internal_input=name=="quarter",previously_observed_input=name!="quarter",
                errors=audit["errors"][f"{branch}:{name}:{index}"],new_fitted_parameters=0,
                semantic_labels="unassigned",terminal_status="unassessed",full_pipeline_continuation=False))
        assert len(catalog.find(applicable_to=context)) == 1
        assert not catalog.find(applicable_to={**context,"execution":{**context["execution"],"backend":"cuda"}})
        assert not catalog.find(applicable_to={**context,"clock":{**context["clock"],"step":3}})
        records.append(record["fingerprint"])
        label=f"{branch}-{name}-j{index}"
        queries.append(dict(label=label,record=record["fingerprint"],context=context))
        with (output/("context-"+label+".json")).open("x") as stream:
            json.dump(context,stream,indent=2,sort_keys=True);stream.write("\n")
    result = dict(schema="mdb-flux-state-closure-publication-v1",equation_occurrences=len(records),
        exact_context_queries=len(records),changed_backend_abstentions=len(records),changed_clock_abstentions=len(records),
        ids=records,queries=queries,new_model_calls=0,audit_sha256=sha(root/"audit-v1.json"),
        source_hashes={Path(__file__).name:sha(Path(__file__))},
        scope="Supplied exact-context conditions; no automatic semantic or cross-context rule induction.")
    with (output/"provenance.json").open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write("\n")
    return {k:result[k] for k in ("equation_occurrences","exact_context_queries","changed_backend_abstentions","changed_clock_abstentions","new_model_calls")}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","collection","output"):p.add_argument(n,type=Path)
    a=p.parse_args();print(json.dumps(publish(a.root.resolve(),a.collection.resolve(),a.output.resolve()),sort_keys=True))
