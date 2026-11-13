"""Publish context-qualified equations and curated causal counterexamples."""
from __future__ import annotations
import argparse
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def publish(run,audit,output):
    from mdb.equation_comparison import EquationCatalog,compare_equations
    from mdb.computation_atlas import ComputationAtlas
    data=run/"extracted/six-areas-result";a=read(audit);summary=read(data/"summary.json");source=read(data/"source.json")
    assert a["mechanics_status"]=="verified" and a["request_sha256"]==summary["request_sha256"]
    output.mkdir(exist_ok=False);catalog=EquationCatalog(output/"catalog");atlas=ComputationAtlas(output/"atlas")
    here=Path(__file__).resolve().parent;typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));route=runpy.run_path(str(here/"analyze_flux_concept_routes.py"))
    ref=dict(path=str(audit),sha256=sha(audit));req=dict(path=str(run/"request.json"),sha256=sha(run/"request.json"))
    declared=read(data/"transaction/declarations.json");forecasts=read(data/"forecasts.json");observed=read(data/"observed.json");request=read(run/"request.json")
    records=[];contexts={};futures={}
    for name,fields in forecasts.items():
        parent=declared["frontier"]["initial_cuts"][name];execution=dict(backend="diffusers",**source["execution"])
        for boundary,key in (("project","project"),("joint0","joint")):
            horizon=dict(equation=boundary,consumer="native-complete-image-and-text-state")
            context=dict(model_identity=source["model_identity"],source_identity=request["source_hashes"],execution=execution,
                horizon=horizon,clock=read(data/"store/cuts"/(parent+".json"))["execution_point"],entry_cut=parent,receiving_state=parent,
                prompt_case=name,source_boundary="observed-Qwen-conditioning/initial-noise/native-weights")
            predicted={k:np.load(data/fields[key][k]["path"],allow_pickle=False) for k in ("image","text")}
            native={k:route["slot_array"](data/"store",observed[name][boundary],k,typed) for k in ("image","text")}
            report=compare_equations(predicted,native,max_elements=8_000_000)
            record=catalog.record(law_id="flux."+boundary+".own-state-reference.v1",route_id="flux.observed-conditioning-to-first-joint.v1",
                alias="contextual-program/formation/"+boundary,binding=dict(model_identity=source["model_identity"],execution=execution,
                    input_identity=parent,source_identity=context["source_identity"],horizon=horizon,
                    frontier=dict(kind="observed-encoder-and-initial-state",entry_cut=parent,writer="independent FP64 NumPy reductions with declared BF16 boundaries",parameter_recipes_sha256=sha(data/"parameter-recipes.json"))),
                report=report,conditions=[dict(field=k,op="eq",value=v) for k,v in context.items()],tested_context=context,
                readers=["native.image-state","native.text-state"],evidence=dict(audit=ref,request=req,
                    seal=summary["seal"],native_target_exclusion=True,forecast=a["numerical_forecasts"][name][boundary],
                    scope="Encoder is observed. Joint forecast consumes own predicted complete project/context. Native VAE future from predicted I/T only tested for b0r0 on native context scaffold."))
            records.append(record);contexts[record["fingerprint"]]=context
            futures[record["fingerprint"]]=dict(status="native-futures-observed",row_ids=[name]+(["math.bridge"] if name=="b0r0" and boundary=="joint0" else []),
                provenance=[ref],scope="Natural future observed; predicted-I/T bridge specifically b0r0 joint0 only.")
    imported=atlas.ingest(output/"catalog",[r["fingerprint"] for r in records],provenance=dict(run_id=summary["job_id"],artifacts=[ref,req]),continuation=futures)
    queries=[]
    for fp,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1
        changed=atlas.find(applicable_to={**context,"receiving_state":"unknown"});clock=atlas.find(applicable_to={**context,"clock":{**context["clock"],"logical_step":1}})
        assert not changed and not clock
        queries.append(dict(equation_id=fp,exact_hits=1,changed_receiving_hits=0,changed_clock_hits=0))
    causal=[]
    for name,effect in a["interventions"].items():
        causal.append(dict(id="flux.J04.route-and-write."+name,entry_cut=observed["b0r0"]["joint4_entry"],
            model_identity=source["model_identity"],clock=dict(step=1,phase="joint",index=4),
            receiver="b0r0",source="b0r0" if name=="route.sham" else "b1r0",consumer="native-base-text-and-conditioning-suffix/VAE",
            signed_effect=effect,collateral=a["collateral"][name],propagation=a["propagation"][name],
            applicability="one exact source/recipient occurrence; no automatic semantic portability",
            counterexamples=[] if name=="route.sham" or (effect["target_projection"] is not None and effect["target_projection"]>0) else ["source-directed pixel-projection expectation did not hold"],
            evidence=[ref,req],terminal_status="unassessed"))
    result=dict(schema="mdb-six-areas-publication-v1",records=len(records),selected_occurrences=[r["fingerprint"] for r in records],
        atlas_ids=[r["fingerprint"] for r in imported],queries=queries,curated_causal_records=len(causal),new_model_calls=0,audit=ref,
        scope="8 exact-context numerical equations;7 curated causal route/write observations including signed misses. No semantic rule induction or cross-model portability.")
    for name,value in (("provenance.json",result),("causal-records.json",dict(records=causal))):
        with (output/name).open("x") as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    return dict(records=len(records),exact_queries=len(queries),negative_queries=2*len(queries),causal_records=len(causal),new_model_calls=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("audit",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(publish(a.run.resolve(),a.audit.resolve(),a.output.resolve()),sort_keys=True))
