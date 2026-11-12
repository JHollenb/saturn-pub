"""Index conditional writer equations and their actual restored native futures."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb.computation_atlas import ComputationAtlas
from mdb.equation_comparison import EquationCatalog,compare_equations
from mdb.edit_search import file_digest


def publish(panels,output):
    here=Path(__file__).resolve().parent
    read=lambda p:json.loads(p.read_text())
    readslot=runpy.run_path(str(here/"analyze_flux_binding_minimal.py"))["read_slot"]
    B=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/"catalog");atlas=ComputationAtlas(output/"atlas")
    contexts={};all_ids=[];jobs=[]
    for root in panels:
        audit=root/"audit-v1.json";review=root/"image-review-v1.json";run=root/"collection-01";data=run/"extracted/binding-minimal-result"
        a=read(audit);inspection=read(review);request=read(run/"request.json");summary=read(data/"summary.json")
        if a["mechanics_status"]!="verified":raise ValueError("audited panel required")
        refs=[dict(path=str(p),sha256=file_digest(p))for p in(audit,review,run/"request.json",run/"archive-custody.json")]
        rows={r["id"]:r for r in read(data/"rows.json")};chain=read(data/"receipts.json")
        execution=dict(backend="diffusers",**next(r["body"]["execution"]for r in chain if r["body"]["operation"]=="continue"))
        model=next(r["body"]["model_identity"]for r in chain)
        records=[];effects={};futures={};counters={}
        for w in read(data/"writes.json"):
            name=w["row"];row=rows[name];assert inspection["rows"][name]["png_sha256"]==row["png_sha256"]==file_digest(data/row["png"])
            ports={}
            for k,r in w["fields"].items():
                p=data/r["path"];assert file_digest(p)==r["sha256"];ports[k]=np.load(p,allow_pickle=False)
            entry=readslot(data/"store",w["entry"],"image");actual=readslot(data/"store",w["exit"],"image")
            predicted=B(B(entry+B(ports["gate_attention"]*ports["attention"]))+B(ports["gate_mlp"]*ports["mlp"]))
            report=compare_equations({"image.exit":predicted},{"image.exit":actual},max_elements=4_000_000)
            horizon=dict(equation="step2/joint0/image-exit",consumer="declared restoration program, native suffix and VAE")
            context=dict(model_identity=model,source_identity=request["source_hashes"],execution=execution,horizon=horizon,
                receiving_state=w["entry"],clock=dict(step=2,joint=0),job_id=summary["job_id"],row=name,
                restoration={k:row[k]for k in("restore_slots","restore_index","rows_by_step")},text_keep_rows=row.get("text_keep_rows"))
            record=catalog.record(law_id="flux.J00.conditional-bf16-residual.v1",route_id="flux.early-conditioning.working-carrier.native-future.v1",
                alias="cmp.flux.binding.carrier."+name,binding=dict(model_identity=model,source_identity=request["source_hashes"],execution=execution,
                    input_identity=w["entry"],horizon=horizon,frontier=dict(kind="observed-native-ports",scope="Post-hoc equation closure using native attention/MLP returns; no independent port or RGB prediction",entry_cut=w["entry"])),
                report=report,conditions=[dict(field=k,op="eq",value=v)for k,v in context.items()],tested_context=context,
                readers=["native.image.exit","rgb.binding.donor-projection"],evidence=dict(artifacts=refs,seal=summary["seal"],
                    observation=inspection["rows"][name],restoration=a["carrier_restoration"][name],
                    scope="Equation is conditional and post-hoc. Categorical candidate predictions were frozen prospectively. Continuation uses native computation after declared typed Acts."))
            fp=record["fingerprint"];records.append(record);contexts[fp]=context;all_ids.append(fp)
            effects[fp]=[dict(consumer_id="rgb.binding.donor-projection",contrast_id="branch-minus-matched-control",units="RGB projection toward natural donor; not semantic correctness",
                value=a["outcomes"][name]["rgb"]["target_projection"]-a["outcomes"]["control"]["rgb"]["target_projection"],provenance=refs)]
            futures[fp]=dict(status="native-RGB-observed",row_id=name,png_sha256=row["png_sha256"],provenance=refs,
                categorical_prediction_matched=a["outcomes"][name]["color_prediction_matched"],final_baseline_exact=a["carrier_restoration"][name]["final_payload_exact"])
            counters[fp]=([]if a["outcomes"][name]["color_prediction_matched"]else[dict(claim_id="frozen-categorical-continuation",status="counterexample-retained",provenance=refs,
                predicted=a["outcomes"][name]["prediction"],observed=inspection["rows"][name]["observation"])])
        atlas.ingest(output/"catalog",[r["fingerprint"]for r in records],provenance=dict(run_id=summary["job_id"],artifacts=refs),signed_effects=effects,continuation=futures,counterexamples=counters)
        jobs.append(summary["job_id"])
    for fp,c in contexts.items():
        assert len(atlas.find(applicable_to=c))==1
        assert not atlas.find(applicable_to={**c,"clock":dict(step=2,joint=1)})
        assert not atlas.find(applicable_to={**c,"receiving_state":"unknown-parent"})
    result=dict(schema="mdb-carrier-atlas-publication-v1",jobs=jobs,equation_occurrences=len(all_ids),exact_queries=len(all_ids),negative_queries=2*len(all_ids),
        selected_equation_ids=all_ids,new_model_calls=0,scope="Curated exact-context observations, not rule induction or independent port/RGB forecasts. Repeated J00 native computations across later-restored branches are not independent equations.")
    with (output/"provenance.json").open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    return {k:result[k]for k in("equation_occurrences","exact_queries","negative_queries","new_model_calls")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path);p.add_argument("panels",nargs="+",type=Path);a=p.parse_args()
    print(json.dumps(publish([v.resolve()for v in a.panels],a.output.resolve()),sort_keys=True))
