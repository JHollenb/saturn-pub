"""Index native stream-factorial corner and effect records without new model work."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
from mdb.equation_comparison import EquationCatalog,compare_equations
from mdb.forecast_transaction import load_forecast_tree
from saturn_pub.store import LocalStore


def publish(root,collection,output):
    H=runpy.run_path(str(Path(__file__).with_name("analyze_flux_attention_state_law.py")))
    read,sha,tree=[H[n] for n in ("read","sha","tree")]
    split=runpy.run_path(str(Path(__file__).with_name("flux_stream_factorial.py")))["decompose"]
    audit=read(root/"audit-v1.json");assert audit["mechanics_status"]=="verified"
    data=collection/"extracted/binding-minimal-result";config,summary,rows=[read(data/n) for n in ("plan.json","summary.json","rows.json")]
    assert summary["request_sha256"]==audit["request_sha256"]
    tx=data/"transaction"
    predicted={c:tree(load_forecast_tree(tx,read(tx/("forecasts/corner"+c+".json"))["predictions"])) for c in config["corners"]}
    store=LocalStore(data/"store");native={};rowmap={(r["case"],r["corner"]):r for r in rows}
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/"equations")
    refs=[dict(path=str(p),sha256=sha(p)) for p in (root/"audit-v1.json",data/"transaction/seal.json",data/"source-bindings.json",data/"parameter-artifacts.json",data/"source-projection.json")]
    ids,queries=[],[]
    def index(label,law,context,forecast,observed,**evidence):
        report=compare_equations(forecast,observed,max_elements=20_000_000)
        r=catalog.record(law_id=law,route_id="route.same-parent.image-text.joint-read.receiving-mlp.v1",alias="cmp.flux.stream-factorial."+label,
            binding=dict(model_identity=config["model"],execution=context["execution"],input_identity=context["input_identity"],
                source_identity=audit["request_sha256"],frontier=context["frontier"],horizon=context["horizon"]),
            report=report,conditions=[dict(field=k,op="eq",value=v) for k,v in context.items()],tested_context=context,
            readers=["native.block:return.image[complete]","native.block:return.text[complete]"],
            evidence=dict(artifacts=refs,semantic_labels="unassigned",new_fitted_parameters=0,terminal_status="unassessed",**evidence))
        assert len(catalog.find(applicable_to=context))==1
        assert not catalog.find(applicable_to={**context,"execution":{**context["execution"],"backend":"cuda"}})
        assert not catalog.find(applicable_to={**context,"clock":{**context["clock"],"block":3}})
        ids.append(r["fingerprint"]);queries.append(dict(label=label,record=r["fingerprint"],context=context))
        with (output/("context-"+label+".json")).open("x") as f:json.dump(context,f,indent=2,sort_keys=True);f.write("\n")
    def context(case,corner,parent,horizon):
        return dict(model_identity=config["model"],checkpoint=config["checkpoint_sha256"],case=case,corner=corner,input_identity=parent,
            execution=dict(backend="cpu",device="cpu",dtype="bfloat16",threads=4,queries=1536,keys=1536),
            clock=dict(step=2,phase="joint",block=2),horizon=horizon,
            frontier=dict(kind="observed native joint2-entry crossed with prior sealed predicted joint1 streams",source_job=config["source_job"],
                source_cut=rowmap[(case,"00")]["source_cut"]))
    for row in rows:
        case,corner=row["case"],row["corner"]
        with np.load(data/row["archive"]["path"],allow_pickle=False) as z:values={n:z[n] for n in z.files}
        p=store.load(row["result"]).payload;values.update({s:p[s].float().numpy() for s in ("image","text")});native[(case,corner)]=values
        c=context(case,corner,row["parent"],"one original native joint2 full image/text exit and port witnesses")
        index(case+"-"+corner,"law.flux.joint-state.v1",c,predicted[corner][case],values,
            new_mixed_state=corner in {"10","01"},calibration_endpoint=corner in {"00","11"},
            native_receipt=row["receipt"]["fingerprint"],acts=[a["receipt"]["fingerprint"] for a in row["acts"]],
            numerical_errors=audit["errors"][case+":"+corner])
    for case in config["cases"]:
        actual,forecast={},{}
        for stream in ("image","text"):
            a=split({c:native[(case,c)][stream] for c in config["corners"]})
            p=split({c:predicted[c][case][stream] for c in config["corners"]})
            for term in ("image","text","interaction","total"):
                actual[stream+"."+term]=a[term];forecast[stream+"."+term]=p[term]
        c=context(case,"factorial",rowmap[(case,"00")]["root"],"same-parent image/text factorial effects at complete joint2 exits")
        index(case+"-factorial","law.flux.image-text-factorial.v1",c,forecast,actual,
            factorial_errors={s:audit["factorials"][case][s] for s in ("image","text")},
            interpretation="Observed-native factorial versus independently predicted factorial. Baseline-dependent signed effects; no causal percentages or semantic conjunction inferred.")
    result=dict(schema="mdb-flux-stream-factorial-publication-v1",equation_occurrences=len(ids),ids=ids,queries=queries,
        exact_context_queries=len(ids),changed_backend_abstentions=len(ids),changed_clock_abstentions=len(ids),
        new_model_calls=0,audit_sha256=sha(root/"audit-v1.json"),source_hashes={Path(__file__).name:sha(Path(__file__))})
    with (output/"provenance.json").open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    return {k:result[k] for k in ("equation_occurrences","exact_context_queries","changed_backend_abstentions","changed_clock_abstentions","new_model_calls")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("root","collection","output"):p.add_argument(n,type=Path)
    a=p.parse_args();print(json.dumps(publish(a.root.resolve(),a.collection.resolve(),a.output.resolve()),sort_keys=True))
