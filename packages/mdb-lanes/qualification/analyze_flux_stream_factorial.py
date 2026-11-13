"""Saved-only state, port, factorial and cross-stream locality audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

import numpy as np
from mdb_lanes.forecast_transaction import load_forecast_tree,verify_transaction
from saturn_pub.store import LocalStore


def audit(collection,output):
    here=Path(__file__).resolve().parent
    H=runpy.run_path(str(here/"analyze_flux_attention_state_law.py"))
    read,sha,tree,B,metric,error,bits=[H[n] for n in ("read","sha","tree","B","metric","error","bits")]
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](collection)
    data=collection/"extracted/binding-minimal-result"
    config,summary,rows,receipts,bindings=[read(data/n) for n in ("plan.json","summary.json","rows.json","receipts.json","source-bindings.json")]
    request=read(collection/"request.json")
    assert config==request["config"] and summary==read(collection/"report.json")["report"]
    assert summary["request_sha256"]==request["request_sha256"]==custody["request_sha256"]
    assert summary["counts"]==config["expected_counts"]
    for n,h in request["source_hashes"].items():assert sha(collection/"payload"/n)==h
    tx=data/"transaction";verified=verify_transaction(tx,expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert verified["verified"] and verified["phase"]=="closed"
    predicted={c:tree(load_forecast_tree(tx,read(tx/("forecasts/corner"+c+".json"))["predictions"])) for c in config["corners"]}
    census=tree(load_forecast_tree(tx,read(tx/"forecasts/census.json")["predictions"]))
    expected_at_seal={k:v if k in {"component_loads","numpy_block_forecasts"} else 0 for k,v in config["expected_counts"].items()}
    assert census["counts_at_seal"]==read(data/"seal-census.json")["counts"]==expected_at_seal
    assert census["hypotheses"]==config["hypotheses"]
    observed=tree(load_forecast_tree(tx,read(tx/"observations/native.json")["value"]))
    assert observed==dict(rows=rows,counts=summary["counts"])
    projection=read(data/"source-projection.json")
    assert projection["source_job"]==config["source_job"] and projection["source_request"]==config["source_request_sha256"]
    assert projection["source_seal"]==config["source_seal_sha256"] and projection["source_files"]==config["source_files_sha256"]
    assert projection["source_transaction"]["verified"] and projection["source_transaction"]["seal_sha256"]==config["source_seal_sha256"]
    parameters=read(data/"native-source.json")["parameters"]
    recipes={r["source"]["tensor"].removeprefix("transformer_blocks.2."):r["source"] for r in read(data/"parameter-artifacts.json")["artifacts"]}
    assert set(parameters)==set(recipes) and len(recipes)==16
    for n,p in parameters.items():
        assert p["sha256"]==recipes[n]["payload_sha256"] and p["shape"]==recipes[n]["shape"] and p["dtype"]=="torch.bfloat16"
    receipt_check=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    expected_receipts=[r["receipt"] for r in rows]+[a["receipt"] for r in rows for a in r["acts"]]
    assert len(receipts)==len(expected_receipts)==27 and len({r["fingerprint"] for r in receipts})==27
    assert {r["fingerprint"] for r in receipts}=={r["fingerprint"] for r in expected_receipts}
    for r in receipts:
        receipt_check(r);assert r==read(data/"store/receipts"/(r["fingerprint"]+".json"))
    store=LocalStore(data/"store")
    def fields(fp):
        m=read(data/"store/cuts"/(fp+".json"))["manifest"]
        return {i["key"]["value"]:i["value"] for i in m["payload"]["items"]}
    def ports():
        p={}
        for stream in ("image","text"):
            ff="ff" if stream=="image" else "ff_context"
            for field,module in (("projected",ff+".linear_in"),("activated",ff+".act_fn.gate_fn"),("product",ff+".act_fn"),("mlp",ff)):
                p[stream+"."+field]=module+":return[witness]"
            p[stream+".mlp_input"]=ff+":input[witness]"
            p[stream+".norm"]="attn:input."+("hidden_states[image-witness]" if stream=="image" else "encoder_hidden_states[text-witness]")
            p[stream+".attention"]="attn:return."+stream+"[witness]"
            for field in "qkv":p[stream+"."+field]="native.dispatch_attention_fn:input."+field+"["+stream+"-witness]"
            p[stream+".head"]="native.dispatch_attention_fn:return["+stream+"-witness]"
            p[stream]="native.block:return."+stream+"[complete]"
        return p
    native,comparisons,joins,equations,calibration={},{},[],{},{}
    rowmap={(r["case"],r["corner"]):r for r in rows}
    assert len(rowmap)==len(rows)==12
    assert set(rowmap)=={(case,c) for case in config["cases"] for c in config["corners"]}
    for row in rows:
        case,corner=row["case"],row["corner"];key=case+":"+corner;b=bindings[case]
        source=store.load(row["source_cut"])
        assert source.fingerprint==b["source_cut"]==projection["loaded_cut_ids"][case]
        assert source.model_identity==b["source_model"] and dict(source.execution_point)==b["source_point"]
        assert source.manifest()["payload"]==b["native_payload"]
        root,fork=store.load(row["root"]),store.load(row["fork_parent"])
        assert root.model_identity==fork.model_identity==config["model"] and fork.parent==root.fingerprint
        rf,sf=fields(root.fingerprint),fields(source.fingerprint)
        assert all(rf[n]==sf[n] for n in rf if n!="advanced")
        assert root.payload["advanced"]==0 and root.execution_point["index"]==fork.execution_point["index"]==2
        assert fields(fork.fingerprint)==rf
        last=fork.fingerprint
        stream_order=["image"] if corner=="00" else [s for s,bit in zip(("image","text"),corner,strict=True) if bit=="1"]
        assert [a["stream"] for a in row["acts"]]==stream_order
        for act in row["acts"]:
            stream=act["stream"];assert act["before"]==last
            ar=act["receipt"];assert ar["operation"]=="apply" and ar["parent"]==last and ar["result"]==act["after"]
            assert ar["act"]==act["act"] and act["act"]["writes"]==[stream]
            pre,post=fields(last),fields(act["after"])
            assert all(pre[n]==post[n] for n in pre if n!=stream)
            if corner=="00":assert pre==post
            else:assert all(post[stream][k]==v for k,v in b["predicted"][stream].items())
            last=act["after"]
        assert last==row["parent"]
        before,after=store.load(row["parent"]),store.load(row["result"])
        bp,ap=before.payload,after.payload
        assert before.model_identity==after.model_identity==config["model"]
        assert before.execution["device"]==after.execution["device"]=="cpu"
        assert bp["advanced"]==0 and ap["advanced"]==1
        assert before.execution_point["logical_step"]==after.execution_point["logical_step"]==2
        assert before.execution_point["index"]==2 and after.execution_point["index"]==3
        pre,post=fields(row["parent"]),fields(row["result"])
        assert all(pre[n]==post[n] for n in pre if n not in {"image","text","advanced"})
        for stream,bit in zip(("image","text"),corner,strict=True):
            if bit=="1":assert all(pre[stream][k]==v for k,v in b["predicted"][stream].items())
            else:assert pre[stream]==rf[stream]
        nr=row["receipt"];assert nr["operation"]=="continue" and nr["steps"]==1 and nr["parent"]==row["parent"] and nr["result"]==row["result"]
        p=data/row["archive"]["path"];assert sha(p)==row["archive"]["sha256"]
        with np.load(p,allow_pickle=False) as z:values={n:z[n] for n in z.files}
        assert not {"image","text"}&set(values) and row["complete_fields_from_state"]==["image","text"]
        values.update({s:ap[s].float().numpy() for s in ("image","text")})
        assert set(values)==set(row["ports"])==set(ports())==set(predicted[corner][case])
        for field,value in values.items():
            port=row["ports"][field]
            assert port["port"]==ports()[field]
            assert port["descriptor"]["sha256"]==bits(value) and port["descriptor"]["shape"]==list(value.shape) and port["descriptor"]["dtype"]=="torch.bfloat16"
            assert np.isfinite(value).all()
        native[(case,corner)]=values
        comparisons[key]={n:error(v,predicted[corner][case][n]) for n,v in values.items()}
        equations[key]={}
        for stream,modname,selected in (("image","img_mod",config["indices"]),("text","txt_mod",config["text_indices"])):
            mod=np.split(bp[modname].float().numpy().reshape(1,1,-1),6,-1)
            u=B(bp[stream].float().numpy()[:,selected]+B(mod[2]*values[stream+".attention"]))
            rebuilt=B(u+B(mod[5]*values[stream+".mlp"]))
            equations[key][stream]=error(values[stream][:,selected],rebuilt)
        if corner in {"00","11"}:
            calibration[key]={n:dict(exact_descriptor=row["ports"][n]["descriptor"]==b["calibration_outputs"][corner][n]["descriptor"]) for n in values}
        joins.append(dict(case=case,corner=corner,ports=len(values),root=row["root"],parent=row["parent"],result=row["result"],acts=len(row["acts"]),source_input_exact=True))
        del source,root,fork,before,after,bp,ap
    def parts(values):
        x={k:np.asarray(values[k],np.float64) for k in config["corners"]}
        return dict(image=x["10"]-x["00"],text=x["01"]-x["00"],interaction=x["11"]-x["10"]-x["01"]+x["00"],total=x["11"]-x["00"])
    factorials,locality={},{ }
    for case in config["cases"]:
        factorials[case]={};locality[case]={}
        for stream,pairs in (("image",(("00","01"),("10","11"))),("text",(("00","10"),("01","11")))):
            locality[case][stream]={a+"-"+b:{f:error(native[(case,a)][stream+"."+f],native[(case,b)][stream+"."+f]) for f in ("norm","q","k","v")} for a,b in pairs}
        for field in native[(case,"00")]:
            actual=parts({c:native[(case,c)][field] for c in config["corners"]})
            forecast=parts({c:predicted[c][case][field] for c in config["corners"]})
            errors={}
            for label,value in actual.items():
                e=error(value,forecast[label]);rms=metric(value)["rms"]
                e["relative_to_observed_effect"]=e["rms"]/rms if rms else None
                errors[label]=e
            factorials[case][field]=dict(observed={n:metric(v) for n,v in actual.items()},
                predicted={n:metric(v) for n,v in forecast.items()},errors=errors,
                signed_closure=metric(actual["image"]+actual["text"]+actual["interaction"]-actual["total"]))
    result=dict(schema="mdb-flux-stream-factorial-audit-v1",mechanics_status="verified",
        instrument_status="actual native ports and complete typed same-parent corner states; no ATen replay",
        trend_status="all corner/effect/interaction/locality/replay outcomes retained",terminal_status="unassessed",
        job_id=summary["job_id"],request_sha256=request["request_sha256"],source_job=config["source_job"],custody=custody,transaction=verified,
        counts=summary["counts"],parameter_recipe_joins=16,state_port_joins=joins,errors=comparisons,
        actual_port_residual_equations=equations,calibration_replay=calibration,structural_locality=locality,factorials=factorials,
        worker_seconds=summary["worker_seconds"],new_model_calls_in_audit=0,
        scope="Four same-run joint2 corners on3known source contexts; six new mixed internal states, six known calibration endpoints. Full1024image/512text outputs,32image/16text port witnesses. Observed-prefix conditional forecasts, no upstream/raw-prompt or semantic prediction claim.",
        audit_sources={p.name:sha(p) for p in (Path(__file__),here/"analyze_flux_attention_state_law.py",here/"flux_family_custody.py",here/"summarize_context_window.py")})
    with output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    return dict(mechanics_status=result["mechanics_status"],counts=result["counts"],
        exits={k:{s:v[s] for s in ("image","text")} for k,v in comparisons.items()},
        factorials={c:{s:factorials[c][s] for s in ("image","text")} for c in factorials},structural_locality=locality)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("collection",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();print(json.dumps(audit(a.collection.resolve(),a.output.resolve()),sort_keys=True))
