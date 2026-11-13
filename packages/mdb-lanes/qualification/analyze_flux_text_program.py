"""Saved-only text equation, RAM intervention and native continuation audit."""
import argparse
import itertools
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/"extracted/text-program-result"
    old=runpy.run_path(str(here/"analyze_flux_eye_program.py"));json_tree=old["json_tree"]
    shared=runpy.run_path(str(here/"analyze_flux_scene_dependencies.py"));sha,metric=shared["sha"],shared["metric"]
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    summary,config,binding,rows,receipts=[read(data/n) for n in ("summary.json","plan.json","binding.json","rows.json","receipts.json")]
    request=read(run/"request.json");assert config==request["config"] and summary==read(run/"report.json")["report"]
    assert summary["request_sha256"]==request["request_sha256"]==custody["request_sha256"]
    for name,h in request["source_hashes"].items():assert sha(run/"payload"/name)==h
    tx=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"]);assert tx["verified"] and tx["phase"]=="closed"
    assert json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/frontiers/binding.json")["value"]))==binding
    sealed=json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/text_fields_and_RAM.json")["predictions"]))
    forecasts=read(data/"forecasts.json");assert sealed["fields"]==forecasts and sealed["hypotheses"]==config["hypotheses"]
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));routes=runpy.run_path(str(here/"analyze_flux_concept_routes.py"));store=typed["audit_store"](data/"store")
    def slot(cut,name):return routes["slot_array"](data/"store",cut,name,typed)
    def fields(cut):return {v["key"]["value"]:v["value"] for v in store["cuts"][cut]["payload"]["items"]}
    arr=lambda record:load_tensor_archive(data,record)
    native=read(data/"native-fields.json");independent=runpy.run_path(str(here/"analyze_flux_attention_replacement.py"));error=independent["field_error"]
    sourcefields={k:fields(v) for k,v in binding["entries"].items()}
    assert all(sourcefields["low"][k]==sourcefields["high"][k] for k in sourcefields["low"] if k!="text")
    equations={}
    for role in ("low","high"):
        t,i=arr(binding["inputs"][role]["text"]),arr(binding["inputs"][role]["image"])
        assert np.array_equal(t,slot(binding["entries"][role],"text")) and np.array_equal(i,slot(binding["entries"][role],"image"))
        tr=t.shape[1];packed=np.concatenate((i,t),axis=1);cos,sin=arr(binding["rope_cos"]),arr(binding["rope_sin"])
        a,_=independent["rebuild_attention"](packed,arr(binding["mod"]),arr(binding["qkv_weight"]),None if binding["qkv_bias"] is None else arr(binding["qkv_bias"]),
            text_rows=i.shape[1],heads=binding["heads"],norm_q_weight=arr(binding["norm_q"]),norm_k_weight=arr(binding["norm_k"]),eps=binding["eps"],qk_eps=binding["qk_eps"],
            rope_cos=np.concatenate((cos[tr:],cos[:tr])),rope_sin=np.concatenate((sin[tr:],sin[:tr])),queries=[0,31,127,511])
        m=independent["GENERIC"]["rebuild_full_mlp"](t,arr(binding["mod"]),arr(binding["mlp_weight"]),None if binding["mlp_bias"] is None else arr(binding["mlp_bias"]),
            eps=binding["eps"],mlp_width=9216,quantize=typed["quantize_bf16"])
        equations[role]={k:{"sealed_minus_independent":error(arr(forecasts[role][k]),value),"sealed_minus_native":error(arr(forecasts[role][k]),arr(native[role][k]))} for k,value in (("A",a),("M",m))}
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    for r in receipts:public(r)
    assert len(receipts)==182 and sum(r["operation"]=="continue" for r in receipts)==171 and sum(r["operation"]=="apply" for r in receipts)==11
    by_receipt={r["fingerprint"]:r for r in receipts};images,latents={},{}
    assert len(rows)==13 and len({r["id"] for r in rows})==13
    raw={r["id"]:r for r in rows};common_parent=None
    for row in rows:
        assert sha(data/row["png"])==row["png_sha256"]
        with Image.open(data/row["png"]) as im:assert im.mode=="RGB" and im.size==(512,512);images[row["id"]]=np.asarray(im).copy()
        latents[row["id"]]=arr(row["latent"]);assert np.array_equal(latents[row["id"]],slot(row["final_cut"],"latent"))
        previous=None
        for r in row["path"]:
            assert r==by_receipt[r["fingerprint"]]
            if previous:assert r["parent"]==previous
            previous=r["result"]
        assert previous==row["final_cut"] and len(row["path"])==(14 if row["id"].startswith("native.") else 13)
        for action in row["actions"]:
            assert action["slot"]=="text" and action["act"]["writes"]==["text"] and action["receipt"]==by_receipt[action["receipt"]["fingerprint"]]
            before,after=fields(action["parent"]),fields(action["post_act"])
            assert all(before[k]==after[k] for k in before if k!="text")
            if common_parent is None:common_parent=action["parent"]
            assert action["parent"]==common_parent and row["path"][0]["parent"]==action["post_act"]
    controls={name:sha(data/(name+".png"))==h for name,h in config["historical_pngs"].items()};assert all(controls.values())
    assert np.array_equal(slot(raw["R0A0M0"]["actions"][-1]["post_act"],"text"),arr(native["low"]["text_exit"]))
    assert np.array_equal(slot(raw["R1A1M1"]["actions"][-1]["post_act"],"text"),arr(native["high"]["text_exit"]))
    shams={name:np.array_equal(images[name],images["native.high"]) and np.array_equal(latents[name],latents["native.high"]) for name in ("R1A1M1","sham.high")};assert all(shams.values())
    def norm(x):return float(np.linalg.norm(np.asarray(x,np.float64).reshape(-1)))
    dr=arr(binding["inputs"]["high"]["text"]).astype(np.float64)-arr(binding["inputs"]["low"]["text"])
    dout=arr(native["high"]["text_exit"]).astype(np.float64)-arr(native["low"]["text_exit"])
    effective=dout-dr
    carried={"incoming_text_contrast_l2":norm(dr),"outgoing_text_contrast_l2":norm(dout),"effective_update_contrast_l2":norm(effective),
        "effective_update_over_incoming_contrast":norm(effective)/norm(dr),"scope":"Effectiveupdate=Tnative_exit-Rentry includes residual-addition BF16rounding; it is not rawgate*projection. Normratios are not causal percentages."}
    observations={}
    def window(im,box):x0,y0,x1,y1=box;return im[y0:y1,x0:x1]
    for name,im in images.items():
        views={"whole":(im,images["native.low"],images["native.high"])}
        views.update({k:(window(im,box),window(images["native.low"],box),window(images["native.high"],box)) for k,box in config["observers"]["windows"].items()})
        observations[name]={}
        for label,(v,lo,hi) in views.items():
            d=hi.astype(np.float64)-lo;energy=float(np.sum(d*d))
            observations[name][label]={"to_low":metric(v,lo),"to_high":metric(v,hi),"high_contrast_projection":float(np.sum((v.astype(np.float64)-lo)*d)/energy) if energy else None}
    effects={}
    for variable,axis in (("R",0),("A",1),("M",2)):
        effects[variable]=[]
        for other in itertools.product((0,1),repeat=2):
            bits=list(other);bits.insert(axis,0);low_name="R%dA%dM%d"%tuple(bits);bits[axis]=1;high_name="R%dA%dM%d"%tuple(bits)
            effects[variable].append({"low":low_name,"high":high_name,"rgb":metric(images[high_name],images[low_name]),
                "eye_projection_delta":observations[high_name]["eyes"]["high_contrast_projection"]-observations[low_name]["eyes"]["high_contrast_projection"]})
    prediction_pairs={role:{"rgb":metric(images["predicted."+role],images[corner]),"latent":metric(latents["predicted."+role],latents[corner])} for role,corner in (("low","R0A0M0"),("high","R1A1M1"))}
    report={"schema":"mdb-text-program-audit-v1","job_id":summary["job_id"],"request_sha256":request["request_sha256"],"mechanics_status":"verified","instrument_status":"applicable",
        "trend_status":"preserve-per-case","terminal_status":"unassessed","custody":custody,"store_counts":store["counts"],"transaction":tx,"counts":summary["counts"],
        "historical_controls":controls,"shams":shams,"writer_validations":read(data/"writer-validations.json"),"full_text_equations":equations,"carried_contrast":carried,
        "common_edited_parent":common_parent,"observations":observations,"conditional_RAM_effects":effects,"prediction_pairs":prediction_pairs,"worker_seconds":summary["worker_seconds"],
        "coverage":"Complete textA/M independentreference reconstruction and nativefield errorcensus, actualtext-only Acts/commonimageparent/cornerclosure,13nativeRGB. Packedattention math checked against originalorder tiny native; FP64key reductionorder differs. Nativeprojection/gate/residual and source states retained. No sourceprogramformation forecast, semanticsegmentation or generalized eye-mechanism claim."}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x") as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    for kind in ("full","eyes"):
        width,height=(384,415) if kind=="full" else (460,150);sheet=Image.new("RGB",(4*width,4*height),(245,245,245));draw=ImageDraw.Draw(sheet)
        for index,row in enumerate(rows):
            im=Image.fromarray(images[row["id"]]);x,y=(index%4)*width,(index//4)*height
            im=im.resize((384,384)) if kind=="full" else im.crop((140,155,370,215)).resize((460,120))
            sheet.paste(im,(x,y+26));draw.text((x+5,y+7),row["id"],fill="black")
        sheet.save(destination.parent/("text-program-"+kind+".png"))
    return {"audit":str(destination),"carried_contrast":carried,"prediction_pairs":prediction_pairs,
        "RAM_eye_deltas":{k:[v["eye_projection_delta"] for v in values] for k,values in effects.items()},
        "equation_relative_rms":{role:{k:values["sealed_minus_native"]["relative_rms"] for k,values in body.items()} for role,body in equations.items()}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve()),sort_keys=True))
