"""Saved-only independent conditioning composition and linear text projection audit."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination,source):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/"extracted/conditioning-program-result";origin=source/"run-01/extracted/predicted-future-result"
    old=runpy.run_path(str(here/"analyze_flux_eye_program.py"));json_tree=old["json_tree"]
    util=runpy.run_path(str(here/"analyze_flux_scene_dependencies.py"));sha,metric=util["sha"],util["metric"]
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    summary,config,binding,rows,receipts=[read(data/n) for n in ("summary.json","plan.json","binding.json","rows.json","receipts.json")]
    request=read(run/"request.json");assert config==request["config"] and summary==read(run/"report.json")["report"]
    assert summary["request_sha256"]==request["request_sha256"]==custody["request_sha256"]
    for name,h in request["source_hashes"].items():assert sha(run/"payload"/name)==h
    for name,meta in config["input_files"].items():assert (origin/name).stat().st_size==meta["bytes"] and sha(origin/name)==meta["sha256"]
    tx=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"]);assert tx["verified"] and tx["phase"]=="closed"
    assert json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/frontiers/binding.json")["value"]))==binding
    sealed=json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/conditioning_to_text.json")["predictions"]))
    forecast=read(data/"forecasts.json");assert sealed["initial_text"]==forecast and sealed["hypotheses"]==config["hypotheses"]
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));q=typed["quantize_bf16"]
    route=runpy.run_path(str(here/"analyze_flux_concept_routes.py"));store=typed["audit_store"](data/"store")
    def slot(cut,name):return route["slot_array"](data/"store",cut,name,typed)
    def fields(cut):return {v["key"]["value"]:v["value"] for v in store["cuts"][cut]["payload"]["items"]}
    arr=lambda record:load_tensor_archive(data,record)
    values={k:arr(v) for k,v in binding["conditioning"].items()}
    for role in ("base","color","shape","joint"):
        expected=route["slot_array"](origin/"store",config["input_cuts"][role],"conditioning",typed);assert np.array_equal(expected,values[role])
    assert np.array_equal(values["additive"],q((values["color"].astype(np.float64)+values["shape"]-values["base"]).astype(np.float32)))
    assert np.array_equal(values["interaction_only"],q((values["base"].astype(np.float64)+values["joint"]-values["additive"]).astype(np.float32)))
    native=read(data/"native-fields.json");error=runpy.run_path(str(here/"analyze_flux_mlp_replacement.py"))["field_error"]
    weight=arr(binding["weight"]);bias=None if binding["bias"] is None else arr(binding["bias"]);equations={}
    for role,value in values.items():
        rebuilt=np.empty((*value.shape[:2],weight.shape[0]),np.float32)
        for start in range(0,weight.shape[0],128):
            stop=min(start+128,weight.shape[0]);tile=value.astype(np.float64)@weight[start:stop].astype(np.float64).T
            if bias is not None:tile+=bias[start:stop].astype(np.float64)
            rebuilt[:,:,start:stop]=q(tile.astype(np.float32))
        prediction=arr(forecast[role]);equations[role]={"sealed_minus_independent":error(prediction,rebuilt),"sealed_minus_native":error(prediction,arr(native[role]))}
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    for r in receipts:public(r)
    assert len(receipts)==242 and sum(r["operation"]=="continue" for r in receipts)==234 and sum(r["operation"]=="apply" for r in receipts)==8
    by_receipt={r["fingerprint"]:r for r in receipts};images,latents={},{}
    assert len(rows)==8 and len({r["id"] for r in rows})==8
    raw={r["id"]:r for r in rows}
    for row in rows:
        assert sha(data/row["png"])==row["png_sha256"]
        with Image.open(data/row["png"]) as im:assert im.mode=="RGB" and im.size==(512,512);images[row["id"]]=np.asarray(im).copy()
        latents[row["id"]]=arr(row["latent"]);assert np.array_equal(latents[row["id"]],slot(row["final_cut"],"latent"))
        previous=None
        for r in row["path"]:
            assert r==by_receipt[r["fingerprint"]]
            if previous:assert r["parent"]==previous
            previous=r["result"]
        assert previous==row["final_cut"] and len(row["path"])==(28 if row["id"].startswith("native.") else 27)
        for action in row["actions"]:
            assert action["receipt"]==by_receipt[action["receipt"]["fingerprint"]]
            assert action["act"]["writes"]==[action["slot"]]
            before,after=fields(action["parent"]),fields(action["post_act"]);assert all(before[k]==after[k] for k in before if k!=action["slot"])
            assert row["path"][0]["parent"]==action["post_act"]
        if row["id"].startswith("native."):
            role=row["id"].split(".",1)[1];assert row["actions"][0]["slot"]=="conditioning" and row["actions"][0]["parent"]==binding["derived_project_frontier"]
            assert np.array_equal(slot(row["actions"][0]["post_act"],"conditioning"),values[role])
            assert np.array_equal(slot(row["path"][0]["result"],"text"),arr(native[role]))
        else:
            role=row["id"].split(".",1)[1];assert row["actions"][0]["slot"]=="text" and row["actions"][0]["parent"]==raw["native."+role]["path"][0]["result"]
            assert np.array_equal(slot(row["actions"][0]["post_act"],"text"),arr(forecast[role]))
    reconstruction=read(data/"reconstruction.json");assert fields(reconstruction["reconstructed"])=={i["key"]["value"]:i["value"] for i in read(origin/"store/cuts"/(config["input_cuts"]["base"]+".json"))["manifest"]["payload"]["items"]}
    assert sha(data/"native.base.png")==config["historical_base_png"]
    def window(v,box):x0,y0,x1,y1=box;return v[y0:y1,x0:x1]
    observations={}
    for name,image in images.items():
        views={"whole":(image,images["native.additive"],images["native.joint"])}
        views.update({k:(window(image,box),window(images["native.additive"],box),window(images["native.joint"],box)) for k,box in config["observers"]["windows"].items()})
        observations[name]={}
        for label,(value,base,target) in views.items():
            d=target.astype(np.float64)-base;energy=float(np.sum(d*d));observations[name][label]={"to_additive":metric(value,base),"to_joint":metric(value,target),
                "joint_contrast_projection":float(np.sum((value.astype(np.float64)-base)*d)/energy) if energy else None}
    pairs={role:{"rgb":metric(images["predicted."+role],images["native."+role]),"latent":metric(latents["predicted."+role],latents["native."+role])} for role in ("additive","joint")}
    report={"schema":"mdb-conditioning-program-audit-v1","job_id":summary["job_id"],"request_sha256":request["request_sha256"],"mechanics_status":"verified","instrument_status":"applicable",
        "trend_status":"preserve-per-case","terminal_status":"unassessed","custody":custody,"transaction":tx,"store_counts":store["counts"],"counts":summary["counts"],
        "reconstruction":reconstruction,"historical_base_png_exact":True,"full_projection_equations":equations,"observations":observations,"prediction_pairs":pairs,"worker_seconds":summary["worker_seconds"],
        "coverage":"All6complete initialT references independently rebuilt;6nativefinalstepRGB/2predictedTnativeconsumer futures; commonbaseL/time/history and conditioning-only Acts; exactderivedfrontier12-stepreconstruction. Sources are observedpromptencodings; no rawtextencoder prediction, fullprompt trajectory match, independent specimens or semantic segmentation."}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x") as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    for kind in ("full","eyes"):
        width,height=(384,415) if kind=="full" else (460,150);sheet=Image.new("RGB",(4*width,2*height),(245,245,245));draw=ImageDraw.Draw(sheet)
        for index,row in enumerate(rows):
            im=Image.fromarray(images[row["id"]]);x,y=(index%4)*width,(index//4)*height
            im=im.resize((384,384)) if kind=="full" else im.crop((140,155,370,215)).resize((460,120))
            sheet.paste(im,(x,y+26));draw.text((x+5,y+7),row["id"],fill="black")
        sheet.save(destination.parent/("conditioning-"+kind+".png"))
    return {"audit":str(destination),"projection_relative_rms":{k:v["sealed_minus_native"]["relative_rms"] for k,v in equations.items()},"prediction_pairs":pairs,
        "image_observations":{k:{"eye_projection":v["eyes"]["joint_contrast_projection"],"whole_joint_mae":v["whole"]["to_joint"]["mae"]} for k,v in observations.items()}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("output",type=Path);p.add_argument("--source",type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve(),a.source.resolve()),sort_keys=True))
