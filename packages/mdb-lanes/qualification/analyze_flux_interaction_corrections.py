"""Saved-only custody, correction algebra, continuation and window observations."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def analyze(run,destination):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb_lanes.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/"extracted/interaction-corrections-result"
    old=runpy.run_path(str(here/"analyze_flux_eye_program.py"));json_tree=old["json_tree"]
    shared=runpy.run_path(str(here/"analyze_flux_scene_dependencies.py"));sha,metric=shared["sha"],shared["metric"]
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    summary,config,rows,receipts,binding=[read(data/n) for n in ("summary.json","plan.json","rows.json","receipts.json","binding.json")]
    request=read(run/"request.json")
    assert summary==read(run/"report.json")["report"] and config==request["config"]
    assert summary["request_sha256"]==request["request_sha256"]==custody["request_sha256"]
    for n,h in request["source_hashes"].items():assert sha(run/"payload"/n)==h
    tx=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert tx["verified"] and tx["phase"]=="closed"
    assert json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/frontiers/binding.json")["value"]))==binding
    sealed=json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/correction_panel.json")["predictions"]))
    forecasts=read(data/"forecasts.json");assert sealed["fields"]==forecasts and sealed["branches"]==config["branches"] and sealed["hypotheses"]==config["hypotheses"]
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));q=typed["quantize_bf16"]
    routes=runpy.run_path(str(here/"analyze_flux_concept_routes.py"));store=typed["audit_store"](data/"store")
    def slot(cut,key):return routes["slot_array"](data/"store",cut,key,typed)
    def fields(cut):return {i["key"]["value"]:i["value"] for i in store["cuts"][cut]["payload"]["items"]}
    arr=lambda r:load_tensor_archive(data,r)
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    for r in receipts:public(r)
    assert len(receipts)==228 and sum(r["operation"]=="continue" for r in receipts)==213 and sum(r["operation"]=="apply" for r in receipts)==15
    by_receipt={r["fingerprint"]:r for r in receipts};images,latents={},{}
    assert len(rows)==15 and len({r["id"] for r in rows})==15
    for row in rows:
        assert sha(data/row["png"])==row["png_sha256"]
        with Image.open(data/row["png"]) as im:assert im.mode=="RGB" and im.size==(512,512);images[row["id"]]=np.asarray(im).copy()
        latents[row["id"]]=arr(row["latent"]);assert np.array_equal(latents[row["id"]],slot(row["final_cut"],"latent"))
        previous=None
        for r in row["path"]:
            assert r==by_receipt[r["fingerprint"]]
            if previous:assert r["parent"]==previous
            previous=r["result"]
        assert previous==row["final_cut"]
        for act in row["actions"]:
            assert act["receipt"]==by_receipt[act["receipt"]["fingerprint"]]
            before,after=fields(act["parent"]),fields(act["post_act"])
            assert before.keys()==after.keys() and act["act"]["writes"]==[act["slot"]]
            assert all(before[k]==after[k] for k in before if k!=act["slot"])
        if row["actions"]:assert row["path"][0]["parent"]==row["actions"][-1]["post_act"]
        assert len(row["path"])==(15 if row["id"].startswith("native.") else 14)
    raw={r["id"]:r for r in rows}
    controls={name:sha(data/(name+".png"))==h for name,h in config["historical_pngs"].items()};assert all(controls.values())
    shams={p:np.array_equal(images[p+".sham"],images["native."+p]) and np.array_equal(latents[p+".sham"],latents["native."+p]) for p in ("additive","joint")};assert all(shams.values())
    # Prior fields are data supplied with exact hashes, not newly observed
    # block outputs. Independently verify each correction formula and cast.
    source={}
    for role,meta in config["prediction_fields"].items():
        if role=="additive":
            file=run/"source-fields"/"additive.npy"
            # The new sealed r0c1 field must equal the old file/native hashes.
            source[role]=arr(forecasts["r0c1"])
            assert forecasts["r0c1"]["descriptor"]["sha256"]==meta["descriptor"]["sha256"]
        else:
            file=run/"payload"/meta["path"];assert sha(file)==meta["sha256"]
            source[role]=(np.load(file,allow_pickle=False).astype(np.uint32)<<16).view(np.float32)
    wide={k:v.astype(np.float64) for k,v in source.items()};linear=wide["color"]+wide["shape"]-wide["base"]
    c=wide["additive"]-linear;r=wide["joint"]-wide["additive"]
    for name,rdose,cdose in (("r0c0",0,0),("r0c1",0,1),("r1c0",1,0),("r1c1",1,1),("r-1c0",-1,0),("r0c-1",0,-1)):
        assert np.array_equal(arr(forecasts[name]),q((linear+rdose*r+cdose*c).astype(np.float32)))
    error=runpy.run_path(str(here/"analyze_flux_mlp_replacement.py"))["field_error"]
    measured=read(data/"native-fields.json");field_errors={role:error(source[role],arr(meta)) for role,meta in measured.items()}
    # The matched program grid changes only outgoing T: candidate I is shared.
    matched_state={}
    for field in ("r0c0","r0c1","r1c0","r1c1"):
        lo,hi=[raw[p+"."+field]["actions"][-1]["post_act"] for p in ("additive","joint")]
        before,after=fields(lo),fields(hi)
        differing=[k for k in before if before[k]!=after[k]]
        assert differing==["text"] or set(differing)=={"text"}
        matched_state[field]={"parents":[lo,hi],"differing_payload_slots":differing}
    def window(value,box):
        x0,y0,x1,y1=box;return value[y0:y1,x0:x1]
    references={p:images["native."+p] for p in ("base","additive","joint")}
    assert sha(data/"native.joint.png")==config["observers"]["signed_contrast_reference_pngs"]["joint"]
    assert sha(data/"native.additive.png")==config["observers"]["signed_contrast_reference_pngs"]["additive"]
    observations={}
    for name,image in images.items():
        groups={"whole":(image,references["additive"],references["joint"])}
        groups.update({k:(window(image,box),window(references["additive"],box),window(references["joint"],box)) for k,box in config["observers"]["windows"].items()})
        observations[name]={}
        for key,(v,base,target) in groups.items():
            delta=target.astype(np.float64)-base;actual=v.astype(np.float64)-base;energy=float(np.sum(delta*delta))
            observations[name][key]={"to_joint":metric(v,target),"to_additive":metric(v,base),"joint_contrast_projection":float(np.sum(actual*delta)/energy) if energy else None,
                "contrast_energy":energy,"scope":"Projection is0at additive and1at joint; supplied historical RGB contrast includes color/geometry/shading and is not a semantic classifier."}
    factorial={}
    for program in ("additive","joint"):
        factorial[program]={}
        for window_name in observations["native.base"]:
            scores={k:observations[program+"."+k][window_name]["joint_contrast_projection"] for k in ("r0c0","r0c1","r1c0","r1c1")}
            factorial[program][window_name]={"R_at_C0":scores["r1c0"]-scores["r0c0"],"R_at_C1":scores["r1c1"]-scores["r0c1"],
                "C_at_R0":scores["r0c1"]-scores["r0c0"],"C_at_R1":scores["r1c1"]-scores["r1c0"],
                "mixed":scores["r1c1"]-scores["r1c0"]-scores["r0c1"]+scores["r0c0"]}
    matched_program_effects={k:{"rgb":metric(images["joint."+k],images["additive."+k]),
        "joint_mae_reduction":observations["additive."+k]["whole"]["to_joint"]["mae"]-observations["joint."+k]["whole"]["to_joint"]["mae"],
        "eye_projection_change":observations["joint."+k]["eyes"]["joint_contrast_projection"]-observations["additive."+k]["eyes"]["joint_contrast_projection"]} for k in matched_state}
    middle={}
    for row in rows:
        record=row["intermediate"];assert record["point"]["index"]==11 and record["point"]["phase"]=="single"
        if row["id"].startswith("native."):
            for slot_name,ref in record["slots"].items():
                value=arr(ref["array"]);assert typed["matches"](value,ref["descriptor"],bf16=True)
                np.testing.assert_array_equal(value.reshape(-1)[::max(1,value.size//64)][:64],ref["witness_flat"])
        else:middle[row["id"]]=record["comparisons"]
    validations=read(data/"projection-validations.json")
    report={"schema":"mdb-interaction-corrections-audit-v1","job_id":summary["job_id"],"request_sha256":request["request_sha256"],
        "mechanics_status":"verified","instrument_status":"applicable","trend_status":"preserve-per-case","terminal_status":"unassessed",
        "custody":custody,"store_counts":store["counts"],"transaction":tx,"historical_controls":controls,"shams":shams,"counts":summary["counts"],
        "projection_validations":validations,"correction_algebra":"all6complete fields exact to independently assembled/cast recipe","full_attention_errors":field_errors,
        "matched_program_payloads":matched_state,"observations":observations,"factorial_signed_contrasts":factorial,"matched_program_effects":matched_program_effects,
        "intermediate_single11":middle,"worker_seconds":summary["worker_seconds"],
        "coverage":"New intervention outcomes; known specimen/source-assisted fields. Complete native attention error census for5roles,4source-bound full-block consumer comparisons. FixednativeM/text/projection retained, not independently predicted.3fullreferenceI/T observations atsingle11; candidates have worker metrics/descriptor64witnesses without retained complete intermediate arrays. Pixelwindows/historicalcontrast are supplied observers; semanticreview separate."}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x") as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    for kind in ("full","eyes"):
        width,height=(384,420) if kind=="full" else (460,146)
        sheet=Image.new("RGB",(4*width,4*height),(245,245,245));draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            im=Image.fromarray(images[row["id"]]);x,y=(i%4)*width,(i//4)*height
            if kind=="eyes":im=im.crop(tuple(config["observers"]["windows"]["eyes"])).resize((460,120))
            else:im=im.resize((384,384))
            sheet.paste(im,(x,y+26));draw.text((x+5,y+7),row["id"],fill="black")
        sheet.save(destination.parent/("corrections-"+kind+".png"))
    return {"audit":str(destination),"matched_program_effects":matched_program_effects,"factorial_eyes":{p:v["eyes"] for p,v in factorial.items()},
        "image_errors":{k:{"whole_joint_mae":v["whole"]["to_joint"]["mae"],"eye_alpha":v["eyes"]["joint_contrast_projection"]} for k,v in observations.items()},
        "attention_relative_rms":{k:v["relative_rms"] for k,v in field_errors.items()},"projection_validations":validations}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve()),sort_keys=True))
