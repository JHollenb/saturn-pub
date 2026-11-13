"""Saved-only audit and descriptive scoring of current versus continuing binding."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def analyze(root,destination):
    from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
    here=Path(__file__).resolve().parent;run=root/"binding-02";data=run/"extracted/binding-program-result";old=root/"run-02/extracted/six-areas-result"
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    config,summary,rows,emissions=[read(data/n) for n in ("plan.json","summary.json","rows.json","receipts.json")]
    assert read(run/"request.json")["config"]==config and summary==read(run/"report.json")["report"]
    assert summary["request_sha256"]==custody["request_sha256"]
    assert config["source_audit_sha256"]==sha(root/"analysis-v1.json")
    for name,v in config["input_files"].items():assert sha(old/name)==v["sha256"] and (old/name).stat().st_size==v["bytes"]
    tx=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"]);assert tx["verified"] and tx["phase"]=="closed"
    tree=runpy.run_path(str(here/"analyze_flux_eye_program.py"))["json_tree"]
    frozen=tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/program_hypotheses.json")["predictions"]))
    assert frozen["hypotheses"]==config["hypotheses"] and frozen["arms"]==config["arms"]
    assert frozen["counts_at_seal"]==dict(loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0)
    expected=dict(loads=1,encodes=0,noise_draws=0,native_transitions=711,acts=13,receipts=724,images=9)
    assert summary["counts"]==expected and len(emissions)==724 and len(rows)==9
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    for r in emissions:public(r["body"])
    by=runpy.run_path(str(here/"analyze_flux_edit_onset.py"))["index_receipts"]([r["body"] for r in emissions])
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));store=typed["audit_store"](data/"store")
    routes=runpy.run_path(str(here/"analyze_flux_concept_routes.py"));slot=lambda fp,s:routes["slot_array"](data/"store",fp,s,typed)
    def fields(fp):return {v["key"]["value"]:v["value"] for v in store["cuts"][fp]["payload"]["items"]}
    def arr(record):
        p=data/record["path"];assert sha(p)==record["sha256"];a=np.load(p,allow_pickle=False);assert list(a.shape)==record["shape"] and str(a.dtype)==record["dtype"];return a
    base,donor=[config["parents"][s] for s in ("base","donor")];baseC,donorC=[slot(fp,"conditioning") for fp in (base,donor)]
    images={name:np.asarray(Image.open(old/(name+".png"))).copy() for name in ("b0r0","b1r0")}
    inv=runpy.run_path(str(here/"flux_scene_inventory.py"))["pixel_inventory"]
    outcomes={};predictions={};footprints={};commit_fields={}
    for spec,row in zip(config["arms"],rows,strict=True):
        name=row["id"];assert all(row[k]==v for k,v in spec.items()) and row["parent"]==base
        parent=base;chain=[r["body"] for r in emissions if r["row"]==name]
        assert sum(r["operation"]=="continue" for r in chain)==79
        for r in chain:assert r==by[r["fingerprint"]] and r["parent"]==parent;parent=r["result"]
        assert parent==row["final"]
        expectedC=baseC.copy()
        if spec["C"]!="base":
            indices=list(range(512)) if spec["C"]=="all" else config["conditioning_rows"][spec["C"]]
            expectedC[:,indices]=donorC[:,indices]
        for action in row["actions"]:
            s=action["slot"];assert action["receipt"]==by[action["receipt"]["fingerprint"]] and action["act"]["writes"]==[s]
            before,after=fields(action["before"]),fields(action["after"])
            assert all(before[k]==after[k] for k in before if k!=s)
            expectedvalue=expectedC if s=="conditioning" else slot(base if name=="sham" else donor,s)
            assert np.array_equal(slot(action["after"],s),expectedvalue)
            if s=="conditioning":
                point=typed["unpack"](store["cuts"][action["before"]]["execution_point"]);assert point["logical_step"]==2 and point["phase"]=="project"
        assert np.array_equal(slot(row["final"],"conditioning"),expectedC)
        assert [v["step"] for v in row["commits"]]==[2,3,4]
        for c in row["commits"]:assert np.array_equal(arr(c["latent"]),slot(c["cut"],"latent"))
        commit_fields[name]={str(c["step"]):arr(c["latent"]) for c in row["commits"]}
        assert np.array_equal(arr(row["latent"]),slot(row["final"],"latent")) and sha(data/row["png"])==row["png_sha256"]
        images[name]=np.asarray(Image.open(data/row["png"])).copy();inventory,_=inv(images[name])
        major={color:max([c for c in inventory["components"] if c["color"]==color],key=lambda c:c["area"],default=None) for color in ("red","blue")}
        redleft=None if any(v is None for v in major.values()) else major["red"]["centroid"][0]<major["blue"]["centroid"][0]
        outcomes[name]=dict(rgb=routes["comparison"](images[name],images["b0r0"],images["b1r0"]),inventory=inventory,
            red_centroid_left_of_blue=redleft,scope="chromatic position diagnostic; shape binding and object count need visual inspection")
        if name in ("C.all","ITC"):predictions[name]=dict(predicted_red_left=False,observed_red_left=redleft,matched=None if redleft is None else not redleft,claim="H1 color-position component only; geometry separately reviewed")
        if name in ("I","T","IT"):predictions[name]=dict(predicted_red_left=True,observed_red_left=redleft,matched=redleft,claim="H2 color-position component only; geometry separately reviewed")
        footprints[name]=dict(actions=len(row["actions"]),complete_undeclared_closure_exact=True,future_conditioning_exact=True)
    assert np.array_equal(images["sham"],images["b0r0"]) and summary["sham_exact"]
    old_observed=read(old/"observed.json");committed_effects={}
    def oldarr(record):
        p=old/record["path"];assert sha(p)==record["sha256"];return np.load(p,allow_pickle=False)
    for name,steps in commit_fields.items():
        committed_effects[name]={}
        for step,value in steps.items():
            b=oldarr(old_observed["b0r0"]["step_latents"][int(step)-1]["latent"])
            d=oldarr(old_observed["b1r0"]["step_latents"][int(step)-1]["latent"])
            committed_effects[name][step]=routes["comparison"](value,b,d)
    traces=read(data/"traces.json");text_reads={}
    for r in traces:
        point=r["point"]
        if point["phase"]=="joint" and point["index"]==0 and point["logical_step"] in (2,3):
            text_reads.setdefault(str(point["logical_step"]),{})[r["row"]]=arr(r["fields"]["text"])
    sampled_text_reset={step:{name:bool(np.array_equal(value,byrow["sham"])) for name,value in byrow.items()} for step,byrow in text_reads.items()}
    report=dict(schema="mdb-binding-program-audit-v1",job_id=summary["job_id"],request_sha256=summary["request_sha256"],mechanics_status="verified",
        instrument_status="chromatic components,manual shape review pending",trend_status="all cases and forecast misses retained",terminal_status="unassessed",
        custody=custody,transaction=tx,counts=expected,distinct_receipt_ids=len(by),footprints=footprints,
        outcomes=outcomes,frozen_prediction_scores=predictions,committed_effects=committed_effects,sampled_text_reset=sampled_text_reset,
        text_witness_scope="8 uniformly spaced text rows; changed color-token rows are not directly sampled. Equality of witnesses is not whole-T equality; use C Acts/declared rowwise projection and scene outcomes separately.",worker_seconds=summary["worker_seconds"],
        scope="Common step1/joint4 parent,carried base latent. Current I/T transplanted before J4; repeated conditioning first changed at project2. Observed donor states; no independent numerical forecasts of edited futures. Valid/padding/color arm outcomes are exploratory.")
    with destination.open("x") as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    names=["b0r0","b1r0",*[r["id"] for r in rows]];w,hp=256,282;sheet=Image.new("RGB",(4*w,3*hp),"#f5f5f5");draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(names):
        x,y=i%4*w,i//4*hp;sheet.paste(Image.fromarray(images[name]).resize((256,256)),(x,y+26));draw.text((x+4,y+7),name,fill="black")
    sheet.save(destination.parent/"six-areas-binding-program.png")
    return dict(job_id=summary["job_id"],counts=expected,predictions=predictions,RGB={k:v["rgb"] for k,v in outcomes.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.root.resolve(),a.output.resolve()),sort_keys=True))
