"""Verify scheduled row edits and native continuations without model execution."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from mdb_lanes.edit_search import file_digest, digest
from mdb_lanes.forecast_transaction import verify_transaction, load_forecast_tree


def analyze(run, source, output, metadata=None):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text())
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    worker=runpy.run_path(str(run/"payload/flux_binding_minimal.py"))
    data=run/"extracted/binding-minimal-result";config=read(data/"plan.json");indexes=data if metadata is None else metadata
    summary,rows,receipts,writes=[read(indexes/n)for n in("summary.json","rows.json","receipts.json","writes.json")]
    request=read(run/"request.json")
    assert config==request["config"]
    if metadata is None:assert summary==read(run/"report.json")["report"]
    else:
        recovery=read(metadata/"recovery.json")
        assert recovery["original_failure"]==read(run/"report.json")["report"] and recovery["original_failure"]["error"]=="ValueError: same-parent historical control differs"
        assert file_digest(data/"transaction/observations/native.json")==recovery["observation_sha256"] and file_digest(data/"files.json")==recovery["source_manifest_sha256"]
        decoder=runpy.run_path(str(here/"analyze_flux_eye_program.py"))["json_tree"]
        native=decoder(load_forecast_tree(data/"transaction",read(data/"transaction/observations/native.json")["value"]))
        assert native["rows"]==rows and native["counts"]==summary["counts"]
        for receipt in receipts:assert receipt["body"]==read(data/"store/receipts"/(receipt["body"]["fingerprint"]+".json"))
    for name,h in request["source_hashes"].items():assert file_digest(run/"payload"/name)==h
    for name,h in read(run/"payload/payload-seal.json")["wheels"].items():assert file_digest(run/"payload"/name)==h
    expected=worker["expected_counts"](config["arms"])
    assert summary["counts"]==expected and summary["target_entries"]==dict(blocks=71*len(rows))
    tx=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"]);assert tx["verified"] and tx["phase"]=="closed"
    tree=runpy.run_path(str(here/"analyze_flux_eye_program.py"))["json_tree"]
    frozen=tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/program_hypotheses.json")["predictions"]))
    assert frozen["target_entries_before_seal"]==dict(blocks=0) and frozen["hypotheses"]==config["hypotheses"]
    assert frozen["equation_forecasts"]==config["search"]
    search=config["search"];plan=search["search_plan"]
    assert plan["identity"]==digest({k:v for k,v in plan.items()if k!="identity"})
    assert plan["binding"]["parent"]==config["parents"]["base"] and plan["binding"]["model"]==config["checkpoint_sha256"]
    for arm in config["arms"]:
        f=search["freezes"][arm["id"]]
        assert f["identity"]==digest({k:v for k,v in f.items()if k!="identity"}) and f["plan"]==plan["identity"]
        assert f["candidate"]==digest(dict(plan=plan["identity"],atoms=f["atoms"]))
        if search.get("kind") in {"carrier-restoration","carrier-row-restoration"}:
            atoms={f"early.C{r}"for r in arm["rows_by_step"]["2"]}
            if search["kind"]=="carrier-row-restoration":
                if "image"not in arm["restore_slots"]:atoms.add("retain.image")
                groups=search["text_groups"]
                for group,ix in groups.items():
                    kept=set(arm.get("text_keep_rows",range(512)if"text"not in arm["restore_slots"]else[]))
                    assert not kept.intersection(ix) or set(ix)<=kept
                    if set(ix)<=kept:atoms.add("retain.text."+group)
                assert kept==set().union(*(set(ix)for group,ix in groups.items()if "retain.text."+group in atoms))
            else:atoms.update("retain."+s for s in ("image","text")if s not in arm["restore_slots"])
        else:atoms={f"row{row}.step{step}"for step,indices in arm["rows_by_step"].items()for row in indices}
        assert set(f["atoms"])==atoms
        assert f["prediction"]["categorical"]==config["hypotheses"]["categorical"][arm["id"]]
    assert frozen["counts_at_seal"]==dict(loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0)
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));store=typed["audit_store"](data/"store")
    index=runpy.run_path(str(here/"analyze_flux_edit_onset.py"))["index_receipts"]([r["body"]for r in receipts])
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    for r in receipts:public(r["body"])
    readslot=runpy.run_path(str(here/"analyze_flux_binding_minimal.py"))["read_slot"]
    slot=lambda fp,k:readslot(data/"store",fp,k)
    if "projection_forecasts"in search:
        cached=search["projection_forecasts"];origin=Path(cached["source"])
        assert file_digest(origin/"rows.json")==cached["source_rows_sha256"]
        cachedI=readslot(origin/"store",cached["base_cut"],"image")
        cachedT=readslot(origin/"store",cached["base_cut"],"text")
        editedT=readslot(origin/"store",cached["edited_cut"],"text")
        for arm in config["arms"]:
            predictedT=cachedT.copy();ix=arm["rows_by_step"]["2"];predictedT[:,ix]=editedT[:,ix]
            for field,independent in(("image",cachedI),("text",predictedT)):
                record=cached["fields"][arm["id"]][field]
                assert file_digest(record["path"])==record["sha256"] and np.array_equal(independent,np.load(record["path"],allow_pickle=False))
    def fields(fp):return {v["key"]["value"]:v["value"]for v in store["cuts"][fp]["payload"]["items"]}
    compare=runpy.run_path(str(here/"analyze_flux_concept_routes.py"))["comparison"]
    inv=runpy.run_path(str(here/"flux_scene_inventory.py"))["pixel_inventory"]
    original=source/"run-02/extracted/six-areas-result"
    base,target=[np.asarray(Image.open(original/(n+".png"))).copy() for n in ("b0r0","b1r0")]
    outcomes={};images={};commits={};equations={};projection_predictions={}
    B=typed["quantize_bf16"]
    for spec,row in zip(config["arms"],rows,strict=True):
        name=row["id"];assert all(row[k]==v for k,v in spec.items())
        parent=config["parents"]["base"];chain=[r["body"]for r in receipts if r["row"]==name]
        assert sum(r["operation"]=="continue" for r in chain)==79 and len(chain)==82+len(spec.get("restore_slots",[]))
        for r in chain:assert r==index[r["fingerprint"]] and r["parent"]==parent;parent=r["result"]
        assert parent==row["final"]
        for action in row["actions"]:
            before,after=fields(action["before"]),fields(action["after"]);s=action["slot"]
            assert all(before[k]==after[k]for k in before if k!=s)
            assert action["receipt"]==index[action["receipt"]["fingerprint"]]
            point=typed["unpack"](store["cuts"][action["before"]]["execution_point"])
            if "source_cut" in action:
                expected_point=("joint",spec["restore_index"]+1)if spec["restore_index"]<4 else("single",0)
                assert s in spec["restore_slots"] and point["logical_step"]==2 and (point["phase"],point["index"])==expected_point
                assert action["source_cut"]==rows[0]["marks"]["restore_boundary"]==row["marks"]["restoration_source"]
                expected_value=slot(action["source_cut"],s).copy()
                if "preserved_rows"in action:
                    assert s=="text"and action["preserved_rows"]==spec["text_keep_rows"]
                    ix=action["preserved_rows"];expected_value[:,ix]=slot(action["before"],s)[:,ix]
                assert np.array_equal(slot(action["after"],s),expected_value)
            elif s=="image":assert np.array_equal(slot(action["after"],s),slot(config["parents"]["donor"],s))
            else:
                assert s=="conditioning" and point["phase"]=="project" and point["logical_step"]in(2,3)
                c=slot(config["parents"]["base"],s).copy();indices=spec["rows_by_step"][str(point["logical_step"])]
                c[:,indices]=slot(config["parents"]["donor"],s)[:,indices]
                assert np.array_equal(slot(action["after"],s),c)
                assert np.array_equal(slot(row["marks"]["project"+str(point["logical_step"])+"_input"],s),c)
        # Actual ports independently close the BF16 block equation. These are
        # conditional observations, not independent suffix predictions.
        w=next(w for w in writes if w["row"]==name)
        if "projection_forecasts"in config["search"]:
            projection_predictions[name]={}
            for field,record in config["search"]["projection_forecasts"]["fields"][name].items():
                p=Path(record["path"]);assert file_digest(p)==record["sha256"]
                prediction=np.load(p,allow_pickle=False);actual=slot(w["entry"],field)
                projection_predictions[name][field]=dict(exact=bool(np.array_equal(prediction,actual)),
                    scalar_count=int(actual.size),rms=float(np.sqrt(np.mean((prediction.astype(float)-actual)**2))))
        ports={}
        for k,r in w["fields"].items():
            p=data/r["path"];assert file_digest(p)==r["sha256"];ports[k]=np.load(p,allow_pickle=False)
            if metadata is not None:assert p.name==name+".J00."+k+".npy"
        v=B(B(slot(w["entry"],"image")+B(ports["gate_attention"]*ports["attention"]))+B(ports["gate_mlp"]*ports["mlp"]))
        equations[name]=bool(np.array_equal(v,slot(w["exit"],"image")));assert equations[name]
        assert file_digest(data/row["png"])==row["png_sha256"]
        image=np.asarray(Image.open(data/row["png"])).copy();images[name]=image
        inventory,_=inv(image)
        sides={}
        for side,a in (("left",image[:,:256]),("right",image[:,256:])):
            f=a.astype(float)/255;mask=(f.max(-1)-f.min(-1)>.18)&(f.max(-1)>.12)
            counts={color:int(np.count_nonzero(mask&(f.argmax(-1)==channel)))for color,channel in (("red",0),("blue",2))}
            sides[side]=dict(counts=counts,dominant=max(counts,key=counts.get)if max(counts.values()) else None)
        predicted=config["hypotheses"]["categorical"][name]
        pred_left=predicted.split()[0];pred_right=predicted.split(" of a ")[-1].split()[0] if " of a "in predicted else predicted.split(" of ")[-1].split()[0]
        outcomes[name]=dict(rgb=compare(image,base,target),inventory=inventory,side_diagnostics=sides,
            prediction=predicted,predicted_colors=[pred_left,pred_right],observed_colors=[sides[k]["dominant"]for k in("left","right")],
            color_prediction_matched=[pred_left,pred_right]==[sides[k]["dominant"]for k in("left","right")],
            scope="hue components/half-frame dominance only; shapes and categorical sufficiency require image review")
        commits[name]={str(c["step"]):np.load(data/c["latent"]["path"],allow_pickle=False)for c in row["commits"]}
        assert all(np.array_equal(v,slot(c["cut"],"latent"))for c,v in zip(row["commits"],commits[name].values()))
    if rows[0]["id"]=="control":assert rows[0]["png_sha256"]==config["control_png_sha256"]
    prefix=next(iter(commits.values()))["2"]
    assert all(np.array_equal(v["2"],prefix)for v in commits.values())
    effects={name:{step:dict(rms=float(np.sqrt(np.mean((v.astype(float)-commits[rows[0]["id"]][step])**2))))for step,v in cs.items()}for name,cs in commits.items()}
    result=dict(schema="mdb-binding-search-audit-v1",mechanics_status="verified",instrument_status="diagnostic hue/position; manual shape review separate",terminal_status="unassessed",
        custody=custody,transaction=tx,counts=expected,target_entries=summary["target_entries"],equations_exact=equations,outcomes=outcomes,
        receiving_Z2_exact_across_arms=True,committed_effects=effects,new_model_calls_in_audit=0,worker_seconds=summary["worker_seconds"])
    result["projection_predictions"]=projection_predictions
    if search.get("kind") in {"carrier-restoration","carrier-row-restoration"}:
        baseline=rows[0];restorations={}
        for row in rows:
            name=row["id"];source_fp=baseline["marks"]["restore_boundary"]
            actual=fields(row["marks"]["post_restore"]);reference=fields(source_fp)
            before=fields(row["marks"]["restore_boundary"])
            at3=fields(row["marks"]["project3_input"]);ref3=fields(baseline["marks"]["project3_input"])
            final=fields(row["final"]);ref_final=fields(baseline["final"])
            restorations[name]=dict(source=source_fp,before_different_fields=[k for k in reference if before[k]!=reference[k]],
                post_restore_different_fields=[k for k in reference if actual[k]!=reference[k]],
                project3_different_fields=[k for k in ref3 if at3[k]!=ref3[k]],
                final_different_fields=[k for k in ref_final if final[k]!=ref_final[k]],
                final_payload_exact=final==ref_final,png_exact=row["png_sha256"]==baseline["png_sha256"],
                restored_slots=row["restore_slots"],restore_index=row["restore_index"],retained_text_rows=row.get("text_keep_rows"))
            trajectories=[]
            trace_rows=read(data/"traces.json")
            references={tuple(sorted(t["point"].items())):t for t in trace_rows if t["row"]==baseline["id"]}
            for t in trace_rows:
                if t["row"]!=name:continue
                ref=references[tuple(sorted(t["point"].items()))];measure={}
                for field in("image","text"):
                    assert t["fields"][field]["rows"]==ref["fields"][field]["rows"]
                    values=[]
                    for record in(t["fields"][field],ref["fields"][field]):
                        p=data/record["path"];assert file_digest(p)==record["sha256"]
                        values.append(np.load(p,allow_pickle=False))
                    delta=values[0].astype(float)-values[1]
                    measure[field]=dict(rms=float(np.sqrt(np.mean(delta**2))),max_abs=float(np.max(np.abs(delta))),
                        changed_scalars=int(np.count_nonzero(delta)),sample_rows=t["fields"][field]["rows"])
                trajectories.append(dict(point=t["point"],vs_control=measure))
            restorations[name]["sampled_trajectory"]=trajectories
        result["carrier_restoration"]=restorations
    if metadata is not None:result["post_completion_index_recovery"]=dict(metadata=str(metadata),recovery_sha256=file_digest(metadata/"recovery.json"),limitations=recovery["limitations"],original_job_state=recovery["original_job_state"])
    with output.open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    sheet=Image.new("RGB",(3*256,((len(images)+2)//3)*280),"white");draw=ImageDraw.Draw(sheet)
    for i,(name,image)in enumerate(images.items()):
        x,y=i%3*256,i//3*280;sheet.paste(Image.fromarray(image).resize((256,256)),(x,y+24));draw.text((x+4,y+6),name,fill="black")
    sheet.save(output.with_suffix(".png"))
    return {k:result[k]for k in("mechanics_status","counts","outcomes")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in("run","source","output"):p.add_argument(n,type=Path)
    p.add_argument("--metadata",type=Path)
    a=p.parse_args();v=analyze(a.run.resolve(),a.source.resolve(),a.output.resolve(),a.metadata.resolve()if a.metadata else None)
    print(json.dumps(dict(mechanics_status=v["mechanics_status"],counts=v["counts"],outcomes={k:dict(colors=x["observed_colors"],prediction_matched=x["color_prediction_matched"],rgb=x["rgb"])for k,x in v["outcomes"].items()}),sort_keys=True))
