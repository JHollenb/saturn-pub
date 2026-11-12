"""Verify predicted causal writes, native candidates and complete native futures."""
import argparse
import hashlib
import json
import runpy
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_slot(root, fingerprint, name):
    cut=json.loads((root/"cuts"/(fingerprint+".json")).read_text())
    page=json.loads((root/"pages"/(cut["slots"][name]+".json")).read_text())
    tensors={}
    if page["tensor_blob"] is not None:
        with (root/"blobs"/(page["tensor_blob"]+".safetensors")).open("rb") as f:
            n=struct.unpack("<Q",f.read(8))[0];header=json.loads(f.read(n));payload=f.read()
        for key,item in header.items():
            if key=="__metadata__":continue
            a,b=item["data_offsets"];raw=payload[a:b]
            if item["dtype"]=="BF16":v=(np.frombuffer(raw,"<u2").astype(np.uint32)<<16).view(np.float32)
            else:v=np.frombuffer(raw,dict(F32="<f4",F64="<f8",I64="<i8",I32="<i4",BOOL="?",F16="<f2")[item["dtype"]])
            tensors[key]=v.reshape(item["shape"])
    def decode(v):
        k=v["kind"]
        if k=="tensor":return tensors[v["name"]]
        if k=="mapping":return {a:decode(b)for a,b in v["items"].items()}
        if k in ("tuple","list"):return [decode(x)for x in v["items"]]
        if k=="scalar":return v["value"]
        raise ValueError("unknown native page kind")
    return decode(page["encoded"])


def analyze(run,source,forecasts,output):
    from mdb.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    worker=runpy.run_path(str(run/"payload/flux_binding_writers.py"))
    data=run/"extracted/binding-writers-result";config,summary,rows,receipts,writes=[read(data/n)for n in ("plan.json","summary.json","rows.json","receipts.json","writes.json")]
    assert config==read(run/"request.json")["config"] and summary==read(run/"report.json")["report"]
    expected=worker["expected_counts"](config["arms"]);assert summary["counts"]==expected and len(receipts)==expected["receipts"]
    assert summary["target_entries"]==dict(blocks=71*len(rows)+expected["candidate_blocks"])
    tx=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"]);assert tx["verified"] and tx["phase"]=="closed"
    tree=runpy.run_path(str(here/"analyze_flux_eye_program.py"))["json_tree"]
    frozen=tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/program_hypotheses.json")["predictions"]))
    assert frozen["target_entries_before_seal"]==dict(blocks=0)
    assert frozen["counts_at_seal"]==dict(loads=1,encodes=0,noise_draws=0,native_transitions=0,acts=0,receipts=0,images=0,candidate_blocks=0)
    assert frozen["hypotheses"]==config["hypotheses"] and frozen["equation_forecasts"]==read(forecasts/"manifest.json")
    assert sha(forecasts/"manifest.json")==config["equation_forecasts_sha256"]
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));store=typed["audit_store"](data/"store")
    by=runpy.run_path(str(here/"analyze_flux_edit_onset.py"))["index_receipts"]([x["body"]for x in receipts])
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    for r in receipts:public(r["body"])
    original=source/"run-02/extracted/six-areas-result";oldbinding=source/"binding-02/extracted/binding-program-result"
    compare=runpy.run_path(str(here/"analyze_flux_concept_routes.py"))["comparison"]
    error=runpy.run_path(str(here/"flux_six_areas.py"))["error"]
    inv=runpy.run_path(str(here/"flux_scene_inventory.py"))["pixel_inventory"]
    images={k:np.asarray(Image.open(original/(k+".png"))).copy() for k in ("b0r0","b1r0")}
    slot=lambda fp,k:read_slot(data/"store",fp,k)
    def fields(fp):return {v["key"]["value"]:v["value"] for v in store["cuts"][fp]["payload"]["items"]}
    def array(record,root=data):
        p=root/record["path"];assert sha(p)==record["sha256"];a=np.load(p,allow_pickle=False)
        assert list(a.shape)==record["shape"] and str(a.dtype)==record["dtype"]
        return a
    outcomes={};equations={};ports={};prefix_checks={};scored={};commits={};texts={}
    candidates={r["row"]:r for r in read(data/"candidates.json")}
    reference=run.parent/"streams-02/extracted/binding-minimal-result"
    assert sha(run.parent/"streams-analysis-v1.json")==config["writer_manifest"]["source_audit_sha256"]
    reference_rows={r["id"]:r for r in read(reference/"rows.json")}
    native_equivalence={}

    B=typed["quantize_bf16"]
    for spec,row in zip(config["arms"],rows,strict=True):
        name=row["id"];assert all(row[k]==v for k,v in spec.items())
        parent=config["parents"]["base"]
        chain=[r["body"]for r in receipts if r["row"]==name]
        assert sum(r["operation"]=="continue" for r in chain)==79
        for r in chain:assert r==by[r["fingerprint"]] and r["parent"]==parent;parent=r["result"]
        assert parent==row["final"] and row["parent"]==config["parents"]["base"]
        for action in row["actions"]:
            before,after=fields(action["before"]),fields(action["after"]);s=action["slot"]
            assert all(before[k]==after[k]for k in before if k!=s)
            assert action["receipt"]==by[action["receipt"]["fingerprint"]]
        baseC=slot(config["parents"]["base"],"conditioning");donorC=slot(config["parents"]["donor"],"conditioning")
        expectedC=baseC.copy()
        if spec["C"]!="base":
            indices=list(range(512))if spec["C"]=="all"else config["conditioning_rows"][spec["C"]];expectedC[:,indices]=donorC[:,indices]
        assert np.array_equal(slot(row["marks"]["project2_input"],"conditioning"),expectedC)
        assert np.array_equal(slot(row["final"],"conditioning"),expectedC)
        for a in row["actions"]:
            if a["after"]==row["marks"].get("post_writer_Act"):
                candidate=candidates[name]
                expected_value=(array(candidate["native_candidate"]) if name=="A.recomputed.native" else load_tensor_archive(forecasts,config["writer_manifest"]["vectors"][name]))
                assert np.array_equal(slot(a["after"],"image"),expected_value)
            elif a["slot"]!="conditioning":
                assert np.array_equal(slot(a["after"],a["slot"]),slot(config["parents"]["donor"],a["slot"]))
            else:
                assert np.array_equal(slot(a["after"],"conditioning"),expectedC)
                point=typed["unpack"](store["cuts"][a["before"]]["execution_point"])
                assert point["phase"]=="project" and point["logical_step"]==2
        entry,exit=row["marks"]["joint0_entry"],row["marks"]["joint0_exit"]
        context=slot(entry,"context");texts[name]=slot(entry,"text")
        prefix_checks[name]=all(np.array_equal(slot(entry,k),read_slot(reference/"store",reference_rows["IC"]["marks"]["joint0_entry"],k)) for k in ("image","text","latent","conditioning"))
        assert prefix_checks[name]
        if name in candidates:
            c=candidates[name];assert c["entry"]==entry
            vecname="AM.predicted" if name=="A.recomputed.native" else name
            predicted=load_tensor_archive(forecasts,config["writer_manifest"]["vectors"][vecname])
            assert np.array_equal(predicted,array(c["predicted"]))
            observed=array(c["native_candidate"])
            equations[name]=error(predicted,observed)
            assert c["predicted_image_exact"]==bool(np.array_equal(predicted,observed))
            assert np.array_equal(array(c["native_candidate_text"]),slot(exit,"text"))
        w=next(w for w in writes if w["row"]==name);assert w["entry"]==entry and w["exit"]==exit
        p={k:array(r)for k,r in w["fields"].items()}
        computed=B(B(slot(entry,"image")+B(p["gate_attention"]*p["attention"]))+B(p["gate_mlp"]*p["mlp"]))
        assert np.array_equal(computed,slot(exit,"image"))
        ports[name]=p
        assert [c["step"]for c in row["commits"]]==[2,3,4]
        commits[name]={}
        for c in row["commits"]:
            v=array(c["latent"]);assert np.array_equal(v,slot(c["cut"],"latent"));commits[name][str(c["step"])]=v
        images[name]=np.asarray(Image.open(data/row["png"])).copy();assert sha(data/row["png"])==row["png_sha256"]
        inventory,_=inv(images[name]);major={k:max([c for c in inventory["components"]if c["color"]==k],key=lambda c:c["area"],default=None)for k in ("red","blue")}
        redleft=None if any(x is None for x in major.values())else major["red"]["centroid"][0]<major["blue"]["centroid"][0]
        outcomes[name]=dict(rgb=compare(images[name],images["b0r0"],images["b1r0"]),inventory=inventory,red_centroid_left_of_blue=redleft)
        if name in config["hypotheses"]["categorical"]:
            expectedred=config["hypotheses"]["categorical"][name].startswith("red sphere")
            scored[name]=dict(prediction=config["hypotheses"]["categorical"][name],predicted_red_left=expectedred,observed_red_left=redleft,matched=None if redleft is None else redleft==expectedred)
    assert rows[0]["png_sha256"]==config["control_png_sha256"]
    if config["panel"]=="streams":assert rows[-1]["png_sha256"]==config["ITC_png_sha256"]
    first=rows[0]["id"]
    port_effects={name:{k:error(p[k],ports[first][k])for k in p}for name,p in ports.items()}
    commit_effects={name:{step:compare(v,commits[first][step],commits[rows[-1]["id"]][step])for step,v in cs.items()}for name,cs in commits.items()}
    source_text_equal={name:bool(np.array_equal(v,texts[rows[-1]["id"]]))for name,v in texts.items()}
    traces=read(data/"traces.json")
    assert all(set(config["conditioning_rows"]["color"])<=set(r["fields"]["text"]["rows"])for r in traces)
    byrow={r["id"]:r for r in rows}
    pair=[byrow[n] for n in ("AM.predicted","A.recomputed.native")]
    native_equivalence=dict(
        complete_post_Act_payload_exact=fields(pair[0]["marks"]["post_writer_Act"])==fields(pair[1]["marks"]["post_writer_Act"]),
        final_payload_exact=fields(pair[0]["final"])==fields(pair[1]["final"]),
        png_exact=pair[0]["png_sha256"]==pair[1]["png_sha256"])
    reference_projection=outcomes["IC.sham"]["rgb"]["target_projection"]
    signed_scores={name:dict(predicted_lower_projection=True,observed_projection=v["rgb"]["target_projection"],
                             reference_projection=reference_projection,matched=v["rgb"]["target_projection"]<reference_projection)
                   for name,v in outcomes.items() if name!="IC.sham"}
    rank=dict(predicted_M_lower_than_A=True,
              observed_M_lower_than_A=outcomes["M.fixed"]["rgb"]["target_projection"]<outcomes["A.fixed"]["rgb"]["target_projection"])
    result=dict(schema="mdb-binding-writer-audit-v1",mechanics_status="verified",instrument_status="chromatic components; manual shape review separate",terminal_status="unassessed",
        custody=custody,counts=expected,target_entries=summary["target_entries"],transaction=tx,prefix_exact=prefix_checks,numerical_forecasts=equations,
        native_sequential_writer_equations_exact=True,lexical_color_rows_observed=True,project2_text_equal_to_last=source_text_equal,
        mathematical_vs_native_future=native_equivalence,signed_prediction_scores=signed_scores,signed_rank_prediction=rank,
        outcomes=outcomes,frozen_prediction_scores=scored,port_effects_vs_first=port_effects,committed_effects_vs_first=commit_effects,
        new_model_calls_in_audit=0,worker_seconds=summary["worker_seconds"],scope="Saved full native operands feed elementary BF16 causal write calculations sealed before suffixes. Four native candidate blocks are explicitly counted. AM mathematical write and native A recomputation are compared through complete post-Act/final payloads and PNG. One exact parent/seed; no independent edited-RGB forecast.")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    names=["b0r0","b1r0",*[r["id"]for r in rows]];sheet=Image.new("RGB",(4*256,((len(names)+3)//4)*282),"#f5f5f5");draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(names):
        x,y=i%4*256,i//4*282;sheet.paste(Image.fromarray(images[name]).resize((256,256)),(x,y+26));draw.text((x+4,y+7),name,fill="black")
    sheet.save(output.parent/"binding-writers.png")
    return dict(mechanics_status="verified",counts=expected,predictions=scored,write_errors=equations,mathematical_vs_native_future=native_equivalence,signed_prediction_scores=signed_scores,signed_rank_prediction=rank)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","source","forecasts","output"):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.source.resolve(),a.forecasts.resolve(),a.output.resolve()),sort_keys=True))

