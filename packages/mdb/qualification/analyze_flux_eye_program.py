"""Saved-only audit of contextual program composition and numerical forecasts."""
import argparse
import json
import runpy
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(p):return json.loads(p.read_text())


def json_tree(value):
    # Public loader deliberately freezes JSON lists as tuples and dicts as
    # mapping proxies. Compare their JSON values without discarding metadata.
    if isinstance(value,Mapping):return {k:json_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_tree(v) for v in value]
    return value


def analyze(run,destination):
    from mdb.forecast_transaction import verify_transaction,load_forecast_tree
    from mdb.tensor_archive import load_tensor_archive
    here=Path(__file__).resolve().parent;data=run/"extracted/eye-program-result"
    shared=runpy.run_path(str(here/"analyze_flux_scene_dependencies.py"));sha,metric=shared["sha"],shared["metric"]
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    summary,config,rows,receipts,binding=[read(data/n) for n in ("summary.json","plan.json","rows.json","receipts.json","binding.json")]
    request=read(run/"request.json")
    assert summary==read(run/"report.json")["report"] and config==request["config"]
    assert summary["request_sha256"]==request["request_sha256"]==custody["request_sha256"]
    for n,h in request["source_hashes"].items():assert sha(run/"payload"/n)==h
    transaction=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"])
    assert transaction["verified"] and transaction["phase"]=="closed"
    assert json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/frontiers/binding.json")["value"]))==binding
    sealed=json_tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/additive_fields.json")["predictions"]))
    assert sealed["fields"]==read(data/"forecasts.json") and sealed["hypotheses"]==config["hypotheses"]
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));q=typed["quantize_bf16"]
    routes=runpy.run_path(str(here/"analyze_flux_concept_routes.py"))
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    store=typed["audit_store"](data/"store")
    def slot(cut,key):return routes["slot_array"](data/"store",cut,key,typed)
    def fields(cut):return {i["key"]["value"]:i["value"] for i in store["cuts"][cut]["payload"]["items"]}
    def arr(r):return load_tensor_archive(data,r)
    for r in receipts:public(r)
    assert len(receipts)==209 and sum(r["operation"]=="continue" for r in receipts)==196
    assert sum(r["operation"]=="apply" for r in receipts)==13
    by_receipt={r["fingerprint"]:r for r in receipts};images,latents={},{}
    assert len(rows)==13 and len({r["id"] for r in rows})==13
    for row in rows:
        assert sha(data/row["png"])==row["png_sha256"]
        with Image.open(data/row["png"]) as im:
            assert im.mode=="RGB" and im.size==(512,512)
            images[row["id"]]=np.asarray(im).copy()
        latents[row["id"]]=arr(row["latent"])
        assert np.array_equal(latents[row["id"]],slot(row["final_cut"],"latent"))
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
        assert len(row["path"])==(14 if row["id"]=="additive.predictedAM" else 15)
    controls={name:sha(data/(name+".png"))==h for name,h in config["historical_pngs"].items()}
    assert all(controls.values())
    base=config["input_cuts"]["post.base"]["cut"];prep=read(data/"preparation.json")
    color,shape=[slot(prep[role]["post_cut"],"text") for role in ("color","shape")]
    additive=q((color+shape-slot(base,"text")).astype(np.float32))
    assert np.array_equal(additive,slot(binding["parent"],"text"))
    assert np.array_equal(additive,arr(binding["text"])) and np.array_equal(slot(base,"image"),arr(binding["image"]))
    # Distinct implementation rebuilds full fields from operands; native outputs
    # participate only in the separate error census below.
    independent=runpy.run_path(str(here/"analyze_flux_attention_replacement.py"))
    rebuilt_a,_=independent["rebuild_attention"](np.concatenate((arr(binding["text"]),arr(binding["image"])),axis=1),arr(binding["mod"]),arr(binding["qkv_weight"]),
        None if binding["qkv_bias"] is None else arr(binding["qkv_bias"]),text_rows=512,heads=binding["heads"],norm_q_weight=arr(binding["norm_q"]),norm_k_weight=arr(binding["norm_k"]),
        eps=binding["layernorm_eps"],qk_eps=binding["qk_eps"],rope_cos=arr(binding["rope_cos"]),rope_sin=arr(binding["rope_sin"]),queries=independent["QUERIES"])
    rebuilt_m=independent["GENERIC"]["rebuild_full_mlp"](arr(binding["image"]),arr(binding["mod"]),arr(binding["mlp_weight"]),
        None if binding["mlp_bias"] is None else arr(binding["mlp_bias"]),eps=binding["layernorm_eps"],mlp_width=9216,quantize=q)
    predictions=read(data/"forecasts.json");observations=read(data/"field-observations.json")
    field_error=independent["field_error"];numerical={}
    for key,rebuilt in (("A",rebuilt_a),("M",rebuilt_m)):
        predicted,measured=arr(predictions[key]),arr(observations["fields"][key])
        numerical[key]={"sealed_minus_independent":field_error(predicted,rebuilt),"sealed_minus_native":field_error(predicted,measured)}
        # Reconstruction disagreement is retained as an instrument observation,
        # not a semantic failure or an uncalibrated scientific quality gate.
    comparisons={name:{"to_base":metric(im,images["native.base"]),"to_full_program":metric(im,images["full.program"])} for name,im in images.items()}
    pairs={name:{"rgb":metric(images[a],images[b]),"latent":metric(latents[a],latents[b])} for name,a,b in (
        ("predicted_vs_measured_additive","additive.predictedAM","additive"),
        ("additive_vs_joint_program","additive","text.only"),
        ("image_only_vs_base","image.only","native.base"),
        ("text_only_vs_full_program","text.only","full.program"))}
    report={"schema":"mdb-eye-program-audit-v1","job_id":summary["job_id"],"request_sha256":request["request_sha256"],
        "mechanics_status":"verified","instrument_status":"applicable","trend_status":"preserve-per-case","terminal_status":"unassessed",
        "custody":custody,"store_counts":store["counts"],"transaction":transaction,"historical_controls":controls,"counts":summary["counts"],
        "program_composition_exact":True,"field_errors":numerical,"text_dependence_observations":{k:v for k,v in observations.items() if k!="fields"},
        "rgb_comparisons":comparisons,"pairs":pairs,"worker_seconds":summary["worker_seconds"],
        "coverage":"Complete A/M fields independently rebuilt, complete native error census. Post-block7 text and native output projection remain observed consumers; only image A/M independently forecast. Every image uses native suffix and VAE. Semantic review is unblinded and separate. One seed/context/late clock; no automatic object segmentation or global edit guarantee."}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x") as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    sheet=Image.new("RGB",(4*384,4*420),(240,240,240));draw=ImageDraw.Draw(sheet)
    for index,row in enumerate(rows):
        x,y=(index%4)*384,(index//4)*420;sheet.paste(Image.fromarray(images[row["id"]]).resize((384,384)),(x,y+30));draw.text((x+8,y+8),row["id"],fill=(0,0,0))
    sheet.save(destination.parent/"eye-program-futures.png")
    return {"audit":str(destination),"pairs":pairs,"text_dependence":report["text_dependence_observations"],"field_errors":{k:{name:{f:v for f,v in value.items() if f!="per_position_error_rms"} for name,value in item.items()} for k,item in numerical.items()}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve()),sort_keys=True))
