"""Independent saved-only six-area audit; no model execution or Torch import."""
from __future__ import annotations
import argparse
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def read(path):return json.loads(path.read_text())


def analyze(run,destination):
    from mdb.forecast_transaction import verify_transaction,load_forecast_tree
    here=Path(__file__).resolve().parent; data=run/"extracted/six-areas-result"
    custody=runpy.run_path(str(here/"flux_family_custody.py"))["verify"](run)
    h=runpy.run_path(str(run/"payload/flux_six_areas.py")); error=h["error"]
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py")); q=typed["quantize_bf16"]
    routes=runpy.run_path(str(here/"analyze_flux_concept_routes.py")); compare=routes["comparison"]
    tree=runpy.run_path(str(here/"analyze_flux_eye_program.py"))["json_tree"]
    public=runpy.run_path(str(here/"summarize_context_window.py"))["public_receipt"]
    index=runpy.run_path(str(here/"analyze_flux_edit_onset.py"))["index_receipts"]
    inventory=runpy.run_path(str(here/"flux_scene_inventory.py"))["pixel_inventory"]
    summary,config,forecasts,observed,witness,rows,tokens,receipts,raw,decoder=[read(data/n) for n in (
        "summary.json","plan.json","forecasts.json","observed.json","witnesses.json","rows.json","tokens.json","receipts.json","raw-pixels.json","decoder-inputs.json")]
    request=read(run/"request.json"); assert summary==read(run/"report.json")["report"] and config==request["config"]
    assert summary["request_sha256"]==request["request_sha256"]==custody["request_sha256"]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def arr(record):
        p=data/record["path"]; assert sha(p)==record["sha256"]
        a=np.load(p,allow_pickle=False); assert list(a.shape)==record["shape"] and str(a.dtype)==record["dtype"] and np.isfinite(a).all()
        return a
    tx=verify_transaction(data/"transaction",expected_seal_sha256=summary["seal"]["seal_sha256"])
    dtx=verify_transaction(data/"decoder-transaction",expected_seal_sha256=summary["decoder_seal"]["seal_sha256"])
    assert tx["verified"] and dtx["verified"] and tx["phase"]==dtx["phase"]=="closed"
    frozen=tree(load_forecast_tree(data/"transaction",read(data/"transaction/forecasts/own_state_math.json")["predictions"]))
    assert frozen["fields"]==forecasts and frozen["hypotheses"]==config["hypotheses"] and frozen["native_targets_before_seal"]==0 and not frozen["encoder_predicted"]
    assert frozen["parameter_recipes_sha256"]==sha(data/"parameter-recipes.json")
    df=tree(load_forecast_tree(data/"decoder-transaction",read(data/"decoder-transaction/forecasts/doses.json")["predictions"]))
    assert df["inputs"]==decoder
    timeline=[json.loads(s) for s in (data/"timeline.jsonl").read_text().splitlines()]
    assert [v["sequence"] for v in timeline]==list(range(len(timeline)))
    sealed=next(v for v in timeline if v["kind"]=="sealed")
    assert sealed["guard"]==dict(released=False,blocked=0,target_calls=0)
    assert sealed["counts"]["native_transitions"]==sealed["counts"]["candidate_blocks"]==sealed["counts"]["images"]==0
    assert len([v for v in timeline if v["kind"]=="forecast_complete" and v["sequence"]<sealed["sequence"]])==4
    assert all(v["sequence"]>sealed["sequence"] for v in timeline if v["kind"]=="image_complete")
    expected=dict(loads=1,encodes=4,noise_draws=1,native_transitions=1111,acts=9,receipts=1120,candidate_blocks=11,images=27,native_vae=17,fp32_vae=10)
    assert summary["counts"]==expected and len(rows)==27
    assert summary["guard"]==dict(released=True,blocked=0,target_calls=1007)
    for r in receipts:public(r["body"])
    by=index([r["body"] for r in receipts]); assert len(receipts)==1120
    assert sum(r["body"]["operation"]=="continue" for r in receipts)==1111
    store=typed["audit_store"](data/"store")
    def slot(fp,name):return routes["slot_array"](data/"store",fp,name,typed)
    def fields(fp):return {v["key"]["value"]:v["value"] for v in store["cuts"][fp]["payload"]["items"]}
    numerical={}; state_factorials={}; conditioning={}; mlp_equations={}; route_checks={}; route_mass={}
    for name,t in tokens.items():
        conditioning[name]=arr(t["conditioning"])
        assert conditioning[name].shape==(1,512,7680) and t["packing_exact"] and t["layers"]==[9,18,27]
        assert len(t["input_ids"])==len(t["attention_mask"])==len(t["tokens"])==len(t["offsets"])==512
        for word,indices in t["groups"].items():
            if word=="padding":assert indices==[i for i,v in enumerate(t["attention_mask"]) if not v]
            else:
                for i in indices:
                    a,b=t["offsets"][i]; assert word.strip() in t["formatted"][max(0,a-8):min(len(t["formatted"]),b+8)]
        numerical[name]={}
        for boundary,key in (("project","project"),("joint0","joint")):
            fp=observed[name][boundary]
            numerical[name][boundary]={k:error(arr(forecasts[name][key][k]),slot(fp,k)) for k in ("image","text")}
        assert numerical[name]["project"]==observed[name]["project_errors"] and numerical[name]["joint0"]==observed[name]["joint_errors"]
        qr=observed[name]["joint4_sources"]; qkv={k:arr(v) for k,v in qr["qkv"].items()}
        ref=h["route_witness"](qkv,512,t["groups"])
        route_checks[name]={k:bool(np.array_equal(v,arr(qr["route"][k]))) for k,v in ref.items()}
        assert all(route_checks[name].values())
        route_mass[name]={k:dict(mean=float(v.mean()),max=float(v.max()),
            key_count=len(t["groups"].get(k,[])) if k not in {"entropy","all_text","all_image"} else (512 if k=="all_text" else (1024 if k=="all_image" else None)),
            scope="head/query average mass; number of key rows is a confound") for k,v in ref.items() if k not in {"probability","rows"}}
    for boundary in ("project","joint0","joint4_entry","joint4_post"):
        state_factorials[boundary]={s:h["factorial"]({name:slot(v[boundary],s) for name,v in observed.items()}) for s in ("image","text")}
    state_factorials["conditioning"]={f"layer{layer}":h["factorial"]({name:v[:,:,i*2560:(i+1)*2560] for name,v in conditioning.items()}) for i,layer in enumerate((9,18,27))}
    contextual_formation={}
    for lo,hi in (("b0r0","b1r0"),("b0r0","b0r1"),("b0r1","b1r1"),("b1r0","b1r1")):
        key=lo+"->"+hi; contextual_formation[key]={}
        ids0,ids1=tokens[lo]["input_ids"],tokens[hi]["input_ids"]
        unchanged=[i for i,(a,b) in enumerate(zip(ids0,ids1)) if a==b and tokens[lo]["attention_mask"][i] and tokens[hi]["attention_mask"][i]]
        changed=[i for i,(a,b) in enumerate(zip(ids0,ids1)) if a!=b]
        padding=sorted(set(tokens[lo]["groups"]["padding"])&set(tokens[hi]["groups"]["padding"]))
        for i,layer in enumerate((9,18,27)):
            x,y=conditioning[lo][:,:,i*2560:(i+1)*2560],conditioning[hi][:,:,i*2560:(i+1)*2560]
            contextual_formation[key][str(layer)]={label:dict(rows=len(indices),**error(y[:,indices],x[:,indices]))
                for label,indices in (("unchanged_token_ID_valid_rows",unchanged),("changed_token_ID_rows",changed),("common_padding_rows",padding)) if indices}
    for step in range(4):
        state_factorials[f"z{step+1}"]=h["factorial"]({name:arr(v["step_latents"][step]["latent"]) for name,v in observed.items()})
    # Independent local MLP equations. Affine projections use FP64 reductions;
    # nonlinear checks retain native paired inputs, so scopes stay separate.
    def affine(x,w,b):
        y=x.astype(np.float64)@w.astype(np.float64).T
        if b is not None:y+=b.astype(np.float64)
        return q(y)
    def silu(x):
        x=x.astype(np.float64); positive=x>=0; y=np.empty_like(x)
        y[positive]=x[positive]/(1+np.exp(-x[positive])); e=np.exp(x[~positive]);y[~positive]=x[~positive]*e/(1+e)
        return q(y)
    for r in witness["records"]:
        w=witness["weights"][r["law"]]; a={k:arr(v) for k,v in r["fields"].items()}; hidden=w["hidden"]
        pin=affine(a["input"],arr(w["paired_weight"]),None if w["paired_bias"] is None else arr(w["paired_bias"]))
        target=np.concatenate((a["paired"][...,:16],a["paired"][...,hidden:hidden+16]),axis=-1)
        activation=q(silu(a["paired"][...,:hidden])*a["paired"][...,hidden:])
        native_activation=a["consumer_input"][...,w["attention_width"]:] if w["attention_width"] else a["consumer_input"]
        pout=affine(a["consumer_input"],arr(w["output_weight"]),None if w["output_bias"] is None else arr(w["output_bias"]))
        mlp_equations[r["id"]]=dict(selected_paired_affine=error(pin,target),activation_from_observed_paired=error(activation,native_activation),
            selected_output_affine=error(pout,a["output"][...,:16]),
            input_rms=float(np.sqrt(np.mean(a["input"].astype(np.float64)**2))),
            activation_rms=float(np.sqrt(np.mean(native_activation.astype(np.float64)**2))),
            output_rms=float(np.sqrt(np.mean(a["output"].astype(np.float64)**2))),
            law=r["law"],scope="8 sampled spatial rows;16 independently projected neuron pairs/16 output channels; full sampled nonlinear activation conditional on observed paired preactivation")
    assert len(mlp_equations)==12
    images={}; inventories={}; rowmap={r["id"]:r for r in rows}; latent_comparisons={}; interventions={}
    installed_exits={}; write_equations={}
    for r in rows:
        p=data/r["png"]; assert sha(p)==r["png_sha256"]
        with Image.open(p) as im:
            assert im.mode=="RGB" and im.size==(512,512); images[r["id"]]=np.asarray(im).copy()
        inventories[r["id"]]=inventory(images[r["id"]])[0]
        if "latent" in r:assert np.array_equal(arr(r["latent"]),slot(r["final"],"latent"))
        if r["kind"]=="image-only-causal-port":
            action=r["action"]; assert action["receipt"]==by[action["receipt"]["fingerprint"]]
            assert action["act"]["writes"]==["image"]
            before,after=fields(action["parent"]),fields(action["post_act"])
            assert all(before[k]==after[k] for k in before if k!="image")
            assert before["text"]==fields(observed["b0r0"]["joint4_post"])["text"]
            p={k:arr(v) for k,v in r["ports"].items()};spec=r["spec"];donor=observed[spec["source"]]["joint4_ports"]
            usedA=arr(donor["attention"]) if spec.get("attention") else p["attention"]
            usedM=arr(donor["mlp"]) if spec.get("mlp") else p["mlp"]
            R=slot(observed["b0r0"]["joint4_entry"],"image")
            expected_exit=q(q(R+q(p["gate_attention"]*usedA))+q(p["gate_mlp"]*usedM))
            installed_exits[r["id"]]=slot(action["post_act"],"image")
            write_equations[r["id"]]=error(expected_exit,installed_exits[r["id"]]);assert write_equations[r["id"]]["exact"]
        if r["id"]=="math.bridge":
            for action,role in zip(r["actions"],("image","text"),strict=True):
                assert action["receipt"]==by[action["receipt"]["fingerprint"]]
                before,after=fields(action["before"]),fields(action["after"])
                assert all(before[k]==after[k] for k in before if k!=role)
                assert np.array_equal(slot(action["after"],role),arr(forecasts["b0r0"]["joint"][role]))
    assert np.array_equal(images["route.sham"],images["b0r0"]) and np.array_equal(images["decode.native.+0.00"],images["b0r0"])
    for spec in config["interventions"]:
        name=spec["id"]; interventions[name]=compare(images[name],images["b0r0"],images["b1r0"])
        latent_comparisons[name]=compare(arr(rowmap[name]["latent"]),arr(rowmap["b0r0"]["latent"]),arr(rowmap["b1r0"]["latent"]))
    base_inventory,base_labels=inventory(images["b0r0"])
    collateral={}
    for spec in config["interventions"]:
        name=spec["id"]; delta=np.abs(images[name].astype(np.float64)-images["b0r0"].astype(np.float64));collateral[name]={}
        for c in base_inventory["components"]:
            mask=base_labels==c["component_label"]
            collateral[name][c["id"]]=dict(base_color=c["color"],pixels=int(mask.sum()),mae=float(delta[mask].mean()),
                scope="base chromatic component footprint; displaced/new object pixels outside footprint reported separately")
        background=base_labels==0
        collateral[name]["base_nonchromatic_pixels"]=dict(pixels=int(background.sum()),mae=float(delta[background].mean()))
    traces=read(data/"trajectories.json"); indexed={}
    for r in traces:
        if "image" in r:
            point=r["point"];clock=(point["logical_step"],point["phase"],point["index"])
            indexed.setdefault(r["history"],{})[clock]=r
    propagation={}
    for name in [s["id"] for s in config["interventions"]]+["math.bridge"]:
        propagation[name]=[]
        for point,r in indexed[name].items():
            if point in indexed["b0r0"]:
                a=arr(r["image"]);b=arr(indexed["b0r0"][point]["image"])
                propagation[name].append(dict(point=r["point"],sampled_image_effect=error(a,b)))
    bridge=compare(images["math.bridge"],images["b0r0"],images["b1r0"])
    # Attention transplantation changes the joint MLP's receiving state. The
    # ideal FP32 interaction of these particular writer arms is -gM*(MA-M0),
    # plus the independently visible finite BF16 additions/rounding residual.
    I0=slot(observed["b0r0"]["joint4_post"],"image").astype(np.float64)
    IA,IM,IAM=[installed_exits[k].astype(np.float64) for k in ("write.attention","write.mlp","write.both")]
    interaction=IAM-IA-IM+I0
    M0=arr(observed["b0r0"]["joint4_ports"]["mlp"]).astype(np.float64)
    MA=arr(rowmap["write.attention"]["ports"]["mlp"]).astype(np.float64)
    gm=arr(rowmap["write.attention"]["ports"]["gate_mlp"]).astype(np.float64)
    nonlinear_mlp_term=-gm*(MA-M0)
    writer_interaction=dict(state_interaction_rms=float(np.sqrt(np.mean(interaction**2))),
        recomputed_MLP_term_rms=float(np.sqrt(np.mean(nonlinear_mlp_term**2))),
        finite_rounding_remainder=error(interaction,nonlinear_mlp_term),
        all_QKV_and_attention_return_equivalence=dict(image_exit_exact=bool(np.array_equal(installed_exits["route.qkv"],installed_exits["write.attention"])),
            PNG_exact=bool(np.array_equal(images["route.qkv"],images["write.attention"]))),
        scope="Specific donor-A/donor-M arms at a fixed recipient; replacing A recomputes M unless M explicitly locked. Downstream RGB has additional nonlinear interactions.")
    # Symmetric curvature is computed on raw pre-PIL pixels, not uint8 output.
    # Continuous FP32 curvature excludes packed-input BF16 rounding. Differences
    # between same-lane continuous/rounded outputs isolate that input rounding.
    decoder_effects={}; decoder_curvature={}
    rawnp={k:arr(v) for k,v in raw.items()}
    rms=lambda v:float(np.sqrt(np.mean(np.asarray(v,np.float64)**2)))
    for key,rec in decoder.items():
        z=arr(rec["continuous"]); b=arr(rec["rounded"]); assert np.array_equal(q(z),b)
        assert error(b,z)==rec["rounding"]
        native=rawnp["decode.native."+key]; rounded=rawnp["decode.fp32."+key+".rounded"]; continuous=rawnp["decode.fp32."+key+".continuous"]
        decoder_effects[key]=dict(latent_input_rounding=error(b,z),
            same_FP32_consumer_input_rounding_effect=error(rounded,continuous),
            BF16_native_preparation_and_consumer_vs_FP32=error(native,rounded),
            FP32_continuous_response_rms=rms(continuous-rawnp["decode.fp32.+0.00.continuous"]))
    for dose in (.05,.1):
        plus,minus=f"{dose:+.2f}",f"{-dose:+.2f}"
        decoder_curvature[str(dose)]={}
        for lane,suffix in (("native",""),("fp32",".rounded"),("fp32_continuous",".continuous")):
            label="fp32" if lane.startswith("fp32") else "native"
            a,b,c=[rawnp["decode."+label+"."+key+suffix].astype(np.float64) for key in (plus,minus,"+0.00")]
            curvature=a+b-2*c; response=(a-b)/2
            decoder_curvature[str(dose)][lane]=dict(curvature_rms=rms(curvature),antisymmetric_response_rms=rms(response),
                curvature_over_response=rms(curvature)/rms(response) if rms(response) else None)
    report=dict(schema="mdb-six-areas-audit-v1",job_id=summary["job_id"],request_sha256=summary["request_sha256"],
        mechanics_status="verified",instrument_status="equations,route-witnesses,chromatic-components;shape identity visual only",
        trend_status="per-area results and counterexamples retained",terminal_status="unassessed",custody=custody,
        transaction=tx,decoder_transaction=dtx,counts=expected,distinct_receipt_ids=len(by),
        numerical_forecasts=numerical,MLP_equations=mlp_equations,state_factorials=state_factorials,contextual_formation=contextual_formation,
        route_witnesses_exact=route_checks,route_group_mass=route_mass,interventions=interventions,write_equations=write_equations,writer_interaction=writer_interaction,
        intervention_latents=latent_comparisons,predicted_bridge=bridge,pixel_inventories=inventories,collateral=collateral,propagation=propagation,
        decoder_effects=decoder_effects,decoder_curvature=decoder_curvature,worker_seconds=summary["worker_seconds"],
        scope="Four own-state project/joint0 predictions audited against native full fields; source-qualified independent NumPy program ran before targets. Full parameter banks retained as hash-bound recipes, not exported/recomputed offline. Twelve sampled MLP equation witnesses independently recalculated. Native encoder, downstream native futures, and VAE consumers remain observed. QK/V causal arm rankings are numerical RGB projections, not semantic scores.")
    with destination.open("x") as stream:json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
    panels={"natural":["b0r0","b1r0","b0r1","b1r1","math.bridge"],
            "causal":["b0r0","b1r0",*[s["id"] for s in config["interventions"]]],
            "decoder":["decode.native."+f"{a:+.2f}" for a in config["decoder_alphas"]]}
    for label,names in panels.items():
        cols=min(5,len(names)); w,hp=256,282; sheet=Image.new("RGB",(cols*w,((len(names)+cols-1)//cols)*hp),"#f5f5f5");draw=ImageDraw.Draw(sheet)
        for i,name in enumerate(names):
            x,y=(i%cols)*w,(i//cols)*hp;sheet.paste(Image.fromarray(images[name]).resize((256,256)),(x,y+26));draw.text((x+4,y+7),name,fill="black")
        sheet.save(destination.parent/("six-areas-"+label+".png"))
    return dict(job_id=summary["job_id"],counts=expected,numerical_forecasts=numerical,
        interventions=interventions,predicted_bridge=bridge,decoder_curvature=decoder_curvature)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.run.resolve(),a.output.resolve()),sort_keys=True))
