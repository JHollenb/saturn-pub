"""Saved-only row and clock interactions for the native binding search."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image
from mdb.edit_search import file_digest


def factorial(values):
    zero,a,b,ab=[np.asarray(v,dtype=np.float64)for v in values]
    da,db,dab=a-zero,b-zero,ab-zero
    interaction=dab-da-db
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    scale=rms(dab)
    return dict(a_effect_rms=rms(da),b_effect_rms=rms(db),joint_effect_rms=scale,
        single_effect_cosine=float(np.sum(da*db)/np.sqrt(np.sum(da*da)*np.sum(db*db)))if rms(da) and rms(db)else None,
        interaction_rms=rms(interaction),interaction_over_joint_rms=rms(interaction)/scale if scale else None,
        interaction_dot_joint_over_joint_energy=float(np.sum(interaction*dab)/np.sum(dab*dab))if scale else None,
        scope="finite observed vector contrasts, not semantic percentages or an independent predictor")


def analyze(run,audit,output):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text())
    assert read(audit)["mechanics_status"]=="verified"
    root=run/"extracted/binding-minimal-result";rows={r["id"]:r for r in read(root/"rows.json")};writes={r["row"]:r for r in read(root/"writes.json")}
    slot=runpy.run_path(str(here/"analyze_flux_binding_minimal.py"))["read_slot"]
    def array(record):
        p=root/record["path"];assert file_digest(p)==record["sha256"];return np.load(p,allow_pickle=False)
    def committed(name,step):return array(next(c["latent"]for c in rows[name]["commits"]if c["step"]==step))
    names=("control","row4","row11","positive");results={}
    for key in("attention","mlp_input","mlp","attention_text","mlp_text"):
        results["joint0."+key]=factorial([array(writes[n]["fields"][key])for n in names])
    for key in("image","text"):
        results["joint0.exit."+key]=factorial([slot(root/"store",writes[n]["exit"],key)for n in names])
    for step in(2,3,4):results["Z"+str(step)]=factorial([committed(n,step)for n in names])
    for key in("gate_attention","gate_mlp"):
        arrays=[array(writes[n]["fields"][key])for n in names]
        assert all(np.array_equal(arrays[0],v)for v in arrays[1:])
    B=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"))["quantize_bf16"]
    states={}
    for n in names:
        p={k:array(writes[n]["fields"][k])for k in("gate_attention","gate_mlp","attention","mlp")}
        R=slot(root/"store",writes[n]["entry"],"image")
        A=B(p["gate_attention"]*p["attention"]);M=B(p["gate_mlp"]*p["mlp"]);U=B(R+A)
        exit=slot(root/"store",writes[n]["exit"],"image")
        assert np.array_equal(B(U+M),exit)
        states[n]=dict(gated_attention=A,gated_mlp=M,after_attention=U,
            addition_rounding=exit.astype(float)-R.astype(float)-A.astype(float)-M.astype(float))
    for key in states[names[0]]:results["writer."+key]=factorial([states[n][key]for n in names])
    images={n:np.asarray(Image.open(root/r["png"])).copy()for n,r in rows.items()}
    results["RGB"]=factorial([images[n]for n in names])
    clocknames=("control","step2","step3","positive")
    clocks={"Z"+str(step):factorial([committed(n,step)for n in clocknames])for step in(2,3,4)}
    clocks["RGB"]=factorial([images[n]for n in clocknames])
    late_delta=images["positive"].astype(float)-images["step2"].astype(float)
    result=dict(schema="mdb-binding-search-factorial-v1",mechanics_status="verified",source_audit_sha256=file_digest(audit),new_model_calls=0,
        row_factorial=results,clock_factorial=clocks,gates_equal=True,
        late_read_after_success=dict(rgb_mae=float(np.mean(np.abs(late_delta))),rgb_max=float(np.max(np.abs(late_delta))),changed_pixel_fraction=float(np.mean(np.any(late_delta!=0,axis=-1))),
            Z3_exact=bool(np.array_equal(committed("positive",3),committed("step2",3)))),
        scope="One observed I-primed receiver. Row factorial holds both later reads; clock factorial holds both rows. Coarse visible sufficiency and fine vector interactions are separate.")
    with output.open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    return dict(row_factorial=results,clock_factorial=clocks,late_read_after_success=result["late_read_after_success"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in("run","audit","output"):p.add_argument(n,type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.run.resolve(),a.audit.resolve(),a.output.resolve()),sort_keys=True))
