"""Rebuild new forecasts with frozen independent equations and reused banks.

No Torch, model execution, campaign predictor import or native future operand.
This consumes only the new initial cuts and prior immutable native parameters.
"""
import argparse
import hashlib
import json
import runpy
import time
from pathlib import Path
import numpy as np


def audit(root,prior,output):
    from mdb.tensor_archive import load_tensor_archive
    started=time.monotonic();read=lambda p:json.loads(p.read_text())
    def sha(p):
        d=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(8<<20),b""):d.update(b)
        return d.hexdigest()
    here=Path(__file__).resolve().parent;data=root/"run-02/extracted/six-areas-result";old=prior/"extracted/flux-final-latent-result"
    source,oldsource=read(data/"source.json"),read(old/"source.json")
    assert source["checkpoint_sha256"]==oldsource["checkpoint_sha256"]
    request=read(prior/"request.json");independent=prior/"independent-math-freeze/flux_reentry_audit_math.py"
    assert sha(independent)==request["independent_math_sources"][independent.name]
    manifest=read(prior/"collection-local-inventory.json")["files"]
    assert sha(old/"binding.json")==manifest["binding.json"]["sha256"]
    binding=read(old/"binding.json");recipes=read(data/"parameter-recipes.json");weights={};used={}
    for role,key in (("project","project"),("joint0","joint0")):
        spec=binding[key];weights[role]={};used[role]={}
        assert set(spec["parameters"])==set(recipes[role])
        for name,archive in spec["parameters"].items():
            p=old/archive["path"]
            assert sha(p)==manifest[archive["path"]]["sha256"]==archive["sha256"]
            v=load_tensor_archive(old,archive);r=recipes[role][name];bits=np.ascontiguousarray(v).view(np.uint32)
            assert v.dtype==np.float32 and not np.any(bits&65535) and list(v.shape)==r["shape"]
            assert hashlib.sha256(np.asarray(v,dtype="<f4").tobytes()).hexdigest()==r["fp32_le_sha256"]
            assert hashlib.sha256((bits>>16).astype("<u2").tobytes()).hexdigest()==r["bf16_le_sha256"]==archive["descriptor"]["sha256"]
            weights[role][name]=v;used[role][name]=dict(archive=archive["path"],archive_sha256=archive["sha256"],native_bf16_sha256=r["bf16_le_sha256"])
    independent_math=runpy.run_path(str(independent))
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"));helpers=runpy.run_path(str(here/"analyze_flux_concept_routes.py"))
    initial=read(data/"transaction/declarations.json")["frontier"]["initial_cuts"];forecasts=read(data/"forecasts.json");records={};scalars=0
    def slot(fp,name):return helpers["slot_array"](data/"store",fp,name,typed)
    def check(name,actual,reference):
        nonlocal scalars
        p=data/reference["path"];assert sha(p)==reference["sha256"];expected=np.load(p,allow_pickle=False)
        assert actual.shape==expected.shape and np.isfinite(actual).all()
        d=actual.astype(np.float64)-expected;scalars+=actual.size
        return dict(exact=bool(np.array_equal(actual,expected)),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(np.abs(d).max()),
            independent_FP32_sha256=hashlib.sha256(np.asarray(actual,dtype="<f4").tobytes()).hexdigest(),sealed_archive_sha256=reference["sha256"])
    spec=binding["project"]
    for name,fp in initial.items():
        tick=time.monotonic()
        p=independent_math["independent_project"](slot(fp,"latent"),slot(fp,"conditioning"),slot(fp,"img_ids"),slot(fp,"txt_ids"),
            slot(fp,"timesteps")[0],weights["project"],axes_dims_rope=spec["axes_dims_rope"],rope_theta=spec["rope_theta"],timestep_channels=spec["timestep_channels"])
        j=independent_math["independent_joint"](p["image"],p["text"],p["context"],weights["joint0"],**{k:binding["joint0"][k] for k in ("heads","layernorm_eps","qk_eps")})
        fields={"project."+k:check(name,p[k],forecasts[name]["project"][k]) for k in ("image","text")}
        fields.update({"context."+k:check(name,p["context"][k],record) for k,record in forecasts[name]["context"].items()})
        fields.update({"rope."+str(i):check(name,p["context"]["rope"][i],record) for i,record in enumerate(forecasts[name]["rope"])})
        fields.update({"joint."+k:check(name,j[k],forecasts[name]["joint"][k]) for k in ("image","text")})
        records[name]=dict(seconds=time.monotonic()-tick,initial_cut=fp,fields=fields)
        print(json.dumps(dict(case=name,seconds=records[name]["seconds"],exact_fields=sum(v["exact"] for v in fields.values()),fields=len(fields))),flush=True)
    result=dict(schema="mdb-six-areas-reused-weights-audit-v1",mechanics_status="verified",terminal_status="unassessed",source_job=read(data/"summary.json")["job_id"],
        prior_parameter_job=read(old/"summary.json")["job_id"],prior_request_sha256=request["request_sha256"],prior_binding_sha256=sha(old/"binding.json"),
        independent_math=dict(path=str(independent),sha256=sha(independent),authority="prior request-frozen independent source; no campaign predictor import"),
        parameter_banks=used,records=records,fields=sum(len(v["fields"]) for v in records.values()),scalars=scalars,
        exact_fields=sum(f["exact"] for v in records.values() for f in v["fields"].values()),
        new_model_calls=0,new_parameter_exports=0,seconds=time.monotonic()-started,
        scope="New initial state/C/IDs/schedule only. All project and joint0 fields consume independently predicted predecessors. Exact native parameter banks reused by content hash across jobs. Numerical discrepancies, if any, retained; native CUDA agreement is reported in the separate primary audit.")
    with output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    return {k:result[k] for k in ("fields","scalars","exact_fields","new_model_calls","new_parameter_exports","seconds")}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("prior",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(audit(a.root.resolve(),a.prior.resolve(),a.output.resolve()),sort_keys=True))
