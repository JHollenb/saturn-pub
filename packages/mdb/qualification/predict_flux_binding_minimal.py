"""Predict project2/J00 from observed old prefixes, with no target/native calls."""
import argparse
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"


def predict(source, prior, output, *, panel="streams", prime="I"):
    import numpy as np
    from mdb.tensor_archive import load_tensor_archive
    from saturn_pub.shared_store import configured_store,share_configured_file

    started=time.monotonic();here=Path(__file__).resolve().parent
    helper=runpy.run_path(str(here/"flux_binding_minimal.py"));sha=helper["sha"];save=helper["save"]
    read=lambda p:json.loads(p.read_text())
    old=prior/"extracted/flux-final-latent-result";data=source/"run-02/extracted/six-areas-result"
    binding_data=source/"binding-02/extracted/binding-program-result"
    binding=read(old/"binding.json");recipes=read(data/"parameter-recipes.json")
    inventory=read(prior/"collection-local-inventory.json")["files"]
    independent=prior/"independent-math-freeze/flux_reentry_audit_math.py"
    request=read(prior/"request.json")
    assert sha(independent)==request["independent_math_sources"][independent.name]
    assert read(old/"source.json")["checkpoint_sha256"]==read(data/"source.json")["checkpoint_sha256"]
    weights={};used={}
    for role in ("project","joint0"):
        weights[role]={};used[role]={}
        for name,record in binding[role]["parameters"].items():
            p=old/record["path"];archive_root,archive_record=old,record
            if not p.exists():
                pool=configured_store(p)
                if pool is None:raise ValueError("parameter archive absent and no shared store configured")
                p=pool.object_path(record["sha256"]);archive_root=pool.root
                archive_record=dict(record,path=p.relative_to(pool.root).as_posix())
            assert sha(p)==record["sha256"]==inventory[record["path"]]["sha256"]
            v=load_tensor_archive(archive_root,archive_record);r=recipes[role][name];bits=np.ascontiguousarray(v).view(np.uint32)
            assert not np.any(bits&65535) and list(v.shape)==r["shape"]
            assert hashlib.sha256((bits>>16).astype("<u2").tobytes()).hexdigest()==r["bf16_le_sha256"]
            weights[role][name]=v;used[role][name]=dict(sha256=record["sha256"],native_bf16_sha256=r["bf16_le_sha256"],resolved_archive=str(p))
    typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"))
    routes=runpy.run_path(str(here/"analyze_flux_concept_routes.py"))
    slot=lambda root,fp,k:routes["slot_array"](root/"store",fp,k,typed)
    rows={r["id"]:r for r in read(binding_data/"rows.json")}
    observed=read(data/"observed.json");donor=observed["b1r0"]["joint4_entry"]
    donorC=slot(data,donor,"conditioning")
    tokens=read(data/"tokens.json")["b0r0"]
    groups=dict(color=sorted(set(tokens["groups"]["red"]+tokens["groups"]["blue"])),
                valid=[i for i,v in enumerate(tokens["attention_mask"]) if v],padding=tokens["groups"]["padding"])
    equations=runpy.run_path(str(independent));cases={};output.mkdir(parents=True,exist_ok=False)
    def array(label,v):
        path=output/(label+".npy");np.save(path,np.asarray(v,np.float32),allow_pickle=False);share_configured_file(path)
        return dict(path=path.name,sha256=sha(path),shape=list(v.shape),dtype="float32")
    for spec in helper["specs"](panel,prime):
        name=spec["id"];prefix=("I" if name in ("I","IC") else "T" if name=="TC" else "IT" if name=="ITC" else "sham") if panel=="streams" else prime
        fp=rows[prefix]["commits"][0]["cut"]
        z=slot(binding_data,fp,"latent");c=slot(binding_data,fp,"conditioning").copy()
        if spec["C"]!="base":
            indices=list(range(512)) if spec["C"]=="all" else groups[spec["C"]];c[:,indices]=donorC[:,indices]
        ps=binding["project"]
        p=equations["independent_project"](z,c,slot(binding_data,fp,"img_ids"),slot(binding_data,fp,"txt_ids"),
            slot(binding_data,fp,"timesteps")[2],weights["project"],**{k:ps[k] for k in ("axes_dims_rope","rope_theta","timestep_channels")})
        j=equations["independent_joint"](p["image"],p["text"],p["context"],weights["joint0"],
            **{k:binding["joint0"][k] for k in ("heads","layernorm_eps","qk_eps")})
        fields={"project."+k:array(name+".project."+k,p[k]) for k in ("image","text")}
        fields.update({"joint."+k:array(name+".joint."+k,j[k]) for k in ("image","text")})
        fields.update({"context."+k:array(name+".context."+k,v) for k,v in p["context"].items() if k!="rope"})
        fields.update({"rope."+str(i):array(name+".rope."+str(i),v) for i,v in enumerate(p["context"]["rope"])})
        cases[name]=dict(observed_prefix=fp,prefix_history=prefix,latent_descriptor=read(binding_data/"store/cuts"/(fp+".json"))["manifest"]["payload"],fields=fields)
        print(json.dumps(dict(case=name,fields=len(fields),elapsed=time.monotonic()-started)),flush=True)
    result=dict(schema="mdb-binding-minimal-independent-forecast-v1",panel=panel,prime=prime,
        independent_source=dict(path=str(independent),sha256=sha(independent)),parameter_banks=used,cases=cases,
        new_model_calls=0,new_parameter_exports=0,seconds=time.monotonic()-started,
        scope="Observed old committed Z2, schedule and IDs; edited observed C. Own independently predicted project/context feeds joint0. No new native target read. Prior ITC/sham/I are historical controls; IC/TC continuations are new.")
    save(output/"manifest.json",result)
    return dict(manifest=str(output/"manifest.json"),sha256=sha(output/"manifest.json"),cases=len(cases),seconds=result["seconds"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("prior",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--panel",choices=["streams","subsets"],default="streams");p.add_argument("--prime",choices=["I","T","IT"],default="I");a=p.parse_args()
    print(json.dumps(predict(a.source.resolve(),a.prior.resolve(),a.output.resolve(),panel=a.panel,prime=a.prime),sort_keys=True))
