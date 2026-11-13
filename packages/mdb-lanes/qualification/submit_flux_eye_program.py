"""Stage a finite same-parent eye-program composition and attention forecast."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")


def stage(run,dependencies,predicted):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    dd=dependencies/"extracted/scene-dependencies-result";pd=predicted/"extracted/predicted-future-result"
    if read(dependencies.parent/"analysis-v1.json")["mechanics_status"]!="verified":raise ValueError("verified dependency result required")
    drows={r["id"]:r for r in read(dd/"rows.json")};pb=read(pd/"binding.json");pp=read(pd/"plan.json")
    inputs={"pre."+role:{"group":"pre","cut":pb["inputs"][role]["parent"]} for role in ("base","color","shape")}
    inputs.update({"post.base":{"group":"post","cut":drows["L0I0T0"]["actions"][-1]["post_act"]},
        "post.program":{"group":"post","cut":drows["L0I0T1"]["actions"][-1]["post_act"]}})
    groups={}
    for group,data,remote in (("pre",pd,pp["output_scratch"]+"/"+read(pd/"summary.json")["job_id"]+"/predicted-future-result"),
        ("post",dd,read(dd/"summary.json")["output_root"]+"/scene-dependencies-result")):
        manifest=read(data/"files.json");names=set()
        for spec in inputs.values():
            if spec["group"]!=group:continue
            name="store/cuts/"+spec["cut"]+".json";names.add(name);c=read(data/name)
            names.update("store/pages/"+h+".json" for h in c["slots"].values());names.update("store/blobs/"+h+".safetensors" for h in c["physical_blobs"])
        files={}
        for name in sorted(names):
            if sha(data/name)!=manifest[name]:raise ValueError("source CAS differs")
            files[name]={"bytes":(data/name).stat().st_size,"sha256":manifest[name]}
        groups[group]={"root":remote,"files":files}
    config={"schema":"mdb-eye-program-plan-v1","model":pp["model"],"model_path":pp["model_path"],"checkpoint_sha256":pp["checkpoint_sha256"],
        "input_groups":groups,"input_cuts":inputs,"prompts":{role:pp["prompts"][role] for role in ("base","joint")},
        "output_scratch":"/home/beast/.cache/mdb-research","max_wall_seconds":600,
        "historical_pngs":{"native.base":sha(dd/"L0I0T0.png"),"full.program":sha(dd/"L0I0T1.png")},
        "hypotheses":{
            "program_vs_write":"Tentative prediction: source post6 working text alone retains violet/narrow response, whereas its current image write alone leaves eyes amber. These roles are future-program carrier versus immediate image write, not text-only global causality.",
            "composition":"Color-only contextual T should favor violet/open eyes; shape-only T should favor amber/narrow eyes. Additive Tcolor+Tshape-Tbase is predicted to give violet/narrow with coarse copper hair/plum clothing/frontal portrait/gray background retained. Preserve partials and collateral.",
            "address":"Lexical changed-row T patch is predicted weaker than full contextual T. Active/padding/top64/rolled supports are exploration; no guaranteed destructive-null interpretation.",
            "math":"Forecast complete image attention and MLP fields of the previously unexecuted additive T input at block7 from entry state and actual weights. Seal before every new block7 call. Actual joint T is a comparison only and never enters additive rule. Native image M should remain exactly base because image entry is fixed; image attention changes through contextual text K/V.",
            "consumer":"Predicted A/M image-exit replacement should closely reproduce measured additive-program continuation, with native text and suffix retained; report full errors and both semantics."},
        "scope":"Second same-artifact iteration after T rather than I governs late eye-color response. Split current image write from continuing working text; compose standardized color/shape contextual programs; lower lexical/active/padding/energy supports. Full A/M prediction on a new additive state, not raw-prompt RGB prediction.13 dependent native futures, no encodes, tokenizer-only lexical coordinates, all failures retained."}
    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config)
    here=Path(__file__).resolve().parent;worker=here/"flux_eye_program_campaign.py"
    names=["flux_predicted_future_campaign.py","flux_scene_dependencies.py","flux_single_contraction.py","flux_mlp_replacement.py","flux_attention_reference.py","flux_family_custody.py"]
    extras=[(here/n,n) for n in names]+[(run/"plan.json","plan.json")];pre=run/"preflight.v1.json"
    save(pre,{"schema_version":"mrun-preflight-config-v1","hashed_roots":[worker.name,*names,"plan.json",pre.name],
        "path_token_allowlist":["eye-program-result","input-data",config["model_path"],config["output_scratch"],*(g["root"] for g in groups.values())]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request={"schema":"mdb-eye-program-request-v1","config":config,"model":config["model"],"task_family":"forward","backend":"diffusers","device":"cuda","dtype":"bfloat16",
        "batch_size":1,"height":512,"width":512,"needs":{"cuda":True,"host":"beast"},"execution_environment":intent(),
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"source_hashes":{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=13,logical_branches=13,logical_checkpoint_replays=13,max_local_work_items=24,
            checkpoint_policy="Two1-step contextual color/shape preparations.12 native15-step futures, one14-step predicted-image suffix:196transitions13Acts209receipts.2 direct source blocks,1 projection,13VAE. One lease/no encode/no retries."),
        "resource_exception":{"ram_mb":10000,"vram_mb":13000,"disk_gb":6,"calibration_job":"job-d75bfb5263fe",
            "reason":"Same no-encoder resident512px Klein4B B1 lane10GB RAM/13GB VRAM as qualified dependency panel. Existing full attention reference tiled64queries/256weights, bounded source ports;13 versus12VAE, fewer retained branches. No physical batching."}}
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/"request.json",request)
    stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {"run_dir":str(run),"request_sha256":request["request_sha256"],"submitted":False}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    r=read(run/"request.json")
    if (run/"submission.json").exists() or any(sha(run/"payload"/p)!=v for p,v in r["source_hashes"].items()):raise ValueError("admitted or mutated stage")
    cmd=worker_command("flux_eye_program_campaign.py");cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(cmd,request=r,model=r["model"],task="forward",experiment="mdb-eye-program",payload=run/"payload",prefer_host="beast",gpu=True,needs=r["needs"],
        ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note="Klein CUDA BF16 B1 512px eye-program iteration. Reuse exact post7/pre6 cuts,0encodes,2tokenizations.13RGB196nativeSteps13Acts209receipts,2sourceblocks1projection. Seal full A/M prediction for additive color+shape T before block7. Split image/text writes, test contextual/lexical/padding supports.10GB RAM/13GB VRAM calibrated,600s worker/720s lease,no retries. Retain semantic misses and collateral.")
    save(run/"submission.json",{"job_id":str(job)});return {"job_id":str(job),"run_dir":str(run)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--dependencies",type=Path);p.add_argument("--predicted",type=Path);p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.dependencies.resolve(),a.predicted.resolve())
    print(json.dumps(r,sort_keys=True))
