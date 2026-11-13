"""Admit one same-artifact L/I/T restoration and suffix-reuse assay."""
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


def stage(run,old):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    data=old/"extracted/predicted-future-result"
    audit=read(old.parent/"analysis-v1.json")
    if audit["mechanics_status"]!="verified":raise ValueError("verified predecessor required")
    binding, forecasts, oldplan, manifest=[read(data/n) for n in ("binding.json","forecasts.json","plan.json","files.json")]
    cuts={role:binding["inputs"][role]["parent"] for role in ("base","joint")}
    names=set()
    for cut in cuts.values():
        rel="store/cuts/"+cut+".json";names.add(rel);record=read(data/rel)
        names.update("store/pages/"+v+".json" for v in record["slots"].values())
        names.update("store/blobs/"+v+".safetensors" for v in record["physical_blobs"])
    prediction=forecasts["fields"]["joint"];names.add(prediction["path"])
    external={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError("predecessor hash differs")
        external[name]={"bytes":(data/name).stat().st_size,"sha256":manifest[name]}
    summary=read(data/"summary.json")
    config={"schema":"mdb-scene-dependencies-plan-v1","model":oldplan["model"],"model_path":oldplan["model_path"],
        "checkpoint_sha256":oldplan["checkpoint_sha256"],"input_cuts":cuts,"mlp_prediction":prediction,
        "source_job":summary["job_id"],"source_seal":summary["seal"],"external_inputs":external,
        "external_input_root":oldplan["output_scratch"]+"/"+summary["job_id"]+"/predicted-future-result",
        "output_scratch":"/home/beast/.cache/mdb-research","max_wall_seconds":600,
        "historical_pngs":{"base":sha(data/"base.png"),"joint":sha(data/"joint.png"),"old_M_transfer":sha(data/"transfer.predicted.png")},
        "hypotheses":{
            "replay":"Derived step3 project closure reproduces saved pre6 payload exactly in12 native transitions; base/joint and fixedM baseline PNGs reproduce historical results.",
            "working_image":"Tentative: joint working image carries more of the late eye appearance than joint text alone. Preserve partial and reversed cases; no hard effect threshold.",
            "joint_parents":"Joint L/I/T with fixed predicted joint M yields violet narrow eyes and approaches native joint, retaining numerical approximation error and collateral.",
            "suffix_reuse":"Matched I/T/context yields identical transformer action across L0/L1. Predict every L1 terminal latent via explicit staged-BF16 Euler from observed L0 velocity; validate00/11 natively after sealing, serve01/10 through forecast+nativeVAE without transformer suffix.",
            "preserve":"Assess copper hair, plum clothing, one frontal adult portrait and gray background separately from eye change; no pixel-preservation or learned object-address claim."},
        "scope":"Same-artifact iterative diagnosis after failed late M-only eye edit. Fixed previously sealed predicted M, legal L-at-project and I/T-at-pre6 Acts. Eight L/I/T combinations, four L1 complete latent forecasts before native L1 corners; two middle L1 cases have forecast-state/nativeVAE only, explicitly no native suffix validation. Two historical references plus ten derived/validation images. Source-assisted conditional dependency reuse, not raw-prompt velocity prediction."}
    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config)
    here=Path(__file__).resolve().parent;worker=here/"flux_scene_dependencies_campaign.py"
    helper_names=["flux_scene_dependencies.py","flux_predicted_future_campaign.py","flux_single_contraction.py","flux_mlp_replacement.py","flux_family_custody.py"]
    extras=[(here/n,n) for n in helper_names]+[(run/"plan.json","plan.json")]
    pre=run/"preflight.v1.json"
    save(pre,{"schema_version":"mrun-preflight-config-v1","hashed_roots":[worker.name,*helper_names,"plan.json",pre.name],
        "path_token_allowlist":["scene-dependencies-result","input-data",config["model_path"],config["external_input_root"],config["output_scratch"]]})
    extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request={"schema":"mdb-scene-dependencies-request-v1","config":config,"model":config["model"],"task_family":"forward",
        "backend":"diffusers","device":"cuda","dtype":"bfloat16","batch_size":1,"height":512,"width":512,
        "needs":{"cuda":True,"host":"beast"},"execution_environment":intent(),"config_bytes_sha256":hashlib.sha256(raw).hexdigest(),
        "source_hashes":{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=12,logical_branches=12,
            logical_checkpoint_replays=12,max_local_work_items=24,checkpoint_policy="Saved pre6 cuts;36 project-reconstruction transitions,32 native reference,64 L0 factorial,32 L1 corner validation=164.19Acts183receipts,8candidateblocks,12VAE. Four computed L1 images, two also validated. No encode/no retry."),
        "resource_exception":{"ram_mb":10000,"vram_mb":13000,"disk_gb":6,"calibration_job":"job-15fbdff68553",
            "measured_peak_rss_mb":5538.1,"measured_peak_vram_mb":9342,
            "reason":"Reuse qualified no-encoder resident Klein B1 512px20-case sequential port profile. This12image panel uses fewer suffixes/candidates; load transformer/VAE only, retain bounded four source action tensors.10GB RAM/13GB VRAM existing qualification exception."}}
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/"request.json",request)
    stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {"run_dir":str(run),"request_sha256":request["request_sha256"],"submitted":False}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    r=read(run/"request.json")
    if (run/"submission.json").exists():raise ValueError("already admitted")
    if any(sha(run/"payload"/n)!=v for n,v in r["source_hashes"].items()):raise ValueError("payload source changed")
    command=worker_command("flux_scene_dependencies_campaign.py");command.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(command,request=r,model=r["model"],task="forward",experiment="mdb-scene-dependencies",payload=run/"payload",
        prefer_host="beast",gpu=True,needs=r["needs"],ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,
        est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note="Klein CUDA BF16 B1 512px no encoder. Reuse audited eye-square pre6 parents/fixed predicted M. Legal latent-at-project then I/T restoration;164nativeSteps19Acts8directBlocks12VAE. Four complete L1 latent forecasts from observed L0 action; native-check00/11, skip transformer suffix for01/10. Preserve semantic misses.600s worker/720s lease,no retries,10GB RAM/13GB VRAM calibrated.")
    save(run/"submission.json",{"job_id":str(job)});return {"job_id":str(job),"run_dir":str(run)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--predecessor",type=Path)
    p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:result=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:result=submit(a.run.resolve())
    elif a.predecessor:result=stage(a.run.resolve(),a.predecessor.resolve())
    else:p.error("--predecessor required for staging")
    print(json.dumps(result,sort_keys=True))
