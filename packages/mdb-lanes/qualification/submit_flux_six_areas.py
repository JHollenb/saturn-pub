"""Stage, admit and collect the finite six-area investigation."""
from __future__ import annotations
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def helpers():
    return runpy.run_path(str(Path(__file__).with_name("flux_six_areas_campaign.py")))


def stage(run):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    h=helpers(); save,sha=h["save"],h["sha"]; here=Path(__file__).resolve().parent
    config=dict(schema="mdb-flux-six-areas-plan-v1",model="flux2-klein-4b-distilled",
        model_path="/mnt/big/llm-models/FLUX.2-klein-4B",
        checkpoint_sha256="dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        output_scratch="/mnt/hdd8tb/mdb/research",max_wall_seconds=840,
        **runpy.run_path(str(here/"flux_six_areas.py"))["plan"]())
    run.mkdir(parents=True,exist_ok=False); save(run/"plan.json",config)
    names=["flux_six_areas.py","flux_reentry_reference.py","flux_joint_writes.py","flux_scene_dependencies.py","flux_family_custody.py"]
    worker=here/"flux_six_areas_campaign.py"; extras=[(here/n,n) for n in names]+[(run/"plan.json","plan.json")]
    pre=run/"preflight.v1.json"; save(pre,dict(schema_version="mrun-preflight-config-v1",
        hashed_roots=[worker.name,*names,"plan.json",pre.name],
        path_token_allowlist=["six-areas-result",config["model_path"],config["output_scratch"]]))
    extras.append((pre,pre.name)); raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request=dict(schema="mdb-flux-six-areas-request-v1",config=config,model=config["model"],task_family="forward",
        backend="diffusers",device="cuda",dtype="bfloat16",batch_size=1,height=512,width=512,
        needs={"cuda":True},execution_environment=intent(),config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=27,logical_branches=27,
            logical_checkpoint_replays=8,max_local_work_items=27,
            checkpoint_policy="4 actual encodes; 4 project+joint0 mathematical forecasts before denoiser;4 natural futures+1 numerical bridge+7 image-only causal futures;5 native VAE dose probes+10 FP32 consumer diagnostics. All serial in one finite lease,no retries/training."),
        resource_exception=dict(ram_mb=16000,vram_mb=13000,disk_gb=2,calibration_job="job-1f1587e6983b",
            measured_peak_rss_mb=16059,measured_peak_vram_mb=9282,
            reason="Existing 4/8-encode Klein4B B1 512px BF16 resident lane. Encoder released before denoising; two independent NumPy parameter banks only during forecast; no full-weight exports. Sparse witnesses/CAS and 27 PNGs. FP32 VAE after all denoiser work, widening same BF16 weights. HDD scratch/cache/output;2GB root admission for bounded transport. 4 CPUs,13GB VRAM,16GB RAM plus normal kill envelope."))
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/"request.json",request)
    stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request["request_sha256"])


def submit(run):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    h=helpers(); sha=h["sha"]; r=json.loads((run/"request.json").read_text())
    if r["execution_environment"]!=intent(): raise ValueError("staged execution environment differs")
    if (run/"submission.json").exists() or any(sha(run/"payload"/n)!=v for n,v in r["source_hashes"].items()):
        raise ValueError("admitted or changed stage")
    if hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()!=r["request_sha256"]:
        raise ValueError("request seal differs")
    cmd=worker_command("flux_six_areas_campaign.py"); cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(cmd,request=r,model=r["model"],task="forward",experiment="mdb-flux-six-areas",
        payload=run/"payload",prefer_host="beast",gpu=True,needs=r["needs"],ram_mb=16000,vram_mb=13000,disk_gb=2,
        cpu_threads=4,est_wall_s=750,timeout_s=960,retry_on_kill=False,preflight=True,
        note="Klein4B BF16 B1 512px4step seed109.4 binding/relation prompts;4 project+joint0 math forecasts sealed before targets.4 natural+1 math bridge+7 QK/V/A/M futures.5 native+10 FP32 VAE probes.1111 transitions9 Acts11 candidate blocks27 images.1load4encodes,serial onelease,no retries.4CPU16GBRAM13GBVRAM2GBdisk,HDDscratch/cache,no fullweight export. Expected750s;840worker960outer stop.")
    h["save"](run/"submission.json",dict(job_id=str(job))); return dict(job_id=str(job),run=str(run))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run",type=Path); p.add_argument("--submit",action="store_true");p.add_argument("--collect")
    a=p.parse_args(); run=a.run.resolve()
    if a.collect: result=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](run,a.collect)
    elif a.submit: result=submit(run)
    else:result=stage(run)
    print(json.dumps(result,sort_keys=True))
