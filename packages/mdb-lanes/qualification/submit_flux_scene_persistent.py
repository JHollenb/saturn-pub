"""Launch a finite latent/conditioning cross from audited natural project cuts."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def stage(run_dir,predecessor,*,submit=True):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    audit = json.loads((predecessor/"analysis-v1.json").read_text())
    if audit["mechanics_status"]!="valid" or audit["job_id"]!="job-2bde111d1343":
        raise ValueError("independently audited state-closure predecessor required")
    data = predecessor/"extracted/flux-scene-state-closure-result"
    old = json.loads((predecessor/"plan.json").read_text())
    manifest = json.loads((data/"files.json").read_text())
    factors = (("shapeA","s1c0r0"),("colorB","s0c1r0"),("relation","s0c0r1"))
    source_rows = {h:json.loads((data/"rows"/(name+".json")).read_text())
                   for name,h in (("base","s0c0r0"),*(("reference."+f,h) for f,h in factors))}
    cuts = {h:r["schedulers"][0]["cut"] for h,r in source_rows.items()}
    relatives = set()
    for value in cuts.values():
        relative = "typed-store/cuts/"+value+".json"
        record = json.loads((data/relative).read_text())
        relatives.add(relative)
        relatives.update("typed-store/pages/"+h+".json" for h in record["slots"].values())
        relatives.update("typed-store/blobs/"+h+".safetensors" for h in record["physical_blobs"])
    external_inputs = {}
    for relative in sorted(relatives):
        if sha(data/relative)!=manifest[relative]:
            raise ValueError("source CAS differs")
        external_inputs[relative] = {"sha256":manifest[relative],"bytes":(data/relative).stat().st_size}
    historical = {"base":source_rows["s0c0r0"]["image"]["sha256"]}
    for factor,history in factors:
        historical[factor+".11"] = source_rows[history]["image"]["sha256"]
        historical[factor+".01"] = json.loads((data/"rows"/(factor+".001.json")).read_text())["image"]["sha256"]
    run_dir.mkdir(parents=True,exist_ok=False)
    save(run_dir/"source-selection.json",{"job_id":audit["job_id"],"audit_sha256":sha(predecessor/"analysis-v1.json"),
        "raw_manifest_sha256":sha(data/"files.json"),"cuts":cuts,
        "known_outcomes":"Seven replay controls have been observed; three source-latent/base-conditioning futures remain unseen."})
    config = {k:old[k] for k in ("model","model_path","expected_checkpoint_sha256","seed","height","width","steps","cpu_threads")}
    config.update(schema="mdb-flux-scene-persistent-plan-v1",input_cuts=cuts,
        external_input_root=json.loads((predecessor/"job.json").read_text())["result"]["diagnostics"]["cwd"]+"/flux-scene-state-closure-result",
        external_inputs=external_inputs,rows=runpy.run_path(str(here/"flux_scene_persistent_campaign.py"))["make_rows"](),
        historical_pngs=historical,max_adapter_steps=700,max_wall_seconds=600,
        hypotheses={
            "H-replay":"00 base, three 01 C-only arms, and three 11 coherent-source arms reproduce seven known PNGs exactly; mechanics controls, not discoveries.",
            "H-layout":"Tentative new prediction: relation.10 (source latent, base C) keeps red-right/blue-left; prior C-only relation.01 retains red-left/blue-right. Quantify actual component centroids, count, and collateral.",
            "H-color":"Tentative new prediction: colorB.10 has greater B median green-minus-blue than colorB.01. Preserve magnitude and ambiguity; no exact color forecast.",
            "H-shape":"Tentative new prediction: shapeA.10 looks sphere-like despite base cube conditioning; informal arm-blinded shape observation, not an automatic shape classifier.",
            "H-scope":"Bits are L,C at the same step2 project boundary. All nuisance slots are exact and working I/T/context/noise_prediction are cleared. Three new dependent donor-assisted futures; no RGB equation forecast, held-out seed, pure semantic object register, or raw-input understanding claim."},
        terminal_status="unassessed")
    save(run_dir/"plan.json",config)
    worker = here/"flux_scene_persistent_campaign.py"
    extras = [(here/"flux_scene_inventory_campaign.py","flux_scene_inventory_campaign.py"),
              (here/"discovery_upgrades.py","campaign_support.py"),
              (run_dir/"source-selection.json","source-selection.json"),(run_dir/"plan.json","plan.json")]
    preflight = run_dir/"preflight.v1.json"
    save(preflight,{"schema_version":"mrun-preflight-config-v1",
        "hashed_roots":[worker.name,"flux_scene_inventory_campaign.py","campaign_support.py","source-selection.json","plan.json",preflight.name],
        "path_token_allowlist":["input-store","flux-scene-persistent-result",config["model_path"],config["external_input_root"]]})
    extras.append((preflight,preflight.name))
    raw = (json.dumps(config,sort_keys=True)+"\n").encode()
    environment = intent()
    if submit and environment["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("configure measured Linux MDB_WORKER cache profile")
    request = {"schema":"mdb-flux-scene-persistent-request-v1","config":config,"task_family":"forward",
        "model":config["model"],"needs":{"cuda":True,"host":"beast"},"backend":"diffusers","device":"cuda","dtype":"bfloat16",
        "batch_size":1,"height":512,"width":512,"execution_environment":environment,
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"source_hashes":{name:sha(path) for path,name in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=10,logical_branches=10,
            logical_checkpoint_replays=10,max_local_work_items=10,
            checkpoint_policy="Four audited natural step2 project cuts; three L/C crosses x3factors plusbase=10nativefutures;3new/7historycontrols.1load/0encodes/B1/no retries."),
        "resource_exception":{"ram_mb":8000,"vram_mb":13000,"calibration_job":"job-2bde111d1343",
            "measured_peak_rss_mb":2369.9,"measured_peak_vram_mb":9250,
            "reason":"Same resident B1 CUDA BF16 consumer; fewer cuts/steps/outputs, no encodes; measured RSS was2369.9MB.8GB RAM includes hydration/serialization margin;13GB VRAM retained from measured profile."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run_dir/"request.json",request)
    stage_worker(run_dir/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    if not submit:
        return {"submitted":False,"run_dir":str(run_dir)}
    job = guarded_mrun_launch(worker_command(worker.name),request=request,model=config["model"],task="forward",
        experiment="mdb-flux-scene-persistent",payload=run_dir/"payload",prefer_host="beast",gpu=True,
        needs=request["needs"],ram_mb=8000,vram_mb=13000,cpu_threads=4,est_wall_s=240,timeout_s=600,
        retry_on_kill=False,preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed47.10RGB L/C factorial step2project:3new source-L/base-C futures,7exacthistorycontrols. Whole-state nuisance closure checked.1load/0encodes/no retries;stop10images/700steps/600s;host-local audited CAS.")
    save(run_dir/"submission.json",{"job_id":str(job)})
    return {"job_id":str(job),"run_dir":str(run_dir)}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir",type=Path)
    p.add_argument("--predecessor",type=Path)
    p.add_argument("--stage-only",action="store_true")
    p.add_argument("--collect")
    p.add_argument("--status")
    args = p.parse_args()
    if args.status:
        from mdb_lanes.job_support import terminal_job

        _,job,logs = terminal_job(args.status)
        phases = [line for line in logs.splitlines() if line.startswith("MDB_FLUX_SCENE_PERSISTENT_PHASE ")]
        result = {"job_id":args.status,"state":job["state"],"last_phases":phases[-2:],
                  "sealed_phase":next((line for line in phases if '"kind": "predictions_sealed"' in line),None),
                  "last_error_lines":logs[-2400:] if job["state"]=="failed" else None}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir,args.collect,log_storage="gzip",max_compressed_bytes=512 << 20,max_expanded_bytes=1 << 30)
    else:
        if args.predecessor is None:
            p.error("--predecessor required")
        result = stage(args.run_dir.resolve(),args.predecessor.resolve(),submit=not args.stage_only)
    print(json.dumps(result,sort_keys=True))
