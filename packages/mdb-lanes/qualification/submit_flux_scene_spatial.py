"""Launch output-assisted spatial latent contrasts from audited project cuts."""
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
    if audit["mechanics_status"]!="valid" or audit["job_id"]!="job-2b05cfb69822":
        raise ValueError("independently audited persistent-state predecessor required")
    data = predecessor/"extracted/flux-scene-persistent-result"
    old = json.loads((predecessor/"plan.json").read_text())
    manifest = json.loads((data/"files.json").read_text())
    factors = (("shapeA","s1c0r0"),("colorB","s0c1r0"),("relation","s0c0r1"))
    cuts = old["input_cuts"]
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
    from saturn_pub.store import LocalStore

    helper = runpy.run_path(str(here/"flux_scene_spatial.py"))
    base = LocalStore(data/"typed-store").load(cuts["s0c0r0"],device="cpu")
    inventories = {"base":audit["inventories"]["base"],
                   **{factor:audit["inventories"][factor+".11"] for factor,_ in factors}}
    windows = {factor:helper["supports"](inventories["base"],inventories[factor],
        base.payload["img_ids"].numpy(),height=old["height"],width=old["width"]) for factor,_ in factors}
    historical = {name:json.loads((data/"rows/base.json").read_text())["image"]["sha256"] for name in ("base","sham")}
    for factor,_ in factors:
        historical[factor+".full"] = json.loads((data/"rows"/(factor+".10.json")).read_text())["image"]["sha256"]
    run_dir.mkdir(parents=True,exist_ok=False)
    save(run_dir/"source-selection.json",{"job_id":audit["job_id"],"audit_sha256":sha(predecessor/"analysis-v1.json"),
        "raw_manifest_sha256":sha(data/"files.json"),"cuts":cuts,
        "known_outcomes":"Five replay controls and natural reference RGB inventories are observed; fifteen spatial patch futures remain unseen."})
    config = {k:old[k] for k in ("model","model_path","expected_checkpoint_sha256","seed","height","width","steps","cpu_threads")}
    config.update(schema="mdb-flux-scene-spatial-plan-v1",input_cuts=cuts,
        external_input_root=json.loads((predecessor/"job.json").read_text())["result"]["diagnostics"]["cwd"]+"/flux-scene-persistent-result",
        external_inputs=external_inputs,supports=windows,historical_inventories=inventories,rows=runpy.run_path(str(here/"flux_scene_spatial_campaign.py"))["make_rows"](),
        historical_pngs=historical,max_adapter_steps=1300,max_wall_seconds=600,
        hypotheses={
            "H-replay":"Base and sham reproduce base PNG; three full-L patches reproduce prior source-L/base-C PNGs. Five exact replay controls are mechanics, not discoveries.",
            "H-shape":"Tentative: shapeA.left and shapeA.both render sphere-like red A; shapeA.right retains cube-like A. Record companion shape and all collateral under arm-blinded informal inspection.",
            "H-color":"Tentative: colorB.right has greater right-object median green-minus-blue than colorB.left, and looks green/teal; left-only retains blue B. Instrument may abstain if components fragment.",
            "H-order-left":"Tentative: relation.left produces two blue-dominant objects, red count zero; retain actual shapes, component counts and repairs.",
            "H-order-right":"Tentative: relation.right produces two red-dominant objects, blue count zero; retain actual shapes, component counts and repairs.",
            "H-order-both":"Tentative: relation.both produces blue-left/red-right with two major objects and correct coarse shape binding. RGB agreement is measured separately.",
            "H-location":"Tentative: relation.outside and relation.shifted retain recipient red-left/blue-right ordering; any collateral or contrary outcome remains evidence.",
            "H-scope":"All branches retain base C and exact nuisance slots. Windows use already observed natural-reference RGB bounds plus16px halo and native H/W IDs, not newly inferred latent object IDs. Both stacks two disjoint Acts. Fifteen new dependent donor-assisted futures; no raw-input RGB forecast, held-out seed or minimal relation register claim."},
        terminal_status="unassessed")
    save(run_dir/"plan.json",config)
    worker = here/"flux_scene_spatial_campaign.py"
    extras = [(here/"flux_scene_spatial.py","flux_scene_spatial.py"),(here/"flux_scene_inventory_campaign.py","flux_scene_inventory_campaign.py"),
              (here/"discovery_upgrades.py","campaign_support.py"),
              (run_dir/"source-selection.json","source-selection.json"),(run_dir/"plan.json","plan.json")]
    preflight = run_dir/"preflight.v1.json"
    save(preflight,{"schema_version":"mrun-preflight-config-v1",
        "hashed_roots":[worker.name,"flux_scene_spatial.py","flux_scene_inventory_campaign.py","campaign_support.py","source-selection.json","plan.json",preflight.name],
        "path_token_allowlist":["input-store","flux-scene-spatial-result",config["model_path"],config["external_input_root"]]})
    extras.append((preflight,preflight.name))
    raw = (json.dumps(config,sort_keys=True)+"\n").encode()
    environment = intent()
    if submit and environment["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("configure measured Linux MDB_WORKER cache profile")
    request = {"schema":"mdb-flux-scene-spatial-request-v1","config":config,"task_family":"forward",
        "model":config["model"],"needs":{"cuda":True,"host":"beast"},"backend":"diffusers","device":"cuda","dtype":"bfloat16",
        "batch_size":1,"height":512,"width":512,"execution_environment":environment,
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"source_hashes":{name:sha(path) for path,name in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=20,logical_branches=20,
            logical_checkpoint_replays=20,max_local_work_items=20,
            checkpoint_policy="Four audited natural step2 project cuts;3factors x6supportpolicies plusbase/sham=20futures.15new/5known.1load/0encodes/B1/1120transitions/22Acts/no retries."),
        "resource_exception":{"ram_mb":10000,"vram_mb":13000,"calibration_job":"job-2b05cfb69822",
            "measured_peak_rss_mb":6448,"measured_peak_vram_mb":9260,
            "reason":"Same resident B1 CUDA BF16 consumer and4sourcecuts; twice as many suffix outputs and compact native-row masks.10GB RAM includes serialization margin over measured6448MB;13GB VRAM retained from9260MB profile."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run_dir/"request.json",request)
    stage_worker(run_dir/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    if not submit:
        return {"submitted":False,"run_dir":str(run_dir)}
    job = guarded_mrun_launch(worker_command(worker.name),request=request,model=config["model"],task="forward",
        experiment="mdb-flux-scene-spatial",payload=run_dir/"payload",prefer_host="beast",gpu=True,
        needs=request["needs"],ram_mb=10000,vram_mb=13000,cpu_threads=4,est_wall_s=360,timeout_s=600,
        retry_on_kill=False,preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed47.20RGB:15new left/right/joint/background/shifted latent windows,5exacthistorycontrols. BaseC;knownRGB-assisted supports;22Acts/1120steps;1load/0encodes/no retries;stop20images/1300steps/600s;audited host-local CAS.")
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
        phases = [line for line in logs.splitlines() if line.startswith("MDB_FLUX_SCENE_SPATIAL_PHASE ")]
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
