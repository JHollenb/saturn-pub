"""Submit the matched I/T/future-C state-coalition panel from audited cuts."""
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
    from mrun.protocol import kill_ceiling_mb

    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    audit = json.loads((predecessor/"analysis-v1.json").read_text())
    if audit["mechanics_status"] != "valid" or audit["job_id"] != "job-1424b0346278":
        raise ValueError("an independently audited scene-inventory predecessor is required")
    data = predecessor/"extracted/flux-scene-inventory-result"
    source = json.loads((data/"source.json").read_text())
    old = json.loads((predecessor/"plan.json").read_text())
    manifest = json.loads((data/"files.json").read_text())
    histories = ("s0c0r0","s1c0r0","s0c1r0","s0c0r1")
    cuts = {name:source["histories"][name]["post"] for name in histories}
    relatives = set()
    for value in cuts.values():
        relative = "typed-store/cuts/"+value+".json"
        record = json.loads((data/relative).read_text())
        relatives.add(relative)
        relatives.update("typed-store/pages/"+h+".json" for h in record["slots"].values())
        relatives.update("typed-store/blobs/"+h+".safetensors" for h in record["physical_blobs"])
    extras, external_inputs = [], {}
    for relative in sorted(relatives):
        if sha(data/relative) != manifest[relative]:
            raise ValueError("source CAS differs")
        external_inputs[relative] = {"sha256":manifest[relative],"bytes":(data/relative).stat().st_size}
    run_dir.mkdir(parents=True,exist_ok=False)
    selection = {"job_id":audit["job_id"],"audit_sha256":sha(predecessor/"analysis-v1.json"),
        "raw_manifest_sha256":sha(data/"files.json"),"cuts":cuts,
        "known_outcomes":"Four natural references and three image-only interventions have been observed. New I/T/C coalitions remain unseen."}
    save(run_dir/"source-selection.json",selection)
    historical = {"base":json.loads((data/"rows/native.s0c0r0.json").read_text())["image"]["sha256"]}
    for factor,history in (("shapeA","s1c0r0"),("colorB","s0c1r0"),("relation","s0c0r1")):
        historical["reference."+factor] = json.loads((data/"rows"/("native."+history+".json")).read_text())["image"]["sha256"]
        historical[factor+".100"] = json.loads((data/"rows"/(factor+".full.json")).read_text())["image"]["sha256"]
    config = {k:old[k] for k in ("model","model_path","expected_checkpoint_sha256","seed","height","width","steps","cpu_threads")}
    config.update(schema="mdb-flux-scene-state-closure-plan-v1",input_cuts=cuts,
        external_input_root=json.loads((predecessor/"job.json").read_text())["result"]["diagnostics"]["cwd"]+"/flux-scene-inventory-result",
        external_inputs=external_inputs,
        rows=runpy.run_path(str(here/"flux_scene_state_closure_campaign.py"))["make_rows"](),
        historical_pngs=historical,max_adapter_steps=2200,max_wall_seconds=840,
        hypotheses={
            "H-replay":"Four natural references and image-only 100 arms reproduce seven historical PNGs exactly.",
            "H-color-authority":"Tentative: colorB.001 (future C only, installed before step2 projection) produces a green B, while known colorB.100 retains blue. Retain fragmentation/ambiguity/collateral.",
            "H-order-authority":"Tentative: relation.001 produces red-right/blue-left ordering, while known relation.100 retains red-left/blue-right. Centroid ordering and visual shapes remain separate.",
            "H-shape-authority":"Tentative: shapeA.001 produces a sphere-like red A, while known shapeA.100 remains rounded-cube-like. Informal shape observations are arm-blinded and separate from pixel metrics.",
            "H-coherence-rank":"Tentative: final source-directed RGB projection(111) > projection(100) for each factor. No exact source equivalence is predicted because scheduler latent/history remains recipient.",
            "H-working-text":"Exploratory: compare 110 vs100 (current text contribution) and111 vs101 (text given futureC). Preserve either direction rather than assuming text adds semantic purity.",
            "H-scope":"Bits select image state I, working text T and future persistent C independently. I/T acts at step1 joint4 exit; C act only at step2 preprojection. All recipient scheduler latent/history remains live; no future latent donor, raw-prompt numerical RGB forecast or new semantic rule is supplied."},
        terminal_status="unassessed")
    save(run_dir/"plan.json",config)
    worker = here/"flux_scene_state_closure_campaign.py"
    extras += [(here/"flux_scene_inventory_campaign.py","flux_scene_inventory_campaign.py"),
               (here/"discovery_upgrades.py","campaign_support.py"),
               (run_dir/"source-selection.json","source-selection.json"),(run_dir/"plan.json","plan.json")]
    preflight = run_dir/"preflight.v1.json"
    save(preflight,{"schema_version":"mrun-preflight-config-v1",
        "hashed_roots":[worker.name,"flux_scene_inventory_campaign.py","campaign_support.py","source-selection.json","plan.json",preflight.name],
        "path_token_allowlist":["input-store","flux-scene-state-closure-result",config["model_path"],config["external_input_root"]]})
    extras.append((preflight,preflight.name))
    raw = (json.dumps(config,sort_keys=True)+"\n").encode()
    if kill_ceiling_mb(16000) >= 66512*.95:
        raise ValueError("RAM kill envelope cannot fit host")
    environment = intent()
    if submit and environment["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("Configure the measured Linux MDB_WORKER cache profile before a Beast launch")
    request = {"schema":"mdb-flux-scene-state-closure-request-v1","config":config,"task_family":"forward",
        "model":config["model"],"needs":{"cuda":True,"host":"beast"},"backend":"diffusers","device":"cuda","dtype":"bfloat16",
        "batch_size":1,"height":512,"width":512,"execution_environment":environment,
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),
        "source_hashes":{name:sha(path) for path,name in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=25,logical_branches=25,
            logical_checkpoint_replays=25,max_local_work_items=25,
            checkpoint_policy="4 observed post-joint4 parents;3 factors x7 I/T/futureC coalitions plus4refs=25nativefutures.18newinterventions,7historicalcontrols.1load/0encodes/B1/no retries;C only at legal step2 project boundary."),
        "resource_exception":{"ram_mb":16000,"vram_mb":13000,"calibration_job":"job-1424b0346278",
            "measured_peak_rss_mb":16323,"measured_peak_vram_mb":9632,
            "reason":"Same B1 CUDA BF16 512px4step resident native consumer. Four retained source cuts, no encoding, bounded2250native operations and compact evidence; normal RAM kill envelope fits66512MB host."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run_dir/"request.json",request)
    stage_worker(run_dir/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    if not submit:
        return {"submitted":False,"run_dir":str(run_dir)}
    job = guarded_mrun_launch(worker_command(worker.name),request=request,model=config["model"],task="forward",
        experiment="mdb-flux-scene-state-closure",payload=run_dir/"payload",prefer_host="beast",gpu=True,
        needs=request["needs"],ram_mb=16000,vram_mb=13000,cpu_threads=4,est_wall_s=600,timeout_s=840,
        retry_on_kill=False,preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed47.25RGB:4refs +3factors*7 image/text/futureC coalitions.18new,7exacthistorycontrols. Source post-j4 cuts; preserve recipient L/history; C only step2project.1load/0encodes/no retries. Stop25images/2200steps/840s;gzip1.")
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

        _, job, logs = terminal_job(args.status)
        phases = [line for line in logs.splitlines() if line.startswith("MDB_FLUX_SCENE_STATE_CLOSURE_PHASE ")]
        result = {"job_id":args.status,"state":job["state"],"last_phases":phases[-2:],
                  "sealed_phase":next((line for line in phases if '"kind": "predictions_sealed"' in line),None),
                  "last_error_lines":logs[-1800:] if job["state"]=="failed" else None}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir,args.collect,log_storage="gzip",max_compressed_bytes=512 << 20,max_expanded_bytes=1 << 30)
    else:
        if args.predecessor is None:
            p.error("--predecessor required")
        result = stage(args.run_dir.resolve(),args.predecessor.resolve(),submit=not args.stage_only)
    print(json.dumps(result,sort_keys=True))
