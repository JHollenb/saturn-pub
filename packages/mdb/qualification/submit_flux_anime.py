"""Submit native anime reference or prospective spatial-intervention panels."""
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


def stage(run_dir,*,predecessor=None,annotation_path=None,submit=True):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    helper = runpy.run_path(str(here/"flux_anime_campaign.py"))
    scenes = helper["specimens"]()
    config = {"schema":"mdb-flux-anime-plan-v1","stage":"references","model":"flux2-klein-4b-distilled",
        "model_path":"/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256":"dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "height":512,"width":512,"steps":4,"cpu_threads":4,"scenes":scenes,
        "rows":helper["reference_rows"](scenes),"max_adapter_steps":700,"max_wall_seconds":600,
        "allowed_information":"Four own-state prefixes through step2 project under two scenes/seeds; no final native or crossed RGB observed",
        "hypotheses":{
            "H-native":"Prompt-compliance expectation: base has A-left/B-right, swap B-left/A-right, two visible adult characters in the specified scene. Observe attributes independently; no correctness gate.",
            "H-carried":"Tentative from primitive experiments: full source latent with base conditioning preserves source B-left/A-right cast arrangement, including major outfit/hair/prop associations. Inspect partial transfer and hybrid identity separately.",
            "H-scope":"New anime contexts, same qualified checkpoint/backend/clock. Observed source prefixes and native continuation; no numerical RGB forecast, independent face-recognition instrument or complete semantic computation account."},
        "terminal_status":"unassessed"}
    additional = []
    ram,calibration,rss = 16000,"job-1424b0346278",16323
    if predecessor is not None:
        from saturn_pub.store import LocalStore

        if annotation_path is None:
            raise ValueError("spatial stage needs explicit reference annotations")
        audit = json.loads((predecessor/"analysis-v1.json").read_text())
        if audit["mechanics_status"]!="valid" or audit["stage"]!="references":
            raise ValueError("independently audited anime reference panel required")
        data = predecessor/"extracted/flux-anime-result"
        source = json.loads((data/"source.json").read_text())
        old = json.loads((data/"plan.json").read_text())
        annotations = json.loads(annotation_path.read_text())
        if old["scenes"]!=scenes or annotations["source_job_id"]!=audit["job_id"]:
            raise ValueError("reference scene/annotation source differs")
        manifest = json.loads((data/"files.json").read_text())
        cuts,relatives = source["parents"],set()
        for value in cuts.values():
            relative = "typed-store/cuts/"+value+".json"
            record = json.loads((data/relative).read_text())
            relatives.add(relative)
            relatives.update("typed-store/pages/"+h+".json" for h in record["slots"].values())
            relatives.update("typed-store/blobs/"+h+".safetensors" for h in record["physical_blobs"])
        external_inputs = {}
        for relative in sorted(relatives):
            path = data/relative
            if sha(path)!=manifest[relative]:
                raise ValueError("historical CAS differs")
            external_inputs[relative] = {"sha256":manifest[relative],"bytes":path.stat().st_size}
        store = LocalStore(data/"typed-store")
        geometry = runpy.run_path(str(here/"flux_scene_spatial.py"))
        supports,historical = {},{}
        for scene in scenes:
            key = scene["id"]
            annotation = annotations["scenes"][key]
            for order,label in (("base","base"),("swap","source")):
                row = json.loads((data/"rows"/(key+".native."+order+".json")).read_text())
                if annotation[label+"_image_sha256"]!=row["image"]["sha256"] or sha(data/row["image"]["path"])!=row["image"]["sha256"]:
                    raise ValueError("annotated reference RGB differs")
            cut = store.load(cuts[key+".base"],device="cpu")
            supports[key] = geometry["supports"](annotation["base"],annotation["source"],cut.payload["img_ids"].numpy(),height=512,width=512)
            native = json.loads((data/"rows"/(key+".native.base.json")).read_text())["image"]["sha256"]
            historical.update({key+".native":native,key+".sham":native,
                key+".full":json.loads((data/"rows"/(key+".full.json")).read_text())["image"]["sha256"]})
        config.update(stage="spatial",rows=helper["spatial_rows"](scenes),max_adapter_steps=1100,
            input_cuts=cuts,annotations=annotations["scenes"],supports=supports,historical_pngs=historical,
            external_input_root=json.loads((predecessor/"job.json").read_text())["result"]["diagnostics"]["cwd"]+"/flux-anime-result",
            external_inputs=external_inputs,annotation_sha256=sha(annotation_path),predecessor_audit_sha256=sha(predecessor/"analysis-v1.json"),
            allowed_information="Audited four anime prefix cuts, six known final references/full-L crosses and explicit manual reference-image windows; no new spatial patch outcome",
            hypotheses={
                "H-replay":"Native and sham PNGs match known base; full latent PNGs match preceding full-L/base-C crossed futures. Six mechanics controls, ten new spatial futures.",
                "H-left":"Tentative: left-only source window makes the left character B-like while right retains B-like appearance: duplicated source B hair/outfit/prop associations. Judge count, hair, clothes, props and face drift separately.",
                "H-right":"Tentative: right-only source window makes right character A-like while left retains A-like appearance. Partial identities and borrowed limbs/props remain observations.",
                "H-both":"Tentative: stacked left/right windows preserve source B-left/A-right appearance associations under base conditioning. No exact face or full RGB equivalence forecast.",
                "H-location":"Tentative: outside-window edit preserves base A-left/B-right major outfit/hair associations. Shifted control is exploratory if its support overlaps character windows; preserve measured overlap.",
                "H-scope":"Two anime scenes/seeds, one checkpoint and clock; manually supplied reference windows, not automatic person segmentation or latent semantic IDs. Native future rendering and appearance observations, no independently calculated images."})
        additional.append((annotation_path,"reference-annotations.json"))
        ram,calibration,rss = 10000,"job-79259af4ad40",7520
    run_dir.mkdir(parents=True,exist_ok=False)
    save(run_dir/"plan.json",config)
    worker = here/"flux_anime_campaign.py"
    extras = [(here/"flux_scene_inventory_campaign.py","flux_scene_inventory_campaign.py"),
              (here/"flux_scene_spatial.py","flux_scene_spatial.py"),(here/"discovery_upgrades.py","campaign_support.py"),
              (run_dir/"plan.json","plan.json"),*additional]
    preflight = run_dir/"preflight.v1.json"
    save(preflight,{"schema_version":"mrun-preflight-config-v1","hashed_roots":[worker.name,*(n for _,n in extras),preflight.name],
        "path_token_allowlist":["flux-anime-result","input-store",config["model_path"],*([config["external_input_root"]] if predecessor else [])]})
    extras.append((preflight,preflight.name))
    environment = intent()
    if submit and environment["uv_cache_dir"].startswith("/Users/"):
        raise ValueError("configure measured Linux MDB_WORKER cache profile")
    raw = (json.dumps(config,sort_keys=True)+"\n").encode()
    needs = {"cuda":True,**({"host":"beast"} if predecessor else {})}
    request = {"schema":"mdb-flux-anime-request-v1","config":config,"task_family":"forward","model":config["model"],
        "needs":needs,"backend":"diffusers","device":"cuda","dtype":"bfloat16","batch_size":1,"height":512,"width":512,
        "execution_environment":environment,"config_bytes_sha256":hashlib.sha256(raw).hexdigest(),
        "source_hashes":{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=len(config["rows"]),
            logical_branches=len(config["rows"]),logical_checkpoint_replays=len(config["rows"]),max_local_work_items=len(config["rows"]),
            checkpoint_policy="Two original adult anime casts/scenes, finite sequential native public Navigator. References:4encodes/4prefixes/6RGB. Spatial:0encodes/16RGB/6knowncontrols/10new. One lease/load/B1/no retries."),
        "resource_exception":{"ram_mb":ram,"vram_mb":13000,"calibration_job":calibration,"measured_peak_rss_mb":rss,
            "measured_peak_vram_mb":9632 if predecessor is None else 9262,
            "reason":"Same qualified resident CUDA BF16 B1 512px four-step consumer. Reference lane serializes4encodes; spatial lane imports4bounded cuts without encoding. Measured envelope plus serialization margin; no guessed model batching."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run_dir/"request.json",request)
    stage_worker(run_dir/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    if not submit:
        return {"submitted":False,"run_dir":str(run_dir)}
    job = guarded_mrun_launch(worker_command(worker.name),request=request,model=config["model"],task="forward",
        experiment="mdb-flux-anime-"+config["stage"],payload=run_dir/"payload",prefer_host="beast",gpu=True,
        needs=needs,ram_mb=ram,vram_mb=13000,cpu_threads=4,est_wall_s=300,timeout_s=600,retry_on_kill=False,preflight=True,
        note=f"Klein4B anime {config['stage']} CUDA BF16 B1 512px4steps.2scenes seeds47/83.1load;finite{len(config['rows'])}RGB;step2project cuts;receiverC retained in edits.4encodes in refs/0spatial;stop{config['max_adapter_steps']}transitions/600s;no retries;no generated/repainted evidence.")
    save(run_dir/"submission.json",{"job_id":str(job)})
    return {"job_id":str(job),"run_dir":str(run_dir)}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir",type=Path)
    p.add_argument("--predecessor",type=Path)
    p.add_argument("--annotations",type=Path)
    p.add_argument("--stage-only",action="store_true")
    p.add_argument("--status")
    p.add_argument("--collect")
    args = p.parse_args()
    if args.status:
        from mdb.job_support import terminal_job

        _,job,logs = terminal_job(args.status)
        phases = [line for line in logs.splitlines() if line.startswith("MDB_FLUX_ANIME_PHASE ")]
        result = {"job_id":args.status,"state":job["state"],"last_phases":phases[-2:],
            "sealed_phase":next((line for line in phases if '"kind": "predictions_sealed"' in line),None),
            "last_error_lines":logs[-2400:] if job["state"]=="failed" else None}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir,args.collect,log_storage="gzip",max_compressed_bytes=512 << 20,max_expanded_bytes=1 << 30)
    else:
        result = stage(args.run_dir.resolve(),predecessor=args.predecessor.resolve() if args.predecessor else None,
            annotation_path=args.annotations.resolve() if args.annotations else None,submit=not args.stage_only)
    print(json.dumps(result,sort_keys=True))
