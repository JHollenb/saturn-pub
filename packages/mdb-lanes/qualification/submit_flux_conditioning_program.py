"""Stage causal conditioning compositions at a fixed native latent frontier."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")


def stage(run,source):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    data=source/"run-01/extracted/predicted-future-result";audit=read(source/"analysis-v1.json")
    if audit["mechanics_status"]!="verified":raise ValueError("verified sources required")
    plan,binding,summary=[read(data/n) for n in ("plan.json","binding.json","summary.json")]
    cuts={k:binding["inputs"][k]["parent"] for k in ("base","color","shape","joint")};manifest=read(data/"files.json");names=set()
    for fingerprint in cuts.values():
        name="store/cuts/"+fingerprint+".json";names.add(name);record=read(data/name)
        names.update("store/pages/"+h+".json" for h in record["slots"].values());names.update("store/blobs/"+h+".safetensors" for h in record["physical_blobs"])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError("historical CAS differs")
        files[name]={"bytes":(data/name).stat().st_size,"sha256":manifest[name]}
    config={"schema":"mdb-conditioning-program-plan-v1","model":plan["model"],"model_path":plan["model_path"],"checkpoint_sha256":plan["checkpoint_sha256"],
        "input_root":plan["output_scratch"]+"/"+summary["job_id"]+"/predicted-future-result","input_files":files,"input_cuts":cuts,"historical_base_png":sha(data/"base.png"),
        "hypotheses":{
            "causal_interaction":"With commonbaseL3 and freshly projected final-step workingstate, observedjointE is predicted to yield a more joint-like violet-eye response than additiveE. Interaction-only Ebase+(Ejoint-Eadd) may give a partial or incompatible response; retain any sign/collateral ratherthan assume universal additivity.",
            "numerical":"All6complete initialT fields predicted from E and actualcontext_embedderweights beforeany newprojector entry. NativeL/context/otherworkingstate and remainingdenoiser/VAE consumers retained. Predictedadditive/joint T replacements should yield close futures to respective measuredT, with numericalerrors retained.",
            "controls":"Derived step3project frontier must reproduce storedpre6 basepayload in12native operations afterseal. BaseE nativefuture matcheshistoricalnaturalbasePNG. Sources remainobservedE fromfourprompts; no rawtextencoderforecast or donorfree semanticformation claim."},
        "observers":{"windows":{"eyes":[140,155,370,215],"hair":[140,20,380,145],"clothing":[150,375,360,500],"background":[0,0,85,290]},
            "scope":"Supplied pixelwindows/directRGBreview, no calibrated hue/shape/identity classifier. FixedL3 isbasehistory; current E effects are last-step causal sufficiency, not fullprompt trajectory equivalence."},
        "output_scratch":"/home/beast/.cache/mdb-research","max_wall_seconds":600,
        "scope":"Causalconditioning square+additive+interaction-only atsamelegalproject frontier, 6nativefinalsteps plus2predictedT suffixes and12-stepreconstruction.8images234transitions8Acts242receipts1load0encodes. FullinitialTforecasts beforecurrentprojector; observedsourceE/actualweights/nativenonlineardenoisersuffix."}
    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config);here=Path(__file__).resolve().parent;worker=here/"flux_conditioning_program_campaign.py"
    names=["flux_conditioning_program.py","flux_predicted_future_campaign.py","flux_scene_dependencies.py","flux_family_custody.py"];extras=[(here/n,n) for n in names]+[(run/"plan.json","plan.json")];pre=run/"preflight.v1.json"
    save(pre,{"schema_version":"mrun-preflight-config-v1","hashed_roots":[worker.name,*names,"plan.json",pre.name],
        "path_token_allowlist":["conditioning-program-result","input-data",config["input_root"],config["model_path"],config["output_scratch"]]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request={"schema":"mdb-conditioning-program-request-v1","config":config,"model":config["model"],"task_family":"forward","backend":"diffusers","device":"cuda","dtype":"bfloat16","batch_size":1,"height":512,"width":512,
        "needs":{"cuda":True,"host":"beast"},"execution_environment":intent(),"config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"source_hashes":{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=8,logical_branches=8,logical_checkpoint_replays=8,max_local_work_items=24,
            checkpoint_policy="6native28-step finals+2predicted27-step suffixes+12reconstruction=234transitions8Acts242receipts8VAE;1load0encodes;6independentlinearfieldforecasts beforeprojector; no retries."),
        "resource_exception":{"ram_mb":10000,"vram_mb":13000,"disk_gb":6,"calibration_job":"job-da6d4cedfd1d",
            "reason":"Same noencoder Klein4B resident512pxBF16B1 lane, actualcontext_embedder47MBweight/full6initialTfields, no fullMLP/QKVparameterwide copies;6serial finalsteps+2suffixes/8VAE.10GBRAM13GBVRAM measured admission lane, not speedupclaim."}}
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/"request.json",request)
    stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {"run_dir":str(run),"request_sha256":request["request_sha256"],"submitted":False}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    request=read(run/"request.json")
    if (run/"submission.json").exists() or any(sha(run/"payload"/n)!=h for n,h in request["source_hashes"].items()):raise ValueError("admitted or mutated stage")
    cmd=worker_command("flux_conditioning_program_campaign.py");cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(cmd,request=request,model=request["model"],task="forward",experiment="mdb-conditioning-program",payload=run/"payload",prefer_host="beast",gpu=True,needs=request["needs"],
        ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note="Klein4BBF16B1512px residentnoencoder. CommonbaseL3/legalproject: conditioningbase/color/shape/joint/additive/interaction-only;8nativeRGB234transitions8Acts242receipts. Seal6completeE->T linearforecasts beforeprojector/reconstruction;2predictedTnativefuturecomparisons. Retainsemanticmisses/collateral.10GBRAM13GBVRAMknownlane,600sworker720slease,noretries.")
    save(run/"submission.json",{"job_id":str(job)});return {"job_id":str(job),"run_dir":str(run)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--source",type=Path);p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
