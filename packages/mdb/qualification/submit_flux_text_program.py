"""Stage one finite text-program writer and prediction assay from matched cuts."""
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
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    data=source/"run-01/extracted/interaction-corrections-result";audit=read(source/"analysis-v1.json")
    if audit["mechanics_status"]!="verified":raise ValueError("verified source required")
    plan,summary=read(data/"plan.json"),read(data/"summary.json");rows={r["id"]:r for r in read(data/"rows.json")}
    source_rows={"low":"additive.r1c1","high":"joint.r1c1"};cuts={role:rows[name]["actions"][-1]["post_act"] for role,name in source_rows.items()}
    manifest=read(data/"files.json");names=set()
    for fingerprint in cuts.values():
        name="store/cuts/"+fingerprint+".json";names.add(name);cut=read(data/name)
        names.update("store/pages/"+h+".json" for h in cut["slots"].values());names.update("store/blobs/"+h+".safetensors" for h in cut["physical_blobs"])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError("source CAS differs")
        files[name]={"bytes":(data/name).stat().st_size,"sha256":manifest[name]}
    config={"schema":"mdb-text-program-plan-v1","model":plan["model"],"model_path":plan["model_path"],"checkpoint_sha256":plan["checkpoint_sha256"],
        "input_root":summary["output_root"]+"/interaction-corrections-result","input_files":files,"input_cuts":cuts,
        "historical_pngs":{"native."+role:sha(data/(name+".png")) for role,name in source_rows.items()},
        "source_audit_sha256":sha(source/"analysis-v1.json"),"conditions":{"seed":60665,"height":512,"width":512,"steps":4,"batch_size":1,"logical_step":3,"block":8},
        "hypotheses":{
            "carrier":"Incoming text residual R is predicted to carry most of the tested high-versus-low continuing-program contrast. R1 corners should be closer to native.high than R0 corners; A/Mtext differences can alter details and collateral without a monotone universal sign.",
            "writes":"Hold post8 image equal to native.high for all11edited futures. Factorial textT=Rselected + nativegate*P[Aselected;Mselected] separates inheritedprogram from currenttextwrites; R0A0M0 is not assumed equal to native.low because their image exits differ.",
            "prediction":"Complete textA/M forecasts for both observed pre8T states from actualweights/modulation/position should approximate their native fields and produce close native continuations to measuredR0A0M0/R1A1M1 under the same fixedimage. Reference arithmetic is not CUDA emulation.",
            "controls":"Two historical sourcePNGs, highsham and highnativecube corner exact. Low/high cachedT corners match measuredT;2mixed cachedT endpoints match actualfullblock textoverride. Preserve all numerical errors and semanticcounterexamples."},
        "observers":{"windows":plan["observers"]["windows"],"scope":"Supplied fixed pixelwindows and high-minus-low RGBcontrast; unblinded semanticreview separate, no calibrated hue/shape/identity or automaticsegmentation."},
        "output_scratch":"/home/beast/.cache/mdb-research","max_wall_seconds":600,
        "scope":"One knowncontext lateclock textcarrier/write investigation. FulltextA/M forecasts sealedbeforecurrentblock; source-assisted RAMfactorial; fixedhighimageexit/nativetextAct/suffix/VAE.13dependentfutures171transitions11Acts182receipts2sourceblocks2validationblocks6projections. One load/noencodes/no retries."}
    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config);here=Path(__file__).resolve().parent;worker=here/"flux_text_program_campaign.py"
    names=["flux_text_program.py","flux_predicted_future_campaign.py","flux_scene_dependencies.py","flux_single_contraction.py","flux_attention_reference.py","flux_mlp_replacement.py","flux_family_custody.py"]
    extras=[(here/n,n) for n in names]+[(run/"plan.json","plan.json")];pre=run/"preflight.v1.json"
    save(pre,{"schema_version":"mrun-preflight-config-v1","hashed_roots":[worker.name,*names,"plan.json",pre.name],
        "path_token_allowlist":["text-program-result","input-data",config["model_path"],config["input_root"],config["output_scratch"]]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request={"schema":"mdb-text-program-request-v1","config":config,"model":config["model"],"task_family":"forward","backend":"diffusers","device":"cuda","dtype":"bfloat16","batch_size":1,"height":512,"width":512,
        "needs":{"cuda":True,"host":"beast"},"execution_environment":intent(),"config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"source_hashes":{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=13,logical_branches=13,logical_checkpoint_replays=13,max_local_work_items=24,
            checkpoint_policy="2native14-step references+11edited13-step suffixes=171transitions,11Acts182receipts.2directsourceblocks,2validationblocks6projections13VAE. Predictionsinprocess beforeblock8, noencodes/retries."),
        "resource_exception":{"ram_mb":10000,"vram_mb":13000,"disk_gb":6,"calibration_job":"job-20c6f6c6773c",
            "reason":"Same noencoder residentKlein4B CUDA BF16B1 512px lane.2sourcecaptures instead5, fulltextA/M512queries ratherthanimage1024, tiledNumPypredictors and fullweightarchive.10GBRAM13GBVRAM measuredadmission lane; not a throughputclaim."}}
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/"request.json",request)
    stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {"run_dir":str(run),"request_sha256":request["request_sha256"],"submitted":False}


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    r=read(run/"request.json")
    if (run/"submission.json").exists() or any(sha(run/"payload"/n)!=h for n,h in r["source_hashes"].items()):raise ValueError("admitted or mutated stage")
    cmd=worker_command("flux_text_program_campaign.py");cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(cmd,request=r,model=r["model"],task="forward",experiment="mdb-text-program",payload=run/"payload",prefer_host="beast",gpu=True,needs=r["needs"],
        ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note="Klein512pxBF16B1 resident noencoder:13nativefutures171transitions11Acts182receipts2directblocks2validationblocks6projections13VAE. Predictcomplete textA/Mbeforeblock8, then R/A/Mtextcube withfixedimagehigh. Recordedparents differonlyT; preservefine/collateral/controls.10GBRAM13GBVRAMcalibratedlane;600sworker720slease, no retries.")
    save(run/"submission.json",{"job_id":str(job)});return {"job_id":str(job),"run_dir":str(run)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--source",type=Path);p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
