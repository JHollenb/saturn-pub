"""Stage and admit a finite native interaction/program correction experiment."""
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


def stage(run,source):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    data=source/"run-01/extracted/eye-program-result";diag=source/"interaction-diagnostic-v1"
    audit=read(source/"analysis-v2.json")
    if audit["mechanics_status"]!="verified":raise ValueError("verified source required")
    plan,binding,prep,summary=[read(data/n) for n in ("plan.json","binding.json","preparation.json","summary.json")]
    inputs={"base":plan["input_cuts"]["post.base"]["cut"],"joint":plan["input_cuts"]["post.program"]["cut"],"additive":binding["parent"],
        **{role:prep[role]["post_cut"] for role in ("color","shape")}}
    manifest=read(data/"files.json");names=set()
    for fingerprint in inputs.values():
        name="store/cuts/"+fingerprint+".json";names.add(name);cut=read(data/name)
        names.update("store/pages/"+h+".json" for h in cut["slots"].values());names.update("store/blobs/"+h+".safetensors" for h in cut["physical_blobs"])
    add=read(data/"forecasts.json")["A"];names.add(add["path"])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError("historical CAS differs")
        files[name]={"bytes":(data/name).stat().st_size,"sha256":manifest[name]}
    diag_report=read(diag/"analysis.json");fields={"additive":add};extras=[]
    for role in ("base","color","shape","joint"):
        filename=role+".A.bf16.npy";meta=diag_report["files"][filename]
        if sha(diag/filename)!=meta["sha256"]:raise ValueError("prior equation field differs")
        fields[role]={**meta,"path":filename};extras.append((diag/filename,filename))
    branches=[{"id":program+"."+field,"program":program,"kind":"predicted-current-write","field":field}
        for program in ("additive","joint") for field in ("r0c0","r0c1","r1c0","r1c1")]
    branches.extend({"id":"additive."+field,"program":"additive","kind":"negative-dose","field":field} for field in ("r-1c0","r0c-1"))
    branches.extend({"id":program+".sham","program":program,"kind":"sham"} for program in ("additive","joint"))
    config={"schema":"mdb-interaction-corrections-plan-v1","model":plan["model"],"model_path":plan["model_path"],"checkpoint_sha256":plan["checkpoint_sha256"],
        "input_root":summary["output_root"]+"/eye-program-result","input_files":files,"input_cuts":inputs,"prediction_fields":fields,
        "prior_diagnostic":{"sha256":sha(diag/"analysis.json"),"plan_sha256":sha(diag/"plan.json"),"audit_sha256":sha(source/"analysis-v2.json"),
            "scope":"Prior saved-state conditional equation fields; additive/nativebase outputs were previously observed, the four diagnostic fields had no native validation. New correction-intervention outcomes are not supplied."},
        "conditions":{"seed":60665,"logical_step":3,"single_block":7,"height":512,"width":512,"batch_size":1,"diffusion_steps":4,
            "receiving_state":"Complete common baseI/L/E/time/position; Tadd and Tjoint at blockentry differ. Native imageM is fixed; native post7 T selects continuing program.",
            "observed":"Saved prefixes, actualparameters/position operands, equationfields fromprioranalysis. Afterseal native M/text/current block and projection are observed consumers.",
            "predicted":"BF16(Alinear + r*R + c*C) image attention operands; generic equation references, no raw-prompt or whole-model imageforecast."},
        "hypotheses":{
            "continuation":"Full R+C current-A correction under continuingTadd is predicted insufficient to reproduce the native joint-program future. ContinuingTjoint should reduce joint-reference discrepancy more than changing just the currentA, averaged over the2x2write grid.",
            "semantic":"Both continuingprograms retain violet/pink-looking eyes. Current corrections may change hue/eyelid details and collateral; no monotone semantic ordering is asserted by local norms.",
            "opposition":"R and C oppose as vectors. Their image effects and signed projection along historical joint-minus-additiveRGB may oppose, cancel or depend on continuingT. Preserve all dose/corner responses.",
            "controls":"Nativebase/additive/joint PNGs reproduce historical controls, shams exact; predictedr0c1 andr1c1 should approximate measuredA on matchingT parents. All equation errors retained."},
        "observers":{"scope":"Supplied fixed pixel windows, not learned object segmentation or a calibrated identity/hue/shape model. Historical joint-minus-additive RGB defines signed contrast, including geometry/shading.",
            "windows":{"eyes":[140,155,370,215],"hair":[140,20,380,145],"clothing":[150,375,360,500],"background":[0,0,85,290]},
            "signed_contrast_reference_pngs":{"joint":sha(data/"text.only.png"),"additive":sha(data/"additive.png")},
            "intermediate":"At pre-single11 retain3reference fullI/T arrays; candidates retain descriptor/64witnesses/fullonlineL2+signedmean. Candidate complete intermediate arrays are not retained."},
        "branches":branches,"historical_pngs":{"native.base":sha(data/"native.base.png"),"native.additive":sha(data/"additive.png"),"native.joint":sha(data/"text.only.png")},
        "output_scratch":"/home/beast/.cache/mdb-research","max_wall_seconds":600,
        "scope":"One source-assisted same-specimen writer/program factorial;15dependent native futures/213transitions/15Acts/5directblocks/4validationblocks/6unique projections/15VAE. No encodes/new raw prompts. Candidate correction fields sealed before new currentblock comparisons; no prior-outcome claim erased."}
    run.mkdir(parents=True,exist_ok=False);save(run/"plan.json",config)
    here=Path(__file__).resolve().parent;worker=here/"flux_interaction_corrections_campaign.py"
    names=["flux_predicted_future_campaign.py","flux_scene_dependencies.py","flux_single_contraction.py","flux_interaction_corrections.py","flux_family_custody.py"]
    extras += [(here/n,n) for n in names]+[(run/"plan.json","plan.json")];pre=run/"preflight.v1.json"
    save(pre,{"schema_version":"mrun-preflight-config-v1","hashed_roots":[worker.name,*[n for _,n in extras],pre.name],
        "path_token_allowlist":["interaction-corrections-result","input-data",config["model_path"],config["input_root"],config["output_scratch"]]});extras.append((pre,pre.name))
    raw=(json.dumps(config,sort_keys=True)+"\n").encode()
    request={"schema":"mdb-interaction-corrections-request-v1","config":config,"model":config["model"],"task_family":"forward","backend":"diffusers","device":"cuda","dtype":"bfloat16",
        "batch_size":1,"height":512,"width":512,"needs":{"cuda":True,"host":"beast"},"execution_environment":intent(),"config_bytes_sha256":hashlib.sha256(raw).hexdigest(),
        "source_hashes":{n:sha(p) for p,n in [(worker,worker.name),*extras]},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_operations=15,logical_branches=15,logical_checkpoint_replays=15,max_local_work_items=24,
            checkpoint_policy="3sourceTActs,3native15-step references,12native14-step futures:213transitions15Acts228receipts.5direct sourceblocks4validationblocks6uniqueprojectioncalls. NativeM/T/projection/suffix retained; no encodes/retries."),
        "resource_exception":{"ram_mb":10000,"vram_mb":13000,"disk_gb":6,"calibration_job":"job-2f0cd41dfc45",
            "reason":"Same one resident no-encoder Klein4B B1 CUDA BF16 512px lane.15 vs13 decodes; source captures bounded5blocks, no new fullweights retained, intermediate referencearrays only3xI/T. Historical measured10GBRAM13GBVRAM reservation; not a speedup claim."}}
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/"request.json",request)
    stage_worker(run/"payload",worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return {"run_dir":str(run),"request_sha256":request["request_sha256"],"submitted":False}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    r=read(run/"request.json")
    if (run/"submission.json").exists() or any(sha(run/"payload"/n)!=h for n,h in r["source_hashes"].items()):raise ValueError("admitted or mutated stage")
    cmd=worker_command("flux_interaction_corrections_campaign.py");cmd.insert(1,"PYTHONDONTWRITEBYTECODE=1")
    job=guarded_mrun_launch(cmd,request=r,model=r["model"],task="forward",experiment="mdb-interaction-corrections",payload=run/"payload",prefer_host="beast",gpu=True,needs=r["needs"],
        ram_mb=10000,vram_mb=13000,disk_gb=6,cpu_threads=4,est_wall_s=240,timeout_s=720,retry_on_kill=False,preflight=True,
        note="Klein512pxBF16B1 noencoder;15nativefutures,213transitions15Acts228receipts5directblocks4validationblocks6projections15VAE. Seal R/C2x2attentioncorrections beforecurrentblock, compare additive/jointcontinuingT, negative doses/shams. FixedimageM, sourceassistedpriorfields; retaintarget/collateral/midtrajectory.10GBRAM13GBVRAMhistoricalexception;600sworker720slease,no retries.")
    save(run/"submission.json",{"job_id":str(job)});return {"job_id":str(job),"run_dir":str(run)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("--source",type=Path);p.add_argument("--submit",action="store_true");p.add_argument("--collect");a=p.parse_args()
    if a.collect:r=runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](a.run.resolve(),a.collect)
    elif a.submit:r=submit(a.run.resolve())
    else:r=stage(a.run.resolve(),a.source.resolve())
    print(json.dumps(r,sort_keys=True))
