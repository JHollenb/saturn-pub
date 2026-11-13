"""Stage a finite common/differential factorial and fresh fox/cat native source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORKER="flux_common_differential_campaign.py"


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path,value):
    with Path(path).open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({k:v for k,v in request.items() if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()


def bind(environment):
    from mdb_lanes.execution_environment import intent

    names=dict(uv_cache_dir="MDB_WORKER_UV_CACHE_DIR",torch_home="MDB_WORKER_TORCH_HOME",extra_model_roots="MDB_WORKER_MODEL_ROOTS",python="MDB_WORKER_PYTHON",uv_link_mode="MDB_WORKER_UV_LINK_MODE")
    for key,name in names.items():os.environ[name]=environment[key]
    actual=intent()
    if actual!={k:environment[k] for k in actual}:raise ValueError("qualified installed runtime changed")


def command(request):
    from mdb_lanes.payload import worker_command

    bind(request["execution_environment"]);args=worker_command(WORKER)
    args.insert(args.index("python")+1,"-B");args.extend(["--config","saturn_run_config.json"])
    for key,name in dict(tmpdir="TMPDIR",cuda_cache_path="CUDA_CACHE_PATH",torchinductor_cache_dir="TORCHINDUCTOR_CACHE_DIR").items():args.insert(1,name+"="+request["execution_environment"][key])
    return args


def validate(run,request):
    if identity(request)!=request.get("request_sha256") or sha(__file__)!=request["launcher_sha256"]:raise ValueError("request or launcher changed")
    for name,expected in {**request["source_hashes"],**request["runtime_wheels"]}.items():
        if Path(name).name!=name or sha(run/"payload"/name)!=expected:raise ValueError("staged source/wheel changed: "+name)


def stage(run,parent_run,design):
    from mdb_lanes.job_support import one_lease_scheduling_contract, request_sha256
    from mdb_lanes.payload import stage_worker

    run,parent_run,design=(Path(p).resolve() for p in (run,parent_run,design))
    parent=read(parent_run/"request.json");worker=runpy.run_path(str(HERE/WORKER))
    if identity(parent)!=worker["SOURCE_REQUEST"] or parent["source_hashes"].get(worker["BASE"])!=worker["BASE_SHA256"]:raise ValueError("exact previous native request required")
    source_job=read(parent_run/"job.json")
    if source_job["state"]!="succeeded" or source_job["job_id"]!=worker["SOURCE_JOB"] or source_job["config"]["saturn_request_sha256"]!=request_sha256(parent):raise ValueError("succeeded source job custody differs")
    if source_job["result"]["diagnostics"]["command"]!=read(parent_run/"admission-command.json")["command"]:raise ValueError("source native execution command differs")
    publication=read(parent_run/"publication.json")
    if publication["files_manifest_sha256"]!=worker["SOURCE_MANIFEST"] or publication["output_root"]!=worker["SOURCE_ROOT"]:raise ValueError("native source publication differs")
    for name,expected in {**parent["source_hashes"],**parent["runtime_wheels"]}.items():
        if sha(parent_run/"payload"/name)!=expected:raise ValueError("previous staged source/wheel changed: "+name)
    design_data=read(design)
    if design_data["schema"]!="mdb-common-differential-design-v1" or design_data["arms"]!=list(worker["ARMS"]) or design_data["source_manifest_sha256"]!=worker["SOURCE_MANIFEST"]:raise ValueError("source-only design differs")
    config=worker["make_config"](parent["config"]);sources={WORKER:HERE/WORKER}
    for name,expected in worker["HELPERS"].items():
        if sha(HERE/name)!=expected:raise ValueError("frozen worker helper changed: "+name)
        sources[name]=HERE/name
    for key in worker["path_keys"]():
        name=config[key]
        if Path(name).name!=name:raise ValueError("flat metadata names required")
        path=parent_run/"payload"/name
        if sha(path)!=parent["source_hashes"][name]:raise ValueError("source recipe changed: "+name)
        sources[name]=path;config[key+"_sha256"]=sha(path)
    local=dict(config)
    for key in worker["path_keys"]():local[key]=str(sources[config[key]])
    counts=worker["validate_config"](local)
    run.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(design,run/"campaign-design-v1.json")
    save(run/"plan.json",config);save(run/"count-contract.json",counts)
    for name in ("campaign-design-v1.json","plan.json","count-contract.json"):sources[name]=run/name
    roots={v for k,v in config.items() if isinstance(v,str) and (k.endswith(("_root","_path")) or k in ("output_scratch","output_name"))}
    save(run/"preflight.v1.json",dict(schema="mrun-preflight-config-v1",hashed_roots=list(sources),path_token_allowlist=sorted(roots),no_new_native_suffix_targets_before_global_seal=True,no_persistent_parameter_payload_exports=True))
    sources["preflight.v1.json"]=run/"preflight.v1.json"
    request=dict(schema="mdb-common-differential-request-v1",panel="common-differential-flux",config=config,execution_environment=parent["execution_environment"],scheduler_url=parent["scheduler_url"],
        runtime_wheels=parent["runtime_wheels"],source_hashes={name:sha(path) for name,path in sources.items()},launcher_sha256=sha(__file__),campaign_design_sha256=sha(design),source_request_sha256=parent["request_sha256"],
        model="flux2-klein-4b-distilled",backend="torch-native-common-differential",device="cuda",dtype="bfloat16",needs={"cuda":True,"host":"beast"},task_family="analysis",height=512,width=512,steps=4,flux_mode="capture",capture_sites=True,
        workload_geometry={**counts,"peak_physical_batch":1,"image_tokens":1024,"text_tokens":512,"output_bytes_bound":config["transport_limits"]["max_expanded_bytes"]},
        resource_exception=dict(ram_mb=24576,vram_mb=13000,cpu_threads=4,disk_gb=48,est_wall_s=750,timeout_s=2100,
            calibration="Native job-58dbdc76f751 measured474.38s,RSS10977MB,VRAM8242MB with16futures,1760transitions/lane,34decodes,2components. Prior three-component source job-445316274652 measured397.74s,RSS20744.6MB,VRAM8370MB. New18futures,1980transitions/lane,39decodes plus full signed interaction arrays and one native Qwen/project/joint0 prefix;3components,6sequential placements. Expect9-13min; retain measured24GiB/13000MB exception;1800s worker/2100s outer ceilings. Host pin required for immutable source StateCuts/checkpoint custody on Beast."),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_branches=18,logical_operations=120,logical_checkpoint_replays=18,max_local_work_items=256,
            extra={"task_family":"analysis","component_hydrations":3,"count_contract":counts},checkpoint_policy="One finite lease; immutable observed native source cuts, in-process forks/rewinds, global future seal, no nested submissions or retries."))
    request["request_sha256"]=identity(request);save(run/"request.json",request);bind(request["execution_environment"])
    stage_worker(run/"payload",worker=HERE/WORKER,config=config,request=request,additional_files=[(path,name) for name,path in sources.items() if name!=WORKER])
    validate(run,request);save(run/"staged-command.json",dict(command=command(request),request_sha256=request["request_sha256"],submitted=False))
    return dict(request_sha256=request["request_sha256"],counts=counts,source_files=len(sources),submitted=False)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run=Path(run).resolve();request=read(run/"request.json");validate(run,request)
    if (run/"submission.json").exists():raise ValueError("finite campaign already submitted")
    review=read(run/"finite-run-review-v1.json")
    if review.get("request_sha256")!=request["request_sha256"] or review.get("blocking_issues")!=[] or review.get("review_status")!="ready-for-finite-run":raise ValueError("request-bound finite review required")
    os.environ["MRUN_URL"]=request["scheduler_url"];args=command(request)
    save(run/"admission-command.json",dict(command=args,request_sha256=request["request_sha256"],review_sha256=sha(run/"finite-run-review-v1.json")))
    resources={k:v for k,v in request["resource_exception"].items() if k!="calibration"}
    job=guarded_mrun_launch(args,request=request,experiment="mdb-common-differential-flux",task="analysis",model=None,payload=run/"payload",needs=request["needs"],gpu=True,pin="beast",retry_on_kill=False,preflight=True,
        note="Common/differential native-write factorial:2saved roots+new fox/cat821 prefix.6arms/root,18image futures,1980F+1980N transitions,39decodes,full signed interaction arrays and first-consumer ports. Frozen613 selector field transfers tofox;719ownfield. Qwen+FLUX+VAE hydrate once,max1GPU. Expect9-13min;finite2100s lease,no retries/nesting;all suffix forecasts globally sealed before native targets.",**resources)
    save(run/"submission.json",dict(job_id=str(job),request_sha256=request["request_sha256"]));return dict(job_id=str(job))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("run",type=Path);parser.add_argument("--parent-run",type=Path);parser.add_argument("--design",type=Path);parser.add_argument("--submit",action="store_true")
    args=parser.parse_args()
    if not args.submit and (not args.parent_run or not args.design):parser.error("stage requires --parent-run and --design")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run,args.parent_run,args.design),sort_keys=True))

