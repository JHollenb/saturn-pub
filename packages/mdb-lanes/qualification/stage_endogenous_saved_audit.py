"""One finite CPU saved-only audit of a succeeded endogenous-write campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "audit_flux_endogenous_write.py"
WORKER_SHA256 = "7de3f5c6a0107f53004efac45bcee0bc12e5b2ea73c2fe4554b2fa7727501533"
PRODUCER_SHA256 = "8440e481c1ab057eabe643291b75e4d30b79acc5de83f1f5b2fd212537118810"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path,value):
    with Path(path).open("x")as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({k:v for k,v in request.items()if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()


def bind(environment):
    from mdb_lanes.execution_environment import intent

    names=dict(uv_cache_dir="MDB_WORKER_UV_CACHE_DIR",torch_home="MDB_WORKER_TORCH_HOME",extra_model_roots="MDB_WORKER_MODEL_ROOTS",python="MDB_WORKER_PYTHON",uv_link_mode="MDB_WORKER_UV_LINK_MODE")
    for key,name in names.items():
        os.environ[name]=environment[key]
    actual=intent()
    if actual!={key:environment[key]for key in actual}:
        raise ValueError("qualified worker environment changed")


def command(request):
    from mdb_lanes.payload import worker_command

    bind(request["execution_environment"])
    args=worker_command(WORKER);args.insert(1,"CUDA_VISIBLE_DEVICES=")
    for name in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        args.insert(1,name+"=4")
    args.insert(args.index("python")+1,"-B");args.extend(["--config","saturn_run_config.json"])
    return args


def validate_parent(parent):
    if identity(parent)!=parent.get("request_sha256")or parent["config"]["schema"]!="mdb-flux-endogenous-write-v1"or parent["source_hashes"].get("flux_endogenous_write_campaign.py")!=PRODUCER_SHA256:
        raise ValueError("exact endogenous-write parent request/source required")


def validate_native_custody(parent,job,command):
    from mdb_lanes.job_support import request_sha256

    # mrun binds the full request, including its experimental self-identity.
    # The published experiment ID excludes that field and is a different hash.
    if (job["config"].get("saturn_request_sha256")!=request_sha256(parent)
            or job["result"]["diagnostics"]["command"]!=command):
        raise ValueError("actual native command/request custody differs")


REMOTE = r'''
import json,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);result={}
for name in('summary.json','files.json','source.json','native-contract.json','runtime-contract.json','forecast-transaction/seal.json'):
 path=root/name
 if path.is_symlink() or any(p.is_symlink()for p in path.parents) or path.stat().st_size>16<<20:raise ValueError('unsafe source metadata')
 data=path.read_bytes();result[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
summary=json.loads((root/'summary.json').read_text())
print(json.dumps(dict(files=result,job_id=summary['job_id'],request_sha256=summary['request_sha256'])))
'''


def stage(run,source):
    from mdb_lanes.job_support import one_lease_scheduling_contract, scheduler_api
    from mdb_lanes.payload import stage_worker

    run,source=Path(run).resolve(),Path(source).resolve();parent=read(source/"request.json");validate_parent(parent)
    os.environ["MRUN_URL"]=parent["scheduler_url"];bind(parent["execution_environment"])
    for name,expected in {**parent["source_hashes"],**parent["runtime_wheels"]}.items():
        if sha(source/"payload"/name)!=expected:
            raise ValueError("original source/wheel changed: "+name)
    jid=read(source/"submission.json")["job_id"];api=scheduler_api()
    status,raw,_=api.request("GET","/api/jobs/"+jid,timeout_s=30);job=json.loads(raw)
    if status!=200 or job["state"]!="succeeded"or job["assigned_host"]!="beast":
        raise ValueError("one succeeded native source job required")
    validate_native_custody(parent,job,read(source/"admission-command.json")["command"])
    status,log,_=api.request("GET","/api/jobs/"+jid+"/logs?offset=0&wait_s=1",timeout_s=30)
    if status!=200 or len(log)>16<<20:
        raise ValueError("bounded complete native log required")
    declared=job["result"].get("artifacts",{}).get("job_log")
    if declared and(len(log)!=declared["bytes"]or hashlib.sha256(log).hexdigest()!=declared["sha256"]):
        raise ValueError("actual native log checksum differs")
    root=Path(parent["config"]["output_scratch"])/jid/parent["config"]["output_name"]
    info=json.loads(subprocess.run(["ssh","beast",shlex.join(["python3","-I","-c",REMOTE,str(root)])],check=True,capture_output=True,text=True).stdout)
    publications=[json.loads(row.removeprefix("MDB_DIRECTORY_READY "))for row in log.decode().splitlines()if row.startswith("MDB_DIRECTORY_READY ")]
    if len(publications)!=1:
        raise ValueError("one native publication required")
    publication=publications[0]
    if (info["job_id"]!=jid or info["request_sha256"]!=parent["request_sha256"]or publication["job_id"]!=jid
            or publication["request_sha256"]!=parent["request_sha256"]or publication["output_root"]!=str(root)
            or publication["files_manifest_sha256"]!=info["files"]["files.json"]["sha256"]or publication["summary_sha256"]!=info["files"]["summary.json"]["sha256"]):
        raise ValueError("native publication identity differs")
    worker_path=HERE/WORKER
    if sha(worker_path)!=WORKER_SHA256:
        raise ValueError("frozen saved reader changed")
    worker=runpy.run_path(str(worker_path))
    publication_id=hashlib.sha256(json.dumps(publication,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    config=worker["make_config"](dict(native_root=str(root),native_job_id=jid,native_request_sha256=parent["request_sha256"],
        native_manifest_sha256=publication["files_manifest_sha256"],native_summary_sha256=publication["summary_sha256"],
        publication_id=publication_id,native_publication=publication,output_scratch=str(root.parents[1])))
    worker["validate_config"](config)
    run.mkdir(parents=True,exist_ok=False)
    for name,value in(("source-job.json",job),("source-metadata.json",info),("source-request.json",parent),("source-publication.json",publication)):
        save(run/name,value)
    (run/"source-worker.log").write_bytes(log)
    sources={WORKER:worker_path}
    for name, expected in worker.get("HELPERS", {}).items():
        path=HERE/name
        if Path(name).name!=name or sha(path)!=expected:
            raise ValueError("frozen saved-reader helper changed: "+name)
        sources[name]=path
    for name in("source-job.json","source-metadata.json","source-request.json","source-publication.json","source-worker.log"):
        sources[name]=run/name
    save(run/"plan.json",config);sources["plan.json"]=run/"plan.json"
    save(run/"preflight.v1.json",dict(schema="mrun-preflight-config-v1",hashed_roots=[WORKER],
        path_token_allowlist=[str(root),config["output_scratch"],"/mnt/big/llm-models/FLUX.2-klein-4B"]))
    sources["preflight.v1.json"]=run/"preflight.v1.json"
    request=dict(schema="mdb-endogenous-saved-analysis-request-v1",panel="endogenous-flux-audit",config=config,
        scheduler_url=parent["scheduler_url"],execution_environment=parent["execution_environment"],runtime_wheels=parent["runtime_wheels"],
        source_hashes={name:sha(path)for name,path in sources.items()},launcher_sha256=sha(__file__),source_job_id=jid,
        source_request_sha256=parent["request_sha256"],source_terminal_state="succeeded",needs={"cpu":True,"host":"beast"},
        backend="numpy",device="cpu",dtype="float64",task_family="analysis",timeout_s=900,
        resource_exception=dict(ram_mb=4096,vram_mb=0,cpu_threads=4,disk_gb=2,est_wall_s=180),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=128,logical_branches=1,
            logical_checkpoint_replays=0,max_local_work_items=256,extra={"task_family":"analysis"},
            checkpoint_policy="Saved-only original bytes; zero models/forwards; no retries; full comparisons and compact reports only."))
    request["request_sha256"]=identity(request);save(run/"request.json",request)
    stage_worker(run/"payload",worker=worker_path,config=config,request=request,additional_files=[(path,name)for name,path in sources.items()if name!=WORKER])
    for name,expected in {**request["source_hashes"],**request["runtime_wheels"]}.items():
        if sha(run/"payload"/name)!=expected:
            raise ValueError("staged saved audit source/wheel changed")
    save(run/"staged-command.json",dict(command=command(request),request_sha256=request["request_sha256"],submitted=False))
    return dict(request_sha256=request["request_sha256"],native_job_id=jid,submitted=False)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run=Path(run).resolve();request=read(run/"request.json")
    if identity(request)!=request["request_sha256"]or sha(__file__)!=request["launcher_sha256"]or(run/"submission.json").exists():
        raise ValueError("saved audit changed or already submitted")
    for name,expected in {**request["source_hashes"],**request["runtime_wheels"]}.items():
        if sha(run/"payload"/name)!=expected:
            raise ValueError("saved audit source/wheel changed")
    os.environ["MRUN_URL"]=request["scheduler_url"];args=command(request)
    save(run/"admission-command.json",dict(command=args,request_sha256=request["request_sha256"]))
    jid=guarded_mrun_launch(args,request=request,experiment="mdb-endogenous-flux-audit",task="analysis",model=None,payload=run/"payload",
        needs=request["needs"],gpu=False,prefer_host="beast",timeout_s=900,retry_on_kill=False,preflight=True,**request["resource_exception"],
        note="Saved-only endogenous-write audit:16complete images,96first-consumer ports,native projection geometry,own-state Acts and lineage. CPU4/RAM4GiB,180s estimate/900s cap;zero models/forwards,2tinyBNstats transient,no parameter bank,no retry. Original source inventory reverified;compact reports only.")
    save(run/"submission.json",dict(job_id=str(jid),request_sha256=request["request_sha256"]))
    return dict(job_id=str(jid))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("run",type=Path);parser.add_argument("--source",type=Path);parser.add_argument("--submit",action="store_true")
    args=parser.parse_args()
    if not args.submit and args.source is None:
        parser.error("staging requires --source")
    print(json.dumps(submit(args.run)if args.submit else stage(args.run,args.source),sort_keys=True))
