"""Freeze completed discovery artifacts and submit only native recipe replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

from wsp_auto_qwen import save, sha


def stage(root, source):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command
    from mdb_lanes.wsp import verify_weight_study

    for key,value in {"MDB_WORKER_UV_CACHE_DIR":"/mnt/hdd8tb/mdb/uv-cache",
        "MDB_WORKER_TORCH_HOME":"/mnt/hdd8tb/mdb/torch","MDB_WORKER_MODEL_ROOTS":"",
        "MDB_WORKER_PYTHON":"3.14.6","MDB_WORKER_UV_LINK_MODE":"symlink"}.items():
        os.environ[key] = value
    root,source = Path(root).resolve(),Path(source).resolve()
    root.mkdir(parents=True,exist_ok=False)
    frozen = root/"frozen"
    frozen.mkdir()
    for name in ("capture","causal","discovery"):
        verify_weight_study(source/name)
    for name in ("capture","causal","discovery","factors","prior-evidence"):
        shutil.copytree(source/name,frozen/name)
    for name in ("recipe.json","computation-account.json","inputs.json","selection.json","causal-selection.json",
                 "fits.json","parameter-artifacts.json","retrieve_weights.py"):
        shutil.copy2(source/name,frozen/name)
    shutil.copy2(source/"request.json",frozen/"upstream-request.json")
    shutil.copy2(source/"config.json",frozen/"upstream-config.json")
    (frozen/"source-patch-manifests").mkdir()
    for family in ("semantic","random"):
        shutil.copy2(source/(family+"-patch")/"patch.json",frozen/"source-patch-manifests"/(family+".json"))
    save(frozen/"frozen-files.json",{p.relative_to(frozen).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size}
        for p in sorted(frozen.rglob("*")) if p.is_file()})
    archive = root/"frozen-program.tar.gz"
    with tarfile.open(archive,"w:gz") as stream:
        for p in sorted(frozen.iterdir()):
            stream.add(p,arcname=p.name)
    config = json.loads((source/"config.json").read_text())
    config.update(schema="mdb-computation-program-replay-plan-v1",max_wall_seconds=600,
        source_job_id="job-ca1c7ac68a91",frozen_program_sha256=sha(archive),
        replay_policy="no fitting or selection; regenerate and verify exact previously frozen parameter bytes")
    here = Path(__file__).resolve().parent
    sources = [here/name for name in ("wsp_program_replay.py","wsp_auto_qwen.py","audit_wsp_program.py","audit_wsp_selective.py")]
    raw = (json.dumps(config,sort_keys=True)+"\n").encode()
    preflight = {"schema_version":"mrun-preflight-config-v1",
        "path_token_allowlist":[config["model_path"],config["output_scratch"]],
        "hashed_roots":[*(p.name for p in sources),archive.name,"saturn_run_config.json","preflight.v1.json"]}
    save(root/"preflight.v1.json",preflight)
    request = {"schema":"mdb-computation-program-replay-request-v1","config":config,
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"execution_environment":intent(),
        "source_hashes":{p.name:sha(p) for p in [*sources,archive,root/"preflight.v1.json"]},
        "model":config["model"],"task_family":"recorder","backend":"hf","device":"cpu","dtype":"float32",
        "needs":{"cpu":True},"batch_size":32,"seq_len":64,
        "backend_options":{"physical_batch_size":32,"full_prefill_logits":False},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_branches=7,logical_operations=28,
            logical_checkpoint_replays=0,max_local_work_items=64,
            checkpoint_policy="fresh CPU model; seven fixed batch32 arms +24 three-token answers each; no fitting, children or retries"),
        "geometry":{"capture_forwards":0,"candidate_forwards":0,"heldout_arms":7,
            "fresh_rewrite_tasks":16,"position_controls":8,"general_prompts":8,"generation_rows":24,"new_tokens":3}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    save(root/"request.json",request)
    stage_worker(root/"payload",worker=sources[0],config=config,request=request,config_bytes=raw,
        additional_workers=sources[1:],additional_files=[(archive,archive.name),(root/"preflight.v1.json","preflight.v1.json")])
    command = worker_command(sources[0].name)
    command.insert(1,"TMPDIR=/mnt/hdd8tb/mdb/tmp")
    save(root/"command.json",command)
    return {"root":str(root),"request_sha256":request["request_sha256"],"frozen_program_sha256":sha(archive)}


def submit(root):
    from mdb_lanes.job_support import guarded_mrun_launch

    root = Path(root).resolve()
    if (root/"submission.json").exists():
        raise ValueError("already submitted")
    request = json.loads((root/"request.json").read_text())
    for name,expected in request["source_hashes"].items():
        if sha(root/"payload"/name)!=expected:
            raise ValueError("staged source differs")
    job = guarded_mrun_launch(json.loads((root/"command.json").read_text()),request=request,
        experiment="mdb-computation-program-replay",task="recorder",model=request["model"],
        payload=root/"payload",prefer_host="beast",gpu=False,needs={"cpu":True},timeout_s=720,
        retry_on_kill=False,preflight=True,
        note="Frozen semantic recipe replay, Qwen1.5B CPU FP32: seven batch32 observations +24 own-token triples each. No search/fit. Recompile factors and require exact frozen parameter hashes before release; fresh model, no children/retries. 600s science stop; original discovery hit its time budget; all partial evidence retained.")
    result = {"job_id":str(job),"request_sha256":request["request_sha256"]}
    save(root/"submission.json",result)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("stage","submit"))
    parser.add_argument("root",type=Path)
    parser.add_argument("--source",type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.root,args.source) if args.action=="stage" else submit(args.root),sort_keys=True))
