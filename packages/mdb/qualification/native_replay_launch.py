"""Stage a bounded metadata-only native regeneration qualification."""
import argparse
import hashlib
import json
from pathlib import Path
from semantic_scaffold_launch import environment, save, sha


def stage(args):
    from mdb import parameter_artifacts
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    worker=Path(__file__).with_name("native_replay_worker.py")
    helper=args.parameter_helper.resolve()
    config={"model":"qwen2.5-1.5b","model_path":"/mnt/big/llm-models/Qwen2.5-1.5B",
        "expected_weight_file_sha256":"a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
        "output_root":"/mnt/big/mdb/qualification/native-replay-hydration-20261112","seed":20261111}
    files=[(helper,"parameter_source.py"),(Path(parameter_artifacts.__file__),"retrieve_weights.py"),
        (args.metadata/"recipe.json","recipe.json"),(args.metadata/"expectations.json","expectations.json")]
    preflight={"schema_version":"mrun-preflight-config-v1","hashed_roots":[worker.name,"preflight.v1.json",*[n for _,n in files]],
        "path_token_allowlist":[config["model_path"],config["output_root"]]}
    save(root/"preflight.v1.json",preflight)
    request={"schema":"mdb-native-regeneration-request-v1","model":config["model"],"worker":worker.name,
        "task_family":"forward","backend":"hf","device":"cuda","dtype":"bfloat16","batch_size":1,"seq_len":128,
        "backend_options":{"physical_batch_size":1,"full_prefill_logits":False},"needs":{"cuda":True},
        "execution_environment":environment(),"config":config,
        "source_hashes":{worker.name:sha(worker),**{n:sha(p) for p,n in files}},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_branches=8,logical_operations=128,
            logical_checkpoint_replays=8,max_local_work_items=256,
            checkpoint_policy="One resident; four token roots, 100 declared native recipe events, eight three-token historical branches, <=1400 native substeps; no activation roots, no stored-cut imports, no fitting/retries"),
        "workload_geometry":{"model_loads":1,"physical_batch":1,"max_prompt_tokens":128,"explicit_native_steps":1400,
            "source_discovery_transactions":0,"output_archive_max_bytes":8<<20,"output_expanded_max_bytes":16<<20},
        "resource_basis":"Intent-first; prior BF16 native Session measured 3446MB VRAM. Retain small replay cuts transiently; no guessed reservation.",
        "timeout_s":600,"queue_note":"Qualify metadata hydration: one Qwen1.5B BF16/eager CUDA resident; reconstruct 100 native closures from four token roots and ordered native/Act recipes across anchor and inserted-note contexts, eight own-token futures. Compare hashes only; no activation/cut files supplied, no fitting, <=1400 substeps. Stop at report or 600s."}
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    save(root/"request.json",request)
    stage_worker(root/"payload",worker=worker,config=config,request=request,
        additional_files=[*files,(root/"preflight.v1.json","preflight.v1.json")])
    print(json.dumps({"staged":str(root),"request_sha256":request["request_sha256"]}))


def submit(root):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    if (root/"submission.json").exists():raise ValueError("already submitted")
    request=json.loads((root/"request.json").read_text())
    environment()
    job=guarded_mrun_launch(worker_command(request["worker"]),request=request,model=request["model"],task="forward",
        experiment="mdb-native-replay-hydration-20261112",payload=root/"payload",prefer_host="beast",gpu=True,
        needs=request["needs"],timeout_s=600,retry_on_kill=False,preflight=True,detach=True,note=request["queue_note"])
    save(root/"submission.json",{"job_id":str(job),"request_sha256":request["request_sha256"]})
    print(json.dumps({"job_id":str(job)}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("stage","submit"));p.add_argument("root",type=Path)
    p.add_argument("--metadata",type=Path);p.add_argument("--parameter-helper",type=Path)
    a=p.parse_args()
    if a.action=="stage":stage(a)
    else:submit(a.root.resolve())
