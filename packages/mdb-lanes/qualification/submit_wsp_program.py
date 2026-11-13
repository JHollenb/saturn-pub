"""One finite mechanism-guided semantic-edit lease, public installed packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from wsp_auto_qwen import save, sha


def stage(root):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    for key,value in {"MDB_WORKER_UV_CACHE_DIR":"/mnt/hdd8tb/mdb/uv-cache",
        "MDB_WORKER_TORCH_HOME":"/mnt/hdd8tb/mdb/torch", "MDB_WORKER_MODEL_ROOTS":"",
        "MDB_WORKER_PYTHON":"3.14.6", "MDB_WORKER_UV_LINK_MODE":"symlink"}.items():
        os.environ[key] = value
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    here = Path(__file__).resolve().parent
    workspace = here.parents[3]
    # Maintainer extraction is explicit; workers never import sibling projects.
    evidence = [(workspace / relative,name) for relative,name in (
        ("saturn/experiments/2026-10-30-dynamic-symbol-table/FINDINGS.md","reader-roles.md"),
        ("saturn/experiments/2026-11-10-operand-row-compiler/FINDINGS.md","operand-rows.md"),
        ("cmachine-dictionary/research/dictionary/FINDINGS.md","dictionary-evidence.md"))]
    config = {"schema":"mdb-computation-program-plan-v1", "model":"qwen2.5-1.5b-instruct",
        "model_path":"/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "checkpoint_sha256":"dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
        "execution":"hf-cpu-fp32-eager", "device":"cpu", "dtype":"float32", "cpu_threads":4,
        "program":{"operand":"a","column":0,"source":7,"target":3},
        "discovery_seed":20261120,"heldout_seed":20261122,"max_wall_seconds":720,
        "output_scratch":"/mnt/hdd8tb/mdb/research","retry":False}
    account = {"schema":"mdb-supplied-computation-account-v1",
        "checkpoint_sha256":config["checkpoint_sha256"],"execution":config["execution"],
        "prompt_prefix":"12+35=47\n481+307=788\n256+139=395\n604+218=822\n",
        "symbols":{"operand.a.column0":{"token_position":47,
            "consumer":"late operand value reads, especially layer22 own-column readers, followed by layer23 combination",
            "writers":[{"module":"model.layers.21.mlp.down_proj","target":"residual","residual_module":"model.layers.21",
                "reason":"additive write immediately before layer22 read; residual digit translations already causally tested"},
                {"module":"model.layers.22.self_attn.v_proj","target":"value",
                 "reason":"edit content delivered to known readers while leaving this layer key parameters unchanged"}]}},
        "wrong_location":51,
        "evidence":[{"path":name,"sha256":sha(path)} for path,name in evidence],
        "limitations":["authored role binding from historical experiments, not automated semantic discovery",
            "role-map and dictionary panel prompts/backends differ; applicability is tested here",
            "first experiment restricts matched cases to no carry change"]}
    save(root/"computation-account.json",account)
    sources = [here/name for name in ("wsp_program_qwen.py","wsp_auto_qwen.py","qwen_family_campaign.py","retrieve_qwen_family_weights.py")]
    raw = (json.dumps(config,sort_keys=True)+"\n").encode()
    additional = [(root/"computation-account.json","computation-account.json"), *evidence]
    preflight = {"schema_version":"mrun-preflight-config-v1",
        "path_token_allowlist":[config["model_path"],config["output_scratch"]],
        "hashed_roots":[*(p.name for p in sources),*(name for _,name in additional),"saturn_run_config.json","preflight.v1.json"]}
    save(root/"preflight.v1.json",preflight)
    request = {"schema":"mdb-computation-program-request-v1","config":config,
        "config_bytes_sha256":hashlib.sha256(raw).hexdigest(),"execution_environment":intent(),
        "source_hashes":{**{p.name:sha(p) for p in sources},**{name:sha(p) for p,name in additional},
                         "preflight.v1.json":sha(root/"preflight.v1.json")},
        "model":config["model"],"task_family":"recorder","backend":"hf","device":"cpu","dtype":"float32",
        "needs":{"cpu":True},"batch_size":48,"seq_len":64,
        "backend_options":{"physical_batch_size":48,"full_prefill_logits":False},
        "scheduling_contract":one_lease_scheduling_contract(model_loads=1,logical_branches=25,
            logical_operations=64,logical_checkpoint_replays=0,max_local_work_items=128,
            checkpoint_policy="one CPU model; capture48, causal/search32, fresh32 +generation24; no children or retries"),
        "geometry":{"capture_forwards":1,"causal_forwards":5,"candidate_forwards":12,
            "heldout_arms":7,"fresh_rewrite_tasks":16,"fresh_position_controls":8,"fresh_general_prompts":8,
            "generation_rows":24,"new_tokens":3,"science_wall_seconds":720}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    save(root/"request.json",request)
    stage_worker(root/"payload",worker=sources[0],config=config,request=request,config_bytes=raw,
        additional_workers=sources[1:],additional_files=[*additional,(root/"preflight.v1.json","preflight.v1.json")])
    command = worker_command(sources[0].name)
    command.insert(1,"TMPDIR=/mnt/hdd8tb/mdb/tmp")
    save(root/"command.json",command)
    return {"root":str(root),"request_sha256":request["request_sha256"],"geometry":request["geometry"]}


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
        experiment="mdb-computation-program",task="recorder",model=request["model"],
        payload=root/"payload",prefer_host="beast",gpu=False,needs={"cpu":True},
        timeout_s=840,retry_on_kill=False,preflight=True,
        note="Evidence-guided Qwen1.5B semantic operand rewrite 7->3: one capture48, five causal probes, six semantic/six norm-matched random fits, seven fresh32 arms +24 own-token triples. Role map resolves two writers; no layer scan. CPU FP32, one model, 720s science stop, no retries/children. Source evidence and weights recipe-bound.")
    result = {"job_id":str(job),"request_sha256":request["request_sha256"]}
    save(root/"submission.json",result)
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("stage","submit"))
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.root) if args.action=="stage" else submit(args.root),sort_keys=True))
