"""Stage and submit one finite CPU selective-weight-repair experiment."""
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

    for key, value in {"MDB_WORKER_UV_CACHE_DIR": "/mnt/hdd8tb/mdb/uv-cache",
        "MDB_WORKER_TORCH_HOME": "/mnt/hdd8tb/mdb/torch", "MDB_WORKER_MODEL_ROOTS": "",
        "MDB_WORKER_PYTHON": "3.14.6", "MDB_WORKER_UV_LINK_MODE": "symlink"}.items():
        os.environ[key] = value
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    sources = [here / name for name in ("wsp_selective_qwen.py", "wsp_auto_qwen.py",
        "qwen_family_campaign.py", "retrieve_qwen_family_weights.py")]
    config = {"schema": "mdb-selective-qwen-plan-v1", "model": "qwen2.5-0.5b-instruct",
        "model_path": "/mnt/big/llm-models/Qwen2.5-0.5B-Instruct",
        "checkpoint_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "device": "cpu", "dtype": "float32", "cpu_threads": 4,
        "site": "model.layers.6.mlp.down_proj", "heldout_seed": 20261112,
        "preserve_weight": 1., "ranks": [1, 4, 8], "fit_preserve_weights": [1., 100.],
        "doses": [.5, 1., 1.5], "ridge_relative": .001,
        "max_prompt_tokens": 128, "max_wall_seconds": 600,
        "output_scratch": "/mnt/hdd8tb/mdb/research", "retry": False,
        "predecessor": {"job_id": "job-745584d1b1ce",
            "evidence_archive_sha256": "5d19d2a9931c1f112acbe627af0e8878a1d8d3f7af33ed77c1d8ee0385a8a7d1",
            "reuse": "previously discovered site and suppression direction; both prior task splits now development"}}
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    preflight = {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["model_path"], config["output_scratch"]],
        "hashed_roots": [*(p.name for p in sources), "saturn_run_config.json", "preflight.v1.json"]}
    save(root / "preflight.v1.json", preflight)
    request = {"schema": "mdb-selective-qwen-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "execution_environment": intent(),
        "source_hashes": {p.name: sha(p) for p in [*sources, root / "preflight.v1.json"]},
        "model": config["model"], "task_family": "recorder", "backend": "hf", "device": "cpu",
        "dtype": "float32", "needs": {"cpu": True}, "batch_size": 32, "seq_len": 128,
        "backend_options": {"physical_batch_size": 32, "full_prefill_logits": False},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_branches=27,
            logical_operations=64, logical_checkpoint_replays=0, max_local_work_items=128,
            checkpoint_policy="one CPU resident; batch24 development and batch32 fresh evaluation; no children/retries"),
        "geometry": {"discovery_examples": 16, "preservation_examples": 8,
            "fresh_task_examples": 20, "fresh_preservation_examples": 12,
            "capture_forwards": 1, "probe_forwards": 20, "final_arms": 6,
            "generation_rows": 4, "new_tokens": 4}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(root / "request.json", request)
    stage_worker(root / "payload", worker=sources[0], config=config, request=request,
        config_bytes=raw, additional_workers=sources[1:],
        additional_files=[(root / "preflight.v1.json", "preflight.v1.json")])
    command = worker_command(sources[0].name)
    command.insert(1, "TMPDIR=/mnt/hdd8tb/mdb/tmp")
    save(root / "command.json", command)
    return {"root": str(root), "request_sha256": request["request_sha256"], "geometry": request["geometry"]}


def submit(root):
    from mdb_lanes.job_support import guarded_mrun_launch

    root = Path(root).resolve()
    if (root / "submission.json").exists():
        raise ValueError("finite request already submitted")
    request = json.loads((root / "request.json").read_text())
    for name, expected in request["source_hashes"].items():
        if sha(root / "payload" / name) != expected:
            raise ValueError("staged source changed")
    job = guarded_mrun_launch(json.loads((root / "command.json").read_text()), request=request,
        experiment="mdb-selective-weight-repair", task="recorder", model=request["model"],
        payload=root / "payload", prefer_host="beast", gpu=False, needs={"cpu": True},
        timeout_s=840, retry_on_kill=False, preflight=True,
        note="Selective Qwen0.5B CPU FP32 repair: one capture +20 batch24 probes, ranks1/4/8 and doses; six batch32 fresh task/locality arms, four own-state tokens x4 rows. One model, no gradients/children/retries; 600s scientific stop. Original weights recipe-bound; full final logits and near-compute archive retained.")
    result = {"job_id": str(job), "request_sha256": request["request_sha256"]}
    save(root / "submission.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.root) if args.action == "stage" else submit(args.root), sort_keys=True))
