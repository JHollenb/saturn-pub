"""Stage and submit one CPU, native-Qwen automatic edit-discovery lease."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(root):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    # The submitted worker runs on the fleet, not in the staging host's home.
    for key, value in {"MDB_WORKER_UV_CACHE_DIR": "/mnt/hdd8tb/mdb/uv-cache",
        "MDB_WORKER_TORCH_HOME": "/mnt/hdd8tb/mdb/torch", "MDB_WORKER_MODEL_ROOTS": "",
        "MDB_WORKER_PYTHON": "3.14.6", "MDB_WORKER_UV_LINK_MODE": "symlink"}.items():
        os.environ[key] = value
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    sources = [here / name for name in ("wsp_auto_qwen.py", "qwen_family_campaign.py", "retrieve_qwen_family_weights.py")]
    config = {"schema": "mdb-auto-affine-qwen-plan-v1", "model": "qwen2.5-0.5b-instruct",
        "model_path": "/mnt/big/llm-models/Qwen2.5-0.5B-Instruct",
        "checkpoint_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "device": "cpu", "dtype": "float32", "cpu_threads": 4, "max_sites": 12,
        "discovery_seed": 20261110, "heldout_seed": 20261111, "preserve_weight": 1.,
        "max_prompt_tokens": 128, "max_wall_seconds": 600,
        "output_scratch": "/mnt/hdd8tb/mdb/research", "retry": False}
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    preflight = {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["model_path"], config["output_scratch"]],
        "hashed_roots": [*(p.name for p in sources), "saturn_run_config.json", "preflight.v1.json"]}
    save(root / "preflight.v1.json", preflight)
    request = {"schema": "mdb-auto-affine-qwen-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "execution_environment": intent(),
        "source_hashes": {p.name: sha(p) for p in [*sources, root / "preflight.v1.json"]},
        "model": config["model"], "task_family": "recorder", "backend": "hf", "device": "cpu",
        "dtype": "float32", "needs": {"cpu": True}, "batch_size": 16, "seq_len": 128,
        "backend_options": {"physical_batch_size": 16, "full_prefill_logits": False},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_branches=32,
            logical_operations=64, logical_checkpoint_replays=0, max_local_work_items=128,
            checkpoint_policy="one CPU resident; batch16 fresh-prefix discovery; final own-token KV; no children or retries"),
        "geometry": {"discovery_examples": 8, "preservation_examples": 8, "heldout_examples": 8,
                     "max_probe_candidates": 23, "final_arms": 4, "generation_rows": 4, "new_tokens": 4}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(root / "request.json", request)
    stage_worker(root / "payload", worker=sources[0], config=config, request=request,
        config_bytes=raw, additional_workers=sources[1:],
        additional_files=[(root / "preflight.v1.json", "preflight.v1.json")])
    command = worker_command(sources[0].name)
    # Explicitly cleared paths and package wheels are already part of the public
    # worker command. Prefer storage outside /mnt/big for transient work.
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
    command = json.loads((root / "command.json").read_text())
    job = guarded_mrun_launch(command, request=request, experiment="mdb-automatic-affine-discovery",
        task="recorder", model=request["model"], payload=root / "payload", prefer_host="beast",
        gpu=False, needs={"cpu": True}, timeout_s=840,
        retry_on_kill=False, preflight=True,
        note="Automatic affine-edit discovery, Qwen0.5B CPU FP32 eager: max23 batch16 fresh-prefix probes, 12 automatically sampled affine sites, dose/channel refinement; four sealed heldout arms and four own-token choices. One resident model, no training/gradients/retries/children; 600s scientific stop, original weights recipe-bound, near-compute evidence archive with pinned publication.")
    result = {"job_id": str(job), "request_sha256": request["request_sha256"]}
    save(root / "submission.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.root) if args.action == "stage" else submit(args.root), sort_keys=True))
