"""Stage installed wheels and admit one bounded native-discovery lease."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mdb.native_discovery import canonical, file_hash, load_corpus, save_json


def stage(args):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    if not 1 <= args.wall_seconds <= 720:
        raise ValueError("this short-study launcher permits at most 720 science seconds")
    corpus = load_corpus(args.corpus)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    worker = Path(__file__).with_name("worker.py")
    config = {"schema": "mdb-native-discovery-plan-v1", "model": args.model,
        "model_path": args.model_path, "device": "cuda", "dtype": "float32", "cpu_threads": 4,
        "max_tokens": args.max_tokens, "stages": args.stages, "features": args.features,
        "steps": args.steps, "interval_steps": args.interval_steps, "seed": args.seed,
        "max_wall_seconds": args.wall_seconds, "output_scratch": "/mnt/hdd8tb/mdb/research",
        "corpus_fingerprint": corpus["fingerprint"], "retry": False}
    for key, value in {"MDB_WORKER_UV_CACHE_DIR": "/mnt/hdd8tb/mdb/uv-cache",
        "MDB_WORKER_TORCH_HOME": "/mnt/hdd8tb/mdb/torch", "MDB_WORKER_MODEL_ROOTS": "",
        "MDB_WORKER_PYTHON": "3.14.6", "MDB_WORKER_UV_LINK_MODE": "symlink"}.items():
        os.environ[key] = value
    save_json(root / "preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["model_path"], config["output_scratch"]],
        "hashed_roots": [worker.name, "corpus.json", "saturn_run_config.json", "preflight.v1.json"]})
    raw = canonical(config) + b"\n"
    request = {"schema": "mdb-native-discovery-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "execution_environment": intent(),
        "source_hashes": {worker.name: file_hash(worker), "corpus.json": file_hash(args.corpus),
                          "preflight.v1.json": file_hash(root / "preflight.v1.json")},
        "model": args.model, "task_family": "recorder", "backend": "hf", "device": "cuda",
        "dtype": "float32", "needs": {"cuda": True}, "batch_size": 1, "seq_len": args.max_tokens,
        "backend_options": {"physical_batch_size": 1, "full_prefill_logits": False},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1,
            logical_branches=3, logical_operations=len(corpus["documents"]) + 3 * args.steps,
            logical_checkpoint_replays=0, max_local_work_items=len(corpus["documents"]) + 3 * args.steps + 32,
            checkpoint_policy="one resident decoder; three reversible learned predictors; native held traces only after global forecast seal; no children or retries"),
        "geometry": {"documents": len(corpus["documents"]), "physical_batch": 1,
                     "maximum_tokens": args.max_tokens, "fitting_arms": 3, "updates_per_arm": args.steps,
                     "stages": args.stages, "features": args.features, "interventions": 0}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save_json(root / "request.json", request)
    stage_worker(root / "payload", worker=worker, config=config, request=request, config_bytes=raw,
        additional_files=[(args.corpus, "corpus.json"), (root / "preflight.v1.json", "preflight.v1.json")])
    command = worker_command(worker.name)
    command.insert(1, "TMPDIR=/mnt/hdd8tb/mdb/tmp")
    save_json(root / "command.json", command)
    return {"root": str(root), "request_sha256": request["request_sha256"], "geometry": request["geometry"]}


def submit(root):
    from mdb.job_support import guarded_mrun_launch

    root = root.resolve()
    if (root / "submission.json").exists():
        raise ValueError("already submitted; no automatic retry")
    request = json.loads((root / "request.json").read_text())
    for name, expected in request["source_hashes"].items():
        if file_hash(root / "payload" / name) != expected:
            raise ValueError("staged source changed: " + name)
    job = guarded_mrun_launch(json.loads((root / "command.json").read_text()), request=request,
        experiment="mdb-native-discovery", task="recorder", model=request["model"],
        payload=root / "payload", prefer_host="beast", gpu=True, needs={"cuda": True},
        timeout_s=request["config"]["max_wall_seconds"] + 120, retry_on_kill=False, preflight=True,
        note="Ordinary prose/code discovery; " + json.dumps(request["geometry"], sort_keys=True) +
             ". One eager FP32 CUDA decoder; three small predictors; public checkpoint rollback; heldout predictions sealed before native forward; science wall bound; no patching, children, or retries.")
    result = {"job_id": str(job), "request_sha256": request["request_sha256"]}
    save_json(root / "submission.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("stage")
    p.add_argument("root", type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--model", default="qwen2.5-1.5b")
    p.add_argument("--model-path", default="/mnt/big/llm-models/Qwen2.5-1.5B")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--interval-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--wall-seconds", type=int, default=600)
    p = sub.add_parser("submit")
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args) if args.action == "stage" else submit(args.root), sort_keys=True))
