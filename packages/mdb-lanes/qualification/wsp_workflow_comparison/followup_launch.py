"""Sealed finite CPU leases for storage diagnosis and native workflow studies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import secrets
import shutil
import zipfile
from pathlib import Path

from protocol import digest, save

HERE = Path(__file__).resolve().parent
LIMITS = {"max_archive_bytes": 1024 << 20, "max_expanded_bytes": 1536 << 20, "max_member_bytes": 64 << 20}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage(base, phase, *, config_extra=None, extra_files=None):
    from mdb_lanes.job_support import one_lease_scheduling_contract

    base.mkdir(parents=True, exist_ok=False)
    payload = base / "payload"
    payload.mkdir()
    native = phase not in {"storage", "audit", "recover"}
    config = {"schema": "mdb-workflow-followup-plan-v1", "phase": phase,
        "model_path": "/mnt/big/llm-models/Qwen2.5-0.5B-Instruct",
        "checkpoint_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "device": "cpu", "dtype": "float32" if native else "float64", "seed": secrets.randbits(31),
        "output_scratch": "/mnt/hdd8tb/mdb/workflow-followup", "volatile_scratch": "/dev/shm/mdb-workflow-followup",
        "scientific_gates": [], "retry": False, "custody_limits": LIMITS, **(config_extra or {})}
    files = {"worker.py": HERE / "followup_worker.py", "protocol.py": HERE / "protocol.py",
        "worker_common.py": HERE / "worker.py", "qwen_profile.py": HERE / "qwen_profile.py",
        "native_workflow.py": HERE / "native_workflow.py", "native_audit.py": HERE / "native_audit.py",
        "qwen_family_campaign.py": HERE.parent / "qwen_family_campaign.py",
        "retrieve_qwen_family_weights.py": HERE.parent / "retrieve_qwen_family_weights.py",
        "flux_family_custody.py": HERE.parent / "flux_family_custody.py", **(extra_files or {})}
    save(payload / "preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["model_path"], config["output_scratch"], config["volatile_scratch"], *config.get("saved_roots", {}).values(), *([config["recovery_root"]] if "recovery_root" in config else [])],
        "hashed_roots": [*files, "preflight.v1.json"]})
    files["preflight.v1.json"] = payload / "preflight.v1.json"
    helper = runpy.run_path(str(HERE.parent / "wsp_case_study/launch.py"))
    wheels = {p.name: p for p in helper["wheels"]()}
    request = {"schema": "mdb-workflow-followup-request-v1", "config": config,
        "execution_environment": {"python": "3.14.6", "uv_link_mode": "symlink", "uv_cache_dir": "/mnt/hdd8tb/mdb/uv-cache",
            "torch_home": "/mnt/hdd8tb/mdb/torch", "extra_model_roots": "/mnt/big/llm-models",
            "requirements": ["torch==2.13.0", "numpy==2.5.4", "pyyaml==6.0.3", "psutil==7.2.2", "safetensors==0.8.0", "transformers==5.14.1", "accelerate==1.14.0"]},
        "source_hashes": {n: sha(p) for n, p in files.items()}, "runtime_wheels": {n: sha(p) for n, p in wheels.items()},
        "model": "qwen2.5-0.5b-instruct" if native else None, "task_family": "recorder" if native else "analysis",
        "backend": "hf", "device": "cpu", "dtype": config["dtype"], "needs": {"cpu": True},
        "batch_size": 16, "seq_len": 128, "backend_options": {"physical_batch_size": 16, "full_prefill_logits": False},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1 if native else 0, model_free=not native,
            logical_branches=64, logical_operations=256, logical_checkpoint_replays=0, max_local_work_items=256,
            extra={} if native else {"task_family": "analysis"})}
    request["request_sha256"] = digest(request)
    save(base / "request.json", request)
    shutil.copyfile(__file__, base / "frozen-launcher.py")
    for name, path in {**files, **wheels}.items():
        if path.resolve() != (payload / name).resolve():
            shutil.copyfile(path, payload / name)
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    (payload / "saturn_run_config.json").write_bytes(raw)
    save(payload / "saturn_debug_request.json", request)
    packages = {}
    for name, path in wheels.items():
        package = "mdb" if name.startswith("mdb_") else "saturn_pub" if name.startswith("saturn_") else "mrun"
        with zipfile.ZipFile(path) as archive:
            packages[package] = {member[len(package)+1:]: hashlib.sha256(archive.read(member)).hexdigest()
                for member in archive.namelist() if member.startswith(package+"/") and not member.endswith("/")}
    save(payload / "payload-seal.json", {"packages": packages, "wheels": request["runtime_wheels"], "files": request["source_hashes"],
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "worker": "worker.py", "worker_sha256": request["source_hashes"]["worker.py"]})
    save(base / "command.json", helper["command"](request))
    return {"run": str(base), "request_sha256": request["request_sha256"]}


def submit(base):
    from mdb_lanes.job_support import guarded_mrun_launch
    if (base / "submission.json").exists():
        raise ValueError("finite follow-up already submitted")
    request = json.loads((base / "request.json").read_text())
    for n, h in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(base / "payload" / n) != h:
            raise ValueError("staged source differs")
    os.environ.setdefault("MRUN_URL", "http://192.168.68.68:9025")
    phase = request["config"]["phase"]
    job = guarded_mrun_launch(json.loads((base / "command.json").read_text()), request=request,
        experiment="mdb-workflow-followup-"+phase, task=request["task_family"], model=request["model"],
        payload=base / "payload", prefer_host="beast", pin="beast", gpu=False, needs={"cpu": True},
        timeout_s=request["config"].get("timeout_seconds", 840), retry_on_kill=False, preflight=True,
        note=request["config"].get("queue_note", "Workflow follow-up "+phase+": CPU, fixed batch16 prefixes; finite frozen plan, no training, retries or children."))
    result = {"job_id": str(job), "request_sha256": request["request_sha256"]}
    save(base / "submission.json", result)
    return result


def collect(base):
    job = json.loads((base / "submission.json").read_text())["job_id"]
    result = runpy.run_path(str(HERE.parent / "flux_family_custody.py"))["collect"](base, job, limits=LIMITS)
    if result.get("collection_status") == "not-terminal":
        return {k: result[k] for k in ("collection_status", "job_id", "state")}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit", "collect"))
    parser.add_argument("base", type=Path)
    parser.add_argument("--phase", choices=("storage", "calibrate"))
    args = parser.parse_args()
    result = stage(args.base.resolve(), args.phase) if args.action == "stage" else {"submit": submit, "collect": collect}[args.action](args.base.resolve())
    print(json.dumps(result, sort_keys=True))
