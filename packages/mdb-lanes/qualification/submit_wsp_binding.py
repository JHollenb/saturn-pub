"""Finite role-binding discovery and frozen evaluation with installed public wheels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

from wsp_binding_protocol import save, sha

HERE = Path(__file__).resolve().parent


def runtime_identity():
    from mdb_lanes.payload import runtime_distributions, seal_package

    return {package: hashlib.sha256(json.dumps(seal_package(package), sort_keys=True).encode()).hexdigest()
        for package, _, _ in runtime_distributions(require_installed=True)}


def stage(root, source, *, phase, development=None):
    import numpy as np

    from mdb_lanes.execution_environment import intent
    from mdb_lanes.forecast_transaction import load_forecast_tree
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    for key, value in {"MDB_WORKER_UV_CACHE_DIR": "/mnt/hdd8tb/mdb/uv-cache",
        "MDB_WORKER_TORCH_HOME": "/mnt/hdd8tb/mdb/torch", "MDB_WORKER_MODEL_ROOTS": "",
        "MDB_WORKER_PYTHON": "3.14.6", "MDB_WORKER_UV_LINK_MODE": "symlink"}.items():
        os.environ[key] = value
    root.mkdir(parents=True, exist_ok=False)
    read = lambda name: json.loads((source/name).read_text())
    inventory = read("files.json")
    needed = ("config.json", "inputs.json", "selection.json", "computation-account.json",
        "parameter-artifacts.json", "retrieve_weights.py", "capture/observations/contrasts.json", "semantic-patch/patch.json")
    for name in needed:
        if sha(source/name) != inventory[name]["sha256"]:
            raise ValueError("upstream source inventory differs")
    selection, inputs, account = (read(n) for n in ("selection.json", "inputs.json", "computation-account.json"))
    selected = selection["selected"]["semantic"]
    factor_name = "factors/"+selected["key"]+".npz"
    if sha(source/factor_name) != inventory[factor_name]["sha256"]:
        raise ValueError("upstream factor differs")
    # load_forecast_tree verifies the referenced tensor bytes, not just JSON.
    capture = load_forecast_tree(source/"capture", read("capture/observations/contrasts.json")["value"])
    sample = capture["result"]["observations"]["samples"][selection["site"]]
    with np.load(source/factor_name, allow_pickle=False) as values:
        np.savez(root/"upstream.npz", left=values["left"], right=values["right"],
            direction=values["desired"][0], positive=sample["hidden"], preservation=sample["preservation"])
    old_controls = []
    for row in inputs["heldout_controls"]:
        if (row["a"], row["b"]) in {(120, 127), (308, 297), (471, 371)}:
            old_controls.append({**row, "role": "historical-control", "digit": 7, "block": -1,
                "expected": row["original"], "matched": False})
    if len(old_controls) != 3:
        raise ValueError("known diagnostic failures missing")
    upstream = {"source_root": str(source), "source_inventory_sha256": sha(source/"files.json"),
        "source_summary_sha256": sha(source/"summary.json"), "site": selection["site"], "selected": selected,
        "prefix": account["prompt_prefix"], "diagnostic_controls": old_controls,
        "old_patch_manifest": read("semantic-patch/patch.json")["manifest"],
        "exclude_a": sorted({r["a"] for k in ("development", "development_controls", "heldout", "heldout_controls") for r in inputs[k]}),
        "source_identities": {n: sha(source/n) for n in (*needed, factor_name)},
        "direction_policy": "retain original selected semantic response; no new donor response estimation"}
    save(root/"upstream.json", upstream)
    files = [(root/n, n) for n in ("upstream.json", "upstream.npz")]
    files += [(source/n, n) for n in ("parameter-artifacts.json", "retrieve_weights.py")]
    config = {**read("config.json"), "schema": "mdb-role-selector-plan-v1", "phase": phase,
        "development_seed": 202611111, "heldout_seed": 202611112,
        "development_blocks": 4, "heldout_blocks": 8, "max_wall_seconds": 720,
        "retry": False, "scientific_gates": [],
        "batch_geometry_evidence": {"job_id": "job-f9ca124e108d", "physical_batch_size": 32,
            "backend": "hf-cpu-fp32-eager", "peak_rss_mb": 8853.7,
            "policy_override": "GATHER_MAX_BATCH=32, using the prior measured batch32 execution profile"},
        "source_inventory_sha256": upstream["source_inventory_sha256"]}
    for name in ("replay_policy", "frozen_program_sha256", "source_job_id"):
        config.pop(name, None)
    if phase == "evaluate":
        if development is None:
            raise ValueError("completed discovery source required")
        dev_manifest = json.loads((development/"files.json").read_text())
        for name, target in (("selection.json", "frozen-selection.json"), ("selected.npz", "selected.npz"),
            ("selected-patch-manifest.json", "selected-patch-manifest.json")):
            if sha(development/name) != dev_manifest[name]:
                raise ValueError("frozen discovery bytes differ")
            files.append((development/name, target))
        config.update(selection_sha256=sha(development/"selection.json"),
            factors_sha256=sha(development/"selected.npz"), source_discovery=str(development),
            discovery_inventory_sha256=sha(development/"files.json"))
    sources = [HERE/n for n in ("wsp_binding_qwen.py", "wsp_binding_protocol.py", "flux_family_custody.py")]
    preflight = {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["model_path"], config["output_scratch"]],
        "hashed_roots": [p.name for p in sources]+[n for _, n in files]+["saturn_run_config.json", "preflight.v1.json"]}
    save(root/"preflight.v1.json", preflight)
    files.append((root/"preflight.v1.json", "preflight.v1.json"))
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    geometry = {"diagnostic_arms": 4, "candidate_scoring_arms": 6, "rows": 35, "generation_arms": 4} if phase == "discover" else {
        "evaluation_arms": 5, "rows": 64, "generation_arms": 5}
    request = {"schema": "mdb-role-selector-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "execution_environment": intent(),
        "runtime_package_sha256": runtime_identity(),
        "source_hashes": {**{p.name: sha(p) for p in sources}, **{n: sha(p) for p, n in files}},
        "model": config["model"], "task_family": "recorder", "backend": "hf", "device": "cpu", "dtype": "float32",
        "needs": {"cpu": True}, "batch_size": 32, "seq_len": 64,
        "backend_options": {"physical_batch_size": 32, "full_prefill_logits": False},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_branches=10 if phase == "discover" else 5,
            logical_operations=32, logical_checkpoint_replays=0, max_local_work_items=128,
            checkpoint_policy="one resident CPU FP32 model; restart each arm; own-token generation; no children/retries"),
        "geometry": {**geometry, "new_tokens": 3, "source_positions": [46, 47, 50, 51],
            "reader_boundaries": ["model.layers.22.self_attn", "model.layers.23"]}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(root/"request.json", request)
    stage_worker(root/"payload", worker=sources[0], config=config, request=request, config_bytes=raw,
        additional_workers=sources[1:], additional_files=files)
    command = worker_command(sources[0].name)
    command.insert(1, "TMPDIR=/mnt/hdd8tb/mdb/tmp")
    save(root/"command.json", command)
    shutil.copyfile(__file__, root/"frozen-launcher.py")
    return {"root": str(root), "request_sha256": request["request_sha256"], "geometry": request["geometry"]}


def submit(root):
    from mdb_lanes.job_support import guarded_mrun_launch
    if (root/"submission.json").exists():
        raise ValueError("finite lease already submitted")
    request = json.loads((root/"request.json").read_text())
    # The policy's CPU default is 16. This declared workload reuses the prior
    # measured batch32 profile; let policy size that actual geometry instead
    # of silently falling back to an undersized probe-default reservation.
    if "physical_batch_size" in request["backend_options"]:
        os.environ["GATHER_MAX_BATCH"] = str(request["backend_options"]["physical_batch_size"])
    for name, digest in request["source_hashes"].items():
        if sha(root/"payload"/name) != digest:
            raise ValueError("staged source differs")
    job = guarded_mrun_launch(json.loads((root/"command.json").read_text()), request=request,
        experiment="mdb-role-selector-"+request["config"]["phase"], task=request["task_family"], model=request["model"],
        payload=root/"payload", prefer_host="beast", gpu=False, needs={"cpu": True},
        timeout_s=300 if request["config"]["phase"] == "audit" else 840, retry_on_kill=False, preflight=True,
        note="Saved-only role-binding reader: verify pinned result trees, full-vocabulary reductions, selector traces, generation and restoration. No model loads or forwards; one finite CPU lease, no retries/children." if request["config"]["phase"] == "audit" else "Qwen1.5B role binding "+request["config"]["phase"]+
            ": frozen 7->3 response, physical batches <=32. Discovery: 4 diagnostic generation arms on 35 rows, 6 candidate scoring arms. Evaluation: 5 generation arms on 64 fresh rows. CPU FP32 eager, 4 threads; trace selector and L22/L23 consumers. 720s science stop, no retries/children. Historical failures are development diagnostics.")
    result = {"job_id": str(job), "request_sha256": request["request_sha256"]}
    save(root/"submission.json", result)
    return result


def collect(root):
    job = json.loads((root/"submission.json").read_text())["job_id"]
    return runpy.run_path(str(HERE/"flux_family_custody.py"))["collect"](root, job)


def stage_audit(root, development, evaluation, replication=None):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    for key, value in {"MDB_WORKER_UV_CACHE_DIR": "/mnt/hdd8tb/mdb/uv-cache",
        "MDB_WORKER_TORCH_HOME": "/mnt/hdd8tb/mdb/torch", "MDB_WORKER_MODEL_ROOTS": "",
        "MDB_WORKER_PYTHON": "3.14.6", "MDB_WORKER_UV_LINK_MODE": "symlink"}.items():
        os.environ[key] = value
    root.mkdir(parents=True, exist_ok=False)
    studies = {}
    collections = [("discovery", development), ("evaluation", evaluation)]
    if replication is not None:
        collections.append(("initial-evaluation", replication))
    for name, collection in collections:
        report = json.loads((collection/"report.json").read_text())
        job = json.loads((collection/"job.json").read_text())
        if job["state"] != "succeeded" or report["job_id"] != job["job_id"]:
            raise ValueError("completed collected source required")
        source = collection/"extracted"/report["archive_root"]
        studies[name] = {"root": "/mnt/hdd8tb/mdb/research/"+report["archive_root"],
            "files_manifest_sha256": sha(source/"files.json"), "summary_sha256": sha(source/"summary.json")}
    config = {"schema": "mdb-role-selector-reader-plan-v1", "phase": "audit", "studies": studies,
        "output_scratch": "/mnt/hdd8tb/mdb/research", "retry": False, "scientific_gates": []}
    sources = [HERE/n for n in ("wsp_binding_audit_worker.py", "audit_wsp_binding.py", "wsp_binding_protocol.py", "flux_family_custody.py")]
    save(root/"preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [config["output_scratch"], *(v["root"] for v in studies.values())],
        "hashed_roots": [p.name for p in sources]+["saturn_run_config.json", "preflight.v1.json"]})
    files = [(root/"preflight.v1.json", "preflight.v1.json")]
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    request = {"schema": "mdb-role-selector-reader-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "execution_environment": intent(),
        "runtime_package_sha256": runtime_identity(),
        "source_hashes": {**{p.name: sha(p) for p in sources}, "preflight.v1.json": sha(root/"preflight.v1.json")},
        "model": None, "task_family": "analysis", "needs": {"cpu": True},
        "backend_options": {},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True,
            logical_branches=len(studies), logical_operations=len(studies), logical_checkpoint_replays=0, max_local_work_items=4,
            extra={"task_family": "analysis"})}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    save(root/"request.json", request)
    stage_worker(root/"payload", worker=sources[0], config=config, request=request, config_bytes=raw,
        additional_workers=sources[1:], additional_files=files)
    command = worker_command(sources[0].name)
    command.insert(1, "TMPDIR=/mnt/hdd8tb/mdb/tmp")
    save(root/"command.json", command)
    return {"root": str(root), "request_sha256": request["request_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit", "collect", "stage-audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--phase", choices=("discover", "evaluate"), default="discover")
    parser.add_argument("--development", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--replication", type=Path)
    args = parser.parse_args()
    result = stage_audit(args.root.resolve(), args.development.resolve(), args.evaluation.resolve(),
        replication=None if args.replication is None else args.replication.resolve()) if args.action == "stage-audit" else stage(args.root.resolve(), args.source.resolve(), phase=args.phase,
        development=None if args.development is None else args.development.resolve()) if args.action == "stage" else {
        "submit": submit, "collect": collect}[args.action](args.root.resolve())
    print(json.dumps(result, sort_keys=True))
