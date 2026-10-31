"""Admit eleven native futures from a matched pre-single-block cut pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def stage(run_dir, predecessor, block, selection):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    if type(block) is not int or not 0 <= block < 10:
        raise ValueError("selected block must lie in localized single0–9 window")
    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-handoff-result"
    old_plan, report, audit = (json.loads(path.read_text()) for path in
        (data / "plan.json", predecessor / "report.json", predecessor / "analysis-v1.json"))
    if audit["source"]["job_id"] != report["job_id"] or audit["source"]["archive_sha256"] != report["archive_sha256"]:
        raise ValueError("input audit/report mismatch")
    frontiers = json.loads((data / "frontiers.json").read_text())
    if frontiers != audit["frontiers"]:
        raise ValueError("frontiers differ from audited histories")
    key = "after_joint4" if block == 0 else f"after_single{block - 1}"
    cut_ids = {name: frontiers[name][key] for name in ("base", "changed")}
    manifest = json.loads((data / "files.json").read_text())
    relatives = set()
    for cut_id in cut_ids.values():
        relative = f"typed-store/cuts/{cut_id}.json"
        relatives.add(relative)
        cut = json.loads((data / relative).read_text())
        if cut["execution_point"]["logical_step"] != 0 or cut["execution_point"]["phase"] != "single" or cut["execution_point"]["index"] != block:
            raise ValueError("source cut is not before the selected block")
        relatives.update(f"typed-store/pages/{page}.json" for page in cut["slots"].values())
        relatives.update(f"typed-store/blobs/{blob}.safetensors" for blob in cut["physical_blobs"])
    extras = []
    for relative in sorted(relatives):
        path = data / relative
        if sha(path) != manifest[relative]:
            raise ValueError("input CAS changed")
        extras.append((path, relative.replace("typed-store/", "input-store/", 1)))
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: old_plan[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "seed", "height", "width", "steps", "cpu_threads")}
    selection_data = json.loads(selection.read_text())
    config.update(schema="mdb-flux-merge-plan-v1", block=block, selector={"logical_step": 0, "phase": "single", "index": block},
        input_cuts=cut_ids, predecessor={"job_id": report["job_id"], "archive_sha256": report["archive_sha256"]},
        selection={"job_id": selection_data["source"]["job_id"], "audit_sha256": sha(selection)},
        max_images=11, max_adapter_steps=1500, max_wall_seconds=660,
        hypotheses={"H-attention": "Replacing image-row attention operands with base values moves the final latent toward base more than replacing MLP operands alone (projection010 < projection001).",
            "H-ordering": {"merge.001": "changed", "merge.010": "base", "merge.110": "base"},
            "H-residual": "The incoming residual also carries prior scene information; it need not be erased by one changed branch operand.",
            "H-controls": "Native copy sham and self merge000 reproduce changed exactly; merge111 reconstructs native base image state at the block exit, while keeping changed text."},
        scope="one selected block, known seed31 relation histories; image-row residual/attention/MLP factorial with fixed recipient post-block text; six new mixed operand combinations, observed native ports and shared projection kernels; not independent image forecasting",
        terminal_status="unassessed")
    save(run_dir / "plan.json", config)
    worker = here / "flux_merge_campaign.py"
    extras += [(here / "flux_parallel_merge.py", "flux_parallel_merge.py"),
        (here / "flux_trajectory_forecast_campaign.py", "flux_trajectory_forecast_campaign.py"),
        (here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json"), (selection, "selection-audit.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, "flux_parallel_merge.py",
        "flux_trajectory_forecast_campaign.py", "campaign_support.py", "plan.json", "preflight.v1.json", "selection-audit.json", "input-store"],
        "path_token_allowlist": ["flux-merge-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-merge-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=11, logical_branches=11,
            logical_checkpoint_replays=11, max_local_work_items=16, checkpoint_policy="two observed selected blocks; eight image-row R/A/M combinations, copy sham and two native refs; no retries"),
        "resource_exception": {"ram_mb": 32000, "vram_mb": 13000, "calibration_job": "job-c2a467736b50", "measured_peak_vram_mb": 9470.,
            "measured_peak_rss_mb": 10697.6, "measured_elapsed_seconds": 736.55,
            "reason": "same resident CUDA BF16 scalar512px model,11RGB plus bounded merge ports; original24RGB measured below this reservation"}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward", experiment="mdb-flux-parallel-merge",
        payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=32000, vram_mb=13000,
        cpu_threads=4, est_wall_s=480, timeout_s=660, retry_on_kill=False, preflight=True,
        note=f"Klein4B CUDA BF16 B1 512px4steps seed31 single{block}. Reuse matched pre-block cuts; observe2native blocks/actual merge ports.11RGB:2refs+copy sham+8R/A/M image-row combinations; text remains recipient. Original full projection shape/kernels, no additive split approximation. <=1500native steps,660s; one load,no retries.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--block", type=int)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, max_compressed_bytes=256 << 20, max_expanded_bytes=512 << 20, log_storage="gzip")
    else:
        if args.predecessor is None or args.selection is None or args.block is None:
            parser.error("--predecessor, --selection and --block required")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve(), args.block, args.selection.resolve())
    print(json.dumps(result))
