"""Submit one bounded joint1–4 writer localization assay from audited parents."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def stage(run_dir, predecessor):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-handoff-result"
    old, report, audit, inventory = (json.loads(p.read_text()) for p in
        (predecessor / "plan.json", predecessor / "report.json", predecessor / "analysis-v1.json", data / "files.json"))
    if (report["job_id"] != "job-6a6fcb685a20" or report["archive_sha256"] != "9ff1c29917cf6339d1d80bc70a8b429a36bda7f3d301297da97f2c6aa9de4f7d"
            or audit["mechanics_status"] != "valid" or audit["source"]["archive_sha256"] != report["archive_sha256"]):
        raise ValueError("unexpected or unaudited parent")
    frontiers = json.loads((data / "frontiers.json").read_text())
    if frontiers != audit["frontiers"]:
        raise ValueError("frontiers differ from audited source")
    cuts = {k: v["after_joint0"] for k, v in frontiers.items()}
    relatives = set()
    for cut in cuts.values():
        relative = f"typed-store/cuts/{cut}.json"
        record = json.loads((data / relative).read_text())
        if any(record["execution_point"].get(k) != v for k, v in {"logical_step": 1, "phase": "joint", "index": 1}.items()):
            raise ValueError("incorrect source clock")
        relatives.add(relative)
        relatives.update("typed-store/pages/" + h + ".json" for h in record["slots"].values())
        relatives.update("typed-store/blobs/" + h + ".safetensors" for h in record["physical_blobs"])
    extras = []
    for relative in sorted(relatives):
        if sha(data / relative) != inventory[relative]:
            raise ValueError("parent CAS changed")
        extras.append((data / relative, relative.replace("typed-store/", "input-store/", 1)))
    run_dir.mkdir(parents=True, exist_ok=False)
    selection = run_dir / "source-selection.json"
    save(selection, {"job_id": report["job_id"], "archive_sha256": report["archive_sha256"],
                     "audit_sha256": sha(predecessor / "analysis-v1.json"), "cuts": cuts})
    branches = [{"id": f"j{i}.{ports}", "blocks": [i], "ports": ports} for i in range(1, 5) for ports in ("A", "M", "AM")]
    branches += [{"id": "all." + ports, "blocks": [1, 2, 3, 4], "ports": ports} for ports in ("A", "M", "AM")]
    branches += [{"id": "sham", "blocks": [1, 2, 3, 4], "ports": "AM", "sham": True},
                 {"id": "reconstruct.base", "blocks": [1, 2, 3, 4], "ports": "AM", "initial_base_image": True}]
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "seed", "height", "width", "steps", "cpu_threads", "prompts")}
    config.update(schema="mdb-flux-joint-writes-plan-v1", input_cuts=cuts, branches=branches,
        max_images=19, max_adapter_steps=1800, max_wall_seconds=840,
        historical_pngs={"base": json.loads((data / "rows/base.json").read_text())["image"]["sha256"],
                         "changed": json.loads((data / "rows/after_joint4.restore_text.json").read_text())["image"]["sha256"]},
        hypotheses={
            "H-controls": "Base and changed reference PNGs exactly reproduce prior base and after_joint4.restore_text. Sham equals changed at all scheduler arrays and PNG. Initial base image plus all four base A/M outputs reconstructs base exactly.",
            "H-coalitions": "Tentative prediction: restoring all attention outputs moves final latent farther baseward than restoring all MLP outputs: p(all.A)<p(all.M). Mixed shapes and misses retained; p is not a shape score.",
            "H-shape": "Tentative prediction: all.A and all.AM lose the clean red sphere (cube-like or mixed), while all.M retains a sphere. Per-block outcomes exploratory.",
            "H-localization": "No categorical per-block forecast. Native suffix and visible geometry determine whether effects are local, distributed, redundant, or repaired.",
            "H-collateral": "Red-left, blue-right and blue cube expected to persist. Retain pose, size, spacing, shape and palette changes separately."},
        scope="Seed47 step1 joint1–4 image attention/MLP outputs. Matched observed afterjoint0 parents. Each branch clamps T to base afterjoint4 and C to base beforestep2. Native common suffix; counted direct block candidates followed by typed image Acts. Not raw-prompt or independent RGB prediction.",
        terminal_status="unassessed")
    save(run_dir / "plan.json", config)
    worker = here / "flux_joint_writes_campaign.py"
    extras += [(here / n, n) for n in ("flux_joint_writes.py", "flux_handoff_campaign.py", "flux_trajectory_forecast_campaign.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json"), (selection, selection.name)]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name,
        "flux_joint_writes.py", "flux_handoff_campaign.py", "flux_trajectory_forecast_campaign.py", "campaign_support.py",
        "plan.json", selection.name, "input-store", preflight.name],
        "path_token_allowlist": ["flux-joint-writes-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-joint-writes-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=19, logical_branches=19,
            logical_checkpoint_replays=19, max_local_work_items=19,
            checkpoint_policy="2 audited parents;8 native prefix steps/port comparisons;19 in-process RGB futures, at most50 direct candidate blocks, native suffix; no encoder/retries"),
        "resource_exception": {"ram_mb": 16000, "vram_mb": 13000, "calibration_job": report["job_id"],
            "measured_peak_rss_mb": 2651.2, "measured_peak_vram_mb": 9284, "native_calibration_job": "job-c2a467736b50",
            "native_measured_peak_rss_mb": 10697.6,
            "reason": "Same CUDA BF16 B1 512px4step resident model/VAE; bounded retained ports, no encoder or forecast cache. Preserve calibrated16GB RAM/13GB VRAM margin."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-joint-writes", payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=16000, vram_mb=13000, cpu_threads=4, est_wall_s=540, timeout_s=840, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 seed47. Joint1–4 shape writers,2refs+12 per-block A/M/AM+3coalitions+sham+reconstruct=19RGB. Tbase afterjoint4/Cbase beforestep2;<=1800transitions/50candidateblocks/840s. One load/noencoder/no retries. Preserve all mixtures/misses. Complementary to single4 assay.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--predecessor", type=Path)
    p.add_argument("--collect")
    a = p.parse_args()
    if a.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            a.run_dir, a.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=2 << 30)
    else:
        if a.predecessor is None:
            p.error("--predecessor required")
        result = stage(a.run_dir.resolve(), a.predecessor.resolve())
    print(json.dumps(result))
