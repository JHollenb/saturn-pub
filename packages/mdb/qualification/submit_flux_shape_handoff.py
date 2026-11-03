"""Reuse equal-latent step1 parents to localize a shape-sensitive handoff."""
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
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def stage(run_dir, predecessor):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    run_dir.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-shape-timing-result"
    old, report, inventory = (json.loads(p.read_text()) for p in
        (data / "plan.json", predecessor / "report.json", data / "files.json"))
    if report["job_id"] != "job-61e7213615c9" or report["archive_sha256"] != "160e8fb54539f373fc3bf307788ea2f030aa67e4a8123683a9153b977d24ea61":
        raise ValueError("unexpected predecessor")
    cuts = {"base": "439f103395524ee17433cb3ae312f0dfabeb1918e686602c2e2cd4c65594538b",
            "changed": "265cd485c81e6a36ba7769a52c8b5640ed64710c49bbe1606c0d428dc5adff8a"}
    relatives, selected_receipts = set(), {}
    for name, cut_id in cuts.items():
        relative = f"typed-store/cuts/{cut_id}.json"
        record = json.loads((data / relative).read_text())
        if record["execution_point"]["phase"] != "project" or record["execution_point"]["logical_step"] != 1:
            raise ValueError("wrong source clock")
        relatives.add(relative)
        relatives.update("typed-store/pages/" + h + ".json" for h in record["slots"].values())
        relatives.update("typed-store/blobs/" + h + ".safetensors" for h in record["physical_blobs"])
        matches = []
        for path in (data / "typed-store/receipts").glob("*.json"):
            r = json.loads(path.read_text())
            if r["result"] == cut_id and r["operation"] == "continue":
                if sha(path) != inventory[path.relative_to(data).as_posix()] or r["transitions"][0]["before"]["phase"] != "scheduler":
                    raise ValueError("source writer differs")
                matches.append(r)
        if len(matches) != 1:
            raise ValueError("ambiguous source writer")
        selected_receipts[name] = matches[0]
    extras = []
    for relative in sorted(relatives):
        if sha(data / relative) != inventory[relative]:
            raise ValueError("source CAS differs")
        extras.append((data / relative, relative.replace("typed-store/", "input-store/", 1)))
    selection = run_dir / "source-selection.json"
    save(selection, {"predecessor_job": report["job_id"], "archive_sha256": report["archive_sha256"],
        "cuts": cuts, "scheduler_receipts": selected_receipts,
        "history": "Both histories consumed shape working text at0 and have equal Z1; only persistent conditioning differs at step1."})
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "seed", "height", "width", "steps", "guidance_scale", "cpu_threads", "prompts")}
    config.update(schema="mdb-flux-shape-handoff-plan-v1", panel="shape-step1", start_step=1, reset_step=2,
        input_cuts=cuts, max_images=12, max_adapter_steps=1100, max_wall_seconds=840,
        predecessor={"job_id": report["job_id"], "archive_sha256": report["archive_sha256"], "selection_sha256": sha(selection)},
        historical_pngs={"base": report["rows"]["early"]["image"]["sha256"],
                         "changed": report["rows"]["coherent.switch2"]["image"]["sha256"]},
        boundaries=[{"id": "after_joint0", "selector": {"logical_step": 1, "phase": "joint", "index": 1}},
            {"id": "after_joint4", "selector": {"logical_step": 1, "phase": "single", "index": 0}},
            {"id": "after_single4", "selector": {"logical_step": 1, "phase": "single", "index": 5}},
            {"id": "after_single18", "selector": {"logical_step": 1, "phase": "single", "index": 19}}],
        hypotheses={"H-early-text": "After joint0, restoring base text removes the red sphere outcome; restoring base image retains the sphere outcome. This is a prospective guess, not a gate.",
            "H-late-image": "After single18, restoring base text retains the red sphere; restoring base image removes it. A mixed object is recorded separately.",
            "H-midpoints": "Joint4 and single4 are exploratory localization points; no categorical shape prediction is assigned.",
            "H-controls": "Late sham exactly reproduces changed and late both-carrier restoration exactly reproduces base in scheduler arrays and PNG bytes under the common reset at2.",
            "H-replay": "Base exactly reproduces prior early PNG; changed exactly reproduces prior coherent.switch2 PNG. Projected image, context, positions and latent match before joint0 at step1.",
            "H-collateral": "Red-left/blue-right ordering and blue cube-like shape should persist; retain violations, mixed shapes, pose and scale changes."},
        scope="seed47 step1 shape contrast from equal-latent native parents; four complementary text/image restores; reset all stored conditioning to base at2; prospective causal hypotheses, not numerical RGB forecasts",
        terminal_status="unassessed")
    save(run_dir / "plan.json", config)
    worker = here / "flux_handoff_campaign.py"
    extras += [(here / "flux_trajectory_forecast_campaign.py", "flux_trajectory_forecast_campaign.py"),
        (here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json"),
        (selection, selection.name)]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name,
        "flux_trajectory_forecast_campaign.py", "campaign_support.py", "plan.json", "source-selection.json", "input-store", preflight.name],
        "path_token_allowlist": ["flux-handoff-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-shape-handoff-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=12, logical_branches=12,
            logical_checkpoint_replays=12, max_local_work_items=12,
            checkpoint_policy="two matching step1 projections; shared depth cuts; twelve in-process native futures with common source reset at2; no encoder or retries"),
        "resource_exception": {"ram_mb": 16000, "vram_mb": 13000, "calibration_job": report["job_id"],
            "measured_peak_vram_mb": 9316, "measured_peak_rss_mb": 2201.5, "native_calibration_job": "job-c2a467736b50",
            "native_measured_peak_rss_mb": 10697.6, "reason": "same resident CUDA BF16 B1 512px4step model and VAE; no encoder or forecast cache. Bound retained frontier tensors under128MB plus archive. Retain16GB native calibration margin and13GB VRAM exception."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-shape-handoff", payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=16000, vram_mb=13000, cpu_threads=4, est_wall_s=420, timeout_s=840, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed47. Step1 shape handoff from equalZ1 cuts; zero encodings, one load. Two refs+8 complementary text/image restores at4 depths+2 controls=12RGB. Every branch resets C tobase beforestep2. <=1100steps/840s, no retries; preserve mixtures and prediction misses.")
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
            a.run_dir, a.collect, log_storage="gzip", max_compressed_bytes=256 << 20, max_expanded_bytes=1 << 30)
    else:
        if a.predecessor is None:
            p.error("--predecessor required")
        result = stage(a.run_dir.resolve(), a.predecessor.resolve())
    print(json.dumps(result))
