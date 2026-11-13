"""Submit one finite native shape/source timing assay from authenticated saved cuts."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stage(run_dir, predecessor):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    run_dir.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-concept-result"
    inventory, source, old, report = (json.loads(p.read_text()) for p in
        (data / "files.json", data / "source.json", data / "plan.json", predecessor / "report.json"))
    if report["job_id"] != "job-c2a467736b50" or report["archive_sha256"] != "24454082c80c1c646b8a7e4948b948fa4765f49ff5ca4b1f4edff3e964cc409d":
        raise ValueError("unexpected predecessor")
    if sha(data / "source.json") != inventory["source.json"]:
        raise ValueError("source identity differs")
    cuts = {n: source["cuts"][n]["0"]["projected"] for n in ("base", "shape")}
    relatives = set()
    for h in cuts.values():
        relative = f"typed-store/cuts/{h}.json"
        cut = json.loads((data / relative).read_text())
        relatives.add(relative)
        relatives.update(f"typed-store/pages/{h}.json" for h in cut["slots"].values())
        relatives.update(f"typed-store/blobs/{h}.safetensors" for h in cut["physical_blobs"])
    extras = []
    for relative in sorted(relatives):
        if sha(data / relative) != inventory[relative]:
            raise ValueError("source CAS differs")
        extras.append((data / relative, relative.replace("typed-store/", "input-store/", 1)))
    config = {k: old[k] for k in ("model", "model_path", "expected_checkpoint_sha256", "height", "width", "steps", "guidance_scale", "cpu_threads")}
    config.update(schema="mdb-flux-shape-timing-plan-v1", seed=47, input_cuts=cuts,
        predecessor={"job_id": report["job_id"], "archive_sha256": report["archive_sha256"]},
        prompts=[p for p in old["prompts"] if p["id"] in ("base", "shape")],
        max_transitions=800, max_images=10, max_wall_seconds=840, max_archive_bytes=256 << 20,
        scope="fresh seed47; native causal timing futures from observed encoder carriers, no numerical RGB forecast; early working text shape with persistent base versus shape conditioning; reverse source switches before timesteps1/2/3",
        hypotheses={
            "H0_mechanics": "changing unread stored conditioning preserves first scheduler latent exactly; early.switch1 equals coherent, coherent.switch1 equals early; base and early exactly reproduce prior seed47 PNGs",
            "H1_reference": "coherent shape conditioning renders red sphere left and blue cube right; if not, semantics of clean rescue are underdetermined",
            "H2_delayed_rescue": "early.switch2 moves final latent closer to coherent than early.switch3; both measured against their common early endpoint without a scientific cutoff",
            "H3_return_to_base": "coherent.switch2 moves final latent closer to early than coherent.switch3; timing effects may be nonmonotone and counterexamples are retained",
            "H4_source_only": "later shape source without changed initial text changes shape more readily than established ordering; retain individual object outcomes and collateral"},
        alternatives=["first-step shape commitment", "later source-driven repair/refinement", "state/source compatibility", "coherent source itself fails requested shape"],
        terminal_status="unassessed")
    save(run_dir / "plan.json", config)
    worker = here / "flux_shape_timing.py"
    extras += [(here / "flux_trajectory_forecast_campaign.py", "flux_trajectory_forecast_campaign.py"),
        (here / "flux_concept_routes.py", "flux_concept_routes.py"),
        (here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, "input-store", "plan.json",
        "flux_trajectory_forecast_campaign.py", "flux_concept_routes.py", "campaign_support.py", preflight.name],
        "path_token_allowlist": ["flux-shape-timing-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-shape-timing-request-v1", "config": config, "task_family": "forward", "model": config["model"],
        "needs": {"cuda": True}, "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=10, logical_branches=10,
            logical_checkpoint_replays=9, max_local_work_items=10,
            checkpoint_policy="one fresh prefix, shared early/coherent project cuts; finite ten native RGB futures, zero encodings, no retries"),
        "resource_exception": {"ram_mb": 16000, "vram_mb": 13000, "cpu_threads": 4,
            "calibration_job": "job-dc521d90c65d", "measured_peak_rss_mb": 12294.8, "measured_peak_vram_mb": 9316,
            "native_calibration_job": "job-c2a467736b50", "native_measured_peak_rss_mb": 10697.6,
            "failed_prefix_calibration_job": "job-52e1ac69f07e", "prefix_measured_peak_rss_mb": 7705.,
            "reason": "same resident512px B1 four-step CUDA BF16 checkpoint. Native-only panel never allocates the prior7.81GB forecast cache; full native concept assay peaked10.70GB with four encodings. Two saved carriers, bounded project cuts and direct archive emission fit16GB with margin; retain13GB VRAM exception."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-shape-timing", payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=16000, vram_mb=13000, cpu_threads=4, est_wall_s=600, timeout_s=840,
        retry_on_kill=False, preflight=True,
        note="Shape binding timing: Klein4B CUDA BF16 B1 512px4steps seed47. One load, zero encodes, <=800 transitions,10 native images. Base/early/coherent/source-only; reciprocal conditioning switches before1/2/3 from shared project cuts. Hypotheses sealed before blocks. <=840s; no retries, preserve mixtures/counterexamples.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=256 << 20, max_expanded_bytes=1 << 30)
    else:
        if args.predecessor is None:
            parser.error("--predecessor required")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve())
    print(json.dumps(result))
