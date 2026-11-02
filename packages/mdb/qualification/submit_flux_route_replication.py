"""Freeze and execute three new contexts for the source-qualified S04 record."""
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


def campaign(prior):
    suffix = " on a plain white background. Two separate objects, simple studio product photograph."
    wording = {
        "original": {"base": "A red cube to the left of a blue sphere" + suffix,
                     "changed": "A red cube to the right of a blue sphere" + suffix},
        "paraphrase": {"base": "A blue sphere is positioned to the right of a red cube" + suffix,
                       "changed": "A blue sphere is positioned to the left of a red cube" + suffix},
    }
    return {"schema": "mdb-flux-route-replication-campaign-v1",
        "model": prior["model"], "model_path": prior["model_path"],
        "expected_checkpoint_sha256": prior["expected_checkpoint_sha256"],
        "source_record": "flux.merge.S04", "source_commit": "302f6436622b65bb797ccfdfda23d2b6e811d073",
        "cases": [{"id": name, "seed": seed, "wording": label, "prompts": wording[label], "attention_sources": attention}
            for name, seed, label, attention in (("original37", 37, "original", True),
                ("paraphrase31", 31, "paraphrase", False), ("paraphrase37", 37, "paraphrase", False))],
        "historical_record": {"sealed_hypotheses": prior["hypotheses"],
            "attention_only": {"original_prediction": "base", "observed_ordering": "changed", "status": "contradicted"},
            "scope": "Original seed31 wording; historical record remains unchanged. Replication expectations below incorporate its counterexample."},
        "replication_expectations": {
            "H-reference": "Base reference has red cube left; initial changed working text produces red cube right despite base stored conditioning.",
            "H-ordering": {"merge.000": "changed", "merge.001": "changed", "merge.010": "changed",
                "merge.100": "changed", "merge.110": "base", "merge.111": "base"},
            "H-direction": "Final latent projection010 < projection001; projection110 < both projection100 and projection010.",
            "H-interaction": "With changed MLP fixed, p110-p100-p010+p000 is negative.",
            "H-controls": "Copy and merge000 reproduce changed future exactly; merge111 has exact base post-block image but retains changed text.",
            "H-attention-sources": "Exploratory source split at the preselected original37 context: with image residual base and MLP changed, base image K/V moves toward base more than base text K/V within each fixed Q source. This is a new hypothesis, not established by S04.",
        },
        "attention_mask_bits": ["base image-query source (text-query outputs discarded)", "base text K/V", "base image K/V"],
        "attention_controls": "000 joins merge100;111 joins merge110 with exact native attention reconstruction; six remaining masks get native RGB suffixes.",
        "scope": "Three preselected new contexts, same checkpoint and exact single4 boundary. No universal object dictionary or RGB numerical forecast. Both source K and V are replaced together; query source is a separate receiving-state factor. Preserve all counterexamples and nonseparating reference contexts.",
        "terminal_status": "unassessed"}


def stage(run_dir, campaign_path, case_id):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    plan = json.loads(campaign_path.read_text())
    if plan["schema"] != "mdb-flux-route-replication-campaign-v1":
        raise ValueError("unknown campaign plan")
    selected = [case for case in plan["cases"] if case["id"] == case_id]
    if len(selected) != 1:
        raise ValueError("select exactly one frozen case")
    case = selected[0]
    images = 17 if case["attention_sources"] else 11
    config = {key: plan[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "scope")}
    config["hypotheses"] = plan["replication_expectations"]
    config.update(schema="mdb-flux-route-replication-plan-v1", fresh_context=True, case_id=case_id,
        campaign_sha256=sha(campaign_path), seed=case["seed"], prompts=case["prompts"], attention_sources=case["attention_sources"],
        block=4, selector={"logical_step": 0, "phase": "single", "index": 4},
        height=512, width=512, steps=4, cpu_threads=4, max_images=images,
        max_adapter_steps=2200 if images == 17 else 1500, max_wall_seconds=1100 if images == 17 else 840,
        max_archive_bytes=512 << 20)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    names = ["flux_parallel_merge.py", "flux_attention_sources.py", "flux_functional_denoiser.py", "flux_trajectory_forecast_campaign.py"]
    worker = here / "flux_merge_campaign.py"
    extras = [(here / name, name) for name in names] + [(here / "discovery_upgrades.py", "campaign_support.py"),
        (run_dir / "plan.json", "plan.json"), (campaign_path, "campaign-plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *[name for _, name in extras], preflight.name],
        "path_token_allowlist": ["flux-merge-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-route-replication-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True},
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=images,
            logical_branches=images, logical_checkpoint_replays=images, max_local_work_items=32,
            checkpoint_policy="two fresh encodings once; common parent; matched prefixes once; exact scalar suffixes, no retries"),
        "resource_exception": {"ram_mb": 32000, "vram_mb": 13000, "calibration_job": "job-c2a467736b50",
            "measured_peak_vram_mb": 9470., "measured_peak_rss_mb": 10697.6, "measured_elapsed_seconds": 736.55,
            "reason": "same 512px B1 Klein CUDA BF16 contract; sequential encoder then transformer/VAE; bounded observed ports and one context per lease"}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward",
        experiment="mdb-flux-route-replication", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=32000, vram_mb=13000, cpu_threads=4,
        est_wall_s=780 if images == 17 else 540, timeout_s=config["max_wall_seconds"] + 100,
        retry_on_kill=False, preflight=True,
        note=f"Klein4B CUDA BF16 B1 512px4steps {case_id}.2fresh encodings,shared prefixes,single4. {images}RGB: R/A/M factorial+refs/sham" +
            ("+6Q/textKV/imageKV rows;2corners reuse exact merge futures." if images == 17 else ".") +
            " C stays base,T changed. Frozen campaign hypotheses; all rows retained. One load,no retries; bounded native steps/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir), "case": case_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--make-plan", type=Path, help="prior plan supplying pinned checkpoint identity")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.make_plan:
        plan = campaign(json.loads(args.make_plan.read_text()))
        plan["historical_plan_sha256"] = sha(args.make_plan)
        save(args.path, plan)
        result = {"campaign": str(args.path), "sha256": sha(args.path)}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.path, args.collect, max_compressed_bytes=512 << 20, max_expanded_bytes=1024 << 20, log_storage="gzip")
    else:
        if not args.campaign or not args.case:
            parser.error("--campaign and --case required")
        result = stage(args.path.resolve(), args.campaign.resolve(), args.case)
    print(json.dumps(result))
