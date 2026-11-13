"""Stage, strictly admit and collect one finite color/shape prediction assay."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run_dir):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    here = Path(__file__).resolve().parent
    # The caller supplies the exact historical bank as a source file. No worker
    # discovers or imports private workspace modules.
    bank = json.loads(ARGS.prompt_bank.read_text())["characters"][0]
    config = {"schema": "mdb-predicted-future-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "seed": 60665, "height": 512, "width": 512, "steps": 4,
        "prompts": {key: bank[key] for key in ("base", "color", "shape", "joint")},
        "prompt_bank_sha256": sha(ARGS.prompt_bank),
        "output_scratch": "/home/beast/.cache/mdb-research", "max_wall_seconds": 780,
        "semantic_predictions": {
            "native_square": "Color prompts yield violet rather than amber eyes; shape prompts yield narrower eyes. Copper hair, plum clothing, one adult head/shoulders, frontal camera and gray background remain. Assess native competence separately.",
            "write_fidelity": "Independently predicted joint M on base should closely reproduce measured joint-M transfer, including collateral; no exact RGB claim.",
            "strong_edit": "Joint-M transfer alone moves base eyes toward violet/narrow while preserving copper hair, clothing, framing and background. This is a test, not prior qualified single6 semantic authority.",
            "additive": "Independent color+shape M addition is an approximation; retain nonzero interaction and downstream discrepancy instead of imposing linear semantic composition.",
            "reuse": "Cached base attention/text plus selected M and native to_out/gate exactly equals full-block M-only intervention at fixed parent. Scope excludes changed attention/residual/text/time."},
        "scope": "One fresh seed and existing eye-color/shape prompt square. Four observed prefixes, four complete NumPy M forecasts sealed before current target; five frozen-parent interventions and nine native images.17 cached dose endpoints independently validated by17 separately counted full blocks. Cost comparison is instrumented fixed-parent endpoint evaluation, not end-to-end generation speedup or raw-prompt prediction."}
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    worker = here / "flux_predicted_future_campaign.py"
    names = ["flux_predicted_future.py", "flux_mlp_replacement.py", "flux_single_contraction.py", "flux_family_custody.py"]
    extras = [(here / name, name) for name in names] + [(run_dir / "plan.json", "plan.json"), (ARGS.prompt_bank, "prompt-bank.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *names, "plan.json", "prompt-bank.json", preflight.name],
        "path_token_allowlist": ["predicted-future-result", config["model_path"], config["output_scratch"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-predicted-future-request-v1", "config": config, "task_family": "forward",
        "model": config["model"], "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "needs": {"cuda": True}, "execution_environment": intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=26, logical_branches=9,
            logical_checkpoint_replays=5, max_local_work_items=32,
            checkpoint_policy="Four96-step prefixes, four16-step native endpoints and five15-step suffixes:523 transitions,5Acts528receipts. Four native source blocks,17 validation blocks,22 cached projections,9VAE. One lease,no retries."),
        "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "disk_gb": 8,
            "calibration_job": "job-68295609457d", "measured_peak_rss_mb": 6320.6, "measured_peak_vram_mb": 9672,
            "reason": "Reuse qualified full-encoder 24GB RAM/13GB VRAM batch1 512px resident Klein profile, one load. Four serial encodes, encoder released before transformer. Additional four entry states/fields bounded below1GB. NumPy4-thread256-neuron tiles. Actual prior profile also includes startup peak from job-1f1587e6983b RSS16059MB."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    return {"run_dir": str(run_dir), "request_sha256": request["request_sha256"], "submitted": False}


def submit(run_dir):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("assay already submitted")
    request = json.loads((run_dir / "request.json").read_text())
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request self seal differs")
    if any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged source changed")
    resource = request["resource_exception"]
    job = guarded_mrun_launch(worker_command("flux_predicted_future_campaign.py"), request=request,
        model=request["model"], task="forward", experiment="mdb-predicted-future", payload=run_dir / "payload", prefer_host="beast",
        gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"], disk_gb=resource["disk_gb"],
        cpu_threads=4, est_wall_s=600, timeout_s=900, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed60665 eye color/shape square. Four encodes/one noise;384 prefix steps; seal4 full M fields before target.523 native steps5Acts9VAE,4 source blocks,17 cached-dose validations. Cached attention/text native projection bypasses repeated MLP at fixed parent. SSD scratch,4 CPU threads,24GB RAM/13GB VRAM qualified profile. Worker780s/lease900s,no retries. Retain semantic failures and count validation cost.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prompt-bank", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect")
    ARGS = parser.parse_args()
    path = ARGS.run_dir.resolve()
    if ARGS.collect:
        result = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](path, ARGS.collect)
    elif ARGS.submit:
        result = submit(path)
    else:
        if ARGS.prompt_bank is None:
            parser.error("staging requires --prompt-bank")
        result = stage(path)
    print(json.dumps(result))
