"""Submit one resident fresh-context replication of the single0 K/V assay."""
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


def stage(run_dir, *, submit=True):
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    campaign = runpy.run_path(str(here / "flux_single_contexts_campaign.py"))
    panel = runpy.run_path(str(here / "flux_single_contexts.py"))
    config = {
        "schema": "mdb-flux-single-contexts-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "contexts": campaign["CONTEXTS"], "branches": panel["BRANCHES"],
        "height": 512, "width": 512, "steps": 4, "cpu_threads": 4,
        "selector": {"logical_step": 3, "phase": "single", "index": 0},
        "attention_query_indices": [0, 31, 32, 255, 511, 512, 767, 1023],
        "max_adapter_steps": 861, "max_acts": 39, "max_candidate_calls": 15,
        "max_images": 27, "max_vae_calls": 27, "max_wall_seconds": 840,
        "archive_compression_level": 1,
        "hypotheses": {
            "H-entry": "For every freshly built paired single0 entry, only working T differs; image, latent, stored C0, time/context/modulation/gates/RoPE, positions and schedule are identical. Source image Q/K/V and parallel image MLP preactivation/activation match exactly.",
            "H-selection": "K-only changes only positioned text K; V-only only text V; KV both. All edited calls retain recipient Q and image K/V. V-only mathematically preserves attention weights; K-only and KV share weights. A K edit can change normalization over both image and text banks. No ordering among K-only, V-only and KV image endpoints is predicted.",
            "H-local-KV": "In each context, KV reproduces text-source image attention dispatch, native fused attention-plus-parallel-MLP image projection, and postblock image exactly; image MLP features stay fixed for all three edits. Candidate postblock T need not equal text-source T. Preserve native fused concat/projection and BF16 arithmetic.",
            "H-numeric-attention": "After exactly two native source calls reveal QKV, independently calculate NumPy FP64 softmax attention at the fixed eight image queries, all heads, for K/V/KV. Seal actual numeric predictions and forecast-only text-bank masses before edited calls or adapter prefixes. Compare saved samples with native BF16 dispatch using signed/RMS/max errors without a numerical gate. K and V are novel conditional targets; KV is a represented consistency control. Source-reviewed QKV-only dataflow is not hermetic isolation or an independent MLP/RGB/raw-prompt prediction.",
            "H-controls": "Both direct native source candidates equal their actual adapter exits. Copy sham equals zero final latent/PNG; coherent native text-source postblock I/T equals text reference final latent/PNG. KV.image and KV.both install identical I with explicitly different selected T. No historical PNG equality is asserted for new contexts.",
            "H-text-policy": "K.image/V.image/KV.image retain native zero postblock T. KV.both installs KV candidate T; text.text only native text-source T; coherent text-source I/T. All hybrids start at native zero post-single0, stored C0 unchanged, with no further conditioning projection. The text reference is I0/T1/C0, not the separate own-C joint-prefix result I1/T1/C1.",
            "H-generalization": "Apply exactly the same source construction and equations to p0-s53,p1-s47,p1-s53 without fitting per-context parameters. p1 seeds47/53 are a within-prompt seed contrast; p0/p1 seed53 are a within-seed prompt contrast. Historical p0 seed47 is separate evidence, not the missing cell of a current exact-code 2x2 cohort. Local equations and forecast errors are tested; endpoint appearances and rankings other than H-text-direction remain exploratory, with reversals retained.",
            "H-text-direction": "Tentative cross-context directional prediction: text.text ends closer to the text reference than each of K.image,V.image,KV.image, evaluated separately in final latent L2 and RGB MAE. Retain each signed margin, misses and disagreement between observables. This is a forecast of endpoint distances, not a mediation fraction or semantic prediction.",
            "H-scope": "Three fresh own-context altered histories. Two unique native prompt encodings, three native seeded noise draws, one resident model, no foreign prompt donor or imported StateCut. Global plan frozen before model load/encoding/noise; each architectural seal follows observed builder prefix; each numerical seal follows exactly two allowed source captures. Nine dependent futures per context do not enumerate every possible future. No universal object dictionary or independent prompt-to-image numerical prediction."
        },
        "scope": "Fresh connected three-context replication of one final-step single0 K/V law. Per context: native Z2, zero-C history to Z3, paired zero/restored-own-C project+5joint, T-only transplant into I0/C0;96 builder transitions+191 panel transitions,13 Acts,5 direct blocks,9 VAE/finals. Global861 transitions/39 Acts/15 direct/27 VAE. All native futures retained; no semantic gates.",
        "terminal_status": "unassessed",
    }
    campaign["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    worker = here / "flux_single_contexts_campaign.py"
    extras = [(here / name, name) for name in (
        "flux_single_contexts.py", "flux_single_kv.py", "flux_family_custody.py")]
    extras.append((run_dir / "plan.json", "plan.json"))
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *(name for _, name in extras), preflight.name],
        "path_token_allowlist": ["flux-single-contexts-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    ram_mb, vram_mb = 24000, 13000
    request = {"schema": "mdb-flux-single-contexts-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True},
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=27,
            logical_branches=27, logical_checkpoint_replays=27, max_local_work_items=45,
            checkpoint_policy="Three sequential fresh contexts, two cached native prompt encodings/three noise draws; each96 source-building+191 panel transitions and nine native finals. Two seals per context, full QKV/ports/cuts retained, bounded streaming multipart transport; no retries."),
        "resource_exception": {"ram_mb": ram_mb, "vram_mb": vram_mb,
            "disk_gb": 8,
            "ram_kill_factor": RAM_KILL_FACTOR, "ram_kill_ceiling_mb": kill_ceiling_mb(ram_mb),
            "calibration_job": "job-1f1587e6983b", "measured_peak_rss_mb": 16059,
            "measured_peak_vram_mb": 9282,
            "prior_single_panel_job": "job-3cec09435010", "prior_single_panel_peak_rss_mb": 4750.5,
            "prior_single_panel_peak_vram_mb": 9992,
            "reason": "Full native encoder startup previously reached16059MB; use24GB RAM headroom for encoder startup, two cached conditionings, full retained panels and bounded archive spool. Declare8GB transient disk for retained arrays, archive spool and base64 agent log; cold model storage is separate. Encoder released before resident denoising; contexts sequential, not parallel accelerator jobs. CUDA BF16 B1 512px4step lane uses qualified13GB VRAM. Reservations are headroom, not measured speedup."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}
    return submit_staged(run_dir)


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("staged assay already has a scheduler admission")
    request = json.loads((run_dir / "request.json").read_text())
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("staged request digest differs")
    if any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged payload bytes differ")
    config = json.loads((run_dir / "plan.json").read_text())
    if config != request["config"]:
        raise ValueError("staged plan differs from request")
    resource = request["resource_exception"]
    job = guarded_mrun_launch(worker_command("flux_single_contexts_campaign.py"), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-single-contexts",
        payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"], disk_gb=resource.get("disk_gb", 8), cpu_threads=4,
        est_wall_s=600, timeout_s=900, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px4step. FreshP0seed53/P1seed47,53; one resident lease,2encodes/3noise. Each build own alteredZ3 paired single0 sources then seal structural and QKV-only numerical K/V forecasts beforeedits;9futures/context.861adaptersteps/39Acts/15direct/27VAE. Sequential27finals,fullretention/boundedmultipart,900s/no retries; retainerrors/rankreversals.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--submit-staged", action="store_true")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](
            args.run_dir.resolve(), args.collect)
    elif args.submit_staged:
        result = submit_staged(args.run_dir.resolve())
    else:
        result = stage(args.run_dir.resolve(), submit=not args.stage_only)
    print(json.dumps(result))
