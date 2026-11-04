"""Stage, explicitly admit and collect one full attention/MLP replacement assay."""
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


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def stage(run_dir):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    worker = here / "flux_attention_replacement_campaign.py"
    engine = runpy.run_path(str(worker))
    config = {
        "schema": "mdb-flux-attention-replacement-plan-v1",
        "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "context": engine["CONTEXT"], "height": 512, "width": 512, "steps": 4,
        "single_index": 6, "selector": {"logical_step": 3, "phase": "single", "index": 6},
        "cpu_threads": 4, "neuron_chunk": 256,
        "attention_query_chunk": 64, "attention_projection_chunk": 256,
        "query_indices": engine["QUERIES"], "neuron_indices": engine["NEURONS"],
        "branches": engine["BRANCHES"], "branch_execution_order": engine["BRANCH_EXECUTION_ORDER"],
        "observation_plan": engine["OBSERVATION_PLAN"],
        "array_storage": "bf16-bits", "archive_integer_ids": "native-int64",
        "max_adapter_steps": 245, "max_acts": 11, "max_candidate_calls": 9,
        "max_images": 9, "max_vae_calls": 9, "max_wall_seconds": 1500,
        "archive_compression_level": 1,
        "transport_limits": {"max_archive_bytes": 1536 << 20, "max_expanded_bytes": 3072 << 20},
        "hypotheses": {
            "H-fields": "Calculate complete image A and M for both observed pre6 histories from actual incoming I/T, time modulation, all QKV/MLP weight rows, learned Q/K norm weights and upstream positional cos/sin. No current-target native Q/K/V/norm/A/M is a predictor input. Freeze explicit BF16/FP64 arithmetic before target entry; preserve every numerical discrepancy.",
            "H-consumer": "Feed sealed complete image A/M into the actual native fused to_out projection, with recipient text, residual and gate unchanged. Native A and M still execute before overwrite: consumer substitution, not compute skipping or speedup.",
            "H-matched": "Low/high same-history AM drop-ins test conditional numerical fidelity. AM.predicted versus AM.measured transfers the same high-source image fields into one low recipient and measures approximation error relative to the native measured intervention. No effect direction or quality threshold is asserted.",
            "H-decomposition": "A.predicted and M.predicted separately transfer predicted high-source fields into low. Without matched measured single-component branches, these are exploratory combined source-transfer/approximation observations, not pure attention-versus-MLP attribution.",
            "H-trajectory": "After sealing, execute fixed reference suffixes first and retain their arrays. At post6 and fifteen native transitions retain reference-qualified signed errors, descriptors and fixed coordinate witnesses. Full candidate norms are worker observations where candidate arrays are not archived. Mark stale latent state, readout noise and scheduler commit separately. No interpolation or semantic repair claim from norm contraction alone.",
            "H-scope": "One fresh seed73 context; both pre6 image/text may differ. Earlier-clock block6 executions in the108-step builder are permitted. Text-side block computation, projection, residual/gate and suffix remain native. No raw-prompt image prediction, semantic dictionary, automatic family discovery or hermetic isolation claim."
        },
        "terminal_status": "unassessed",
        "scope": "Fresh seed73 single6 full image attention plus MLP forecast and native-consumer substitution. One CUDA lease/load/encode/noise;245 native transitions,11 Acts,256 receipt emissions,nine direct calculations/nine VAE futures. Two native plus seven edited direct calls; reference-qualified16-boundary observations. Trusted pinned dataflow, no native target values in prediction, no skipped-compute claim."
    }
    engine["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    names = ["flux_attention_reference.py", "flux_mlp_replacement.py", "flux_single_mlp.py",
             "flux_single_mlp_campaign.py", "flux_single_contraction.py", "flux_single_contexts.py",
             "flux_single_kv.py", "flux_family_custody.py"]
    extras = [(here / name, name) for name in names] + [(run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *names, "plan.json", preflight.name],
        "path_token_allowlist": ["flux-attention-replacement-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-attention-replacement-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "backend": "diffusers",
        "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "needs": {"cuda": True}, "execution_environment": intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1,
            logical_operations=9, logical_branches=9, logical_checkpoint_replays=9, max_local_work_items=16,
            checkpoint_policy="One fresh108-transition source builder; seal two full A/M fields before current target. Two native and seven edited direct calls; two native prefixes plus nine15-step suffixes. Three reference trajectories retained and per-cursor candidate observations. No retries."),
        "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "disk_gb": 10,
            "calibration_job": "job-68295609457d", "measured_peak_rss_mb": 6320.6,
            "measured_peak_vram_mb": 9672.0,
            "reason": "Same qualified full-encoder24GB RAM/13GB VRAM lane, with encoder released after one encoding. Previous same-geometry full-M run245transitions9VAE completed740.34s pre-export/860.3s runner. Additional QKV rows56.6MB BF16, two25.2MB-total attention fields, bounded64-query scores/256-row projection tiles and CPU-only reference trajectories approximately0.4GB. Retain native candidates only for two source histories and archive edited ports immediately; no measured speedup claim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request,
        config_bytes=raw, additional_files=extras)
    return {"submitted": False, "run_dir": str(run_dir), "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("attention replacement assay already admitted")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or any(sha(run_dir / "payload" / name) != value for name, value in request["source_hashes"].items())
            or read(run_dir / "plan.json") != request["config"]):
        raise ValueError("staged attention request/source/plan changed")
    config, resources = request["config"], request["resource_exception"]
    note = ("Klein4B CUDA BF16 B1 512px freshseed73:108 builder steps/4Acts; seal complete imageA+M for2histories before single6. "
        "245native/11Acts/256receipts/9direct/9VAE; per-cursor paired evidence,3 CPU reference traces. "
        "One load/encode/noise,4CPU,24GB RAM/13GB VRAM. Worker1500s/outer2100s; no retries. "
        "Native attention/MLP still run before overwrite. Archive capped1536MiB/expanded3072MiB; all numerical misses retained.")
    if len(note) > 500:
        raise ValueError("queue note exceeds scheduler limit")
    job = guarded_mrun_launch(worker_command("flux_attention_replacement_campaign.py"), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-attention-replacement",
        payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=resources["ram_mb"], vram_mb=resources["vram_mb"], disk_gb=resources["disk_gb"],
        cpu_threads=4, est_wall_s=1200, timeout_s=2100, retry_on_kill=False, preflight=True, note=note)
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submit-staged", action="store_true")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        limits = read(args.run_dir / "request.json")["config"]["transport_limits"]
        result = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](
            args.run_dir.resolve(), args.collect, limits=limits)
    elif args.submit_staged:
        result = submit_staged(args.run_dir.resolve())
    else:
        result = stage(args.run_dir.resolve())
    print(json.dumps(result))
