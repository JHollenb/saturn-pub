"""Admit, or directly collect, one finite compiled-state residual qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

WORKER = "compiled_state_addition.py"
MODEL_PATH = "/mnt/big/llm-models/Qwen2.5-1.5B"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plan(*, smoke):
    value = {"schema": "mdb-compiled-state-addition-plan-v1", "model": "qwen2.5-1.5b", "model_path": MODEL_PATH,
             "seed": 20261102, "lam": 1.0, "threshold": 0.95, "batch": 256, "cpu_threads": 4,
             "feature_sets": ["F1-digits", "F4-+pairs", "F4-+pairs|c", "F5-+totals"],
             "n_fit": 32768, "n_eval": 2048, "r2_rows": 2048, "layers": [4, 12, 20, 22, 23, 24, 26, 28],
             "smoke": smoke,
             "comparison": {
                 "source": "saturn experiments/2026-11-02-forecast-cost-and-compressed-addition FINDINGS-D.md, "
                           "FINDINGS-E.md, results/e-full1/report.json (job-008821c65712)",
                 "part_e_fit_rows": 160200, "part_e_eval_rows": 32768,
                 "row_agreement": {"F1-digits": {"4": 0.9899, "22": 0.9705, "23": 0.79, "24": 0.0293},
                                   "F4-+pairs": {"24": 0.9812, "26": 0.6922}},
                 "horizon_at_0.95": {"F1-digits": 22, "F4-+pairs": 24, "F4-+pairs|c": 28, "F5-+totals": 28},
                 "r2_L4_F1": [1.0, 0.822, 0.822, 0.906, 0.982, 0.823, 0.831, 0.911, 0.996, 0.867]},
             "limits": ["single seed; fit rows are 20% of the Part E fit, evaluation rows 6% of its sample",
                        "agreement is native greedy top-1 at '=', r0, r1 after the splice; R2 pools hidden dims",
                        "no timing claim"]}
    if smoke:
        value.update({"n_fit": 2048, "n_eval": 512, "r2_rows": 512, "layers": [4, 22, 23, 24, 28]})
    return value


def stage(run_dir, *, smoke, vram_mb, ram_mb):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import package_files, stage_worker, worker_command

    here = Path(__file__).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=False)
    config = plan(smoke=smoke)
    save(run_dir / "plan.json", config)
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
                     "path_token_allowlist": ["compiled-state-result", MODEL_PATH],
                     "hashed_roots": [WORKER, "campaign_support.py", "plan.json", "preflight.v1.json"]})
    worker = here / WORKER
    extras = [(here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json"),
              (preflight, preflight.name)]
    installed = package_files("mdb")
    module_hashes = {f"mdb/{name}": sha(installed[name].read_bytes())
                     for name in ("compiled_state.py", "compiled_state_runner.py")}
    splices = len(config["feature_sets"]) * len(config["layers"])
    batches = -(-config["n_eval"] // config["batch"])
    forwards = (2 * -(-(config["n_fit"] + config["n_eval"]) // config["batch"])
                + -(-config["n_fit"] // config["batch"]) + batches * (1 + splices))
    # logical work: one native fit pass, one native evaluation pass, one splice arm per (feature set, layer)
    scheduling = one_lease_scheduling_contract(logical_operations=2 + splices, logical_branches=splices,
        logical_checkpoint_replays=0, max_local_work_items=64,
        checkpoint_policy="one resident CUDA BF16 Qwen2.5-1.5B; prefix K/V cached once; batched suffix forwards; "
                          "compiled states spliced by layer pre-hook",
        extra={"max_native_batched_forwards": forwards})
    request = {"schema": "mdb-compiled-state-qualification-request-v1", "task_family": "forward",
        "model": config["model"], "needs": {"cuda": True}, "backend": "hf", "device": "cuda", "dtype": "bfloat16",
        "batch_size": config["batch"], "seq_len": 49,
        "execution_environment": intent(), "scheduling_contract": scheduling, "config": config,
        "config_bytes_sha256": sha(raw),
        "source_hashes": {**{name: sha(path.read_bytes()) for path, name in [(worker, worker.name), *extras]},
                          **module_hashes},
        "resource_basis": (f"reserve {ram_mb} MB RAM / {vram_mb} MB VRAM / 4 threads; "
                           + ("first measurement: Part E's manual forward peaked 7.6 GB VRAM at batch 512"
                              if smoke else "smoke-measured peaks x1.3")),
        "selection_policy": "fixed seed-derived Part D/E/H rows; every layer and feature set retained"}
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    note = (f"{'SMOKE ' if smoke else ''}mdb compiled-state residual view, Qwen2.5-1.5B CUDA BF16 addition: "
            f"fit {config['n_fit']} rows, eval {config['n_eval']}, {len(config['feature_sets'])} feature sets x "
            f"layers {config['layers']}; one load, prefix K/V cached, {forwards} batched forwards; no retries.")
    job = guarded_mrun_launch(worker_command(WORKER), request=request, experiment="mdb-compiled-state-qualification",
        task="forward", model=config["model"], payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4,
        est_wall_s=480 if smoke else 900, timeout_s=1200 if smoke else 1800, priority=80,
        retry_on_kill=False, preflight=True, note=note[:500])
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--vram-mb", type=int, default=8000)
    parser.add_argument("--ram-mb", type=int, default=12288)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.vram_mb > 13000:
        raise SystemExit("VRAM reservation above the 13000 MB device policy")
    if args.collect:
        helper = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))
        result = helper["collect"](args.run_dir, args.collect, max_compressed_bytes=32 << 20,
                                   max_expanded_bytes=128 << 20)
    else:
        result = stage(args.run_dir.resolve(), smoke=args.smoke, vram_mb=args.vram_mb, ram_mb=args.ram_mb)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
