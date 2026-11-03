"""Admit, or directly collect, one finite forwardless/timing qualification of the compiled-state view."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

WORKER = "compiled_state_forwardless.py"
MODEL_PATH = "/mnt/big/llm-models/Qwen2.5-1.5B"
WEIGHTS_SHA256 = "1f160d193f8c1b78aa628cd8dfd92d5da5631a2132d1c8074f96c2be52c62601"  # job-5ac69ae2b1d4 identity
SOURCE = "saturn experiments/2026-11-02-forecast-cost-and-compressed-addition"
F4, F4C, F5 = "F4-+pairs", "F4-+pairs|c", "F5-+totals"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


# earlier headline measurements this run is compared against, copied from the cited report files
COMPARISON = {
    "part_e": {"source": f"{SOURCE} results/e-full1/report.json carry (job-008821c65712)",
               "rows": {"fit": 160200, "eval": 32768},
               "agreement": {"F4-+pairs(20)": [0.984, 0.5444], "F4-+pairs|c(20)": [0.9842, 0.5358],
                             "F5-+totals(20)": [0.9842, 0.5415], "F4-+pairs|c(28)": [0.9787, 0.0],
                             "F5-+totals(28)": [0.9787, 0.0]},
               "agreement_fields": ["answer agreement", "error recall"],
               "timing_ms": {"1": {"native_prefix_cached": 10.979, "F4-+pairs(20)+digit_head": 3.714,
                                   "F5-+totals(28)+digit_head": 0.735},
                             "256": {"native_prefix_cached": 105.177, "F4-+pairs(20)+digit_head": 30.383,
                                     "F5-+totals(28)+digit_head": 0.753}},
               "speedup": {"1": {"F4-+pairs(20)+digit_head": 2.96, "F5-+totals(28)+digit_head": 14.93},
                           "256": {"F4-+pairs(20)+digit_head": 3.46, "F5-+totals(28)+digit_head": 139.69}}},
    "part_f": {"source": f"{SOURCE} results/f-full1/report.json gap",
               "agreement": {"T3c(20) sweep 32768": [0.9831, 0.5172], "T3c(20) headline 160200": [0.9832, 0.5334],
                             "T3c-nativeKV(20) sweep 32768": [0.9949, 0.8209],
                             "T3c-F5-+totals(20) sweep 32768": [0.9834, 0.5244]},
               "timing_ms": {"1": {"native_prefix_cached": 10.804, "T3c(20)+digit_head": 3.881},
                             "256": {"native_prefix_cached": 105.112, "T3c(20)+digit_head": 12.235}},
               "speedup": {"1": {"T3c(20)+digit_head": 2.78}, "256": {"T3c(20)+digit_head": 8.59}}},
    "part_g": {"source": f"{SOURCE} results/g-full1/report.json text_families.G1",
               "chain": {"n": 744, "exact_through": [1.0, 1.0, 1.0, 1.0], "answer_acc_pred": 1.0},
               "timing_ms": {"1": {"native_kv_decode": 42.078, "predict_chain_L28": 3.602},
                             "256": {"native_kv_decode": 190.206, "predict_chain_L28": 6.359}},
               "speedup": {"1": {"predict_chain_L28": 11.68}, "256": {"predict_chain_L28": 29.91}}}}


def plan(*, smoke):
    value = {"schema": "mdb-forwardless-plan-v1", "model": "qwen2.5-1.5b", "model_path": MODEL_PATH,
             "expected_weights_sha256": WEIGHTS_SHA256, "device": "cuda", "seed": 20261102, "lam": 1.0,
             "batch": 256, "cpu_threads": 4, "reps": 10, "warmup": 5, "timing_batches": [1, 256], "parity_rows": 64,
             "taus": [None, 1.0, 2.0, 4.0, 8.0],
             "addition": {"n_fit": 160200, "n_eval": 32768, "r2_rows": 2048, "layers": [20, 28],
                          "partial_layers": [20], "feature_sets": [F4, F4C, F5],
                          "timing_full": [[F4, 20, "digit"], [F5, 28, "digit"]], "timing_t3c": [[F4, 20, "digit"]],
                          "serve_warmup": 2048, "serve_refit_every": 1024},
             "g1": {"steps": 4, "expected_holdout": [0, 1, 9, 18, 27, 29, 31, 36, 39],
                    "expected_shape": [12, 33, 738, 744, 384], "serve_warmup": 64, "serve_refit_every": 32,
                    "serve_batch": 64, "serve_window": 500},
             "smoke": smoke, "comparison": COMPARISON,
             "limits": ["single seed", "timings are one process on one GPU; mdb's layer loop calls the HF decoder "
                        "layers, Parts E-G timed their own manual loop",
                        "headline arms only, 10 timed repetitions (Parts E-G used 30); parity gates on 64 rows",
                        "mdb's L28 arms compile only the score positions; Part E compiled all ten",
                        "G1 HO rows contain fillers the chain never saw; they are reported, not compared"]}
    if smoke:
        value["addition"].update({"n_fit": 4096, "n_eval": 1024, "r2_rows": 512, "serve_warmup": 256,
                                  "serve_refit_every": 128})
        value["taus"] = [None, 2.0]
    return value


def logical_operations(config):
    a = config["addition"]
    fsets, layers, partial = len(a["feature_sets"]), len(a["layers"]), len(a["partial_layers"])
    timing = 2 * len(config["timing_batches"])  # one forward group and one serving group per batch
    taus = len(config["taus"])
    # addition: 2 decodes, measure fit + eval passes, full arms, partial arms (compiled + oracle K/V), ladder gate,
    # timing groups, 2 chain scorings, one serve per tau; G1: 2 decodes, gate, 4 chain scorings, timing, serves
    return 2 + 2 + fsets * layers + 2 * fsets * partial + 1 + timing + 2 + taus + 2 + 1 + 4 + timing + taus


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
                     "path_token_allowlist": ["forwardless-result", MODEL_PATH],
                     "hashed_roots": [WORKER, "campaign_support.py", "compiled_state_addition.py", "plan.json",
                                      "preflight.v1.json"]})
    worker = here / WORKER
    extras = [(here / "discovery_upgrades.py", "campaign_support.py"),
              (here / "compiled_state_addition.py", "compiled_state_addition.py"),
              (run_dir / "plan.json", "plan.json"), (preflight, preflight.name)]
    installed = package_files("mdb")
    module_hashes = {f"mdb/{name}": sha(installed[name].read_bytes())
                     for name in ("compiled_state.py", "compiled_state_runner.py", "compiled_state_serving.py",
                                  "timing.py")}
    ops = logical_operations(config)
    a = config["addition"]
    batches = -(-a["n_fit"] // config["batch"]), -(-a["n_eval"] // config["batch"])
    forwards = (2 * batches[0] + 3 * batches[1] + batches[0]
                + batches[1] * (2 + len(a["feature_sets"]) * (len(a["layers"]) + 2 * len(a["partial_layers"])))
                + len(config["taus"]) * 3 * batches[1] + 4 * 9 + 4 * 3)
    scheduling = one_lease_scheduling_contract(logical_operations=ops, logical_branches=ops - 4,
        logical_checkpoint_replays=0, max_local_work_items=64,
        checkpoint_policy="one resident CUDA BF16 Qwen2.5-1.5B; prefix K/V cached once per family; batched suffix "
                          "passes through mdb's manual layer loop; resumes run only layers L..27",
        extra={"max_native_batched_forwards_approx": forwards})
    request = {"schema": "mdb-forwardless-qualification-request-v1", "task_family": "forward",
        "model": config["model"], "needs": {"cuda": True}, "backend": "hf", "device": "cuda", "dtype": "bfloat16",
        "batch_size": config["batch"], "seq_len": 49,
        "execution_environment": intent(), "scheduling_contract": scheduling, "config": config,
        "config_bytes_sha256": sha(raw),
        "source_hashes": {**{name: sha(path.read_bytes()) for path, name in [(worker, worker.name), *extras]},
                          **module_hashes},
        "resource_basis": (f"reserve {ram_mb} MB RAM / {vram_mb} MB VRAM / 4 threads; "
                           + ("first measurement: parity on 64 rows and score-tail hook logits keep the 3.1 GB model under 7 GB; "
                              "job-5ac69ae2b1d4 peaked 4,841 MB RSS"
                              if smoke else "smoke-measured peaks x1.3")),
        "selection_policy": "fixed seed-derived Part D/E/F/G rows; every arm, gate, tau and family retained"}
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw,
                 additional_files=extras)
    note = (f"{'SMOKE ' if smoke else ''}mdb forwardless qualification, Qwen2.5-1.5B CUDA BF16: addition fit "
            f"{a['n_fit']} eval {a['n_eval']} (true resume L{a['layers']}, T3c L{a['partial_layers']}, ladder parity, "
            f"registered timing b{config['timing_batches']}, zero-layer chain, serving taus {config['taus']}) and "
            f"Part G G1 chain/timing/serving; one load, no retries.")
    job = guarded_mrun_launch(worker_command(WORKER), request=request, experiment="mdb-forwardless-qualification",
        task="forward", model=config["model"], payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4,
        est_wall_s=420 if smoke else 1080, timeout_s=1200 if smoke else 2400, priority=80,
        retry_on_kill=False, preflight=True, note=note[:500])
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--vram-mb", type=int, default=7000)
    parser.add_argument("--ram-mb", type=int, default=8000)
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
