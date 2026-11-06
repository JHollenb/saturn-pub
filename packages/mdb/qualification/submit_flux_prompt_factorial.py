"""Stage one finite four-prompt FLUX campaign from qualified immutable helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_LAUNCHER_SHA256 = "068e9706dc47e2e66d2a580228cfbc016f294b402a352022629742062bf0444c"
INITIAL_WORKER_SHA256 = "1ae1a639cb2407adadda597af48ef581d0129f6cf2b87df7e249274247b6129b"
QUALIFIED_READER_SHA256 = "93760e398fe844ca0c2000a902891ee8fd98a0ab9a7496a7f5654202ce04cb45"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


base_path = HERE / "submit_flux_initial_noise.py"
if sha(base_path) != BASE_LAUNCHER_SHA256:
    raise ValueError("qualified worker-environment launcher differs")
B = runpy.run_path(str(base_path))


def command(request):
    result = B["sealed_worker_command"](request)
    if result[-1] != "flux_initial_noise_campaign.py":
        raise ValueError("qualified worker bootstrap ABI differs")
    result[-1] = "flux_prompt_factorial.py"
    return result


def accepted_sources(prior_run):
    request = read(prior_run / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("accepted predecessor request changed")
    names = (*B["PRIOR_HELPERS"], "flux_initial_noise_campaign.py")
    workers = {}
    for name in names:
        source = prior_run / "payload" / name
        if sha(source) != request["source_hashes"][name]:
            raise ValueError("immutable qualified worker helper changed: " + name)
        workers[name] = source
    if request["source_hashes"]["flux_initial_noise_campaign.py"] != INITIAL_WORKER_SHA256:
        raise ValueError("accepted initial-noise execution profile differs")
    independent, authorities = {}, {}
    for name, digest in request["independent_math_sources"].items():
        source = prior_run / "independent-math-freeze" / name
        if sha(source) != digest:
            raise ValueError("immutable independent numerical helper changed: " + name)
        independent[name] = source.read_bytes()
        authorities[name] = {"authority": "accepted predecessor immutable source freeze", "sha256": digest}
    reader = HERE / "analyze_flux_initial_noise.py"
    if sha(reader) != QUALIFIED_READER_SHA256:
        raise ValueError("previously qualified lossless integer reader changed")
    independent[reader.name] = reader.read_bytes()
    authorities[reader.name] = {"authority": "qualified lossless reader published at da129da", "sha256": sha(reader),
        "prior_preoutcome_sha256": request["independent_math_sources"][reader.name]}
    for name in ("analyze_flux_prompt_factorial.py", "collect_flux_prompt_factorial.py"):
        source = HERE / name
        independent[name] = source.read_bytes()
        authorities[name] = {"authority": "new pre-outcome audit/collection source", "sha256": sha(source)}
    return request, workers, independent, authorities


def stage(run_dir, *, prior_run):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    prior_request, workers, independent, authorities = accepted_sources(Path(prior_run))
    engine = runpy.run_path(str(HERE / "flux_prompt_factorial.py"))
    base = {key: value for key, value in prior_request["config"].items()
            if key.startswith("expected_") or key in ("model", "model_path", "sample_indices", "image_sample_indices")}
    config = engine["make_config"](base)
    config["schema"] = "mdb-flux-prompt-factorial-plan-v1"
    config["hypotheses"] = {
        "H-recurrence": "Four fresh seed127 prompt contexts share exact actualZ0 and qualified parameters/profile; all448 predicted stages use their own prior state, not native future activations.",
        "H-prospective": "Seal all four112 trajectories and cohort before all four actual predicted PNGs; all four PNGs precede every native denoiser target entry. Then execute four native112 histories and terminalsham.",
        "H-contrasts": "Freeze full latent/RGB outcomes, then measure two color contrasts, two order contrasts and factorial interaction. Compare predicted versus native changes at fixed raw coordinates; preserve signed errors and undefined zero-denominator ratios.",
        "H-semantics": "Expect supplied color assignment and left/right relation to appear in coarse scene observations. This is diagnostic, not a quality gate; retain model misrenderings and numerical counterexamples.",
        "H-coverage": "Full predicted hashes at every transition; full native errors only at declared landmarks/finals, shape-fixed64witnesses elsewhere. Nonlandmark native manifests are not hydratable replay."}
    engine["validate_config"](config)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    worker = HERE / "flux_prompt_factorial.py"
    extras = [(source, name) for name, source in workers.items()]
    extras.append((run_dir / "plan.json", "plan.json"))
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *workers, "plan.json", preflight.name],
        "path_token_allowlist": [engine["RESULT"], config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-prompt-factorial-request-v1", "config": config,
        "scheduler_url": B["scheduler_endpoint"](), "launcher_source_sha256": sha(Path(__file__)),
        "launcher_helper_sha256": BASE_LAUNCHER_SHA256, "task_family": "forward", "model": config["model"],
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "needs": {"cuda": True}, "execution_environment": B["execution_intent"](),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "source_authorities": {"prior_request_sha256": prior_request["request_sha256"],
            "legacy_sources": {name: sha(path) for name, path in workers.items()}},
        "independent_math_sources": {name: hashlib.sha256(value).hexdigest() for name, value in independent.items()},
        "independent_source_authorities": authorities,
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=9, logical_branches=9,
            logical_checkpoint_replays=9, max_local_work_items=64,
            checkpoint_policy="One pipeline, four encodings, one shared freshZ0. Four sealed own112 forecasts and actual PNGs before all native denoising; four native histories and one terminalsham in-process. No per-condition jobs or automatic retries."),
        "resource_exception": {"ram_mb": 14336, "vram_mb": 13600, "disk_gb": 12, "cpu_threads": 4,
            "est_wall_s": 2500, "outer_timeout_s": 3300, "priority": 50,
            "calibration_job": "job-4350531b1dcd", "measured_peak_rss_mb": 8509.1, "measured_peak_vram_mb": 12936,
            "reason": "Measured512px B1 CUDA BF16 initial-noise program; sequential cells retain same one-bank working geometry. Four forecasts448math and four native448 transitions, sparse full landmark arrays plus every-stage descriptors. No persistent parameter copies. Long-run review and installed tiny storage/census qualification required before admission."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    freeze = run_dir / "independent-math-freeze"
    freeze.mkdir()
    for name, content in independent.items():
        with (freeze / name).open("xb") as stream:
            stream.write(content)
    return {"submitted": False, "run_dir": str(run_dir), "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch

    run_dir = Path(run_dir).resolve()
    if (run_dir / "submission.json").exists():
        raise ValueError("finite campaign already admitted")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or sha(Path(__file__)) != request["launcher_source_sha256"]
            or any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items())
            or any(sha(run_dir / "independent-math-freeze" / name) != digest for name, digest in request["independent_math_sources"].items())
            or read(run_dir / "plan.json") != request["config"]):
        raise ValueError("staged source/plan/independent reader changed")
    review_path = run_dir.parent / "long-run-review-final.json"
    review = read(review_path)
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("source-bound independent long-run review required")
    B["scheduler_endpoint"](request)
    resources = request["resource_exception"]
    worker_command = command(request)
    save(run_dir / "admission-command.json", {"command": worker_command, "request_sha256": request["request_sha256"],
        "launcher_source_sha256": sha(Path(__file__)), "long_run_review_sha256": sha(review_path)})
    note = ("Klein4B CUDA BF16 B1 512 seed127: four prompt color/order cells share actualZ0. "
        "Seal448 own-state math transitions and save4VAE PNGs BEFORE any native target. "
        "Then448native/5Acts/453receipts/9VAE; one load4encodes1noise. "
        "CPU4 RAM14336 VRAM13600 disk12GB; expected2500/worker3000/outer3300s,no retries. "
        "Allstage hashes/witnesses; fullfields at landmarks only;169checkpointrecipes. Preserve misses.")
    if len(note) > 500:
        raise ValueError("queue note oversized")
    job = guarded_mrun_launch(worker_command, request=request, model=request["model"], task="forward",
        experiment="mdb-flux-prompt-factorial", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=resources["ram_mb"], vram_mb=resources["vram_mb"],
        disk_gb=resources["disk_gb"], cpu_threads=resources["cpu_threads"], est_wall_s=resources["est_wall_s"],
        timeout_s=resources["outer_timeout_s"], priority=resources["priority"], retry_on_kill=False, preflight=True, note=note)
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--submit-staged", action="store_true")
    args = parser.parse_args()
    if not args.submit_staged and args.prior_run is None:
        parser.error("--prior-run is required while staging")
    print(json.dumps(submit_staged(args.run_dir) if args.submit_staged else stage(args.run_dir, prior_run=args.prior_run)))
