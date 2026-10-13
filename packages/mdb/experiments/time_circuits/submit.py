"""Source-sealed paper reproductions using installed MDB and one mrun lease."""
import argparse
import base64
import gzip
import json
import tempfile
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from mdb.execution_environment import intent
from mdb.job_support import (
    guarded_mrun_launch,
    one_lease_scheduling_contract,
    source_hash_manifest,
    terminal_job,
)
from mdb.payload import package_files, stage_worker, worker_command
from mdb.tools import scheduler_api

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results/time-circuits"
MARKER = "MDB_PAPER_REPORT_GZIP_B64="
E6_PYTHON = "/mnt/usb-storage/e6-circuit-tracer/venv2/bin/python"
E6_BASE = "/mnt/usb-storage/e6-circuit-tracer"
GEMMA = E6_BASE + "/hf/hub/models--google--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"
IDENTITIES = ["fox", "cat", "dog", "wolf", "bear", "lion", "tiger", "deer", "horse", "rabbit", "mouse", "owl", "pig", "sheep"]
COLORS = ["red", "blue", "green", "black", "white", "brown", "pink", "orange", "purple", "yellow"]
ADMITTED_COLORS = ["red", "blue", "green", "brown", "pink", "orange", "purple", "yellow"]


def specification(experiment, smoke, e6_mode="interchange", unfreeze_attention=False, allow_nondeterminism=False, assay_id=None):
    if allow_nondeterminism and (experiment != "e6" or not smoke):
        raise ValueError("native repeat-variability mode is a two-item E6 smoke diagnostic")
    payload = {"smoke": smoke, "output_root": "/mnt/hdd8tb/mdb/time-circuits"}
    profile = {"ram_mb": 12000., "vram_mb": 12000.}
    if experiment == "e20":
        model = "qwen2.5-1.5b"
        payload.update(model_key="qwen15", seed_layer=12, mediator_layers=list(range(22, 28)),
            outside_layers=list(range(16, 22)),
            expected_weights_sha256="a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796")
        extras = [(HERE / "e20_bridge.py", "e20_bridge.py"),
                  (HERE / "vendor/paper_e20.py", "vendor/paper_e20.py"),
                  (HERE / "vendor/run_sae_comparison.py", "vendor/run_sae_comparison.py")]
        command = worker_command("worker_e20.py")
        basis = "paper job-e1f652e80f43: peak RSS 6761.6 MB, GPU 9412 MB; native FP32 plus durable MDB state headroom"
        environment = intent()
    else:
        model = None  # exact host-local instrument checkpoint, not a registry family substitution
        payload.update(items=[{"behavior": b, "context_idx": 0, "subject": s, "candidates": candidates}
            for b, subjects, candidates in (("identity", IDENTITIES[:1 if smoke else 12], IDENTITIES),
                                           ("color", ADMITTED_COLORS[:1 if smoke else 8], COLORS)) for s in subjects],
            ks=[5, 20, 50], offload="cpu", max_feature_nodes=4096, attr_batch_size=64,
            n_rand_seeds=3, model_name="google/gemma-2-2b", transcoder_set="gemma",
            model_local_path=GEMMA, backend="nnsight", e6_mode=e6_mode, freeze_attention=not unfreeze_attention,
            deterministic_algorithms=not allow_nondeterminism, reference_repeats=2 if smoke else 1,
            expected_weights_sha256="40f7727761523db40b475358377c9a9b0f0d8fcf7ef8b869e71ae4f0ef12a555",
            circuit_tracer_commit="7f66876689f59e92fc3641650d38b8bd41749ec4")
        extras = [(HERE / "vendor/paper_e6.py", "vendor/paper_e6.py"),
                  (HERE / "e6_bridge.py", "e6_bridge.py"),
                  (HERE / "e6_bootstrap.py", "e6_bootstrap.py")]
        command = ["/usr/bin/env", "PYTHONUNBUFFERED=1", "PYTHONNOUSERSITE=1", "PYTHONPATH=",
            "HF_HOME=" + E6_BASE + "/hf", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
            "TOKENIZERS_PARALLELISM=false", "SATURN_RESIDENT_WORKER=1",
            "SATURN_RUN_CONFIG=saturn_run_config.json", "SATURN_DEBUG_REQUEST=saturn_debug_request.json",
            "TMPDIR=" + E6_BASE + "/tmp", "XDG_CACHE_HOME=" + E6_BASE + "/.xdg-cache",
            "TORCH_HOME=/mnt/big/llm-models/.torch-cache", E6_PYTHON, "-I", "-u", "e6_bootstrap.py"]
        basis = "paper interchange job-03225ce474d1: peak RSS 8608.4 MB, GPU 9584 MB; MDB input cut headroom"
        environment = None  # separately pinned legacy instrument toolchain, reported by worker
    worker = HERE / f"worker_{experiment}.py"
    contexts = 1 if smoke else 3
    operations = contexts if experiment == "e20" else len(payload["items"])
    consumer_calls = contexts * 10 if experiment == "e20" else len(payload["items"]) * (70 if e6_mode == "interchange" else 38)
    replays = contexts if experiment == "e20" else len(payload["items"]) * 2
    sources = {name + "/" + relative: path for name in ("mdb", "mrun")
               for relative, path in package_files(name).items()}
    sources[worker.name] = worker
    sources.update({name: path for path, name in extras})
    config = {"payload": payload}
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-paper-reproduction-request-v1", "experiment": "mdb-paper-" + experiment,
        "assay_id": assay_id or uuid4().hex,
        "task_family": "forward", "device": "cuda-fp32" if experiment == "e20" else "cuda-bf16",
        "needs": {"cuda": True}, "payload": payload, "source_hashes": source_hash_manifest(sources),
        "config_bytes_sha256": sha256(raw).hexdigest(), "resource_profile": profile,
        "resource_basis": basis, "timeout_s": 840.,
        "work_geometry": {"logical_operation_unit": "one fixed context/prompt panel",
                          "prompt_panels": operations, "mdb_consumer_calls": consumer_calls,
                          "local_cut_replays": replays},
        "estimated_worker_seconds": 90 if experiment == "e20" or smoke else (650 if e6_mode == "interchange" else 500),
        "timing_basis": ("E20 full job-a7c0623183af: 34.34 worker seconds for 42 items"
                         if experiment == "e20" else
                         ("E6 deterministic interchange smoke job-715744154f27: 73.7 seconds; full job-f953178a6b6e: 658.5 seconds"
                          if e6_mode == "interchange" else
                          "E6 deterministic ablation smoke job-445dd7e71280: 54.2 seconds for two items with two native references; full uses one native reference per arm")),
        "execution_environment": environment, "scheduler_endpoint": scheduler_api()._resolve(),
        "scheduling_contract": one_lease_scheduling_contract(
            checkpoint_policy="one resident paper checkpoint; immutable native-input cuts, private interventions and replay; no retries",
            logical_operations=operations, logical_checkpoint_replays=replays)}
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    return config, raw, request, worker, extras, command, model


def collect(job):
    job_id, record, logs = terminal_job(job, wait=False)
    output = RESULTS / job_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "job.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "worker.log").write_text(logs)
    environment = next((json.loads(line.split("=", 1)[1]) for line in logs.splitlines()
                        if line.startswith("MDB_EXECUTION_ENVIRONMENT=")), None)
    if environment:
        (output / "execution-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    encoded = next((line[len(MARKER):] for line in logs.splitlines() if line.startswith(MARKER)), None)
    if encoded:
        report = json.loads(gzip.decompress(base64.b64decode(encoded)))
        request = json.loads((output / "request.json").read_text())
        if report["mdb"]["request_sha256"] != request["request_sha256"] or report["mdb"]["source_hashes"] != request["source_hashes"]:
            raise RuntimeError("paper reproduction source/request identity mismatch")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"job": job_id, "state": record["state"], "report": bool(encoded), "output": str(output)}))
    return 0 if encoded and record["state"] == "succeeded" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "submit", "collect"))
    parser.add_argument("experiment", choices=("e6", "e20"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--e6-mode", choices=("interchange", "ablation"), default="interchange")
    parser.add_argument("--unfreeze-attention", action="store_true")
    parser.add_argument("--allow-nondeterminism", action="store_true", help="retain the historical native repeat-variability lane")
    parser.add_argument("--assay-id", help="explicit instance identity; use the same value for preflight and submission, a new value for a new assay")
    args = parser.parse_args()
    if args.action == "collect":
        return collect(args.job)
    if args.allow_nondeterminism and (args.experiment != "e6" or not args.smoke):
        parser.error("--allow-nondeterminism requires e6 --smoke")
    config, raw, request, worker, extras, command, model = specification(
        args.experiment, args.smoke, args.e6_mode, args.unfreeze_attention, args.allow_nondeterminism, args.assay_id)
    if args.action == "preflight":
        print(json.dumps({"request": request, "command": command}, sort_keys=True))
        return 0
    profile = request["resource_profile"]
    with tempfile.TemporaryDirectory(prefix="mdb-paper-") as directory:
        stage_worker(directory, worker=worker, config=config, config_bytes=raw, request=request,
                     additional_files=extras)
        command = [command[0], "MRUN_SERVER_URL=" + request["scheduler_endpoint"],
                   "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command[1:]]
        job = guarded_mrun_launch(command, request=request, experiment=request["experiment"], task="forward",
            model=model, payload=Path(directory), prefer_host="beast", pin="beast" if args.experiment == "e6" else None,
            gpu=True, needs={"cuda": True}, ram_mb=profile["ram_mb"], vram_mb=profile["vram_mb"],
            cpu_threads=4, timeout_s=request["timeout_s"], est_wall_s=request["estimated_worker_seconds"],
            detach=True, retry_on_kill=False, preflight=True,
            note=f"MDB paper {args.experiment} {'smoke' if args.smoke else 'full'}: fixed native protocol; private tensor cuts; same-parent replay; one checkpoint, sequential CUDA, no retries; stop after fixed panel.")
    output = RESULTS / job
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (output / "config.json").write_bytes(raw)
    print(json.dumps({"job": job, "experiment": args.experiment, "smoke": args.smoke}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
