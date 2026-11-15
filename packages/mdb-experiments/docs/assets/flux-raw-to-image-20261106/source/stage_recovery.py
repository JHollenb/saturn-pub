"""Stage or submit one explicit CPU recovery of retained failed-producer bytes."""
import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HERE = ROOT.parents[1] / "saturn-pub/packages/mdb/qualification"
WORKER = "recover_flux_raw_to_image.py"
AUDITOR = "audit_flux_raw_to_image.py"
AUDITOR_SHA = "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835"
FINALIZER = "flux_image_tail_microscope.py"
FINALIZER_SHA = "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(value):
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def command(request):
    from mdb.execution_environment import intent
    from mdb.payload import worker_command
    environment = request["execution_environment"]
    for key, name in {
        "uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
        "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
        "uv_link_mode": "MDB_WORKER_UV_LINK_MODE",
    }.items():
        os.environ[name] = environment[key]
    installed = intent()
    if installed != {key: environment[key] for key in installed}:
        raise ValueError("Installed recovery dependency intent changed")
    args = worker_command(WORKER)
    args.insert(1, "CUDA_VISIBLE_DEVICES=")
    args.insert(args.index("python") + 1, "-B")
    return args


def stage(run):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    parent = read(ROOT / "run-01/request.json")
    packet_root = ROOT / "recovery-inputs-v1"
    packet = read(packet_root / "packet-receipt.json")
    if identity(parent) != parent["request_sha256"] or packet["request_sha256"] != parent["request_sha256"]:
        raise ValueError("Original request packet differs")
    for name, row in packet["files"].items():
        if sha(packet_root / name) != row["sha256"] or (packet_root / name).stat().st_size != row["bytes"]:
            raise ValueError("Captured original packet differs")
    if sha(HERE / AUDITOR) != AUDITOR_SHA or sha(HERE / FINALIZER) != FINALIZER_SHA:
        raise ValueError("Frozen recovery reader/finalizer differs")
    auditor = runpy.run_path(str(HERE / AUDITOR))
    helper_pins = {**auditor["PINS"], AUDITOR: AUDITOR_SHA, FINALIZER: FINALIZER_SHA}
    for name, expected in helper_pins.items():
        if sha(HERE / name) != expected:
            raise ValueError("Frozen reader dependency differs")
    config = {
        "schema": "mdb-flux-raw-packaging-recovery-v1",
        "original_root": packet["original_root"],
        "retained_inventory": "retained-inventory.json",
        "retained_inventory_sha256": packet["files"]["retained-inventory.json"]["sha256"],
        "original_job": "original-job.json", "original_job_sha256": packet["files"]["original-job.json"]["sha256"],
        "original_log": "original-worker.log", "original_log_sha256": packet["files"]["original-worker.log"]["sha256"],
        "original_request": "original-request.json", "original_request_sha256": packet["files"]["original-request.json"]["sha256"],
        "expected_source_job_id": packet["job_id"], "expected_source_request_sha256": parent["request_sha256"],
        "expected_source_worker_sha256": parent["config"]["worker_sha256"],
        "expected_source_payload_sha256": packet["payload_sha256"],
        "finalizer": FINALIZER, "finalizer_sha256": FINALIZER_SHA,
        "auditor": AUDITOR, "auditor_sha256": AUDITOR_SHA,
        "output_scratch": "/mnt/big/mdb/research", "view_name": "raw-to-image-recovery-view",
        "max_member_bytes": 134221824, "max_expanded_bytes": 4 << 30,
        "scope": "Original failed producer retained unchanged. New hardlink view, bounded finalization and saved-only audit. Zero neural/model calls; shared-primitive whole-image outcome awaits audit.",
    }
    run.mkdir(exist_ok=False)
    save(run / "plan.json", config)
    save(run / "preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [WORKER, *helper_pins, *packet["files"], "preflight.v1.json"],
        "path_token_allowlist": [config["original_root"], config["output_scratch"],
            config["view_name"], "binding-minimal-result"]})
    files = {WORKER: HERE / WORKER, **{name: HERE / name for name in helper_pins},
        **{name: packet_root / name for name in packet["files"]},
        "preflight.v1.json": run / "preflight.v1.json"}
    request = {
        "schema": "mdb-flux-raw-recovery-request-v1", "config": config,
        "execution_environment": parent["execution_environment"], "scheduler_url": parent["scheduler_url"],
        "runtime_wheels": parent["runtime_wheels"],
        "task_family": "analysis", "backend": "saved-only-numpy", "device": "cpu", "dtype": "float64",
        "needs": {"cpu": True, "host": "beast"},
        "source_hashes": {name: sha(path) for name, path in files.items()}, "launcher_sha256": sha(__file__),
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True,
            logical_branches=1, logical_operations=1, max_local_work_items=32,
            extra={"task_family": "analysis"},
            checkpoint_policy="One failed-source packaging recovery and complete saved-only audit; hardlinks only, original immutable, no neural recurrence or automatic retry."),
        "resource_exception": {"ram_mb": 3072, "vram_mb": 0, "cpu_threads": 2,
            "disk_gb": 4, "est_wall_s": 180, "timeout_s": 600,
            "calibration": "Prior remaining-prefix saved audit:172.97s/1256.7MB for1960fields; this980denoiser+29Qwen+2endpoints and16GB streamed recipe identity. New view shares original inodes; no raw tensor byte copy."},
    }
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
        additional_files=[(path, name) for name, path in files.items() if name != WORKER])
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("Staged recovery payload differs")
    save(run / "staged-command.json", {"command": command(request), "submitted": False})
    return {"request_sha256": request["request_sha256"], "submitted": False}


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    request = read(run / "request.json")
    if (run / "submission.json").exists() or identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("Recovery request/launcher changed or already submitted")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("Frozen recovery payload differs")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"]})
    resources = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-raw-recovery-audit",
        task="analysis", model=None, payload=run / "payload", gpu=False, needs=request["needs"],
        prefer_host="beast", retry_on_kill=False, preflight=True,
        **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        note="Failed-source packaging recovery:1337files/2.62GB, hardlink view/zero raw copies,128MiB+4096B member allowance; original unchanged/failed. Full saved-array and709recipe audit; CPU2/RAM3GB,180s estimate/600s outer; zero model calls/no retries.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve()), sort_keys=True))
