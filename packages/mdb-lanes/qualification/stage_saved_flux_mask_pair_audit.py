"""Paired full-array CPU audit launcher with an explicit measured-work allowance.

Derived from frozen shared launcher 26b031478922a330d85f010bc9cc059e92072bbaf3aa5b0324e41a80a568aef9.
Keep the shared source unchanged for other campaigns. The paired full-array and
four-lane response work uses240s estimated and600s outer, with no model calls.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bind(environment):
    runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))["bind"](environment)


def stage(run, study, *, worker, helpers, experiment, scope, operations):
    from mdb_lanes.job_support import one_lease_scheduling_contract, scheduler_api
    from mdb_lanes.payload import stage_worker

    run, study, worker = Path(run).resolve(), Path(study).resolve(), Path(worker).resolve()
    parent = read(study / "request.json")
    bind(parent["execution_environment"])
    job_id = read(study / "submission.json")["job_id"]
    status, raw, _ = scheduler_api().request("GET", "/api/jobs/" + job_id, timeout_s=30)
    job = json.loads(raw)
    if status != 200 or job["state"] != "succeeded" or job["result"]["diagnostics"]["command"] != read(study / "admission-command.json")["command"]:
        raise ValueError("completed exact native command required")
    collector = HERE / "collect_flux_initial_noise.py"
    if sha(collector) != "e59cb6e3c1f4f0b03f0163f4db8544952def504090a3fdded8e61f530880b9ad":
        raise ValueError("qualified actual-job validator changed")
    runpy.run_path(str(collector))["validate_job"](job, job_id, parent)
    status, log, _ = scheduler_api().request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected = job["result"]["artifacts"]["job_log"]
    if status != 200 or len(log) > 4 << 20 or len(log) != expected["bytes"] or hashlib.sha256(log).hexdigest() != expected["sha256"]:
        raise ValueError("native log checksum differs")
    markers = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    if len(markers) != 1:
        raise ValueError("one exact native directory publication required")
    root = Path(parent["config"]["output_scratch"]) / job_id / parent["config"].get("output_name", "binding-minimal-result")
    script = "import json,hashlib\nfrom pathlib import Path\np=Path(" + repr(str(root)) + ")\nprint(json.dumps(dict(manifest_sha256=hashlib.sha256((p/'files.json').read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((p/'summary.json').read_bytes()).hexdigest(),summary=json.loads((p/'summary.json').read_text()))))\n"
    info = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"], input=script, text=True, capture_output=True, check=True, timeout=60).stdout)
    marker = markers[0]
    if (info["summary"]["job_id"] != job_id or info["summary"]["request_sha256"] != parent["request_sha256"]
            or marker["job_id"] != job_id or marker["request_sha256"] != parent["request_sha256"]
            or marker["output_root"] != str(root) or marker["files_manifest_sha256"] != info["manifest_sha256"]
            or marker["summary_sha256"] != info["summary_sha256"]):
        raise ValueError("native publication and directory identity differ")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "source-job.json", job)
    (run / "source-worker.log").write_bytes(log)
    if not (study / "job.json").exists():
        save(study / "job.json", job)
    if not (study / "worker.log").exists():
        (study / "worker.log").write_bytes(log)
    elif sha(study / "worker.log") != hashlib.sha256(log).hexdigest():
        raise ValueError("previous native log copy differs")
    config = {"native_root": str(root), "native_manifest_sha256": info["manifest_sha256"],
        "native_request_sha256": parent["request_sha256"], "source_job": job_id,
        "output_scratch": str(root.parents[1]), "directory_custody": True, "scope": scope}
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in helpers]
    if Path(__file__).name not in helpers:
        extras.append((Path(__file__), Path(__file__).name))
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *[name for _, name in extras], preflight.name],
        "path_token_allowlist": [str(root), config["output_scratch"], "binding-minimal-result"]})
    extras.append((preflight, preflight.name))
    request = {"schema": "mdb-saved-flux-audit-request-v1", "config": config,
        "experiment": experiment, "execution_environment": parent["execution_environment"],
        "needs": {"cpu": True, "host": "beast"}, "task_family": "analysis", "backend": "numpy", "device": "cpu", "dtype": "float64",
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "runtime_wheels": parent["runtime_wheels"], "controller_sha256": sha(__file__),
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=operations,
            logical_branches=1, logical_checkpoint_replays=0, max_local_work_items=max(32, operations),
            extra={"task_family": "analysis"}, checkpoint_policy="One full-array saved-only CPU audit; compact catalog/retriever, no native/reference recurrence or retries."),
        "resource_exception": {"ram_mb": 3072, "vram_mb": 0, "cpu_threads": 2, "disk_gb": 1, "est_wall_s": 240}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, additional_files=extras)
    for name, digest in request["runtime_wheels"].items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("audit wheel cohort differs")
    return {"request_sha256": request["request_sha256"], "source_job": job_id}


def submit(run, *, worker):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    run, worker = Path(run).resolve(), Path(worker).resolve()
    request = read(run / "request.json")
    bind(request["execution_environment"])
    if (run / "submission.json").exists():
        raise ValueError("audit already submitted")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or sha(__file__) != request["controller_sha256"]):
        raise ValueError("audit request/controller changed")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("audit payload changed")
    args = worker_command(worker.name)
    args.insert(1, "CUDA_VISIBLE_DEVICES=")
    args.insert(args.index("python") + 1, "-B")
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"]})
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment=request["experiment"], task="analysis", model=None,
        payload=run / "payload", needs=request["needs"], gpu=False, prefer_host="beast", timeout_s=600,
        retry_on_kill=False, preflight=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s")},
        note="Saved-only full-array audit; " + request["config"]["scope"] + " CPU2/RAM3GB,240sestimate/600souter;zero model/reference recurrence; compact reports/retriever only.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}

