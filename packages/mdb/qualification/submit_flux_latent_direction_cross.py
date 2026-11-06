"""Stage, review, admit and authenticate six CPU-only mathematical futures."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = runpy.run_path(str(HERE / "submit_flux_matched_pass.py"))
L, C = M["L"], M["C"]
sha, read, save = (M[k] for k in ("sha", "read", "save"))
OUTPUT = "latent-direction-math-result"
WORKER = "flux_latent_direction_cross.py"


def command(request):
    result = L["command"](request)
    if result[-1] != "audit_flux_prompt_factorial.py":
        raise ValueError("qualified CPU bootstrap changed")
    result[-1] = WORKER
    return result


def stage(run, parent, audit, expectations):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, parent, audit, expectations = map(lambda p: Path(p).resolve(), (run, parent, audit, expectations))
    # Reuse qualified source acquisition, original-array custody and audit joins.
    # This sibling directory is staging metadata only, never an admitted job.
    seed = run.with_name(run.name + ".source-stage-v1")
    M["stage"](seed, parent, audit)
    base = read(seed / "request.json")
    run.mkdir(parents=True, exist_ok=False)
    config = {**base["config"], "schema": "mdb-latent-direction-math-plan-v1",
        "mathematical_transitions": 504, "logical_steps": [1, 2, 3], "branches": 6,
        "max_wall_seconds": 7200, "parameter_codec": "float32", "max_decoded_bytes": 18 << 30,
        "expectations_sha256": sha(expectations),
        "scope": "Six frozen-equation own-state forecasts from known native N1 sources, crossed old P1−N1 directions. Four edits plus two shams; low own input is known. No native model forwards or outcome fitting."}
    config.pop("logical_step", None)
    config.pop("max_retained_bytes", None)
    save(run / "plan.json", config)
    worker = HERE / WORKER
    sources = {name: seed / "payload" / name for name in base["source_hashes"]
        if name not in ("flux_matched_pass_worker.py", "plan.json", "preflight.v1.json")}
    sources.update({WORKER: worker, "submit_flux_matched_pass.py": HERE / "submit_flux_matched_pass.py",
        "expectations.json": expectations, "plan.json": run / "plan.json"})
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [*sources, preflight.name],
        "path_token_allowlist": [config["parent_remote_result"], config["checkpoint_root"], "parent-run", "parent-audit",
            "audit-parameter-cache", OUTPUT, "expectations.json"]})
    sources[preflight.name] = preflight
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    new = {**base, "schema": "mdb-latent-direction-math-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "source_hashes": {n: sha(p) for n, p in sources.items()},
        "runtime_wheels": read(seed / "payload/payload-seal.json")["wheels"], "launcher_sha256": sha(__file__),
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=6,
            logical_branches=6, logical_checkpoint_replays=0,
            checkpoint_policy="One CPU-only lease, six independent84-transition mathematical suffixes; all sealed before native experiment. No retries.",
            extra={"task_family": "analysis", "native_framework_forwards": 0, "mathematical_transitions": 504}),
        "resource_exception": {"ram_mb": 28672, "vram_mb": 0, "cpu_threads": 4, "disk_gb": 12,
            "est_wall_s": 2700, "timeout_s": 7500,
            "calibration": "FP32 read-only original parameter cache: accepted448-transition audit04 completed1801.58worker seconds, peakRSS16639.8MB,169decodes/2535hits.504transitions share immutable original7.75GBcheckpoint/15.50GBdecoded constants;28672MB reserves transient/headroom. No controlledspeedup claim. Conservative7200sworker/7500souter; strictpreflight and noCUDA reservation."}}
    new.pop("request_sha256", None)
    new["request_sha256"] = hashlib.sha256(json.dumps(new, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", new)
    stage_worker(run / "payload", worker=worker, config=config, request=new, config_bytes=raw,
        additional_files=[(p, n) for n, p in sources.items() if n != WORKER])
    if read(run / "payload/payload-seal.json")["wheels"] != new["runtime_wheels"]:
        raise ValueError("installed wheel bytes changed during staging")
    validate_payload(run, new)
    return {"submitted": False, "request_sha256": new["request_sha256"]}


def validate_payload(run, request):
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or request["launcher_sha256"] != sha(__file__)
            or any(sha(run / "payload" / n) != d for n, d in request["source_hashes"].items())
            or any(sha(run / "payload" / n) != d for n, d in request["runtime_wheels"].items())):
        raise ValueError("staged request/source/runtime identity differs")
    for name in ("flux_matched_pass_replay.py", "collect_flux_prompt_factorial.py", "audit_flux_prompt_factorial.py"):
        runpy.run_path(str(run / "payload" / name))
    runpy.run_path(str(run / "payload" / WORKER))["helpers"]()


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    if (run / "submission.json").exists():
        raise ValueError("CPU prediction already admitted")
    request = read(run / "request.json")
    validate_payload(run, request)
    review = read(run / "long-run-review-final.json")
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("exact source-bound long-run review required")
    L["B"]["scheduler_endpoint"](request)
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"],
        "review_sha256": sha(run / "long-run-review-final.json")})
    r = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-latent-direction-math", task="analysis", model=None,
        payload=run / "payload", pin="beast", gpu=False, needs={}, ram_mb=r["ram_mb"], vram_mb=0,
        cpu_threads=4, disk_gb=12, est_wall_s=r["est_wall_s"], timeout_s=r["timeout_s"], priority=50,
        retry_on_kill=False, preflight=True,
        note="Receiver×direction cross:6own-state mathematical suffixes/504transitions,0nativeforwards. CPU4/RAM28672MB,FP32immutableparametercache,12GBscratch. KnownN1/P1inputstates only; sealall6beforeseparateCUDA.2700s estimate/7200worker7500outer, no retries; remove ownweightsafterverifiedforecast.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


def collect(run):
    from mdb.job_support import scheduler_api

    run = Path(run).resolve()
    request = read(run / "request.json")
    L["B"]["scheduler_endpoint"](request)
    job_id = read(run / "submission.json")["job_id"]
    bodies = {}
    for suffix, name in (("", "job.json"), ("/events", "events.json"), ("/logs?offset=0&wait_s=1", "worker.log")):
        status, body, _ = scheduler_api().request("GET", "/api/jobs/" + job_id + suffix, timeout_s=30)
        if status != 200 or len(body) > 24 << 20:
            raise ValueError("actual scheduler evidence missing or oversized")
        bodies[name] = body
    job = json.loads(bodies["job.json"])
    remote = str(Path(C["validate_job"](job, job_id, request)).parent / OUTPUT)
    expected_log = job["result"]["artifacts"]["job_log"]
    if (job["state"] != "succeeded" or job["result"]["diagnostics"]["command"] != read(run / "admission-command.json")["command"]
            or len(bodies["worker.log"]) != expected_log["bytes"]
            or hashlib.sha256(bodies["worker.log"]).hexdigest() != expected_log["sha256"]):
        raise ValueError("actual job/command/log differs")
    raw, capsule = L["decode_capsule"](bodies["worker.log"].decode())
    report = json.loads(raw)
    execution = report["near_data_execution"]
    if (report["schema"] != "mdb-latent-direction-math-execution-v1" or execution["job_id"] != job_id
            or execution["request_sha256"] != request["request_sha256"]
            or execution["worker_sha256"] != request["source_hashes"][WORKER]
            or execution["mathematical_transitions"] != 504 or execution["native_framework_forwards"] != 0):
        raise ValueError("CPU execution identity differs")
    for name, body in {**bodies, "result.json": raw}.items():
        C["preserve"](run / name, body)
    script = C["remote_script"](remote)
    def scan():
        result = subprocess.run(["rtk", "proxy", "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "beast", "python3", "-"],
            input=script, text=True, capture_output=True, check=True, timeout=600)
        if len(result.stdout) > 16 << 20:
            raise ValueError("remote inventory oversized")
        return json.loads(result.stdout)
    before = scan()
    save(run / "collection-remote-before.json", before)
    if shutil.disk_usage(run).free < before["expanded_bytes"] + (128 << 20):
        raise ValueError("insufficient local collection space")
    destination = run / "extracted" / OUTPUT
    destination.mkdir(parents=True, exist_ok=False)
    allowlist = run / "collection-files.bin"
    C["preserve"](allowlist, b"".join(str(C["relative"](n)).encode() + b"\0" for n in before["files"]))
    copied = subprocess.run(["rtk", "proxy", "rsync", "--recursive", "--times", "--protect-args", "--no-links", "--no-devices", "--no-specials",
        "--from0", "--files-from=" + str(allowlist), "--max-size=" + str(C["MAX_MEMBER"]), "--timeout=120",
        "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes", "beast:" + remote + "/", str(destination) + "/"],
        text=True, capture_output=True, check=True, timeout=1800)
    save(run / "collection-copy.json", {"returncode": copied.returncode, "stdout": copied.stdout, "stderr": copied.stderr})
    after, local = scan(), C["inventory"](destination)
    if before != after or before != local:
        raise ValueError("remote-before/after/local inventories differ")
    manifest = read(destination / "remote-manifest.json")
    if (sha(destination / "remote-manifest.json") != report["remote_manifest_sha256"]
            or manifest["job_id"] != job_id or manifest["request_sha256"] != request["request_sha256"]
            or manifest["files"] != {k: v for k, v in local["files"].items() if k != "remote-manifest.json"}
            or read(destination / "summary.json") != report["report"]):
        raise ValueError("actual worker manifest/summary differs")
    save(run / "collection-remote-after.json", after)
    save(run / "collection-local-inventory.json", local)
    names = ("request.json", "job.json", "events.json", "worker.log", "result.json", "admission-command.json",
        "collection-remote-before.json", "collection-remote-after.json", "collection-local-inventory.json", "collection-copy.json", "collection-files.bin")
    receipt = {"schema": "mdb-latent-direction-math-collection-v1", "job_id": job_id, "request_sha256": request["request_sha256"],
        "capsule": capsule, "remote_path": remote, "file_count": local["file_count"], "expanded_bytes": local["expanded_bytes"],
        "archive_sha256": None, "native_framework_forwards": 0, "collector_sha256": sha(__file__), "bindings": {n: sha(run / n) for n in names}}
    save(run / "collection.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--parent", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--expectations", type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--collect", action="store_true")
    a = p.parse_args()
    if sum((a.parent is not None, a.submit, a.collect)) != 1 or bool(a.parent) != bool(a.audit) or bool(a.parent) != bool(a.expectations):
        p.error("choose --parent RUN --audit RUN --expectations JSON, --submit, or --collect")
    print(json.dumps(collect(a.run) if a.collect else submit(a.run) if a.submit else stage(a.run, a.parent, a.audit, a.expectations)))
