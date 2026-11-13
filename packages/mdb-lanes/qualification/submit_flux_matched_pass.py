"""Stage, admit and collect a finite retrospective native-input pass comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
L = runpy.run_path(str(HERE / "submit_flux_prompt_factorial_audit.py"))
C = runpy.run_path(str(HERE / "collect_flux_prompt_factorial.py"))
sha, read, save = (L[key] for key in ("sha", "read", "save"))
OUTPUT = "matched-pass-output"
RESOURCES = {"ram_mb": 14336, "vram_mb": 0, "cpu_threads": 4, "disk_gb": 10,
    "est_wall_s": 900, "timeout_s": 2100}


def validate_accepted_audit(parent, audit):
    request, receipt, job, report = [read(audit / name) for name in ("request.json", "collection.json", "job.json", "audit.json")]
    C["validate_job"](job, job["job_id"], request)
    if (job["state"] != "succeeded" or receipt["job_id"] != job["job_id"]
            or receipt["request_sha256"] != request["request_sha256"]
            or job["result"]["diagnostics"]["command"] != read(audit / "admission-command.json")["command"]
            or any(sha(audit / name) != digest for name, digest in receipt["bindings"].items())):
        raise ValueError("accepted audit collection identity differs")
    raw, capsule = L["decode_capsule"]((audit / "worker.log").read_text())
    expected_log = job["result"]["artifacts"]["job_log"]
    if (raw != (audit / "audit.json").read_bytes() or capsule != receipt["capsule"]
            or sha(audit / "worker.log") != expected_log["sha256"]
            or (audit / "worker.log").stat().st_size != expected_log["bytes"]):
        raise ValueError("accepted audit is not its actual scheduler-log report")
    execution = report["near_data_execution"]
    parent_request = read(parent / "request.json")
    if (report["mechanics_status"] != "valid" or report["auditor_source_sha256"] !=
            parent_request["independent_math_sources"]["analyze_flux_prompt_factorial.py"]
            or execution["parent_request_sha256"] != parent_request["request_sha256"]
            or execution["audit_job_id"] != job["job_id"] or execution["native_framework_forwards"] != 0
            or not all(row["complete_field_hashes_match"] for row in report["numeric"].values())):
        raise ValueError("source-bound full reconstruction has not been established")
    return job["job_id"], sha(audit / "audit.json")


def command(request):
    result = L["command"](request)
    if result[-1] != "audit_flux_prompt_factorial.py":
        raise ValueError("qualified CPU bootstrap changed")
    result[-1] = "flux_matched_pass_worker.py"
    return result


def stage(run, parent, audit):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, parent, audit = map(lambda p: Path(p).resolve(), (run, parent, audit))
    audit_job, audit_sha = validate_accepted_audit(parent, audit)
    request, receipt, job = [read(parent / name) for name in ("request.json", "collection.json", "collection-job.json")]
    remote = C["validate_job"](job, job["job_id"], request)
    if C["inventory"](parent / "extracted/flux-prompt-factorial-result") != read(parent / "collection-local-inventory.json"):
        raise ValueError("original parent inventory differs")
    run.mkdir(parents=True, exist_ok=False)
    config = {"schema": "mdb-matched-pass-plan-v1", "parent_job_id": job["job_id"],
        "parent_request_sha256": request["request_sha256"], "parent_remote_result": remote,
        "parent_audit_job_id": audit_job, "parent_audit_sha256": audit_sha,
        "checkpoint_root": request["config"]["model_path"], "cpu_threads": 4,
        "native_framework_forwards": 0, "mathematical_transitions": 56,
        "max_wall_seconds": 1800, "conditions": ["c1-o0", "c0-o1"], "logical_step": 1,
        "max_decoded_bytes": 18 << 30, "max_retained_bytes": 8 << 30,
        "scope": "Retrospective selected high/low cases; both prompt factors differ. Q=G_1(N_1), not a fresh prospective prediction."}
    save(run / "plan.json", config)
    worker = HERE / "flux_matched_pass_worker.py"
    sources = [(worker, worker.name)]
    for name in ("flux_matched_pass_replay.py", "memoize_flux_audit_parameters.py", "audit_flux_prompt_factorial.py",
                 "submit_flux_prompt_factorial_audit.py", "submit_flux_initial_noise.py"):
        sources.append((HERE / name, name))
    for name, digest in request["independent_math_sources"].items():
        frozen = parent / "independent-math-freeze" / name
        if sha(frozen) != digest:
            raise ValueError("pre-outcome numerical cohort changed")
        sources.append((frozen, name))
    for name in sorted(set(receipt["bindings"]) | {"collection.json", "collection-job.json", "request.json"}):
        path = parent / name
        if name in receipt["bindings"] and sha(path) != receipt["bindings"][name]:
            raise ValueError("authenticated parent metadata changed")
        sources.append((path, "parent-run/" + name))
    sources += [(parent / "payload" / name, "parent-run/payload/" + name) for name in request["source_hashes"]]
    sources += [(parent / "independent-math-freeze" / name, "parent-run/independent-math-freeze/" + name)
        for name in request["independent_math_sources"]]
    audit_receipt = read(audit / "collection.json")
    sources += [(audit / name, "parent-audit/" + name) for name in
        sorted(set(audit_receipt["bindings"]) | {"collection.json", "request.json", "audit.json"})]
    sources.append((run / "plan.json", "plan.json"))
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [target for _, target in sources] + [preflight.name],
        "path_token_allowlist": [remote, config["checkpoint_root"], "parent-run", "parent-audit", "audit-parameter-cache", OUTPUT, "matched-pass-result.json"]})
    sources.append((preflight, preflight.name))
    encoded = (json.dumps(config, sort_keys=True) + "\n").encode()
    new = {"schema": "mdb-matched-pass-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(encoded).hexdigest(), "source_hashes": {target: sha(path) for path, target in sources},
        "launcher_sha256": sha(Path(__file__)), "execution_environment": L["B"]["execution_intent"](),
        "scheduler_url": request["scheduler_url"], "needs": {}, "task_family": "analysis", "device": "cpu", "backend": "independent-numpy",
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=2,
            logical_branches=2, logical_checkpoint_replays=0, checkpoint_policy="One finite CPU lease; two matched-input mathematical passes, no native forwards or retries.",
            extra={"task_family": "analysis", "native_framework_forwards": 0, "mathematical_transitions": 56}),
        "resource_exception": {**RESOURCES, "calibration": "Two28-transition passes with original parameters and losslessBF16 cache. Complete four-context audit04 calibrates CPU4 execution;7.219GiB originalBF16 payload plus previously observed2.33GiB transients and landmark arrays fit reviewed14GiB.900s estimate/1800s worker/2100s outer allows checkpoint hydration, hashes and cleanup. No production speedup claim."}}
    new["request_sha256"] = hashlib.sha256(json.dumps(new, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", new)
    stage_worker(run / "payload", worker=worker, config=config, request=new, config_bytes=encoded, additional_files=sources[1:])
    return {"submitted": False, "request_sha256": new["request_sha256"]}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    if (run / "submission.json").exists():
        raise ValueError("matched pass already admitted")
    request = read(run / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or request["launcher_sha256"] != sha(Path(__file__))
            or any(sha(run / "payload" / name) != digest for name, digest in request["source_hashes"].items())):
        raise ValueError("staged identity changed")
    review = read(run / "long-run-review-final.json")
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("exact source-bound review required")
    L["B"]["scheduler_endpoint"](request)
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"],
        "review_sha256": sha(run / "long-run-review-final.json")})
    r = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-matched-pass", task="analysis", model=None,
        payload=run / "payload", pin="beast", gpu=False, needs={}, ram_mb=r["ram_mb"], vram_mb=0,
        cpu_threads=r["cpu_threads"], disk_gb=r["disk_gb"], est_wall_s=r["est_wall_s"], timeout_s=r["timeout_s"],
        priority=50, retry_on_kill=False, preflight=True,
        note="Retrospective Q=G1(N1) for c10/c01:2in-process passes/56 learned-weight math transitions,0nativeforwards. CPU4 RAM14336MB,disk10GB,900s estimate/1800worker2100outer. OriginalBF16-bit cache; all Q sealed before target comparison; no new predictions, no retries; own temporary weights removed after verification.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


def collect(run):
    from mdb_lanes.job_support import scheduler_api

    run = Path(run).resolve()
    request = read(run / "request.json")
    L["B"]["scheduler_endpoint"](request)
    job_id = read(run / "submission.json")["job_id"]
    bodies = {}
    for suffix, name in (("", "job.json"), ("/events", "events.json"), ("/logs?offset=0&wait_s=1", "worker.log")):
        status, body, _ = scheduler_api().request("GET", "/api/jobs/" + job_id + suffix, timeout_s=30)
        if status != 200 or len(body) > 24 << 20:
            raise ValueError("actual scheduler evidence missing/oversized")
        bodies[name] = body
    job = json.loads(bodies["job.json"])
    remote = str(Path(C["validate_job"](job, job_id, request)).parent / OUTPUT)
    expected_log = job["result"]["artifacts"]["job_log"]
    if (job["result"]["diagnostics"]["command"] != read(run / "admission-command.json")["command"]
            or len(bodies["worker.log"]) != expected_log["bytes"]
            or hashlib.sha256(bodies["worker.log"]).hexdigest() != expected_log["sha256"]):
        raise ValueError("actual command/log differs")
    raw, capsule = L["decode_capsule"](bodies["worker.log"].decode())
    report = json.loads(raw)
    execution = report["near_data_execution"]
    if (report["schema"] != "mdb-matched-pass-execution-v1" or execution["job_id"] != job_id
            or execution["request_sha256"] != request["request_sha256"]
            or execution["worker_sha256"] != request["source_hashes"]["flux_matched_pass_worker.py"]
            or execution["mathematical_transitions"] != 56 or execution["native_framework_forwards"] != 0):
        raise ValueError("matched replay execution identity differs")
    for name, body in {**bodies, "result.json": raw}.items():
        C["preserve"](run / name, body)
    script = C["remote_script"](remote)
    scan_command = ["rtk", "proxy", "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "beast", "python3", "-"]

    def scan():
        result = subprocess.run(scan_command, input=script, text=True, capture_output=True, check=True, timeout=600)
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
    C["preserve"](allowlist, b"".join(str(C["relative"](name)).encode() + b"\0" for name in before["files"]))
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
            or read(destination / "report.json") != report["report"]):
        raise ValueError("actual worker manifest/report differs")
    save(run / "collection-remote-after.json", after)
    save(run / "collection-local-inventory.json", local)
    names = ("request.json", "job.json", "events.json", "worker.log", "result.json", "admission-command.json",
        "collection-remote-before.json", "collection-remote-after.json", "collection-local-inventory.json", "collection-copy.json", "collection-files.bin")
    receipt = {"schema": "mdb-matched-pass-collection-v1", "job_id": job_id, "request_sha256": request["request_sha256"],
        "capsule": capsule, "remote_path": remote, "file_count": local["file_count"], "expanded_bytes": local["expanded_bytes"],
        "archive_sha256": None, "native_framework_forwards": 0, "collector_sha256": sha(Path(__file__)),
        "bindings": {name: sha(run / name) for name in names}, "scope": "Original retrospective result bytes; actual scheduler/log, remote-before/after/local inventory, no fabricated archive."}
    save(run / "collection.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--parent", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--collect", action="store_true")
    a = p.parse_args()
    if sum((a.parent is not None, a.submit, a.collect)) != 1 or (a.parent is not None) != (a.audit is not None):
        p.error("choose --parent RUN --audit AUDIT, --submit, or --collect")
    print(json.dumps(collect(a.run) if a.collect else submit(a.run) if a.submit else stage(a.run, a.parent, a.audit)))
