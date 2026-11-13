"""Stage, admit and collect native continuations of sealed predicted FLUX state."""
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
B = L["B"]
sha, read, save = (M[key] for key in ("sha", "read", "save"))
OUTPUT = "flux-native-corner-result"
RESOURCES = {"ram_mb": 14336, "vram_mb": 13600, "cpu_threads": 4, "disk_gb": 4,
    "est_wall_s": 450, "outer_timeout_s": 1200, "priority": 50}
MAX_TOTAL = 4 << 30


def validate_request(request):
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request self digest differs")


def validate_matched(parent, matched):
    """Authenticate the completed Q result before it can anchor new branches."""
    request, receipt, job, result = [read(matched / name) for name in
        ("request.json", "collection.json", "job.json", "result.json")]
    validate_request(request)
    C["validate_job"](job, job["job_id"], request)
    if (receipt["job_id"] != job["job_id"] or receipt["request_sha256"] != request["request_sha256"]
            or any(sha(matched / name) != digest for name, digest in receipt["bindings"].items())
            or job["result"]["diagnostics"]["command"] != read(matched / "admission-command.json")["command"]):
        raise ValueError("matched collection/job identity differs")
    raw, capsule = L["decode_capsule"]((matched / "worker.log").read_text())
    expected_log = job["result"]["artifacts"]["job_log"]
    data = matched / "extracted/matched-pass-output"
    report = read(data / "report.json")
    execution = result["near_data_execution"]
    if (raw != (matched / "result.json").read_bytes() or capsule != receipt["capsule"]
            or sha(matched / "worker.log") != expected_log["sha256"]
            or (matched / "worker.log").stat().st_size != expected_log["bytes"]
            or report != result["report"] or execution["job_id"] != job["job_id"]
            or execution["parent_job_id"] != read(parent / "collection.json")["job_id"]
            or execution["mathematical_transitions"] != 56 or execution["native_framework_forwards"] != 0
            or report["Q_seal_sha256"] != sha(data / "Q-seal.json")
            or C["inventory"](data) != read(matched / "collection-local-inventory.json")):
        raise ValueError("actual completed Q result differs from its custody")
    return job["job_id"], sha(data / "report.json")


def command(request):
    result = B["sealed_worker_command"](request)
    if result[-1] != "flux_initial_noise_campaign.py":
        raise ValueError("qualified CUDA bootstrap changed")
    result[-1] = "flux_native_corner.py"
    return result


def work_contract():
    from mdb_lanes.job_support import one_lease_scheduling_contract

    return one_lease_scheduling_contract(model_loads=1, logical_operations=4,
        logical_branches=4, logical_checkpoint_replays=4, max_local_work_items=64,
        checkpoint_policy="Two source-exact N1 parents; four in-process latent Acts and native84 suffixes. No new encodes, noise, math or automatic retry.",
        extra={"native_adapter_transitions": 336, "mathematical_transitions": 0, "vae_calls": 4})


def validate_payload_helpers(payload):
    """Exercise trusted source imports and pinned helper closure without a model."""
    payload = Path(payload)
    runpy.run_path(str(payload / "collect_flux_prompt_factorial.py"))
    runpy.run_path(str(payload / "flux_native_corner.py"))["support"]()
    return {"collector_import": "valid", "immutable_producer_helper_chain": "valid", "model_calls": 0}


def stage(run, *, parent, audit, matched, expectations, response_contract=None, expectation_correction=None):
    from mdb_lanes.payload import runtime_wheel_names, stage_worker

    if response_contract is None:
        raise ValueError("staging requires the paired response contract")
    run, parent, audit, matched, expectations = map(lambda p: Path(p).resolve(),
        (run, parent, audit, matched, expectations))
    audit_job, audit_sha = M["validate_accepted_audit"](parent, audit)
    matched_job, matched_sha = validate_matched(parent, matched)
    parent_request, receipt, job = [read(parent / name) for name in
        ("request.json", "collection.json", "collection-job.json")]
    validate_request(parent_request)
    remote = C["validate_job"](job, job["job_id"], parent_request)
    if C["inventory"](parent / "extracted/flux-prompt-factorial-result") != read(parent / "collection-local-inventory.json"):
        raise ValueError("original parent inventory changed")
    prediction_plan = read(expectations)
    if (prediction_plan["schema"] != "mdb-native-corner-expectations-v1"
            or prediction_plan["contexts"] != ["c1-o0", "c0-o1"]
            or prediction_plan["counts"]["adapter_steps"] != 336):
        raise ValueError("pre-native expectation geometry differs")
    worker = HERE / "flux_native_corner.py"
    engine = runpy.run_path(str(worker))
    config = engine["make_config"](parent_request["config"])
    config.update(parent_remote_result=remote, parent_job_id=job["job_id"],
        parent_request_sha256=parent_request["request_sha256"], parent_audit_job_id=audit_job,
        parent_audit_sha256=audit_sha, matched_job_id=matched_job, matched_report_sha256=matched_sha,
        matched_result_sha256=sha(matched / "result.json"),
        expectations_sha256=sha(expectations), max_wall_seconds=900,
        transport_method=C["METHOD"], transport_limits={"max_expanded_bytes": MAX_TOTAL, "max_member_bytes": 1 << 30})
    if response_contract is not None:
        response_contract = Path(response_contract).resolve()
        contract = read(response_contract)
        if (contract["previous_expectations_sha256"] != sha(expectations)
                or contract["primary_native_response"] != "D_F=R-M"):
            raise ValueError("paired native-response contract differs")
        config.update(response_contract_sha256=sha(response_contract), response_contract_path="response-contract.json")
    if expectation_correction is not None:
        expectation_correction = Path(expectation_correction).resolve()
        correction = read(expectation_correction)
        if (correction["original_expectations_sha256"] != sha(expectations)
                or correction["response_contract_sha256"] != sha(response_contract)):
            raise ValueError("expectation correction is bound to different sources")
        config.update(expectation_correction_sha256=sha(expectation_correction), expectation_correction_path="expectation-correction.json")
    engine["validate_config"](config)
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    sources = [(worker, worker.name), (expectations, "expectations.json")]
    if response_contract is not None:
        sources.append((response_contract, "response-contract.json"))
    if expectation_correction is not None:
        sources.append((expectation_correction, "expectation-correction.json"))
    helpers = {"audit_flux_prompt_factorial.py", "submit_flux_prompt_factorial_audit.py",
        "submit_flux_initial_noise.py", "submit_flux_matched_pass.py", "collect_flux_prompt_factorial.py", "collect_flux_initial_noise.py"}
    for name in parent_request["source_hashes"]:
        if name.endswith(".py"):
            source = parent / "payload" / name
            if sha(source) != parent_request["source_hashes"][name]:
                raise ValueError("original producer helper changed")
            sources.append((source, name))
            helpers.discard(name)
    sources += [(HERE / name, name) for name in sorted(helpers)]
    for name in sorted(set(receipt["bindings"]) | {"collection.json", "collection-job.json", "request.json"}):
        if name in receipt["bindings"] and sha(parent / name) != receipt["bindings"][name]:
            raise ValueError("parent collection metadata changed")
        sources.append((parent / name, "parent-run/" + name))
    for name in ("request.json", "collection.json", "job.json", "result.json"):
        sources.append((matched / name, "matched-prior/" + name))
    for name in ("report.json", "Q-seal.json"):
        sources.append((matched / "extracted/matched-pass-output" / name, "matched-prior/" + name))
    audit_receipt = read(audit / "collection.json")
    for name in sorted(set(audit_receipt["bindings"]) | {"collection.json", "request.json", "audit.json"}):
        sources.append((audit / name, "parent-audit/" + name))
    sources.append((run / "plan.json", "plan.json"))
    if len({name for _, name in sources}) != len(sources):
        raise ValueError("duplicate staged source name")
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [name for _, name in sources] + [preflight.name],
        "path_token_allowlist": [remote, config["model_path"], "parent-run", "parent-audit", "matched-prior", OUTPUT]})
    sources.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-native-corner-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in sources}, "launcher_sha256": sha(Path(__file__)),
        "execution_environment": parent_request["execution_environment"], "scheduler_url": parent_request["scheduler_url"],
        "model": parent_request["model"], "task_family": "forward", "backend": "diffusers", "device": "cuda",
        "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True, "host": "beast"},
        "runtime_wheels": {name: sha(parent / "payload" / name) for name in runtime_wheel_names()},
        "scheduling_contract": work_contract(),
        "resource_exception": {**RESOURCES, "calibration_job": "job-95453050cf73",
            "measured_peak_rss_mb": 7900.4, "measured_peak_vram_mb": 9266,
            "reason": "Same original Klein4B CUDA BF16 B1 512px profile. One resident pipeline, four sequential84 native suffixes, sparse full landmarks. Prior448native plus448math campaign measured7900.4MB RSS/9266MB VRAM. Reserve prior qualified14GiB/13600MB envelope with finite450s estimate/900s worker/1200s outer. Source evidence is host-local, so Beast placement is required."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw,
        additional_files=[(path, name) for path, name in sources if name != worker.name])
    if any(sha(run / "payload" / name) != digest for name, digest in request["runtime_wheels"].items()):
        raise ValueError("installed public runtime wheels differ from the original native execution")
    save(run / "payload-helper-check.json", validate_payload_helpers(run / "payload"))
    return {"run": str(run), "submitted": False, "request_sha256": request["request_sha256"]}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_request(request)
    if (run / "submission.json").exists():
        raise ValueError("finite native campaign already admitted")
    if (sha(Path(__file__)) != request["launcher_sha256"]
            or any(sha(run / "payload" / name) != digest for name, digest in
                {**request["source_hashes"], **request["runtime_wheels"]}.items())
            or read(run / "plan.json") != request["config"]):
        raise ValueError("staged native source/plan/runtime changed")
    validate_payload_helpers(run / "payload")
    review = read(run / "long-run-review-final.json")
    if (review.get("request_sha256") != request["request_sha256"]
            or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("source-bound native execution review required")
    B["scheduler_endpoint"](request)
    cmd = command(request)
    save(run / "admission-command.json", {"command": cmd, "request_sha256": request["request_sha256"],
        "launcher_sha256": sha(Path(__file__)), "review_sha256": sha(run / "long-run-review-final.json")})
    r = request["resource_exception"]
    note = ("Klein4B BF16 CUDA B1 512: two known N1 parents, same-value shams versus sealed P1 latent. "
        "Seal both old-P forecasts before336native/4Acts/340receipts/4VAE; no encodes/noise/math. "
        "Pass1 full5landmarks; later commits; preserve rank/sign misses. One load, CPU4/RAM14336/VRAM13600, "
        "expected450/worker900/outer1200s; no retries. Tests native response and final image continuation.")
    if len(note) > 500:
        raise ValueError("queue note oversized")
    job = guarded_mrun_launch(cmd, request=request, model=request["model"], task="forward",
        experiment="mdb-flux-native-corner", payload=run / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=r["ram_mb"], vram_mb=r["vram_mb"], disk_gb=r["disk_gb"],
        cpu_threads=r["cpu_threads"], est_wall_s=r["est_wall_s"], timeout_s=r["outer_timeout_s"],
        priority=r["priority"], retry_on_kill=False, preflight=True, note=note)
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


def verify_manifest(data, actual, job_id, request):
    from mdb_lanes.parameter_artifacts import validate_manifest

    manifest = read(data / "remote-manifest.json")
    if (manifest != {"schema": "mdb-step-boundary-remote-manifest-v1", "job_id": job_id,
            "request_sha256": request["request_sha256"],
            "files": {k: v for k, v in actual["files"].items() if k != "remote-manifest.json"}}
            or read(data / "plan.json") != request["config"]):
        raise ValueError("native result manifest/plan differs")
    source, summary = [read(data / name) for name in ("source.json", "summary.json")]
    if (source["source_hashes"] != request["source_hashes"]
            or any(v["job_id"] != job_id or v["request_sha256"] != request["request_sha256"] for v in (source, summary))):
        raise ValueError("native source/summary identity differs")
    recipe = validate_manifest(read(data / "parameter-artifacts.json"))
    if (len(recipe["artifacts"]) != 169
            or sha(data / "retrieve_weights.py") != request["config"]["expected_parameter_artifacts_module_sha256"]
            or any(row["path"] in actual["files"] for row in recipe["artifacts"])):
        raise ValueError("standalone retrieval recipes and omitted parameter arrays required")
    return manifest


def collect(run):
    from mdb_lanes.job_support import scheduler_api

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_request(request)
    B["scheduler_endpoint"](request)
    job_id = read(run / "submission.json")["job_id"]
    if any(sha(run / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged native source changed")
    bodies = {}
    for suffix, name, bound in (("", "job.json", 24 << 20), ("/events", "events.json", 8 << 20),
                                ("/logs?offset=0&wait_s=1", "worker.log", 16 << 20)):
        status, body, _ = scheduler_api().request("GET", "/api/jobs/" + job_id + suffix, timeout_s=30)
        if status != 200 or len(body) > bound:
            raise ValueError("actual scheduler evidence missing/oversized")
        bodies[name] = body
    job = json.loads(bodies["job.json"])
    remote = str(Path(C["validate_job"](job, job_id, request)).parent / OUTPUT)
    expected = job["result"]["artifacts"]["job_log"]
    if (job["result"]["diagnostics"]["command"] != read(run / "admission-command.json")["command"]
            or len(bodies["worker.log"]) != expected["bytes"]
            or hashlib.sha256(bodies["worker.log"]).hexdigest() != expected["sha256"]):
        raise ValueError("native actual command/log differs")
    for name, raw in bodies.items():
        C["preserve"](run / name, raw)
    script = C["remote_script"](remote)

    def scan():
        result = subprocess.run(["rtk", "proxy", "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "beast", "python3", "-"], input=script, text=True, capture_output=True, check=True, timeout=600)
        if len(result.stdout) > 16 << 20:
            raise ValueError("remote inventory oversized")
        value = json.loads(result.stdout)
        if value["expanded_bytes"] > MAX_TOTAL:
            raise ValueError("native output exceeds declared collection bound")
        return value

    before = scan()
    save(run / "collection-remote-before.json", before)
    if shutil.disk_usage(run).free < before["expanded_bytes"] + (128 << 20):
        raise ValueError("insufficient local collection space")
    destination = run / "extracted" / OUTPUT
    destination.mkdir(parents=True, exist_ok=False)
    allowlist = run / "collection-files.bin"
    C["preserve"](allowlist, b"".join(C["relative"](name).encode() + b"\0" for name in before["files"]))
    copied = subprocess.run(["rtk", "proxy", "rsync", "--recursive", "--times", "--protect-args", "--no-links",
        "--no-devices", "--no-specials", "--from0", "--files-from=" + str(allowlist), "--max-size=" + str(1 << 30),
        "--timeout=120", "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes", "beast:" + remote + "/", str(destination) + "/"],
        text=True, capture_output=True, check=True, timeout=1800)
    save(run / "collection-copy.json", {"returncode": copied.returncode, "stdout": copied.stdout, "stderr": copied.stderr})
    after, local = scan(), C["inventory"](destination)
    if before != after or before != local:
        raise ValueError("native remote-before/after/local inventories differ")
    verify_manifest(destination, local, job_id, request)
    save(run / "collection-remote-after.json", after)
    save(run / "collection-local-inventory.json", local)
    for name in ("parameter-artifacts.json", "retrieve_weights.py"):
        C["preserve"](run / name, (destination / name).read_bytes())
    names = ("request.json", "job.json", "events.json", "worker.log", "admission-command.json",
        "collection-remote-before.json", "collection-remote-after.json", "collection-local-inventory.json",
        "collection-copy.json", "collection-files.bin", "parameter-artifacts.json", "retrieve_weights.py")
    result = {"schema": "mdb-native-corner-collection-v1", "job_id": job_id, "request_sha256": request["request_sha256"],
        "remote_path": remote, "file_count": local["file_count"], "expanded_bytes": local["expanded_bytes"], "archive_sha256": None,
        "manifest_sha256": sha(destination / "remote-manifest.json"), "summary_sha256": sha(destination / "summary.json"),
        "collector_sha256": sha(Path(__file__)), "bindings": {name: sha(run / name) for name in names},
        "scope": "Original native result bytes authenticated through actual scheduler command/payload/log and unchanged remote-before/after/local inventories; no numerical interpretation during collection."}
    save(run / "collection.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--matched", type=Path)
    parser.add_argument("--expectations", type=Path)
    parser.add_argument("--response-contract", type=Path)
    parser.add_argument("--expectation-correction", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if not args.submit and not args.collect and any(v is None for v in (args.parent, args.audit, args.matched, args.expectations, args.response_contract)):
        parser.error("staging requires --parent, --audit, --matched, --expectations and --response-contract")
    result = collect(args.run) if args.collect else submit(args.run) if args.submit else stage(args.run,
        parent=args.parent, audit=args.audit, matched=args.matched, expectations=args.expectations,
        response_contract=args.response_contract, expectation_correction=args.expectation_correction)
    print(json.dumps(result, sort_keys=True))
