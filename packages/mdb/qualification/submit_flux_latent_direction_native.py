"""Stage/admit/collect one finite native receiver × direction cross."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_SHA256 = "d27d31215f258847e6a15b92a81b230ed1e1cdf66d572a475690d7507081c264"
prior_path = HERE / "submit_flux_native_corner.py"
if hashlib.sha256(prior_path.read_bytes()).hexdigest() != PRIOR_SHA256:
    raise ValueError("qualified native launch/collection primitive differs")
P = runpy.run_path(str(prior_path))
C, B, L = (P[key] for key in ("C", "B", "L"))
sha, read, save, validate_request = (P[key] for key in ("sha", "read", "save", "validate_request"))
OUTPUT = "latent-direction-native-result"
MATH_OUTPUT = "latent-direction-math-result"
MAX_TOTAL = 4 << 30
PARAMETER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
CUSTODY_SHA256 = "9a25178d14ed776f56daa2a609bd4a967c21587a451721dcd75f0a48f03c0121"
RESOURCES = {"ram_mb": 14336, "vram_mb": 13600, "cpu_threads": 4, "disk_gb": 4,
    "est_wall_s": 900, "outer_timeout_s": 1500, "priority": 50}


def engine(path=None):
    return runpy.run_path(str(path or HERE / "flux_latent_direction_native.py"))


def custody():
    path = HERE / "flux_latent_direction_custody.py"
    if sha(path) != CUSTODY_SHA256:
        raise ValueError("qualified native recovery custody source differs")
    return runpy.run_path(str(path))


def validate_collected_job(job, job_id, request, collection):
    if collection.get("schema") == "mdb-latent-direction-native-recovery-collection-v1":
        return custody()["validate_job"](job, job_id, request, collection=collection)
    return C["validate_job"](job, job_id, request)


def verify_collected_manifest(data, actual, job_id, request, collection, *, run):
    if collection.get("schema") == "mdb-latent-direction-native-recovery-collection-v1":
        return custody()["verify_manifest"](data, actual, job_id, request, collection=collection, run=run)
    return verify_manifest(data, actual, job_id, request)


def verify_math_manifest(data, actual, job_id, request):
    from mdb.parameter_artifacts import validate_manifest

    data = Path(data)
    manifest = read(data / "remote-manifest.json")
    if manifest != {"schema": "mdb-latent-direction-math-manifest-v1", "job_id": job_id,
            "request_sha256": request["request_sha256"], "files": {k: v for k, v in actual["files"].items() if k != "remote-manifest.json"}}:
        raise ValueError("CPU math manifest differs from actual inventory")
    source = read(data / "source.json")
    if (source["job_id"] != job_id or source["request_sha256"] != request["request_sha256"]
            or source["source_hashes"] != request["source_hashes"] or read(data / "plan.json") != request["config"]):
        raise ValueError("CPU math source/plan differs")
    recipe = validate_manifest(read(data / "parameter-artifacts.json"))
    # CPU math's matched-pass config has no expected_* module fields. The
    # standalone retriever is the same explicitly qualified source authority.
    if (len(recipe["artifacts"]) != 169 or sha(data / "retrieve_weights.py") != PARAMETER_SHA256
            or any(r["path"] in actual["files"] for r in recipe["artifacts"])):
        raise ValueError("CPU math must retain169 recipes without weight arrays")
    return manifest


def validate_sources(parent, math_run, *, audit):
    """Authenticate two existing jobs and actual CPU log capsule, without math."""
    parent, math_run, audit = map(Path, (parent, math_run, audit))
    pr, pc, pj = [read(parent / name) for name in ("request.json", "collection.json", "collection-job.json")]
    mr, mc, mj, capsule = [read(math_run / name) for name in ("request.json", "collection.json", "job.json", "result.json")]
    for directory, request, collection, job in ((parent, pr, pc, pj), (math_run, mr, mc, mj)):
        validate_request(request)
        C["validate_job"](job, job["job_id"], request)
        if (collection["job_id"] != job["job_id"] or collection["request_sha256"] != request["request_sha256"]
                or any(sha(directory / name) != digest for name, digest in collection["bindings"].items())
                or job["result"]["diagnostics"]["command"] != read(directory / "admission-command.json")["command"]):
            raise ValueError("actual source collection/job/command changed")
    audit_job, audit_sha = P["M"]["validate_accepted_audit"](parent, audit)
    raw, encoded = L["decode_capsule"]((math_run / "worker.log").read_text())
    expected = mj["result"]["artifacts"]["job_log"]
    execution = capsule["near_data_execution"]
    summary = capsule["report"]
    remote = str(Path(C["validate_job"](mj, mj["job_id"], mr)).parent / MATH_OUTPUT)
    if (mc["schema"] != "mdb-latent-direction-math-collection-v1" or mc["remote_path"] != remote
            or raw != (math_run / "result.json").read_bytes() or encoded != mc["capsule"]
            or sha(math_run / "worker.log") != expected["sha256"] or (math_run / "worker.log").stat().st_size != expected["bytes"]
            or capsule["schema"] != "mdb-latent-direction-math-execution-v1" or capsule["output_directory"] != MATH_OUTPUT
            or execution["job_id"] != mj["job_id"] or execution["request_sha256"] != mr["request_sha256"]
            or execution["parent_job_id"] != pj["job_id"] or execution["parent_request_sha256"] != pr["request_sha256"]
            or execution["mathematical_transitions"] != 504 or execution["native_framework_forwards"] != 0
            or execution["parent_inventory_unchanged"] is not True or summary["source_job_id"] != mj["job_id"]
            or execution["worker_sha256"] != mr["source_hashes"]["flux_latent_direction_cross.py"]
            or summary["parent_job_id"] != pj["job_id"] or summary["math_transitions"] != 504 or summary["native_forwards"] != 0):
        raise ValueError("actual CPU sealed-math job/result/capsule identity differs")
    inventory = read(math_run / "collection-local-inventory.json")
    if (inventory != read(math_run / "collection-remote-before.json") or inventory != read(math_run / "collection-remote-after.json")
            or inventory["file_count"] != mc["file_count"] or inventory["expanded_bytes"] != mc["expanded_bytes"]):
        raise ValueError("CPU original inventory snapshots differ")
    return {"parent_remote_result": C["validate_job"](pj, pj["job_id"], pr), "parent_job_id": pj["job_id"],
        "parent_request_sha256": pr["request_sha256"], "parent_audit_job_id": audit_job, "parent_audit_sha256": audit_sha,
        "math_remote_result": remote, "math_job_id": mj["job_id"], "math_request_sha256": mr["request_sha256"],
        "math_collection_sha256": sha(math_run / "collection.json"), "math_result_sha256": sha(math_run / "result.json"),
        "math_cohort_sha256": summary["cohort_seal_sha256"], "math_inputs_sha256": summary["inputs_sha256"]}


def command(request):
    result = B["sealed_worker_command"](request)
    if result[-1] != "flux_initial_noise_campaign.py":
        raise ValueError("qualified bootstrap ABI changed")
    result[-1] = "flux_latent_direction_native.py"
    return result


def work_contract():
    from mdb.job_support import one_lease_scheduling_contract

    return one_lease_scheduling_contract(model_loads=1, logical_operations=6, logical_branches=6, logical_checkpoint_replays=6,
        max_local_work_items=64, checkpoint_policy="Six CPU-sealed finals via6renderActs before native0;6original N1 latent Acts then84native each; no math/encodes/noise/retry.",
        extra={"native_adapter_transitions": 504, "render_acts": 6, "native_branch_acts": 6, "receipt_occurrences": 516, "vae_calls": 12})


def validate_payload_helpers(payload):
    payload = Path(payload)
    runpy.run_path(str(payload / "collect_flux_prompt_factorial.py"))
    engine(payload / "flux_latent_direction_native.py")["support"]()
    return {"collector_import": "valid", "immutable_helper_closure": "valid", "model_calls": 0}


def stage(run, *, parent, math_run, audit):
    from mdb.payload import runtime_wheel_names, stage_worker

    run, parent, math_run, audit = map(lambda p: Path(p).resolve(), (run, parent, math_run, audit))
    metadata = validate_sources(parent, math_run, audit=audit)
    pr, mr = read(parent / "request.json"), read(math_run / "request.json")
    for directory, basename in ((parent, "flux-prompt-factorial-result"), (math_run, MATH_OUTPUT)):
        if C["inventory"](directory / "extracted" / basename) != read(directory / "collection-local-inventory.json"):
            raise ValueError("authenticated source data changed before native staging")
    math_data = math_run / "extracted" / MATH_OUTPUT
    verify_math_manifest(math_data, C["inventory"](math_data), metadata["math_job_id"], mr)
    worker = HERE / "flux_latent_direction_native.py"
    e = engine(worker)
    config = e["make_config"](pr["config"])
    config.update(metadata)
    e["load_math"](math_data, parent_job_id=metadata["parent_job_id"], parent_request_sha256=metadata["parent_request_sha256"])
    e["validate_config"](config)
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    files = {worker.name: worker, Path(__file__).name: Path(__file__), "flux_native_corner.py": HERE / "flux_native_corner.py",
        "flux_latent_direction_custody.py": HERE / "flux_latent_direction_custody.py",
        "submit_flux_native_corner.py": prior_path, "expectations.json": math_data / "expectations.json"}
    closure = {"audit_flux_prompt_factorial.py", "submit_flux_prompt_factorial_audit.py", "submit_flux_initial_noise.py",
        "submit_flux_matched_pass.py", "collect_flux_prompt_factorial.py", "collect_flux_initial_noise.py"}
    for name, digest in pr["source_hashes"].items():
        if name.endswith(".py"):
            source = parent / "payload" / name
            if sha(source) != digest:
                raise ValueError("original immutable producer source changed")
            files[name] = source
            closure.discard(name)
    files.update({name: HERE / name for name in closure})
    for directory, prefix, job_file, extra in ((parent, "parent-run", "collection-job.json", ()),
            (math_run, "math-run", "job.json", ("result.json",)), (audit, "parent-audit", "job.json", ("audit.json",))):
        collection = read(directory / "collection.json")
        for name in sorted(set(collection["bindings"]) | {"request.json", "collection.json", job_file, *extra}):
            if name in collection["bindings"] and sha(directory / name) != collection["bindings"][name]:
                raise ValueError("staged source metadata changed")
            files[prefix + "/" + name] = directory / name
    files["plan.json"] = run / "plan.json"
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": list(files) + [preflight.name],
        "path_token_allowlist": [metadata["parent_remote_result"], metadata["math_remote_result"], config["model_path"],
            "parent-run", "math-run", "parent-audit", OUTPUT]})
    files[preflight.name] = preflight
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-latent-direction-native-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "source_hashes": {name: sha(path) for name, path in files.items()},
        "launcher_sha256": sha(__file__), "execution_environment": pr["execution_environment"], "scheduler_url": pr["scheduler_url"],
        "model": pr["model"], "task_family": "forward", "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "needs": {"cuda": True, "host": "beast"},
        "runtime_wheels": {name: sha(parent / "payload" / name) for name in runtime_wheel_names()}, "scheduling_contract": work_contract(),
        "resource_exception": {**RESOURCES, "calibration_job": "job-f0c4066835bb", "measured_peak_rss_mb": 3482.1,
            "measured_peak_vram_mb": 9284, "prior_resident_seconds": 342.97425347333774,
            "reason": "Same pinned original native pipeline/profile. Prior336native measured342.97s resident/735.97s runner; new504native plus6extraVAE, no math/encoder/noise. Finite1200s worker/1500outer; reviewer required."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw,
        additional_files=[(path, name) for name, path in files.items() if name != worker.name])
    if any(sha(run / "payload" / name) != digest for name, digest in request["runtime_wheels"].items()):
        raise ValueError("public runtime wheels differ from original native history")
    save(run / "payload-helper-check.json", validate_payload_helpers(run / "payload"))
    return {"run": str(run), "request_sha256": request["request_sha256"], "submitted": False}


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_request(request)
    if (run / "submission.json").exists():
        raise ValueError("finite native direction cross already submitted")
    if (sha(__file__) != request["launcher_sha256"] or read(run / "plan.json") != request["config"]
            or any(sha(run / "payload" / name) != digest for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items())):
        raise ValueError("staged native source/config/runtime changed")
    validate_payload_helpers(run / "payload")
    review = read(run / "long-run-review-final.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("source-bound independent long-run review required")
    B["scheduler_endpoint"](request)
    cmd = command(request)
    save(run / "admission-command.json", {"command": cmd, "request_sha256": request["request_sha256"],
        "launcher_sha256": sha(__file__), "review_sha256": sha(run / "long-run-review-final.json")})
    r = request["resource_exception"]
    note = ("Klein4B CUDA BF16 B1 512 receiver×direction cross. Six CPU math finals sealed/rendered atnative0, "
        "then6original N1 latent Acts→84native each.504native/12Acts/516receipts/12VAE,1load,0encodes/noise/math. "
        "Fullpass1 five landmarks/later commits; all signed outcomes retained. CPU4/RAM14336/VRAM13600, expected900/worker1200/outer1500s; no retry.")
    if len(note) > 500:
        raise ValueError("queue note oversized")
    job = guarded_mrun_launch(cmd, request=request, model=request["model"], task="forward", experiment="mdb-flux-latent-direction-native",
        payload=run / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=r["ram_mb"], vram_mb=r["vram_mb"],
        disk_gb=r["disk_gb"], cpu_threads=r["cpu_threads"], est_wall_s=r["est_wall_s"], timeout_s=r["outer_timeout_s"],
        priority=r["priority"], retry_on_kill=False, preflight=True, note=note)
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


def verify_manifest(data, actual, job_id, request):
    from mdb.parameter_artifacts import validate_manifest

    if (read(data / "remote-manifest.json") != {"schema": "mdb-step-boundary-remote-manifest-v1", "job_id": job_id,
            "request_sha256": request["request_sha256"], "files": {k: v for k, v in actual["files"].items() if k != "remote-manifest.json"}}
            or read(data / "plan.json") != request["config"]):
        raise ValueError("native manifest/plan differs")
    source, summary = [read(data / name) for name in ("source.json", "summary.json")]
    if (source["source_hashes"] != request["source_hashes"] or summary["counts"] != request["config"]["counts"]
            or any(v["job_id"] != job_id or v["request_sha256"] != request["request_sha256"] for v in (source, summary))):
        raise ValueError("native actual source/summary/census differs")
    recipe = validate_manifest(read(data / "parameter-artifacts.json"))
    if len(recipe["artifacts"]) != 169 or sha(data / "retrieve_weights.py") != request["config"]["expected_parameter_artifacts_module_sha256"] or any(row["path"] in actual["files"] for row in recipe["artifacts"]):
        raise ValueError("standalone169 recipes required; parameter bank must not be persisted")
    for cell in request["config"]["conditions"]:
        if "transactions/" + cell + "/seal.json" not in actual["files"]:
            raise ValueError("receiver seal absent")


def collect(run):
    from mdb.job_support import scheduler_api

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_request(request)
    B["scheduler_endpoint"](request)
    job_id = read(run / "submission.json")["job_id"]
    if any(sha(run / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged native source changed")
    bodies = {}
    for suffix, name, bound in (("", "job.json", 24 << 20), ("/events", "events.json", 8 << 20), ("/logs?offset=0&wait_s=1", "worker.log", 16 << 20)):
        status, body, _ = scheduler_api().request("GET", "/api/jobs/" + job_id + suffix, timeout_s=30)
        if status != 200 or len(body) > bound:
            raise ValueError("actual job evidence absent/oversized")
        bodies[name] = body
    job = json.loads(bodies["job.json"])
    if job.get("state") == "failed":
        return custody()["recover"](run)
    remote = str(Path(C["validate_job"](job, job_id, request)).parent / OUTPUT)
    expected = job["result"]["artifacts"]["job_log"]
    if (job["result"]["diagnostics"]["command"] != read(run / "admission-command.json")["command"]
            or len(bodies["worker.log"]) != expected["bytes"] or hashlib.sha256(bodies["worker.log"]).hexdigest() != expected["sha256"]):
        raise ValueError("actual admitted native command/log differs")
    for name, raw in bodies.items():
        C["preserve"](run / name, raw)
    script = C["remote_script"](remote)
    def scan():
        value = subprocess.run(["rtk", "proxy", "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "beast", "python3", "-"],
            input=script, text=True, capture_output=True, check=True, timeout=600)
        if len(value.stdout) > 16 << 20:
            raise ValueError("remote native inventory oversized")
        result = json.loads(value.stdout)
        if result["expanded_bytes"] > MAX_TOTAL:
            raise ValueError("native result exceeds4GiB")
        return result
    before = scan()
    save(run / "collection-remote-before.json", before)
    if shutil.disk_usage(run).free < before["expanded_bytes"] + (128 << 20):
        raise ValueError("insufficient local collection disk")
    destination = run / "extracted" / OUTPUT
    destination.mkdir(parents=True, exist_ok=False)
    allowlist = run / "collection-files.bin"
    C["preserve"](allowlist, b"".join(C["relative"](name).encode() + b"\0" for name in before["files"]))
    copied = subprocess.run(["rtk", "proxy", "rsync", "--recursive", "--times", "--protect-args", "--no-links", "--no-devices", "--no-specials",
        "--from0", "--files-from=" + str(allowlist), "--max-size=" + str(1 << 30), "--timeout=120", "-e",
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes", "beast:" + remote + "/", str(destination) + "/"],
        text=True, capture_output=True, check=True, timeout=1800)
    save(run / "collection-copy.json", {"returncode": copied.returncode, "stdout": copied.stdout, "stderr": copied.stderr})
    after, actual = scan(), C["inventory"](destination)
    if before != after or before != actual:
        raise ValueError("native before/after/local inventories differ")
    verify_manifest(destination, actual, job_id, request)
    save(run / "collection-remote-after.json", after)
    save(run / "collection-local-inventory.json", actual)
    for name in ("parameter-artifacts.json", "retrieve_weights.py"):
        C["preserve"](run / name, (destination / name).read_bytes())
    names = ("request.json", "job.json", "events.json", "worker.log", "admission-command.json", "collection-remote-before.json",
        "collection-remote-after.json", "collection-local-inventory.json", "collection-copy.json", "collection-files.bin", "parameter-artifacts.json", "retrieve_weights.py")
    result = {"schema": "mdb-latent-direction-native-collection-v1", "job_id": job_id, "request_sha256": request["request_sha256"],
        "remote_path": remote, "file_count": actual["file_count"], "expanded_bytes": actual["expanded_bytes"], "archive_sha256": None,
        "manifest_sha256": sha(destination / "remote-manifest.json"), "summary_sha256": sha(destination / "summary.json"),
        "collector_sha256": sha(__file__), "bindings": {name: sha(run / name) for name in names}, "model_calls_during_collection": 0}
    save(run / "collection.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--math-run", type=Path)
    parser.add_argument("--audit", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if not args.submit and not args.collect and any(v is None for v in (args.parent, args.math_run, args.audit)):
        parser.error("staging requires --parent --math-run --audit")
    result = collect(args.run) if args.collect else submit(args.run) if args.submit else stage(args.run, parent=args.parent, math_run=args.math_run, audit=args.audit)
    print(json.dumps(result, sort_keys=True))
