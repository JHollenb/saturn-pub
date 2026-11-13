"""Admit and collect one CPU-only, near-data independent factorial audit."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_SHA = "e75b722394ec8cd2ac5b6b505c337b3f043ed68036ac7c1691be1dbdd535980f"
base = HERE / "submit_flux_initial_noise.py"
if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError("qualified environment launcher changed")
B = runpy.run_path(str(base))
MARKER = "MDB_FACTORIAL_AUDIT_CAPSULE="
MAX_REPORT = 32 << 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def command(request):
    result = B["sealed_worker_command"](request)
    if result[-1] != "flux_initial_noise_campaign.py":
        raise ValueError("qualified bootstrap entrypoint changed")
    result[-1] = "audit_flux_prompt_factorial.py"
    result.insert(1, "CUDA_VISIBLE_DEVICES=")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result.insert(1, variable + "=" + str(request["config"]["cpu_threads"]))
    return result


def audit_contract():
    from mdb_lanes.job_support import one_lease_scheduling_contract
    return one_lease_scheduling_contract(model_loads=0, model_free=True,
        logical_operations=4, logical_branches=4, logical_checkpoint_replays=0,
        checkpoint_policy="One CPU lease near authenticated parent/checkpoint. Four independent 112-transition programs (448 total); no native framework object or forwards, no retries.",
        extra={"task_family": "analysis", "native_framework_forwards": 0, "mathematical_transitions": 448})


def audit_resources(memoize_parameters=True, *, parameter_codec="float32"):
    if parameter_codec not in ("float32", "bf16-bits"):
        raise ValueError("unknown parameter storage codec")
    if memoize_parameters and parameter_codec == "bf16-bits":
        return {"ram_mb": 14336, "vram_mb": 0, "cpu_threads": 4, "disk_gb": 10,
            "est_wall_s": 4500, "timeout_s": 6300, "max_wall_seconds": 6000,
            "calibration": "Lossless storage of the original BF16 parameter bits retains7,751,089,152bytes (7.219GiB), rather than15,502,178,304bytes of decoded FP32. Largest expanded joint bank0.914GiB; measured uncached transient RSS about2.33GiB. Independent review recommends14GiB rather than borderline12GiB for decoder copies, allocator overhead and temporary expansions. Logical decoded cap18GiB, retained allocation cap8GiB. No numerical casting, rounding, equation changes, reuse after failure or production speedup claim."}
    if memoize_parameters:
        return {"ram_mb": 28672, "vram_mb": 0, "cpu_threads": 4, "disk_gb": 10,
            "est_wall_s": 4500, "timeout_s": 6300, "max_wall_seconds": 6000,
            "calibration": "Uncached job-3e3cefedf464 spent about1030s on custody, 7.75GB recipe hydration/reverification and native capture validation; recent mathematical throughput6-7s/transition projected about4000s total. Explicit read-only parameter memoization retains up to18GiB of decoded constants (actual original FP32 parameters about15.5GB) plus observed2.5GB transient RSS and reclaimable file cache. RAM28GiB bounds that live working set; finite6000s worker/6300s outer includes cold hydration, unchanged checks and cleanup. No speedup claim before measurement."}
    return {"ram_mb": 12288, "vram_mb": 0, "cpu_threads": 4, "disk_gb": 10,
        "est_wall_s": 2500, "timeout_s": 3300, "max_wall_seconds": 3000,
        "calibration": "Historical uncached profile: prior168-transition independent saved-only audit588.8s; new448-transition producer math about1460s; declared HDD custody overhead. Job-3e3cefedf464 later showed repeated parameter rehash/read overhead can exceed this time estimate. Retained for explicit historical reproduction; no native model or speedup claim."}


def stage(run, parent, *, memoize_parameters=True, parameter_codec="float32"):
    from mdb_lanes.payload import stage_worker

    run, parent = Path(run).resolve(), Path(parent).resolve()
    parent_request, collection, job = [read(parent / name) for name in ("request.json", "collection.json", "collection-job.json")]
    independent = parent_request["independent_math_sources"]
    for name, digest in independent.items():
        if sha(parent / "independent-math-freeze" / name) != digest:
            raise ValueError("pre-outcome numerical cohort changed")
    frozen_collector = runpy.run_path(str(parent / "independent-math-freeze/collect_flux_prompt_factorial.py"))
    remote = frozen_collector["validate_job"](job, job["job_id"], parent_request)
    if frozen_collector["inventory"](parent / "extracted" / "flux-prompt-factorial-result") != read(parent / "collection-local-inventory.json"):
        raise ValueError("parent collection is not byte exact")
    run.mkdir(parents=True, exist_ok=False)
    resources = audit_resources(memoize_parameters, parameter_codec=parameter_codec)
    config = {"schema": "mdb-near-data-factorial-plan-v1", "parent_job_id": job["job_id"],
        "parent_request_sha256": parent_request["request_sha256"], "parent_remote_result": remote,
        "checkpoint_root": parent_request["config"]["model_path"], "cpu_threads": 4,
        "native_framework_forwards": 0, "mathematical_transitions": 448, "max_wall_seconds": resources["max_wall_seconds"],
        "parameter_memoization": {"enabled": memoize_parameters, "max_decoded_bytes": 18 << 30 if memoize_parameters else 0,
            "storage_codec": parameter_codec,
            "max_retained_bytes": (8 << 30) if memoize_parameters and parameter_codec == "bf16-bits" else None,
            "scope": "Original validated checkpoint parameters only; no native activations, forecasts or targets are cached."},
        "scope": "Legacy scheduling model_free means no native model object/forwards; 448 learned-weight mathematical transitions still execute."}
    save(run / "plan.json", config)
    worker = HERE / "audit_flux_prompt_factorial.py"
    sources = [(worker, worker.name)]
    if memoize_parameters:
        adapter = HERE / "memoize_flux_audit_parameters.py"
        sources.append((adapter, adapter.name))
    # Use precisely the parent's pre-outcome mathematical cohort, never live
    # numerical source files when staging this post-outcome execution wrapper.
    sources += [(parent / "independent-math-freeze" / name, name) for name in independent]
    for name in sorted(set(collection["bindings"]) | {"collection.json", "collection-job.json", "request.json"}):
        path = parent / name
        if name in collection["bindings"] and sha(path) != collection["bindings"][name]:
            raise ValueError("authenticated parent metadata changed")
        sources.append((path, "parent-run/" + name))
    sources += [(parent / "payload" / name, "parent-run/payload/" + name) for name in parent_request["source_hashes"]]
    sources += [(parent / "independent-math-freeze" / name, "parent-run/independent-math-freeze/" + name) for name in independent]
    sources.append((run / "plan.json", "plan.json"))
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [target for _, target in sources] + [preflight.name],
        "path_token_allowlist": [remote, config["checkpoint_root"], "parent-run", "audit-parameter-cache", "audit.json"]})
    sources.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-near-data-factorial-audit-request-v1", "config": config,
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(), "source_hashes": {target: sha(path) for path, target in sources},
        "execution_environment": B["execution_intent"](), "scheduler_url": parent_request["scheduler_url"],
        "needs": {}, "task_family": "analysis", "device": "cpu", "backend": "independent-numpy", "launcher_sha256": sha(Path(__file__)),
        "parent_collection_sha256": sha(parent / "collection.json"),
        "scheduling_contract": audit_contract(),
        "resource_exception": {key: value for key, value in resources.items() if key != "max_wall_seconds"}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=sources[1:])
    return {"submitted": False, "request_sha256": request["request_sha256"], "run_dir": str(run)}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    if (run / "submission.json").exists():
        raise ValueError("audit already admitted")
    request = read(run / "request.json")
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or request["launcher_sha256"] != sha(Path(__file__))
            or any(sha(run / "payload" / name) != digest for name, digest in request["source_hashes"].items())):
        raise ValueError("staged audit identity changed")
    review = read(run / "long-run-review-final.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("source-bound long-run review required")
    B["scheduler_endpoint"](request)
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"],
        "review_sha256": sha(run / "long-run-review-final.json")})
    r = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-prompt-factorial-audit", task="analysis",
        model=None, payload=run / "payload", pin="beast", gpu=False, needs={}, ram_mb=r["ram_mb"], vram_mb=0,
        cpu_threads=r["cpu_threads"], disk_gb=r["disk_gb"], est_wall_s=r["est_wall_s"], timeout_s=r["timeout_s"],
        priority=50, retry_on_kill=False, preflight=True,
        note=f"CPU-only frozen independent audit of four same-Z0 prompt forecasts:448 learned-weight math transitions,0native framework forwards. Parent/checkpoint Beast-local; authenticated regular-file hardlinks. CPU{r['cpu_threads']} RAM{r['ram_mb']}MB disk{r['disk_gb']}GB expected{r['est_wall_s']}s/timeout{r['timeout_s']}s. Parameter-only read memoization={request['config'].get('parameter_memoization', {}).get('enabled', False)} codec={request['config'].get('parameter_memoization', {}).get('storage_codec', 'float32')}; no numerical function changes, native forwards or retries; own temporary cache removed after verification.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


def decode_capsule(log):
    rows = [line[len(MARKER):] for line in log.splitlines() if line.startswith(MARKER)]
    if len(rows) != 1:
        raise ValueError("one actual audit capsule required")
    meta = json.loads(rows[0])
    if not 0 < meta["bytes"] <= MAX_REPORT or not 0 < meta["gzip_bytes"] <= 8 << 20:
        raise ValueError("audit capsule exceeds bounds")
    compressed = base64.b64decode(meta["base64_gzip"], validate=True)
    if len(compressed) != meta["gzip_bytes"] or hashlib.sha256(compressed).hexdigest() != meta["gzip_sha256"]:
        raise ValueError("compressed report differs")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(MAX_REPORT + 1)
    if len(raw) != meta["bytes"] or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise ValueError("audit report differs")
    return raw, {k: v for k, v in meta.items() if k != "base64_gzip"}


def collect(run):
    from mdb_lanes.job_support import scheduler_api
    run = Path(run).resolve()
    request = read(run / "request.json")
    B["scheduler_endpoint"](request)
    job_id = read(run / "submission.json")["job_id"]
    api = scheduler_api()
    bodies = {}
    for endpoint, name in (("", "job.json"), ("/events", "events.json"), ("/logs?offset=0&wait_s=1", "worker.log")):
        status, body, _ = api.request("GET", "/api/jobs/" + job_id + endpoint, timeout_s=30)
        if status != 200 or len(body) > 24 << 20:
            raise ValueError("audit scheduler evidence unavailable/oversized")
        bodies[name] = body
    job = json.loads(bodies["job.json"])
    if (job["job_id"] != job_id or job["state"] != "succeeded" or job["assigned_host"] != "beast"
            or job["result"]["diagnostics"]["command"] != read(run / "admission-command.json")["command"]):
        raise ValueError("actual audit job/command differs")
    # The qualified producer collector validates request fingerprint and all
    # declared/sealed/executed payload identities; its returned path is unused.
    verifier = runpy.run_path(str(run / "payload/collect_flux_initial_noise.py"))
    verifier["validate_job"](job, job_id, request)
    expected = job["result"]["artifacts"]["job_log"]
    if len(bodies["worker.log"]) != expected["bytes"] or hashlib.sha256(bodies["worker.log"]).hexdigest() != expected["sha256"]:
        raise ValueError("actual audit log hash differs")
    raw, capsule = decode_capsule(bodies["worker.log"].decode())
    result = json.loads(raw)
    execution = result["near_data_execution"]
    if (execution["audit_job_id"] != job_id or execution["audit_request_sha256"] != request["request_sha256"]
            or execution["audit_worker_sha256"] != request["source_hashes"]["audit_flux_prompt_factorial.py"]
            or execution["parent_request_sha256"] != request["config"]["parent_request_sha256"]
            or result["auditor_source_sha256"] != request["source_hashes"]["analyze_flux_prompt_factorial.py"]):
        raise ValueError("independent report/source identity differs")
    for name, body in {**bodies, "audit.json": raw}.items():
        with (run / name).open("xb") as stream:
            stream.write(body)
    receipt = {"schema": "mdb-actual-near-data-audit-collection-v1", "job_id": job_id,
        "request_sha256": request["request_sha256"], "capsule": capsule,
        "bindings": {name: sha(run / name) for name in (*bodies, "audit.json", "request.json", "admission-command.json")},
        "native_framework_forwards_during_collection": 0, "scope": "Actual source-bound CPU audit report transported in authenticated scheduler log; no fabricated archive."}
    save(run / "collection.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--parent", type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--collect", action="store_true")
    memoization = p.add_mutually_exclusive_group()
    memoization.add_argument("--memoize-parameters", dest="memoize_parameters", action="store_true", default=True,
        help="Stage the default bounded immutable parameter read cache; equations remain the original frozen source")
    memoization.add_argument("--no-memoize-parameters", dest="memoize_parameters", action="store_false",
        help="Use the historical uncached profile for explicit reproduction")
    p.add_argument("--parameter-codec", choices=("float32", "bf16-bits"), default="float32",
        help="Parameter cache storage only; bf16-bits rejects values not exactly representable by original BF16 bits")
    a = p.parse_args()
    if sum((a.parent is not None, a.submit, a.collect)) != 1:
        p.error("choose --parent, --submit or --collect")
    print(json.dumps(collect(a.run) if a.collect else submit(a.run) if a.submit else stage(a.run, a.parent,
        memoize_parameters=a.memoize_parameters, parameter_codec=a.parameter_codec)))
