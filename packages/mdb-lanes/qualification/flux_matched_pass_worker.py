"""One finite near-data retrospective pass comparison, without native forwards."""
from __future__ import annotations

import base64
import faulthandler
import gzip
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = "matched-pass-output"
MARKER = "MDB_FACTORIAL_AUDIT_CAPSULE="


def main():
    started = time.monotonic()
    faulthandler.dump_traceback_later(90, repeat=True)
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    helpers = runpy.run_path(str(HERE / "audit_flux_prompt_factorial.py"))
    read, sha = helpers["read"], helpers["sha"]
    collector = runpy.run_path(str(HERE / "collect_flux_prompt_factorial.py"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or config["native_framework_forwards"] != 0
            or config["mathematical_transitions"] != 56):
        raise ValueError("CPU-only two-pass work declaration differs")
    parent = HERE / "parent-run"
    parent_request = read(parent / "request.json")
    source = Path(config["parent_remote_result"])
    actual = collector["validate_job"](read(parent / "collection-job.json"), config["parent_job_id"], parent_request)
    expected = read(parent / "collection-local-inventory.json")
    if str(source) != actual or collector["inventory"](source) != expected:
        raise ValueError("original producer source differs")
    target = parent / "extracted/flux-prompt-factorial-result"
    helpers["hardlink_inventory"](source, target, expected)
    if collector["inventory"](target) != expected:
        raise ValueError("hardlinked producer evidence differs")
    accepted = read(HERE / "parent-audit/audit.json")
    if (sha(HERE / "parent-audit/audit.json") != config["parent_audit_sha256"]
            or accepted["auditor_source_sha256"] != request["source_hashes"]["analyze_flux_prompt_factorial.py"]
            or accepted["near_data_execution"]["audit_job_id"] != config["parent_audit_job_id"]):
        raise ValueError("accepted numerical audit binding differs")
    cache = HERE / "audit-parameter-cache"
    if cache.exists():
        raise ValueError("fresh job-owned parameter cache required")
    core_path = HERE / "flux_matched_pass_replay.py"
    adapter_path = HERE / "memoize_flux_audit_parameters.py"
    if any(sha(path) != request["source_hashes"][path.name] for path in (core_path, adapter_path)):
        raise ValueError("staged retrospective core or I/O adapter changed")

    def progress(value):
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite matched-pass wall budget exhausted")
        print("MDB_MATCHED_PASS_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    report = runpy.run_path(str(core_path))["run"](target,
        analyzer_source=HERE / "analyze_flux_prompt_factorial.py", parameter_root=cache,
        checkpoint=Path(config["checkpoint_root"]), output=HERE / OUTPUT,
        memoization_source=adapter_path, progress=progress,
        expected_analyzer_sha256=request["source_hashes"]["analyze_flux_prompt_factorial.py"],
        expected_memoization_sha256=request["source_hashes"][adapter_path.name],
        max_decoded_bytes=config["max_decoded_bytes"], max_retained_bytes=config["max_retained_bytes"])
    if (report["math_transitions"] != 56 or report["native_forwards"] != 0
            or report["logical_step"] != 1 or report["cells"] != config["conditions"]):
        raise ValueError("actual matched replay census differs from its declaration")
    if collector["inventory"](source) != expected or collector["inventory"](target) != expected:
        raise ValueError("original evidence changed during matched replay")
    released = helpers["release_cache"](cache, read(target / "parameter-artifacts.json"))
    execution = {"schema": "mdb-near-data-matched-pass-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "worker_sha256": sha(Path(__file__)),
        "parent_job_id": config["parent_job_id"], "parent_audit_job_id": config["parent_audit_job_id"],
        "parent_audit_sha256": config["parent_audit_sha256"], "parent_inventory_unchanged": True,
        "mathematical_transitions": 56, "native_framework_forwards": 0,
        "parameter_cache_cleanup": released, "seconds": time.monotonic() - started,
        "scope": "Retrospective native-input numerical decomposition; targets were already observed. No prospective accuracy claim."}
    output = HERE / OUTPUT
    for name, value in (("source.json", {"job_id": execution["job_id"], "request_sha256": request["request_sha256"],
            "source_hashes": request["source_hashes"]}), ("plan.json", config), ("execution.json", execution)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
    inventory = collector["inventory"](output)
    manifest = {"schema": "mdb-matched-pass-manifest-v1", "job_id": execution["job_id"],
        "request_sha256": request["request_sha256"], "files": inventory["files"]}
    with (output / "remote-manifest.json").open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    capsule = {"schema": "mdb-matched-pass-execution-v1", "report": report, "near_data_execution": execution,
        "remote_manifest_sha256": sha(output / "remote-manifest.json"), "output_directory": OUTPUT}
    raw = (json.dumps(capsule, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > 32 << 20:
        raise ValueError("report exceeds transport bound")
    compressed = gzip.compress(raw, mtime=0)
    if len(compressed) > 8 << 20:
        raise ValueError("compressed report exceeds transport bound")
    with (HERE / "matched-pass-result.json").open("xb") as stream:
        stream.write(raw)
    print(MARKER + json.dumps({"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "gzip_bytes": len(compressed),
        "gzip_sha256": hashlib.sha256(compressed).hexdigest(), "base64_gzip": base64.b64encode(compressed).decode()}), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
