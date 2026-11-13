"""Authenticated, byte-bounded collection of one completed prompt factorial."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import runpy
import shutil
import subprocess
from pathlib import Path

RESULT = "flux-prompt-factorial-result"
MAX_TOTAL, MAX_MEMBER, MAX_FILES = 8 << 30, 1 << 30, 20000
METHOD = "planned-authenticated-ssh-manifest"
LEGACY_SHA256 = "e59cb6e3c1f4f0b03f0163f4db8544952def504090a3fdded8e61f530880b9ad"
HERE = Path(__file__).resolve().parent
legacy = HERE / "collect_flux_initial_noise.py"
if hashlib.sha256(legacy.read_bytes()).hexdigest() != LEGACY_SHA256:
    raise ValueError("qualified collection primitive source differs")
C = runpy.run_path(str(legacy))
sha, relative, preserve, save = (C[key] for key in ("sha", "relative", "preserve", "save"))
CELLS = ("c0-o0", "c0-o1", "c1-o0", "c1-o1")


def inventory(root):
    return C["inventory"](root, max_total=MAX_TOTAL, max_member=MAX_MEMBER, max_files=MAX_FILES)


def validate_job(job, job_id, request):
    prior_path = C["validate_job"](job, job_id, request)
    if not prior_path.endswith("/flux-initial-noise-result"):
        raise ValueError("qualified authenticated working-directory ABI differs")
    return str(Path(prior_path).parent / RESULT)


def verify_manifest(root, actual, job_id, request):
    from mdb_lanes.parameter_artifacts import validate_manifest

    root = Path(root)
    manifest = json.loads((root / "remote-manifest.json").read_text())
    expected = {key: value for key, value in actual["files"].items() if key != "remote-manifest.json"}
    if (set(manifest) != {"schema", "job_id", "request_sha256", "files"}
            or manifest["schema"] != "mdb-step-boundary-remote-manifest-v1"
            or manifest["job_id"] != job_id or manifest["request_sha256"] != request["request_sha256"]
            or manifest["files"] != expected):
        raise ValueError("producer manifest does not cover exact factorial bytes")
    source, summary = [json.loads((root / name).read_text()) for name in ("source.json", "summary.json")]
    if (any(value["job_id"] != job_id or value["request_sha256"] != request["request_sha256"]
            for value in (source, summary)) or source["source_hashes"] != request["source_hashes"]
            or json.loads((root / "plan.json").read_text()) != request["config"]):
        raise ValueError("producer source/summary/plan identity differs")
    for name in ("cohort-seal.json", "parameter-artifacts.json", "retrieve_weights.py",
                 *("transactions/" + cell + "/seal.json" for cell in CELLS)):
        if name not in actual["files"]:
            raise ValueError("factorial forecast/retrieval metadata absent: " + name)
    recipe = validate_manifest(json.loads((root / "parameter-artifacts.json").read_text()))
    if (sha(root / "retrieve_weights.py") != request["config"]["expected_parameter_artifacts_module_sha256"]
            or len(recipe["artifacts"]) != 169
            or any(entry["path"] in actual["files"] for entry in recipe["artifacts"])):
        raise ValueError("exact checkpoint recipes and omitted parameter archives required")
    provenance = recipe.get("provenance", {})
    if (provenance.get("job_id") != job_id or provenance.get("request_sha256") != request["request_sha256"]
            or provenance.get("producer_source_sha256") != request["source_hashes"]["flux_prompt_factorial.py"]):
        raise ValueError("checkpoint recipe producer identity differs")
    return manifest


def remote_script(remote_path):
    header = "import os,stat,json,hashlib\nfrom pathlib import Path,PurePosixPath\n"
    header += f"MAX_TOTAL={MAX_TOTAL}\nMAX_MEMBER={MAX_MEMBER}\nMAX_FILES={MAX_FILES}\n"
    return header + inspect.getsource(C["relative"]) + "\n" + inspect.getsource(C["inventory"]) + (
        "\nprint(json.dumps(inventory(" + repr(remote_path) + "),sort_keys=True))\n")


def collect(run_dir, job_id, *, api=None, runner=None):
    if not isinstance(job_id, str) or not re.fullmatch(r"job-[a-zA-Z0-9_-]+", job_id):
        raise ValueError("invalid job id")
    runner = runner or subprocess.run
    run = Path(run_dir).resolve()
    request = json.loads((run / "request.json").read_text())
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("staged request digest differs")
    if os.environ.setdefault("MRUN_URL", request["scheduler_url"]) != request["scheduler_url"]:
        raise ValueError("scheduler endpoint differs from sealed request")
    config = request["config"]
    if (config["transport_method"] != METHOD
            or config["transport_limits"] != {"max_expanded_bytes": MAX_TOTAL, "max_member_bytes": MAX_MEMBER}):
        raise ValueError("predeclared collection bounds differ")
    for name, digest in request["source_hashes"].items():
        if sha(run / "payload" / relative(name)) != digest:
            raise ValueError("staged producer source changed")
    if api is None:
        from mdb_lanes.job_support import scheduler_api
        api = scheduler_api()
    status, raw, _ = api.request("GET", "/api/jobs/" + job_id, timeout_s=30)
    if status != 200:
        raise ValueError("scheduler job unavailable")
    job = json.loads(raw)
    remote = validate_job(job, job_id, request)
    command = json.loads((run / "admission-command.json").read_text())["command"]
    if job["result"]["diagnostics"]["command"] != command:
        raise ValueError("actual command differs from admitted command")
    preserve(run / "collection-job.json", raw)
    for endpoint, name, bound in (("/events", "collection-events.json", 8 << 20),
                                  ("/logs?offset=0&wait_s=1", "collection-worker.log", 16 << 20)):
        status, body, _ = api.request("GET", "/api/jobs/" + job_id + endpoint, timeout_s=30)
        if status != 200 or len(body) > bound:
            raise ValueError("scheduler evidence missing or oversized: " + name)
        if name.endswith(".log"):
            expected_log = job["result"].get("artifacts", {}).get("job_log", {})
            if expected_log and (expected_log["bytes"] != len(body)
                                or expected_log["sha256"] != hashlib.sha256(body).hexdigest()):
                raise ValueError("scheduler log identity differs")
        preserve(run / name, body)
    script = remote_script(remote)
    remote_command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "beast", "python3", "-"]

    def scan():
        response = runner(remote_command, input=script, text=True, capture_output=True, check=True, timeout=600)
        if len(response.stdout) > 16 << 20:
            raise ValueError("remote inventory oversized")
        value = json.loads(response.stdout)
        if (value["file_count"] != len(value["files"]) or value["file_count"] > MAX_FILES
                or value["expanded_bytes"] != sum(item["bytes"] for item in value["files"].values())
                or value["expanded_bytes"] > MAX_TOTAL):
            raise ValueError("remote inventory bounds differ")
        for name, item in value["files"].items():
            relative(name)
            if (set(item) != {"sha256", "bytes", "filetype"} or item["filetype"] != "regular"
                    or type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_MEMBER
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("invalid remote member")
        return value

    before = scan()
    save(run / "collection-remote-before.json", before)
    if shutil.disk_usage(run).free < before["expanded_bytes"] + (128 << 20):
        raise ValueError("insufficient local collection space")
    destination = run / "extracted" / RESULT
    destination.mkdir(parents=True, exist_ok=False)
    allowlist = run / "collection-file-list.bin"
    preserve(allowlist, b"".join(name.encode() + b"\0" for name in before["files"]))
    copied = runner(["rsync", "--recursive", "--times", "--protect-args", "--no-links", "--no-devices", "--no-specials",
        "--from0", "--files-from=" + str(allowlist), "--max-size=" + str(MAX_MEMBER), "--timeout=120",
        "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes", "beast:" + remote + "/", str(destination) + "/"],
        text=True, capture_output=True, check=True, timeout=1800)
    save(run / "collection-copy.json", {"returncode": copied.returncode, "stdout": copied.stdout, "stderr": copied.stderr})
    after = scan()
    save(run / "collection-remote-after.json", after)
    local = inventory(destination)
    if before != after or local != before:
        raise ValueError("remote/local evidence changed during collection")
    verify_manifest(destination, local, job_id, request)
    save(run / "collection-local-inventory.json", local)
    for name in ("parameter-artifacts.json", "retrieve_weights.py"):
        preserve(run / name, (destination / name).read_bytes())
    receipt = {"schema": "mdb-prompt-factorial-ssh-collection-v1", "collection_mode": METHOD,
        "job_id": job_id, "scheduler_status": job["state"], "request_sha256": request["request_sha256"],
        "remote_path": remote, "file_count": local["file_count"], "expanded_bytes": local["expanded_bytes"],
        "archive_sha256": None, "model_calls_during_collection": 0,
        "executed_payload_sha256": job["result"]["executed_payload"]["sha256"],
        "manifest_sha256": sha(destination / "remote-manifest.json"), "summary_sha256": sha(destination / "summary.json"),
        "cohort_seal_sha256": sha(destination / "cohort-seal.json"),
        "seals": {cell: sha(destination / "transactions" / cell / "seal.json") for cell in CELLS},
        "limits": {"max_expanded_bytes": MAX_TOTAL, "max_member_bytes": MAX_MEMBER, "max_files": MAX_FILES},
        "collector_source_sha256": sha(Path(__file__)), "legacy_collection_primitive_sha256": LEGACY_SHA256,
        "remote_inventory_source_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "bindings": {name: sha(run / name) for name in ("request.json", "collection-job.json", "collection-events.json",
            "collection-worker.log", "collection-remote-before.json", "collection-remote-after.json", "collection-local-inventory.json",
            "collection-copy.json", "collection-file-list.bin", "admission-command.json", "parameter-artifacts.json", "retrieve_weights.py")},
        "scope": "Predeclared authenticated direct collection; unchanged producer bytes and inventories. Numerical audit is separate; nonlandmark native manifests/witnesses are metadata-only, not complete replay."}
    save(run / "collection.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("job_id")
    args = parser.parse_args()
    print(json.dumps(collect(args.run_dir, args.job_id)))
