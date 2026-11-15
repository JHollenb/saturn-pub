"""Saved-only packaging recovery; the original failed producer remains failed.

The original regular files are hard-linked into a separately named same-device
view. Only that view receives the frozen finalizer's missing wrapper manifest.
All reductions run through the unchanged saved-only auditor, never a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import stat
import time
from collections import Counter
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
SCHEMA = "mdb-flux-raw-packaging-recovery-v1"
INVENTORY_SCHEMA = "mdb-failed-raw-to-image-files-v1"
MEMBER_BYTES = (128 << 20) + 4096
EXPANDED_BYTES = 4 << 30
FINALIZER_SHA256 = "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926"
AUDITOR_SHA256 = "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835"
CORE = ("plan.json", "request.json", "source.json", "summary.json", "native-contract.json")
COMPACT = (*CORE, "files.json", "runtime-contract.json", "timeline.jsonl",
    "forecast-qwen.json", "native-qwen.json", "frontend-forecast.json", "frontend-native.json",
    "forecast-state-rows.json", "native-state-rows.json", "decoder-forecast.json",
    "decoder-native-own.json", "decoder-native-reference.json", "comparisons.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def member(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name and relative.parts
        and not any(part in {".", ".."} for part in relative.parts), "unsafe retained member")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "retained member escapes root")
    return path


def regular_members(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "original/view root must be a real directory")
    result = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), "symlink forbidden in retained evidence")
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "nonregular retained evidence forbidden")
        if stat.S_ISREG(mode):
            result[path.relative_to(root).as_posix()] = path
    return result


def inventory(root, *, job_id, request_sha256):
    """Read-only stdlib inventory, suitable for authenticated SSH stdout."""
    rows = {}
    for name, path in regular_members(root).items():
        before = path.stat()
        rows[name] = {"sha256": sha(path), "bytes": before.st_size}
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "source changed during inventory")
    return dict(schema=INVENTORY_SCHEMA, job_id=job_id, request_sha256=request_sha256,
        original_job_state="failed", inventory_stage="Post-failure read-only custody inventory, not producer finalization",
        files=rows, expanded_bytes=sum(row["bytes"] for row in rows.values()))


def verify_inventory(root, expected):
    files = regular_members(root)
    require("files.json" not in files, "original failed tree already has a wrapper manifest")
    require(set(files) == set(expected["files"]), "retained member set differs")
    snapshot = {}
    for name, row in expected["files"].items():
        path = member(root, name)
        before = path.stat()
        require(before.st_size == row["bytes"] and sha(path) == row["sha256"], "retained evidence bytes differ: " + name)
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "retained evidence changed while hashing")
        snapshot[name] = dict(device=after.st_dev, inode=after.st_ino, bytes=after.st_size,
            mtime_ns=after.st_mtime_ns, nlink=after.st_nlink)
    require(sum(row["bytes"] for row in expected["files"].values()) == expected["expanded_bytes"], "inventory expanded bytes differ")
    return snapshot


def authenticate(config):
    """Join authenticated scheduler bytes to the untouched completed core evidence."""
    from mdb.forecast_transaction import verify_transaction

    require(config["schema"] == SCHEMA and config["max_member_bytes"] == MEMBER_BYTES
        and config["max_expanded_bytes"] == EXPANDED_BYTES, "recovery scope/bounds differ")
    for key in ("retained_inventory", "original_job", "original_log", "original_request"):
        require(sha(config[key]) == config[key + "_sha256"], "bound recovery input differs: " + key)
    root = Path(config["original_root"])
    job, request, retained = (read(config[key]) for key in ("original_job", "original_request", "retained_inventory"))
    require(job["job_id"] == config["expected_source_job_id"] and job["state"] == "failed"
        and job["result"]["returncode"] == 1, "original failed job must remain failed")
    require(identity(request) == request["request_sha256"] == config["expected_source_request_sha256"], "original request identity differs")
    require(all(job["config"][key] == value for key, value in request.items())
        and job["cmd"] == job["submitted_request"]["cmd"] == job["result"]["diagnostics"]["command"], "executed original request/command differs")
    custody = job["payload_custody"]
    require(custody["declared"]["sha256"] == custody["sealed"]["sha256"] == custody["executed"]["sha256"]
        == job["result"]["executed_payload"]["sha256"] == config["expected_source_payload_sha256"], "executed payload custody differs")
    log = Path(config["original_log"]).read_text()
    require("ValueError: output member bound exceeded" in log and "MDB_DIRECTORY_READY " not in log,
        "original failure is not the declared finalizer-only failure")
    require(retained["schema"] == INVENTORY_SCHEMA and retained["job_id"] == job["job_id"]
        and retained["request_sha256"] == request["request_sha256"] and retained["original_job_state"] == "failed", "retained inventory custody differs")
    snapshot = verify_inventory(root, retained)
    require(retained["expanded_bytes"] <= EXPANDED_BYTES
        and max(row["bytes"] for row in retained["files"].values()) <= MEMBER_BYTES, "retained tree exceeds bounded recovery")
    plan, raw_request, source, summary, contract = (read(root / name) for name in CORE)
    require(raw_request == request and all(plan[key] == value for key, value in request["config"].items()), "raw executed request/plan differs")
    require(all(value["job_id"] == job["job_id"] and value["request_sha256"] == request["request_sha256"]
        for value in (plan, source, summary)), "original core job/request join differs")
    require(plan["output"] == str(root) and summary["mechanics_status"] == "complete", "original completed science/output identity differs")
    require(source["worker_sha256"] == request["source_hashes"]["flux_raw_to_image.py"]
        == config["expected_source_worker_sha256"], "original producer source differs")
    require(source["helper_source_sha256"] == plan["helper_source_sha256"]
        and all(request["source_hashes"][name] == digest for name, digest in source["helper_source_sha256"].items()), "original helper source joins differ")
    require(source["physical_counts"] == summary["counts"] == contract["counts"], "completed physical census sidecars differ")
    require(source["forecast_seal"] == summary["forecast_seal"] == contract["forecast_seal"], "original forecast seals differ")
    for component in ("qwen", "flux", "vae"):
        before, after = (source[component + "_parameter_identity_" + side] for side in ("before", "after"))
        require(before and before == after and contract[component + "_identity_before"] == before
            and contract[component + "_identity_after"] == after, "saved final resident identity check differs: " + component)
    require(source["qwen_tied_alias"] is contract["qwen_tied_alias"] is True, "saved final tied alias differs")
    require(source["placement_records"] == contract["placement_records"], "saved placement identity sidecars differ")
    require(sha(root / "runtime-contract.json") == source["runtime_contract_sha256"], "runtime contract hash differs")
    for records, path_key in ((source["source_files"], "path"), (source["tokenizer_files"], "retained")):
        for row in records.values():
            path = member(root, row[path_key])
            require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "original source/tokenizer metadata differs")
    transaction = root / "forecast-transaction"
    verified = verify_transaction(transaction, expected_seal_sha256=summary["forecast_seal"]["seal_sha256"],
        expected_event_head=summary["transaction"]["event_head"], max_bytes=plan["forecast_transaction_bytes"])
    require(verified["verified"] and verified["phase"] == "closed", "original transaction is not verified closed")
    events = [read(path) for path in sorted((transaction / "events").glob("*.json"))]
    kinds = [event["kind"] for event in events]
    require(kinds == ["opened", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"]
        and not read(transaction / "seal.json")["violations"], "persisted transaction failure/violation or chronology differs")
    return retained, snapshot, dict(original_job_id=job["job_id"], original_request_sha256=request["request_sha256"],
        original_producer_status="failed", original_returncode=1, original_failure="output member bound exceeded",
        original_job_sha256=config["original_job_sha256"], original_log_sha256=config["original_log_sha256"],
        original_request_file_sha256=config["original_request_sha256"], original_payload_sha256=config["expected_source_payload_sha256"],
        original_worker_sha256=source["worker_sha256"], final_resident_parameter_check="established-by-saved-before-after-core-records",
        transaction_verification=verified, transaction_event_kinds=kinds,
        transaction_health_basis="Verified persisted complete event chain contains no failed/violation events; seal violations empty.",
        final_parameter_scope="Runtime before/after identities retained before packaging failure; frozen saved auditor independently restores original recipes.")


def prepare(config, view, receipt_path, *, recovery_identity=None):
    """Verify, hard-link and finalize a new view; never write an original member."""
    retained, snapshot, receipt = authenticate(config)
    root, view, receipt_path = Path(config["original_root"]), Path(view), Path(receipt_path)
    require(not view.exists() and not view.is_symlink() and not receipt_path.exists(), "recovery destination already exists")
    require(not view.resolve().is_relative_to(root.resolve()) and not root.resolve().is_relative_to(view.resolve())
        and not receipt_path.resolve().is_relative_to(root.resolve()) and not receipt_path.resolve().is_relative_to(view.resolve()), "recovery evidence must be outside original/view")
    view.parent.mkdir(parents=True, exist_ok=True)
    require(view.parent.stat().st_dev == root.stat().st_dev, "hard-link recovery requires same filesystem")
    finalizer = Path(config["finalizer"])
    require(config["finalizer_sha256"] == FINALIZER_SHA256 and sha(finalizer) == FINALIZER_SHA256,
        "frozen finalizer source differs")
    view.mkdir()
    for name in retained["files"]:
        original, target = member(root, name), member(view, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(original, target, follow_symlinks=False)
        current, linked = original.stat(), target.stat()
        require((current.st_dev, current.st_ino) == (linked.st_dev, linked.st_ino)
            == (snapshot[name]["device"], snapshot[name]["inode"]), "hard-link inode/device join differs")
    ready = runpy.run_path(str(finalizer))["finalize"](view, job_id=receipt["original_job_id"],
        request_sha256=receipt["original_request_sha256"], max_member_bytes=MEMBER_BYTES, max_expanded_bytes=EXPANDED_BYTES)
    wrapper = read(view / "files.json")
    require(wrapper["files"] == retained["files"] and wrapper["expanded_bytes"] == retained["expanded_bytes"], "derived wrapper does not exactly cover original bytes")
    after = verify_inventory(root, retained)
    multiplicity = Counter((row["device"], row["inode"]) for row in snapshot.values())
    require(all(all(after[name][key] == row[key] for key in ("device", "inode", "bytes", "mtime_ns"))
        and after[name]["nlink"] == row["nlink"] + multiplicity[(row["device"], row["inode"])]
        for name, row in snapshot.items()), "original contents/mtime/inode or expected link delta changed")
    receipt.update(schema="mdb-flux-raw-packaging-recovery-receipt-v1", original_root=str(root), derived_view=str(view),
        retained_inventory_sha256=config["retained_inventory_sha256"], original_file_count=len(retained["files"]),
        original_expanded_bytes=retained["expanded_bytes"], original_largest_member_bytes=max(row["bytes"] for row in retained["files"].values()),
        wrapper=ready, max_member_bytes=MEMBER_BYTES, max_expanded_bytes=EXPANDED_BYTES,
        finalizer_sha256=FINALIZER_SHA256, new_neural_calls=0, new_parameter_hydrations=0,
        recovery_execution=recovery_identity, original_shared_inode_groups=sum(count > 1 for count in multiplicity.values()),
        custody_status="derived-hardlink-view-finalized; original producer remains failed",
        original_mutation_scope="No original file/directory writes, chmod, symlinks or content copies. Hard-link creation necessarily increments file link count and changes inode ctime; contents, mtime and inode preserved and rehashed.")
    save(receipt_path, receipt)
    return receipt


def compact(view, output, config, request, receipt_path):
    """Use unchanged audit mathematics; export only small metadata and PNGs."""
    view, output = Path(view), Path(output)
    require(config["auditor_sha256"] == AUDITOR_SHA256 and sha(config["auditor"]) == AUDITOR_SHA256,
        "frozen saved auditor source differs")
    auditor = runpy.run_path(config["auditor"])
    result, files = auditor["audit"](view, output, expected_manifest=sha(view / "files.json"),
        expected_request=config["expected_source_request_sha256"])
    for name in COMPACT:
        shutil.copyfile(view / name, output / ("source-" + name))
    for key, row in result["source"]["source_files"].items():
        target = output / "source-metadata" / (key + "-" + Path(row["path"]).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(view / row["path"], target)
    for lane, name in (("forecast", "decoder-forecast.json"), ("native-own", "decoder-native-own.json"), ("native-reference", "decoder-native-reference.json")):
        shutil.copyfile(view / read(view / name)["png"]["path"], output / (lane + ".png"))
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    shutil.copyfile(receipt_path, output / "recovery-receipt.json")
    for key, name in (("original_job", "original-failed-job.json"), ("original_log", "original-failed-worker.log"),
        ("retained_inventory", "original-retained-inventory.json")):
        shutil.copyfile(config[key], output / name)
    save(output / "external-evidence.json", dict(host="beast", roots=dict(native=str(view)),
        manifests=dict(native={name: row["sha256"] for name, row in files["files"].items()}),
        original_immutable_root=config["original_root"], custody="Derived hardlink view; only its wrapper is newly finalized. Original producer failed packaging.",
        scope="All saved full tensors remain near data. No model calls or new scientific outcomes."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-flux-raw-to-image-recovery-compact-v1", job_id=os.environ["MRUN_JOB_ID"],
        request_sha256=request["request_sha256"], source_job=config["expected_source_job_id"], original_producer_status="failed",
        derived_view=str(view), recovery_receipt_sha256=sha(receipt_path), records=2, model_calls=0,
        mechanics_status="verified-saved-evidence-from-derived-custody", final_resident_parameter_check=read(receipt_path)["final_resident_parameter_check"]))
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output),
        files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json"))


def worker(config_path):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    started = time.monotonic()
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    config = read(config_path)
    require(identity(request) == request["request_sha256"] and request["config"] == config
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU request/config differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged recovery source differs: " + name)
    base = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    view, receipt_path = base / config.get("view_name", "raw-to-image-recovery-view"), base / "recovery-receipt.json"
    receipt = prepare(config, view, receipt_path, recovery_identity={"job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "worker_sha256": sha(__file__)})
    print("MDB_RAW_PACKAGING_RECOVERY " + json.dumps(dict(view=str(view), receipt_sha256=sha(receipt_path),
        original_producer_status="failed", source_job=receipt["original_job_id"], seconds=time.monotonic() - started), sort_keys=True), flush=True)
    ready = compact(view, base / "binding-minimal-result", config, request, receipt_path)
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"))
    args = parser.parse_args()
    if not args.config:
        parser.error("--config or SATURN_RUN_CONFIG is required")
    worker(args.config)
