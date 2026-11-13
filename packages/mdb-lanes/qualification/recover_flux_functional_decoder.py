"""Recover saved decoder comparisons from a failed reporting job, without models.

The original failed directory is read only. Missing final metadata is derived in
a separate directory from authenticated executed source, closed transactions and
original recipes, with unavailable final resident checks explicitly unclaimed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILED_WORKER = "8de957590606dfbe39fdd5500075151025f266401fdac51238f891bfe13f9067"
FAILED_JOB = "job-406f417b8512"
FAILED_REQUEST = "8740beb5bc338e9ac4fff678a80713f8fb27dd63a336caa52753edb16c671fc8"
RAW_ROOT = "/mnt/big/mdb/research/job-406f417b8512/functional-decoder-result"
UNESTABLISHED = "unestablished; original check after failing catalog was not executed"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def authenticate_packet(packet, original_request, *, synthetic=False):
    auth, job, request = read(packet / "authentication.json"), read(packet / "job.json"), read(original_request)
    require(sha(packet / "job.json") == auth["job_sha256"] and sha(packet / "worker.log") == auth["log_sha256"], "failed job/log packet differs")
    require(job["state"] == auth["original_job_state"] == "failed" and job["result"]["returncode"] == auth["original_returncode"] == 1, "original failed status must remain failed")
    require(identity(request) == request["request_sha256"] == auth["request_sha256"] and request["source_hashes"]["flux_functional_decoder_campaign.py"] == FAILED_WORKER, "executed failed request/source differs")
    if not synthetic:
        require(job["job_id"] == auth["job_id"] == FAILED_JOB and request["request_sha256"] == FAILED_REQUEST and auth["actual_command_and_request_verified"] is True, "authenticated production failure differs")
    custody = job["payload_custody"]
    require(custody["declared"]["sha256"] == custody["sealed"]["sha256"] == custody["executed"]["sha256"] == job["result"]["executed_payload"]["sha256"] == auth["executed_payload_sha256"], "failed executed payload custody differs")
    return auth, job, request


def constants(path):
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def journal_state(transaction, expected_kinds):
    events = [read(path) for path in sorted((transaction / "events").glob("*.json"))]
    failures = [row for row in events if row["kind"] == "failed"]
    violations = [row for row in events if row["kind"] == "violation"]
    seal_violations = read(transaction / "seal.json")["violations"]
    require(not failures and not violations and not seal_violations, "persisted transaction failure/violation cannot be treated as clean")
    kinds = [row["kind"] for row in events]
    require(kinds == expected_kinds, "executed transaction journal occurrence differs")
    return dict(failure=None if not failures else failures[-1]["detail"], violations=[row["detail"] for row in violations] + seal_violations,
        event_kinds=kinds, basis="Absence derived from verified persisted failed/violation event kinds and sealed violation list; not inferred merely from closed phase.")


def prepare(root, metadata, *, inventory_path, expected_manifest, packet, original_request, assets, executed_worker, decoder_helper, synthetic=False):
    """Derive metadata only; no failed-root writes, links, chmod or tensor reads."""
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    root, metadata, packet = Path(root), Path(metadata), Path(packet)
    auth, job, request = authenticate_packet(packet, Path(original_request), synthetic=synthetic)
    require(sha(inventory_path) == expected_manifest and sha(executed_worker) == FAILED_WORKER, "fresh inventory/executed source differs")
    inventory = read(inventory_path)
    require(inventory["job_id"] == job["job_id"] and inventory["request_sha256"] == request["request_sha256"] and inventory["original_job_state"] == "failed", "retained inventory failure identity differs")
    config = read(root / "plan.json")
    expected_config = dict(request["config"], request_sha256=request["request_sha256"])
    if "output" not in expected_config:
        expected_config["output"] = str(root)
    require(config == expected_config, "retained plan differs from failed request")
    require(not any((root / name).exists() for name in ("source.json", "native-contract.json", "summary.json", "files.json")), "recovery expects absent original final exports")
    events = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    seal_event = next(row for row in events if row["kind"] == "cohort_globally_sealed")
    require(seal_event == auth["global_forecast_seal"] and [r for r in events if r["kind"] == "forecast_case_completed"] == auth["observed_forecast_completions"]
        and [r for r in events if r["kind"] == "native_case_completed"] == auth["observed_native_completions"], "saved chronology/authenticated log differs")
    verifies = {name: verify_transaction(root / (name + "-transaction"), expected_seal_sha256=next(row["seal"]["seal_sha256"] for row in events if row["kind"] == kind), max_bytes=256 << 20)
        for name, kind in (("input", "inputs_sealed"), ("forecast", "cohort_globally_sealed"))}
    require(all(row["phase"] == "closed" and row["verified"] for row in verifies.values()), "original saved transactions must be closed")
    journals = {"input": journal_state(root / "input-transaction", ["opened", "forecast_persisted", "globally_sealed", "closed"]),
        "forecast": journal_state(root / "forecast-transaction", ["opened", "forecast_persisted", "globally_sealed", "native_execution_released", "observation_persisted", "closed"])}
    audit = runpy.run_path(str(HERE / "audit_flux_functional_decoder.py"))
    tree = lambda tx, path, key: audit["plain"](load_forecast_tree(root / tx, read(root / tx / path)[key]))
    inputs = tree("input-transaction", "forecasts/inputs.json", "predictions")
    observation = tree("forecast-transaction", "observations/native_decoders_and_optional_diagnostics.json", "value")
    cohort = tree("forecast-transaction", "forecasts/cohort.json", "predictions")
    require(observation["native_macros"] == read(root / "native-macros.json") and cohort["macros"] == read(root / "forecast-macros.json") and cohort["native_decode_calls_at_seal"] == 0, "retained closed observation differs")
    metadata.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(inventory_path, metadata / "retained-files.json")
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        require(sha(assets[key]) == config[key + "_sha256"], "authenticated staged metadata differs")
        shutil.copyfile(assets[key], metadata / Path(config[key]).name)
    manifest = read(assets["parameter_manifest"])
    names = sorted(row["source"]["tensor"] for row in manifest["artifacts"])
    digests = {row["source"]["tensor"]: row["source"]["payload_sha256"] for row in manifest["artifacts"]}
    require(sha(decoder_helper) == config["helper_source_sha256"]["flux_functional_decoder.py"], "unchanged original decoder helper differs")
    helper = constants(decoder_helper)
    origin = "Post-failure derived metadata; not an original runtime final export. Counts are the original persisted execution observation; profile declarations come from authenticated executed source/config."
    contract = dict(metadata_origin=origin, decoder=dict(prior_helper_sha256=helper["PRIOR_HELPER_SHA256"], parameter_names=[n for n in names if not n.startswith("bn.")], files=helper["SOURCE_FILES"], methods=helper["SOURCE_METHODS"], profile=helper["PROFILE"]),
        parameter_names=names, original_parameter_bytes=sum(2 * math.prod(row["source"]["shape"]) for row in manifest["artifacts"]), upstream_source_sha256=config["upstream_source_sha256"],
        dtype="torch.bfloat16", device=config["device"], inference_mode=True, autocast=False, tf32=False, source_reduction_policy=config["source_reduction_policy"], replication=True,
        frontend=dict(input_state_device="cpu", unpack_device="cpu", bn_statistics_device="cpu", bn_sqrt_device="cpu", bn_multiply_device="cpu", bn_add_device="cpu", unpatch_device="cpu", decoder_argument_device=config["device"], transfer_boundary="After CPU unpatch, before post_quant_conv; applies to forecast, native and diagnostic lanes."),
        profile_claim_status="Declared authenticated executed-source path, not a recovered device/flag probe", final_resident_parameter_check=UNESTABLISHED)
    source = dict(schema="mdb-recovered-decoder-source-v1", metadata_origin=origin, producer_status="failed", job_id=job["job_id"], request_sha256=request["request_sha256"], worker_sha256=FAILED_WORKER,
        helper_source_sha256=config["helper_source_sha256"], source_provenance=config["source_provenance"], input_records=inputs["cases"], input_seal=cohort["input_seal"], forecast_seal=seal_event["seal"],
        original_decoder_tensor_sha256=digests, parameter_digest_basis="Standalone original-archive recipe digests; not final resident tensor hashes", final_resident_parameter_check=UNESTABLISHED)
    sidecars = read(root / "comparisons.json")
    summary = dict(schema="mdb-recovered-decoder-summary-v1", metadata_origin=origin, producer_status="failed", job_id=job["job_id"], request_sha256=request["request_sha256"],
        counts=observation["counts"], totals=observation["totals"], input_seal=source["input_seal"], forecast_seal=source["forecast_seal"], cases=list(audit["CASES"]),
        transaction=dict(event_head=verifies["forecast"]["event_head"], **journals["forecast"]),
        complete_macro_cuts={lane: sum(len(panel) for panel in read(root / (lane + "-macros.json")).values()) for lane in ("forecast", "native")},
        first_micro_difference=observation["first_micro_difference"], first_macro_difference=sidecars["first_macro_difference"], diagnostics=observation["diagnostics"],
        act_receipts=0, session_continue_receipts=0, receipt_count_basis="No Session/Act receipts in the declared component-only executed source or retained evidence", final_resident_parameter_check=UNESTABLISHED)
    save(metadata / "source.json", source)
    save(metadata / "native-contract.json", contract)
    save(metadata / "summary.json", summary)
    recovery = dict(schema="mdb-failed-decoder-recovery-metadata-v1", original_job_state="failed", original_returncode=1, original_job_id=job["job_id"], original_request_sha256=request["request_sha256"],
        retained_manifest_sha256=expected_manifest, failed_authentication_sha256=sha(packet / "authentication.json"), failed_job_sha256=sha(packet / "job.json"), failed_log_sha256=sha(packet / "worker.log"),
        failed_executed_payload_sha256=auth["executed_payload_sha256"], final_resident_parameter_check=UNESTABLISHED, original_runtime_metadata_export="absent after optional catalog failure",
        persisted_transaction_health=journals,
        original_recipe_restoration="independent audit will verify source checkpoint bytes; no final resident-version claim", tensor_read_root=str(root), raw_root_mutations=0, raw_tensor_copies=0, new_model_calls=0,
        derived_files={p.name: sha(p) for p in metadata.iterdir() if p.is_file() and p.name != "retained-files.json"})
    save(metadata / "recovery.json", recovery)
    return recovery


def worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    require(request["config"] == config and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "saved-only CPU recovery request differs")
    for name, digest in request["source_hashes"].items():
        require(sha(name) == digest, "staged recovery source differs")
    started = time.monotonic()
    base = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    metadata, output = base / "decoder-recovery-metadata", base / "binding-minimal-result"
    prepare(config["native_root"], metadata, inventory_path=config["retained_inventory"], expected_manifest=config["native_manifest_sha256"], packet=Path(config["failed_packet"]),
        original_request=config["original_request"], assets=config["assets"], executed_worker=config["executed_worker"], decoder_helper=config["decoder_helper"], synthetic=config.get("synthetic_fixture", False))
    audit = runpy.run_path(str(HERE / "audit_flux_functional_decoder.py"))
    result, inventory = audit["audit"](config["native_root"], output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"], metadata_root=metadata)
    root = Path(config["native_root"])
    for name in ("plan.json", "comparisons.json", "signed-responses.json", "forecast-macros.json", "native-macros.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for name in ("source.json", "summary.json", "native-contract.json"):
        shutil.copyfile(metadata / name, output / ("source-" + name))
    shutil.copyfile(metadata / "retained-files.json", output / "source-files.json")
    shutil.copytree(metadata, output / "recovery-metadata")
    shutil.copytree(config["failed_packet"], output / "failed-source-authentication")
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        original = read(root / "plan.json")
        shutil.copyfile(metadata / Path(original[key]).name, output / Path(original[key]).name)
    for directory in ("forecast-images", "native-images"):
        shutil.copytree(root / directory, output / directory)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", dict(host="beast", roots={"native": str(root)}, manifests={"native": {name: row["sha256"] for name, row in inventory["files"].items()}}, scope="Immutable failed raw root; fresh saved-only inventory, not a successful runtime manifest. Full tensors remain near compute."))
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", dict(schema="mdb-functional-decoder-recovery-compact-v1", job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], source_job=result["job"], original_producer_status="failed",
        records=4, original_pngs=8, output_root=str(output), seconds=time.monotonic() - started, model_calls=0, mechanics_status="verified-recovered-saved-evidence", final_resident_parameter_check=UNESTABLISHED))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"], output_root=str(output), files_manifest_sha256=sha(output / "files.json"), summary_sha256=sha(output / "summary.json")), sort_keys=True), flush=True)


def remote_inventory():
    script = r'''import hashlib,json
from pathlib import Path
r=Path(ROOT);rows={}
for p in sorted(r.rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(r.resolve())
 s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 t=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
 rows[p.relative_to(r).as_posix()]=dict(bytes=s.st_size,sha256=h.hexdigest())
print(json.dumps(dict(schema='mdb-failed-decoder-retained-files-v1',job_id=JOB,request_sha256=REQUEST,original_job_state='failed',inventory_stage='Fresh post-failure read-only recovery inventory; not runtime finalization',files=rows),sort_keys=True))
'''
    script = "ROOT=" + repr(RAW_ROOT) + "\nJOB=" + repr(FAILED_JOB) + "\nREQUEST=" + repr(FAILED_REQUEST) + "\n" + script
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"], input=script, text=True, capture_output=True, check=True, timeout=120)
    return json.loads(result.stdout)


def stage(run, study, packet):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, study, packet = map(lambda p: Path(p).resolve(), (run, study, packet))
    auth, _, parent = authenticate_packet(packet, study / "request.json")
    launcher = runpy.run_path(str(HERE / "stage_saved_flux_audit.py"))
    launcher["bind"](parent["execution_environment"])
    audit = runpy.run_path(str(HERE / "audit_flux_functional_decoder.py"))
    run.mkdir(parents=True, exist_ok=False)
    save(run / "retained-files.json", remote_inventory())
    extras = [(HERE / name, name) for name in ("audit_flux_functional_decoder.py", *audit["HELPERS"], "stage_saved_flux_audit.py")]
    for name in ("job.json", "worker.log", "authentication.json"):
        extras.append((packet / name, "failed-source/" + name))
    extras.extend([(study / "request.json", "failed-request.json"), (study / "payload/flux_functional_decoder_campaign.py", "executed-decoder-worker.py"),
        (study / "payload/flux_functional_decoder.py", "executed-decoder-helper.py"), (run / "retained-files.json", "retained-files.json")])
    assets = {}
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        name = Path(parent["config"][key]).name
        require(sha(study / "payload" / name) == parent["config"][key + "_sha256"], "original staged recipe/config differs")
        extras.append((study / "payload" / name, name))
        assets[key] = name
    config = dict(native_root=RAW_ROOT, native_manifest_sha256=sha(run / "retained-files.json"), native_request_sha256=FAILED_REQUEST, source_job=FAILED_JOB,
        retained_inventory="retained-files.json", failed_packet="failed-source", original_request="failed-request.json", assets=assets, executed_worker="executed-decoder-worker.py", decoder_helper="executed-decoder-helper.py",
        output_scratch="/mnt/big/mdb/research", directory_custody=True, source_authentication_sha256=sha(packet / "authentication.json"), scope="Saved-only recovery of4decoder endpoint comparisons; immutable failed root, original failed status retained, closed pretarget forecasts and all fullmacro/micro evidence; finalresidentparametercheck unestablished.")
    save(run / "plan.json", config)
    preflight = run / "preflight.v1.json"
    save(preflight, dict(schema_version="mrun-preflight-config-v1", hashed_roots=[Path(__file__).name, *[name for _, name in extras], preflight.name], path_token_allowlist=[RAW_ROOT, parent["config"]["parameter_root"], "/mnt/big/mdb/research", "binding-minimal-result", "decoder-recovery-metadata"]))
    extras.append((preflight, preflight.name))
    request = dict(schema="mdb-failed-decoder-recovery-request-v1", config=config, experiment="mdb-functional-decoder-recovery", execution_environment=parent["execution_environment"],
        needs={"cpu": True, "host": "beast"}, task_family="analysis", backend="saved-only-numpy", device="cpu", dtype="float64", runtime_wheels=parent["runtime_wheels"],
        source_hashes={name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]}, controller_sha256=sha(HERE / "stage_saved_flux_audit.py"),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=4, logical_branches=1, logical_checkpoint_replays=0, max_local_work_items=32,
            extra={"task_family": "analysis"}, checkpoint_policy="One saved-only CPU recovery lease; no model construction/forward, immutable failed root, no retries."),
        resource_exception=dict(ram_mb=3072, vram_mb=0, cpu_threads=2, disk_gb=1, est_wall_s=180), failed_authentication_sha256=sha(packet / "authentication.json"), original_failed_payload_sha256=auth["executed_payload_sha256"])
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=Path(__file__), config=config, request=request, additional_files=extras)
    for name, digest in request["runtime_wheels"].items():
        require(sha(run / "payload" / name) == digest, "saved-only wheel cohort differs")
    return dict(request_sha256=request["request_sha256"], source_job=FAILED_JOB, original_producer_status="failed", submitted=False)


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--failed-auth", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        if args.submit:
            print(json.dumps(runpy.run_path(str(HERE / "stage_saved_flux_audit.py"))["submit"](args.run, worker=Path(__file__))))
        else:
            if args.study is None or args.failed_auth is None:
                parser.error("failed original study and authenticated failed-source packet required")
            print(json.dumps(stage(args.run, args.study, args.failed_auth)))
