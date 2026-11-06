"""Audit sealed image-reader forecasts near saved arrays; collect compact evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPERS = ("audit_flux_image_tail_microscope.py", "analyze_flux_reader_arithmetic.py",
           "flux_source_receiver.py", "retrieve_reader_evidence.py", "flux_family_custody.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def audit(root, output, *, expected_manifest, expected_request):
    import numpy as np
    from mdb.equation_comparison import EquationCatalog, compare_equations
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    root, output = Path(root), Path(output)
    inventory = runpy.run_path(str(HERE / "audit_flux_image_tail_microscope.py"))["validate_inventory"](root, expected_manifest)
    measure = runpy.run_path(str(HERE / "analyze_flux_reader_arithmetic.py"))["measure"]
    load = runpy.run_path(str(HERE / "flux_source_receiver.py"))["load_array"]
    summary, config, source, inputs = (read(root / name) for name in ("summary.json", "plan.json", "source.json", "input-seal.json"))
    if (any(row["request_sha256"] != expected_request for row in (summary, config, source, inventory))
            or len({summary["job_id"], source["job_id"], inventory["job_id"]}) != 1
            or summary["input_seal_sha256"] != sha(root / "input-seal.json")
            or source["input_seal_sha256"] != summary["input_seal_sha256"]):
        raise ValueError("native forecast lineage differs")
    for role in ("sham", "edit"):
        if (source["source_cuts"][role]["cut"] != config["inputs"][role]["cut"]
                or inputs["original_source_cuts"][role] != config["inputs"][role]["cut"]):
            raise ValueError("source role join differs")
    if inputs["target_entries"] != 0 or inputs["prefix_entries"] != 0 or inputs["recipe"] != config["midpoint_recipe"]:
        raise ValueError("input recipe was not sealed before observation")
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    kinds = [row["kind"] for row in timeline]
    if kinds != ["input_sealed", "prefix_stopped", "prefix_stopped", "image_cohort_sealed", "target_completed", "target_completed"]:
        raise ValueError("forecast/target sequence differs")
    if any(row["target_entries"] != 0 for row in timeline[:4]):
        raise ValueError("complete target entered before prediction seal")
    transaction = root / "transaction"
    verified = verify_transaction(transaction, expected_seal_sha256=summary["forecast_seal_sha256"], max_bytes=config["transaction_max_bytes"])
    encoded = read(transaction / "forecasts/image_reader.json")
    predictions = load_forecast_tree(transaction, encoded["predictions"])
    frontier_record = read(transaction / "frontiers/observed_qkv.json")
    frontiers = load_forecast_tree(transaction, frontier_record["value"])
    if predictions["target_entries_before_seal"] != 0:
        raise ValueError("target exposure before forecast seal")
    targets = {role: read(root / "targets" / (role + ".json")) for role in ("sham", "midpoint")}
    prefixes = {role: read(root / "prefixes" / (role + ".json")) for role in targets}
    errors, ports, qkv = {}, {}, {}
    native_images = {}
    for role, target in targets.items():
        expected_cut = inputs["inputs"][role]["cut"]
        if any(row["input_cut"] != expected_cut for row in (target, prefixes[role], predictions["cases"][role])):
            raise ValueError("forecast source/case join differs")
        frontier = frontiers[role]
        if frontier["input_cut"] != expected_cut or frontier["input_seal_sha256"] != summary["input_seal_sha256"]:
            raise ValueError("sealed frontier source join differs")
        if not np.array_equal(frontier["image"], load(root, inputs["inputs"][role]["image"])):
            raise ValueError("sealed frontier image differs from input seal")
        for name in ("q", "k", "v"):
            if not np.array_equal(frontier["values"][name], load(root, prefixes[role]["values"][name])):
                raise ValueError("sealed frontier QKV differs from prefix sidecar")
        counts = prefixes[role]["counts"]
        if any(counts[key] != 0 for key in ("native_attention", "completed_blocks", "output_projection", "mlp")):
            raise ValueError("prefix computed a future consumer")
        if prefixes[role]["original_dispatch_called"] or prefixes[role]["native_complete_outputs_computed"]:
            raise ValueError("prefix computed native output")
        native_images[role] = load(root, target["actual"]["image"])
        errors[role] = measure(native_images[role], predictions["cases"][role]["image"])
        ports[role] = {port: measure(load(root, target["actual"][port]), load(root, value))
                       for port, value in predictions["cases"][role]["primitive_ports"].items()}
        qkv[role] = {name: measure(load(root, target["actual"][name]), load(root, prefixes[role]["values"][name])) for name in ("q", "k", "v")}
    native_effect = native_images["midpoint"].astype(np.float64) - native_images["sham"].astype(np.float64)
    predicted_effect = predictions["cases"]["midpoint"]["image"].astype(np.float64) - predictions["cases"]["sham"]["image"].astype(np.float64)
    effect = measure(native_effect, predicted_effect)
    controls = {}
    wrong = predictions["controls"].get("wrong_fixed_head")
    if wrong is not None:
        controls = {"endpoint": measure(native_images["midpoint"], wrong["image"]),
                    "effect": measure(native_effect, wrong["image"].astype(np.float64) - predictions["cases"]["sham"]["image"].astype(np.float64))}
    result = {"schema": "mdb-image-reader-audit-v1", "job": summary["job_id"], "native_request_sha256": expected_request,
        "native_manifest_sha256": expected_manifest, "input_seal_sha256": summary["input_seal_sha256"],
        "forecast_seal_sha256": summary["forecast_seal_sha256"], "transaction_verification": verified,
        "errors": errors, "port_errors": ports, "prefix_replay": qkv, "response": effect,
        "wrong_fixed_head": controls, "midpoint_input_observation": inputs["input_comparisons"],
        "novel_input": inputs["novel_input"],
        "historical_sham_replay": targets["sham"]["historical_replay"],
        "prefix_counts": summary["prefix_counts"], "forecast_primitive_counts": summary["forecast_primitive_counts"],
        "target_counts": summary["target_counts"], "native_contract": read(root / "native-contract.json"),
        "selected_aten_operations": {role: {"forecast": [dict(item) for item in predictions["cases"][role]["selected_aten_operations"]],
            "target": target["selected_aten_operations"]} for role, target in targets.items()},
        "orientation": "native minus candidate; paired effect is midpoint minus sham",
        "new_model_calls": 0, "new_reference_forward_calls": 0, "mechanics_status": "authenticated-saved-array-audit",
        "terminal_status": "unassessed", "auditor_sha256": sha(__file__),
        "limitations": ["One new numerical midpoint and one known sham, fixed receiver/clock.",
            "Predictions use observed native positioned QKV and native SDPA/tail primitives.",
            "No independent Flash arithmetic, upstream full-block forecast, suffix, VAE or semantic generalization."]}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    catalog, queries = EquationCatalog(output / "catalog/equations"), []

    def record(case, lane, predicted, actual, kind):
        context = {"model_identity": config["expected_model_identity"], "case": case, "lane": lane,
            "comparison": kind, "input_seal_sha256": summary["input_seal_sha256"],
            "forecast_seal_sha256": summary["forecast_seal_sha256"],
            "clock": {"step": 1, "phase": "joint", "index": 0},
            "observed_frontier": {role: prefixes[role]["values"] for role in targets},
            "native_contract": result["native_contract"], "horizon": 1}
        item = catalog.record(law_id="law.flux.image-reader.native-primitives.v1",
            route_id="route.flux.joint0.image-attention-residual-mlp.v1",
            alias="cmp.flux.image-reader." + case + "." + lane + "." + kind,
            binding={"model_identity": config["expected_model_identity"], "input_identity": summary["input_seal_sha256"],
                "source_identity": expected_request, "execution": result["native_contract"],
                "frontier": {"clock": context["clock"], "positioned_qkv": context["observed_frontier"]}, "horizon": 1},
            report=compare_equations({"image": predicted}, {"image": actual}, max_elements=25_000_000),
            conditions=[{"field": key, "op": "eq", "value": value} for key, value in context.items()], tested_context=context,
            readers=["joint0.image-reader"], evidence={"audit_sha256": sha(output / "audit.json"),
                "role": "prospective conditional native-primitive composition", "new_learned_parameters": 0,
                "independent_kernel_arithmetic": False, "quality_gate": False})
        changed = {**context, "clock": {"step": 2, "phase": "joint", "index": 0}}
        exact, wrong_clock = len(catalog.find(applicable_to=context)), len(catalog.find(applicable_to=changed))
        if exact != 1 or wrong_clock != 0:
            raise ValueError("equation query mechanics differ")
        queries.append({"fingerprint": item["fingerprint"], "case": case, "lane": lane, "kind": kind,
                        "exact": exact, "wrong_clock": wrong_clock})
        if case == "midpoint" and lane == "own-head":
            save(output / "catalog/context-exact.json", context)
            save(output / "catalog/context-wrong-clock.json", changed)

    for role in targets:
        record(role, "own-head", predictions["cases"][role]["image"], native_images[role], "endpoint")
    record("paired", "own-head", predicted_effect, native_effect, "effect")
    if wrong is not None:
        record("midpoint", "fixed-sham-head", wrong["image"], native_images["midpoint"], "endpoint")
        record("paired", "fixed-sham-head", wrong["image"].astype(np.float64) - predictions["cases"]["sham"]["image"].astype(np.float64), native_effect, "effect")
    save(output / "catalog/provenance.json", {"records": len(queries), "queries": queries, "family_induction": False,
        "source_native_manifest_sha256": expected_manifest, "new_model_calls": 0, "applicability": "supplied exact predicates"})
    return result, len(queries), inventory


def worker():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    request, config = read(os.environ["SATURN_DEBUG_REQUEST"]), read(os.environ["SATURN_RUN_CONFIG"])
    if request["config"] != config or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU audit environment differs")
    for name, digest in request["source_hashes"].items():
        if sha(name) != digest:
            raise ValueError("audit source changed")
    started, root = time.monotonic(), Path(config["native_root"])
    output = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "binding-minimal-result"
    result, count, inventory = audit(root, output, expected_manifest=config["native_manifest_sha256"], expected_request=config["native_request_sha256"])
    for name in ("summary.json", "source.json", "files.json", "input-seal.json", "native-contract.json", "transaction-report.json", "transaction-verification.json", "timeline.jsonl"):
        shutil.copyfile(root / name, output / ("source-" + name))
    for name in ("parameter-artifacts.json", "retrieve_weights.py", "source-binding.json"):
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(HERE / "retrieve_reader_evidence.py", output / "retrieve_evidence.py")
    save(output / "external-evidence.json", {"host": "beast", "roots": {"native": str(root)},
        "manifests": {"native": {name: value["sha256"] for name, value in inventory["files"].items()}},
        "scope": "Full native operands/predictions/outcomes remain near data; individual checksum retrieval."})
    save(output / "request.json", request)
    save(output / "plan.json", config)
    save(output / "summary.json", {"schema": "mdb-image-reader-compact-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "source_job": result["job"], "records": count,
        "output_root": str(output), "seconds": time.monotonic() - started, "model_calls": 0,
        "reference_forward_calls": 0, "mechanics_status": "verified-saved-array-custody", "terminal_status": "unassessed"})
    save(output / "files.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps({"job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "output_root": str(output), "files_manifest_sha256": sha(output / "files.json"), "summary_sha256": sha(output / "summary.json")}, sort_keys=True), flush=True)


def stage(run, study):
    from mdb.job_support import one_lease_scheduling_contract, scheduler_api
    from mdb.payload import stage_worker
    parent = read(study / "request.json")
    runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))["bind"](parent["execution_environment"])
    job_id = read(study / "submission.json")["job_id"]
    status, raw, _ = scheduler_api().request("GET", "/api/jobs/" + job_id, timeout_s=30)
    job = json.loads(raw)
    if status != 200 or job["state"] != "succeeded" or job["result"]["diagnostics"]["command"] != read(study / "admission-command.json")["command"]:
        raise ValueError("completed exact native command required")
    collector = HERE / "collect_flux_initial_noise.py"
    if sha(collector) != "ce3930b71b58dab832c09b7d4b0c4e3eefb8bc6435a72e4605992d415be38765":
        raise ValueError("qualified scheduler validator changed")
    runpy.run_path(str(collector))["validate_job"](job, job_id, parent)
    status, log, _ = scheduler_api().request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected = job["result"]["artifacts"]["job_log"]
    if status != 200 or len(log) > 4 << 20 or len(log) != expected["bytes"] or hashlib.sha256(log).hexdigest() != expected["sha256"]:
        raise ValueError("native log checksum differs")
    markers = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    if len(markers) != 1:
        raise ValueError("one native publication required")
    save(study / "job.json", job)
    with (study / "worker.log").open("xb") as stream:
        stream.write(log)
    root = Path(parent["config"]["output_scratch"]) / job_id / "binding-minimal-result"
    script = "import json,hashlib\nfrom pathlib import Path\np=Path(" + repr(str(root)) + ")\nprint(json.dumps(dict(manifest_sha256=hashlib.sha256((p/'files.json').read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((p/'summary.json').read_bytes()).hexdigest(),summary=json.loads((p/'summary.json').read_text()))))\n"
    info = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"], input=script, text=True, capture_output=True, check=True, timeout=60).stdout)
    marker = markers[0]
    if (info["summary"]["job_id"] != job_id or info["summary"]["request_sha256"] != parent["request_sha256"]
            or marker["job_id"] != job_id or marker["request_sha256"] != parent["request_sha256"]
            or marker["output_root"] != str(root) or marker["files_manifest_sha256"] != info["manifest_sha256"]
            or marker["summary_sha256"] != info["summary_sha256"]):
        raise ValueError("native publication/directory identity differs")
    run.mkdir(parents=True, exist_ok=False)
    config = {"native_root": str(root), "native_manifest_sha256": info["manifest_sha256"],
        "native_request_sha256": parent["request_sha256"], "source_job": job_id,
        "output_scratch": "/mnt/big/mdb/research", "directory_custody": True,
        "scope": "Saved-only full-array sealed forecast/target audit; zero model or reference recurrence calls."}
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in HELPERS]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [Path(__file__).name, *HELPERS, preflight.name],
        "path_token_allowlist": [str(root), config["output_scratch"], "binding-minimal-result"]})
    extras.append((preflight, preflight.name))
    request = {"schema": "mdb-image-reader-audit-request-v1", "config": config, "execution_environment": parent["execution_environment"],
        "needs": {"cpu": True, "host": "beast"}, "task_family": "analysis", "backend": "numpy", "device": "cpu", "dtype": "float64",
        "source_hashes": {name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        "runtime_wheels": parent["runtime_wheels"],
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=5, logical_branches=1,
            logical_checkpoint_replays=0, max_local_work_items=16, extra={"task_family": "analysis"},
            checkpoint_policy="One full-array CPU forecast audit; compact catalog/retriever collection; no native/reference forwards or retries."),
        "resource_exception": {"ram_mb": 3072, "vram_mb": 0, "cpu_threads": 2, "disk_gb": 1, "est_wall_s": 120}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=Path(__file__), config=config, request=request, additional_files=extras)
    for name, digest in request["runtime_wheels"].items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("audit wheel cohort differs")
    return {"request_sha256": request["request_sha256"], "source_job": job_id}


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    request = read(run / "request.json")
    runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))["bind"](request["execution_environment"])
    if (run / "submission.json").exists():
        raise ValueError("audit already submitted")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("audit request changed")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("audit payload changed")
    command = worker_command(Path(__file__).name)
    command.insert(1, "CUDA_VISIBLE_DEVICES=")
    command.insert(command.index("python") + 1, "-B")
    save(run / "admission-command.json", {"command": command, "request_sha256": request["request_sha256"]})
    resource = request["resource_exception"]
    job = guarded_mrun_launch(command, request=request, experiment="mdb-image-reader-audit", task="analysis", model=None,
        payload=run / "payload", needs=request["needs"], gpu=False, prefer_host="beast", timeout_s=360, retry_on_kill=False, preflight=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s")},
        note="Saved-only image-reader audit:full sealed forecasts/prefixQKV/native returns, wrongfixedhead control and signed paired effects;5curated equationrecords. CPU2/RAM3GB,120sestimate/360souter;zero new native/reference forwards. Rawarrays remainBeast;compactcustody only.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run", type=Path)
        parser.add_argument("--study", type=Path)
        parser.add_argument("--submit", action="store_true")
        args = parser.parse_args()
        print(json.dumps(submit(args.run.resolve()) if args.submit else stage(args.run.resolve(), args.study.resolve())))
