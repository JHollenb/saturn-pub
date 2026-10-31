"""Finite saved-array forecast-storage benchmark; no model or equation execution.

One lease runs three isolated children sequentially: frozen old files-v1,
current files-v1, and opt-in current bundle-v1. Filesystem cache/order effects
are retained; warm reads are explicitly labeled. Staging submits a job, so
invoke it only after reviewing the frozen input and runtime source snapshots.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.util
import json
import os
import platform
import pstats
import resource
import runpy
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

MODES = ("baseline-files-v1", "current-files-v1", "current-bundle-v1")
MAX_INPUT_BYTES = 32 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def support():
    path = Path(__file__).with_name("campaign_support.py")
    return runpy.run_path(str(path if path.exists() else path.with_name("discovery_upgrades.py")))


def descriptors(value):
    if isinstance(value, dict):
        if value.get("kind") == "array":
            yield value
        else:
            for child in value.values():
                yield from descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from descriptors(child)


def decode(module, root, node):
    # The public tree reader owns bundle verification/caching. Calling the leaf
    # reader repeatedly would invent a per-leaf rehash cost for bundled trees.
    return module.load_forecast_tree(root, node)


def load_module(path, name):
    """Explicit source snapshot, using installed public MDB for relative imports."""
    import mdb

    if not mdb.__file__:
        raise ValueError("installed public MDB package required")
    spec = importlib.util.spec_from_file_location("mdb." + name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load explicit forecast source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def leaf_manifest(value):
    """Every decoded scalar and array byte; excludes storage descriptor layout."""
    import numpy as np

    result = {}

    def walk(node, path):
        key = json.dumps(path, separators=(",", ":"))
        if isinstance(node, np.ndarray):
            result[key] = {"kind": "array", "dtype": str(node.dtype), "shape": list(node.shape),
                           "bytes": int(node.nbytes), "sha256": digest(node.tobytes(order="C"))}
        elif isinstance(node, Mapping):
            result[key] = {"kind": "mapping", "keys": sorted(node)}
            for name in sorted(node):
                walk(node[name], [*path, name])
        elif isinstance(node, (tuple, list)):
            result[key] = {"kind": "sequence", "length": len(node)}
            for index, child in enumerate(node):
                walk(child, [*path, index])
        elif node is None or type(node) in (str, bool, int, float):
            result[key] = {"kind": "scalar", "type": type(node).__name__, "sha256": digest(canonical(node))}
        else:
            raise ValueError("unsupported decoded leaf: " + type(node).__name__)

    walk(value, [])
    return result


def compare_leaves(expected, actual):
    keys = set(expected) | set(actual)
    differences = [key for key in sorted(keys) if expected.get(key) != actual.get(key)]
    return {"compared_nodes": len(keys), "differences": len(differences), "first_differences": differences[:20],
            "byte_exact": not differences, "expected_sha256": digest(canonical(expected)),
            "actual_sha256": digest(canonical(actual))}


def workload_stats(manifest):
    arrays = [value for value in manifest.values() if value["kind"] == "array"]
    unique = {(value["dtype"], tuple(value["shape"]), value["sha256"]): value["bytes"] for value in arrays}
    return {"array_occurrences": len(arrays), "unique_array_values": len(unique),
            "repeated_array_occurrences": len(arrays) - len(unique),
            "logical_array_bytes": sum(value["bytes"] for value in arrays),
            "unique_array_bytes": sum(unique.values()), "tree_nodes": len(manifest)}


class Measurements:
    """Synchronous syscall wall/CPU time; instrumentation is equal in all modes."""

    def __init__(self):
        self.phases = []
        self.fsync = {"calls": 0, "wall_seconds": 0.0, "cpu_seconds": 0.0}

    @contextmanager
    def fsync_probe(self):
        original = os.fsync

        def measured(fd):
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return original(fd)
            finally:
                self.fsync["calls"] += 1
                self.fsync["wall_seconds"] += time.perf_counter() - wall
                self.fsync["cpu_seconds"] += time.process_time() - cpu

        os.fsync = measured
        try:
            yield
        finally:
            os.fsync = original

    def timed(self, name, callback):
        before = dict(self.fsync)
        wall, cpu = time.perf_counter(), time.process_time()
        print("MDB_STORAGE_PHASE " + json.dumps({"phase": name, "event": "start"}), flush=True)
        try:
            return callback()
        finally:
            row = {"phase": name, "wall_seconds": time.perf_counter() - wall,
                   "cpu_seconds": time.process_time() - cpu,
                   "fsync": {key: self.fsync[key] - before[key] for key in self.fsync}}
            self.phases.append(row)
            print("MDB_STORAGE_PHASE " + json.dumps({**row, "event": "end"}), flush=True)


def profile_summary(profile, source):
    stats = pstats.Stats(profile)
    rows = []
    groups = Counter()
    for (filename, line, name), (primitive, calls, self_time, cumulative, callers) in stats.stats.items():
        # Nonoverlapping function-self-time groups; cumulative times are shown
        # per function only and never summed as if disjoint phases.
        category = "other"
        if "fsync" in name or name == "measured":
            category = "sync_and_probe_overhead"
        elif "hash" in name or name in ("_sha", "digest"):
            category = "hash"
        elif "write" in name or name in ("_publish", "link", "mkdir", "open"):
            category = "write_publication_and_filesystem"
        elif any(token in name.lower() for token in ("encode", "canonical", "dumps", "save", "tobytes")):
            category = "serialization"
        elif any(token in name.lower() for token in ("read", "decode", "load")):
            category = "read_decode"
        groups[category] += self_time
        rows.append({"file": filename, "line": line, "function": name, "calls": calls,
                     "primitive_calls": primitive, "self_seconds": self_time, "cumulative_seconds": cumulative})
    rows.sort(key=lambda row: row["self_seconds"], reverse=True)
    return {"source": str(source), "grouped_self_seconds": dict(groups), "top_functions": rows[:100],
            "classification_scope": "function-name classification; instrumented wall/self times, not independent disk bandwidth",
            "total_profile_seconds": stats.total_tt, "total_calls": stats.total_calls}


def read_workload(module, root):
    declarations = json.loads((root / "declarations.json").read_text())
    forecast = json.loads((root / "forecasts/campaign.json").read_text())
    observed = json.loads((root / "observations/native.json").read_text())
    frontiers = {name: decode(module, root, json.loads((root / ("frontiers/" + name + ".json")).read_text())["value"])
                 for name in forecast["frontiers"]}
    payloads = {name: {"original": decode(module, root, row["original"]), "dtype": row["cast"]["to"],
                       "provenance": row["provenance"], "value": decode(module, root, row["value"])}
                for name, row in forecast["payloads"].items()}
    return {"weights": decode(module, root, declarations["actual_weight_manifest"]), "frontiers": frontiers,
            "predictions": decode(module, root, forecast["predictions"]), "payloads": payloads,
            "observations": decode(module, root, observed["value"]), "declarations": declarations}


def logical_tree(workload):
    return {"weights": workload["weights"], "frontiers": workload["frontiers"],
            "predictions": workload["predictions"], "observations": workload["observations"],
            "payload_values": {name: row["value"] for name, row in workload["payloads"].items()},
            "payload_originals": {name: row["original"] for name, row in workload["payloads"].items()},
            "payload_metadata": {name: {"dtype": row["dtype"], "provenance": row["provenance"]}
                                 for name, row in workload["payloads"].items()}}


def encoded_array_identities(root):
    """Retain stable NPY leaf identities while ignoring physical storage layout."""
    declaration = json.loads((root / "declarations.json").read_text())
    forecast = json.loads((root / "forecasts/campaign.json").read_text())
    observation = json.loads((root / "observations/native.json").read_text())
    values = {"weights": declaration["actual_weight_manifest"], "predictions": forecast["predictions"],
              "observations": observation["value"], "payloads": forecast["payloads"],
              "frontiers": {name: json.loads((root / ("frontiers/" + name + ".json")).read_text())["value"]
                            for name in forecast["frontiers"]}}
    result = {}

    def visit(value, path):
        if isinstance(value, dict):
            if value.get("kind") == "array":
                result[json.dumps(path, separators=(",", ":"))] = {
                    key: value[key] for key in ("sha256", "content_sha256", "shape", "dtype")}
            else:
                for key, child in value.items():
                    visit(child, [*path, key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, index])

    visit(values, [])
    return result


def child(mode):
    if mode not in MODES:
        raise ValueError("unknown benchmark mode")
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    for name, expected in request["source_hashes"].items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError("staged file differs: " + name)
    current = load_module(Path("current_forecast_transaction.py"), "storage_benchmark_current")
    baseline = load_module(Path("baseline_forecast_transaction.py"), "storage_benchmark_baseline")
    import mdb.forecast_storage

    storage_source = Path(mdb.forecast_storage.__file__)
    if digest(storage_source.read_bytes()) != request["source_hashes"]["current_forecast_storage.py"]:
        raise ValueError("installed public storage dependency differs from the staged current source")
    module = baseline if mode == MODES[0] else current
    output = Path("storage-benchmark-result") / mode
    output.mkdir(parents=True, exist_ok=False)
    measurements = Measurements()
    workload = measurements.timed("saved_input_load_once", lambda: read_workload(current, Path("input")))
    if len(workload["weights"]) != 87 or len(workload["predictions"]["rows"]) != 10 or len(workload["observations"]) != 10:
        raise ValueError("representative saved workload geometry differs")
    expected = leaf_manifest(logical_tree(workload))
    save(output / "expected-leaves.json", expected)
    expected_npy = encoded_array_identities(Path("input"))
    save(output / "expected-npy-identities.json", expected_npy)
    profile = cProfile.Profile()
    tx = None
    kwargs = {} if mode == MODES[0] else {"array_storage": "bundle-v1" if mode == MODES[2] else "files-v1"}

    def construct():
        return module.ForecastTransaction(output / "transaction", model_identity=config["model_identity"],
            input_identity=config["workload_identity"], source_identity=request["source_hashes"],
            frontier={"kind": "empty", "scope": "posthoc saved-array storage mechanics only"},
            weights=workload["weights"], budgets={"max_native_calls": 1, "max_wall_ms": 500000,
                "max_bytes": 64 * 1024 * 1024, "max_files": 4096}, **kwargs)

    def predict(context):
        return {"predictions": workload["predictions"], "payloads": {name: module.ForecastPayload(
            row["original"], dtype=row["dtype"], provenance=row["provenance"])
            for name, row in workload["payloads"].items()}}

    try:
        with measurements.fsync_probe():
            profile.enable()
            try:
                tx = measurements.timed("initialize_weights", construct)
                with tx:
                    measurements.timed("frontier_publish", lambda: tx.run_script([
                        {"op": "frontier", "name": name, "value": value} for name, value in workload["frontiers"].items()]))
                    measurements.timed("predict_publish", lambda: tx.run_script([
                        {"op": "predict", "name": "campaign", "callback": predict}]))
                    measurements.timed("seal", lambda: tx.run_script([{"op": "seal"}]))
                    measurements.timed("execute_saved_observation_publish", lambda: tx.run_script([
                        {"op": "execute", "name": "native", "callback": lambda context: workload["observations"]}]))
                report = tx.report()
                verification = measurements.timed("offline_verify", lambda: module.verify_transaction(output / "transaction",
                    expected_seal_sha256=report["seal_sha256"], expected_event_head=report["event_head"]))
                npy_identity = compare_leaves(expected_npy, encoded_array_identities(output / "transaction"))
                if not npy_identity["byte_exact"]:
                    raise ValueError("stable original NPY leaf identity differs: " + json.dumps(npy_identity))
                encoded_prediction = json.loads((output / "transaction/forecasts/campaign.json").read_text())["predictions"]
                representative_leaf = next(descriptors(encoded_prediction))
                for index in range(2):
                    leaf = measurements.timed("warm_public_array_read_" + str(index + 1),
                        lambda: current.load_forecast_array(output / "transaction", representative_leaf))
                    if digest(leaf.tobytes(order="C")) != representative_leaf["content_sha256"]:
                        raise ValueError("public array reader leaf bytes differ")
                    loaded = measurements.timed("warm_public_read_" + str(index + 1), lambda: {
                        "predictions": tx.prediction("campaign"), "observations": tx.observation("native"),
                        "payload_values": {name: tx.payload("campaign", name) for name in workload["payloads"]}})
                    # Weights/frontiers/original payloads use the public saved-array
                    # reader as well, ensuring every logical leaf is compared.
                    saved = measurements.timed("warm_all_leaf_read_" + str(index + 1),
                                               lambda: logical_tree(read_workload(current, output / "transaction")))
                    result = compare_leaves(expected, leaf_manifest(saved))
                    for key in loaded:
                        if compare_leaves(leaf_manifest(saved[key]), leaf_manifest(loaded[key]))["differences"]:
                            raise ValueError("public transaction read differs from public descriptor read: " + key)
                    if not result["byte_exact"]:
                        raise ValueError("saved logical leaf byte mismatch: " + json.dumps(result))
                    del saved, loaded, leaf
                    gc.collect()
            finally:
                profile.disable()
        if report["work"]["native_calls"] != 0:
            raise ValueError("storage benchmark unexpectedly entered a native model")
        paths = [path for path in (output / "transaction").rglob("*") if path.is_file()]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
        summary = {"schema": "mdb-forecast-storage-measurement-v1", "mode": mode,
            "phases": measurements.phases, "fsync": measurements.fsync, "workload": workload_stats(expected),
            "equivalence": result, "transaction_verification": verification, "transaction_report": report,
            "stable_npy_leaf_identities": npy_identity,
            "transaction_files": len(paths), "transaction_bytes": sum(path.stat().st_size for path in paths),
            "max_rss_bytes": peak_bytes, "max_rss_scope": "isolated child process lifetime including input decode/profile",
            "filesystem": {"working_directory": str(Path.cwd()), "device": os.stat(output).st_dev,
                           "block_size": os.statvfs(output).f_bsize},
            "runtime_source_sha256": digest(Path(module.__file__).read_bytes()), "model_calls": 0, "equation_evaluations": 0}
        summary["current_public_storage_source_sha256"] = digest(storage_source.read_bytes())
        save(output / "summary.json", summary)
    finally:
        profile.dump_stats(str(output / "profile.prof"))
        save(output / "profile.json", profile_summary(profile, Path(module.__file__)))
        if tx is not None:
            save(output / "transaction-report.json", tx.report())


def worker():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    output = Path("storage-benchmark-result")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    results = []
    for mode in MODES:
        remaining = 550 - (time.monotonic() - started)
        if remaining <= 1:
            raise ValueError("finite total benchmark wall budget exhausted")
        command = [sys.executable, str(Path(__file__).resolve()), "--benchmark-child", mode]
        print("MDB_STORAGE_CHILD " + json.dumps({"mode": mode, "event": "start"}), flush=True)
        completed = subprocess.run(command, timeout=remaining, check=False)
        if completed.returncode:
            raise ValueError("storage benchmark child failed: " + mode)
        results.append(json.loads((output / mode / "summary.json").read_text()))
    reference = json.loads((output / MODES[0] / "expected-leaves.json").read_text())
    for mode in MODES[1:]:
        other = json.loads((output / mode / "expected-leaves.json").read_text())
        if not compare_leaves(reference, other)["byte_exact"]:
            raise ValueError("child input workloads differ")
    for name in ("baseline_forecast_transaction.py", "current_forecast_transaction.py", "current_forecast_storage.py", "input-manifest.json"):
        with (output / name).open("xb") as stream:
            stream.write(Path(name).read_bytes())
    summary = {"schema": "mdb-forecast-storage-benchmark-result-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "mechanics_status": "valid", "modes": results,
        "worker_seconds": time.monotonic() - started, "model_calls": 0, "equation_evaluations": 0,
        "host": platform.node(), "python": sys.version, "order": list(MODES),
        "limits": ["One measured filesystem/host/lease; fixed order with warm OS cache, not a cold-disk general speed claim.",
                   "Fsync probe and cProfile overhead are enabled equally in all modes.",
                   "Saved forecast/observation values are copied posthoc; no new predictions or model/equation evaluation.",
                   "Model guard/signature overhead is excluded because this transaction has no bound model."]}
    save(output / "summary.json", summary)
    support()["emit"](output, summary)


def stage(run_dir, source_result, current_source, baseline_ref, ram_mb=2304):
    from mdb.execution_environment import intent
    from mdb.forecast_transaction import verify_transaction
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    if ram_mb != 2304:
        raise ValueError("this bounded qualification retains the explicit2304MiB measured-parent legacy-bridge exception")
    root = source_result / "transaction"
    source_report = json.loads((source_result / "transaction-report.json").read_text())
    verification = verify_transaction(root, expected_seal_sha256=source_report["seal_sha256"],
                                      expected_event_head=source_report["event_head"])
    relative = {"declarations.json", "forecasts/campaign.json", "observations/native.json"}
    forecast = json.loads((root / "forecasts/campaign.json").read_text())
    relative.update("frontiers/" + name + ".json" for name in forecast["frontiers"])
    for name in list(relative):
        for descriptor in descriptors(json.loads((root / name).read_text())):
            path = Path(descriptor["path"])
            if path.is_absolute() or ".." in path.parts or descriptor.get("storage"):
                raise ValueError("benchmark input must be the original files-v1 saved workload")
            relative.add(path.as_posix())
    size = sum((root / name).stat().st_size for name in relative)
    if size > MAX_INPUT_BYTES or len(relative) > 2048:
        raise ValueError("saved benchmark input exceeds finite geometry")
    run_dir.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[3]
    git_path = "packages/mdb/src/mdb/forecast_transaction.py"
    baseline = subprocess.run(["git", "-C", str(repository), "show", baseline_ref + ":" + git_path],
                              capture_output=True, check=True).stdout
    baseline_path = run_dir / "baseline_forecast_transaction.py"
    baseline_path.write_bytes(baseline)
    files = [(root / name, "input/" + name) for name in sorted(relative)]
    storage_source = current_source.with_name("forecast_storage.py")
    files += [(baseline_path, baseline_path.name), (current_source, "current_forecast_transaction.py"),
              (storage_source, "current_forecast_storage.py"),
              (Path(__file__).with_name("discovery_upgrades.py"), "campaign_support.py")]
    manifest = {"schema": "mdb-storage-benchmark-input-v1", "source_transaction": verification,
        "source_report_sha256": digest((source_result / "transaction-report.json").read_bytes()),
        "files": {name: digest((root / name).read_bytes()) for name in sorted(relative)},
        "bytes": size, "baseline_ref": baseline_ref, "baseline_source_sha256": digest(baseline),
        "current_source_sha256": digest(current_source.read_bytes()),
        "current_storage_source_sha256": digest(storage_source.read_bytes()),
        "scope": "posthoc storage mechanics; no model/equation computation"}
    manifest_path = run_dir / "input-manifest.json"
    save(manifest_path, manifest)
    files.append((manifest_path, manifest_path.name))
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "path_token_allowlist": ["storage-benchmark-result"],
        "hashed_roots": ["input", "baseline_forecast_transaction.py", "current_forecast_transaction.py", "current_forecast_storage.py",
                         "forecast_storage_benchmark.py", "input-manifest.json", "campaign_support.py", "preflight.v1.json"]})
    files.append((preflight, preflight.name))
    declarations = json.loads((root / "declarations.json").read_text())
    config = {"schema": "mdb-forecast-storage-config-v1", "model_identity": declarations["model_identity"],
              "workload_identity": digest(canonical(manifest))}
    request = {"schema": "mdb-forecast-storage-request-v1", "task_family": "mdb-forecast-storage",
        "config": config, "needs": {"cpu": True}, "execution_environment": intent(),
        "source_hashes": {name: digest(path.read_bytes()) for path, name in files},
        "input_manifest": manifest,
        "resource_policy": {"cpu_threads": 1, "ram_mb": ram_mb,
            "ram_basis": {"policy": "retained measured CPU-parent exception for the legacy reservation bridge",
                          "job_id": "job-053353849ab7", "measured_peak_ram_mb": 1028.2,
                          "saved_input_bytes": size, "native_model_calls": 0}},
        "scheduling_contract": one_lease_scheduling_contract(logical_operations=3,
            logical_checkpoint_replays=0, max_local_work_items=3, model_free=True, model_loads=0,
            checkpoint_policy="no model branches or native checkpoints; three sequential finite storage-only helper processes within one outer lease",
            extra={"max_physical_model_forwards": 0, "max_saved_input_bytes": MAX_INPUT_BYTES,
                   "sequential_storage_helper_processes": 3, "helper_process_order": list(MODES),
                   "helper_process_scope": "posthoc storage measurement; no model loads or scientific branches"}),
        "selection_policy": "one same-host sequential lease; three isolated storage-only children; default/opt-in storage contrast"}
    request["source_hashes"]["forecast_storage_benchmark.py"] = digest(Path(__file__).read_bytes())
    request["request_sha256"] = digest(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=files)
    job = guarded_mrun_launch(worker_command("forecast_storage_benchmark.py") + ["--worker"], request=request,
        experiment="mdb-forecast-storage", task="sealed-ten-row-files-vs-bundle-storage-only", model=None,
        payload=run_dir / "payload", prefer_host="beast", gpu=False, needs=request["needs"], cpu_threads=1,
        ram_mb=ram_mb,
        timeout_s=600, retry_on_kill=False, preflight=True,
        note="One finiteCPU1 storage-only lease:no model/native/equation calls. 3sequential isolated children:29669c7files/currentfiles/currentbundle. Same10savedforecasts+10observations+87weights; cProfile/fsync/warmread/peakRSS+allleafbytes. <=32MiB inputs,64MiB retained pertransaction,550sinner/600souter,no retries.2304MiB retainedlegacybridge exception:CPUparent job053353849ab7 measured1028.2MiB; current9MiB savedinput/no model. Stopafter3children/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--benchmark-child", choices=MODES)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--current-source", type=Path)
    parser.add_argument("--baseline-ref", default="29669c7")
    parser.add_argument("--ram-mb", type=float, default=2304, help="fixed2304MiB measured-parent exception required by the legacy CPU1 reservation bridge")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.benchmark_child:
        child(args.benchmark_child)
    elif args.worker:
        try:
            worker()
        except Exception as error:
            output = Path("storage-benchmark-result")
            if output.is_dir():
                report = {"schema": "mdb-forecast-storage-benchmark-result-v1", "mechanics_status": "errored",
                    "job_id": os.environ["MRUN_JOB_ID"], "error": f"{type(error).__name__}: {error}",
                    "model_calls": 0, "equation_evaluations": 0}
                save(output / "failure.json", report)
                support()["emit"](output, report)
            raise
    elif args.collect:
        print(json.dumps(support()["collect"](args.run_dir, args.collect)))
    else:
        if args.run_dir is None or args.source_result is None or args.current_source is None:
            parser.error("staging requires run-dir, source-result and explicit current-source")
        print(json.dumps(stage(args.run_dir.resolve(), args.source_result.resolve(), args.current_source.resolve(), args.baseline_ref, args.ram_mb)))


if __name__ == "__main__":
    main()
