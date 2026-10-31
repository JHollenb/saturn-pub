"""Compare completed storage reproductions using saved arrays only.

Custody verification reuses the public qualification auditor. Random Acts,
transition receipts and run-specific observation metadata are excluded from
the native-value comparison. No model or equation is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mdb import forecast_storage, forecast_transaction
from mdb.forecast_transaction import load_forecast_tree


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(run):
    tx = run["transaction"]
    declaration = read(tx / "declarations.json")
    forecast = read(tx / "forecasts/campaign.json")
    observation = read(tx / "observations/native.json")["value"]["items"]
    fields = ("heads", "stages", "roots", "decoded", "before")
    encoded = {"weights": declaration["actual_weight_manifest"], "predictions": forecast["predictions"],
               "payloads": forecast["payloads"], "observations": {
                   row: {field: value["items"][field] for field in fields} for row, value in observation.items()}}
    decoded = {"weights": load_forecast_tree(tx, encoded["weights"]),
               "predictions": load_forecast_tree(tx, encoded["predictions"]),
               "payloads": {name: {"value": load_forecast_tree(tx, item["value"]),
                                    "original": load_forecast_tree(tx, item["original"]),
                                    "cast": item["cast"], "provenance": item["provenance"]}
                            for name, item in forecast["payloads"].items()},
               "observations": {row: {field: load_forecast_tree(tx, value) for field, value in values.items()}
                                for row, values in encoded["observations"].items()}}
    identities, locations = {}, {}

    def visit(value, path):
        if isinstance(value, dict):
            if value.get("kind") == "array":
                key = json.dumps(path, separators=(",", ":"))
                identities[key] = {name: value[name] for name in ("dtype", "shape", "sha256", "content_sha256")}
                locations[key] = {name: value[name] for name in ("path", "bundle_sha256", "offset", "nbytes") if name in value}
                locations[key]["storage"] = value.get("storage", "files-v1")
            else:
                for key, child in value.items():
                    visit(child, [*path, key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, index])

    visit(encoded, [])
    return decoded, identities, locations


def timing(run):
    phases, pending = Counter(), {}
    events = read(run["data"] / "phase-timings.json")
    for event in events:
        key = (event["phase"], event.get("row"))
        if event["event"] == "start":
            if key in pending:
                raise ValueError("overlapping duplicate phase timing")
            pending[key] = event["elapsed_seconds"]
        elif event["event"] == "end":
            if key not in pending:
                raise ValueError("unpaired phase timing")
            phases[event["phase"]] += event["elapsed_seconds"] - pending.pop(key)
    if pending:
        raise ValueError("unfinished phase timing")
    summary = read(run["data"] / "summary.json")
    counts = summary["counts"]
    if counts["physical_forwards"] != 7 or counts["prior_rows_reused"] != 3:
        raise ValueError("performance reproduction must retain seven native calls and three reused rows")
    tx_files = [path for path in run["transaction"].rglob("*") if path.is_file()]
    arrays = [path for path in tx_files if path.parent == run["transaction"] / "arrays"]
    return {"phase_wall_seconds": dict(phases), "worker_seconds": summary["worker_seconds"], "counts": counts,
            "transaction_files": len(tx_files), "transaction_bytes": sum(path.stat().st_size for path in tx_files),
            "array_storage_files": len(arrays), "array_storage_bytes": sum(path.stat().st_size for path in arrays)}


def compare(reference, candidate):
    # These are sibling public qualification helpers, not private runtime imports.
    from forecast_storage_benchmark import compare_leaves, leaf_manifest
    from summarize_context_window import source_run

    runs = [source_run(path, "context-window-result") for path in (reference, candidate)]
    values = [records(run) for run in runs]
    comparisons = {name: compare_leaves(leaf_manifest(values[0][0][name]), leaf_manifest(values[1][0][name]))
                   for name in ("weights", "predictions", "payloads", "observations")}
    native_rows = {row: compare_leaves(leaf_manifest(values[0][0]["observations"].get(row)),
                                      leaf_manifest(values[1][0]["observations"].get(row)))
                   for row in sorted(set(values[0][0]["observations"]) | set(values[1][0]["observations"]))}
    npy = compare_leaves(values[0][1], values[1][1])
    left, right = values[0][2], values[1][2]
    location_changes = [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    timings = [timing(run) for run in runs]
    phases = sorted(set(timings[0]["phase_wall_seconds"]) | set(timings[1]["phase_wall_seconds"]))
    deltas = {phase: {"reference_seconds": timings[0]["phase_wall_seconds"].get(phase, 0),
                      "candidate_seconds": timings[1]["phase_wall_seconds"].get(phase, 0)} for phase in phases}
    for row in deltas.values():
        row["candidate_minus_reference_seconds"] = row["candidate_seconds"] - row["reference_seconds"]
    intact = npy["byte_exact"] and all(row["byte_exact"] for row in comparisons.values())
    provenance = [{key: run["source"][key] for key in ("job_id", "request_sha256", "request_file_sha256",
                   "report_file_sha256", "archive_sha256", "manifest_sha256", "transaction")} for run in runs]
    return {"schema": "mdb-forecast-storage-reproduction-comparison-v1", "mechanics_status": "valid" if intact else "differs",
            "scope": "performance reproduction of seven known native calls plus three prior controls; no new scientific specimens",
            "logical_comparisons": comparisons, "native_observations_by_row": native_rows,
            "original_npy_leaf_identities": npy,
            "storage_changes": {"changed_occurrences": len(location_changes),
                "reference_kinds": dict(Counter(value["storage"] for value in left.values())),
                "candidate_kinds": dict(Counter(value["storage"] for value in right.values())),
                "first_changes": [{"tree_path": key, "reference": left.get(key), "candidate": right.get(key)}
                                  for key in location_changes[:8]]},
            "reference": {"source": provenance[0], "measurements": timings[0]},
            "candidate": {"source": provenance[1], "measurements": timings[1]}, "phase_comparison": deltas,
            "analysis_sources": {name: sha(Path(__file__).with_name(name)) for name in
                (Path(__file__).name, "forecast_storage_benchmark.py", "summarize_context_window.py")},
            "public_reader_sources": {Path(module.__file__).name: sha(Path(module.__file__))
                                      for module in (forecast_transaction, forecast_storage)},
            "model_calls": 0, "equation_evaluations": 0, "terminal_status": "unassessed",
            "limits": ["Same planned values and native outputs are checked; random Acts/receipts/run metadata are not compared.",
                       "Phase clocks include their declared callback/publication boundaries, not isolated I/O or pure kernel time.",
                       "Separate sequential jobs can differ in filesystem cache and host conditions; no cold-disk/general speed claim.",
                       "Independent per-row native/Act auditing remains the separate context-window auditor's responsibility."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to((run / "extracted").resolve())
           for run in (args.reference_run, args.candidate_run)):
        parser.error("comparison output must remain outside immutable extracted results")
    result = compare(args.reference_run, args.candidate_run)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "mechanics_status": result["mechanics_status"]}))
    return 0 if result["mechanics_status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
