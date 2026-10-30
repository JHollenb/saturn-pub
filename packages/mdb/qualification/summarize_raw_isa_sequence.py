"""Audit saved raw-input forecasts and publish a compact result; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from mdb.equation_comparison import EquationCatalog
from mdb.forecast_transaction import verify_transaction
from mdb.forward_ports import NativePortCatalog


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(run_dir):
    data = run_dir / "extracted/sequence-result"
    summary, request, report = read(data / "summary.json"), read(run_dir / "request.json"), read(run_dir / "report.json")
    job = read(run_dir / "job.json")
    assert summary["job_id"] == report["job_id"] == job["job_id"] == "job-62804158d4e4"
    assert job["state"] == "succeeded" and summary["request_sha256"] == request["request_sha256"]
    unsigned = dict(request)
    expected = unsigned.pop("request_sha256")
    assert sha(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()) == expected
    for name, expected in request["source_hashes"].items():
        assert sha((run_dir / "payload" / name).read_bytes()) == expected, name
    manifest = read(data / "files.json")
    for name, expected in manifest.items():
        assert sha((data / name).read_bytes()) == expected, name
    tx = data / "transaction"
    transaction = read(data / "transaction-report.json")
    verification = verify_transaction(tx, expected_seal_sha256=transaction["seal_sha256"],
                                      expected_event_head=transaction["event_head"])
    sealed_event = next(event for event in transaction["events"] if event["kind"] == "globally_sealed")
    assert all(event["work"]["native_calls"] == 0 for event in transaction["events"]
               if event["sequence"] <= sealed_event["sequence"])
    cache = {}

    def decode(node):
        kind = node["kind"]
        if kind == "mapping":
            return {key: decode(value) for key, value in node["items"].items()}
        if kind in ("list", "tuple"):
            return [decode(value) for value in node["items"]]
        if kind == "scalar":
            return node["value"]
        if kind == "array":
            path = tx / node["path"]
            if node["path"] not in cache:
                raw = path.read_bytes()
                assert sha(raw) == node["sha256"]
                array = np.load(io.BytesIO(raw), allow_pickle=False)
                assert list(array.shape) == node["shape"] and str(array.dtype) == node["dtype"]
                assert sha(array.tobytes(order="C")) == node["content_sha256"]
                cache[node["path"]] = array
            return cache[node["path"]]
        raise ValueError("unknown numerical descriptor")

    forecasts = decode(read(tx / "forecasts/campaign.json")["predictions"])
    actual = decode(read(tx / "observations/native.json")["value"])
    comparisons = {row["row_id"]: row for row in read(data / "comparisons.json")["rows"]}
    catalog = EquationCatalog(data / "equations")
    rows = []
    for identifier, observation in actual.items():
        case, arm, step = identifier.split(".")
        prediction = forecasts[case + "." + arm]["steps"][step]
        retained = comparisons[identifier]
        record = catalog.show(retained["equation_record"])
        assert record["report"] == retained["head_report"]
        assert record["binding"]["input_identity"] == observation["input_identity"]
        trace = read(data / "captures" / identifier / "trace.json")
        ports = NativePortCatalog(trace)
        assert trace["fingerprint"] == observation["trace_fingerprint"]
        for name, binding in observation["actual_ports"].items():
            assert ports.resolve(binding["selector"]) == binding
        roots_equal = all(np.array_equal(prediction["roots"][name], observation["roots"][name])
                          for name in ("window", "regs", "flags"))
        assert roots_equal == observation["forecast_root_join"]
        maximum, decision_differences = {}, {}
        for name, value in observation["heads"].items():
            forecast = prediction["heads"][name]
            residual = value.astype(np.float64) - forecast
            maximum[name] = float(np.abs(residual).max())
            assert maximum[name] == retained["head_report"]["fields"][name]["residual"]["max_abs"]
            left, right = ((forecast.argmax(-1), value.argmax(-1)) if name in ("dest", "src", "width", "length")
                           else (forecast > 0, value > 0))
            decision_differences[name] = int(np.count_nonzero(left != right))
        for name, value in observation["stages"].items():
            error = float(np.abs(value.astype(np.float64) - prediction["stages"][name]).max())
            assert error == retained["stage_report"]["fields"][name]["residual"]["max_abs"]
        assert all(observation["replay_native_exact"].values())
        assert prediction["decoded"] == observation["decoded"]
        if int(step) > 0:
            previous = actual[case + "." + arm + "." + str(int(step) - 1)]
            assert previous["driver_commit"]["committed"]
            assert observation["before"]["register_words"] == previous["decoded"]["register_words"]
            assert observation["before"]["flags"] == previous["decoded"]["flags"]
            assert observation["before"]["offset"] == previous["before"]["offset"] + previous["decoded"]["length"]
        native = actual[case + ".native." + step]
        rsi = {candidate: {"forecast": float(forecasts[case + "." + candidate]["steps"][step]["heads"]["regs"][0, 6, 1]),
                           "observed": float(actual[case + "." + candidate + "." + step]["heads"]["regs"][0, 6, 1])}
               for candidate in ["native", "U_swap", "U_swap_P2native"]}
        rows.append({"row_id": identifier, "before": observation["before"],
            "forecast_decode": prediction["decoded"], "actual_decode": observation["decoded"],
            "all_roots_equal": roots_equal, "head_max_abs": maximum,
            "head_decision_differences": decision_differences, "native_replay_exact": True,
            "actual_port_count": len(observation["actual_ports"]), "equation_record": record["fingerprint"],
            "RSI_bit1": rsi,
            "effect_vs_native_max_abs": {name: float(np.abs(value - native["heads"][name]).max())
                                         for name, value in observation["heads"].items()},
            "regs_target_effect": retained["head_report"]["fields"]["regs"]["observed_effect"]["target"],
            "regs_collateral_effect": retained["head_report"]["fields"]["regs"]["observed_effect"]["collateral"]})
    assert len(rows) == summary["rows"] == 18
    assert sum(sum(row["head_decision_differences"].values()) for row in rows) == summary["head_decision_mismatches"] == 0
    result = {"schema": "mdb-raw-isa-sequence-publication-v1", "summary": summary,
        "source": {"job_id": job["job_id"], "archive_sha256": report["archive_sha256"],
                   "request_sha256": request["request_sha256"], "plan_file_sha256": sha((data / "plan.json").read_bytes()),
                   "verified_source_hashes": request["source_hashes"], "verified_artifact_files": len(manifest),
                   "publication_tool_sha256": sha(Path(__file__).read_bytes())},
        "verification": verification, "phase_events": transaction["events"], "rows": rows,
        "scope": "two related starting states x three branches x three successive own-state instructions",
        "countertrend": "The previous repeated-SUB RSI bit error does not recur under this mixed-program context; all branches share decoded states.",
        "timing_scope": "phase differences include equation calculation and durable persistence; no isolated I/O benchmark"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "decision_mismatches": result["summary"]["head_decision_mismatches"],
                      "verified_arrays": "all forecast/observed referenced numerical arrays", "output": str(args.output)}))


if __name__ == "__main__":
    main()
