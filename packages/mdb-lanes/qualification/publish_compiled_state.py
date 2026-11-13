"""Write the compiled-state qualification record from one collected run directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mdb_lanes.compiled_state import render, verify_report


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "report.json").read_text())  # collector's copy of the worker summary
    report_path = run_dir / "extracted" / "compiled-state-result" / "report.json"
    report = verify_report(json.loads(report_path.read_text()))
    if file_sha(report_path) != summary["report_sha256"] or report["fingerprint"] != summary["report_fingerprint"]:
        raise ValueError("collected report differs from the worker summary")
    job = json.loads((run_dir / "job.json").read_text())
    expected = summary["config"]["comparison"]
    rows = []
    for fset, fs in report["feature_sets"].items():
        for L, row in sorted(fs["layers"].items(), key=lambda item: int(item[0])):
            a = row["agreement"]
            rows.append({"feature_set": fset, "layer": int(L), "row_agreement": a["row_agreement"],
                         "row_agreement_ci": a["row_agreement_ci"], "error_recall": a.get("error_recall"),
                         "n_native_wrong": a.get("n_native_wrong"),
                         "part_e_row_agreement": expected["row_agreement"].get(fset, {}).get(L),
                         "r2": [p["r2"] for p in row["positions"]]})
    return {"schema": "mdb-compiled-state-qualification-record-v1", "job_id": summary["job_id"],
            "job_state": job.get("state"), "request_sha256": summary["request_sha256"],
            "report_fingerprint": report["fingerprint"], "report_sha256": summary["report_sha256"],
            "model": report["model"], "execution": report["execution"], "environment": summary["environment"],
            "tokenization_probe": summary["tokenization_probe"], "rows": report["rows"], "lam": report["lam"],
            "threshold": report["threshold"], "native": report["native"], "noise_ceiling": report["noise_ceiling"],
            "native_splice_control": report["native_splice_control"],
            "horizons": {f: {**fs["horizon"], "part_e": expected["horizon_at_0.95"].get(f)}
                         for f, fs in report["feature_sets"].items()},
            "r2_L4_F1": {"measured": [p["r2"] for p in report["feature_sets"]["F1-digits"]["layers"]["4"]["positions"]]
                         if "4" in report["feature_sets"].get("F1-digits", {}).get("layers", {}) else None,
                         "part_d_e": expected["r2_L4_F1"]},
            "ladder": rows, "comparison_source": expected["source"], "limits": summary["config"]["limits"],
            "rendered": render(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = record(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(value["rendered"])


if __name__ == "__main__":
    main()
