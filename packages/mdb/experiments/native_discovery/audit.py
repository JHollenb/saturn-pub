"""Artifact-only audit: hashes, sealed forecast scores, and saved training closures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mdb.forecast_transaction import load_forecast_tree
from mdb.native_discovery import error_metrics, file_hash, save_json, verify_run


def logsoftmax(a):
    a = a.astype(np.float64)
    a = a - a.max()
    return a - np.log(np.exp(a).sum())


def audit(stage, *, closures=False):
    root = stage / "evidence"
    marker = json.loads((stage / "result.json").read_text())
    if file_hash(stage / "evidence.tar.gz") != marker["artifact"]["sha256"]:
        raise ValueError("archive does not match the job's result marker")
    verification = verify_run(root)
    report = json.loads((root / "report.json").read_text())
    documents = {r["document"]: r for r in report["heldout"]}
    checks = 0
    for forecast in sorted((root / "forecasts").iterdir()):
        observed = json.loads((forecast / "observations/ordinary_native.json").read_text())
        native = load_forecast_tree(forecast, observed["value"])
        row = documents[native["trace"]["document_id"]]
        target_logp = logsoftmax(native["consumer_logits"]) if "consumer_logits" in native else None
        for path in sorted((forecast / "forecasts").glob("*.json")):
            sealed = json.loads(path.read_text())
            prediction = load_forecast_tree(forecast, sealed["predictions"])
            kind = sealed["name"]
            for i, state in enumerate(prediction["state"]):
                expected = row["arms"][kind][i]["state"] if kind in row["arms"] else row[kind][i]
                actual = error_metrics(state, native["native_states"][i + 1])
                for key in ("mse", "relative_mse", "max_abs"):
                    np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-12)
                    checks += 1
            if target_logp is not None:
                predicted_logp = logsoftmax(prediction["consumer_logits"])
                kl = float((np.exp(target_logp) * (target_logp - predicted_logp)).sum())
                expected = (row["native_output_consumer"][kind] if kind in row["arms"]
                            else row[kind + "_consumer"])["native_to_predicted_kl"]
                np.testing.assert_allclose(kl, expected, rtol=1e-10, atol=1e-12)
                checks += 1
    snapshots = []
    if closures:
        from mdb.native_discovery_learning import load_snapshot
        paths = [*root.glob("predictor/*/candidates/*"), *root.glob("predictor/*/accepted")]
        for path in sorted(paths):
            snapshot = load_snapshot(path)
            snapshot.verify()
            snapshots.append({"path": str(path.relative_to(root)), "fingerprint": snapshot.fingerprint})
    result = {"schema": "mdb-native-discovery-artifact-audit-v1", "verification": verification,
        "source_archive_sha256": marker["artifact"]["sha256"], "metric_checks": checks,
        "closure_checks": len(snapshots), "closures": snapshots, "native_model_loads": 0,
        "native_forwards": 0, "mechanics_status": "valid", "terminal_status": "not-requested"}
    save_json(stage / "artifact-audit.json", result)
    return {k: result[k] for k in ("metric_checks", "closure_checks", "native_model_loads", "mechanics_status")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--closures", action="store_true")
    print(json.dumps(audit(**vars(parser.parse_args())), sort_keys=True))
