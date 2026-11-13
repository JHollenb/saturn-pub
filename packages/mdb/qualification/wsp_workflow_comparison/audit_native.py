"""Read saved native-smoke arrays without loading or executing the target model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root):
    root = Path(root)
    inventory = json.loads((root / "files.json").read_text())
    for name, expected in inventory.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("inventory path escapes result")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("saved inventory differs: " + name)
    summary = json.loads((root / "summary.json").read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    with np.load(root / "anchor.npz", allow_pickle=False) as saved:
        anchor = saved["logp"].astype(np.float64)
    if not np.isfinite(anchor).all():
        raise ValueError("nonfinite anchor")
    targets = np.asarray(inputs["target_ids"], dtype=np.int64)
    rows, arrays = {}, {}
    for name, receipt in summary["outcomes"].items():
        if "readout" not in receipt:
            rows[name] = {"instrument_status": "evidence-incomplete"}
            continue
        readout = receipt["readout"]
        path = root / readout["path"]
        if readout["sha256"] != inventory.get(readout["path"]):
            raise ValueError("readout receipt differs")
        with np.load(path, allow_pickle=False) as saved:
            data = {k: saved[k].copy() for k in saved.files}
        if not all(np.isfinite(a).all() for a in data.values()):
            raise ValueError("nonfinite readout: " + name)
        if data["logp"].shape != anchor.shape or not np.array_equal(data["rows"], np.arange(16)):
            raise ValueError("native row geometry differs")
        logp = data["logp"].astype(np.float64)
        correct = logp[np.arange(16), targets]
        kl = (np.exp(anchor) * (anchor-logp)).sum(-1)
        if not np.array_equal(correct, data["correct_logp"]):
            raise ValueError("saved target-token readout differs")
        if not np.array_equal(np.argmax(logp, axis=-1), data["winner_ids"]):
            raise ValueError("saved winner readout differs")
        arrays[name] = logp
        rows[name] = {"instrument_status": "observed", "finite": True,
            "task_nll_fp64": float(-correct[:8].mean()),
            "preservation_kl_fp64": float(kl[8:].mean()),
            "maximum_kl_reduction_difference_from_native_fp32": float(np.max(np.abs(kl-data["kl"]))),
            "task_nll_difference_from_native_fp32": float(-correct[:8].mean()-receipt["task_mse"]),
            "preservation_kl_difference_from_native_fp32": float(kl[8:].mean()-receipt["preservation_kl"]),
            "correct": int(np.count_nonzero(data["winner_ids"][:8] == targets[:8]))}
    comparisons = {}
    for a, b in (("native", "restored"), ("virtual", "compiled")):
        if a not in arrays or b not in arrays:
            comparisons[a+"_vs_"+b] = {"instrument_status": "evidence-incomplete"}
            continue
        delta = arrays[a]-arrays[b]
        comparisons[a+"_vs_"+b] = {"maximum_logp_difference": float(np.max(np.abs(delta))),
            "different_logp_elements": int(np.count_nonzero(delta)), "compared_elements": int(delta.size)}
    return {"schema": "mdb-native-saved-readout-audit-v1", "job_id": summary["job_id"],
        "request_sha256": summary["request_sha256"],
        "inventory_sha256": hashlib.sha256((root / "files.json").read_bytes()).hexdigest(),
        "verified_files": len(inventory), "rows": rows, "comparisons": comparisons,
        "scope": "saved-array identity, finite values, native row geometry, target/winner extraction and independent FP64 NLL/KL arithmetic; no parameter-restoration reexecution or editor ranking",
        "work": {"target_model_loads": 0, "target_model_forwards": 0},
        "mechanics_status": "valid", "terminal_status": "not-evaluated"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True))
