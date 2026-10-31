"""Offline saved-array comparison only; never imports Torch or model equations.

Historical sub_258_new means a fresh suffix specimen in the OLD repeated-SUB
context. It must not be confused with current rows named new.* (mixed context).
Run without --current-root to publish an index and a waiting status; run with
the collected context-window-result to compare the first two sequence rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("native", "U_swap", "U_swap_P2native")
STAGES = ("S", "U", "M", "P2", "h", "z")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def find(rows, **fields):
    found = [row for row in rows if all(row.get(key) == value for key, value in fields.items())]
    if len(found) != 1:
        raise ValueError("expected exactly one saved row: " + str(fields))
    return found[0]


def array(root, descriptor):
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid array path")
    path = root / relative
    if path.is_symlink() or sha(path.read_bytes()) != descriptor["sha256"]:
        raise ValueError("saved array byte identity differs: " + str(path))
    result = np.load(path, allow_pickle=False)
    if list(result.shape) != descriptor["shape"] or str(result.dtype) != descriptor["dtype"]:
        raise ValueError("saved array geometry differs")
    if "content_sha256" in descriptor and sha(result.tobytes()) != descriptor["content_sha256"]:
        raise ValueError("saved array content identity differs")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite saved evidence")
    return result


def decode(root, node):
    kind = node["kind"]
    if kind == "array":
        return array(root, node)
    if kind == "mapping":
        return {key: decode(root, child) for key, child in node["items"].items()}
    if kind in ("list", "tuple"):
        return [decode(root, child) for child in node["items"]]
    if kind == "scalar":
        return node["value"]
    raise ValueError("unsupported transaction encoding")


def compare(left, right):
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    return {"shape": list(left.shape), "array_equal": bool(np.array_equal(left, right)),
            "max_abs": float(np.abs(delta).max(initial=0)), "l2": float(np.linalg.norm(delta)),
            "mean_signed_current_minus_historical": float(delta.mean()),
            "positive_decision_mismatches": int(np.count_nonzero((left > 0) != (right > 0)))}


def low(array_value, field):
    if field == "regs" or field.startswith("register_terms."):
        return array_value[:, :, :2]
    if array_value.ndim == 3 and array_value.shape[1] == 64:
        return array_value[:, :2, :]
    return array_value


def historical_index(root):
    files = ("plan.json", "prefixes.json", "forecasts.json", "primary-observations.json", "forecast-seal.json")
    plan = read_json(root / "plan.json")
    specimen = find(plan["cases"], case_id="sub_258_new")
    prefix = find(read_json(root / "prefixes.json")["rows"], case_id="sub_258_new")
    forecasts = read_json(root / "forecasts.json")["rows"]
    observed = read_json(root / "primary-observations.json")["rows"]
    rows = {}
    for arm in ARMS:
        actual = find(observed, case_id="sub_258_new", arm=arm)
        forecast = find(forecasts, case_id="sub_258_new", arm=arm)
        rows[arm] = {"stages": {key: actual["stages"][key] for key in STAGES},
                     "heads": actual["heads"], "reader": {"head": "regs", "register": 6, "bit": 1,
                         "descriptor": actual["heads"]["regs"]},
                     "conditional_payloads": {key: forecast["payloads"][key] for key in ("U_native", "P2_native")},
                     "conditional_register_terms": forecast["register_terms"],
                     "decoded": actual["decoded"], "mechanics_status": actual["mechanics_status"],
                     "intervention_recipe": actual["intervention_recipe"],
                     "replay": actual["replay"], "audit_exact_stages": actual["audit"]["exact_stages"]}
        for descriptor in [*rows[arm]["stages"].values(), *rows[arm]["heads"].values(),
                           *rows[arm]["conditional_payloads"].values(), *forecast["register_terms"].values()]:
            array(root, descriptor)
    return {"schema": "mdb-historical-low-prefix-index-v1", "historical_root": str(root), "specimen": specimen,
            "semantic_warning": "sub_258_new is OLD repeated-SUB context at RDI258; 'new' names the historical holdout",
            "metadata_files": {name: sha((root / name).read_bytes()) for name in files},
            "prefixes": {key: prefix[key] for key in ("Sn", "Se", "z", "source_zero", "roots", "trace_fingerprint")},
            "rows": rows, "model_calls": 0, "equation_evaluations": 0}


def current_comparison(historical_root, current_root, index):
    tx = current_root / "transaction"
    report = read_json(current_root / "transaction-report.json")
    if report["phase"] != "closed" or report.get("failure"):
        raise ValueError("current transaction must be complete and retained before comparison")
    observed = decode(tx, read_json(tx / "observations/native.json")["value"])
    forecast_doc = read_json(tx / "forecasts/campaign.json")
    predictions = decode(tx, forecast_doc["predictions"])["rows"]
    comparisons = {}
    for context in ("old", "new"):
        for arm in ARMS:
            row_id = context + "." + arm
            current = observed[row_id]
            prior = index["rows"][arm]
            stages = {}
            for key in STAGES:
                native_name = "z" if key == "z" else "layer2." + key
                stages[key] = compare(low(array(historical_root, prior["stages"][key]), key),
                                      low(current["stages"][native_name], key))
            saved_heads = {key: array(historical_root, descriptor) for key, descriptor in prior["heads"].items()}
            terms = {key: compare(low(array(historical_root, descriptor), "register_terms." + key),
                                  low(predictions[row_id]["register_terms"][key], "register_terms." + key))
                     for key, descriptor in prior["conditional_register_terms"].items()}
            payloads = {key: compare(low(array(historical_root, prior["conditional_payloads"][key]), key),
                                      low(array(tx, forecast_doc["payloads"][context + "_" + key]["value"]), key))
                        for key in ("U_native", "P2_native")}
            comparisons[row_id] = {"evidence_role": current["evidence_role"], "low_rows": [0, 1],
                "stages": stages, "heads": {key: compare(low(value, key), low(current["heads"][key], key))
                                             for key, value in saved_heads.items()},
                "reader_RSI_bit1": {"historical": float(saved_heads["regs"][0, 6, 1]),
                                     "current": float(current["heads"]["regs"][0, 6, 1])},
                "register_terms": terms, "conditional_vs_full_raw_payloads": payloads}
    return {"status": "saved-array-comparisons-complete", "current_root": str(current_root),
            "current_seal": report["seal_sha256"], "current_event_head": report["event_head"], "rows": comparisons,
            "limits": ["Historical RDI258 versus current RDI262 differ at input bit2, beyond the two compared low positions.",
                       "Historical payloads condition on observed FP32 prefix; current payloads use full raw FP64 equations.",
                       "Comparisons are diagnostics, not equivalence certification or new prospective forecasts.",
                       "Flags and non-register auxiliary heads are compared in full, not mislabeled position-local."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = historical_index(args.historical_root)
    result = {"index": index, "comparison": {"status": "waiting-for-collected-current-evidence"}}
    if args.current_root is not None:
        result["comparison"] = current_comparison(args.historical_root, args.current_root, index)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "comparison_status": result["comparison"]["status"],
                      "model_calls": 0, "equation_evaluations": 0}))


if __name__ == "__main__":
    main()

