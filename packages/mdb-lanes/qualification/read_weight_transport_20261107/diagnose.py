"""Saved-only FP64 transport diagnosis; no model imports or forecast claims.

Usage: python diagnose.py RUN_DIR OUTPUT_JSON
The output is a post-hoc analysis of already paid smooth/native controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(value):
    return float(np.linalg.norm(value))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def logsoftmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def diagnose(run):
    report_path = run / "report.json"
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    payloads, rows = {}, []

    def load(case_id, descriptor, used, label):
        digest = descriptor["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid payload digest")
        if not descriptor["ref"].endswith("/" + digest):
            raise ValueError("payload reference/digest mismatch")
        path = run / "results" / case_id / "vm/radix/cas/objects" / (digest + ".bin")
        raw = path.read_bytes()
        if sha(raw) != digest or len(raw) != descriptor["bytes"]:
            raise ValueError(f"payload integrity mismatch: {path}")
        if descriptor["dtype"] == "torch.bfloat16":
            values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4")
        else:
            dtype = {"torch.float32": "<f4", "torch.float64": "<f8"}[descriptor["dtype"]]
            values = np.frombuffer(raw, dtype=dtype)
        values = values.reshape(descriptor["shape"]).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite input: {path}")
        relative = path.relative_to(run).as_posix()
        payloads[relative] = {"path": relative, "sha256": digest, "descriptor": descriptor}
        used[label] = relative
        return values

    for case in report["cases"]:
        case_id = case["case"]["id"]
        for phase in ("first", "second_candidate_world"):
            bundle, used, fields = case[phase], {}, {}
            baseline = bundle["arms"]["00"]["capture"]
            native = bundle["arms"]["A-1.0"]["capture"]
            compiled = bundle["weight_forecast"]["compiled"]
            dense = bundle["dense_fp32_control_qk"]
            arrays = {}
            for index, field in enumerate(("q", "k")):
                descriptors = {"native0": baseline[field + "22"], "nativeA": native[field + "22"],
                               "shadow0": compiled["shadow_" + field + "0"],
                               "shadowA": dense[index], "tangent": compiled[field + "_dot"]}
                values = {name: load(case_id, ref, used, field + "." + name)
                          for name, ref in descriptors.items()}
                arrays[field] = values
                n0, nA = values["native0"], values["nativeA"]
                if field == "k":
                    n0, nA = n0[:, :, -1:], nA[:, :, -1:]
                s0, sA, tangent = (values[name] for name in ("shadow0", "shadowA", "tangent"))
                remainder = sA - s0 - tangent
                lift_difference = (sA - nA) - (s0 - n0)
                error = n0 + tangent - nA
                closure = error - (-remainder + lift_difference)
                fields[field] = {"tangent_error_l2": norm(error), "smooth_remainder_l2": norm(remainder),
                    "finite_profile_difference_l2": norm(lift_difference),
                    "remainder_over_error": ratio(norm(remainder), norm(error)),
                    "profile_difference_over_error": ratio(norm(lift_difference), norm(error)),
                    "anchored_nonlinear_relative_l2": ratio(norm(lift_difference), norm(nA)),
                    "decomposition_residual_l2": norm(closure),
                    "decomposition_max_abs": float(np.abs(closure).max()),
                    "decomposition_exact_fp64": bool(np.count_nonzero(closure) == 0)}
            program = bundle["weight_forecast"]["score_program"]
            heads, group, source = list(program["query_heads"]), program["kv_head"], program["source_row"]
            q, k = arrays["q"], arrays["k"]
            q0, k0 = q["native0"][0, heads, 0], k["native0"][0, group]
            qhat = q0 + (q["shadowA"] - q["shadow0"])[0, heads, 0]
            khat = k0.copy()
            khat[-1] += (k["shadowA"] - k["shadow0"])[0, group, 0]
            score0 = load(case_id, baseline["scores22"], used, "scores.native0")[0, heads, 0]
            scoreA = load(case_id, native["scores22"], used, "scores.nativeA")[0, heads, 0]
            predicted = logsoftmax(score0 + (qhat @ khat.T - q0 @ k0.T) * program["scaling"])[:, source]
            actual = logsoftmax(scoreA)[:, source]
            errors = predicted - actual
            primary = next(d for d in bundle["dose_rows"] if d["dose"] == 1.0)["methods"]["transport"]
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            rows.append({"case_id": case_id, "phase": phase, "input_payloads": used, "fields": fields,
                "weight_forecast_ref": bundle["weight_forecast_ref"],
                "posthoc_nonlinear_logweight_rmse_fp64": rmse,
                "sealed_transport_logweight_rmse_reported": primary["log_change_rmse"],
                "posthoc_lower_than_reported_transport": rmse < primary["log_change_rmse"],
                "head_rows": [{"query_head": head, "predicted_logp": float(p), "native_logp": float(a),
                               "error": float(e)} for head, p, a, e in zip(heads, predicted, actual, errors, strict=True)]})
    return {"schema": "mdb-transport-remainder-diagnostic-v1", "status": "post-hoc-paid-control-diagnostic",
        "scope": "Saved payloads only; zero model executions. Not a sealed forecast or serving improvement.",
        "report_sha256": sha(report_raw), "job_id": report["job_id"], "numpy_version": np.__version__,
        "script_sha256": sha(Path(__file__).read_bytes()),
        "decomposition": "error = native0 + Jd - nativeA = -R + E; R = shadowA - shadow0 - Jd; E = (shadowA-nativeA) - (shadow0-native0)",
        "interpretation": "Norm ratios are not additive causal percentages. The nonlinear smooth controls were already paid. The primary comparator is its saved FP32 report metric; this diagnostic uses FP64.",
        "receiver_count": len(rows), "rows": rows, "payloads": [payloads[key] for key in sorted(payloads)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "receivers": result["receiver_count"], "verified_payloads": len(result["payloads"])}))
