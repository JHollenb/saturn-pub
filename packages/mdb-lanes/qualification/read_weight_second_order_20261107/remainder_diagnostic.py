"""Framework-free saved-only Taylor diagnosis (NumPy; use common Python).

Usage: python remainder_diagnostic.py RUN_DIR OUTPUT_JSON
No model loading or execution. Complete smooth controls were paid after sealing;
their use here is post hoc, not a new forecast or an economical implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rmse(value):
    return float(np.sqrt(np.mean(value * value)))


def logsoftmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def diagnose(run):
    report_raw = (run / "report.json").read_bytes()
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
            values = np.frombuffer(raw, dtype={"torch.float32": "<f4", "torch.float64": "<f8"}[descriptor["dtype"]])
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
            bundle, used = case[phase], {}
            forecast = bundle["weight_forecast"]
            program, compiled = forecast["taylor_program"], forecast["compiled"]
            baseline, native = (bundle["arms"][arm]["capture"] for arm in ("00", "A-1.0"))
            heads, group, source = program["query_heads"], program["kv_head"], program["source_row"]
            q0 = load(case_id, baseline["q22"], used, "q.native0")[0, heads, 0]
            k0 = load(case_id, baseline["k22"], used, "k.native0")[0, group]
            displacements = []
            for index, field in enumerate(("q", "k")):
                s0 = load(case_id, compiled["shadow_" + field + "0"], used, field + ".shadow0")
                sA = load(case_id, bundle["dense_fp32_control_qk"][index], used, field + ".shadowA")
                displacements.append(sA - s0)
            qbar = q0 + displacements[0][0, heads, 0]
            kbar = k0.copy()
            kbar[program["current_row"]] += displacements[1][0, group, 0]
            scores0 = load(case_id, baseline["scores22"], used, "scores.native0")[0, heads, 0]
            scoresA = load(case_id, native["scores22"], used, "scores.nativeA")[0, heads, 0]
            smooth = logsoftmax(scores0 + (qbar @ kbar.T - q0 @ k0.T) * program["scaling"])[:, source]
            actual = logsoftmax(scoresA)[:, source]
            polynomial = sum(np.asarray(program[key], dtype=np.float64)[:, 0] * scale
                             for key, scale in (("logp0", 1), ("c1", 1), ("c2", 0.5)))
            error, remainder, profile = polynomial - actual, polynomial - smooth, smooth - actual
            closure = error - (remainder + profile)
            primary = next(row for row in bundle["dose_rows"] if row["dose"] == 1.0)["methods"]["second-order"]
            rows.append({"case_id": case_id, "phase": phase, "dose": 1.0, "input_payloads": used,
                "weight_forecast_ref": bundle["weight_forecast_ref"], "report_program": program,
                "total_rmse_fp64": rmse(error), "taylor_to_smooth_rmse_fp64": rmse(remainder),
                "smooth_to_native_rmse_fp64": rmse(profile),
                "cross_mean": float(np.mean(remainder * profile)),
                "decomposition_max_abs": float(np.abs(closure).max()),
                "decomposition_exact_fp64": bool(np.count_nonzero(closure) == 0),
                "sealed_second_order_rmse_reported": primary["log_change_rmse"],
                "head_rows": [{"query_head": head, "polynomial_logp": float(t),
                    "anchored_smooth_logp": float(s), "native_logp": float(n),
                    "total_error": float(e), "taylor_to_smooth": float(r), "smooth_to_native": float(p)}
                    for head, t, s, n, e, r, p in zip(heads, polynomial, smooth, actual, error, remainder, profile, strict=True)]})
    return {"schema": "mdb-taylor-remainder-diagnostic-v1", "status": "post-hoc-paid-control-diagnostic",
        "scope": "Saved payloads only; zero model executions. Not a sealed forecast or serving improvement.",
        "report_sha256": sha(report_raw), "script_sha256": sha(Path(__file__).read_bytes()),
        "job_id": report["job_id"], "numpy_version": np.__version__, "receiver_count": len(rows),
        "decomposition": "T-N=(T-S)+(S-N): T=g0+c1+c2/2; S=logsoftmax(native_scores0 + (qbar kbar^T-q0 k0^T)/sqrt(d))[B]; N=logsoftmax(native_scoresA)[B].",
        "interpretation": "T widens saved FP32 coefficients to FP64; this differs slightly from sealed FP32 evaluation. T-S includes truncation and smooth numerical/coefficient errors. S-N is the anchored smooth/native finite-profile discrepancy, not BF16 rounding alone. RMSE components are not additive causal percentages: total MSE = remainder MSE + profile MSE + 2*cross_mean.",
        "anchor": "qbar=q_native0+(q_smoothA-q_smooth0); current kbar likewise; old native keys fixed. Saved smoothA is a paid full-dose post-seal control.",
        "rows": rows, "payloads": [payloads[key] for key in sorted(payloads)]}


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
