"""Framework-free summary and scalar projection for the read-weight assay.

The worker has already paid for native captures, the sealed smooth producer,
and the downstream consumer arms.  This module only reads ``report.json`` and
keeps those observations separate: dose-zero anchors are retained, while the
transport-versus-control counts use nonzero doses only and never become a
scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

FORECAST_METHODS = ("transport", "linear-logweight", "stale", "wrong-sign", "query-only")
CONSUMER_METHODS = ("transport", "linear-logweight", "stale", "oracle")
BOUNDARIES = ("attention", "mlp", "exit")
PHASES = ("first", "second_candidate_world")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _json_scalar(value: Any) -> str | int | float | bool | None:
    """Encode a provenance value as one flat, deterministic scalar."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metric_row(
    report: dict[str, Any],
    report_sha256: str,
    case: dict[str, Any],
    phase: str,
    bundle: dict[str, Any],
    dose: float,
    method: str,
    metrics: dict[str, Any],
    weight_forecast: dict[str, Any],
) -> dict[str, Any]:
    case_spec = case.get("case", {})
    receiver = f"{case_spec.get('id', 'unknown')}:{phase}"
    head_rows = []
    for head in metrics.get("head_rows", []) or []:
        head_rows.append({
            "head_offset": int(head["head_offset"]),
            "true_delta_logp": _float(head.get("true_delta_logp")),
            "predicted_delta_logp": _float(head.get("predicted_delta_logp")),
            "error": _float(head.get("error")),
        })
    sites = bundle.get("sites")
    return {
        "schema": "mdb-read-weight-transport-forecast-row-v1",
        "job_id": report.get("job_id"),
        "request_sha256": report.get("request_sha256"),
        "report_sha256": report_sha256,
        "source_hashes": report.get("source_hashes"),
        "case": case_spec.get("id"),
        "phase": phase,
        "receiver": receiver,
        "owner_relation": case_spec.get("previous_current_owner_relation"),
        "dose": float(dose),
        "dose_zero": float(dose) == 0.0,
        "method": method,
        "native_parent_cut": bundle.get("parent_cut"),
        "native_sites": sites,
        "weight_forecast_ref": bundle.get("weight_forecast_ref"),
        "consumer_forecast_ref": bundle.get("consumer_forecast_ref"),
        "log_change_rmse": _float(metrics.get("log_change_rmse")),
        "log_change_max_abs": _float(metrics.get("log_change_max_abs")),
        "true_log_change_rms": _float(metrics.get("true_log_change_rms")),
        "probability_error": metrics.get("probability_error"),
        "predicted_probability_in_range": _bool(metrics.get("predicted_probability_in_range")),
        "head_count": len(head_rows),
        "head_rows": head_rows,
        "producer_wall_seconds": _float(weight_forecast.get("producer_wall_seconds")),
        "evaluation_wall_seconds": _float(
            weight_forecast.get("twenty_scalar_evaluations_wall_seconds")
        ),
        "serving_tensor_bytes": weight_forecast.get("serving_tensor_bytes"),
        "serving_tensor_elements": weight_forecast.get("serving_tensor_elements"),
    }


def _local_rows(
    report: dict[str, Any],
    report_sha256: str,
    case: dict[str, Any],
    phase: str,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    case_spec = case.get("case", {})
    receiver = f"{case_spec.get('id', 'unknown')}:{phase}"
    rows = []
    for method, boundaries in (bundle.get("local_errors") or {}).items():
        for boundary in BOUNDARIES:
            metrics = boundaries.get(boundary) if isinstance(boundaries, dict) else None
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "schema": "mdb-read-weight-transport-consumer-row-v1",
                "job_id": report.get("job_id"),
                "request_sha256": report.get("request_sha256"),
                "report_sha256": report_sha256,
                "case": case_spec.get("id"),
                "phase": phase,
                "receiver": receiver,
                "method": method,
                "boundary": boundary,
                "native_parent_cut": bundle.get("parent_cut"),
                "weight_forecast_ref": bundle.get("weight_forecast_ref"),
                "l2": _float(metrics.get("l2")),
                "max_abs": _float(metrics.get("max_abs")),
                "relative_l2": _float(metrics.get("relative_l2")),
                "exact": _bool(metrics.get("exact")),
                "injected_mlp_exact": _bool(boundaries.get("injected_mlp_exact")),
                "injected_exit_exact": _bool(boundaries.get("injected_exit_exact")),
            })
    return rows


def _future_rows(
    report: dict[str, Any],
    report_sha256: str,
    case: dict[str, Any],
    phase: str,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    case_spec = case.get("case", {})
    receiver = f"{case_spec.get('id', 'unknown')}:{phase}"
    rows = []
    for method, comparisons in (bundle.get("future_comparisons") or {}).items():
        for comparison in comparisons or []:
            rows.append({
                "schema": "mdb-read-weight-transport-future-row-v1",
                "job_id": report.get("job_id"),
                "request_sha256": report.get("request_sha256"),
                "report_sha256": report_sha256,
                "case": case_spec.get("id"),
                "phase": phase,
                "receiver": receiver,
                "method": method,
                "offset": comparison.get("offset"),
                "same_next": _bool(comparison.get("same_next")),
                "same_consumed": _bool(comparison.get("same_consumed")),
                "l2": _float(comparison.get("l2")),
                "max_abs": _float(comparison.get("max_abs")),
                "relative_l2": _float(comparison.get("relative_l2")),
                "native_parent_cut": bundle.get("parent_cut"),
                "weight_forecast_ref": bundle.get("weight_forecast_ref"),
            })
    return rows


def _history_rows(
    report: dict[str, Any],
    report_sha256: str,
    case: dict[str, Any],
    phase: str,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    case_spec = case.get("case", {})
    receiver = f"{case_spec.get('id', 'unknown')}:{phase}"
    rows = []
    for arm_name, arm in sorted((bundle.get("arms") or {}).items()):
        trajectory = []
        for item in arm.get("trajectory", []) or []:
            trajectory.append({
                "offset": item.get("offset"),
                "authority": item.get("authority"),
                "consumed_token_id": item.get("consumed_token_id"),
                "next_token_id": item.get("next_token_id"),
                "next_text": item.get("next_text"),
                "logits_sha256": item.get("logits_sha256"),
                "cut": item.get("cut"),
            })
        rows.append({
            "schema": "mdb-read-weight-transport-own-history-v1",
            "job_id": report.get("job_id"),
            "request_sha256": report.get("request_sha256"),
            "report_sha256": report_sha256,
            "case": case_spec.get("id"),
            "phase": phase,
            "receiver": receiver,
            "arm": arm_name,
            "native_parent_cut": arm.get("parent_cut", bundle.get("parent_cut")),
            "native_end_cut": arm.get("end_cut"),
            "trajectory": trajectory,
        })
    return rows


def _comparison(winner: list[dict[str, Any]], loser: list[dict[str, Any]]) -> dict[str, int]:
    pairs = []
    left_by_key = {(row.get("receiver"), float(row["dose"])): row for row in winner}
    right_by_key = {(row.get("receiver"), float(row["dose"])): row for row in loser}
    for key in sorted(left_by_key.keys() & right_by_key.keys(), key=repr):
        left, right = left_by_key[key], right_by_key[key]
        a, b = left.get("log_change_rmse"), right.get("log_change_rmse")
        if a is None or b is None:
            continue
        pairs.append((float(a), float(b)))
    return {
        "wins": sum(a < b for a, b in pairs),
        "ties": sum(a == b for a, b in pairs),
        "losses": sum(a > b for a, b in pairs),
        "eligible": len(pairs),
    }


def _aggregate_transport_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonzero = [row for row in rows if not row["dose_zero"]]
    by_receiver: dict[str, list[dict[str, Any]]] = {}
    for row in nonzero:
        by_receiver.setdefault(row["receiver"], []).append(row)

    receiver_summary = {}
    all_transport, all_stale, all_linear = [], [], []
    for receiver, receiver_rows in sorted(by_receiver.items()):
        methods = {
            method: sorted((row for row in receiver_rows if row["method"] == method), key=lambda row: row["dose"])
            for method in ("transport", "stale", "linear-logweight")
        }
        stale_comparison = _comparison(methods["transport"], methods["stale"])
        linear_comparison = _comparison(methods["transport"], methods["linear-logweight"])
        receiver_summary[receiver] = {
            "nonzero_dose_rows": len(receiver_rows),
            "nonzero_doses": sorted({row["dose"] for row in receiver_rows}),
            "transport_lower_than_stale": stale_comparison,
            "transport_lower_than_linear": linear_comparison,
        }
        all_transport.extend(methods["transport"])
        all_stale.extend(methods["stale"])
        all_linear.extend(methods["linear-logweight"])
    return {
        "nonzero_rows": len(nonzero),
        "dose_zero_rows_retained_but_excluded": len(rows) - len(nonzero),
        "transport_lower_than_stale": _comparison(all_transport, all_stale),
        "transport_lower_than_linear": _comparison(all_transport, all_linear),
        "by_receiver": receiver_summary,
        "gate": None,
        "interpretation": "directional observations over nonzero doses; no promotion gate",
    }


def _aggregate_local(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in sorted({row["method"] for row in rows}):
        result[method] = {}
        for boundary in BOUNDARIES:
            values = [row["relative_l2"] for row in rows if row["method"] == method and row["boundary"] == boundary
                      and row["relative_l2"] is not None]
            result[method][boundary] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "exact_count": sum(
                    row["exact"] is True for row in rows
                    if row["method"] == method and row["boundary"] == boundary
                ),
            }
    return result


def _aggregate_future(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        generated = [row for row in selected if row.get("offset", 0) > 0]
        result[method] = {
            "same_next": sum(row["same_next"] is True for row in selected),
            "next_decisions": len(selected),
            "same_consumed": sum(row["same_consumed"] is True for row in generated),
            "generated_decisions": len(generated),
            "receivers": len({row["receiver"] for row in selected}),
        }
    return result


def _costs(report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    receiver_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        receiver_rows.setdefault(row["receiver"], {
            "producer_wall_seconds": row["producer_wall_seconds"],
            "evaluation_wall_seconds": row["evaluation_wall_seconds"],
            "serving_tensor_bytes": row["serving_tensor_bytes"],
            "serving_tensor_elements": row["serving_tensor_elements"],
        })
    values = list(receiver_rows.values())
    return {
        "producer_wall_seconds": sum((v["producer_wall_seconds"] or 0.0) for v in values),
        "evaluation_wall_seconds": sum((v["evaluation_wall_seconds"] or 0.0) for v in values),
        "serving_tensor_bytes": sum((v["serving_tensor_bytes"] or 0) for v in values),
        "serving_tensor_elements": sum((v["serving_tensor_elements"] or 0) for v in values),
        "per_receiver": receiver_rows,
        "compilation_paid": {
            "smooth_dense_linears": 19,
            "smooth_directional_tangent": True,
            "local_source_o_projection_linears": 4,
            "after_seal_dense_control_linears": 19,
            "forecast_mlp_linears": 12,
            "forecast_scalar_evaluations": 20,
            "scope": "per receiver unless explicitly stated; source and consumer counts are declared worker costs",
        },
        "worker_scope": report.get("compilation_scope"),
        "consumer_scope": report.get("consumer_scope"),
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    """Read one completed run and return a compact, immutable-friendly summary."""

    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    report = read(report_path)
    report_sha256 = _sha256(report_path)
    forecast_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    receivers = []
    for case in report.get("cases", []):
        case_spec = case.get("case", {})
        for phase in PHASES:
            bundle = case.get(phase)
            if not isinstance(bundle, dict):
                continue
            receiver = f"{case_spec.get('id', 'unknown')}:{phase}"
            weight_forecast = bundle.get("weight_forecast") or {}
            rows_for_receiver = []
            for dose_row in bundle.get("dose_rows", []) or []:
                dose = float(dose_row.get("dose"))
                for method, metrics in (dose_row.get("methods") or {}).items():
                    if not isinstance(metrics, dict):
                        continue
                    row = _metric_row(
                        report, report_sha256, case, phase, bundle, dose, method, metrics, weight_forecast
                    )
                    forecast_rows.append(row)
                    rows_for_receiver.append(row)
            receiver_local = _local_rows(report, report_sha256, case, phase, bundle)
            receiver_future = _future_rows(report, report_sha256, case, phase, bundle)
            receiver_history = _history_rows(report, report_sha256, case, phase, bundle)
            local_rows.extend(receiver_local)
            future_rows.extend(receiver_future)
            history_rows.extend(receiver_history)
            receivers.append({
                "receiver": receiver,
                "case": case_spec.get("id"),
                "phase": phase,
                "owner_relation": case_spec.get("previous_current_owner_relation"),
                "parent_cut": bundle.get("parent_cut"),
                "sites": bundle.get("sites"),
                "weight_forecast_ref": bundle.get("weight_forecast_ref"),
                "consumer_forecast_ref": bundle.get("consumer_forecast_ref"),
                "forecast_rows": rows_for_receiver,
                "local_consumer_errors": receiver_local,
                "next_token_agreement": [row for row in receiver_future],
                "own_candidate_history": receiver_history,
                "q_transport_error": bundle.get("q_transport_error"),
                "k_transport_error": bundle.get("k_transport_error"),
                "dense_fp32_control_errors": bundle.get("dense_fp32_control_errors"),
                "native_module_entries_during_forecast": bundle.get("native_module_entries_during_forecast"),
                "mechanics_status": bundle.get("mechanics_status"),
                "terminal_status": bundle.get("terminal_status", "not-assessed"),
            })

    return {
        "schema": "mdb-read-weight-transport-summary-v1",
        "job_id": report.get("job_id"),
        "request_sha256": report.get("request_sha256"),
        "report_sha256": report_sha256,
        "source_hashes": report.get("source_hashes"),
        "model": report.get("model"),
        "weights_sha256": report.get("weights_sha256"),
        "parameter_identity": report.get("parameter_identity"),
        "custody": report.get("artifact_custody"),
        "receiver_count": len(receivers),
        "receivers": receivers,
        "forecast_rows": forecast_rows,
        "local_consumer_errors": local_rows,
        "next_token_rows": future_rows,
        "own_candidate_history": history_rows,
        "forecast_aggregates": _aggregate_transport_rows(forecast_rows),
        "local_error_aggregates": _aggregate_local(local_rows),
        "next_token_agreement": _aggregate_future(future_rows),
        "costs": _costs(report, forecast_rows),
        "physical_token_steps": report.get("physical_token_steps"),
        "prefill_calls": report.get("prefill_calls"),
        "prefill_token_count": report.get("prefill_token_count"),
        "dose_count": report.get("dose_count"),
        "compilation_scope": report.get("compilation_scope"),
        "consumer_scope": report.get("consumer_scope"),
        "mechanics_status": report.get("mechanics_status", "observed"),
        "instrument_status": "native-capture-and-sealed-forecast-rows",
        "trend_status": "all-rows-retained; comparisons are observations",
        "terminal_status": report.get("terminal_status", "not-assessed"),
        "semantic_cot_qualified": report.get("semantic_cot_qualified", False),
        "economical_full_query_qualified": report.get("economical_full_query_qualified", False),
    }


def _binding(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row.get("job_id"),
        "request_sha256": row.get("request_sha256"),
        "report_sha256": row.get("report_sha256"),
        "case": row.get("case"),
        "phase": row.get("phase"),
        "receiver": row.get("receiver"),
        "native_parent_cut": row.get("native_parent_cut"),
        "weight_forecast_ref": _json_scalar(row.get("weight_forecast_ref")),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def write_evidence_projection(result: dict[str, Any], destination: Path) -> None:
    """Write flat scalar JSONL projections with native source binding."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    observations = []
    head_observations = []
    for row in result["forecast_rows"]:
        observations.append({
            "schema": "mdb-read-weight-transport-scalar-v1",
            **_binding(row),
            "dose": row["dose"],
            "dose_zero": row["dose_zero"],
            "method": row["method"],
            "head_count": row["head_count"],
            "log_change_rmse": row["log_change_rmse"],
            "log_change_max_abs": row["log_change_max_abs"],
            "true_log_change_rms": row["true_log_change_rms"],
            "producer_wall_seconds": row["producer_wall_seconds"],
            "evaluation_wall_seconds": row["evaluation_wall_seconds"],
            "serving_tensor_bytes": row["serving_tensor_bytes"],
            "serving_tensor_elements": row["serving_tensor_elements"],
        })
        for head in row["head_rows"]:
            head_observations.append({
                "schema": "mdb-read-weight-transport-head-scalar-v1",
                **_binding(row),
                "dose": row["dose"],
                "dose_zero": row["dose_zero"],
                "method": row["method"],
                **head,
            })
    local_observations = []
    for row in result["local_consumer_errors"]:
        local_observations.append({"schema": "mdb-read-weight-transport-local-scalar-v1", **_binding(row),
                                   **{key: value for key, value in row.items()
                                      if key not in {"schema", "job_id", "request_sha256", "report_sha256", "case", "phase", "receiver", "native_parent_cut", "weight_forecast_ref"}}})
    future_observations = []
    for row in result["next_token_rows"]:
        future_observations.append({"schema": "mdb-read-weight-transport-future-scalar-v1", **_binding(row),
                                    **{key: value for key, value in row.items()
                                       if key not in {"schema", "job_id", "request_sha256", "report_sha256", "case", "phase", "receiver", "native_parent_cut", "weight_forecast_ref"}}})
    _write_jsonl(destination / "observations.jsonl", observations)
    _write_jsonl(destination / "head-observations.jsonl", head_observations)
    _write_jsonl(destination / "local-observations.jsonl", local_observations)
    _write_jsonl(destination / "future-observations.jsonl", future_observations)
    manifest = {
        "schema": "mdb-read-weight-transport-evidence-projection-v1",
        "job_id": result["job_id"],
        "request_sha256": result["request_sha256"],
        "report_sha256": result["report_sha256"],
        "observations": len(observations),
        "head_observations": len(head_observations),
        "local_observations": len(local_observations),
        "future_observations": len(future_observations),
        "dose_zero_retained": True,
        "comparison_gate": None,
        "native_source_binding": "job/request/report/case/phase/parent-cut/forecast-ref on every row",
        "terminal_status": "not-assessed",
    }
    (destination / "source.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep aggregates and unique scalar rows in Git, without duplicate tables."""
    omitted = {"receivers", "forecast_rows", "local_consumer_errors", "next_token_rows", "own_candidate_history"}
    compact = {key: value for key, value in result.items() if key not in omitted}
    compact["schema"] = "mdb-read-weight-transport-compact-summary-v1"
    compact["forecast_rows"] = [{key: row[key] for key in (
        "receiver", "dose", "method", "log_change_rmse", "log_change_max_abs",
        "true_log_change_rms", "predicted_probability_in_range")} for row in result["forecast_rows"]]
    compact["full_dose_transport_heads"] = [
        {"receiver": row["receiver"], **head} for row in result["forecast_rows"]
        if row["dose"] == 1.0 and row["method"] == "transport" for head in row["head_rows"]]
    compact["receivers"] = []
    for receiver in result["receivers"]:
        record = {key: value for key, value in receiver.items() if key not in (
            "forecast_rows", "local_consumer_errors", "next_token_agreement", "own_candidate_history")}
        record["first_token_text"] = {arm["arm"]: arm["trajectory"][0]["next_text"]
                                      for arm in receiver["own_candidate_history"]}
        compact["receivers"].append(record)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--compact-output", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.evidence_dir:
        write_evidence_projection(result, args.evidence_dir)
    if args.compact_output:
        args.compact_output.write_text(json.dumps(compact_summary(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "job_id": result["job_id"],
        "receivers": result["receiver_count"],
        "forecast_rows": len(result["forecast_rows"]),
        "local_rows": len(result["local_consumer_errors"]),
        "future_rows": len(result["next_token_rows"]),
        "nonzero_comparisons": result["forecast_aggregates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
