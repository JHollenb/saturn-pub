"""Framework-free scalar summary for the Qwen read-weight closure assay.

The analyzer reads a custodied ``report.json`` only.  It retains every receiver,
dose, method, head, native-profile field, and future comparison.  Its paired
comparisons are descriptive observations; they do not assign scientific gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

DOSES = (-0.25, 0.0, 0.2, 0.6, 1.0)
FORECAST_METHODS = (
    "midpoint",
    "baseline-corrected",
    "zero-centered",
    "stale",
    "wrong-sign",
    "wrong-curvature",
)
CONSUMER_METHODS = (
    "midpoint",
    "baseline-corrected",
    "zero-centered",
    "stale",
    "functional-native",
)
PHASES = ("first", "second_candidate_world")
PROFILE_BOUNDARIES = ("input", "attention", "mlp", "exit")
SPAN_ARMS = (*CONSUMER_METHODS, "functional-zero")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _matrix(value: Any, *, label: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            return []
        converted = []
        for item in row:
            number = _float(item)
            if number is None:
                raise ValueError(f"{label} contains a missing or non-finite value")
            converted.append(number)
        rows.append(converted)
    return rows


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _metric_copy(metrics: Any) -> dict[str, Any]:
    return _json_safe(metrics) if isinstance(metrics, dict) else {}


def _forecast_rows(
    report: dict[str, Any], report_sha: str, case: dict[str, Any], phase: str, bundle: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = case.get("case", {})
    case_id = spec.get("id", "unknown")
    receiver = f"{case_id}:{phase}"
    forecast_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    for dose_row in bundle.get("dose_rows", []) or []:
        dose = _float(dose_row.get("dose"))
        if dose is None:
            continue
        for method, metrics in (dose_row.get("methods") or {}).items():
            if not isinstance(metrics, dict):
                continue
            row = {
                "schema": "mdb-read-weight-closure-forecast-row-v1",
                "job_id": report.get("job_id"),
                "request_sha256": report.get("request_sha256"),
                "report_sha256": report_sha,
                "source_hashes": report.get("source_hashes"),
                "case": case_id,
                "split": spec.get("split", "unknown"),
                "phase": phase,
                "receiver": receiver,
                "owner_relation": spec.get("previous_current_owner_relation"),
                "parent_cut": bundle.get("parent_cut"),
                "sites": bundle.get("sites"),
                "dose": dose,
                "dose_zero": dose == 0.0,
                "method": method,
                "log_change_rmse": _float(metrics.get("log_change_rmse")),
                "log_change_max_abs": _float(metrics.get("log_change_max_abs")),
                "true_log_change_rms": _float(metrics.get("true_log_change_rms")),
                "probability_error": _metric_copy(metrics.get("probability_error")),
                "predicted_probability_in_range": _bool(metrics.get("predicted_probability_in_range")),
                "head_count": len(metrics.get("head_rows", []) or []),
            }
            forecast_rows.append(row)
            for head in metrics.get("head_rows", []) or []:
                head_rows.append({
                    "schema": "mdb-read-weight-closure-head-row-v1",
                    "job_id": report.get("job_id"),
                    "request_sha256": report.get("request_sha256"),
                    "report_sha256": report_sha,
                    "case": case_id,
                    "split": spec.get("split", "unknown"),
                    "phase": phase,
                    "receiver": receiver,
                    "dose": dose,
                    "dose_zero": dose == 0.0,
                    "method": method,
                    "observation": "forecast-head-error",
                    "head_offset": int(head.get("head_offset", -1)),
                    "true_delta_logp": _float(head.get("true_delta_logp")),
                    "predicted_delta_logp": _float(head.get("predicted_delta_logp")),
                    "error": _float(head.get("error")),
                })
    return forecast_rows, head_rows


def _program_diagnostics(
    report: dict[str, Any], report_sha: str, case: dict[str, Any], phase: str, bundle: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = case.get("case", {})
    case_id = spec.get("id", "unknown")
    receiver = f"{case_id}:{phase}"
    forecast = bundle.get("forecast") or {}
    midpoint = forecast.get("midpoint_program") or {}
    zero_program = forecast.get("zero_program") or {}
    matrix = _matrix(midpoint.get("zero_centered_residual"), label="midpoint zero residual")
    t0 = _matrix(midpoint.get("centered_zero_logp"), label="midpoint centered zero prediction")
    native0 = _matrix(midpoint.get("logp_native0"), label="native zero-dose logweights")
    zero_matrix = _matrix(zero_program.get("zero_centered_residual"), label="zero-centered zero residual")
    if matrix and t0 and native0 and not (len(matrix) == len(t0) == len(native0)):
        raise ValueError(f"program diagnostic head count differs for {receiver}")
    residual_values = [value for row in matrix for value in row]
    residual_rows: list[dict[str, Any]] = []
    for head_index, row in enumerate(matrix):
        for selected_index, residual in enumerate(row):
            residual_rows.append({
                "schema": "mdb-read-weight-closure-head-row-v1",
                "job_id": report.get("job_id"),
                "request_sha256": report.get("request_sha256"),
                "report_sha256": report_sha,
                "case": case_id,
                "split": spec.get("split", "unknown"),
                "phase": phase,
                "receiver": receiver,
                "dose": 0.0,
                "dose_zero": True,
                "method": "midpoint",
                "observation": "raw-midpoint-zero-residual",
                "head_offset": head_index,
                "selected_row_index": selected_index,
                "logp_native0": native0[head_index][selected_index],
                "centered_zero_logp": t0[head_index][selected_index],
                "zero_centered_residual": residual,
            })
    diagnostic = {
        "center": _float(midpoint.get("center")),
        "raw_midpoint_zero_residual": matrix,
        "raw_midpoint_zero_residual_rmse": (
            math.sqrt(statistics.fmean(value * value for value in residual_values)) if residual_values else None
        ),
        "raw_midpoint_zero_residual_max_abs": max((abs(value) for value in residual_values), default=None),
        "zero_centered_zero_residual": zero_matrix,
        "zero_centered_zero_residual_max_abs": max(
            (abs(value) for row in zero_matrix for value in row), default=None
        ),
        "midpoint_program_cost": _json_safe(midpoint.get("cost", {})),
        "zero_program_cost": _json_safe(zero_program.get("cost", {})),
        "baseline_corrected_zero_delta_exact_by_construction": True,
        "source_refs": {
            "midpoint_program": _json_safe(midpoint),
            "zero_program": _json_safe(zero_program),
        },
    }
    return diagnostic, residual_rows


def _local_rows(
    report: dict[str, Any], report_sha: str, case: dict[str, Any], phase: str, bundle: dict[str, Any]
) -> list[dict[str, Any]]:
    spec = case.get("case", {})
    case_id = spec.get("id", "unknown")
    receiver = f"{case_id}:{phase}"
    rows: list[dict[str, Any]] = []
    binding = {
        "job_id": report.get("job_id"),
        "request_sha256": report.get("request_sha256"),
        "report_sha256": report_sha,
        "case": case_id,
        "split": spec.get("split", "unknown"),
        "phase": phase,
        "receiver": receiver,
        "parent_cut": bundle.get("parent_cut"),
    }
    for dose_row in bundle.get("dose_rows", []) or []:
        dose = _float(dose_row.get("dose"))
        if dose is None:
            continue
        for field, metrics in (dose_row.get("functional_profile_errors") or {}).items():
            rows.append({
                "schema": "mdb-read-weight-closure-local-row-v1",
                **binding,
                "row_kind": "native-profile-field",
                "dose": dose,
                "field": field,
                "metrics": _metric_copy(metrics),
                "exact": _bool(metrics.get("exact")) if isinstance(metrics, dict) else None,
            })
        rows.append({
            "schema": "mdb-read-weight-closure-local-row-v1",
            **binding,
            "row_kind": "functional-profile-selected-logweight",
            "dose": dose,
            "field": "selected-logweight",
            "rmse": _float(dose_row.get("functional_logp_rmse")),
        })
        rows.append({
            "schema": "mdb-read-weight-closure-local-row-v1",
            **binding,
            "row_kind": "source-write-reconstruction",
            "dose": dose,
            "field": "attention19",
            "metrics": _metric_copy(dose_row.get("local_source_error")),
        })
    for method, boundaries in (bundle.get("local_errors") or {}).items():
        for boundary, metrics in (boundaries or {}).items():
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "schema": "mdb-read-weight-closure-local-row-v1",
                **binding,
                "row_kind": "consumer-boundary-error",
                "dose": 1.0,
                "method": method,
                "boundary": boundary,
                "metrics": _metric_copy(metrics),
                "injected_exit_exact": _bool(boundaries.get("injected_exit_exact")),
                "injected_current_k_exact": _bool(boundaries.get("injected_current_k_exact")),
            })
    return rows


def _future_rows(
    report: dict[str, Any], report_sha: str, case: dict[str, Any], phase: str, bundle: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = case.get("case", {})
    case_id = spec.get("id", "unknown")
    receiver = f"{case_id}:{phase}"
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    binding = {
        "job_id": report.get("job_id"),
        "request_sha256": report.get("request_sha256"),
        "report_sha256": report_sha,
        "case": case_id,
        "split": spec.get("split", "unknown"),
        "phase": phase,
        "receiver": receiver,
        "parent_cut": bundle.get("parent_cut"),
    }
    for method, comparisons in (bundle.get("future_comparisons") or {}).items():
        for comparison in comparisons or []:
            rows.append({
                "schema": "mdb-read-weight-closure-future-row-v1",
                **binding,
                "row_kind": "candidate-vs-native-AB",
                "method": method,
                "offset": comparison.get("offset"),
                "same_next": _bool(comparison.get("same_next")),
                "same_consumed": _bool(comparison.get("same_consumed")),
                "l2": _float(comparison.get("l2")),
                "max_abs": _float(comparison.get("max_abs")),
                "relative_l2": _float(comparison.get("relative_l2")),
            })
    arms = bundle.get("arms") or {}
    for arm_name, arm in sorted(arms.items()):
        for step in arm.get("trajectory", []) or []:
            history = {
                "schema": "mdb-read-weight-closure-future-row-v1",
                **binding,
                "row_kind": "own-history",
                "method": arm_name,
                "arm": arm_name,
                "offset": step.get("offset"),
                "authority": step.get("authority"),
                "consumed_token_id": step.get("consumed_token_id"),
                "next_token_id": step.get("next_token_id"),
                "next_text": step.get("next_text"),
                "target_margin": _float(step.get("target_margin")),
                "direct_object_token_match": _bool(step.get("direct_object_token_match")),
                "logits_sha256": step.get("logits_sha256"),
                "cut": step.get("cut"),
                "end_cut": arm.get("end_cut"),
            }
            histories.append(history)
            if arm_name in (*CONSUMER_METHODS, "AB"):
                rows.append(history)
    return rows, histories


def _comparison(rows: list[dict[str, Any]], method_a: str, method_b: str) -> dict[str, int]:
    by_key = {(row["receiver"], float(row["dose"])): row for row in rows if row.get("method") == method_a}
    other = {(row["receiver"], float(row["dose"])): row for row in rows if row.get("method") == method_b}
    pairs = []
    for key in by_key.keys() & other.keys():
        left = _float(by_key[key].get("log_change_rmse"))
        right = _float(other[key].get("log_change_rmse"))
        if left is not None and right is not None:
            pairs.append((left, right))
    return {
        "midpoint_lower_rmse": sum(left < right for left, right in pairs),
        "ties": sum(left == right for left, right in pairs),
        "midpoint_higher_rmse": sum(left > right for left, right in pairs),
        "eligible_pairs": len(pairs),
    }


def _aggregate_forecasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dose: dict[str, Any] = {}
    by_split: dict[str, Any] = {}
    for dose in sorted({float(row["dose"]) for row in rows}):
        dose_rows = [row for row in rows if float(row["dose"]) == dose]
        method_stats = {}
        for method in FORECAST_METHODS:
            selected = [row for row in dose_rows if row["method"] == method]
            rmse = [value for row in selected if (value := _float(row.get("log_change_rmse"))) is not None]
            method_stats[method] = {
                **_stats(rmse),
                "receiver_count": len({row["receiver"] for row in selected}),
                "predicted_probability_in_range_count": sum(
                    row.get("predicted_probability_in_range") is True for row in selected
                ),
            }
        by_dose[str(dose)] = {
            "receiver_count": len({row["receiver"] for row in dose_rows}),
            "method_rmse": method_stats,
            "paired_lower_rmse_counts_vs_midpoint": {
                method: _comparison(dose_rows, "midpoint", method)
                for method in FORECAST_METHODS
                if method != "midpoint"
            },
        }
        for split in sorted({row.get("split", "unknown") for row in dose_rows}):
            selected = [row for row in dose_rows if row.get("split", "unknown") == split]
            method_stats = {}
            for method in FORECAST_METHODS:
                method_rows = [row for row in selected if row["method"] == method]
                values = [value for row in method_rows if (value := _float(row.get("log_change_rmse"))) is not None]
                method_stats[method] = {
                    **_stats(values),
                    "receiver_count": len({row["receiver"] for row in method_rows}),
                }
            by_split.setdefault(split, {})[str(dose)] = {
                "receiver_count": len({row["receiver"] for row in selected}),
                "method_rmse": method_stats,
                "paired_lower_rmse_counts_vs_midpoint": {
                    method: _comparison(selected, "midpoint", method)
                    for method in FORECAST_METHODS
                    if method != "midpoint"
                },
            }
    return {
        "forecast_row_count": len(rows),
        "receiver_count": len({row["receiver"] for row in rows}),
        "dose_zero_rows_retained": sum(row.get("dose_zero") is True for row in rows),
        "zero_dose_included_in_paired_comparisons": True,
        "methods": list(FORECAST_METHODS),
        "by_dose": by_dose,
        "by_split": by_split,
        "comparison_authority": "descriptive paired observations; no scientific gate",
    }


def _aggregate_heads(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forecasts = [row for row in rows if row.get("observation") == "forecast-head-error"]
    result: dict[str, Any] = {}
    for dose in sorted({float(row["dose"]) for row in forecasts}):
        dose_rows = [row for row in forecasts if float(row["dose"]) == dose]
        result[str(dose)] = {}
        for method in FORECAST_METHODS:
            selected = [row for row in dose_rows if row.get("method") == method]
            matches = []
            for row in selected:
                true_value = _float(row.get("true_delta_logp"))
                predicted = _float(row.get("predicted_delta_logp"))
                if true_value is not None and predicted is not None:
                    matches.append((true_value == 0.0 and predicted == 0.0) or true_value * predicted > 0.0)
            result[str(dose)][method] = {
                "head_observation_count": len(selected),
                "direction_match_count": sum(matches),
                "direction_mismatch_count": len(matches) - sum(matches),
                "direction_unavailable_count": len(selected) - len(matches),
            }
    return result


def _aggregate_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [row for row in rows if row.get("row_kind") == "native-profile-field"]
    aggregate: dict[str, Any] = {}
    for dose in sorted({float(row["dose"]) for row in profiles}):
        dose_rows = [row for row in profiles if float(row["dose"]) == dose]
        fields: dict[str, Any] = {}
        for field in sorted({str(row.get("field")) for row in dose_rows}):
            selected = [row for row in dose_rows if row.get("field") == field]
            values = [value for row in selected
                      if isinstance(row.get("metrics"), dict)
                      and (value := _float(row["metrics"].get("relative_l2"))) is not None]
            fields[field] = {
                "relative_l2": _stats(values),
                "exact_count": sum(row.get("exact") is True for row in selected),
                "observation_count": len(selected),
            }
        aggregate[str(dose)] = fields
    return {
        "row_count": len(profiles),
        "by_dose_field": aggregate,
        "gate": None,
        "interpretation": "functional/native profile diagnostics by exact field and dose",
    }


def _aggregate_future(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [row for row in rows if row.get("row_kind") == "candidate-vs-native-AB"]
    histories = [row for row in rows if row.get("row_kind") == "own-history"]
    result: dict[str, Any] = {}
    for method in CONSUMER_METHODS:
        selected = [row for row in comparisons if row.get("method") == method]
        own = [row for row in histories if row.get("method") == method]
        generated = [row for row in selected if (_float(row.get("offset")) or 0.0) > 0]
        result[method] = {
            "receivers": len({row["receiver"] for row in selected}),
            "same_next_count": sum(row.get("same_next") is True for row in selected),
            "next_decision_count": sum(row.get("same_next") is not None for row in selected),
            "same_consumed_count": sum(row.get("same_consumed") is True for row in generated),
            "consumed_decision_count": sum(row.get("same_consumed") is not None for row in generated),
            "comparison_rows": len(selected),
            "own_history_receivers": len({row["receiver"] for row in own}),
            "own_history_rows": len(own),
            "own_history_generated_rows": sum((_float(row.get("offset")) or 0.0) > 0 for row in own),
        }
    return {"methods": result, "raw_rows_retained": len(rows), "own_history_rows": sum(row.get("row_kind") == "own-history" for row in rows)}


def _aggregate_local(rows: list[dict[str, Any]]) -> dict[str, Any]:
    consumer = [row for row in rows if row.get("row_kind") == "consumer-boundary-error"]
    result: dict[str, Any] = {}
    for method in sorted({str(row.get("method")) for row in consumer}):
        result[method] = {}
        for boundary in PROFILE_BOUNDARIES:
            selected = [row for row in consumer if row.get("method") == method and row.get("boundary") == boundary]
            values = [value for row in selected if (value := _float((row.get("metrics") or {}).get("relative_l2"))) is not None]
            result[method][boundary] = {
                "relative_l2": _stats(values),
                "injected_exit_exact_count": sum(row.get("injected_exit_exact") is True for row in selected),
                "injected_current_k_exact_count": sum(row.get("injected_current_k_exact") is True for row in selected),
                "observation_count": len(selected),
            }
    return {
        "native_profile": _aggregate_profiles(rows),
        "consumer_boundaries": result,
        "all_local_rows_retained": len(rows),
    }


def _aggregate_costs(rows: list[dict[str, Any]], cost_scope: Any) -> dict[str, Any]:
    numeric_fields = (
        "source_seconds",
        "prepare_seconds",
        "two_compiles_seconds",
        "six_native_profile_spans_seconds",
        "native_profile_linears_per_span",
        "native_profile_spans",
        "extra_consumer_mlp_linears",
    )
    return {
        "receiver_count": len(rows),
        **{
            field: _stats([value for row in rows if (value := _float(row.get(field))) is not None])
            for field in numeric_fields
        },
        "cost_scope": cost_scope,
        "baseline_capture_timing": "not emitted separately; prefill counts are reported independently",
        "same_state_benchmark_scope": "one prepared state/direction; same-state dose amortization only",
    }


def _benchmark_rows(
    report: dict[str, Any], report_sha: str, case: dict[str, Any], phase: str, bundle: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = case.get("case", {})
    case_id = spec.get("id", "unknown")
    receiver = f"{case_id}:{phase}"
    costs = (bundle.get("forecast") or {}).get("costs") or {}
    benchmark = costs.get("same_state_benchmark") or {}
    compile_samples = [_float(value) for value in benchmark.get("compile_seconds", []) or []]
    compile_samples = [value for value in compile_samples if value is not None]
    compile_median = statistics.median(compile_samples) if compile_samples else None
    rows: list[dict[str, Any]] = []
    for item in benchmark.get("rows", []) or []:
        k = int(item.get("K", 0))
        direct = [_float(value) for value in item.get("direct_seconds", []) or []]
        scalar = [_float(value) for value in item.get("scalar_seconds", []) or []]
        direct = [value for value in direct if value is not None]
        scalar = [value for value in scalar if value is not None]
        paired_count = min(len(direct), len(scalar))
        paired_ratios = []
        if compile_median is not None:
            paired_ratios = [
                direct[index] / (compile_median + scalar[index])
                for index in range(paired_count)
                if compile_median + scalar[index] > 0
            ]
        rows.append({
            "schema": "mdb-read-weight-closure-benchmark-row-v1",
            "job_id": report.get("job_id"),
            "request_sha256": report.get("request_sha256"),
            "report_sha256": report_sha,
            "case": case_id,
            "split": spec.get("split", "unknown"),
            "phase": phase,
            "receiver": receiver,
            "K": k,
            "doses": _json_safe(item.get("doses", [])),
            "compile_seconds": compile_samples,
            "compile_median_seconds": compile_median,
            "direct_seconds": direct,
            "scalar_seconds": scalar,
            "direct_median_seconds": _float(item.get("direct_median_seconds")),
            "scalar_median_seconds": statistics.median(scalar) if scalar else None,
            "compile_plus_scalar_median_seconds": _float(item.get("compile_plus_scalar_median_seconds")),
            "direct_over_compile_plus_scalar": _float(item.get("direct_over_compile_plus_scalar")),
            "paired_direct_over_compile_plus_scalar_ratios": paired_ratios,
            "median_paired_direct_over_compile_plus_scalar": statistics.median(paired_ratios) if paired_ratios else None,
            "scope": benchmark.get("scope"),
            "native_model_calls": benchmark.get("native_model_calls"),
        })
    cost_row = {
        "receiver": receiver,
        "case": case_id,
        "split": spec.get("split", "unknown"),
        "phase": phase,
        "source_seconds": _float(costs.get("source_seconds")),
        "prepare_seconds": _float(costs.get("prepare_seconds")),
        "two_compiles_seconds": _float(costs.get("two_compiles_seconds")),
        "six_native_profile_spans_seconds": _float(costs.get("six_native_profile_spans_seconds")),
        "native_profile_linears_per_span": costs.get("native_profile_linears_per_span"),
        "native_profile_spans": costs.get("native_profile_spans"),
        "extra_consumer_mlp_linears": costs.get("extra_consumer_mlp_linears"),
        "same_state_benchmark": _json_safe(benchmark),
    }
    return rows, cost_row


def _closure_row(case: dict[str, Any], phase: str, bundle: dict[str, Any]) -> dict[str, Any]:
    spec = case.get("case", {})
    return {
        "receiver": f"{spec.get('id', 'unknown')}:{phase}",
        "case": spec.get("id"),
        "split": spec.get("split", "unknown"),
        "phase": phase,
        "functional_zero_closure_exact": _bool(bundle.get("functional_zero_closure_exact")),
        "functional_AB_native_closure_exact": _bool(bundle.get("functional_AB_native_closure_exact")),
        "functional_AB_full_logit_signature_exact": _bool(bundle.get("functional_AB_full_logit_signature_exact")),
    }


def _span_rows(report: dict[str, Any], report_sha: str, case: dict[str, Any], phase: str, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    spec = case.get("case", {})
    receiver = f"{spec.get('id', 'unknown')}:{phase}"
    rows = []
    arms = bundle.get("arms") or {}
    for arm_name in SPAN_ARMS:
        arm = arms.get(arm_name)
        if not isinstance(arm, dict):
            continue
        replacements = arm.get("span_replacements", []) or []
        rows.append({
            "schema": "mdb-read-weight-closure-span-row-v1",
            "job_id": report.get("job_id"),
            "request_sha256": report.get("request_sha256"),
            "report_sha256": report_sha,
            "case": spec.get("id"),
            "split": spec.get("split", "unknown"),
            "phase": phase,
            "receiver": receiver,
            "arm": arm_name,
            "replacement_count": len(replacements),
            "replacements": _json_safe(replacements),
            "native_layer_transitions": arm.get("native_layer_transitions"),
            "replaced_layer_transitions": arm.get("replaced_layer_transitions"),
            "end_cut": arm.get("end_cut"),
            "closure_sha256": arm.get("closure_sha256"),
        })
    return rows


def _closure_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "functional_zero_closure_exact",
        "functional_AB_native_closure_exact",
        "functional_AB_full_logit_signature_exact",
    )
    return {
        field: {
            "true_count": sum(row.get(field) is True for row in rows),
            "false_count": sum(row.get(field) is False for row in rows),
            "unknown_count": sum(row.get(field) is None for row in rows),
            "receiver_count": len(rows),
        }
        for field in fields
    }


def _benchmark_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for k in sorted({int(row["K"]) for row in rows}):
        selected = [row for row in rows if int(row["K"]) == k]
        metrics = (
            "compile_median_seconds",
            "direct_median_seconds",
            "scalar_median_seconds",
            "compile_plus_scalar_median_seconds",
            "direct_over_compile_plus_scalar",
            "median_paired_direct_over_compile_plus_scalar",
        )
        result[str(k)] = {
            "receiver_count": len({row["receiver"] for row in selected}),
            **{
                metric: _stats([value for row in selected if (value := _float(row.get(metric))) is not None])
                for metric in metrics
            },
            "scope": "same prepared state and direction; same-state dose amortization only",
        }
    return result


def summarize(run_dir: Path) -> dict[str, Any]:
    """Read one completed job and return raw-row-preserving scalar summaries."""

    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    report = read(report_path)
    report_sha = _sha256(report_path)
    forecast_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    own_history: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    receivers: list[dict[str, Any]] = []

    for case in report.get("cases", []) or []:
        spec = case.get("case", {})
        for phase in PHASES:
            bundle = case.get(phase)
            if not isinstance(bundle, dict):
                continue
            receiver = f"{spec.get('id', 'unknown')}:{phase}"
            current_forecasts, current_heads = _forecast_rows(report, report_sha, case, phase, bundle)
            diagnostic, residual_rows = _program_diagnostics(report, report_sha, case, phase, bundle)
            current_local = _local_rows(report, report_sha, case, phase, bundle)
            current_future, current_history = _future_rows(report, report_sha, case, phase, bundle)
            current_benchmarks, current_cost = _benchmark_rows(report, report_sha, case, phase, bundle)
            current_spans = _span_rows(report, report_sha, case, phase, bundle)
            current_heads.extend(residual_rows)
            forecast_rows.extend(current_forecasts)
            head_rows.extend(current_heads)
            local_rows.extend(current_local)
            future_rows.extend(current_future)
            own_history.extend(current_history)
            benchmark_rows.extend(current_benchmarks)
            cost_rows.append(current_cost)
            span_rows.extend(current_spans)
            closure_rows.append(_closure_row(case, phase, bundle))
            receivers.append({
                "receiver": receiver,
                "case": spec.get("id"),
                "split": spec.get("split", "unknown"),
                "phase": phase,
                "owner_relation": spec.get("previous_current_owner_relation"),
                "parent_cut": bundle.get("parent_cut"),
                "sites": _json_safe(bundle.get("sites")),
                "forecast_rows": current_forecasts,
                "midpoint_program_diagnostic": diagnostic,
                "local_rows": current_local,
                "future_rows": current_future,
                "own_candidate_history": current_history,
                "span_rows": current_spans,
                "closure_observations": _closure_row(case, phase, bundle),
                "forecast_ref": bundle.get("forecast_ref"),
                "forecast": _json_safe(bundle.get("forecast", {})),
                "mechanics_status": bundle.get("mechanics_status"),
                "terminal_status": bundle.get("terminal_status", "not-assessed"),
            })

    return {
        "schema": "mdb-read-weight-closure-summary-v1",
        "job_id": report.get("job_id"),
        "request_sha256": report.get("request_sha256"),
        "report_sha256": report_sha,
        "source_hashes": _json_safe(report.get("source_hashes")),
        "model": _json_safe(report.get("model")),
        "weights_sha256": report.get("weights_sha256"),
        "parameter_identity": _json_safe(report.get("parameter_identity")),
        "custody": _json_safe(report.get("artifact_custody")),
        "receiver_count": len(receivers),
        "receivers": receivers,
        "forecast_rows": forecast_rows,
        "head_rows": head_rows,
        "local_rows": local_rows,
        "future_rows": future_rows,
        "own_candidate_history": own_history,
        "benchmark_rows": benchmark_rows,
        "cost_rows": cost_rows,
        "closure_rows": closure_rows,
        "span_rows": span_rows,
        "forecast_aggregates": _aggregate_forecasts(forecast_rows),
        "head_direction_observations": _aggregate_heads(head_rows),
        "local_aggregates": _aggregate_local(local_rows),
        "future_aggregates": _aggregate_future(future_rows),
        "closure_observation_counts": _closure_counts(closure_rows),
        "span_replacement_summary": {
            "rows": len(span_rows),
            "replacement_record_count": sum(int(row.get("replacement_count", 0)) for row in span_rows),
            "by_arm": {
                arm: {
                    "receiver_count": len([row for row in span_rows if row.get("arm") == arm]),
                    "replacement_record_count": sum(
                        int(row.get("replacement_count", 0)) for row in span_rows if row.get("arm") == arm
                    ),
                }
                for arm in SPAN_ARMS
            },
            "report_replaced_layer_transitions": report.get("replaced_layer_transitions"),
            "report_native_layer_transitions": report.get("native_layer_transitions"),
        },
        "benchmark_aggregates": _benchmark_aggregates(benchmark_rows),
        "cost_aggregates": _aggregate_costs(cost_rows, report.get("cost_scope")),
        "cost_scope": report.get("cost_scope"),
        "consumer_scope": report.get("consumer_scope"),
        "prefill_calls": report.get("prefill_calls"),
        "prefill_token_count": report.get("prefill_token_count"),
        "physical_token_steps": report.get("physical_token_steps"),
        "native_layer_transitions": report.get("native_layer_transitions"),
        "replaced_layer_transitions": report.get("replaced_layer_transitions"),
        "mechanics_status": report.get("mechanics_status", "not-recorded-by-worker"),
        "instrument_status": report.get("instrument_status", "not-recorded-by-worker"),
        "trend_status": report.get("trend_status", "not-recorded-by-worker"),
        "terminal_status": report.get("terminal_status", "not-assessed"),
        "economical_full_query_qualified": report.get("economical_full_query_qualified", False),
        "semantic_cot_qualified": report.get("semantic_cot_qualified", False),
        "baseline_capture_timing": "not emitted separately; prefill counts and measured source/preparation/profile costs are retained",
        "interpretation": "all rows are observations; raw midpoint zero residual, native-profile fields, and heterogeneous futures remain visible",
    }


def _binding(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": row.get("job_id"),
        "request_sha256": row.get("request_sha256"),
        "report_sha256": row.get("report_sha256"),
        "case": row.get("case"),
        "split": row.get("split"),
        "phase": row.get("phase"),
        "receiver": row.get("receiver"),
        "parent_cut": row.get("parent_cut"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def write_evidence_projection(result: dict[str, Any], destination: Path) -> None:
    """Write four hash-bound scalar evidence streams: forecast, head, local, future."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "forecast.jsonl", result["forecast_rows"])
    _write_jsonl(destination / "head.jsonl", result["head_rows"])
    _write_jsonl(destination / "local.jsonl", result["local_rows"])
    _write_jsonl(destination / "future.jsonl", result["future_rows"])


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the aggregate report and per-receiver diagnostics without duplicate raw rows."""

    summary = {key: value for key, value in result.items() if key not in (
        "forecast_rows", "head_rows", "local_rows", "future_rows", "own_candidate_history",
        "benchmark_rows", "cost_rows", "span_rows",
    )}
    summary["receivers"] = [
        {key: value for key, value in receiver.items() if key not in (
            "forecast_rows", "local_rows", "future_rows", "own_candidate_history", "span_rows", "forecast",
        )}
        for receiver in result.get("receivers", [])
    ]
    for receiver in summary["receivers"]:
        diagnostic = receiver.get("midpoint_program_diagnostic")
        if isinstance(diagnostic, dict):
            diagnostic.pop("source_refs", None)
    return summary


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
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_output.write_text(
            json.dumps(compact_summary(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps({
        "job_id": result["job_id"],
        "receivers": result["receiver_count"],
        "forecast_rows": len(result["forecast_rows"]),
        "head_rows": len(result["head_rows"]),
        "local_rows": len(result["local_rows"]),
        "future_rows": len(result["future_rows"]),
        "closure_observation_counts": result["closure_observation_counts"],
        "trend_status": result["trend_status"],
        "terminal_status": result["terminal_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
