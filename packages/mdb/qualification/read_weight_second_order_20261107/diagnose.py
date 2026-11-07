"""Framework-free coefficient and head-level diagnostics for the Taylor assay.

The collector's report contains the per-dose native outcomes.  The small
Taylor program is normally stored as a CAS JSON artifact inside the result
archive, so this reader searches the report, result tree, and JSON members of
``result.tar.gz`` for ``mdb-smooth-qwen-taylor-program-v1``.  It never imports
Torch or executes a model.  Missing program artifacts are reported explicitly
so a partial custody tree remains useful evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PROGRAM_SCHEMA = "mdb-smooth-qwen-taylor-program-v1"
REPORT_SCHEMA = "mdb-read-weight-second-order-report-v1"
PROGRAM_FIELDS = (
    "logp0",
    "c1",
    "c2",
    "c1_score",
    "c2_route",
    "c2_upstream",
    "selected_rows",
    "query_heads",
    "kv_head",
    "source_row",
    "current_row",
    "scaling",
    "cost",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean(value: Any) -> Any:
    """Copy JSON values while dropping non-finite numeric leaves."""

    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _flatten_matrix(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for row in value:
        if isinstance(row, (list, tuple)):
            result.extend(item for item in row if _finite(item) is not None)
        elif _finite(row) is not None:
            result.append(row)
    return [float(item) for item in result]


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "mean_abs": None, "rms": None, "positive": 0, "negative": 0, "zero": 0}
    absolute = [abs(value) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "mean_abs": statistics.fmean(absolute),
        "rms": math.sqrt(statistics.fmean([value * value for value in values])),
        "positive": sum(value > 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
    }


def _matrix_shape(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        return None
    width = len(value[0])
    if width == 0 or any(len(row) != width for row in value):
        return None
    return len(value), width


def _matrix_difference(left: Any, right: Any) -> list[float]:
    if _matrix_shape(left) != _matrix_shape(right):
        return []
    return [float(a) - float(b) for a_row, b_row in zip(left, right)
            for a, b in zip(a_row, b_row)
            if _finite(a) is not None and _finite(b) is not None]


def _program_record(program: dict[str, Any], *, source: str, context: dict[str, Any]) -> dict[str, Any]:
    record = {
        "schema": PROGRAM_SCHEMA,
        "source": source,
        **{key: _clean(program[key]) for key in PROGRAM_FIELDS if key in program},
    }
    c1 = _flatten_matrix(program.get("c1"))
    c2 = _flatten_matrix(program.get("c2"))
    c2_route = _flatten_matrix(program.get("c2_route"))
    c2_upstream = _flatten_matrix(program.get("c2_upstream"))
    record["context"] = context
    record["program_float32_values"] = 3 * len(c1)
    record["program_float32_equivalent_bytes"] = 4 * 3 * len(c1)
    record["coefficient_stats"] = {
        "c1": _stats(c1),
        "c2": _stats(c2),
        "c2_route": _stats(c2_route),
        "c2_upstream": _stats(c2_upstream),
    }
    c1_delta = _matrix_difference(program.get("c1"), program.get("c1_score"))
    record["c1_agreement"] = {
        "shape": _matrix_shape(program.get("c1")),
        "score_shape": _matrix_shape(program.get("c1_score")),
        "difference": _stats(c1_delta),
        "exact": bool(c1_delta) and max(abs(value) for value in c1_delta) == 0.0,
    }
    if c2 and c2_route and len(c2) == len(c2_route):
        upstream = _matrix_difference(program.get("c2"), program.get("c2_route"))
        record["c2_upstream_check"] = {
            "shape": _matrix_shape(program.get("c2")),
            "route_shape": _matrix_shape(program.get("c2_route")),
            "difference": _stats(upstream),
            "full_abs": _stats(c2),
            "route_abs": _stats(c2_route),
        }
    else:
        record["c2_upstream_check"] = {
            "shape": _matrix_shape(program.get("c2")),
            "route_shape": _matrix_shape(program.get("c2_route")),
            "difference": _stats([]),
            "full_abs": _stats(c2),
            "route_abs": _stats(c2_route),
        }
    return record


def _walk(value: Any, *, source: str, context: dict[str, Any], found: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        local = dict(context)
        case = value.get("case")
        if isinstance(case, dict) and case.get("id") is not None:
            local["case"] = case["id"]
        if isinstance(case, str):
            local["case"] = case
        if value.get("phase") is not None:
            local["phase"] = value["phase"]
        if value.get("schema") == PROGRAM_SCHEMA:
            found.append(_program_record(value, source=source, context=local))
        for key, item in value.items():
            child = dict(local)
            if key in {"first", "second_candidate_world"}:
                child["phase"] = key
            if key == "case" and isinstance(item, dict) and item.get("id") is not None:
                child["case"] = item["id"]
            _walk(item, source=source, context=child, found=found)
    elif isinstance(value, list):
        for item in value:
            _walk(item, source=source, context=context, found=found)


def _scan_tree(run_dir: Path, found: list[dict[str, Any]]) -> None:
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in {"report.json", "summary.json", "compact-summary.json", "diagnose.json"}:
            continue
        try:
            value = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        _walk(value, source=str(path), context={}, found=found)


def _scan_archive(path: Path, found: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    try:
        archive = tarfile.open(path, "r:*")
    except (OSError, tarfile.TarError):
        return
    with archive:
        for member in archive:
            if not member.isfile() or member.size > 4 * 1024 * 1024:
                continue
            if not (member.name.endswith(".json") or "taylor" in member.name.lower()):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                value = json.loads(handle.read())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            _walk(value, source=f"{path}:{member.name}", context={}, found=found)


def _dedupe_programs(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in found:
        payload = {key: item[key] for key in PROGRAM_FIELDS if key in item}
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        current = result.get(identity)
        if current is None:
            current = dict(item)
            current["sources"] = [item["source"]]
            current["contexts"] = []
            if item.get("context"):
                current["contexts"].append(item["context"])
            current["record_sha256"] = identity
            result[identity] = current
            continue
        if item["source"] not in current["sources"]:
            current["sources"].append(item["source"])
        context = item.get("context")
        if context and context not in current["contexts"]:
            current["contexts"].append(context)
        if not current.get("context") and context:
            current["context"] = context
    return list(result.values())


def _dose_outcomes(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in report.get("cases", []):
        spec = case.get("case", {})
        case_id = spec.get("id")
        for phase in ("first", "second_candidate_world"):
            bundle = case.get(phase)
            if not isinstance(bundle, dict):
                continue
            receiver = f"{case_id}:{phase}"
            for dose_row in bundle.get("dose_rows", []) or []:
                dose = _finite(dose_row.get("dose"))
                for method, metrics in (dose_row.get("methods") or {}).items():
                    if not isinstance(metrics, dict):
                        continue
                    head_rows = []
                    for head in metrics.get("head_rows", []) or []:
                        head_rows.append({
                            "head_offset": head.get("head_offset"),
                            "true_delta_logp": _finite(head.get("true_delta_logp")),
                            "predicted_delta_logp": _finite(head.get("predicted_delta_logp")),
                            "error": _finite(head.get("error")),
                        })
                    rows.append({
                        "case": case_id,
                        "phase": phase,
                        "receiver": receiver,
                        "dose": dose,
                        "method": method,
                        "log_change_rmse": _finite(metrics.get("log_change_rmse")),
                        "log_change_max_abs": _finite(metrics.get("log_change_max_abs")),
                        "true_log_change_rms": _finite(metrics.get("true_log_change_rms")),
                        "probability_error": _clean(metrics.get("probability_error")),
                        "predicted_probability_in_range": metrics.get("predicted_probability_in_range"),
                        "head_rows": head_rows,
                    })
    return rows


def _outcome_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("log_change_rmse")
        if value is not None and row.get("dose") is not None:
            grouped[(str(row["method"]), float(row["dose"]))].append(float(value))
    result = {}
    for (method, dose), values in sorted(grouped.items()):
        result.setdefault(method, {})[str(dose)] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def _head_counterexamples(rows: list[dict[str, Any]], *, limit: int = 24) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        for head in row.get("head_rows", []):
            error = head.get("error")
            if error is None:
                continue
            candidates.append({
                "case": row["case"],
                "phase": row["phase"],
                "receiver": row["receiver"],
                "dose": row["dose"],
                "method": row["method"],
                **head,
                "abs_error": abs(float(error)),
            })
    candidates.sort(key=lambda item: (-item["abs_error"], item["receiver"], str(item["dose"]), item["method"]))
    return candidates[:limit]


def diagnose(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    report = _read_json(report_path)
    if report.get("schema") not in {None, REPORT_SCHEMA}:
        raise ValueError(f"unexpected report schema: {report.get('schema')!r}")
    report_sha = _sha256(report_path)
    found: list[dict[str, Any]] = []
    _walk(report, source=str(report_path), context={}, found=found)
    _scan_tree(run_dir, found)
    _scan_archive(run_dir / "result.tar.gz", found)
    programs = _dedupe_programs(found)
    outcomes = _dose_outcomes(report)
    full_dose = [row for row in outcomes if row.get("dose") == 1.0]
    return {
        "schema": "mdb-read-weight-second-order-diagnostics-v1",
        "job_id": report.get("job_id"),
        "report_sha256": report_sha,
        "program_count": len(programs),
        "programs": programs,
        "program_float32_values_total": sum(item["program_float32_values"] for item in programs),
        "program_float32_equivalent_bytes_total": sum(item["program_float32_equivalent_bytes"] for item in programs),
        "per_dose_outcomes": outcomes,
        "full_dose_outcomes": full_dose,
        "outcome_aggregates": _outcome_aggregates(outcomes),
        "head_counterexamples": _head_counterexamples(outcomes),
        "program_artifact_status": "found" if programs else "not-found-in-report-or-json-custody",
        "terminal_status": "not-assessed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "job_id": result["job_id"],
        "program_count": result["program_count"],
        "outcome_rows": len(result["per_dose_outcomes"]),
        "full_dose_rows": len(result["full_dose_outcomes"]),
        "program_artifact_status": result["program_artifact_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
