"""Hyperion: compile dynamic causal panels into circuit symbol-table entries.

Hyperion is artifact-only. It does not load a model or manufacture causal evidence.
It normalizes already measured same-parent intervention rows, fits a residual kernel
over an ordered execution coordinate on discovery arms, evaluates held-out schedules,
classifies the observed consumer-response shape, and emits a fingerprinted,
context-qualified symbol.

The native consumer remains authoritative. A Hyperion symbol is a compiled evidence
object, not a claim that a physical alias has universal semantic meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

HYPERION_PANEL_SCHEMA = "saturn-hyperion-panel-v1"
HYPERION_REPORT_SCHEMA = "saturn-hyperion-report-v1"
HYPERION_SYMBOL_SCHEMA = "saturn-hyperion-dynamic-circuit-symbol-v1"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")


class HyperionError(ValueError):
    """Raised when Hyperion receives unsafe or insufficient circuit evidence."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HyperionError("Hyperion artifacts cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise HyperionError(f"value is not JSON-safe: {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result or not _SAFE_ID.fullmatch(result):
        raise HyperionError(f"{label} must be a non-empty safe identifier")
    return result


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise HyperionError(f"{label} must be non-empty")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise HyperionError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HyperionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise HyperionError(f"{label} must be finite")
    return result


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HyperionError(f"{label} must be an object")
    result = _jsonable(value)
    if not isinstance(result, dict):
        raise HyperionError(f"{label} must be an object")
    return result


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HyperionError(f"{label} must be an array")
    return list(value)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, places: int = 6) -> float | None:
    return None if value is None else round(float(value), places)


@dataclass(frozen=True, slots=True)
class HyperionObservation:
    """One native-consumer response under a declared residual schedule."""

    observation_id: str
    cell_id: str
    arm_id: str
    phase: Literal["discovery", "heldout", "control"]
    arm_kind: str
    sites: tuple[str, ...]
    schedule: tuple[tuple[int, float], ...]
    target_effect: float
    consumer_id: str
    evidence_ref: str
    collateral_effect: float | None = None
    repair_effect: float | None = None
    continuation_effect: float | None = None
    endpoint_equivalence_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("observation_id", "cell_id", "arm_id", "arm_kind", "consumer_id"):
            object.__setattr__(self, name, _safe_id(getattr(self, name), name))
        if self.phase not in {"discovery", "heldout", "control"}:
            raise HyperionError("phase must be discovery, heldout, or control")
        sites = tuple(_safe_id(site, "site") for site in self.sites)
        if not sites:
            raise HyperionError("an observation requires at least one site")
        if len(sites) != len(set(sites)):
            raise HyperionError("observation sites cannot contain duplicates")
        object.__setattr__(self, "sites", sites)
        normalized_schedule: list[tuple[int, float]] = []
        seen_times: set[int] = set()
        for raw_time, raw_weight in self.schedule:
            if isinstance(raw_time, bool):
                raise HyperionError("schedule times must be integers")
            try:
                time_index = int(raw_time)
            except (TypeError, ValueError) as exc:
                raise HyperionError("schedule times must be integers") from exc
            if time_index < 0 or time_index in seen_times:
                raise HyperionError("schedule times must be unique non-negative integers")
            seen_times.add(time_index)
            weight = _finite(raw_weight, "schedule weight")
            if weight == 0:
                continue
            normalized_schedule.append((time_index, weight))
        normalized_schedule.sort()
        object.__setattr__(self, "schedule", tuple(normalized_schedule))
        object.__setattr__(self, "target_effect", _finite(self.target_effect, "target_effect"))
        for name in ("collateral_effect", "repair_effect", "continuation_effect"):
            object.__setattr__(
                self,
                name,
                _optional_finite(getattr(self, name), name),
            )
        object.__setattr__(self, "evidence_ref", _text(self.evidence_ref, "evidence_ref"))
        if self.endpoint_equivalence_id is not None:
            object.__setattr__(
                self,
                "endpoint_equivalence_id",
                _safe_id(self.endpoint_equivalence_id, "endpoint_equivalence_id"),
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def schedule_map(self) -> dict[int, float]:
        return dict(self.schedule)

    @property
    def nominal_exposure(self) -> float:
        return sum(abs(weight) for _, weight in self.schedule)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "cell_id": self.cell_id,
            "arm_id": self.arm_id,
            "phase": self.phase,
            "arm_kind": self.arm_kind,
            "sites": list(self.sites),
            "schedule": {str(time): weight for time, weight in self.schedule},
            "nominal_exposure": self.nominal_exposure,
            "target_effect": self.target_effect,
            "collateral_effect": self.collateral_effect,
            "repair_effect": self.repair_effect,
            "continuation_effect": self.continuation_effect,
            "consumer_id": self.consumer_id,
            "endpoint_equivalence_id": self.endpoint_equivalence_id,
            "evidence_ref": self.evidence_ref,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HyperionObservation:
        raw_schedule = value.get("schedule") or {}
        if not isinstance(raw_schedule, Mapping):
            raise HyperionError("observation schedule must be an object")
        return cls(
            observation_id=str(value.get("observation_id") or ""),
            cell_id=str(value.get("cell_id") or ""),
            arm_id=str(value.get("arm_id") or ""),
            phase=str(value.get("phase") or ""),  # type: ignore[arg-type]
            arm_kind=str(value.get("arm_kind") or "candidate"),
            sites=tuple(str(site) for site in value.get("sites") or ()),
            schedule=tuple((int(time), weight) for time, weight in raw_schedule.items()),
            target_effect=value.get("target_effect"),
            collateral_effect=value.get("collateral_effect"),
            repair_effect=value.get("repair_effect"),
            continuation_effect=value.get("continuation_effect"),
            consumer_id=str(value.get("consumer_id") or ""),
            endpoint_equivalence_id=value.get("endpoint_equivalence_id"),
            evidence_ref=str(value.get("evidence_ref") or ""),
            metadata=value.get("metadata") or {},
        )


def _solve_ridge(rows: Sequence[Sequence[float]], values: Sequence[float]) -> list[float]:
    """Solve a small ridge-regularized least-squares system without heavy imports."""

    if not rows or len(rows) != len(values):
        raise HyperionError("kernel fit requires aligned non-empty rows and values")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise HyperionError("kernel fit rows must have a stable non-zero width")
    matrix = [[0.0 for _ in range(width + 1)] for _ in range(width)]
    for left in range(width):
        for right in range(width):
            matrix[left][right] = sum(row[left] * row[right] for row in rows)
        matrix[left][left] += 1e-10
        matrix[left][width] = sum(
            row[left] * value for row, value in zip(rows, values, strict=True)
        )
    for pivot in range(width):
        best = max(range(pivot, width), key=lambda row: abs(matrix[row][pivot]))
        if abs(matrix[best][pivot]) < 1e-12:
            raise HyperionError("kernel design matrix is rank deficient")
        matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
        scale = matrix[pivot][pivot]
        matrix[pivot] = [value / scale for value in matrix[pivot]]
        for row_index in range(width):
            if row_index == pivot:
                continue
            factor = matrix[row_index][pivot]
            matrix[row_index] = [
                current - factor * reference
                for current, reference in zip(matrix[row_index], matrix[pivot], strict=True)
            ]
    return [matrix[index][width] for index in range(width)]


def _predict(weights: Sequence[float], observation: HyperionObservation) -> float:
    return sum(weights[time] * value for time, value in observation.schedule)


def _kernel_cells(
    observations: Sequence[HyperionObservation],
    time_indices: Sequence[int],
) -> dict[str, dict[str, Any]]:
    if list(time_indices) != list(range(len(time_indices))):
        raise HyperionError("Hyperion v1 requires a dense zero-based time lattice")
    by_cell: dict[str, list[HyperionObservation]] = defaultdict(list)
    for observation in observations:
        by_cell[observation.cell_id].append(observation)
    cells: dict[str, dict[str, Any]] = {}
    for cell_id, rows in sorted(by_cell.items()):
        discovery = [
            row for row in rows if row.phase == "discovery" and row.arm_kind == "candidate"
        ]
        if not discovery:
            continue
        design = [[row.schedule_map.get(time, 0.0) for time in time_indices] for row in discovery]
        weights = _solve_ridge(design, [row.target_effect for row in discovery])
        discovery_errors = [_predict(weights, row) - row.target_effect for row in discovery]
        heldout = [row for row in rows if row.phase == "heldout" and row.arm_kind == "candidate"]
        heldout_rows = [
            {
                "arm_id": row.arm_id,
                "observed": _round(row.target_effect),
                "predicted": _round(_predict(weights, row)),
                "error": _round(_predict(weights, row) - row.target_effect),
                "endpoint_equivalence_id": row.endpoint_equivalence_id,
            }
            for row in heldout
        ]
        heldout_errors = [_predict(weights, row) - row.target_effect for row in heldout]
        cells[cell_id] = {
            "weights": [_round(weight) for weight in weights],
            "discovery_rows": len(discovery),
            "discovery_mse": _round(
                sum(error * error for error in discovery_errors) / len(discovery_errors)
            ),
            "discovery_residual_l2": _round(
                math.sqrt(sum(error * error for error in discovery_errors))
            ),
            "heldout_rows": heldout_rows,
            "heldout_mse": _round(
                sum(error * error for error in heldout_errors) / len(heldout_errors)
            )
            if heldout_errors
            else None,
        }
    if not cells:
        raise HyperionError("no discovery cells were available for a kernel fit")
    return cells


def _arm_effect(
    observations: Sequence[HyperionObservation], cell_id: str, arm_id: str
) -> float | None:
    matches = [
        row.target_effect for row in observations if row.cell_id == cell_id and row.arm_id == arm_id
    ]
    return matches[0] if len(matches) == 1 else None


def _final_dose_effect(
    observations: Sequence[HyperionObservation], cell_id: str, final_time: int
) -> float | None:
    """Return the unit dose applied at the final coordinate alone, when unique."""
    matches = [
        row.target_effect
        for row in observations
        if row.cell_id == cell_id
        and row.phase == "discovery"
        and len(row.schedule) == 1
        and row.schedule[0][0] == final_time
        and abs(row.schedule[0][1]) == 1.0
    ]
    return matches[0] if len(matches) == 1 else None


def _classify(
    observations: Sequence[HyperionObservation],
    cells: Mapping[str, Mapping[str, Any]],
    time_indices: Sequence[int],
    *,
    coordinate_kind: Literal["temporal", "depth"],
) -> dict[str, Any]:
    discovery = [row for row in observations if row.phase == "discovery"]
    cell_ids = sorted(cells)
    full_effects = [
        effect
        for cell_id in cell_ids
        if (effect := _arm_effect(discovery, cell_id, "all")) is not None
    ]
    singleton_effects = [
        abs(row.target_effect)
        for row in discovery
        if len(row.schedule) == 1 and abs(row.schedule[0][1]) == 1.0
    ]
    weights_by_cell = {
        cell_id: [float(value) for value in cells[cell_id].get("weights") or ()]
        for cell_id in cell_ids
    }
    split = max(1, len(time_indices) // 2)
    front_loaded_cells = [
        cell_id
        for cell_id, weights in weights_by_cell.items()
        if sum(weights[:split]) > sum(weights[split:])
    ]
    back_loaded_cells = [
        cell_id
        for cell_id, weights in weights_by_cell.items()
        if sum(weights[:split]) < sum(weights[split:])
    ]
    mean_weights = [
        sum(weights[index] for weights in weights_by_cell.values()) / len(weights_by_cell)
        for index in time_indices
    ]
    heldout_mses = [
        float(cell["heldout_mse"]) for cell in cells.values() if cell.get("heldout_mse") is not None
    ]
    heldout_mse_mean = _mean(heldout_mses)

    equal_exposure: list[dict[str, Any]] = []
    synergies: list[dict[str, Any]] = []
    path_pairs: list[dict[str, Any]] = []
    half_response_comparisons: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        all_effect = _arm_effect(discovery, cell_id, "all")
        late_double = _arm_effect(discovery, cell_id, "s23_double")
        if all_effect is not None and late_double is not None:
            equal_exposure.append(
                {
                    "cell_id": cell_id,
                    "all": _round(all_effect),
                    "late_double": _round(late_double),
                    "difference": _round(all_effect - late_double),
                }
            )
        s0 = _arm_effect(discovery, cell_id, "s0")
        s1 = _arm_effect(discovery, cell_id, "s1")
        s01 = _arm_effect(discovery, cell_id, "s01")
        if s0 is not None and s1 is not None and s01 is not None:
            synergies.append(
                {
                    "cell_id": cell_id,
                    "observed_pair": _round(s01),
                    "singleton_sum": _round(s0 + s1),
                    "interaction_residual": _round(s01 - s0 - s1),
                }
            )
        early_half = _arm_effect(observations, cell_id, "first_half")
        late_half = _arm_effect(observations, cell_id, "second_half")
        if early_half is None or late_half is None:
            early_half = _arm_effect(observations, cell_id, "s01")
            late_half = _arm_effect(observations, cell_id, "s23")
        if early_half is not None and late_half is not None:
            half_response_comparisons.append(
                {
                    "cell_id": cell_id,
                    "early_half": _round(early_half),
                    "late_half": _round(late_half),
                    "early_minus_late": _round(early_half - late_half),
                }
            )
        heldout = [
            row
            for row in observations
            if row.cell_id == cell_id
            and row.phase == "heldout"
            and row.endpoint_equivalence_id == "net-zero"
        ]
        by_arm = {row.arm_id: row for row in heldout}
        if "pos01_neg23" in by_arm and "neg01_pos23" in by_arm:
            positive = by_arm["pos01_neg23"].target_effect
            negative = by_arm["neg01_pos23"].target_effect
            path_pairs.append(
                {
                    "cell_id": cell_id,
                    "endpoint_equivalence_id": "net-zero",
                    "early_positive": _round(positive),
                    "early_negative": _round(negative),
                    "order_gap": _round(positive - negative),
                }
            )

    full_min = min(full_effects) if full_effects else None
    singleton_max = max(singleton_effects) if singleton_effects else None
    coordinate_coalition = bool(
        full_min is not None
        and singleton_max is not None
        and full_min >= 0.5
        and singleton_max < 0.5
    )
    final_time = max(time_indices)
    final_dose_comparisons: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        all_effect = _arm_effect(discovery, cell_id, "all")
        final_effect = _final_dose_effect(discovery, cell_id, final_time)
        if all_effect is None or final_effect is None or all_effect == 0.0:
            continue
        final_dose_comparisons.append(
            {
                "cell_id": cell_id,
                "all": _round(all_effect),
                "final_dose": _round(final_effect),
                "share": _round(abs(final_effect) / abs(all_effect)),
            }
        )
    final_dose_dominant = bool(
        final_dose_comparisons
        and all(float(row["share"]) >= 0.5 for row in final_dose_comparisons)
    )
    front_loaded = len(front_loaded_cells) == len(cells)
    back_loaded = len(back_loaded_cells) == len(cells)
    nonexchangeable = bool(
        equal_exposure and all(float(row["difference"]) >= 0.20 for row in equal_exposure)
    )
    path_order_sensitive = bool(
        len(path_pairs) >= 2 and sum(float(row["order_gap"]) >= 0.25 for row in path_pairs) >= 2
    )
    interaction_residual = bool(
        synergies
        and sum(abs(float(row["interaction_residual"])) >= 0.05 for row in synergies)
        >= max(1, len(synergies) // 2)
    )
    weighted_predictive = bool(heldout_mse_mean is not None and heldout_mse_mean <= 0.05)
    early_half_dominant = bool(
        half_response_comparisons
        and all(float(row["early_minus_late"]) > 0.0 for row in half_response_comparisons)
    )
    late_half_dominant = bool(
        half_response_comparisons
        and all(float(row["early_minus_late"]) < 0.0 for row in half_response_comparisons)
    )
    if coordinate_coalition and final_dose_dominant:
        # A final-coordinate-only dose that reproduces most of the full-path
        # effect leaves ordered integration unproven: the simplest rival
        # explains the panel (Mimas leave-one-arm-out, five of six depth
        # panels). Retain the coalition as an observation, not an integral.
        primary = f"final-dose-dominant-{coordinate_kind}-coalition"
    elif coordinate_coalition and (front_loaded or back_loaded) and weighted_predictive:
        primary = f"weighted-{coordinate_kind}-integral"
    elif coordinate_coalition:
        primary = f"{coordinate_kind}-coalition"
    elif full_min is not None and full_min >= 0.5:
        primary = f"{coordinate_kind}-addressed-carrier"
    else:
        primary = "insufficient-dynamic-evidence"
    coalition_property = (
        "distributed-temporal-coalition"
        if coordinate_kind == "temporal"
        else "distributed-depth-coalition"
    )
    nonexchangeable_property = (
        "dose-nonexchangeable-across-time"
        if coordinate_kind == "temporal"
        else "dose-nonexchangeable-across-depth"
    )
    properties = [
        name
        for name, present in (
            (coalition_property, coordinate_coalition),
            ("front-loaded-kernel", front_loaded and weighted_predictive),
            ("back-loaded-kernel", back_loaded and weighted_predictive),
            ("early-half-dominant-response", early_half_dominant),
            ("late-half-dominant-response", late_half_dominant),
            ("heldout-weighted-kernel-predictive", weighted_predictive),
            ("final-dose-dominant-response", final_dose_dominant),
            (nonexchangeable_property, nonexchangeable),
            ("path-order-sensitive", path_order_sensitive),
            ("nonlinear-interaction-residual", interaction_residual),
        )
        if present
    ]
    return {
        "primary": primary,
        "coordinate_kind": coordinate_kind,
        "properties": properties,
        "thresholds": {
            "strong_full_effect": 0.5,
            "subthreshold_singleton": 0.5,
            "equal_exposure_gap": 0.20,
            "path_order_gap": 0.25,
            "interaction_residual": 0.05,
            "heldout_weighted_kernel_mse": 0.05,
            "final_dose_share": 0.5,
        },
        "observed": {
            "full_effect_min": _round(full_min),
            "singleton_effect_max": _round(singleton_max),
            "mean_kernel_weights": [_round(value) for value in mean_weights],
            "heldout_weighted_kernel_mse_mean": _round(heldout_mse_mean),
            "front_loaded_cells": front_loaded_cells,
            "back_loaded_cells": back_loaded_cells,
            "equal_exposure_comparisons": equal_exposure,
            "path_order_pairs": path_pairs,
            "interaction_residuals": synergies,
            "half_response_comparisons": half_response_comparisons,
            "final_dose_comparisons": final_dose_comparisons,
        },
    }


def _source_record(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    execution = payload.get("execution_receipt")
    execution = execution if isinstance(execution, Mapping) else {}
    return {
        "path": str(path),
        "sha256": _file_digest(path),
        "schema": str(payload.get("schema") or "unknown"),
        "mrun_job_id": execution.get("mrun_job_id"),
        "mrun_ok": execution.get("mrun_ok"),
        "mrun_state": execution.get("mrun_state"),
        "report_present": execution.get("report_present"),
    }


def _require_successful_receipt(payload: Mapping[str, Any], label: str) -> None:
    execution = payload.get("execution_receipt")
    if not isinstance(execution, Mapping):
        raise HyperionError(f"{label} is missing its execution receipt")
    if execution.get("mrun_ok") is not True or execution.get("report_present") is not True:
        raise HyperionError(f"{label} did not finish as a successful collected mrun report")


def adapt_ird_kernel_report(
    payload: Mapping[str, Any],
    *,
    path: Path,
    consumer_id: str,
) -> tuple[list[HyperionObservation], dict[str, Any]]:
    """Adapt the successful FLUX IRD write-presence report into Hyperion rows."""

    if payload.get("schema") != "saturn-ird-p4-kernel-v1":
        raise HyperionError("unsupported IRD kernel report schema")
    _require_successful_receipt(payload, "IRD kernel report")
    kernel = payload.get("kernel")
    if not isinstance(kernel, Mapping):
        raise HyperionError("IRD kernel report is missing kernel data")
    branches = _sequence(kernel.get("branches"), "kernel.branches")
    observations: list[HyperionObservation] = []
    for raw in branches:
        if not isinstance(raw, Mapping):
            raise HyperionError("kernel branch rows must be objects")
        target = _safe_id(raw.get("target"), "kernel target")
        seed = int(raw.get("seed"))
        arm = _safe_id(raw.get("arm"), "kernel arm")
        alpha = _finite(raw.get("alpha", 1.0), "kernel alpha")
        schedule = tuple((int(step), alpha) for step in raw.get("sched") or ())
        cell_id = f"{target}-{seed}"
        observations.append(
            HyperionObservation(
                observation_id=f"kernel:{cell_id}:{arm}",
                cell_id=cell_id,
                arm_id=arm,
                phase="discovery",
                arm_kind="candidate",
                sites=("conditioner.subject-row",),
                schedule=schedule,
                target_effect=raw.get("progress"),
                consumer_id=consumer_id,
                evidence_ref=f"{path}#/kernel/branches/{len(observations)}",
                metadata={"target": target, "seed": seed, "calls": raw.get("calls")},
            )
        )
    gates = kernel.get("gates")
    gates = gates if isinstance(gates, Mapping) else {}
    target_seed_pairs = sorted(
        {(str(row.metadata.get("target")), int(row.metadata.get("seed"))) for row in observations}
    )
    for target, seed in target_seed_pairs:
        noop = gates.get(f"noop_{seed}")
        if noop is None:
            continue
        observations.append(
            HyperionObservation(
                observation_id=f"kernel:{target}-{seed}:noop",
                cell_id=f"{target}-{seed}",
                arm_id="noop",
                phase="control",
                arm_kind="noop",
                sites=("conditioner.subject-row",),
                schedule=(),
                target_effect=noop,
                consumer_id=consumer_id,
                evidence_ref=f"{path}#/kernel/gates/noop_{seed}",
                metadata={"target": target, "seed": seed},
            )
        )
    return observations, _source_record(path, payload)


def adapt_ird_path_report(
    payload: Mapping[str, Any],
    *,
    path: Path,
    consumer_id: str,
) -> tuple[list[HyperionObservation], dict[str, Any]]:
    """Adapt the held-out IRD order panel without leaking it into the kernel fit."""

    if payload.get("schema") != "saturn-ird-allmodels-p1-v1":
        raise HyperionError("unsupported IRD path report schema")
    _require_successful_receipt(payload, "IRD path report")
    p1 = payload.get("p1")
    if not isinstance(p1, Mapping):
        raise HyperionError("IRD path report is missing p1 data")
    schedules: dict[str, tuple[tuple[int, float], ...]] = {
        "s01": ((0, 1.0), (1, 1.0)),
        "all": ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)),
        "pos01_neg23": ((0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0)),
        "neg01_pos23": ((0, -1.0), (1, -1.0), (2, 1.0), (3, 1.0)),
    }
    observations: list[HyperionObservation] = []
    for index, raw in enumerate(_sequence(p1.get("branches"), "p1.branches")):
        if not isinstance(raw, Mapping):
            raise HyperionError("p1 branch rows must be objects")
        target = _safe_id(raw.get("target"), "p1 target")
        seed = int(raw.get("seed"))
        arm = _safe_id(raw.get("arm"), "p1 arm")
        if arm not in schedules:
            raise HyperionError(f"unsupported p1 arm: {arm}")
        observations.append(
            HyperionObservation(
                observation_id=f"p1:{target}-{seed}:{arm}",
                cell_id=f"{target}-{seed}",
                arm_id=arm,
                phase="heldout",
                arm_kind="candidate"
                if arm in {"pos01_neg23", "neg01_pos23"}
                else "heldout-reference",
                sites=("conditioner.subject-row",),
                schedule=schedules[arm],
                target_effect=raw.get("progress"),
                consumer_id=consumer_id,
                endpoint_equivalence_id="net-zero"
                if arm in {"pos01_neg23", "neg01_pos23"}
                else None,
                evidence_ref=f"{path}#/p1/branches/{index}",
                metadata={"target": target, "seed": seed, "calls": raw.get("calls")},
            )
        )
    return observations, _source_record(path, payload)


def adapt_ird_depth_report(
    payload: Mapping[str, Any],
    *,
    path: Path,
    model_key: str,
    behavior: str,
    consumer_id: str,
) -> tuple[list[HyperionObservation], dict[str, Any]]:
    """Adapt an IRD autoregressive residual-deletion depth panel.

    Effects are normalized by the all-layer target-logprob loss so that the
    generic Hyperion thresholds retain their dimensionless meaning. Raw logprob
    effects remain in observation metadata.
    """

    if payload.get("schema") != "saturn-ird-allmodels-p1-v1":
        raise HyperionError("unsupported IRD depth report schema")
    _require_successful_receipt(payload, "IRD depth report")
    model_key = _safe_id(model_key, "model_key")
    behavior = _safe_id(behavior, "behavior")
    grid = payload.get("grid")
    if not isinstance(grid, Mapping) or not isinstance(grid.get(model_key), Mapping):
        raise HyperionError(f"IRD depth report has no model cell for {model_key}")
    model = grid[model_key]
    cells = model.get("cells")
    if not isinstance(cells, Mapping) or not isinstance(cells.get(behavior), Mapping):
        raise HyperionError(f"IRD depth report has no {model_key}/{behavior} behavior cell")
    cell = cells[behavior]
    n_layers = int(cell.get("n_layers") or 0)
    if n_layers < 2:
        raise HyperionError("IRD depth cell requires at least two layers")
    necessity = cell.get("necessity")
    if not isinstance(necessity, Mapping):
        raise HyperionError("IRD depth cell is missing necessity arms")
    required = {"all", "first_half", "second_half"}
    if not required.issubset(necessity):
        raise HyperionError("IRD depth cell is missing all/first_half/second_half arms")

    def raw_effect(arm: str) -> float:
        row = necessity.get(arm)
        if not isinstance(row, Mapping):
            raise HyperionError(f"IRD depth arm {arm} must be an object")
        return -_finite(row.get("delta_lp_base"), f"IRD depth {arm} delta_lp_base")

    full_effect = raw_effect("all")
    if full_effect <= 0.0:
        raise HyperionError("IRD depth normalization requires a positive all-layer effect")
    split = n_layers // 2
    cell_id = f"{model_key}-{behavior}"
    observations: list[HyperionObservation] = []
    for index, (arm, raw) in enumerate(necessity.items()):
        if not isinstance(raw, Mapping):
            raise HyperionError(f"IRD depth arm {arm} must be an object")
        arm = _safe_id(arm, "IRD depth arm")
        if arm == "all":
            schedule = tuple((layer, 1.0) for layer in range(n_layers))
            phase: Literal["discovery", "heldout", "control"] = "discovery"
        elif arm == "first_half":
            schedule = tuple((layer, 1.0) for layer in range(split))
            phase = "heldout"
        elif arm == "second_half":
            schedule = tuple((layer, 1.0) for layer in range(split, n_layers))
            phase = "heldout"
        elif re.fullmatch(r"L\d+", arm):
            layer = int(arm[1:])
            if layer >= n_layers:
                raise HyperionError(f"IRD depth arm {arm} exceeds the model depth")
            schedule = ((layer, 1.0),)
            phase = "discovery"
        else:
            continue
        raw_target_effect = -_finite(
            raw.get("delta_lp_base"), f"IRD depth {arm} delta_lp_base"
        )
        observations.append(
            HyperionObservation(
                observation_id=f"ird-depth:{cell_id}:{arm}",
                cell_id=cell_id,
                arm_id=arm,
                phase=phase,
                arm_kind="candidate",
                sites=("residual-stream.subject-token",),
                schedule=schedule,
                target_effect=raw_target_effect / full_effect,
                consumer_id=consumer_id,
                evidence_ref=f"{path}#/grid/{model_key}/cells/{behavior}/necessity/{arm}",
                metadata={
                    "model": model_key,
                    "target": behavior,
                    "raw_target_effect": raw_target_effect,
                    "normalization_effect": full_effect,
                    "metric": "clean-minus-ablated-target-logprob",
                    "source_index": index,
                },
            )
        )
    gates = cell.get("gates")
    gates = gates if isinstance(gates, Mapping) else {}
    if gates.get("noop_max_abs") is not None:
        observations.append(
            HyperionObservation(
                observation_id=f"ird-depth:{cell_id}:noop",
                cell_id=cell_id,
                arm_id="noop",
                phase="control",
                arm_kind="noop",
                sites=("residual-stream.subject-token",),
                schedule=(),
                target_effect=_finite(gates["noop_max_abs"], "noop_max_abs") / full_effect,
                consumer_id=consumer_id,
                evidence_ref=f"{path}#/grid/{model_key}/cells/{behavior}/gates/noop_max_abs",
                metadata={"model": model_key, "target": behavior},
            )
        )
    source = _source_record(path, payload)
    source.update({"adapter": "ird-residual-depth", "model_key": model_key, "behavior": behavior})
    return observations, source


def _certificate_effect(certificate: Mapping[str, Any], *, label: str) -> tuple[float, float]:
    panels = _sequence(certificate.get("panels"), f"{label}.panels")
    effects: list[float] = []
    repair_effects: list[float] = []
    for index, raw_panel in enumerate(panels):
        if not isinstance(raw_panel, Mapping):
            raise HyperionError(f"{label}.panels[{index}] must be an object")
        metrics = raw_panel.get("metrics")
        if not isinstance(metrics, Mapping):
            raise HyperionError(f"{label}.panels[{index}] is missing metrics")
        clean = _finite(metrics.get("clean_accuracy"), f"{label} clean_accuracy")
        deleted = _finite(
            metrics.get("source_deletion_accuracy"), f"{label} source_deletion_accuracy"
        )
        repaired = _finite(
            metrics.get("correct_repair_accuracy"), f"{label} correct_repair_accuracy"
        )
        effects.append(clean - deleted)
        repair_effects.append(repaired - deleted)
    effect = _mean(effects)
    repair = _mean(repair_effects)
    if effect is None or repair is None:
        raise HyperionError(f"{label} has no certificate panels")
    return effect, repair


def adapt_ar_induction_report(
    payload: Mapping[str, Any],
    *,
    path: Path,
    model_key: str,
    consumer_id: str,
) -> tuple[list[HyperionObservation], dict[str, Any]]:
    """Adapt the certified AR induction source-edge layer panel."""

    if payload.get("schema") != "saturn-ar-induction-circuit-certificate-report-v1":
        raise HyperionError("unsupported AR induction report schema")
    if payload.get("status") != "cross-model-ar-source-circuit-certified":
        raise HyperionError("AR induction report is not the completed cross-model certificate")
    model_key = _safe_id(model_key, "model_key")
    models = _sequence(payload.get("models"), "AR induction models")
    matches = [
        row
        for row in models
        if isinstance(row, Mapping)
        and isinstance(row.get("model"), Mapping)
        and model_key in {str(row["model"].get("name")), str(row["model"].get("model_id"))}
    ]
    if len(matches) != 1:
        raise HyperionError(f"AR induction report does not resolve one model named {model_key}")
    model = matches[0]
    execution = model.get("execution")
    if not isinstance(execution, Mapping) or execution.get("consumer") != consumer_id:
        raise HyperionError("AR induction native consumer does not match consumer_id")
    gates = model.get("gates")
    if not isinstance(gates, Mapping) or gates.get("full_universe_certified") is not True:
        raise HyperionError("AR induction model lacks a certified full-universe circuit")
    architecture = model.get("architecture")
    if not isinstance(architecture, Mapping):
        raise HyperionError("AR induction model is missing architecture")
    n_layers = int(architecture.get("layers") or 0)
    if n_layers < 2:
        raise HyperionError("AR induction model requires at least two layers")
    discovery = model.get("discovery")
    confirmation = model.get("confirmation")
    if not isinstance(discovery, Mapping) or not isinstance(confirmation, Mapping):
        raise HyperionError("AR induction model is missing discovery or confirmation data")
    full = confirmation.get("full_universe")
    if not isinstance(full, Mapping) or not isinstance(full.get("certificate"), Mapping):
        raise HyperionError("AR induction model is missing its full-universe certificate")
    full_effect, full_repair = _certificate_effect(
        full["certificate"], label="confirmation.full_universe.certificate"
    )
    if full_effect <= 0.0:
        raise HyperionError("AR induction normalization requires a positive full-universe effect")
    cell_id = f"{model_key}-induction"
    observations: list[HyperionObservation] = []
    profiles = _sequence(discovery.get("layer_profile"), "discovery.layer_profile")
    for index, raw in enumerate(profiles):
        if not isinstance(raw, Mapping):
            raise HyperionError("AR induction layer-profile rows must be objects")
        layer = int(raw.get("layer"))
        if not 0 <= layer < n_layers:
            raise HyperionError("AR induction layer-profile coordinate is out of range")
        raw_effect = _finite(raw.get("necessity_effect"), "necessity_effect")
        observations.append(
            HyperionObservation(
                observation_id=f"ar-induction:{cell_id}:L{layer}",
                cell_id=cell_id,
                arm_id=f"L{layer}",
                phase="discovery",
                arm_kind="candidate",
                sites=("attention.direct-source-edge",),
                schedule=((layer, 1.0),),
                target_effect=raw_effect / full_effect,
                repair_effect=_finite(raw.get("repair_accuracy"), "repair_accuracy") / full_effect,
                consumer_id=consumer_id,
                evidence_ref=f"{path}#/models/{models.index(model)}/discovery/layer_profile/{index}",
                metadata={
                    "model": model_key,
                    "target": "relational-induction",
                    "raw_target_effect": raw_effect,
                    "normalization_effect": full_effect,
                    "metric": "clean-minus-source-deletion-accuracy",
                },
            )
        )
    observations.append(
        HyperionObservation(
            observation_id=f"ar-induction:{cell_id}:all",
            cell_id=cell_id,
            arm_id="all",
            phase="discovery",
            arm_kind="candidate",
            sites=("attention.direct-source-edge",),
            schedule=tuple((layer, 1.0) for layer in range(n_layers)),
            target_effect=1.0,
            repair_effect=full_repair / full_effect,
            consumer_id=consumer_id,
            evidence_ref=f"{path}#/models/{models.index(model)}/confirmation/full_universe",
            metadata={
                "model": model_key,
                "target": "relational-induction",
                "raw_target_effect": full_effect,
                "normalization_effect": full_effect,
                "metric": "clean-minus-source-deletion-accuracy",
            },
        )
    )
    envelopes = _sequence(discovery.get("nested_envelopes"), "discovery.nested_envelopes")
    for index, raw in enumerate(envelopes):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("family"), Mapping):
            raise HyperionError("AR induction nested-envelope rows require a family map")
        layers = sorted(int(layer) for layer in raw["family"])
        if not layers or any(layer < 0 or layer >= n_layers for layer in layers):
            raise HyperionError("AR induction nested envelope has invalid layer support")
        certificate = raw.get("certificate")
        if not isinstance(certificate, Mapping):
            raise HyperionError("AR induction nested envelope is missing its certificate")
        raw_effect, repair_effect = _certificate_effect(
            certificate, label=f"discovery.nested_envelopes[{index}].certificate"
        )
        layer_count = int(raw.get("layer_count") or len(layers))
        observations.append(
            HyperionObservation(
                observation_id=f"ar-induction:{cell_id}:envelope-{layer_count}",
                cell_id=cell_id,
                arm_id=f"envelope-{layer_count}",
                phase="heldout",
                arm_kind="candidate" if len(layers) < n_layers else "heldout-reference",
                sites=("attention.direct-source-edge",),
                schedule=tuple((layer, 1.0) for layer in layers),
                target_effect=raw_effect / full_effect,
                repair_effect=repair_effect / full_effect,
                consumer_id=consumer_id,
                evidence_ref=(
                    f"{path}#/models/{models.index(model)}/discovery/nested_envelopes/{index}"
                ),
                metadata={
                    "model": model_key,
                    "target": "relational-induction",
                    "raw_target_effect": raw_effect,
                    "normalization_effect": full_effect,
                    "metric": "clean-minus-source-deletion-accuracy",
                    "family_id": raw.get("family_id"),
                },
            )
        )
    source = _source_record(path, payload)
    source.update(
        {
            "adapter": "ar-induction-source-edge-depth",
            "model_key": model_key,
            "report_status": payload.get("status"),
            "content_sha256": payload.get("content_sha256"),
        }
    )
    return observations, source


def load_hyperion_panel(
    payload: Mapping[str, Any], *, path: Path
) -> tuple[list[HyperionObservation], dict[str, Any]]:
    """Load Hyperion's model-agnostic normalized panel schema."""

    if payload.get("schema") != HYPERION_PANEL_SCHEMA:
        raise HyperionError("unsupported Hyperion panel schema")
    rows = _sequence(payload.get("observations"), "panel.observations")
    observations = [HyperionObservation.from_dict(row) for row in rows if isinstance(row, Mapping)]
    if len(observations) != len(rows):
        raise HyperionError("panel observations must all be objects")
    return observations, _source_record(path, payload)


def _coordinate_kind(axis: str) -> Literal["temporal", "depth"]:
    lowered = axis.lower()
    return "depth" if "layer" in lowered or "depth" in lowered else "temporal"


def compile_dynamic_circuit(
    observations: Sequence[HyperionObservation],
    *,
    symbol_id: str,
    logical_symbol: str,
    model_id: str,
    consumer_id: str,
    time_axis: str,
    address_space: str,
    residual_definition: str,
    source_artifacts: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile measured responses into a dynamic circuit symbol and analysis report."""

    symbol_id = _safe_id(symbol_id, "symbol_id")
    logical_symbol = _safe_id(logical_symbol, "logical_symbol")
    model_id = _text(model_id, "model_id")
    consumer_id = _safe_id(consumer_id, "consumer_id")
    time_axis = _safe_id(time_axis, "time_axis")
    address_space = _safe_id(address_space, "address_space")
    residual_definition = _text(residual_definition, "residual_definition")
    rows = tuple(observations)
    if not rows:
        raise HyperionError("Hyperion requires at least one observation")
    if len({row.observation_id for row in rows}) != len(rows):
        raise HyperionError("observation IDs must be unique")
    if {row.consumer_id for row in rows} != {consumer_id}:
        raise HyperionError("every observation must use the declared native consumer")
    time_indices = sorted({time for row in rows for time, _ in row.schedule})
    if len(time_indices) < 2:
        raise HyperionError("dynamic compilation requires at least two time coordinates")
    cells = _kernel_cells(rows, time_indices)
    coordinate_kind = _coordinate_kind(time_axis)
    classification = _classify(
        rows,
        cells,
        time_indices,
        coordinate_kind=coordinate_kind,
    )
    target_values = {str(row.metadata.get("target")) for row in rows if row.metadata.get("target")}
    seed_values = {str(row.metadata.get("seed")) for row in rows if row.metadata.get("seed")}
    noops = [row.target_effect for row in rows if row.arm_kind == "noop"]
    heldout_mses = [
        float(cell["heldout_mse"]) for cell in cells.values() if cell.get("heldout_mse") is not None
    ]
    evidence_level = (
        "convergent_trend"
        if len(target_values) >= 2
        and len(seed_values) >= 2
        and noops
        and max(abs(value) for value in noops) == 0.0
        and heldout_mses
        else "trend"
    )
    mean_weights = classification["observed"]["mean_kernel_weights"]
    context_payload = _mapping(context or {}, "context")
    symbol_body: dict[str, Any] = {
        "schema": HYPERION_SYMBOL_SCHEMA,
        "symbol_id": symbol_id,
        "logical_symbol": logical_symbol,
        "model_id": model_id,
        "context": context_payload,
        "context_fingerprint": _digest(context_payload),
        "address": {
            "space": address_space,
            "sites": sorted({site for row in rows for site in row.sites}),
            "binding_policy": "context-qualified; recompile on model/input/schedule drift",
        },
        "residual": {
            "definition": residual_definition,
            "carrier": "execution residual relative to the declared parent/scaffold",
        },
        "execution_coordinate": {
            "axis": time_axis,
            "kind": coordinate_kind,
            "support": time_indices,
            "kernel_kind": classification["primary"],
            "mean_weights": mean_weights,
            "properties": classification["properties"],
        },
        "consumer": {
            "consumer_id": consumer_id,
            "authority": "unchanged native consumer",
        },
        "writer": {
            "status": "not-lowered",
            "recipe_authority": "source receipt intervention schedule",
            "executable": False,
            "boundary": "Hyperion compiles evidence but does not yet lower a writable Act",
        },
        "evidence": {
            "level": evidence_level,
            "cells": sorted(cells),
            "targets": sorted(target_values),
            "seeds": sorted(seed_values),
            "source_artifacts": [_mapping(row, "source artifact") for row in source_artifacts],
            "controls": {
                "noop_count": len(noops),
                "noop_max_abs": _round(max((abs(value) for value in noops), default=0.0)),
            },
        },
        "status": "compiled",
        "terminal_claim_status": "not_assessed",
        "abstention": {
            "required_on": [
                "model or consumer mismatch",
                "context fingerprint drift",
                "time lattice mismatch",
                "address resolution failure",
                "missing native-consumer controls",
            ]
        },
        "claim_boundary": (
            "Hyperion compiles immutable causal receipts; it does not add model executions, "
            "prove minimality, certify universal semantic equivalence, or promote a terminal claim."
        ),
    }
    symbol_body["fingerprint"] = _digest(symbol_body)
    report_body: dict[str, Any] = {
        "schema": HYPERION_REPORT_SCHEMA,
        "compiler": {
            "name": "Hyperion",
            "version": 2,
            "method": (
                "discovery-only ordered-coordinate ridge kernel plus held-out "
                "consumer-response evaluation"
            ),
            "implementation": "src/saturn/hyperion.py",
            "implementation_sha256": _file_digest(Path(__file__)),
        },
        "symbol": symbol_body,
        "classification": classification,
        "kernel_cells": cells,
        "evaluation": {
            "heldout_mse_mean": _round(_mean(heldout_mses)),
            "heldout_cell_count": len(heldout_mses),
            "noop_max_abs": _round(max((abs(value) for value in noops), default=0.0)),
        },
        "coverage": {
            "observation_count": len(rows),
            "discovery_count": sum(row.phase == "discovery" for row in rows),
            "heldout_count": sum(row.phase == "heldout" for row in rows),
            "control_count": sum(row.phase == "control" for row in rows),
            "cell_count": len(cells),
            "target_count": len(target_values),
            "seed_count": len(seed_values),
            "time_count": len(time_indices),
        },
        "observations": [row.to_dict() for row in rows],
        "source_artifacts": [_mapping(row, "source artifact") for row in source_artifacts],
        "claim_boundary": symbol_body["claim_boundary"],
    }
    report_body["fingerprint"] = _digest(report_body)
    return report_body


def verify_hyperion_report(
    payload: Mapping[str, Any], *, check_sources: bool = False
) -> dict[str, Any]:
    """Verify report and nested-symbol fingerprints, optionally checking source bytes."""

    if payload.get("schema") != HYPERION_REPORT_SCHEMA:
        raise HyperionError("unsupported Hyperion report schema")
    report_fingerprint = str(payload.get("fingerprint") or "")
    report_body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if report_fingerprint != _digest(report_body):
        raise HyperionError("Hyperion report fingerprint mismatch")
    symbol = payload.get("symbol")
    if not isinstance(symbol, Mapping) or symbol.get("schema") != HYPERION_SYMBOL_SCHEMA:
        raise HyperionError("Hyperion report is missing its dynamic symbol")
    symbol_fingerprint = str(symbol.get("fingerprint") or "")
    symbol_body = {key: value for key, value in symbol.items() if key != "fingerprint"}
    if symbol_fingerprint != _digest(symbol_body):
        raise HyperionError("Hyperion symbol fingerprint mismatch")
    compiler = payload.get("compiler")
    if not isinstance(compiler, Mapping):
        raise HyperionError("Hyperion report is missing compiler provenance")
    expected_implementation = str(compiler.get("implementation_sha256") or "")
    observed_implementation = _file_digest(Path(__file__))
    if expected_implementation != observed_implementation:
        raise HyperionError("Hyperion compiler implementation drifted")
    source_checks: list[dict[str, Any]] = []
    if check_sources:
        for raw in payload.get("source_artifacts") or ():
            if not isinstance(raw, Mapping):
                raise HyperionError("source artifact rows must be objects")
            path = Path(str(raw.get("path") or "")).expanduser()
            expected = str(raw.get("sha256") or "")
            observed = _file_digest(path) if path.is_file() else None
            source_checks.append(
                {
                    "path": str(path),
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "passed": observed == expected,
                }
            )
        if not all(row["passed"] for row in source_checks):
            raise HyperionError("one or more Hyperion source artifacts drifted")
    return {
        "schema": HYPERION_REPORT_SCHEMA,
        "fingerprint": report_fingerprint,
        "symbol_fingerprint": symbol_fingerprint,
        "implementation_sha256": observed_implementation,
        "classification": _mapping(payload.get("classification"), "classification").get("primary"),
        "source_checks": source_checks,
        "passed": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HyperionError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compile_command(args: argparse.Namespace) -> int:
    observations: list[HyperionObservation] = []
    sources: list[dict[str, Any]] = []
    for raw_path in args.panel:
        path = Path(raw_path)
        rows, source = load_hyperion_panel(_read_json(path), path=path)
        observations.extend(rows)
        sources.append(source)
    for raw_path in args.ird_kernel:
        path = Path(raw_path)
        rows, source = adapt_ird_kernel_report(
            _read_json(path), path=path, consumer_id=args.consumer_id
        )
        observations.extend(rows)
        sources.append(source)
    for raw_path in args.ird_path:
        path = Path(raw_path)
        rows, source = adapt_ird_path_report(
            _read_json(path), path=path, consumer_id=args.consumer_id
        )
        observations.extend(rows)
        sources.append(source)
    if args.ird_depth:
        if not args.ird_depth_model or not args.ird_depth_behavior:
            raise HyperionError(
                "--ird-depth requires --ird-depth-model and --ird-depth-behavior"
            )
        path = Path(args.ird_depth)
        rows, source = adapt_ird_depth_report(
            _read_json(path),
            path=path,
            model_key=args.ird_depth_model,
            behavior=args.ird_depth_behavior,
            consumer_id=args.consumer_id,
        )
        observations.extend(rows)
        sources.append(source)
    if args.ar_induction:
        if not args.ar_induction_model:
            raise HyperionError("--ar-induction requires --ar-induction-model")
        path = Path(args.ar_induction)
        rows, source = adapt_ar_induction_report(
            _read_json(path),
            path=path,
            model_key=args.ar_induction_model,
            consumer_id=args.consumer_id,
        )
        observations.extend(rows)
        sources.append(source)
    context = json.loads(args.context_json)
    if not isinstance(context, dict):
        raise HyperionError("--context-json must decode to an object")
    report = compile_dynamic_circuit(
        observations,
        symbol_id=args.symbol_id,
        logical_symbol=args.logical_symbol,
        model_id=args.model_id,
        consumer_id=args.consumer_id,
        time_axis=args.time_axis,
        address_space=args.address_space,
        residual_definition=args.residual_definition,
        source_artifacts=sources,
        context=context,
    )
    _write_json(Path(args.output), report)
    print(json.dumps(verify_hyperion_report(report), indent=2, sort_keys=True))
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    result = verify_hyperion_report(_read_json(Path(args.report)), check_sources=args.check_sources)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saturn-hyperion",
        description="Compile causal response panels into dynamic circuit symbols.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--symbol-id", required=True)
    compile_parser.add_argument("--logical-symbol", required=True)
    compile_parser.add_argument("--model-id", required=True)
    compile_parser.add_argument("--consumer-id", required=True)
    compile_parser.add_argument("--time-axis", required=True)
    compile_parser.add_argument("--address-space", required=True)
    compile_parser.add_argument("--residual-definition", required=True)
    compile_parser.add_argument("--context-json", default="{}")
    compile_parser.add_argument("--panel", action="append", default=[])
    compile_parser.add_argument("--ird-kernel", action="append", default=[])
    compile_parser.add_argument("--ird-path", action="append", default=[])
    compile_parser.add_argument("--ird-depth")
    compile_parser.add_argument("--ird-depth-model")
    compile_parser.add_argument("--ird-depth-behavior")
    compile_parser.add_argument("--ar-induction")
    compile_parser.add_argument("--ar-induction-model")
    compile_parser.add_argument("--output", required=True)
    compile_parser.set_defaults(func=_compile_command)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("report")
    verify_parser.add_argument("--check-sources", action="store_true")
    verify_parser.set_defaults(func=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (HyperionError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"hyperion: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HYPERION_PANEL_SCHEMA",
    "HYPERION_REPORT_SCHEMA",
    "HYPERION_SYMBOL_SCHEMA",
    "HyperionError",
    "HyperionObservation",
    "adapt_ar_induction_report",
    "adapt_ird_depth_report",
    "adapt_ird_kernel_report",
    "adapt_ird_path_report",
    "compile_dynamic_circuit",
    "load_hyperion_panel",
    "main",
    "verify_hyperion_report",
]
