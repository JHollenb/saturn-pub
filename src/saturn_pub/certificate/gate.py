"""Fail-closed certificate gate for a bounded autoregressive induction circuit.

This module is deliberately runtime-neutral and stdlib-only: it imports no model
framework and no other part of ``saturn_pub``. A researcher freezes a
:class:`InductionCircuitPolicy` (candidate circuit family + control arms + held-out
panel thresholds) *in advance*; a worker then supplies the per-example forced-choice
outcomes of a controlled relational-induction workload; and this module decides whether
the declared direct-source attention family is necessary, position-specific,
repair-sufficient, donor-specific, replay-faithful, and connected to the model's own
native lexical consumer on multiple unopened panels.

The verdict is a signed (content-hashed) certificate dict, or -- the moment any gate
fails -- the same dict with ``certified: False`` and the failing gate named. Nothing is
tuned after the outcomes arrive: the policy is the whole decision rule.

``measure_induction_panel`` in :mod:`saturn_pub.certificate.panel` produces the panels
this gate consumes by running the panel on a native ``saturn_pub`` decoder adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

INDUCTION_CERTIFICATE_SCHEMA = "saturn-pub-induction-certificate-v1"
INDUCTION_PANEL_SCHEMA = "saturn-pub-induction-panel-v1"

# The native lexical consumer every branch must terminate at: the model's own final
# normalization followed by its own output embedding (lm head). A learned student,
# surrogate readout, or Saturn token predictor is not this consumer.
NATIVE_CONSUMER = "native_final_norm_lm_head"


class CertificateError(ValueError):
    """Raised when a purported circuit panel is incomplete or malformed."""


@dataclass(frozen=True)
class ConsumerContract:
    """Declared identity of the consumer every panel branch must terminate at.

    ``consumer`` is matched exactly. ``backends`` and ``device_prefixes`` are optional
    allow-lists frozen in advance; ``None`` means "do not constrain this axis" so the
    same policy certifies a tiny CPU specimen and a real-weight GPU run without being
    re-tuned. Constrain them when a certificate must pin its execution substrate.
    """

    consumer: str = NATIVE_CONSUMER
    backends: tuple[str, ...] | None = None
    device_prefixes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.consumer, str) or not self.consumer:
            raise CertificateError("consumer must be a non-empty string")
        for name, value in (("backends", self.backends), ("device_prefixes", self.device_prefixes)):
            if value is None:
                continue
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise CertificateError(f"{name} must be a sequence of strings or None")
            if not value or any(not isinstance(item, str) or not item for item in value):
                raise CertificateError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, tuple(value))

    def matches(self, execution: Mapping[str, Any]) -> bool:
        if not isinstance(execution, Mapping):
            return False
        if execution.get("consumer") != self.consumer:
            return False
        if self.backends is not None and str(execution.get("backend", "")) not in self.backends:
            return False
        if self.device_prefixes is not None:
            device = str(execution.get("device", ""))
            if not any(device.startswith(prefix) for prefix in self.device_prefixes):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer,
            "backends": None if self.backends is None else sorted(self.backends),
            "device_prefixes": (
                None if self.device_prefixes is None else sorted(self.device_prefixes)
            ),
        }


_FRACTIONAL_FIELDS = (
    "clean_min_accuracy",
    "source_closure_slack",
    "match_min_accuracy",
    "match_max_clean_drop",
    "repair_max_clean_drop",
    "wrong_repair_slack",
    "repair_specificity_min",
    "replay_max_correctness_mismatch",
    "replay_max_margin_mae",
)


@dataclass(frozen=True)
class InductionCircuitPolicy:
    """Preregistered empirical gates for one bounded AR source circuit."""

    min_panels: int = 2
    confidence_z: float = 1.96
    clean_min_accuracy: float = 0.85
    source_closure_slack: float = 0.03
    match_min_accuracy: float = 0.85
    match_max_clean_drop: float = 0.10
    repair_max_clean_drop: float = 0.03
    wrong_repair_slack: float = 0.05
    repair_specificity_min: float = 0.75
    replay_max_correctness_mismatch: float = 0.01
    replay_max_margin_mae: float = 1.0
    consumer: ConsumerContract = field(default_factory=ConsumerContract)

    def __post_init__(self) -> None:
        if isinstance(self.min_panels, bool) or not isinstance(self.min_panels, int):
            raise CertificateError("min_panels must be an integer")
        if self.min_panels < 1:
            raise CertificateError("min_panels must be positive")
        if self.confidence_z < 0:
            raise CertificateError("confidence_z must be non-negative")
        for name in _FRACTIONAL_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CertificateError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise CertificateError(f"{name} must be between zero and one")
        if isinstance(self.consumer, Mapping):
            object.__setattr__(self, "consumer", ConsumerContract(**dict(self.consumer)))
        if not isinstance(self.consumer, ConsumerContract):
            raise CertificateError("consumer must be a ConsumerContract")

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"min_panels": self.min_panels, "confidence_z": self.confidence_z}
        for name in _FRACTIONAL_FIELDS:
            body[name] = float(getattr(self, name))
        body["consumer"] = self.consumer.to_dict()
        return body


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_vector(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CertificateError(f"{label} must be an array")
    result = tuple(float(item) for item in value)
    if not result or any(not math.isfinite(item) for item in result):
        raise CertificateError(f"{label} must be non-empty and finite")
    return result


def _branch(panel: Mapping[str, Any], name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    branches = panel.get("branches")
    if not isinstance(branches, Mapping):
        raise CertificateError("panel.branches must be an object")
    branch = branches.get(name)
    if not isinstance(branch, Mapping):
        raise CertificateError(f"panel is missing branch {name!r}")
    correct = _finite_vector(branch.get("correct"), label=f"branches.{name}.correct")
    margin = _finite_vector(branch.get("margin"), label=f"branches.{name}.margin")
    if len(correct) != len(margin):
        raise CertificateError(f"branch {name!r} vectors must align")
    if any(item not in {0.0, 1.0} for item in correct):
        raise CertificateError(f"branch {name!r} correctness must be binary")
    return correct, margin


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _wilson_upper(successes: int, total: int, z: float) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise CertificateError("Wilson inputs are invalid")
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = estimate + z * z / (2.0 * total)
    radius = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total))
    return float((center + radius) / denominator)


def _panel_result(panel: Mapping[str, Any], policy: InductionCircuitPolicy) -> dict[str, Any]:
    if panel.get("schema") != INDUCTION_PANEL_SCHEMA:
        raise CertificateError("panel schema is not recognized")
    panel_id = panel.get("panel_id")
    if not isinstance(panel_id, str) or not panel_id:
        raise CertificateError("panel_id must be a non-empty string")
    classes = panel.get("classes")
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise CertificateError("classes must be an integer of at least two")

    clean_correct, clean_margin = _branch(panel, "clean")
    source_correct, source_margin = _branch(panel, "source_deletion")
    match_correct, match_margin = _branch(panel, "match_deletion")
    repair_correct, repair_margin = _branch(panel, "correct_repair")
    wrong_correct, wrong_margin = _branch(panel, "wrong_repair")
    lengths = {
        len(clean_correct),
        len(source_correct),
        len(match_correct),
        len(repair_correct),
        len(wrong_correct),
    }
    if len(lengths) != 1:
        raise CertificateError("all panel branches must contain the same rows")
    total = lengths.pop()
    declared_n = panel.get("examples")
    if declared_n != total:
        raise CertificateError("panel example count does not match branch rows")

    execution = panel.get("execution")
    if not isinstance(execution, Mapping):
        raise CertificateError("panel.execution must be an object")

    clean_accuracy = _mean(clean_correct)
    source_accuracy = _mean(source_correct)
    match_accuracy = _mean(match_correct)
    repair_accuracy = _mean(repair_correct)
    wrong_accuracy = _mean(wrong_correct)
    chance = 1.0 / classes
    source_upper = _wilson_upper(int(sum(source_correct)), total, policy.confidence_z)
    wrong_upper = _wilson_upper(int(sum(wrong_correct)), total, policy.confidence_z)
    correctness_mismatch = _mean(
        [float(left != right) for left, right in zip(clean_correct, repair_correct, strict=True)]
    )
    margin_mae = _mean(
        [abs(left - right) for left, right in zip(clean_margin, repair_margin, strict=True)]
    )

    gates = {
        "clean_behavior": clean_accuracy >= policy.clean_min_accuracy,
        "source_necessity": source_upper <= chance + policy.source_closure_slack,
        "position_specificity": (
            match_accuracy >= policy.match_min_accuracy
            and clean_accuracy - match_accuracy <= policy.match_max_clean_drop
        ),
        "correct_repair_sufficiency": (
            clean_accuracy - repair_accuracy <= policy.repair_max_clean_drop
        ),
        "wrong_donor_specificity": (
            wrong_upper <= chance + policy.wrong_repair_slack
            and repair_accuracy - wrong_accuracy >= policy.repair_specificity_min
        ),
        "repair_replay_fidelity": (
            correctness_mismatch <= policy.replay_max_correctness_mismatch
            and margin_mae <= policy.replay_max_margin_mae
        ),
        "native_consumer_continuation": policy.consumer.matches(execution),
    }
    return {
        "panel_id": panel_id,
        "examples": total,
        "classes": classes,
        "chance_accuracy": chance,
        "metrics": {
            "clean_accuracy": clean_accuracy,
            "clean_mean_margin": _mean(clean_margin),
            "source_deletion_accuracy": source_accuracy,
            "source_deletion_mean_margin": _mean(source_margin),
            "source_deletion_upper_confidence_bound": source_upper,
            "match_deletion_accuracy": match_accuracy,
            "match_deletion_mean_margin": _mean(match_margin),
            "correct_repair_accuracy": repair_accuracy,
            "correct_repair_mean_margin": _mean(repair_margin),
            "wrong_repair_accuracy": wrong_accuracy,
            "wrong_repair_mean_margin": _mean(wrong_margin),
            "wrong_repair_upper_confidence_bound": wrong_upper,
            "repair_specificity_margin": repair_accuracy - wrong_accuracy,
            "repair_correctness_mismatch_fraction": correctness_mismatch,
            "repair_margin_mae": margin_mae,
        },
        "gates": gates,
        "certified": all(gates.values()),
    }


def certify_induction_route(
    panels: Sequence[Mapping[str, Any]],
    policy: InductionCircuitPolicy | None = None,
) -> dict[str, Any]:
    """Certify one fixed route family over independently declared panels.

    Returns a content-sealed certificate dict. ``certified`` is ``True`` only when every
    per-panel gate passed on at least ``policy.min_panels`` distinct panels. A failing
    gate is reported, never tuned away.
    """

    selected_policy = policy or InductionCircuitPolicy()
    if not isinstance(panels, Sequence) or isinstance(panels, (str, bytes, bytearray)):
        raise CertificateError("panels must be an array")
    panel_results = tuple(_panel_result(panel, selected_policy) for panel in panels)
    panel_ids = [row["panel_id"] for row in panel_results]
    if len(panel_ids) != len(set(panel_ids)):
        raise CertificateError("panel IDs must be unique")
    gates = {
        "replication": len(panel_results) >= selected_policy.min_panels,
        "all_panels_certified": bool(panel_results)
        and all(row["certified"] for row in panel_results),
    }
    body = {
        "schema": INDUCTION_CERTIFICATE_SCHEMA,
        "policy": selected_policy.to_dict(),
        "panel_count": len(panel_results),
        "panels": list(panel_results),
        "gates": gates,
        "certified": all(gates.values()),
        "claim_boundary": (
            "bounded direct-source attention circuit for the declared arbitrary-token "
            "relational-induction workload, model bytes, route family, prompt panels, "
            "native lexical consumer, and thresholds; not a globally minimal, unique, "
            "natural-language-general, or architecture-universal circuit"
        ),
    }
    return {**body, "content_sha256": _canonical_sha256(body)}


__all__ = [
    "INDUCTION_CERTIFICATE_SCHEMA",
    "INDUCTION_PANEL_SCHEMA",
    "NATIVE_CONSUMER",
    "CertificateError",
    "ConsumerContract",
    "InductionCircuitPolicy",
    "certify_induction_route",
]
