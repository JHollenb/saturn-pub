"""Standalone offline exporter for token-time x layer-depth causal evidence.

The exporter renders existing measurement receipts and circuit thought traces.  It
does not execute a model, infer semantic labels, or turn an observation into causal
evidence.  A report row is marked causal only when it is an actual same-parent arm
whose native consumer ran and whose signed effect was recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VIEW_SCHEMA = "mdb-dual-time-causal-view-v1"
MEASUREMENT_SCHEMA = "saturn-bench-circuit-measurement-v1"
OBSERVATION_SCHEMA = "saturn-bench-circuit-observation-v1"
THOUGHT_SCHEMA = "saturn-thott-circuit-thought-v1"
PASS_DIFF_SCHEMAS = {
    "saturn-pass-causal-state-diff-v1",
    "mdb-native-pass-diff-v1",
    "saturn-native-pass-diff-v1",
    "saturn-bench-pass-diff-v1",
}
TYPED_EVENTS_SCHEMA = "mdb-dual-time-causal-events-v1"


class TemporalViewerError(ValueError):
    """Raised when evidence cannot be represented without weakening its binding."""


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TemporalViewerError("viewer inputs must contain finite JSON values") from error


def _schema(payload: Mapping[str, Any]) -> str:
    return str(payload.get("schema") or "")


def _unwrap(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    report = payload.get("report")
    if isinstance(report, Mapping) and report.get("schema") == MEASUREMENT_SCHEMA:
        return report
    return payload


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _first(mapping: Any, *paths: Sequence[str]) -> Any:
    for path in paths:
        value = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            if value is not None:
                return value
    return None


def _normalize_axis(value: Any) -> int | str:
    if isinstance(value, bool):
        return "?"
    if isinstance(value, int):
        return value if value >= 0 else "?"
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        parsed = int(value)
        return parsed if parsed >= 0 else "?"
    return "?"


def _role(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("_", "-").replace(" ", "-")


def _explicit_arm_kind(row: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    """Return only kinds supported by explicit row/spec vocabulary."""
    explicit = row.get("arm_kind") or row.get("intervention_kind")
    if explicit:
        return _role(explicit)
    role = _role(row.get("role"))
    name = _role(row.get("name"))
    if role in {"reference", "native"} or name == "native":
        return "native"
    if role in {"noop", "sham"} or name in {"noop", "sham"}:
        return "sham"
    if role == "matched-random":
        return "matched-random-control"
    if role in {"memory-transfer", "memory-restore"}:
        return role
    if "swap" in {role, name} or role.endswith("-swap") or name.startswith("swap:"):
        return "swap"
    contrasts = spec.get("contrasts") if isinstance(spec, Mapping) else None
    if isinstance(contrasts, list):
        clean_name = name.removeprefix("random:")
        for contrast in contrasts:
            if not isinstance(contrast, Mapping) or _role(contrast.get("name")) != clean_name:
                continue
            if contrast.get("mode") == "scale" and _finite_number(contrast.get("value")) == 0.0:
                return "ablation"
            return "intervention"
    return role


def _dose(row: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("dose")
    if _finite_number(value) is not None:
        return {"value": float(value), "mode": row.get("mode"), "source": "row"}
    name = _role(row.get("name")).removeprefix("random:")
    contrasts = spec.get("contrasts") if isinstance(spec, Mapping) else None
    if isinstance(contrasts, list):
        for contrast in contrasts:
            if isinstance(contrast, Mapping) and _role(contrast.get("name")) == name:
                return {
                    "value": _finite_number(contrast.get("value")),
                    "mode": contrast.get("mode"),
                    "source": "measurement_spec",
                }
    return {"value": None, "mode": None, "source": "unknown"}


def _effect(row: Mapping[str, Any]) -> dict[str, Any]:
    first_effect = _first(row, ("metrics", "first_effect", "signed_effect"))
    future_effect = _first(row, ("metrics", "future_effect", "signed_effect"))
    paired_effect = _first(row, ("metrics", "paired_future", "signed_effect"))
    first_margin = _first(row, ("first", "value"))
    future_margin = _first(row, ("future", "value"))
    value = _finite_number(future_effect)
    return {
        "sign": "positive" if value is not None and value > 0 else "negative" if value is not None and value < 0 else "zero" if value == 0 else "unknown",
        "first_signed": _finite_number(first_effect),
        "future_signed": value,
        "paired_future_signed": _finite_number(paired_effect),
        "native_first_margin": _finite_number(first_margin),
        "native_future_margin": _finite_number(future_margin),
    }


def _refs(value: Any, *, limit: int = 80) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []

    def visit(node: Any, path: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                key_text = str(key).lower()
                if isinstance(child, str) and (
                    key_text.endswith(("ref", "sha256", "digest", "cut", "cut-id", "cut_id"))
                    or child.startswith(("cas://", "ar://"))
                ):
                    found.append({"path": child_path, "value": child})
                elif isinstance(child, (Mapping, list)):
                    visit(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                child_path = f"{path}[{index}]"
                if isinstance(child, str) and child.startswith(("cas://", "ar://")):
                    found.append({"path": child_path, "value": child})
                else:
                    visit(child, child_path)

    visit(value, "")
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in found:
        unique[(item["path"], item["value"])] = item
    return list(unique.values())


def _row_token(row: Mapping[str, Any]) -> int | str:
    return _normalize_axis(
        _first(
            row,
            ("token_time",),
            ("token",),
            ("first", "closure", "cursor", "token"),
            ("reference_first", "closure", "cursor", "token"),
        )
    )


def _same_parent(row: Mapping[str, Any], parent_cut: Any) -> bool:
    return parent_cut is not None and row.get("parent_cut") == parent_cut


def _measurement(payload: Mapping[str, Any], source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if _schema(payload) not in {MEASUREMENT_SCHEMA, OBSERVATION_SCHEMA}:
        raise TemporalViewerError("unsupported circuit measurement schema")
    rows = payload.get("rows") if _schema(payload) == MEASUREMENT_SCHEMA else [payload]
    if not isinstance(rows, list):
        raise TemporalViewerError("circuit measurement rows must be a list")
    parent_cut = payload.get("parent_cut")
    spec = payload.get("spec") or {}
    if not isinstance(spec, Mapping):
        raise TemporalViewerError("circuit measurement spec must be an object")
    branches: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TemporalViewerError(f"measurement row {index} must be an object")
        row = _copy_json(raw)
        row_parent = row.get("parent_cut")
        if row_parent is None and parent_cut is not None:
            row_parent = parent_cut
            row["parent_cut"] = parent_cut
        if parent_cut is not None and row_parent != parent_cut:
            raise TemporalViewerError(f"measurement row {index} differs from report parent_cut")
        instrument = str(row.get("instrument_status") or payload.get("instrument_status") or "unknown")
        same_parent = _same_parent(row, parent_cut)
        measured_effect = _effect(row)
        has_signed_effect = measured_effect["future_signed"] is not None
        arm_kind = _explicit_arm_kind(row, spec)
        causal = (
            arm_kind != "native"
            and same_parent
            and instrument == "native-consumer-executed"
            and has_signed_effect
        )
        arm_id = str(row.get("name") or f"row-{index}")
        branch = {
            "branch_id": arm_id,
            "row_index": index,
            "role": _role(row.get("role")),
            "arm_kind": arm_kind,
            "parent_cut": row_parent,
            "same_parent": same_parent,
            "instrument_status": instrument,
            "evidence_class": "causal_evidence" if causal else "observation",
            "effect": measured_effect,
            "dose": _dose(row, spec),
            "evidence_ref": row.get("evidence_ref"),
            "site_cut": row.get("site_cut"),
            "replay_cut": row.get("replay_cut"),
            "donor_cut": row.get("donor_cut"),
            "recipient_cut": row.get("recipient_cut"),
            "receipt_ref": _first(row, ("act", "receipt_ref")),
            "source": source,
        }
        branches.append(branch)
        event_kind = "native" if branch["arm_kind"] == "native" else "intervention"
        if isinstance(row.get("act"), Mapping) and row["act"].get("receipt_ref"):
            event_kind = "commit"
        events.append(
            {
                "event_id": f"measurement:{source}:{index}",
                "token_time": _row_token(row),
                "layer_depth": _normalize_axis(row.get("site")),
                "event_kind": event_kind,
                "stage": str(row.get("replay_stage") or "measurement"),
                "node_role": None,
                "branch_id": arm_id,
                "evidence_class": branch["evidence_class"],
                "effect": branch["effect"],
                "dose": branch["dose"],
                "native_margin": {
                    "first": branch["effect"]["native_first_margin"],
                    "future": branch["effect"]["native_future_margin"],
                },
                "logical_address": row.get("source_address"),
                "physical_address": row.get("site"),
                "memory": row.get("metrics", {}).get("memory") if isinstance(row.get("metrics"), Mapping) else None,
                "payload": row.get("random_control") or row.get("act"),
                "refs": _refs(row),
                "raw": row,
            }
        )
    binding = {
        "model_identity": payload.get("model_identity") or {},
        "input_binding": {
            "spec_sha256": payload.get("spec_sha256"),
            "spec": _copy_json(spec),
            "binding_context": payload.get("binding_context"),
        },
        "numerical_contract": payload.get("numerical_contract"),
        "parent_cut": parent_cut,
        "source_schema": _schema(payload),
    }
    return branches, events, binding


def _positions(metadata: Any) -> list[int | str]:
    if not isinstance(metadata, Mapping):
        return ["?"]
    value = metadata.get("positions")
    if isinstance(value, list) and value:
        return [_normalize_axis(item) for item in value]
    if isinstance(value, Mapping) and value.get("count") and value.get("first") is not None:
        first, last = value.get("first"), value.get("last")
        if isinstance(first, int) and isinstance(last, int) and 0 <= last - first <= 64:
            return list(range(first, last + 1))
    return ["?"]


def _thought(payload: Mapping[str, Any], source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _schema(payload) != THOUGHT_SCHEMA:
        raise TemporalViewerError("unsupported circuit thought trace schema")
    nodes = payload.get("nodes")
    raw_events = payload.get("events")
    if not isinstance(nodes, list) or not isinstance(raw_events, list):
        raise TemporalViewerError("thought trace requires node and event lists")
    node_map: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if isinstance(node, Mapping) and node.get("node_id") is not None:
            node_map[str(node["node_id"])] = node
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise TemporalViewerError(f"thought event {index} must be an object")
        node = node_map.get(str(raw.get("node_id")), {})
        role = _role(node.get("role"))
        stage = _role(raw.get("stage"))
        kind = role if role in {"producer", "carrier", "consumer", "readout", "control"} else "observation"
        if any(word in stage.split("-") for word in ("commit", "write", "store")):
            kind = "commit"
        elif any(word in stage.split("-") for word in ("read", "load", "consume")):
            kind = "read"
        token_values = _positions(raw.get("metadata"))
        if token_values == ["?"]:
            token_values = [_normalize_axis(node.get("position"))]
        if token_values == ["?"]:
            token_values = [_normalize_axis(raw.get("execution_step"))]
        layer = _normalize_axis(node.get("layer"))
        for offset, token in enumerate(token_values):
            events.append(
                {
                    "event_id": f"thought:{source}:{index}:{offset}",
                    "token_time": token,
                    "layer_depth": layer,
                    "event_kind": kind,
                    "stage": str(raw.get("stage") or "unknown"),
                    "node_role": role,
                    "branch_id": None,
                    "evidence_class": "observation",
                    "effect": None,
                    "dose": None,
                    "native_margin": None,
                    "logical_address": node.get("address"),
                    "physical_address": {
                        "layer": node.get("layer"),
                        "position": node.get("position"),
                        "component": node.get("component"),
                    },
                    "memory": node.get("metadata"),
                    "payload": raw.get("summary"),
                    "refs": _refs({"node": node, "event": raw}),
                    "raw": {"node": _copy_json(node), "event": _copy_json(raw)},
                }
            )
    interventions = payload.get("interventions")
    if isinstance(interventions, list):
        for index, raw in enumerate(interventions):
            if not isinstance(raw, Mapping):
                raise TemporalViewerError(f"thought intervention {index} must be an object")
            node = node_map.get(str(raw.get("node_id")), {})
            consumer_effect = raw.get("consumer_effect")
            signed = _finite_number(
                _first(
                    consumer_effect,
                    ("signed_effect",),
                    ("effect",),
                    ("delta",),
                )
            )
            events.append(
                {
                    "event_id": f"thought-intervention:{source}:{index}",
                    "token_time": _normalize_axis(node.get("position")),
                    "layer_depth": _normalize_axis(node.get("layer")),
                    "event_kind": "intervention",
                    "stage": str(raw.get("status") or "intervention"),
                    "node_role": _role(node.get("role")),
                    "branch_id": raw.get("branch"),
                    "evidence_class": "observation",
                    "effect": {
                        "sign": "positive" if signed is not None and signed > 0 else "negative" if signed is not None and signed < 0 else "zero" if signed == 0 else "unknown",
                        "future_signed": signed,
                        "consumer_effect": consumer_effect,
                        "local_effect": raw.get("local_effect"),
                    },
                    "dose": {"value": _finite_number(raw.get("dose")), "source": "thought_trace"},
                    "native_margin": None,
                    "logical_address": node.get("address"),
                    "physical_address": {
                        "layer": node.get("layer"),
                        "position": node.get("position"),
                        "component": node.get("component"),
                    },
                    "memory": node.get("metadata"),
                    "payload": raw.get("behavior_effect"),
                    "refs": _refs({"node": node, "intervention": raw}),
                    "raw": {"node": _copy_json(node), "intervention": _copy_json(raw)},
                }
            )
    binding = {
        "model_identity": payload.get("model_identity") or {},
        "input_binding": {
            "circuit_id": payload.get("circuit_id"),
            "specimen": payload.get("specimen"),
            "content_sha256": payload.get("content_sha256"),
        },
        "numerical_contract": None,
        "parent_cut": _first(payload, ("specimen", "parent_cut")),
        "source_schema": THOUGHT_SCHEMA,
    }
    return events, binding


def _pass_diff(payload: Mapping[str, Any], source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the intentionally small, forward-compatible pass-diff interchange."""
    if _schema(payload) not in PASS_DIFF_SCHEMAS:
        raise TemporalViewerError("unsupported pass diff schema")
    if _schema(payload) == "saturn-pass-causal-state-diff-v1":
        return _causal_state_pass_diff(payload, source)
    raw_events = payload.get("events") or payload.get("rows") or payload.get("differences")
    if not isinstance(raw_events, list):
        raise TemporalViewerError("pass diff requires events, rows, or differences")
    events = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise TemporalViewerError(f"pass diff row {index} must be an object")
        effect = raw.get("effect") if isinstance(raw.get("effect"), Mapping) else {}
        signed = _finite_number(effect.get("signed_effect") or raw.get("signed_effect"))
        events.append(
            {
                "event_id": f"pass-diff:{source}:{index}",
                "token_time": _normalize_axis(raw.get("token_time", raw.get("token"))),
                "layer_depth": _normalize_axis(raw.get("layer_depth", raw.get("layer"))),
                "event_kind": str(raw.get("event_kind") or raw.get("kind") or "pass-diff"),
                "stage": str(raw.get("stage") or raw.get("pass") or "pass-diff"),
                "node_role": raw.get("node_role"),
                "branch_id": raw.get("branch_id") or raw.get("arm_id"),
                "evidence_class": str(raw.get("evidence_class") or "observation"),
                "effect": {"sign": "positive" if signed and signed > 0 else "negative" if signed and signed < 0 else "zero" if signed == 0 else "unknown", "future_signed": signed},
                "dose": raw.get("dose"),
                "native_margin": raw.get("native_margin"),
                "logical_address": raw.get("logical_address"),
                "physical_address": raw.get("physical_address"),
                "memory": raw.get("memory"),
                "payload": raw.get("payload"),
                "refs": _refs(raw),
                "raw": _copy_json(raw),
            }
        )
    binding = {
        "model_identity": payload.get("model_identity") or {},
        "input_binding": payload.get("input_binding") or {},
        "numerical_contract": payload.get("numerical_contract"),
        "parent_cut": payload.get("parent_cut"),
        "source_schema": _schema(payload),
    }
    return events, binding


def _causal_state_pass_diff(payload: Mapping[str, Any], source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt pass-state rows without guessing layer or semantic meaning."""
    binding_raw = payload.get("binding") if isinstance(payload.get("binding"), Mapping) else {}
    state = payload.get("state") if isinstance(payload.get("state"), Mapping) else {}
    events: list[dict[str, Any]] = []

    def append(raw: Mapping[str, Any], *, kind: str, layer: Any = None, token: Any = None) -> None:
        index = len(events)
        events.append(
            {
                "event_id": f"pass-diff:{source}:{index}",
                "token_time": _normalize_axis(
                    token if token is not None else raw.get("logical_position", raw.get("token_time"))
                ),
                "layer_depth": _normalize_axis(
                    layer if layer is not None else raw.get("layer_depth", raw.get("layer"))
                ),
                "event_kind": kind,
                "stage": str(raw.get("stage") or "pass-state-diff"),
                "node_role": raw.get("role"),
                "branch_id": None,
                "evidence_class": "observation",
                "effect": None,
                "dose": None,
                "native_margin": None,
                "logical_address": raw.get("logical_position"),
                "physical_address": {
                    "parent": raw.get("parent_physical_position"),
                    "child": raw.get("child_physical_position"),
                    "physical": raw.get("physical_position"),
                },
                "memory": raw.get("per_layer") or raw.get("layers") or raw,
                "payload": None,
                "refs": _refs({"row": raw, "receipts": payload.get("receipt_links")}),
                "raw": _copy_json(raw),
            }
        )

    old_rows = state.get("old_rows") if isinstance(state.get("old_rows"), Mapping) else {}
    input_authority = (
        payload.get("input_authority")
        if isinstance(payload.get("input_authority"), Mapping)
        else {}
    )
    for position in (
        input_authority.get("positions", [])
        if isinstance(input_authority.get("positions"), list)
        else []
    ):
        if isinstance(position, Mapping):
            append(position, kind="position-binding")
    for row in old_rows.get("rows", []) if isinstance(old_rows.get("rows"), list) else []:
        if isinstance(row, Mapping):
            append(row, kind="memory-old-row")
    appended = state.get("appended_rows")
    if isinstance(appended, Mapping):
        groups = appended.values()
    elif isinstance(appended, list):
        groups = [appended]
    else:
        groups = []
    for group in groups:
        rows = group.get("rows", []) if isinstance(group, Mapping) else group
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    append(row, kind="memory-appended-row")
    captures = payload.get("captures") if isinstance(payload.get("captures"), Mapping) else {}
    for capture in captures.get("layers", []) if isinstance(captures.get("layers"), list) else []:
        if not isinstance(capture, Mapping):
            continue
        rows = capture.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    append(row, kind=str(row.get("kind") or "pass-capture"), layer=capture.get("layer"))
        else:
            append(capture, kind=str(capture.get("kind") or "pass-capture"))
    readout = payload.get("native_readout")
    if isinstance(readout, Mapping):
        append(readout, kind="native-readout")
        events[-1]["native_margin"] = readout.get("raw_margin")
        events[-1]["payload"] = readout
    binding = {
        "model_identity": binding_raw.get("model_identity") or {},
        "input_binding": {
            "report_id": payload.get("report_id"),
            "pass_transition": payload.get("pass_transition"),
            "input_authority": payload.get("input_authority"),
            "identical_followup_query": binding_raw.get("identical_followup_query"),
            "native_consumer": binding_raw.get("native_consumer"),
            "status": payload.get("status"),
        },
        "numerical_contract": payload.get("numerical_contract"),
        "parent_cut": binding_raw.get("common_parent_statecut_id"),
        "source_schema": _schema(payload),
    }
    return events, binding


def _typed_events(payload: Mapping[str, Any], source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accept explicit experiment events while preserving the causal evidence gate."""
    raw_events = payload.get("events")
    binding_raw = payload.get("binding")
    if not isinstance(raw_events, list) or not isinstance(binding_raw, Mapping):
        raise TemporalViewerError("typed events require binding and events")
    events = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise TemporalViewerError(f"typed event {index} must be an object")
        evidence_class = str(raw.get("evidence_class") or "observation")
        if evidence_class not in {"observation", "causal_evidence"}:
            raise TemporalViewerError(f"typed event {index} has unsupported evidence_class")
        signed = _finite_number(_first(raw, ("effect", "signed_effect"), ("effect", "future_signed")))
        if evidence_class == "causal_evidence" and not (
            raw.get("same_parent") is True
            and raw.get("native_consumer_executed") is True
            and signed is not None
        ):
            raise TemporalViewerError(
                f"typed event {index} causal_evidence requires same_parent, native_consumer_executed and signed effect"
            )
        copied = _copy_json(raw)
        copied.update(
            {
                "event_id": str(raw.get("event_id") or f"typed:{source}:{index}"),
                "token_time": _normalize_axis(raw.get("token_time")),
                "layer_depth": _normalize_axis(raw.get("layer_depth")),
                "event_kind": str(raw.get("event_kind") or "typed-event"),
                "stage": str(raw.get("stage") or "unknown"),
                "node_role": raw.get("node_role"),
                "branch_id": raw.get("branch_id"),
                "evidence_class": evidence_class,
                "effect": raw.get("effect"),
                "dose": raw.get("dose"),
                "native_margin": raw.get("native_margin"),
                "logical_address": raw.get("logical_address"),
                "physical_address": raw.get("physical_address"),
                "memory": raw.get("memory"),
                "payload": raw.get("payload"),
                "refs": _refs(raw),
                "raw": _copy_json(raw),
            }
        )
        events.append(copied)
    binding = {
        "model_identity": binding_raw.get("model_identity") or {},
        "input_binding": binding_raw.get("input_binding") or {},
        "numerical_contract": binding_raw.get("numerical_contract"),
        "parent_cut": binding_raw.get("parent_cut"),
        "source_schema": TYPED_EVENTS_SCHEMA,
    }
    return events, binding


def _typed_branches(payload: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    raw_branches = payload.get("branches", [])
    if not isinstance(raw_branches, list):
        raise TemporalViewerError("typed branches must be a list")
    branches = []
    for index, raw in enumerate(raw_branches):
        if not isinstance(raw, Mapping) or not raw.get("branch_id"):
            raise TemporalViewerError(f"typed branch {index} requires branch_id")
        copied = _copy_json(raw)
        copied.setdefault("role", "unknown")
        copied.setdefault("arm_kind", copied["role"])
        copied.setdefault("evidence_class", "observation")
        copied.setdefault("source", source)
        branches.append(copied)
    return branches


def _merge_bindings(bindings: list[dict[str, Any]]) -> dict[str, Any]:
    models = [item["model_identity"] for item in bindings if item.get("model_identity")]
    if models and any(_json_key(value) != _json_key(models[0]) for value in models[1:]):
        raise TemporalViewerError("viewer inputs bind different model identities")
    contracts = [item["numerical_contract"] for item in bindings if item.get("numerical_contract") is not None]
    if contracts and any(_json_key(value) != _json_key(contracts[0]) for value in contracts[1:]):
        raise TemporalViewerError("viewer inputs bind different numerical contracts")
    parents = sorted({_json_key(item["parent_cut"]): item["parent_cut"] for item in bindings if item.get("parent_cut") is not None}.values(), key=str)
    return {
        "model_identity": models[0] if models else {},
        "input_bindings": [item["input_binding"] for item in bindings],
        "numerical_contract": contracts[0] if contracts else None,
        "parent_cuts": parents,
        "source_schemas": [item["source_schema"] for item in bindings],
    }


def normalize_evidence(inputs: Sequence[Mapping[str, Any]], *, sources: Sequence[str] | None = None) -> dict[str, Any]:
    """Normalize supported evidence into the stable offline viewer schema."""
    if not inputs:
        raise TemporalViewerError("at least one evidence input is required")
    names = list(sources or [f"input-{index}" for index in range(len(inputs))])
    if len(names) != len(inputs):
        raise TemporalViewerError("source labels must match evidence inputs")
    branches: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for raw, source in zip(inputs, names, strict=True):
        if not isinstance(raw, Mapping):
            raise TemporalViewerError(f"{source}: top-level evidence must be an object")
        payload = _unwrap(raw)
        schema = _schema(payload)
        if schema in {MEASUREMENT_SCHEMA, OBSERVATION_SCHEMA}:
            new_branches, new_events, binding = _measurement(payload, source)
            branches.extend(new_branches)
            events.extend(new_events)
        elif schema == THOUGHT_SCHEMA:
            new_events, binding = _thought(payload, source)
            events.extend(new_events)
        elif schema in PASS_DIFF_SCHEMAS:
            new_events, binding = _pass_diff(payload, source)
            events.extend(new_events)
        elif schema == TYPED_EVENTS_SCHEMA:
            new_events, binding = _typed_events(payload, source)
            events.extend(new_events)
            branches.extend(_typed_branches(payload, source))
        else:
            raise TemporalViewerError(f"{source}: unsupported evidence schema {schema!r}")
        bindings.append(binding)
    binding = _merge_bindings(bindings)
    tokens = sorted({event["token_time"] for event in events if event["token_time"] != "?"})
    layers = sorted({event["layer_depth"] for event in events if event["layer_depth"] != "?"})
    if any(event["token_time"] == "?" for event in events) or not tokens:
        tokens.append("?")
    if any(event["layer_depth"] == "?" for event in events) or not layers:
        layers.append("?")
    by_cell: dict[tuple[int | str, int | str], list[str]] = {}
    for event in events:
        by_cell.setdefault((event["token_time"], event["layer_depth"]), []).append(event["event_id"])
    cells = [
        {
            "token_time": token,
            "layer_depth": layer,
            "status": "observed" if (token, layer) in by_cell else "missing",
            "event_ids": by_cell.get((token, layer), []),
        }
        for token in tokens
        for layer in layers
    ]
    bundle = {
        "schema": VIEW_SCHEMA,
        "binding": binding,
        "axes": {"token_time": tokens, "layer_depth": layers},
        "branches": branches,
        "events": events,
        "cells": cells,
        "claim_boundary": {
            "observations": "recorded trace/pass events and incomplete or unpaired rows",
            "causal_evidence": "same-parent non-native arms with a recorded native-consumer signed effect",
            "missing": "no supplied event at this token/layer coordinate; absence is not a null result",
            "semantics": "labels are copied from evidence vocabulary; the viewer does not infer semantic roles",
            "execution": "static HTML; rerun controls export a proposed spec/command and never execute interventions",
        },
    }
    digest_base = _copy_json(bundle)
    bundle["content_sha256"] = hashlib.sha256(_json_key(digest_base).encode()).hexdigest()
    return bundle


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_html(bundle: Mapping[str, Any], *, title: str = "Dual-time causal path viewer") -> str:
    if bundle.get("schema") != VIEW_SCHEMA:
        raise TemporalViewerError("render_html requires a normalized viewer bundle")
    data = _script_json(bundle)
    safe_title = html.escape(title, quote=True)
    return _HTML.replace("__TITLE__", safe_title).replace("__VIEW_DATA__", data)


def export_html(inputs: Sequence[str | Path], output: str | Path, *, title: str = "Dual-time causal path viewer") -> dict[str, Any]:
    paths = [Path(item) for item in inputs]
    payloads = []
    for path in paths:
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise TemporalViewerError(f"cannot read JSON evidence {path}: {error}") from error
    bundle = normalize_evidence(payloads, sources=[str(path) for path in paths])
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(bundle, title=title), encoding="utf-8")
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export bound circuit evidence as standalone offline HTML")
    parser.add_argument("inputs", nargs="+", help="measurement report, thought trace, or pass-diff JSON")
    parser.add_argument("-o", "--output", required=True, help="standalone HTML destination")
    parser.add_argument("--title", default="Dual-time causal path viewer")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle = export_html(args.inputs, args.output, title=args.title)
    except TemporalViewerError as error:
        raise SystemExit(f"temporal viewer: {error}") from error
    print(json.dumps({"output": str(Path(args.output)), "content_sha256": bundle["content_sha256"], "events": len(bundle["events"]), "branches": len(bundle["branches"])}))
    return 0


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>__TITLE__</title><style>
:root{color-scheme:dark;--bg:#0b1018;--panel:#121a26;--line:#293548;--text:#e6edf7;--muted:#92a2b8;--obs:#59a6ff;--cause:#e6b450;--missing:#222b37;--pos:#4fd18b;--neg:#ff6b79}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}header{padding:18px 22px;border-bottom:1px solid var(--line);background:#0e1520;position:sticky;top:0;z-index:3}h1{font:600 19px system-ui;margin:0 0 4px}.sub{color:var(--muted)}main{padding:18px 22px;display:grid;gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}.binding{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}.binding div{background:#0d141f;padding:8px;border-radius:6px;overflow-wrap:anywhere}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,button{background:#172233;color:var(--text);border:1px solid #3a4960;border-radius:6px;padding:7px 9px}button{cursor:pointer}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted)}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}.matrix-wrap{overflow:auto;max-height:65vh}.matrix{display:grid;gap:3px;min-width:max-content}.axis,.cell{width:118px;min-height:54px;border:1px solid var(--line);border-radius:5px;padding:6px}.axis{background:#101925;color:var(--muted);position:sticky;z-index:2}.axis.top{top:0}.axis.side{left:0}.cell{background:#111a25;cursor:pointer}.cell:hover{outline:1px solid #6e87aa}.cell.missing{background:repeating-linear-gradient(135deg,var(--missing),var(--missing) 5px,#1b2430 5px,#1b2430 10px);color:#6f7e92}.pill{display:inline-block;border-radius:10px;padding:1px 6px;margin:1px;font-size:11px}.observation{background:#183653;color:#9bcaff}.causal_evidence{background:#4c3a18;color:#ffd98b}.positive{border-left:3px solid var(--pos)}.negative{border-left:3px solid var(--neg)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b111a;padding:10px;border-radius:6px;max-height:50vh;overflow:auto}.detail-grid{display:grid;grid-template-columns:minmax(220px,1fr) minmax(300px,2fr);gap:12px}.refs a,.refs code{display:block;overflow-wrap:anywhere;color:#9bcaff}@media(max-width:760px){.detail-grid{grid-template-columns:1fr}.axis,.cell{width:98px}}
</style></head><body><header><h1>__TITLE__</h1><div class="sub">Offline evidence surface · static HTML does not run model interventions</div></header><main>
<section class="panel"><div id="binding" class="binding"></div></section>
<section class="panel controls"><label>Branch <select id="branch"></select></label><label>Evidence <select id="evidence"><option value="all">all</option><option value="causal_evidence">causal evidence</option><option value="observation">observations</option></select></label><button id="spec">Download proposed rerun spec</button><button id="command">Download proposed command</button></section>
<section class="panel legend"><span><i class="sw" style="background:var(--cause)"></i>causal evidence</span><span><i class="sw" style="background:var(--obs)"></i>observation</span><span><i class="sw" style="background:var(--missing)"></i>missing / unobserved</span><span>Axes use “?” when source coordinates are unknown.</span></section>
<section class="panel matrix-wrap"><div id="matrix" class="matrix"></div></section>
<section class="panel detail-grid"><div><h3>Selected cell</h3><div id="summary" class="sub">Select a cell.</div><div id="refs" class="refs"></div></div><div><h3>Evidence payload</h3><pre id="detail">No cell selected.</pre></div></section>
<section class="panel"><strong>Claim boundary</strong><pre id="claims"></pre></section>
</main><script id="view-data" type="application/json">__VIEW_DATA__</script><script>
(()=>{'use strict';const D=JSON.parse(document.getElementById('view-data').textContent);const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const binding=D.binding;$('binding').innerHTML=[['model',binding.model_identity],['input / contract',binding.input_bindings],['numerical contract',binding.numerical_contract],['parent cuts',binding.parent_cuts],['bundle digest',D.content_sha256]].map(([k,v])=>`<div><b>${esc(k)}</b><br>${esc(typeof v==='string'?v:JSON.stringify(v))}</div>`).join('');$('claims').textContent=JSON.stringify(D.claim_boundary,null,2);
const branch=$('branch');branch.innerHTML='<option value="all">all actual rows</option>'+D.branches.map(b=>`<option value="${esc(b.branch_id)}">${esc(b.branch_id)} · ${esc(b.arm_kind)} · ${esc(b.evidence_class)}</option>`).join('');
const eventMap=new Map(D.events.map(e=>[e.event_id,e]));function visible(e){return(branch.value==='all'||e.branch_id===null||e.branch_id===branch.value)&&($('evidence').value==='all'||e.evidence_class===$('evidence').value)}
function render(){const T=D.axes.token_time,L=D.axes.layer_depth,M=$('matrix');M.style.gridTemplateColumns=`118px repeat(${T.length},118px)`;let out='<div class="axis top side">layer ↓ / token →</div>'+T.map(t=>`<div class="axis top">token ${esc(t)}</div>`).join('');for(const l of L){out+=`<div class="axis side">layer ${esc(l)}</div>`;for(const t of T){const c=D.cells.find(x=>String(x.token_time)===String(t)&&String(x.layer_depth)===String(l));const ev=(c?c.event_ids:[]).map(id=>eventMap.get(id)).filter(visible);if(!ev.length){out+=`<div class="cell missing" data-t="${esc(t)}" data-l="${esc(l)}">${c&&c.status==='observed'?'filtered':'missing'}</div>`;continue}const signs=ev.map(e=>e.effect&&e.effect.sign).filter(Boolean);const cls=signs.includes('positive')?' positive':signs.includes('negative')?' negative':'';out+=`<div class="cell${cls}" data-t="${esc(t)}" data-l="${esc(l)}">${ev.map(e=>`<span class="pill ${esc(e.evidence_class)}">${esc(e.event_kind)}</span>`).join('')}<br><small>${ev.length} event${ev.length===1?'':'s'}</small></div>`}}M.innerHTML=out;M.querySelectorAll('.cell').forEach(el=>el.onclick=()=>show(el.dataset.t,el.dataset.l))}
function show(t,l){const ev=D.events.filter(e=>String(e.token_time)===t&&String(e.layer_depth)===l&&visible(e));$('summary').textContent=`token ${t}, layer ${l}: ${ev.length?ev.length+' supplied event(s)':'missing / no supplied event'}`;$('detail').textContent=ev.length?JSON.stringify(ev,null,2):'No supplied evidence at this coordinate. This is missing data, not a null effect.';const refs=ev.flatMap(e=>e.refs||[]);$('refs').innerHTML=refs.map(r=>/^https?:/.test(r.value)?`<a href="${esc(r.value)}">${esc(r.path)}: ${esc(r.value)}</a>`:`<code>${esc(r.path)}: ${esc(r.value)}</code>`).join('')}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}
$('spec').onclick=()=>{const selected=branch.value==='all'?D.branches:D.branches.filter(b=>b.branch_id===branch.value);download('proposed-rerun-spec.json',JSON.stringify({schema:'mdb-temporal-viewer-rerun-proposal-v1',source_bundle_sha256:D.content_sha256,binding:D.binding,selected_branches:selected,note:'Proposal only. Submit through a qualified live bench after review.'},null,2),'application/json')};$('command').onclick=()=>download('proposed-rerun-command.txt','python -m mdb.temporal_causal_viewer <REPORT.json> [- optional TRACE.json] -o viewer.html\n# This command only rebuilds static HTML. It does not execute model interventions.\n','text/plain');branch.onchange=render;$('evidence').onchange=render;render();})();
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
