"""Standalone offline pass-indexed diffs over retained StateCut artifacts.

This module has no model or tensor-library imports. Native StateCut objects are
accepted by protocol: their tensor slots need ``detach()``, ``cpu()``,
``contiguous()``, and ``tolist()`` methods. JSON mappings use nested lists.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PASS_CAUSAL_STATE_DIFF_SCHEMA = "saturn-pass-causal-state-diff-v1"
POSITION_ROLES = frozenset(
    {"immutable_context", "appended_question", "appended_model_answer"}
)


class PassCausalStateDiffError(ValueError):
    """Raised when identity, geometry, or logical alignment is ambiguous."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise PassCausalStateDiffError("value is not finite canonical JSON") from error


def _copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical(value))
    except PassCausalStateDiffError as error:
        raise PassCausalStateDiffError(f"{label} is not JSON-like") from error


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _shape_and_flat(value: Any, *, label: str) -> tuple[tuple[int, ...], tuple[int | float, ...]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        children = [_shape_and_flat(item, label=label) for item in value]
        if not children:
            return (0,), ()
        shape = children[0][0]
        if any(child_shape != shape for child_shape, _flat in children):
            raise PassCausalStateDiffError(f"{label} must be rectangular")
        return (len(children), *shape), tuple(
            item for _child_shape, flat in children for item in flat
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PassCausalStateDiffError(f"{label} contains a non-numeric value")
    number: int | float = value
    if isinstance(number, float) and not math.isfinite(number):
        raise PassCausalStateDiffError(f"{label} contains a non-finite value")
    return (), (number,)


@dataclass(frozen=True, slots=True)
class _Array:
    shape: tuple[int, ...]
    flat: tuple[int | float, ...]
    dtype: str

    @classmethod
    def from_value(cls, value: Any, *, label: str) -> _Array:
        if isinstance(value, cls):
            return value
        dtype = type(value).__name__
        current = value
        for method in ("detach", "cpu", "contiguous"):
            action = getattr(current, method, None)
            if callable(action):
                current = action()
        raw_dtype = getattr(current, "dtype", None)
        if raw_dtype is not None:
            dtype = str(raw_dtype)
        tolist = getattr(current, "tolist", None)
        if callable(tolist):
            current = tolist()
        shape, flat = _shape_and_flat(current, label=label)
        declared_shape = getattr(value, "shape", None)
        if declared_shape is not None:
            try:
                expected = tuple(int(size) for size in declared_shape)
            except (TypeError, ValueError) as error:
                raise PassCausalStateDiffError(f"{label} has invalid shape") from error
            if shape != expected:
                raise PassCausalStateDiffError(f"{label} shape disagrees with values")
        return cls(shape=shape, flat=flat, dtype=dtype)

    @property
    def digest(self) -> str:
        return _digest({"dtype": self.dtype, "shape": self.shape, "values": self.flat})

    def select(self, axis: int, indices: Sequence[int]) -> _Array:
        if not self.shape:
            raise PassCausalStateDiffError("cannot select from a scalar")
        axis = axis if axis >= 0 else len(self.shape) + axis
        if axis < 0 or axis >= len(self.shape):
            raise PassCausalStateDiffError("array selection axis is invalid")
        if any(index < 0 or index >= self.shape[axis] for index in indices):
            raise PassCausalStateDiffError("array selection index is invalid")
        output_shape = list(self.shape)
        output_shape[axis] = len(indices)
        source_strides = [
            math.prod(self.shape[index + 1 :]) for index in range(len(self.shape))
        ]
        output_strides = [
            math.prod(output_shape[index + 1 :]) for index in range(len(output_shape))
        ]
        selected: list[int | float] = []
        for output_index in range(math.prod(output_shape)):
            coordinates = [
                (output_index // stride) % output_shape[index]
                for index, stride in enumerate(output_strides)
            ]
            coordinates[axis] = indices[coordinates[axis]]
            source_index = sum(
                coordinate * stride
                for coordinate, stride in zip(coordinates, source_strides, strict=True)
            )
            selected.append(self.flat[source_index])
        return _Array(tuple(output_shape), tuple(selected), self.dtype)

    def take(self, axis: int, index: int) -> _Array:
        axis = axis if axis >= 0 else len(self.shape) + axis
        selected = self.select(axis, [index])
        return _Array(
            tuple(size for dimension, size in enumerate(selected.shape) if dimension != axis),
            selected.flat,
            selected.dtype,
        )


@dataclass(frozen=True, slots=True)
class _Cut:
    checkpoint_id: str
    model_identity: dict[str, Any]
    slots: Mapping[str, Any]
    kv: dict[int, tuple[_Array, _Array]]


def _cut(value: Any, *, label: str) -> _Cut:
    if isinstance(value, Mapping):
        checkpoint_id = value.get("checkpoint_id") or value.get("statecut_id") or value.get("id")
        identity = value.get("model_identity")
        slots = value.get("slots", value)
    else:
        checkpoint_id = getattr(value, "checkpoint_id", None)
        identity = getattr(value, "model_identity", None)
        slots = getattr(value, "slots", None)
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise PassCausalStateDiffError(f"{label} needs a checkpoint ID")
    if not isinstance(identity, Mapping) or not identity:
        raise PassCausalStateDiffError(f"{label} needs a model identity")
    if not isinstance(slots, Mapping):
        raise PassCausalStateDiffError(f"{label} needs StateCut slots")
    kv: dict[int, tuple[_Array, _Array]] = {}
    explicit = slots.get("kv")
    if isinstance(explicit, Mapping):
        for raw_layer, row in explicit.items():
            if not isinstance(row, Mapping) or "key" not in row or "value" not in row:
                raise PassCausalStateDiffError(f"{label} KV layer {raw_layer!r} is incomplete")
            layer = int(raw_layer)
            kv[layer] = (
                _Array.from_value(row["key"], label=f"{label}.kv.{layer}.key"),
                _Array.from_value(row["value"], label=f"{label}.kv.{layer}.value"),
            )
    else:
        names = sorted(
            name
            for name in slots
            if isinstance(name, str) and name.startswith("kv.") and name.endswith(".key")
        )
        for name in names:
            layer = int(name.split(".")[1])
            value_name = f"kv.{layer:03d}.value"
            if value_name not in slots:
                raise PassCausalStateDiffError(f"{label} KV layer {layer} has no value")
            kv[layer] = (
                _Array.from_value(slots[name], label=f"{label}.{name}"),
                _Array.from_value(slots[value_name], label=f"{label}.{value_name}"),
            )
    if not kv:
        raise PassCausalStateDiffError(f"{label} has no KV layers")
    lengths: set[int] = set()
    for layer, (key, val) in kv.items():
        if len(key.shape) < 2 or key.shape[-2] != val.shape[-2]:
            raise PassCausalStateDiffError(f"{label} KV layer {layer} has invalid geometry")
        lengths.add(key.shape[-2])
    if len(lengths) != 1:
        raise PassCausalStateDiffError(f"{label} KV layers have unequal lengths")
    return _Cut(checkpoint_id, _copy(identity, label=f"{label}.model_identity"), slots, kv)


def _positions(rows: Sequence[Mapping[str, Any]], *, label: str, length: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    physical: set[int] = set()
    for index, raw in enumerate(rows):
        logical = raw.get("logical_position") if isinstance(raw, Mapping) else None
        position = raw.get("physical_position") if isinstance(raw, Mapping) else None
        role = raw.get("role") if isinstance(raw, Mapping) else None
        authority = raw.get("input_authority") if isinstance(raw, Mapping) else None
        if not isinstance(logical, str) or not logical:
            raise PassCausalStateDiffError(f"{label}[{index}] needs a logical position")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < length:
            raise PassCausalStateDiffError(f"{label}[{index}] has an invalid physical position")
        if role not in POSITION_ROLES or not isinstance(authority, str) or not authority:
            raise PassCausalStateDiffError(f"{label}[{index}] has invalid role/authority")
        if logical in result or position in physical:
            raise PassCausalStateDiffError(f"{label} positions must be unique")
        result[logical] = {
            "logical_position": logical,
            "physical_position": position,
            "role": role,
            "input_authority": authority,
        }
        physical.add(position)
    return result


def _vector(cut: _Cut, name: str) -> list[int] | None:
    if name not in cut.slots:
        return None
    value = _Array.from_value(cut.slots[name], label=f"{cut.checkpoint_id}.{name}")
    if len(value.shape) == 2 and value.shape[0] == 1:
        return [int(item) for item in value.flat]
    if len(value.shape) == 1:
        return [int(item) for item in value.flat]
    raise PassCausalStateDiffError(f"{name} must be a vector or one-row matrix")


def _row_receipt(key: _Array, value: _Array, position: int) -> dict[str, Any]:
    key_row, value_row = key.take(-2, position), value.take(-2, position)
    return {
        "key_sha256": key_row.digest,
        "value_sha256": value_row.digest,
        "key_shape": list(key_row.shape),
        "value_shape": list(value_row.shape),
    }


def _old_rows(parent: _Cut, child: _Cut, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parent_tokens, child_tokens = _vector(parent, "token_ids"), _vector(child, "token_ids")
    for logical in sorted(set(left) & set(right)):
        left_position = left[logical]["physical_position"]
        right_position = right[logical]["physical_position"]
        layers = []
        for layer in sorted(parent.kv):
            left_key, left_value = parent.kv[layer]
            right_key, right_value = child.kv[layer]
            left_receipt = _row_receipt(left_key, left_value, left_position)
            right_receipt = _row_receipt(right_key, right_value, right_position)
            layers.append(
                {
                    "layer": layer,
                    "key_equal": left_receipt["key_sha256"] == right_receipt["key_sha256"],
                    "value_equal": left_receipt["value_sha256"] == right_receipt["value_sha256"],
                    "parent": left_receipt,
                    "child": right_receipt,
                }
            )
        row = {
            "logical_position": logical,
            "parent_physical_position": left_position,
            "child_physical_position": right_position,
            "role": right[logical]["role"],
            "input_authority": right[logical]["input_authority"],
            "layers": layers,
            "all_layers_equal": all(item["key_equal"] and item["value_equal"] for item in layers),
        }
        if parent_tokens is not None and child_tokens is not None:
            left_token, right_token = parent_tokens[left_position], child_tokens[right_position]
            row["token_identity"] = {
                "equal": left_token == right_token,
                "parent_sha256": _digest({"token_id": left_token}),
                "child_sha256": _digest({"token_id": right_token}),
            }
        rows.append(row)
    return {
        "alignment": "logical_position",
        "rows": rows,
        "all_rows_equal": bool(rows)
        and all(
            row["all_layers_equal"] and row.get("token_identity", {}).get("equal", True)
            for row in rows
        ),
        "parent_rows_missing_from_child": sorted(set(left) - set(right)),
    }


def _appended(child: _Cut, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, list[dict[str, Any]]] = {
        "appended_question": [],
        "appended_model_answer": [],
    }
    tokens = _vector(child, "token_ids")
    for logical in sorted(set(right) - set(left)):
        binding = right[logical]
        role, position = binding["role"], binding["physical_position"]
        if role == "immutable_context":
            raise PassCausalStateDiffError("a child-only row cannot be immutable_context")
        row = {
            **binding,
            "layers": [
                {"layer": layer, **_row_receipt(key, value, position)}
                for layer, (key, value) in sorted(child.kv.items())
            ],
        }
        if tokens is not None:
            row["token_sha256"] = _digest({"token_id": tokens[position]})
        result[role].append(row)
    return result


def _derived(parent: _Cut, child: _Cut) -> dict[str, Any]:
    result = {}
    for name in ("attention_mask", "position_ids", "cache_position"):
        left, right = _vector(parent, name), _vector(child, name)
        result[name] = {
            "parent": left,
            "child": right,
            "changed": left != right,
            "availability": "both" if left is not None and right is not None else "partial",
        }
    names = ("pending_token_id", "pending_token")
    left = next((parent.slots[name] for name in names if name in parent.slots), None)
    right = next((child.slots[name] for name in names if name in child.slots), None)
    for label, value in (("parent", left), ("child", right)):
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        result.setdefault("pending_token", {})[label] = value
    result["pending_token"]["changed"] = result["pending_token"]["parent"] != result["pending_token"]["child"]
    return result


def _numeric(left: Any, right: Any, *, label: str) -> dict[str, Any]:
    lhs, rhs = _Array.from_value(left, label=f"{label}.parent"), _Array.from_value(
        right, label=f"{label}.child"
    )
    result = {
        "parent_shape": list(lhs.shape),
        "child_shape": list(rhs.shape),
        "parent_sha256": lhs.digest,
        "child_sha256": rhs.digest,
    }
    if lhs.shape != rhs.shape:
        return {**result, "comparable": False, "reason": "shape-mismatch"}
    deltas = [float(b) - float(a) for a, b in zip(lhs.flat, rhs.flat, strict=True)]
    left_norm = math.sqrt(sum(float(item) ** 2 for item in lhs.flat))
    right_norm = math.sqrt(sum(float(item) ** 2 for item in rhs.flat))
    denominator = left_norm * right_norm
    return {
        **result,
        "comparable": True,
        "delta_l2": math.sqrt(sum(item**2 for item in deltas)),
        "delta_max_abs": max(map(abs, deltas), default=0.0),
        "cosine": (
            sum(float(a) * float(b) for a, b in zip(lhs.flat, rhs.flat, strict=True)) / denominator
            if denominator
            else None
        ),
    }


def _logical_numeric(left: Any, right: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return _numeric(left, right, label=label)
    left_positions, right_positions = left.get("logical_positions"), right.get("logical_positions")
    if not isinstance(left_positions, list) or not isinstance(right_positions, list):
        raise PassCausalStateDiffError(f"{label} labelled values need logical_positions")
    lhs, rhs = _Array.from_value(left.get("values"), label=f"{label}.parent.values"), _Array.from_value(
        right.get("values"), label=f"{label}.child.values"
    )
    if not lhs.shape or lhs.shape[-1] != len(left_positions) or not rhs.shape or rhs.shape[-1] != len(right_positions):
        raise PassCausalStateDiffError(f"{label} logical position axis is invalid")
    left_index = {position: index for index, position in enumerate(left_positions)}
    right_index = {position: index for index, position in enumerate(right_positions)}
    common = [position for position in left_positions if position in right_index]
    if common:
        result = _numeric(
            lhs.select(-1, [left_index[position] for position in common]),
            rhs.select(-1, [right_index[position] for position in common]),
            label=f"{label}.aligned",
        )
    else:
        result = {"comparable": False, "reason": "no-common-logical-positions"}
    child_only = [position for position in right_positions if position not in left_index]
    result.update(
        {
            "alignment": "logical_position",
            "common_logical_positions": common,
            "parent_only_logical_positions": [p for p in left_positions if p not in right_index],
            "child_only_logical_positions": child_only,
            "parent_full_sha256": lhs.digest,
            "child_full_sha256": rhs.digest,
        }
    )
    if child_only:
        result["child_only_sum"] = sum(
            float(item) for item in rhs.select(-1, [right_index[p] for p in child_only]).flat
        )
    return result


def _capture_layers(value: Mapping[str, Any] | None, *, label: str) -> dict[int, Mapping[str, Any]]:
    if value is None:
        return {}
    raw = value.get("layers", value)
    iterator = raw.items() if isinstance(raw, Mapping) else (
        (row.get("layer"), row) for row in raw if isinstance(row, Mapping)
    )
    result = {}
    for layer, row in iterator:
        if not isinstance(row, Mapping):
            raise PassCausalStateDiffError(f"{label} layer {layer!r} must be an object")
        result[int(layer)] = row
    return result


def _captures(parent: Mapping[str, Any] | None, child: Mapping[str, Any] | None, control: Mapping[str, Any] | None) -> dict[str, Any]:
    left = _capture_layers(parent, label="parent_capture")
    right = _capture_layers(child, label="child_capture")
    controls = _capture_layers(control, label="control_capture")
    metrics = ("query_residual", "attention_alpha", "mlp_gate", "norm_scale")
    layers = []
    for layer in sorted(set(left) | set(right)):
        row: dict[str, Any] = {"layer": layer, "metrics": {}}
        for metric in metrics:
            if layer not in left or layer not in right or metric not in left[layer] or metric not in right[layer]:
                row["metrics"][metric] = {"availability": "incomplete"}
                continue
            summary = _logical_numeric(left[layer][metric], right[layer][metric], label=f"layer.{layer}.{metric}")
            summary["availability"] = "both"
            if layer in controls and metric in controls[layer]:
                summary["length_format_matched_control"] = _logical_numeric(
                    left[layer][metric], controls[layer][metric], label=f"layer.{layer}.{metric}.control"
                )
            row["metrics"][metric] = summary
        layers.append(row)
    return {"layers": layers, "capture_authority": "caller-supplied-offline-artifacts"}


def _readout(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    result = {}
    for name in ("target_logit", "competitor_logit", "raw_margin", "target_logprob"):
        if value.get(name) is not None:
            number = float(value[name])
            if not math.isfinite(number):
                raise PassCausalStateDiffError(f"{label}.{name} must be finite")
            result[name] = number
    if "raw_margin" not in result and {"target_logit", "competitor_logit"} <= result.keys():
        result["raw_margin"] = result["target_logit"] - result["competitor_logit"]
    if "raw_logits" in value:
        result["raw_logits"] = value["raw_logits"]
    if value.get("generated_output") is not None:
        output = value["generated_output"]
        result["generated_output"] = {
            "kind": "text" if isinstance(output, str) else "structured",
            "sha256": hashlib.sha256(output.encode()).hexdigest() if isinstance(output, str) else _digest(output),
            **({"length": len(output)} if isinstance(output, str) else {}),
        }
    return result


def _readout_diff(parent: Mapping[str, Any] | None, child: Mapping[str, Any] | None) -> dict[str, Any]:
    left, right = _readout(parent, label="parent_readout"), _readout(child, label="child_readout")
    if left is None or right is None:
        return {"availability": "incomplete"}
    left_logits, right_logits = left.pop("raw_logits", None), right.pop("raw_logits", None)
    result = {
        "availability": "both",
        "raw_logits": _numeric(left_logits, right_logits, label="native_readout.raw_logits")
        if left_logits is not None and right_logits is not None
        else {"availability": "incomplete"},
    }
    for name in ("target_logit", "competitor_logit", "raw_margin", "target_logprob"):
        result[name] = {
            "parent": left.get(name),
            "child": right.get(name),
            "delta": right[name] - left[name] if name in left and name in right else None,
        }
    result["decoded_output"] = {
        "parent": left.get("generated_output"),
        "child": right.get("generated_output"),
        "changed": left.get("generated_output") != right.get("generated_output"),
    }
    result["separation"] = "raw logits/margins are separate from log-probability and decoded output"
    return result


def _links(values: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, Any]]:
    fields = ("schema", "receipt_id", "content_sha256", "path", "uri", "parent_statecut_id", "comparison_role", "control_kind", "arm")
    result = []
    for index, value in enumerate(values):
        link = {field: value[field] for field in fields if field in value} if isinstance(value, Mapping) else {}
        if not link:
            raise PassCausalStateDiffError(f"{label}[{index}] has no durable identity")
        result.append(_copy(link, label=f"{label}[{index}]"))
    return result


def compose_pass_causal_state_diff(
    parent_statecut: Any,
    child_statecut: Any,
    *,
    pass_index: int,
    native_consumer: Mapping[str, Any],
    parent_positions: Sequence[Mapping[str, Any]],
    child_positions: Sequence[Mapping[str, Any]],
    common_parent_statecut_id: str,
    followup_query: Mapping[str, Any],
    thott_context_diff: Mapping[str, Any] | None = None,
    pass_baseline: Mapping[str, Any] | None = None,
    parent_capture: Mapping[str, Any] | None = None,
    child_capture: Mapping[str, Any] | None = None,
    control_capture: Mapping[str, Any] | None = None,
    parent_readout: Mapping[str, Any] | None = None,
    child_readout: Mapping[str, Any] | None = None,
    intervention_receipts: Sequence[Mapping[str, Any]] = (),
    restore_receipts: Sequence[Mapping[str, Any]] = (),
    continuation_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compose a native-consumer-bound N to N+1 report without executing a model."""
    if isinstance(pass_index, bool) or not isinstance(pass_index, int) or pass_index < 0:
        raise PassCausalStateDiffError("pass_index must be a non-negative integer")
    parent, child = _cut(parent_statecut, label="parent_statecut"), _cut(child_statecut, label="child_statecut")
    if parent.model_identity != child.model_identity or set(parent.kv) != set(child.kv):
        raise PassCausalStateDiffError("StateCuts have different model identities or KV layers")
    parent_length = next(iter(parent.kv.values()))[0].shape[-2]
    child_length = next(iter(child.kv.values()))[0].shape[-2]
    left = _positions(parent_positions, label="parent_positions", length=parent_length)
    right = _positions(child_positions, label="child_positions", length=child_length)
    if any(row["role"] != "immutable_context" for row in left.values()):
        raise PassCausalStateDiffError("parent rows must be immutable_context")
    if any(right[name]["role"] != "immutable_context" for name in set(left) & set(right)):
        raise PassCausalStateDiffError("shared child rows must be immutable_context")
    if not isinstance(common_parent_statecut_id, str) or not common_parent_statecut_id:
        raise PassCausalStateDiffError("common parent StateCut ID is required")
    query = _copy(followup_query, label="followup_query")
    for field in ("query_id", "input_sha256", "format_sha256", "input_authority"):
        if not isinstance(query.get(field), str) or not query[field]:
            raise PassCausalStateDiffError(f"followup_query needs {field}")
    positions = [{"side": "parent", **row} for row in left.values()] + [
        {"side": "child", **row} for row in right.values()
    ]
    links = {
        "interventions": _links(intervention_receipts, label="intervention_receipts"),
        "restores": _links(restore_receipts, label="restore_receipts"),
        "continuations": _links(continuation_receipts, label="continuation_receipts"),
        "authority": "linked receipts remain authoritative; this report does not replay them",
    }
    body = {
        "schema": PASS_CAUSAL_STATE_DIFF_SCHEMA,
        "pass_transition": {"from_pass": pass_index, "to_pass": pass_index + 1},
        "binding": {
            "native_consumer": _copy(native_consumer, label="native_consumer"),
            "model_identity": parent.model_identity,
            "parent_statecut_id": parent.checkpoint_id,
            "child_statecut_id": child.checkpoint_id,
            "common_parent_statecut_id": common_parent_statecut_id,
            "identical_followup_query": query,
        },
        "input_authority": {"positions": positions},
        "thott_context_diff": _copy(thott_context_diff, label="thott_context_diff") if thott_context_diff is not None else None,
        "pass_baseline": _copy(pass_baseline, label="pass_baseline") if pass_baseline is not None else None,
        "state": {
            "cache_geometry": {
                "parent_sequence_length": parent_length,
                "child_sequence_length": child_length,
                "unequal_length_supported": True,
                "alignment": "logical_position",
            },
            "old_rows": _old_rows(parent, child, left, right),
            "appended_rows": _appended(child, left, right),
            "derived_state": _derived(parent, child),
        },
        "captures": _captures(parent_capture, child_capture, control_capture),
        "native_readout": _readout_diff(parent_readout, child_readout),
        "receipt_links": links,
        "status": {
            "mechanics_status": "composed",
            "instrument_status": "observational-diff",
            "trend_status": "not-assigned",
            "terminal_status": "not-evaluated",
            "causal_verdict": "not-inferred-from-diffs-alone",
            "receipt_coverage": {
                "interventions": len(links["interventions"]),
                "restores": len(links["restores"]),
                "continuations": len(links["continuations"]),
            },
        },
    }
    body["report_id"] = _digest(body)
    return body


__all__ = [
    "PASS_CAUSAL_STATE_DIFF_SCHEMA",
    "POSITION_ROLES",
    "PassCausalStateDiffError",
    "compose_pass_causal_state_diff",
]
