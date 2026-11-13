"""Evidence-bound execution traces for a candidate Saturn circuit.

This module gives Thott a more precise meaning for an interpretation bubble:
an ordered, producer -> carrier -> consumer trace of a circuit's execution.
It captures bounded tensor summaries and keeps activation values in an
ephemeral in-process patch bank when a counterfactual branch needs them.  No
raw hidden state is serialized, and no prose is inferred from an activation.

The intended evidence ladder is:

``decodable state -> native consumer response -> causal patch/ablation ->
replicated circuit``

The first three are instruments.  The final label remains a scientific claim
that must be earned by the experiment using the trace.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

THOTT_CIRCUIT_THOUGHT_SCHEMA = "saturn-thott-circuit-thought-v1"
THOTT_CIRCUIT_DIFF_SCHEMA = "saturn-thott-circuit-diff-v1"
THOTT_CIRCUIT_BUBBLE_SCHEMA = "saturn-thott-circuit-bubble-v1"
MAX_SUMMARY_ELEMENTS = 4096
VALID_NODE_ROLES = frozenset({"producer", "carrier", "consumer", "readout", "control"})


class CircuitThoughtTraceError(ValueError):
    """Raised when a circuit trace violates its typed evidence contract."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        raise CircuitThoughtTraceError("raw tensor cannot enter a durable circuit receipt")
    return repr(value)


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_leaves(value: Any, path: str = "root") -> list[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(path, value)]
    if isinstance(value, Mapping):
        leaves: list[tuple[str, torch.Tensor]] = []
        for key, item in value.items():
            leaves.extend(_tensor_leaves(item, f"{path}.{key}"))
        return leaves
    if isinstance(value, (tuple, list)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_tensor_leaves(item, f"{path}[{index}]"))
        return leaves
    leaves = []
    for attribute in ("last_hidden_state", "hidden_states", "logits"):
        nested = getattr(value, attribute, None)
        if nested is not None:
            leaves.extend(_tensor_leaves(nested, f"{path}.{attribute}"))
            if leaves:
                return leaves
    return []


def _first_tensor(value: Any) -> tuple[str, torch.Tensor] | None:
    leaves = _tensor_leaves(value)
    return leaves[0] if leaves else None


def _bounded_positions(positions: Sequence[int]) -> list[int] | dict[str, Any]:
    """Keep long sequence coordinates compact in every event receipt."""

    values = tuple(int(item) for item in positions)
    if len(values) <= 16:
        return list(values)
    step = values[1] - values[0] if len(values) > 1 else None
    contiguous = step == 1 and all(
        right - left == step for left, right in zip(values[:-1], values[1:], strict=True)
    )
    result: dict[str, Any] = {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "contiguous": bool(contiguous),
    }
    if not contiguous:
        result["head"] = list(values[:4])
        result["tail"] = list(values[-4:])
    return result


def _replace_first_tensor(value: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return replacement
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            found = _first_tensor(item)
            if found is not None:
                items[index] = _replace_first_tensor(item, replacement)
                return type(value)(items)
        return value
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if _first_tensor(item) is not None:
                items[index] = _replace_first_tensor(item, replacement)
                return items
        return value
    if isinstance(value, Mapping):
        items = dict(value)
        for key, item in items.items():
            if _first_tensor(item) is not None:
                items[key] = _replace_first_tensor(item, replacement)
                return items
    return value


def _sample_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() <= MAX_SUMMARY_ELEMENTS:
        return flat, "all_values"
    indices = torch.linspace(
        0,
        flat.numel() - 1,
        MAX_SUMMARY_ELEMENTS,
        device=flat.device,
        dtype=torch.long,
    )
    return flat.index_select(0, indices), "evenly_spaced_flat_sample"


def summarize_tensor(
    tensor: torch.Tensor | None,
    *,
    feature_direction: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return bounded geometry and signal summaries without retaining values."""

    if not isinstance(tensor, torch.Tensor):
        return {"available": False, "reason": "no_tensor_leaf"}
    sampled, sample_strategy = _sample_tensor(tensor)
    finite = bool(torch.isfinite(sampled).all().item()) if sampled.numel() else True
    if sampled.numel():
        mean = float(sampled.mean().item())
        std = float(sampled.std(unbiased=False).item())
        sample_l2 = float(torch.linalg.vector_norm(sampled).item())
        abs_max = float(sampled.abs().max().item())
    else:
        mean = std = sample_l2 = abs_max = 0.0
    result: dict[str, Any] = {
        "available": True,
        "shape": [int(item) for item in tensor.shape],
        "rank": int(tensor.ndim),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "sampled_numel": int(sampled.numel()),
        "sample_strategy": sample_strategy,
        "fingerprint_scope": "bounded_summary_only",
        "full_value_available": False,
        "finite": finite,
        "mean": mean,
        "std": std,
        "sample_l2": sample_l2,
        "rms": sample_l2 / math.sqrt(max(int(sampled.numel()), 1)),
        "abs_max": abs_max,
    }
    if tensor.ndim >= 3:
        last = tensor.detach().to(torch.float32)[..., -1, :]
        result["last_position_l2"] = float(torch.linalg.vector_norm(last).item())
        result["last_position_mean"] = float(last.mean().item())
    else:
        result["last_position_l2"] = sample_l2
        result["last_position_mean"] = mean
    if feature_direction is not None:
        direction = torch.as_tensor(feature_direction, device=tensor.device, dtype=torch.float32)
        if direction.ndim != 1 or tensor.shape[-1] != direction.numel():
            raise CircuitThoughtTraceError(
                "feature direction must be rank one and match the tensor's final dimension"
            )
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
        values = tensor.detach().to(torch.float32)
        result["feature_projection"] = float((values * direction).sum(dim=-1).mean().item())
        result["last_feature_projection"] = float(
            (values[..., -1, :] * direction).sum(dim=-1).mean().item()
            if values.ndim >= 3
            else (values * direction).sum(dim=-1).mean().item()
        )
    return result


@dataclass(frozen=True, slots=True)
class CircuitNode:
    """A typed producer, carrier, consumer, or readout address."""

    node_id: str
    role: str
    address: str
    layer: int | None = None
    position: int | None = None
    component: str | None = None
    consumer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id or not self.address:
            raise CircuitThoughtTraceError("circuit nodes require an ID and address")
        if self.role not in VALID_NODE_ROLES:
            raise CircuitThoughtTraceError(
                f"unsupported circuit node role {self.role!r}; "
                f"expected one of {sorted(VALID_NODE_ROLES)}"
            )
        object.__setattr__(self, "metadata", dict(_jsonable(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "address": self.address,
            "layer": self.layer,
            "position": self.position,
            "component": self.component,
            "consumer": self.consumer,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CircuitEvent:
    """One bounded observation at one circuit address and execution boundary."""

    sequence: int
    event_key: str
    node_id: str
    stage: str
    phase: str
    execution_step: int
    summary: Mapping[str, Any]
    source_node: str | None = None
    target_node: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "event_key": self.event_key,
            "node_id": self.node_id,
            "stage": self.stage,
            "phase": self.phase,
            "execution_step": int(self.execution_step),
            "summary": dict(self.summary),
            "source_node": self.source_node,
            "target_node": self.target_node,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CircuitEvent:
        if not isinstance(payload, Mapping):
            raise CircuitThoughtTraceError("circuit event must be an object")
        return cls(
            sequence=int(payload.get("sequence", -1)),
            event_key=str(payload.get("event_key") or ""),
            node_id=str(payload.get("node_id") or ""),
            stage=str(payload.get("stage") or ""),
            phase=str(payload.get("phase") or ""),
            execution_step=int(payload.get("execution_step", -1)),
            summary=payload.get("summary") or {},
            source_node=(
                None if payload.get("source_node") is None else str(payload.get("source_node"))
            ),
            target_node=(
                None if payload.get("target_node") is None else str(payload.get("target_node"))
            ),
            metadata=payload.get("metadata") or {},
        )


class CircuitThoughtRecorder:
    """Build a compact, ordered execution trace for one candidate circuit."""

    def __init__(
        self,
        circuit_id: str,
        *,
        specimen: Mapping[str, Any] | None = None,
        model_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if not circuit_id:
            raise CircuitThoughtTraceError("circuit ID must be non-empty")
        self.circuit_id = str(circuit_id)
        self.specimen = dict(_jsonable(specimen or {}))
        self.model_identity = dict(_jsonable(model_identity or {}))
        self.nodes: dict[str, CircuitNode] = {}
        self.edges: list[dict[str, Any]] = []
        self.events: list[CircuitEvent] = []
        self.interventions: list[dict[str, Any]] = []
        self._phase = "unbound"
        self._execution_step = 0
        self._positions: tuple[int, ...] = ()
        self._occurrences: dict[tuple[str, str, str, int], int] = {}

    def add_node(self, node: CircuitNode | None = None, **kwargs: Any) -> CircuitNode:
        value = node if node is not None else CircuitNode(**kwargs)
        if value.node_id in self.nodes and self.nodes[value.node_id] != value:
            raise CircuitThoughtTraceError(f"circuit node {value.node_id!r} was redefined")
        self.nodes[value.node_id] = value
        return value

    def add_edge(
        self,
        source_node: str,
        target_node: str,
        *,
        carrier: str,
        consumer: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if source_node not in self.nodes or target_node not in self.nodes:
            raise CircuitThoughtTraceError(
                "circuit edges require registered source and target nodes"
            )
        self.edges.append(
            {
                "source_node": source_node,
                "target_node": target_node,
                "carrier": str(carrier),
                "consumer": consumer,
                "metadata": dict(_jsonable(metadata or {})),
            }
        )

    def begin(
        self,
        *,
        phase: str,
        execution_step: int,
        positions: Sequence[int] = (),
    ) -> None:
        if not phase:
            raise CircuitThoughtTraceError("execution phase must be non-empty")
        self._phase = str(phase)
        self._execution_step = int(execution_step)
        self._positions = tuple(int(item) for item in positions)

    def next_event_key(self, node_id: str, stage: str) -> str:
        if node_id not in self.nodes:
            raise CircuitThoughtTraceError(f"event references unknown node {node_id!r}")
        key = (self._phase, str(stage), node_id, self._execution_step)
        occurrence = self._occurrences.get(key, 0)
        self._occurrences[key] = occurrence + 1
        return ":".join(
            (
                self._phase,
                str(self._execution_step),
                node_id,
                str(stage),
                str(occurrence),
            )
        )

    def record(
        self,
        node_id: str,
        stage: str,
        value: torch.Tensor | None,
        *,
        event_key: str | None = None,
        source_node: str | None = None,
        target_node: str | None = None,
        feature_direction: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CircuitEvent:
        if node_id not in self.nodes:
            raise CircuitThoughtTraceError(f"event references unknown node {node_id!r}")
        key = event_key or self.next_event_key(node_id, stage)
        event = CircuitEvent(
            sequence=len(self.events),
            event_key=key,
            node_id=node_id,
            stage=str(stage),
            phase=self._phase,
            execution_step=self._execution_step,
            summary=summarize_tensor(value, feature_direction=feature_direction),
            source_node=source_node,
            target_node=target_node,
            metadata={
                "positions": _bounded_positions(self._positions),
                **dict(_jsonable(metadata or {})),
            },
        )
        self.events.append(event)
        return event

    def record_intervention(
        self,
        *,
        name: str,
        node_id: str,
        branch: str,
        dose: float,
        local_effect: Mapping[str, Any],
        consumer_effect: Mapping[str, Any],
        behavior_effect: Mapping[str, Any] | None = None,
        repaired: bool | None = None,
        status: str = "observed_intervention",
    ) -> None:
        if node_id not in self.nodes:
            raise CircuitThoughtTraceError(f"intervention references unknown node {node_id!r}")
        self.interventions.append(
            {
                "name": str(name),
                "node_id": node_id,
                "branch": str(branch),
                "dose": float(dose),
                "local_effect": dict(_jsonable(local_effect)),
                "consumer_effect": dict(_jsonable(consumer_effect)),
                "behavior_effect": dict(_jsonable(behavior_effect or {})),
                "repaired": repaired,
                "status": str(status),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": THOTT_CIRCUIT_THOUGHT_SCHEMA,
            "circuit_id": self.circuit_id,
            "specimen": dict(self.specimen),
            "model_identity": dict(self.model_identity),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": list(self.edges),
            "events": [event.to_dict() for event in self.events],
            "interventions": list(self.interventions),
            "claim_boundary": (
                "bounded mechanistic execution trace; not a private chain-of-thought "
                "transcript, semantic oracle, or training-data extractor"
            ),
        }
        return {**payload, "content_sha256": _digest(payload)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CircuitThoughtRecorder:
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != THOTT_CIRCUIT_THOUGHT_SCHEMA
        ):
            raise CircuitThoughtTraceError("unsupported circuit thought trace schema")
        declared = payload.get("content_sha256")
        base = {key: value for key, value in payload.items() if key != "content_sha256"}
        if declared is not None and str(declared) != _digest(base):
            raise CircuitThoughtTraceError("circuit thought trace content hash mismatch")
        recorder = cls(
            str(payload.get("circuit_id") or ""),
            specimen=payload.get("specimen") or {},
            model_identity=payload.get("model_identity") or {},
        )
        for raw_node in payload.get("nodes", ()):
            recorder.add_node(CircuitNode(**raw_node))
        recorder.edges = [dict(_jsonable(edge)) for edge in payload.get("edges", ())]
        recorder.events = [CircuitEvent.from_dict(raw) for raw in payload.get("events", ())]
        recorder.interventions = [
            dict(_jsonable(item)) for item in payload.get("interventions", ())
        ]
        return recorder


@contextlib.contextmanager
def capture_module_hooks(
    recorder: CircuitThoughtRecorder,
    modules: Mapping[str, torch.nn.Module],
    *,
    capture_bank: MutableMapping[str, torch.Tensor] | None = None,
    capture_nodes: Sequence[str] | None = None,
    patch_bank: Mapping[str, torch.Tensor] | None = None,
    feature_directions: Mapping[str, torch.Tensor] | None = None,
) -> Iterator[None]:
    """Capture selected module boundaries and optionally patch ephemeral outputs.

    ``capture_bank`` and ``patch_bank`` are intentionally caller-owned memory
    objects.  They are never serialized by this module.  A patch replaces the
    first tensor leaf of a module output and preserves tuple/list structure.
    This is a counterfactual execution helper, not an installation mechanism.
    """

    handles: list[Any] = []
    directions = feature_directions or {}
    capture_node_set = None if capture_nodes is None else {str(item) for item in capture_nodes}

    def capture_allowed(node_id: str) -> bool:
        return capture_bank is not None and (
            capture_node_set is None or node_id in capture_node_set
        )

    def install_pre(node_id: str, module: torch.nn.Module) -> None:
        def pre_hook(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            *,
            _node_id: str = node_id,
        ) -> None:
            value = _first_tensor((args, kwargs))
            event_key = recorder.next_event_key(_node_id, "read")
            if capture_allowed(_node_id) and value is not None:
                capture_bank[event_key] = value[1].detach().clone()  # type: ignore[index]
            recorder.record(
                _node_id,
                "read",
                None if value is None else value[1],
                event_key=event_key,
                feature_direction=directions.get(_node_id),
                metadata={"tensor_path": None if value is None else value[0]},
            )

        try:
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        except TypeError:  # pragma: no cover - compatibility with older torch
            def legacy_pre_hook(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                *,
                _node_id: str = node_id,
            ) -> None:
                value = _first_tensor(args)
                event_key = recorder.next_event_key(_node_id, "read")
                if capture_allowed(_node_id) and value is not None:
                    capture_bank[event_key] = value[1].detach().clone()  # type: ignore[index]
                recorder.record(
                    _node_id,
                    "read",
                    None if value is None else value[1],
                    event_key=event_key,
                    feature_direction=directions.get(_node_id),
                    metadata={"tensor_path": None if value is None else value[0]},
                )

            handles.append(module.register_forward_pre_hook(legacy_pre_hook))

    def install_post(node_id: str, module: torch.nn.Module) -> None:
        def post_hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            _node_id: str = node_id,
        ) -> Any:
            value = _first_tensor(output)
            event_key = recorder.next_event_key(_node_id, "write")
            if capture_allowed(_node_id) and value is not None:
                capture_bank[event_key] = value[1].detach().clone()
            recorder.record(
                _node_id,
                "write",
                None if value is None else value[1],
                event_key=event_key,
                feature_direction=directions.get(_node_id),
                metadata={"tensor_path": None if value is None else value[0]},
            )
            if patch_bank is None or event_key not in patch_bank or value is None:
                return output
            replacement = patch_bank[event_key].to(device=value[1].device, dtype=value[1].dtype)
            patched = _replace_first_tensor(output, replacement)
            patched_value = _first_tensor(patched)
            recorder.record(
                _node_id,
                "write_intervened",
                None if patched_value is None else patched_value[1],
                feature_direction=directions.get(_node_id),
                metadata={
                    "tensor_path": None if patched_value is None else patched_value[0],
                    "patch_source_event_key": event_key,
                },
            )
            return patched

        handles.append(module.register_forward_hook(post_hook))

    try:
        for node_id, module in modules.items():
            if not isinstance(module, torch.nn.Module):
                raise CircuitThoughtTraceError(
                    f"module for node {node_id!r} is not torch.nn.Module"
                )
            if node_id not in recorder.nodes:
                raise CircuitThoughtTraceError(f"module node {node_id!r} is not registered")
            install_pre(node_id, module)
            install_post(node_id, module)
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def _summary_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "sample_l2",
        "last_position_l2",
        "mean",
        "std",
        "rms",
        "abs_max",
        "feature_projection",
        "last_feature_projection",
    ):
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            delta = float(right_value) - float(left_value)
            result[key] = {
                "left": float(left_value),
                "right": float(right_value),
                "delta": delta,
                "abs_delta": abs(delta),
            }
    result["shape_equal"] = left.get("shape") == right.get("shape")
    result["change_score"] = float(
        result.get("last_position_l2", {}).get("abs_delta", 0.0)
        + 0.25 * result.get("sample_l2", {}).get("abs_delta", 0.0)
    )
    return result


def compare_circuit_traces(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare compact branch traces without reopening activation tensors."""

    if left.get("schema") != THOTT_CIRCUIT_THOUGHT_SCHEMA:
        raise CircuitThoughtTraceError("left trace has an unsupported schema")
    if right.get("schema") != THOTT_CIRCUIT_THOUGHT_SCHEMA:
        raise CircuitThoughtTraceError("right trace has an unsupported schema")
    left_events = {
        str(row.get("event_key")): row
        for row in left.get("events", ())
        if isinstance(row, Mapping)
    }
    right_events = {
        str(row.get("event_key")): row
        for row in right.get("events", ())
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for event_key in sorted(set(left_events) & set(right_events)):
        left_event = left_events[event_key]
        right_event = right_events[event_key]
        if left_event.get("node_id") != right_event.get("node_id"):
            continue
        delta = _summary_delta(left_event.get("summary") or {}, right_event.get("summary") or {})
        rows.append(
            {
                "event_key": event_key,
                "node_id": left_event.get("node_id"),
                "stage": left_event.get("stage"),
                "phase": left_event.get("phase"),
                "execution_step": left_event.get("execution_step"),
                "delta": delta,
            }
        )
    rows.sort(key=lambda row: (-float(row["delta"].get("change_score", 0.0)), row["event_key"]))
    return {
        "schema": THOTT_CIRCUIT_DIFF_SCHEMA,
        "left_circuit_id": left.get("circuit_id"),
        "right_circuit_id": right.get("circuit_id"),
        "matched_event_count": len(rows),
        "left_only_event_count": len(set(left_events) - set(right_events)),
        "right_only_event_count": len(set(right_events) - set(left_events)),
        "rows": rows,
        "top_changes": rows[:12],
        "claim_boundary": (
            "summary-level branch difference; not an exact tensor difference and not "
            "a semantic interpretation"
        ),
    }


def render_circuit_thought_bubble(
    trace: Mapping[str, Any],
    *,
    diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a deterministic, evidence-bound mechanistic thought bubble."""

    if trace.get("schema") != THOTT_CIRCUIT_THOUGHT_SCHEMA:
        raise CircuitThoughtTraceError("trace has an unsupported circuit thought schema")
    nodes = {
        str(row.get("node_id")): row
        for row in trace.get("nodes", ())
        if isinstance(row, Mapping)
    }
    edges = [row for row in trace.get("edges", ()) if isinstance(row, Mapping)]
    interventions = [
        row for row in trace.get("interventions", ()) if isinstance(row, Mapping)
    ]
    changes = [
        row for row in (diff or {}).get("top_changes", ()) if isinstance(row, Mapping)
    ]
    path = []
    for edge in edges:
        source = str(edge.get("source_node") or "")
        target = str(edge.get("target_node") or "")
        if source and source not in path:
            path.append(source)
        if target and target not in path:
            path.append(target)
    if not path:
        path = [
            str(row.get("node_id"))
            for row in trace.get("nodes", ())
            if isinstance(row, Mapping)
        ]
    path_text = " → ".join(path) if path else "<unconnected circuit>"
    top_text = []
    for row in changes[:4]:
        node_id = str(row.get("node_id") or "unknown")
        node = nodes.get(node_id, {})
        score = float((row.get("delta") or {}).get("change_score", 0.0))
        top_text.append(
            f"{node_id} ({node.get('role', 'unknown')}, change={score:.4g})"
        )
    intervention_status = [str(row.get("status")) for row in interventions]
    causal = any(
        status
        in {
            "causal_candidate",
            "consumer_closed",
            "consumer_closed_candidate",
            "replicated_causal",
        }
        for status in intervention_status
    )
    claim_status = "working_inference" if causal else "observation"
    sentence = (
        f"Observed path {path_text}. "
        + (
            "The branch panel contains a causal candidate, so the path is a working "
            "mechanistic hypothesis."
            if causal
            else "The trace is observational until a native-consumer intervention closes it."
        )
    )
    if top_text:
        sentence += " Largest compact branch changes: " + ", ".join(top_text) + "."
    return {
        "schema": THOTT_CIRCUIT_BUBBLE_SCHEMA,
        "circuit_id": trace.get("circuit_id"),
        "bubble_text": sentence,
        "claim_status": claim_status,
        "path": path,
        "steps": changes,
        "interventions": interventions,
        "evidence": {
            "trace_schema": trace.get("schema"),
            "diff_schema": (diff or {}).get("schema"),
            "matched_event_count": (diff or {}).get("matched_event_count"),
        },
        "limitations": [
            "This is a mechanistic execution annotation, not a private chain-of-thought "
            "transcript.",
            "Summary deltas do not replace exact tensor differences or native-consumer replay.",
            "A decoder or readable label would remain a hypothesis until held-out causal "
            "validation.",
        ],
    }


__all__ = [
    "CircuitEvent",
    "CircuitNode",
    "CircuitThoughtRecorder",
    "CircuitThoughtTraceError",
    "THOTT_CIRCUIT_BUBBLE_SCHEMA",
    "THOTT_CIRCUIT_DIFF_SCHEMA",
    "THOTT_CIRCUIT_THOUGHT_SCHEMA",
    "capture_module_hooks",
    "compare_circuit_traces",
    "render_circuit_thought_bubble",
    "summarize_tensor",
]
