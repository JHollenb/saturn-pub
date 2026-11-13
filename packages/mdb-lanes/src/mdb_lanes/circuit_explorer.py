"""A small, sealed catalogue for exploring executable native circuits.

The explorer is deliberately a control-plane component.  It stores qualified
model, execution, input, and forward-trace identities together with native SSA
routes, supplied role annotations, predictions, and measured outcomes.  It
does not discover semantics from names, execute a private Saturn service, or
serialize Python callbacks.

Native execution is an optional adapter supplied by the caller.  A registered
backend receives a frozen plan, the catalogue record, its bound context, and
the verified frozen trace (when one is attached), and returns a public receipt
and output.  :func:`CircuitCatalog.record_outcome` then seals the actual public
receipt and output digests into the durable catalogue.

The public API is framework-free at import time.  ``torch`` and other model
frameworks are only touched by a caller's backend or by the lazy output
descriptor used when an outcome contains a non-JSON model value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

CIRCUIT_CATALOG_SCHEMA = "mdb-circuit-catalog-v1"
CIRCUIT_RECORD_SCHEMA = "mdb-circuit-record-v1"
CIRCUIT_PLAN_SCHEMA = "mdb-circuit-plan-v1"
CIRCUIT_OUTCOME_SCHEMA = "mdb-circuit-outcome-v1"
BRANCH_FORWARD_CONTRACT = "mdb-forward-intervention-v1"
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_JSON_MAX_DEPTH = 32


class CircuitCatalogError(ValueError):
    """Raised when a circuit, plan, context, or outcome is not admissible."""


def _copy_json(value: Any, label: str = "value", depth: int = 0) -> Any:
    """Validate and copy bounded JSON without importing a model framework."""

    if depth > _JSON_MAX_DEPTH:
        raise CircuitCatalogError(f"{label} exceeds the metadata nesting limit")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if value != value or value in {float("inf"), float("-inf")}:  # noqa: PLR2004
            raise CircuitCatalogError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CircuitCatalogError(f"{label} contains a non-text key")
            result[key] = _copy_json(item, f"{label}.{key}", depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_copy_json(item, f"{label}[{index}]", depth + 1)
                for index, item in enumerate(value)]
    raise CircuitCatalogError(
        f"{label} contains unsupported {type(value).__module__}.{type(value).__name__}"
    )


def _canonical(value: Any, label: str = "value") -> bytes:
    try:
        return json.dumps(
            _copy_json(value, label), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CircuitCatalogError):
            raise
        raise CircuitCatalogError(f"{label} is not canonical JSON") from exc


def _digest(value: Any, label: str = "value") -> str:
    return hashlib.sha256(_canonical(value, label)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CircuitCatalogError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise CircuitCatalogError(f"{label} must be non-empty text")
    return value


def _sealed(body: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise CircuitCatalogError(f"{label} must be an object")
    copied = _copy_json(body, label)
    if copied.get("schema") != schema:
        raise CircuitCatalogError(f"{label} has unsupported schema")
    fingerprint = copied.pop("fingerprint", None)
    if fingerprint is None:
        raise CircuitCatalogError(f"{label} requires a fingerprint")
    _sha(fingerprint, f"{label}.fingerprint")
    if fingerprint != _digest(copied, label):
        raise CircuitCatalogError(f"{label} fingerprint mismatch")
    copied["fingerprint"] = fingerprint
    return copied


def _seal(body: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    copied = _copy_json(body, label)
    copied["schema"] = schema
    copied.pop("fingerprint", None)
    copied["fingerprint"] = _digest(copied, label)
    return copied


def _output_value_refs(tree: Any) -> list[str]:
    if isinstance(tree, Mapping):
        if set(tree) == {"$value"} and isinstance(tree["$value"], str):
            return [tree["$value"]]
        return [ref for item in tree.values() for ref in _output_value_refs(item)]
    if isinstance(tree, list):
        return [ref for item in tree for ref in _output_value_refs(item)]
    return []


def _output_digest(value: Any) -> str:
    """Digest a public output descriptor, lazily supporting tensor outputs."""

    try:
        from saturn_pub.values import describe

        return _digest(describe(value), "output descriptor")
    except (ImportError, ModuleNotFoundError):
        return _digest(value, "output")
    except CircuitCatalogError:
        raise
    except Exception as exc:  # model adapters should fail closed, never guess
        raise CircuitCatalogError(f"cannot describe native output: {exc}") from exc


def _prediction_output_keys(value: Any, label: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise CircuitCatalogError(f"{label} must be a non-empty sequence of output keys")
    keys = [_text(item, f"{label} key") for item in value]
    if len(set(keys)) != len(keys):
        raise CircuitCatalogError(f"{label} must contain unique output keys")
    return keys


def _project_output(value: Any, output_keys: Sequence[str], label: str = "output") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CircuitCatalogError(f"{label} is not a mapping and cannot be projected")
    missing = [key for key in output_keys if key not in value]
    if missing:
        raise CircuitCatalogError(f"{label} is missing scoped keys: {', '.join(missing)}")
    return {key: value[key] for key in output_keys}


def _receipt_payload(receipt: Any) -> tuple[str, dict[str, Any] | None]:
    if isinstance(receipt, str):
        return _sha(receipt, "receipt digest"), None
    if hasattr(receipt, "to_dict") and callable(receipt.to_dict):
        payload = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        payload = receipt
    else:
        raise CircuitCatalogError("receipt must be a public receipt, mapping, or digest")
    copied = _copy_json(payload, "receipt")
    supplied = copied.pop("fingerprint", None)
    declared = getattr(receipt, "fingerprint", None)
    if declared is not None:
        _sha(declared, "receipt fingerprint")
    if supplied is not None:
        _sha(supplied, "receipt fingerprint")
    fingerprint = declared if declared is not None else supplied
    expected = _digest(copied, "receipt")
    if fingerprint is None:
        fingerprint = expected
    if fingerprint != expected or (supplied is not None and supplied != fingerprint):
        raise CircuitCatalogError("receipt fingerprint disagrees with its public payload")
    copied["fingerprint"] = fingerprint
    return fingerprint, copied


def _context(
    *,
    model_identity: Any,
    execution: Any,
    input_identity: Any,
    trace_fingerprint: str,
) -> dict[str, Any]:
    trace_fingerprint = _sha(trace_fingerprint, "trace_fingerprint")
    body = {
        "model_identity": _copy_json(model_identity, "model_identity"),
        "execution": _copy_json(execution, "execution"),
        "input_identity": _copy_json(input_identity, "input_identity"),
        "trace_fingerprint": trace_fingerprint,
    }
    body["context_fingerprint"] = _digest(body, "context")
    return body


def _context_from_mapping(value: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a caller context and refuse even one stale binding field."""

    if not isinstance(value, Mapping):
        raise CircuitCatalogError("context must be a mapping")
    if "context_fingerprint" in value:
        _sha(value["context_fingerprint"], "context.context_fingerprint")
    fields = ("model_identity", "execution", "input_identity", "trace_fingerprint")
    if all(field in value for field in fields):
        observed = _context(
            model_identity=value["model_identity"],
            execution=value["execution"],
            input_identity=value["input_identity"],
            trace_fingerprint=value["trace_fingerprint"],
        )
    else:
        observed = dict(expected)
        for field in fields:
            if field in value:
                observed[field] = _copy_json(value[field], f"context.{field}")
        observed["context_fingerprint"] = _digest(
            {key: observed[key] for key in fields}, "context"
        )
    for field in (*fields, "context_fingerprint"):
        if observed[field] != expected[field]:
            raise CircuitCatalogError(f"stale context: {field} does not match the circuit binding")
    return observed


def _tensor_descriptor(value: Any, label: str) -> dict[str, Any]:
    """Describe a tensor lazily, without making the catalogue framework-bound."""

    try:
        from saturn_pub.values import describe

        descriptor = describe(value)
    except Exception as exc:  # pragma: no cover - adapter-specific failure
        raise CircuitCatalogError(f"{label} is not a describable public tensor: {exc}") from exc
    if not isinstance(descriptor, Mapping) or descriptor.get("kind") != "tensor":
        raise CircuitCatalogError(f"{label} must be a tensor")
    device = getattr(value, "device", None)
    if device is None or str(device) != "cpu":
        raise CircuitCatalogError(f"{label} must be a CPU tensor")
    contiguous = getattr(value, "is_contiguous", None)
    if callable(contiguous) and not bool(contiguous()):
        raise CircuitCatalogError(f"{label} must be contiguous")
    storage_offset = getattr(value, "storage_offset", None)
    if callable(storage_offset) and int(storage_offset()) != 0:
        raise CircuitCatalogError(f"{label} must start at storage offset zero")
    return dict(descriptor)


def _root_binding(
    trace: Mapping[str, Any], roots: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Recompute the public MDB parent seal from exactly the declared roots."""

    if not isinstance(roots, Mapping):
        raise CircuitCatalogError("branch backend requires a roots mapping")
    trace_roots = {
        value_id for value_id, row in trace["values"].items() if row["producer"] is None
    }
    if set(roots) != trace_roots:
        raise CircuitCatalogError("branch roots must be exactly the declared trace roots")
    rows: dict[str, Any] = {}
    for value_id in sorted(roots):
        descriptor = _tensor_descriptor(roots[value_id], f"root {value_id}")
        observed = trace["values"][value_id]
        for field, actual in (
            ("sha256", descriptor.get("sha256")),
            ("shape", descriptor.get("shape")),
            ("dtype", descriptor.get("dtype")),
        ):
            if actual != observed[field]:
                raise CircuitCatalogError(f"root {value_id} differs from the sealed trace ({field})")
        rows[value_id] = {
            "sha256": descriptor["sha256"],
            "shape": list(descriptor["shape"]),
            "dtype": descriptor["dtype"],
            "device": "cpu",
        }
    body = {
        "schema": "mdb-forward-intervention-parent-v1",
        "trace_fingerprint": trace["fingerprint"],
        "model_identity": trace["model_identity"],
        "execution": trace["execution"],
        "input_identity": trace["input_identity"],
        "roots": rows,
    }
    return _digest(body, "branch parent"), rows


def _branch_scalar_digest(value: Any) -> str:
    if isinstance(value, float) and (
        value != value or value in {float("inf"), float("-inf")}
    ):
        value = {"kind": "float", "value": "nan" if value != value else ("inf" if value > 0 else "-inf")}
        return _digest(value, "branch scalar")
    return _digest({"kind": type(value).__name__, "value": value}, "branch scalar")


def _branch_output_leaf_digests(value: Any, path: str = "$") -> dict[str, str]:
    """Return the leaf digests emitted by MDB's native branch receipt."""

    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for key, item in value.items():
            result.update(_branch_output_leaf_digests(item, f"{path}.{key}"))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(_branch_output_leaf_digests(item, f"{path}[{index}]"))
        return result
    try:
        from saturn_pub.values import describe

        descriptor = describe(value)
    except Exception as exc:  # pragma: no cover - adapter-specific failure
        raise CircuitCatalogError(f"cannot describe branch output at {path}: {exc}") from exc
    if isinstance(descriptor, Mapping) and descriptor.get("kind") == "tensor":
        return {path: _sha(descriptor.get("sha256"), f"branch output {path}")}
    return {path: _branch_scalar_digest(value)}


def _site_ids(intervention: Mapping[str, Any], *, required: bool = False) -> list[str]:
    raw = intervention.get("value_ids", intervention.get("intervention_sites"))
    if raw is None:
        if required:
            raise CircuitCatalogError("branch intervention must declare value_ids")
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise CircuitCatalogError("intervention value_ids must be a non-empty sequence")
    result = [_text(item, "intervention value_id") for item in raw]
    if len(set(result)) != len(result):
        raise CircuitCatalogError("intervention value_ids must be unique")
    return result


def _site_doses(intervention: Mapping[str, Any], sites: Sequence[str]) -> dict[str, float]:
    raw = intervention.get("doses", intervention.get("dose", 1.0))
    if isinstance(raw, Mapping):
        if set(raw) != set(sites):
            raise CircuitCatalogError("intervention doses must name every value_id exactly once")
        values = {site: raw[site] for site in sites}
    else:
        values = {site: raw for site in sites}
    result: dict[str, float] = {}
    for site, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CircuitCatalogError(f"dose for {site} must be numeric")
        dose = float(value)
        if dose != dose or dose in {float("inf"), float("-inf")}:
            raise CircuitCatalogError(f"dose for {site} must be finite")
        result[site] = dose
    return result


def _replacement_digests(intervention: Mapping[str, Any], sites: Sequence[str]) -> dict[str, str]:
    raw = intervention.get("replacement_digests")
    if not isinstance(raw, Mapping) or set(raw) != set(sites):
        raise CircuitCatalogError(
            "branch intervention must seal one replacement_digests entry per value_id"
        )
    return {site: _sha(raw[site], f"replacement digest for {site}") for site in sites}


def _value_ids(trace: Mapping[str, Any]) -> set[str]:
    values = trace.get("values")
    if not isinstance(values, Mapping):
        raise CircuitCatalogError("frozen trace must contain a values object")
    return set(values)


def _event_ids(trace: Mapping[str, Any]) -> set[int]:
    events = trace.get("events")
    if not isinstance(events, list):
        raise CircuitCatalogError("frozen trace must contain an events array")
    result = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or event.get("sequence") != index:
            raise CircuitCatalogError("frozen trace event sequence is not contiguous")
        result.add(index)
    return result


def _validate_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a public ForwardSymbolTable trace without importing torch."""

    if not isinstance(trace, Mapping):
        raise CircuitCatalogError("trace must be a mapping")
    try:
        from .forward_symbols import verify_trace

        verify_trace(trace)
    except ImportError as exc:
        raise CircuitCatalogError("forward-symbol trace validation is unavailable") from exc
    except Exception as exc:
        raise CircuitCatalogError(f"invalid frozen forward trace: {exc}") from exc
    return _copy_json(trace, "trace")


class TraceCircuitAdapter:
    """Derive structural routes and native edges from a verified forward table.

    The adapter never invents semantic names.  ``roles`` and evidence supplied
    by an experiment remain marked as supplied/unverified; event/value edges
    are the native structural observations derived from the trace.
    """

    def __init__(self, table: Any):
        if isinstance(table, Mapping):
            try:
                from .forward_symbols import ForwardSymbolTable

                table = ForwardSymbolTable(table)
            except Exception as exc:
                raise CircuitCatalogError(f"cannot construct ForwardSymbolTable: {exc}") from exc
        trace = getattr(table, "trace", None)
        if not isinstance(trace, Mapping):
            raise CircuitCatalogError("table must be a ForwardSymbolTable or sealed trace")
        self.table = table
        self.trace = _validate_trace(trace)
        self.trace_fingerprint = _sha(self.trace["fingerprint"], "trace.fingerprint")
        self._values = _value_ids(self.trace)
        self._events = _event_ids(self.trace)

    @classmethod
    def from_table(cls, table: Any) -> "TraceCircuitAdapter":
        return cls(table)

    @classmethod
    def from_trace(cls, trace: Mapping[str, Any]) -> "TraceCircuitAdapter":
        return cls(trace)

    def _ancestry(self, value_id: str) -> list[int]:
        try:
            return list(self.table.inspect(value_id)["event_ancestry"])
        except Exception as exc:
            raise CircuitCatalogError(f"cannot inspect route value {value_id!r}: {exc}") from exc

    def _structural_edges(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        source_edges: list[dict[str, Any]] = []
        consumer_edges: list[dict[str, Any]] = []
        for event in self.trace["events"]:
            event_data = {
                "event_sequence": event["sequence"],
                "operator": event["operator"],
                "module_path": event["module_path"],
            }
            for source in event["inputs"]:
                for target in event["outputs"]:
                    source_edges.append({**event_data, "source": source, "target": target})
                consumer_edges.append({**event_data, "value_id": source})
        return source_edges, consumer_edges

    def _normalise_route(
        self,
        candidate: Mapping[str, Any],
        index: int,
        source_edges: Sequence[Mapping[str, Any]],
        consumer_edges: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(candidate, Mapping):
            raise CircuitCatalogError("route candidates must be mappings")
        route_id = _text(candidate.get("route_id", candidate.get("id", f"route-{index}")),
                          f"route[{index}].route_id")
        target_raw = candidate.get("target_values", candidate.get("targets", candidate.get("outputs")))
        if target_raw is None:
            target_raw = candidate.get("value_ids")
        if target_raw is None:
            raise CircuitCatalogError(f"route {route_id!r} requires target_values")
        if isinstance(target_raw, str):
            target_raw = [target_raw]
        targets = tuple(_text(item, f"route {route_id}.target_values") for item in target_raw)
        if not targets or not set(targets) <= self._values:
            raise CircuitCatalogError(f"route {route_id!r} references an unknown target value")
        events_raw = candidate.get("event_sequences", candidate.get("events"))
        if events_raw is None:
            events = sorted({event for target in targets for event in self._ancestry(target)})
        else:
            events = sorted({int(item) for item in events_raw})
        if not set(events) <= self._events:
            raise CircuitCatalogError(f"route {route_id!r} references an unknown event")
        if "source_values" in candidate or "sources" in candidate:
            source_raw = candidate.get("source_values", candidate.get("sources"))
            if isinstance(source_raw, str):
                source_raw = [source_raw]
            sources = tuple(_text(item, f"route {route_id}.source_values") for item in source_raw)
        else:
            source_values = {
                source
                for edge in source_edges
                if edge["event_sequence"] in events
                for source in (edge["source"],)
                if self.trace["values"][source]["producer"] is None
            }
            sources = tuple(sorted(source_values))
        if not set(sources) <= self._values:
            raise CircuitCatalogError(f"route {route_id!r} references an unknown source value")
        route_sources = [dict(edge) for edge in source_edges
                         if edge["event_sequence"] in events
                         and edge["source"] in set(sources)]
        route_consumers = [dict(edge) for edge in consumer_edges
                           if edge["event_sequence"] in events
                           and edge["value_id"] in set(targets)]
        if candidate.get("source_edges") is not None:
            route_sources = _copy_json(candidate["source_edges"], f"route {route_id}.source_edges")
        if candidate.get("consumer_edges") is not None:
            route_consumers = _copy_json(candidate["consumer_edges"], f"route {route_id}.consumer_edges")
        roles = _copy_json(candidate.get("roles", {}), f"route {route_id}.roles")
        return {
            "route_id": route_id,
            "source_values": list(sources),
            "target_values": list(targets),
            "event_sequences": events,
            "source_edges": route_sources,
            "consumer_edges": route_consumers,
            "roles": roles,
            "role_provenance": "supplied-unverified" if roles else "none-supplied",
            "structure_provenance": "native-forward-trace",
        }

    def to_record(
        self,
        *,
        circuit_id: str,
        name: str | None = None,
        model_identity: Any = None,
        execution: Any = None,
        input_identity: Any = None,
        route_candidates: Sequence[Mapping[str, Any]] = (),
        roles: Any = None,
        evidence: Sequence[Mapping[str, Any]] = (),
        supported_interventions: Sequence[Mapping[str, Any]] = (),
        predictions: Any = (),
        replay_handle: Mapping[str, Any] | str | None = None,
    ) -> "CircuitRecord":
        for key, override in (("model_identity", model_identity),
                              ("execution", execution), ("input_identity", input_identity)):
            if override is not None and override != self.trace[key]:
                raise CircuitCatalogError(f"{key} does not match the frozen trace binding")
        binding = _context(
            model_identity=self.trace["model_identity"],
            execution=self.trace["execution"],
            input_identity=self.trace["input_identity"],
            trace_fingerprint=self.trace_fingerprint,
        )
        source_edges, consumer_edges = self._structural_edges()
        candidates = list(route_candidates)
        if not candidates:
            outputs = _output_value_refs(self.trace["outputs"])
            candidates = [{"route_id": f"output:{value_id}", "target_values": [value_id]}
                          for value_id in outputs]
        routes = [self._normalise_route(candidate, index, source_edges, consumer_edges)
                  for index, candidate in enumerate(candidates)]
        if len({route["route_id"] for route in routes}) != len(routes):
            raise CircuitCatalogError("route candidates must have unique route IDs")
        supplied_route_roles = {
            route["route_id"]: route["roles"]
            for route in routes
            if route.get("roles")
        }
        supplied_roles = _copy_json(roles or {}, "roles")
        if not supplied_roles and supplied_route_roles:
            supplied_roles = {"route_roles": supplied_route_roles}
        has_supplied_roles = bool(supplied_roles)
        payload = {
            "circuit_id": _text(circuit_id, "circuit_id"),
            "name": _text(name or circuit_id, "name"),
            "binding": binding,
            "trace": {
                "schema": self.trace.get("schema"),
                "fingerprint": self.trace_fingerprint,
                "event_count": len(self.trace["events"]),
                "value_count": len(self.trace["values"]),
            },
            "routes": routes,
            "source_edges": source_edges,
            "consumer_edges": consumer_edges,
            "roles": supplied_roles,
            "role_provenance": "supplied-unverified" if has_supplied_roles else "none-supplied",
            "evidence": _copy_json(list(evidence), "evidence"),
            "supported_interventions": _copy_json(list(supported_interventions),
                                                   "supported_interventions"),
            "predictions": _normalise_predictions(predictions, binding, routes),
            "replay_handle": _normalise_replay_handle(replay_handle),
            "statuses": {
                "structure": "observed-native-trace",
                "roles": "supplied-unverified" if has_supplied_roles else "none-supplied",
                "prediction": "sealed-prediction" if predictions else "none-recorded",
                "measurement": "unmeasured",
            },
        }
        return CircuitRecord(_seal(payload, CIRCUIT_RECORD_SCHEMA, "circuit record"))


def _normalise_replay_handle(value: Mapping[str, Any] | str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"backend": _text(value, "replay_handle.backend")}
    result = _copy_json(value, "replay_handle")
    _text(result.get("backend"), "replay_handle.backend")
    return result


def _normalise_predictions(
    predictions: Any, binding: Mapping[str, Any], routes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if predictions is None:
        return []
    if isinstance(predictions, Mapping):
        if "route_id" in predictions or "prediction_id" in predictions:
            raw_predictions = [predictions]
        else:
            raw_predictions = [
                {**item, "route_id": route_id}
                for route_id, item in predictions.items()
                if isinstance(item, Mapping)
            ]
    else:
        raw_predictions = list(predictions)
    route_ids = {route["route_id"] for route in routes}
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_predictions):
        if not isinstance(raw, Mapping):
            raise CircuitCatalogError("predictions must be mappings")
        body = _copy_json(raw, f"prediction[{index}]")
        route_id = _text(body.get("route_id"), f"prediction[{index}].route_id")
        if route_id not in route_ids:
            raise CircuitCatalogError(f"prediction references unknown route {route_id!r}")
        declared_context = body.get("context_fingerprint", binding["context_fingerprint"])
        if declared_context != binding["context_fingerprint"]:
            raise CircuitCatalogError(f"prediction {route_id!r} has a stale context")
        body["context_fingerprint"] = binding["context_fingerprint"]
        output_keys = _prediction_output_keys(body.get("output_keys"),
                                              f"prediction[{index}].output_keys")
        if output_keys is not None:
            body["output_keys"] = output_keys
        if "output" in body:
            predicted_output = body.pop("output")
            if output_keys is not None:
                predicted_output = _project_output(
                    predicted_output, output_keys, f"prediction[{index}].output"
                )
            body["output_digest"] = _output_digest(predicted_output)
        if "output_digest" not in body:
            raise CircuitCatalogError(
                f"prediction {route_id!r} requires a sealed output_digest or output"
            )
        _sha(body["output_digest"], f"prediction[{index}].output_digest")
        body["prediction_id"] = _text(
            body.get("prediction_id", f"prediction:{route_id}:{index}"),
            f"prediction[{index}].prediction_id",
        )
        body["status"] = "sealed-prediction"
        body["provenance"] = body.get("provenance", "experiment-local-predictor")
        body.pop("fingerprint", None)
        body["fingerprint"] = _digest(body, f"prediction[{index}]")
        result.append(body)
    if len({item["prediction_id"] for item in result}) != len(result):
        raise CircuitCatalogError("prediction IDs must be unique")
    return result


class CircuitRecord:
    """Immutable, sealed catalogue description of one qualified circuit."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = _sealed(payload, CIRCUIT_RECORD_SCHEMA, "circuit record")

    @property
    def circuit_id(self) -> str:
        return self._payload["circuit_id"]

    @property
    def fingerprint(self) -> str:
        return self._payload["fingerprint"]

    @property
    def binding(self) -> dict[str, Any]:
        return _copy_json(self._payload["binding"], "binding")

    @property
    def trace_fingerprint(self) -> str:
        return self._payload["binding"]["trace_fingerprint"]

    def to_dict(self) -> dict[str, Any]:
        return _copy_json(self._payload, "circuit record")

    def route(self, route_id: str | None = None) -> dict[str, Any]:
        routes = self._payload["routes"]
        if route_id is None:
            if len(routes) != 1:
                raise CircuitCatalogError("route_id is required when a circuit has multiple routes")
            return _copy_json(routes[0], "route")
        for route in routes:
            if route["route_id"] == route_id:
                return _copy_json(route, "route")
        raise CircuitCatalogError(f"unknown route {route_id!r} for circuit {self.circuit_id!r}")


class CircuitPlan:
    """Immutable branch plan tied to one circuit binding and prediction."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = _sealed(payload, CIRCUIT_PLAN_SCHEMA, "circuit plan")

    @property
    def plan_id(self) -> str:
        return self._payload["plan_id"]

    @property
    def fingerprint(self) -> str:
        return self._payload["fingerprint"]

    def to_dict(self) -> dict[str, Any]:
        return _copy_json(self._payload, "circuit plan")


class CircuitOutcome:
    """Immutable measured or refused result bound to a plan."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = _sealed(payload, CIRCUIT_OUTCOME_SCHEMA, "circuit outcome")

    @property
    def outcome_id(self) -> str:
        return self._payload["outcome_id"]

    def to_dict(self) -> dict[str, Any]:
        return _copy_json(self._payload, "circuit outcome")


class CircuitBackend(Protocol):
    """Public replay adapter protocol used by :meth:`CircuitCatalog.execute`.

    Implementations may accept additional keyword arguments supplied to
    ``execute``.  The result is a mapping with ``receipt`` and ``output`` or
    an object exposing those two attributes.
    """

    def __call__(
        self,
        *,
        plan: Mapping[str, Any],
        record: Mapping[str, Any],
        context: Mapping[str, Any],
        trace: Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> Any: ...


class CircuitBackendVerifier(Protocol):
    """Explicit evidence verifier for a registered execution backend.

    A catalogue cannot infer whether an arbitrary callback actually ran a
    native consumer.  The callback therefore has to be paired with a verifier
    owned by that backend (or with :func:`verify_branch_forward_result`).
    """

    def __call__(
        self,
        *,
        plan: Mapping[str, Any],
        record: Mapping[str, Any],
        context: Mapping[str, Any],
        trace: Mapping[str, Any] | None,
        result: Any,
        kwargs: Mapping[str, Any],
    ) -> None: ...


@dataclass(frozen=True)
class _BackendRegistration:
    callback: Backend
    verifier: CircuitBackendVerifier | None
    contract: str


def _result_parts(result: Any) -> tuple[Any, Any]:
    if isinstance(result, Mapping):
        if "receipt" not in result or "output" not in result:
            raise CircuitCatalogError("backend result must contain receipt and output")
        return result["receipt"], result["output"]
    if not hasattr(result, "receipt") or not hasattr(result, "output"):
        raise CircuitCatalogError("backend result must expose receipt and output")
    return result.receipt, result.output


def verify_branch_forward_result(
    *,
    plan: Mapping[str, Any],
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    trace: Mapping[str, Any] | None,
    result: Any,
    kwargs: Mapping[str, Any],
) -> None:
    """Verify a result produced by public ``mdb.forward_interventions``.

    This is intentionally a receipt and binding verifier, not an executor. It
    checks the public receipt against the sealed plan and the caller-supplied
    roots/replacements.  The callback remains responsible for actually using
    ``branch_forward_many``; a generic callback must provide its own verifier.
    """

    if trace is None:
        raise CircuitCatalogError("branch-forward backend requires the frozen trace")
    try:
        from saturn_pub import Receipt
    except ImportError as exc:  # pragma: no cover - package dependency failure
        raise CircuitCatalogError("branch-forward verification requires saturn_pub.Receipt") from exc
    receipt, output = _result_parts(result)
    if not isinstance(receipt, Receipt):
        raise CircuitCatalogError("branch-forward backend must return the public Receipt instance")
    _, payload = _receipt_payload(receipt)
    if payload is None or payload.get("schema") != "saturn-pub-receipt-v1":
        raise CircuitCatalogError("branch-forward backend returned an invalid public receipt")
    if payload.get("operation") != "mdb.native-aten.ssa-replacement-branch":
        raise CircuitCatalogError("receipt operation is not the public MDB branch operation")
    if payload.get("status") != "observed":
        raise CircuitCatalogError("branch-forward receipt is not observed")
    if payload.get("trace_fingerprint") != context["trace_fingerprint"]:
        raise CircuitCatalogError("receipt trace fingerprint does not match the plan")
    for field in ("model_identity", "execution", "input_identity"):
        if payload.get(field) != context[field]:
            raise CircuitCatalogError(f"receipt {field} does not match the plan")
    if trace.get("fingerprint") != context["trace_fingerprint"]:
        raise CircuitCatalogError("supplied trace does not match the plan context")
    for field in ("model_identity", "execution", "input_identity"):
        if trace.get(field) != context[field]:
            raise CircuitCatalogError(f"supplied trace {field} does not match the plan")

    parent_fingerprint, root_rows = _root_binding(trace, kwargs.get("roots"))
    if payload.get("root_fingerprints") != root_rows:
        raise CircuitCatalogError("receipt root fingerprints do not match the supplied roots")
    if payload.get("parent_fingerprint") != parent_fingerprint:
        raise CircuitCatalogError("receipt parent fingerprint does not match the supplied roots")

    intervention = plan.get("intervention")
    if not isinstance(intervention, Mapping) or not intervention:
        raise CircuitCatalogError("branch-forward plan must declare an intervention")
    sites = _site_ids(intervention, required=True)
    doses = _site_doses(intervention, sites)
    # The callback must pass the plan's dose through to the public branch
    # engine.  A receipt alone cannot prove that a callback did so: a generic
    # adapter could execute a different dose and then rewrite the receipt.
    observed_doses = _site_doses({"dose": kwargs.get("dose", 1.0)}, sites)
    if observed_doses != doses:
        raise CircuitCatalogError("backend dose does not match the sealed plan")
    replacement_digests = _replacement_digests(intervention, sites)
    replacements = kwargs.get("replacements")
    if not isinstance(replacements, Mapping) or set(replacements) != set(sites):
        raise CircuitCatalogError("supplied replacements do not match the sealed plan sites")
    for value_id in sites:
        value_row = trace["values"].get(value_id)
        if not isinstance(value_row, Mapping) or value_row.get("producer") is None:
            raise CircuitCatalogError(f"branch site {value_id} is not a produced trace value")
        descriptor = _tensor_descriptor(replacements[value_id], f"replacement {value_id}")
        if descriptor.get("sha256") != replacement_digests[value_id]:
            raise CircuitCatalogError(f"replacement digest for {value_id} differs from the plan")
        if list(descriptor.get("shape", ())) != value_row.get("shape"):
            raise CircuitCatalogError(f"replacement shape for {value_id} differs from the trace")
        if descriptor.get("dtype") != value_row.get("dtype"):
            raise CircuitCatalogError(f"replacement dtype for {value_id} differs from the trace")

    prefix = payload.get("native_prefix")
    if not isinstance(prefix, Mapping) or prefix.get("exact") is not True:
        raise CircuitCatalogError("receipt does not prove exact native prefix replay")
    prefix_steps = prefix.get("steps")
    through_event = prefix.get("through_event")
    if not isinstance(through_event, int) or isinstance(through_event, bool):
        raise CircuitCatalogError("receipt native prefix has no integer through_event")
    if not isinstance(prefix_steps, list) or len(prefix_steps) != through_event + 1:
        raise CircuitCatalogError("receipt native prefix contains a non-exact step")
    expected_first_producer = min(trace["values"][site]["producer"] for site in sites)
    if through_event != expected_first_producer:
        raise CircuitCatalogError("receipt native prefix ends at the wrong producer event")
    for index, row in enumerate(prefix_steps):
        if (
            not isinstance(row, Mapping)
            or row.get("pc") != index
            or row.get("operator") != trace["events"][index]["operator"]
            or row.get("exact_outputs") is not True
        ):
            raise CircuitCatalogError("receipt native prefix does not match the frozen trace")

    reuse = payload.get("replay_reuse")
    if reuse is not None:
        from saturn_pub.values import describe

        if not isinstance(reuse, Mapping) or reuse.get("schema") != "mdb-native-prefix-reuse-v1":
            raise CircuitCatalogError("unknown native prefix reuse contract")
        cut = reuse.get("state_cut_manifest")
        cached_through = reuse.get("through_event")
        if (not isinstance(cut, Mapping)
                or _digest(cut) != reuse.get("state_cut_fingerprint")
                or cut.get("schema") != "saturn-pub-statecut-v2"
                or cut.get("boundary") != "mdb.native-aten.produced-prefix-v1"
                or cut.get("parent") != parent_fingerprint
                or cut.get("model_identity") != context["model_identity"]
                or cut.get("execution") != context["execution"]):
            raise CircuitCatalogError("prepared native StateCut identity differs")
        if (not isinstance(cached_through, int) or isinstance(cached_through, bool)
                or not 0 <= cached_through <= expected_first_producer
                or reuse.get("prefix_operators_reused") != cached_through + 1
                or reuse.get("operators_executed") != len(trace["events"]) - cached_through - 1):
            raise CircuitCatalogError("prepared native work accounting differs")
        if cut.get("execution_point") != describe({"through_event":cached_through}):
            raise CircuitCatalogError("prepared native cut clock differs from claimed reuse")
        cut_payload = cut.get("payload", {})
        if not isinstance(cut_payload, Mapping) or cut_payload.get("kind") != "mapping":
            raise CircuitCatalogError("prepared native cut payload is missing")
        fields = {r["key"].get("value"):r["value"] for r in cut_payload.get("items", [])}
        for key,expected in {"trace_fingerprint":context["trace_fingerprint"],
                             "next_event":cached_through+1,"root_fingerprints":root_rows}.items():
            if fields.get(key) != describe(expected):
                raise CircuitCatalogError("prepared native cut payload binding differs")

    suffix_steps = payload.get("suffix_observations")
    expected_suffix_count = len(trace["events"]) - expected_first_producer - 1
    if not isinstance(suffix_steps, list) or len(suffix_steps) != expected_suffix_count:
        raise CircuitCatalogError("receipt native suffix does not cover the frozen trace")
    for offset, row in enumerate(suffix_steps, start=expected_first_producer + 1):
        if (
            not isinstance(row, Mapping)
            or row.get("pc") != offset
            or row.get("operator") != trace["events"][offset]["operator"]
        ):
            raise CircuitCatalogError("receipt native suffix does not match the frozen trace")

    receipt_sites = payload.get("intervention_sites")
    if not isinstance(receipt_sites, list) or len(receipt_sites) != len(sites):
        raise CircuitCatalogError("receipt intervention site count differs from the plan")
    by_value = {}
    for row in receipt_sites:
        if not isinstance(row, Mapping) or row.get("value_id") in by_value:
            raise CircuitCatalogError("receipt intervention sites are malformed or duplicated")
        by_value[row["value_id"]] = row
    if set(by_value) != set(sites):
        raise CircuitCatalogError("receipt intervention sites differ from the plan")
    for value_id in sites:
        row = by_value[value_id]
        if row.get("producer_event") != trace["values"][value_id]["producer"]:
            raise CircuitCatalogError(f"receipt site {value_id} points at the wrong producer")
        try:
            observed_dose = float(row["dose"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CircuitCatalogError(f"receipt site {value_id} has no numeric dose") from exc
        if observed_dose != doses[value_id]:
            raise CircuitCatalogError(f"receipt dose for {value_id} differs from the plan")
        if row.get("replacement_digest") != replacement_digests[value_id]:
            raise CircuitCatalogError(f"receipt replacement digest for {value_id} differs from the plan")
        act = row.get("act")
        operation = act.get("operation") if isinstance(act, Mapping) else None
        if not isinstance(operation, Mapping) or operation.get("id") != "saturn.builtin.replace":
            raise CircuitCatalogError(f"receipt site {value_id} lacks the public replace Act")

    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("intervention") != "declared-ssa-value-replacement":
        raise CircuitCatalogError("receipt lacks the declared SSA replacement authority")
    if authority.get("causal_effect") != "not-established":
        raise CircuitCatalogError("receipt has an unsupported causal authority claim")
    if not isinstance(payload.get("alias_scope"), str):
        raise CircuitCatalogError("receipt lacks an alias-scope declaration")

    differences = payload.get("output_differences")
    if not isinstance(differences, list):
        raise CircuitCatalogError("receipt lacks native output observations")
    candidate_digests = {}
    for row in differences:
        if not isinstance(row, Mapping) or row.get("path") in candidate_digests:
            raise CircuitCatalogError("receipt output observations are malformed or duplicated")
        if "candidate_digest" not in row:
            raise CircuitCatalogError("receipt output observation lacks candidate digest")
        candidate_digests[row["path"]] = row["candidate_digest"]
    actual_digests = _branch_output_leaf_digests(output)
    if candidate_digests != actual_digests:
        raise CircuitCatalogError("receipt output claims do not match the actual backend output")
Backend = Callable[..., Any]


class CircuitCatalog:
    """Persistent, append-only catalogue of circuits, plans, and outcomes."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = None if path is None else Path(path)
        self._records: dict[str, CircuitRecord] = {}
        self._plans: dict[str, CircuitPlan] = {}
        self._outcomes: dict[str, CircuitOutcome] = {}
        self._backends: dict[str, _BackendRegistration] = {}
        self._traces: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            self._load(self.path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CircuitCatalog":
        return cls(path)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CircuitCatalog":
        instance = cls()
        instance._load_payload(payload)
        return instance

    def _load(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CircuitCatalogError(f"cannot load circuit catalogue {path}: {exc}") from exc
        self._load_payload(payload)

    def _load_payload(self, payload: Mapping[str, Any]) -> None:
        sealed = _sealed(payload, CIRCUIT_CATALOG_SCHEMA, "circuit catalog")
        for raw in sealed["records"]:
            record = CircuitRecord(raw)
            if record.circuit_id in self._records:
                raise CircuitCatalogError("catalog contains duplicate circuit IDs")
            self._records[record.circuit_id] = record
        for raw in sealed["plans"]:
            plan = CircuitPlan(raw)
            if plan.plan_id in self._plans:
                raise CircuitCatalogError("catalog contains duplicate plan IDs")
            self._plans[plan.plan_id] = plan
        for raw in sealed["outcomes"]:
            outcome = CircuitOutcome(raw)
            if outcome.outcome_id in self._outcomes:
                raise CircuitCatalogError("catalog contains duplicate outcome IDs")
            self._outcomes[outcome.outcome_id] = outcome
        record_ids = set(self._records)
        for plan in self._plans.values():
            if plan.to_dict()["circuit_id"] not in record_ids:
                raise CircuitCatalogError("plan references an unknown circuit")
        for outcome in self._outcomes.values():
            if outcome.to_dict()["plan_id"] not in self._plans:
                raise CircuitCatalogError("outcome references an unknown plan")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "records": [self._records[key].to_dict() for key in sorted(self._records)],
            "plans": [self._plans[key].to_dict() for key in sorted(self._plans)],
            "outcomes": [self._outcomes[key].to_dict() for key in sorted(self._outcomes)],
        }
        return _seal(body, CIRCUIT_CATALOG_SCHEMA, "circuit catalog")

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise CircuitCatalogError("catalog save requires a path")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        self.path = target
        return target

    def attach_trace(self, trace: Mapping[str, Any]) -> str:
        frozen = _validate_trace(trace)
        fingerprint = frozen["fingerprint"]
        self._traces[fingerprint] = frozen
        return fingerprint

    def register_backend(
        self,
        name: str,
        callback: Backend,
        *,
        verifier: CircuitBackendVerifier | None = None,
        contract: str = "generic",
    ) -> None:
        """Register a callback with an explicit evidence contract.

        ``execute`` refuses an unverified backend.  A generic callback cannot
        be inspected for native execution semantics, so callers must provide a
        backend-owned verifier.  The public branch engine can use
        ``verifier=verify_branch_forward_result`` and an explicit contract.
        """

        name = _text(name, "backend name")
        if not callable(callback):
            raise CircuitCatalogError("backend callback must be callable")
        contract = _text(contract, "backend contract")
        if verifier is not None and not callable(verifier):
            raise CircuitCatalogError("backend verifier must be callable")
        if name in self._backends:
            raise CircuitCatalogError(f"backend already registered: {name}")
        self._backends[name] = _BackendRegistration(callback, verifier, contract)

    # Friendly alias for callers that treat backends as execution adapters.
    register_executor = register_backend

    def register(
        self,
        record: CircuitRecord | Mapping[str, Any] | None = None,
        *,
        circuit_id: str | None = None,
        name: str | None = None,
        trace: Mapping[str, Any] | None = None,
        table: Any = None,
        model_identity: Any = None,
        execution: Any = None,
        input_identity: Any = None,
        route_candidates: Sequence[Mapping[str, Any]] = (),
        roles: Any = None,
        evidence: Sequence[Mapping[str, Any]] = (),
        supported_interventions: Sequence[Mapping[str, Any]] = (),
        predictions: Any = (),
        replay_handle: Mapping[str, Any] | str | None = None,
        persist: bool = True,
    ) -> CircuitRecord:
        """Register one sealed record or derive one from a frozen native trace."""

        if record is not None and any(value is not None for value in (trace, table, circuit_id)):
            raise CircuitCatalogError("record cannot be combined with trace/table registration arguments")
        if record is not None:
            result = record if isinstance(record, CircuitRecord) else CircuitRecord(record)
        else:
            if table is None and trace is None:
                raise CircuitCatalogError("register requires a CircuitRecord, table, or frozen trace")
            adapter = TraceCircuitAdapter(table if table is not None else trace)
            if circuit_id is None:
                raise CircuitCatalogError("trace registration requires circuit_id")
            result = adapter.to_record(
                circuit_id=circuit_id, name=name, model_identity=model_identity,
                execution=execution, input_identity=input_identity,
                route_candidates=route_candidates, roles=roles, evidence=evidence,
                supported_interventions=supported_interventions, predictions=predictions,
                replay_handle=replay_handle,
            )
            self._traces[result.trace_fingerprint] = adapter.trace
        existing = self._records.get(result.circuit_id)
        if existing is not None:
            if existing.fingerprint != result.fingerprint:
                if any(item.to_dict()["circuit_id"] == result.circuit_id
                       for item in self._plans.values()):
                    raise CircuitCatalogError("cannot change a circuit after a plan exists")
                raise CircuitCatalogError(f"circuit already registered with another seal: {result.circuit_id}")
            return existing
        self._records[result.circuit_id] = result
        if persist and self.path is not None:
            self.save()
        return result

    add = register

    def list(
        self,
        *,
        model_identity: Any = None,
        trace_fingerprint: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for circuit_id in sorted(self._records):
            record = self._records[circuit_id]
            body = record.to_dict()
            if model_identity is not None and body["binding"]["model_identity"] != model_identity:
                continue
            if trace_fingerprint is not None and body["trace"]["fingerprint"] != trace_fingerprint:
                continue
            if status is not None and body["statuses"].get(status) != status:
                continue
            related_plans = [p for p in self._plans.values() if p.to_dict()["circuit_id"] == circuit_id]
            outcome_count = sum(
                1 for outcome in self._outcomes.values()
                if outcome.to_dict()["circuit_id"] == circuit_id
            )
            rows.append({
                "circuit_id": circuit_id,
                "name": body["name"],
                "fingerprint": record.fingerprint,
                "trace_fingerprint": record.trace_fingerprint,
                "context_fingerprint": body["binding"]["context_fingerprint"],
                "route_count": len(body["routes"]),
                "plan_count": len(related_plans),
                "outcome_count": outcome_count,
                "statuses": body["statuses"],
            })
        return rows

    def show(self, circuit_id: str, *, include_history: bool = True) -> dict[str, Any]:
        record = self._records.get(circuit_id)
        if record is None:
            raise CircuitCatalogError(f"unknown circuit: {circuit_id}")
        result = record.to_dict()
        if include_history:
            result["plans"] = [
                plan.to_dict() for plan in self._plans.values()
                if plan.to_dict()["circuit_id"] == circuit_id
            ]
            result["outcomes"] = [
                outcome.to_dict() for outcome in self._outcomes.values()
                if outcome.to_dict()["circuit_id"] == circuit_id
            ]
        return result

    def _resolve_plan(self, plan: CircuitPlan | Mapping[str, Any] | str) -> CircuitPlan:
        if isinstance(plan, str):
            result = self._plans.get(plan)
            if result is None:
                raise CircuitCatalogError(f"unknown plan: {plan}")
            return result
        result = plan if isinstance(plan, CircuitPlan) else CircuitPlan(plan)
        existing = self._plans.get(result.plan_id)
        if existing is None or existing.fingerprint != result.fingerprint:
            raise CircuitCatalogError("plan is not the sealed plan registered in this catalog")
        return existing

    def plan(
        self,
        circuit_id: str,
        *,
        route_id: str | None = None,
        intervention: Mapping[str, Any] | str | None = None,
        prediction: Mapping[str, Any] | str | None = None,
        branch_id: str | None = None,
        parent_plan_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        plan_id: str | None = None,
        persist: bool = True,
    ) -> CircuitPlan:
        record = self._records.get(circuit_id)
        if record is None:
            raise CircuitCatalogError(f"unknown circuit: {circuit_id}")
        record_body = record.to_dict()
        binding = _context_from_mapping(context, record_body["binding"]) if context else record.binding
        route = record.route(route_id)
        route_id = route["route_id"]
        if intervention is None:
            intervention_body: dict[str, Any] = {}
        elif isinstance(intervention, str):
            matches = [item for item in record_body["supported_interventions"]
                      if item.get("intervention_id", item.get("id")) == intervention]
            if len(matches) != 1:
                raise CircuitCatalogError(f"unknown intervention: {intervention}")
            intervention_body = _copy_json(matches[0], "intervention")
        else:
            intervention_body = _copy_json(intervention, "intervention")
        self._validate_intervention(record_body, intervention_body)
        available = [item for item in record_body["predictions"] if item["route_id"] == route_id]
        if prediction is None:
            prediction_body = available[0] if len(available) == 1 else None
        elif isinstance(prediction, str):
            matches = [item for item in available if item["prediction_id"] == prediction]
            if len(matches) != 1:
                raise CircuitCatalogError(f"unknown prediction: {prediction}")
            prediction_body = matches[0]
        else:
            prediction_body = _copy_json(prediction, "prediction")
        if prediction_body is not None:
            if prediction_body.get("context_fingerprint") != binding["context_fingerprint"]:
                raise CircuitCatalogError("prediction context is stale for this plan")
            prediction_body = _copy_json(prediction_body, "prediction")
        body = {
            "circuit_id": circuit_id,
            "route_id": route_id,
            "binding": binding,
            "intervention": intervention_body,
            "prediction": prediction_body,
            "prediction_status": "sealed-prediction" if prediction_body else "none-recorded",
            "branch_id": _text(branch_id or f"branch:{route_id}", "branch_id"),
            "parent_plan_id": parent_plan_id,
            "metadata": _copy_json(metadata or {}, "plan.metadata"),
        }
        if plan_id is None:
            plan_id = f"plan:{_digest(body, 'circuit plan')[:32]}"
        body["plan_id"] = _text(plan_id, "plan_id")
        result = CircuitPlan(_seal(body, CIRCUIT_PLAN_SCHEMA, "circuit plan"))
        existing = self._plans.get(result.plan_id)
        if existing is not None:
            if existing.fingerprint != result.fingerprint:
                raise CircuitCatalogError("cannot change a plan after it has been sealed")
            return existing
        self._plans[result.plan_id] = result
        if persist and self.path is not None:
            self.save()
        return result

    def _validate_intervention(self, record: Mapping[str, Any], intervention: Mapping[str, Any]) -> None:
        if not intervention:
            return
        declared_id = intervention.get("intervention_id", intervention.get("id"))
        supported = record["supported_interventions"]
        if supported and declared_id is None:
            raise CircuitCatalogError("intervention must name one supported intervention")
        if supported and declared_id is not None:
            supported_ids = {item.get("intervention_id", item.get("id")) for item in supported}
            if declared_id not in supported_ids:
                raise CircuitCatalogError(f"intervention {declared_id!r} is not supported")
            canonical = next(item for item in supported
                             if item.get("intervention_id", item.get("id")) == declared_id)
        else:
            canonical = None
        sites = _site_ids(intervention, required=True)
        if canonical is not None:
            canonical_sites = _site_ids(canonical)
            if canonical_sites and set(sites) != set(canonical_sites):
                raise CircuitCatalogError("intervention sites differ from the supported declaration")
        _site_doses(intervention, sites)
        if "replacement_digests" in intervention:
            _replacement_digests(intervention, sites)
        trace = self._traces.get(record["trace"]["fingerprint"])
        if trace is not None:
            values = _value_ids(trace)
            for value_id in sites:
                if value_id not in values:
                    raise CircuitCatalogError(f"intervention references unknown SSA value {value_id!r}")
                if trace["values"][value_id]["producer"] is None:
                    raise CircuitCatalogError(f"intervention cannot replace root SSA value {value_id!r}")

    def record_outcome(
        self,
        plan: CircuitPlan | Mapping[str, Any] | str,
        outcome: Any = None,
        *,
        receipt: Any = None,
        output: Any = None,
        receipt_digest: str | None = None,
        output_digest: str | None = None,
        status: str = "measured",
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        persist: bool = True,
    ) -> CircuitOutcome:
        selected_plan = self._resolve_plan(plan)
        plan_body = selected_plan.to_dict()
        binding = _context_from_mapping(context, plan_body["binding"]) if context else plan_body["binding"]
        if outcome is not None:
            if isinstance(outcome, CircuitOutcome):
                body = outcome.to_dict()
                if body["plan_id"] != selected_plan.plan_id:
                    raise CircuitCatalogError("outcome belongs to another plan")
                receipt = receipt if receipt is not None else body.get("receipt")
                output_digest = output_digest if output_digest is not None else body.get("output_digest")
                receipt_digest = receipt_digest if receipt_digest is not None else body.get("receipt_digest")
                status = body.get("status", status)
            elif hasattr(outcome, "receipt") and hasattr(outcome, "output"):
                receipt = receipt if receipt is not None else outcome.receipt
                output = output if output is not None else outcome.output
            elif isinstance(outcome, Mapping):
                supplied = outcome
                receipt = receipt if receipt is not None else supplied.get("receipt")
                output = output if output is not None else supplied.get("output")
                receipt_digest = receipt_digest if receipt_digest is not None else supplied.get("receipt_digest")
                output_digest = output_digest if output_digest is not None else supplied.get("output_digest")
                status = supplied.get("status", status)
                if metadata is None:
                    metadata = supplied.get("metadata")
            else:
                raise CircuitCatalogError("outcome must be a mapping, CircuitOutcome, or replay result")
        if status not in {"measured", "replayed", "failed", "refused"}:
            raise CircuitCatalogError("outcome status must be measured, replayed, failed, or refused")
        receipt_payload = None
        if receipt is not None:
            actual_receipt_digest, receipt_payload = _receipt_payload(receipt)
            if receipt_digest is not None and _sha(receipt_digest, "receipt_digest") != actual_receipt_digest:
                raise CircuitCatalogError("receipt_digest does not match the actual public receipt")
            receipt_digest = actual_receipt_digest
        elif receipt_digest is not None:
            receipt_digest = _sha(receipt_digest, "receipt_digest")
        if output is not None:
            actual_output_digest = _output_digest(output)
            if output_digest is not None and _sha(output_digest, "output_digest") != actual_output_digest:
                raise CircuitCatalogError("output_digest does not match the actual public output")
            output_digest = actual_output_digest
        elif output_digest is not None:
            output_digest = _sha(output_digest, "output_digest")
        if receipt_payload is not None:
            for field in ("model_identity", "execution", "input_identity"):
                if field in receipt_payload and receipt_payload[field] != binding[field]:
                    raise CircuitCatalogError(f"receipt {field} does not match the plan")
            if (
                "trace_fingerprint" in receipt_payload
                and receipt_payload["trace_fingerprint"] != binding["trace_fingerprint"]
            ):
                raise CircuitCatalogError("receipt trace fingerprint does not match the plan")
            if output is not None and "output_differences" in receipt_payload:
                differences = receipt_payload["output_differences"]
                if not isinstance(differences, list):
                    raise CircuitCatalogError("receipt output observations are malformed")
                candidate_digests = {}
                for row in differences:
                    if not isinstance(row, Mapping) or row.get("path") in candidate_digests:
                        raise CircuitCatalogError("receipt output observations are malformed")
                    if "candidate_digest" not in row:
                        raise CircuitCatalogError("receipt output observation lacks candidate digest")
                    candidate_digests[row["path"]] = row["candidate_digest"]
                if candidate_digests != _branch_output_leaf_digests(output):
                    raise CircuitCatalogError("receipt output claims do not match the actual output")
        if status in {"measured", "replayed"} and (receipt_digest is None or output_digest is None):
            raise CircuitCatalogError("measured outcomes require actual receipt and output digests")
        prediction = plan_body.get("prediction")
        prediction_status = "unavailable"
        prediction_observed_digest = None
        if prediction is not None and output is not None:
            output_keys = _prediction_output_keys(
                prediction.get("output_keys"), "prediction.output_keys"
            )
            compared_output = output if output_keys is None else _project_output(
                output, output_keys, "measured output"
            )
            prediction_observed_digest = _output_digest(compared_output)
            prediction_status = (
                "matched" if prediction["output_digest"] == prediction_observed_digest else "diverged"
            )
        elif prediction is not None:
            prediction_status = "not-measured"
        body = {
            "plan_id": selected_plan.plan_id,
            "circuit_id": plan_body["circuit_id"],
            "route_id": plan_body["route_id"],
            "binding": binding,
            "status": status,
            "receipt_digest": receipt_digest,
            "output_digest": output_digest,
            "receipt": receipt_payload,
            "prediction_status": prediction_status,
            "prediction_observed_digest": prediction_observed_digest,
            "metadata": _copy_json(metadata or {}, "outcome.metadata"),
        }
        body["outcome_id"] = f"outcome:{_digest(body, 'circuit outcome')[:32]}"
        result = CircuitOutcome(_seal(body, CIRCUIT_OUTCOME_SCHEMA, "circuit outcome"))
        existing = next((item for item in self._outcomes.values()
                         if item.to_dict()["plan_id"] == selected_plan.plan_id), None)
        if existing is not None:
            if existing.outcome_id == result.outcome_id:
                return existing
            raise CircuitCatalogError("cannot change a plan after an outcome has been recorded")
        self._outcomes[result.outcome_id] = result
        if persist and self.path is not None:
            self.save()
        return result

    def execute(
        self,
        plan: CircuitPlan | Mapping[str, Any] | str,
        *,
        backend: str | None = None,
        context: Mapping[str, Any] | None = None,
        trace: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> CircuitOutcome:
        selected_plan = self._resolve_plan(plan)
        plan_body = selected_plan.to_dict()
        binding = _context_from_mapping(context, plan_body["binding"]) if context else plan_body["binding"]
        if trace is not None:
            trace_fp = self.attach_trace(trace)
            if trace_fp != binding["trace_fingerprint"]:
                raise CircuitCatalogError("stale context: supplied trace differs from the plan")
        record = self._records[plan_body["circuit_id"]]
        record_body = record.to_dict()
        handle = record_body.get("replay_handle") or {}
        backend_name = backend or handle.get("backend")
        if backend_name is None:
            raise CircuitCatalogError("plan has no registered replay backend")
        registration = self._backends.get(backend_name)
        if registration is None:
            raise CircuitCatalogError(f"replay backend is not registered: {backend_name}")
        if registration.verifier is None:
            raise CircuitCatalogError(
                f"backend {backend_name!r} has no evidence verifier; generic callbacks cannot be executed"
            )
        frozen = self._traces.get(binding["trace_fingerprint"])
        result = registration.callback(
            plan=selected_plan.to_dict(), record=record_body, context=binding,
            trace=frozen, **kwargs,
        )
        try:
            registration.verifier(
                plan=selected_plan.to_dict(), record=record_body, context=binding,
                trace=frozen, result=result, kwargs=kwargs,
            )
        except CircuitCatalogError:
            raise
        except Exception as exc:  # backend verifiers must fail closed
            raise CircuitCatalogError(f"backend {backend_name!r} evidence verification failed: {exc}") from exc
        return self.record_outcome(selected_plan, result, context=binding)


def _load_json_argument(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdb circuits", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list sealed circuits")
    listing.add_argument("--catalog", required=True, type=Path)
    listing.add_argument("--trace-fingerprint")
    listing.add_argument("--status")
    show = commands.add_parser("show", help="show one circuit and its history")
    show.add_argument("circuit_id")
    show.add_argument("--catalog", required=True, type=Path)
    plan = commands.add_parser("plan", help="seal one immutable branch plan")
    plan.add_argument("circuit_id")
    plan.add_argument("--catalog", required=True, type=Path)
    plan.add_argument("--route-id")
    plan.add_argument("--intervention")
    plan.add_argument("--prediction")
    plan.add_argument("--branch-id")
    plan.add_argument("--context")
    record = commands.add_parser("record", help="record one measured outcome")
    record.add_argument("plan_id")
    record.add_argument("--catalog", required=True, type=Path)
    record.add_argument("--outcome", required=True, help="JSON or @path JSON outcome")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalog = CircuitCatalog.load(args.catalog)
        if args.command == "list":
            result = catalog.list(trace_fingerprint=args.trace_fingerprint, status=args.status)
        elif args.command == "show":
            result = catalog.show(args.circuit_id)
        elif args.command == "plan":
            result = catalog.plan(
                args.circuit_id,
                route_id=args.route_id,
                intervention=None if args.intervention is None else _load_json_argument(args.intervention),
                prediction=None if args.prediction is None else args.prediction,
                branch_id=args.branch_id,
                context=None if args.context is None else _load_json_argument(args.context),
            ).to_dict()
            catalog.save()
        else:
            result = catalog.record_outcome(
                args.plan_id, _load_json_argument(args.outcome)
            ).to_dict()
            catalog.save()
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (CircuitCatalogError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "BRANCH_FORWARD_CONTRACT", "CIRCUIT_CATALOG_SCHEMA", "CIRCUIT_OUTCOME_SCHEMA",
    "CIRCUIT_PLAN_SCHEMA", "CIRCUIT_RECORD_SCHEMA", "CircuitBackend",
    "CircuitBackendVerifier", "CircuitCatalog", "CircuitCatalogError", "CircuitOutcome",
    "CircuitPlan", "CircuitRecord", "TraceCircuitAdapter",
    "verify_branch_forward_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
