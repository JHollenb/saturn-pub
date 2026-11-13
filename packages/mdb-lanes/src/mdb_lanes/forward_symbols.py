"""Sealed, anonymous records of values observed during native forward execution.

The trace records physical inputs, operation outputs and producer/use edges.
It does not assign semantic meaning or replace public SymbolBinding records.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FORWARD_SYMBOLS_SCHEMA = "mdb-forward-symbols-v1"
_VALUE_ID = re.compile(r"^v[0-9]{3,}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRACE_KEYS = {
    "schema", "model_identity", "execution", "input_identity", "values", "events",
    "outputs", "capture_status", "semantic_labels_supplied", "metadata",
}
_VALUE_KEYS = {"shape", "dtype", "sha256", "origin", "producer", "preview"}
_EVENT_KEYS = {
    "sequence", "operator", "module_path", "inputs", "outputs", "arguments", "result", "clock"
}
_SEMANTIC_KEYS = {
    "semantic_label", "semantic_labels", "semantic_name", "semantic_names",
    "provided_symbol", "supplied_symbol", "symbol", "symbols", "gold_label",
    "target_label", "chain_of_thought", "thought",
}
_MAX_TRACE_BYTES, _MAX_METADATA_BYTES = 16 << 20, 64 << 10
_MAX_EVENTS = _MAX_VALUES = _MAX_ITEMS = 100_000
_MAX_DEPTH, _MAX_STRING_BYTES = 32, 16 << 10


class ForwardSymbolsError(ValueError):
    """Raised when a forward trace is malformed or its seal does not match."""


def _canonical(value: Any) -> bytes:
    from saturn_pub.values import canonical

    return canonical(value)


def _digest(value: Any) -> str:
    from saturn_pub.values import digest

    return digest(value)


def _json_copy(value: Any, label: str = "value", depth: int = 0) -> Any:
    """Copy bounded, strict JSON without importing model frameworks."""
    if depth > _MAX_DEPTH:
        raise ForwardSymbolsError(f"{label} exceeds the metadata nesting limit")
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ForwardSymbolsError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > _MAX_STRING_BYTES:
            raise ForwardSymbolsError(f"{label} string is too large")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_ITEMS:
            raise ForwardSymbolsError(f"{label} has too many keys")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ForwardSymbolsError(f"{label} contains a non-text key")
            result[key] = _json_copy(item, f"{label}.{key}", depth + 1)
        return result
    if isinstance(value, list):
        if len(value) > _MAX_ITEMS:
            raise ForwardSymbolsError(f"{label} has too many items")
        return [_json_copy(item, f"{label}[{i}]", depth + 1) for i, item in enumerate(value)]
    raise ForwardSymbolsError(f"{label} contains unsupported {type(value).__name__}")


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ForwardSymbolsError(f"{label} must be non-empty text")
    return value


def _check_metadata(value: Any, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.casefold() in _SEMANTIC_KEYS:
                raise ForwardSymbolsError(f"{path}.{key} is semantic labeling; use a separate provenance record")
            _check_metadata(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_metadata(item, f"{path}[{index}]")


def _refs(tree: Any, known: set[str], label: str, depth: int = 0) -> list[str]:
    """Validate tagged JSON argument/output trees and collect value references."""
    if depth > _MAX_DEPTH:
        raise ForwardSymbolsError(f"{label} exceeds the metadata nesting limit")
    if tree is None or type(tree) in {bool, int, float, str}:
        if type(tree) is float and not math.isfinite(tree):
            raise ForwardSymbolsError(f"{label} contains a non-finite number")
        return []
    if isinstance(tree, list):
        return [ref for item in tree for ref in _refs(item, known, label, depth + 1)]
    if not isinstance(tree, Mapping):
        raise ForwardSymbolsError(f"{label} contains unsupported {type(tree).__name__}")
    for tag in ("$value", "$tuple", "$dtype", "$device", "$torch", "$float"):
        if tag not in tree:
            continue
        if set(tree) != {tag}:
            raise ForwardSymbolsError(f"{label} has a malformed {tag} tag")
        payload = tree[tag]
        if tag == "$value":
            if not isinstance(payload, str) or payload not in known:
                raise ForwardSymbolsError(f"{label} references unknown value {payload!r}")
            return [payload]
        if tag == "$tuple":
            if not isinstance(payload, list):
                raise ForwardSymbolsError(f"{label} $tuple payload must be an array")
            return [ref for item in payload for ref in _refs(item, known, label, depth + 1)]
        if tag == "$float":
            if not isinstance(payload, str) or payload not in {"inf", "-inf", "nan"}:
                raise ForwardSymbolsError(f"{label} $float payload is not a native non-finite scalar")
            return []
        _text(payload, f"{label} {tag}")
        return []
    return [ref for item in tree.values() for ref in _refs(item, known, label, depth + 1)]


def _unique_refs(tree: Any, known: set[str], label: str) -> list[str]:
    """Match the recorder's ordered, deduplicated tensor-reference lists."""
    return list(dict.fromkeys(_refs(tree, known, label)))


def _validate_body(body: Any) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise ForwardSymbolsError("trace must be an object")
    trace = _json_copy(body, "trace")
    if set(trace) != _TRACE_KEYS:
        missing, extra = sorted(_TRACE_KEYS - set(trace)), sorted(set(trace) - _TRACE_KEYS)
        raise ForwardSymbolsError(f"trace fields differ from schema; missing={missing}, extra={extra}")
    if trace["schema"] != FORWARD_SYMBOLS_SCHEMA:
        raise ForwardSymbolsError("unsupported forward symbols schema")
    if any(trace[key] is None for key in ("model_identity", "execution", "input_identity")):
        raise ForwardSymbolsError("model_identity, execution and input_identity are required")
    if trace["capture_status"] != "complete":
        raise ForwardSymbolsError("forward-symbol trace must record a complete capture")
    if trace["semantic_labels_supplied"] is not False:
        raise ForwardSymbolsError("physical forward trace cannot carry supplied semantic labels")
    metadata = trace["metadata"]
    if not isinstance(metadata, dict):
        raise ForwardSymbolsError("metadata must be an object")
    _check_metadata(metadata)
    if len(_canonical(metadata)) > _MAX_METADATA_BYTES:
        raise ForwardSymbolsError("metadata exceeds its byte limit")

    values, events = trace["values"], trace["events"]
    if not isinstance(values, dict) or len(values) > _MAX_VALUES:
        raise ForwardSymbolsError("values must be a bounded object")
    if not isinstance(events, list) or len(events) > _MAX_EVENTS:
        raise ForwardSymbolsError("events must be a bounded array")
    known, visible, defined = set(values), set(), set()
    for value_id, row in values.items():
        if not _VALUE_ID.fullmatch(value_id):
            raise ForwardSymbolsError(f"value ID is not anonymous: {value_id!r}")
        if not isinstance(row, dict) or set(row) != _VALUE_KEYS:
            raise ForwardSymbolsError(f"value {value_id} has unsupported fields")
        if not isinstance(row["shape"], list) or any(type(n) is not int or n < 0 for n in row["shape"]):
            raise ForwardSymbolsError(f"value {value_id} has an invalid shape")
        _text(row["dtype"], f"value {value_id} dtype")
        if not isinstance(row["sha256"], str) or not _SHA256.fullmatch(row["sha256"]):
            raise ForwardSymbolsError(f"value {value_id} requires a lowercase SHA-256 digest")
        origin, producer = row["origin"], row["producer"]
        if origin in {"input", "parameter", "literal"}:
            if producer is not None:
                raise ForwardSymbolsError(f"root value {value_id} cannot claim an operation producer")
            visible.add(value_id)
        elif origin == "operation-output":
            if type(producer) is not int or producer < 0:
                raise ForwardSymbolsError(f"value {value_id} requires a native producer event")
        else:
            raise ForwardSymbolsError(f"value {value_id} has an unsupported origin")
        preview = row["preview"]
        if not isinstance(preview, list) or len(preview) > 8:
            raise ForwardSymbolsError(f"value {value_id} preview must contain at most eight scalars")
        for item in preview:
            if (item is None or type(item) in {bool, int}
                    or (type(item) is float and math.isfinite(item))
                    or (isinstance(item, str) and len(item.encode("utf-8")) <= 256)):
                continue
            raise ForwardSymbolsError(f"value {value_id} preview must contain bounded scalars")

    prior_clock = None
    for sequence, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
            raise ForwardSymbolsError(f"event {sequence} has unsupported fields")
        if type(event["sequence"]) is not int or event["sequence"] != sequence:
            raise ForwardSymbolsError("event sequence must be contiguous from zero")
        operator = _text(event["operator"], f"event {sequence} operator")
        if not operator.startswith(("aten.", "prims.")) or any(char.isspace() for char in operator):
            raise ForwardSymbolsError(f"event {sequence} operator must be a native ATen/prims name")
        _text(event["module_path"], f"event {sequence} module_path", empty=True)
        clock = event["clock"]
        if (not isinstance(clock, dict) or set(clock) != {"forward_call", "operation"}
                or any(type(clock[key]) is not int or clock[key] < 0 for key in clock)):
            raise ForwardSymbolsError(f"event {sequence} has an invalid native clock")
        clock_pair = (clock["forward_call"], clock["operation"])
        if prior_clock is not None and clock_pair <= prior_clock:
            raise ForwardSymbolsError("native event clocks must increase in execution order")
        prior_clock = clock_pair
        inputs, outputs = event["inputs"], event["outputs"]
        if (not isinstance(inputs, list) or any(not isinstance(x, str) for x in inputs)
                or not isinstance(outputs, list) or any(not isinstance(x, str) for x in outputs)):
            raise ForwardSymbolsError(f"event {sequence} inputs and outputs must be ID arrays")
        for value_id in inputs:
            if value_id not in known:
                raise ForwardSymbolsError(f"event {sequence} reads unknown value {value_id!r}")
            if value_id not in visible:
                raise ForwardSymbolsError(f"event {sequence} reads {value_id!r} before its native producer")
        argument_refs = _unique_refs(event["arguments"], known, f"event {sequence} arguments")
        if len(inputs) != len(set(inputs)) or set(argument_refs) != set(inputs):
            raise ForwardSymbolsError(f"event {sequence} argument references do not match native inputs")
        result_refs = _unique_refs(event["result"], known, f"event {sequence} result")
        if set(result_refs) != set(outputs):
            raise ForwardSymbolsError(f"event {sequence} result references do not match native outputs")
        if len(outputs) != len(set(outputs)):
            raise ForwardSymbolsError(f"event {sequence} defines a duplicate value ID")
        for value_id in outputs:
            if value_id not in known:
                raise ForwardSymbolsError(f"event {sequence} writes unknown value {value_id!r}")
            row = values[value_id]
            if row["origin"] != "operation-output" or row["producer"] != sequence or value_id in defined:
                raise ForwardSymbolsError(f"event {sequence} output producer identity is inconsistent")
            defined.add(value_id)
            visible.add(value_id)

    literal_roots = {key for key, row in values.items() if row["origin"] == "literal"}
    literal_uses = {
        value_id: [event for event in events if value_id in event["inputs"]]
        for value_id in literal_roots
    }
    if any(
        len(uses) != 1
        or uses[0]["operator"] not in {"aten.lift_fresh.default", "aten.lift_fresh_copy.default"}
        for uses in literal_uses.values()
    ):
        raise ForwardSymbolsError("literal roots must originate only in native lift_fresh operations")

    expected = {key for key, row in values.items() if row["origin"] == "operation-output"}
    if defined != expected:
        raise ForwardSymbolsError("operation-output values do not match native event definitions")
    _refs(trace["outputs"], known, "outputs")
    if len(_canonical(trace)) > _MAX_TRACE_BYTES:
        raise ForwardSymbolsError("trace exceeds its byte limit")
    return trace


def seal_trace(body: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and seal a recorder-produced forward trace."""
    trace = _validate_body(body)
    return {**trace, "fingerprint": _digest(trace)}


def verify_trace(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the seal and native producer-before-use closure."""
    if not isinstance(value, Mapping) or "fingerprint" not in value:
        raise ForwardSymbolsError("sealed trace requires a fingerprint")
    fingerprint = value["fingerprint"]
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise ForwardSymbolsError("trace fingerprint must be a lowercase SHA-256 digest")
    trace = _validate_body({key: item for key, item in value.items() if key != "fingerprint"})
    if fingerprint != _digest(trace):
        raise ForwardSymbolsError("forward-symbol trace fingerprint mismatch")
    return {
        "schema": FORWARD_SYMBOLS_SCHEMA, "fingerprint": fingerprint,
        "event_count": len(trace["events"]), "value_count": len(trace["values"]),
        "passed": True, "authority": "read-only-native-forward-trace",
        "semantic_labels_supplied": False,
    }


class ForwardSymbolTable:
    """Stepwise view over a sealed anonymous native forward trace."""

    def __init__(self, trace: Mapping[str, Any]):
        verify_trace(trace)
        self.trace = json.loads(_canonical(dict(trace)))
        self._events = {event["sequence"]: event for event in self.trace["events"]}
        self._uses = {value_id: [] for value_id in self.trace["values"]}
        for event in self.trace["events"]:
            for value_id in event["inputs"]:
                self._uses[value_id].append(event["sequence"])

    def at(self, step: int) -> dict[str, Any]:
        """Return the event, model outputs, definitions, uses and values visible after step."""
        if type(step) is not int or step not in self._events:
            raise ForwardSymbolsError(f"step must identify a captured event: {step!r}")
        event = self._events[step]
        visible = {
            key: row for key, row in self.trace["values"].items()
            if row["origin"] in {"input", "parameter", "literal"} or row["producer"] <= step
        }
        return {
            "schema": FORWARD_SYMBOLS_SCHEMA, "fingerprint": self.trace["fingerprint"],
            "step": step, "event": _json_copy(event), "outputs": _json_copy(self.trace["outputs"]),
            "visible_values": _json_copy(visible), "defs": list(event["outputs"]),
            "uses": list(event["inputs"]),
            "current_outputs": [
                {"value_id": key, "value": _json_copy(self.trace["values"][key])}
                for key in event["outputs"]
            ],
        }

    def inspect(self, value_id: str) -> dict[str, Any]:
        """Inspect one value with native uses and its producer ancestry."""
        row = self.trace["values"].get(value_id)
        if row is None:
            raise ForwardSymbolsError(f"unknown value ID: {value_id}")
        producer = row["producer"]
        ancestry, pending = set(), [] if producer is None else [producer]
        while pending:
            sequence = pending.pop()
            if sequence in ancestry:
                continue
            ancestry.add(sequence)
            for input_id in self._events[sequence]["inputs"]:
                parent = self.trace["values"][input_id]["producer"]
                if parent is not None:
                    pending.append(parent)
        return {
            "schema": FORWARD_SYMBOLS_SCHEMA, "fingerprint": self.trace["fingerprint"],
            "value_id": value_id, "value": _json_copy(row),
            "producer_event": None if producer is None else _json_copy(self._events[producer]),
            "defs": [] if producer is None else [producer],
            "uses": list(self._uses[value_id]), "event_ancestry": sorted(ancestry),
        }

    def format_step(self, step: int) -> str:
        """Render operation, module, and physical value previews for one step."""
        view, event = self.at(step), self._events[step]
        clock, values = event["clock"], view["visible_values"]
        lines = [
            f"step {step}: {event['operator']}  @ {event['module_path'] or '<root>'}",
            f"clock: forward_call={clock['forward_call']} operation={clock['operation']}",
            "value      role   shape        dtype          preview",
        ]
        for role, identifiers in (("read", view["uses"]), ("write", view["defs"])):
            for value_id in identifiers:
                row = values[value_id]
                shape = json.dumps(row["shape"], separators=(",", ":"))
                preview = json.dumps(row["preview"], ensure_ascii=True, separators=(",", ":"))
                lines.append(f"{value_id:<10} {role:<6} {shape:<12} {row['dtype']:<14} {preview}")
        return "\n".join(lines)

    def format_value(self, value_id: str) -> str:
        """Render one value's physical descriptor and native producer/use edges."""
        view = self.inspect(value_id)
        row = view["value"]
        lines = [
            f"{value_id}: {row['origin']}  shape={json.dumps(row['shape'], separators=(',', ':'))}  "
            f"dtype={row['dtype']}  preview={json.dumps(row['preview'], separators=(',', ':'))}",
            f"sha256: {row['sha256']}",
        ]
        producer = view["producer_event"]
        if producer is None:
            lines.append("producer: declared root")
        else:
            lines.append(
                f"producer: event {producer['sequence']} {producer['operator']}  "
                f"@ {producer['module_path'] or '<root>'}"
            )
        lines.append(f"uses: {view['uses']}")
        lines.append(f"event ancestry: {view['event_ancestry']}")
        return "\n".join(lines)

    def list_steps(self) -> list[dict[str, Any]]:
        return [
            {key: _json_copy(event[key]) for key in (
                "sequence", "operator", "module_path", "inputs", "outputs", "clock"
            )}
            for event in self.trace["events"]
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a sealed anonymous native forward trace")
    parser.add_argument("trace", type=Path, help="JSON artifact sealed as mdb-forward-symbols-v1")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--step", type=int, help="render one zero-based operation step")
    action.add_argument("--value", help="inspect one anonymous value ID")
    action.add_argument("--list", action="store_true", help="list captured native operations")
    action.add_argument("--readable", action="store_true", help="render mathematical expressions and evidence annotations")
    action.add_argument("--algebra", action="store_true", help="derive structural algebra candidates from the executed graph")
    parser.add_argument("--annotations", type=Path, help="separate trace-bound interpretation/evidence document")
    parser.add_argument("--scope", help="module-path filter for readable text")
    parser.add_argument("--output-path", help="JSON-pointer output to trace backward in readable view")
    parser.add_argument("--json", action="store_true", help="print the selected view as JSON")
    args = parser.parse_args(argv)
    try:
        table = ForwardSymbolTable(json.loads(args.trace.read_text(encoding="utf-8")))
        if (args.annotations or args.scope or args.output_path) and not args.readable:
            parser.error("--annotations, --scope and --output-path require --readable")
        if args.algebra:
            from .forward_algebra import acquire_symbol_candidates
            from .forward_readable import ForwardReadableProgram

            result = acquire_symbol_candidates(ForwardReadableProgram(table.trace))
            if not args.json:
                lines = [f"algebra candidates for {table.trace['fingerprint']}"]
                lines.extend(f"identity {row['id']}: {row.get('expression', row['kind'])}"
                             for row in result["structural_identities"])
                lines.extend(f"candidate {row['id']}: {row['kind']} events={row['events']}"
                             for row in result["graph_candidates"])
                lines.append("Behavioral names require separate native-consumer evidence.")
                result = "\n".join(lines)
        elif args.readable:
            from .forward_readable import ForwardReadableProgram

            annotations = json.loads(args.annotations.read_text()) if args.annotations else None
            reader = ForwardReadableProgram(table.trace, annotations)
            if args.json:
                if args.scope:
                    parser.error("--scope filters readable text; use the JSON program's module groups")
                result = reader.backward_slice(output_path=args.output_path) if args.output_path else reader.program()
            else:
                result = reader.render(scope=args.scope, output_path=args.output_path)
        elif args.step is not None:
            result = table.at(args.step) if args.json else table.format_step(args.step)
        elif args.value is not None:
            result = table.inspect(args.value) if args.json else table.format_value(args.value)
        else:
            result = table.list_steps()
            if not args.json:
                result = "\n".join(
                    f"{row['sequence']:>5}  {row['operator']}  @ {row['module_path'] or '<root>'}  "
                    f"{row['inputs']} -> {row['outputs']}" for row in result
                ) or "(no native operations captured)"
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


__all__ = [
    "FORWARD_SYMBOLS_SCHEMA", "ForwardSymbolsError", "ForwardSymbolTable", "seal_trace", "verify_trace",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI smoke path
    raise SystemExit(main())
