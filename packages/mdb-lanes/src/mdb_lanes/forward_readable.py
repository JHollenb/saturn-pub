"""Human-readable views over sealed, anonymous native forward traces.

This module translates only recorded operators and value-reference structure.
It does not infer names for values from shapes, module paths, previews, or use
patterns. Optional external claims remain separate, explicitly scoped,
trace-fingerprint-bound annotations.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from .forward_symbols import ForwardSymbolsError, verify_trace

FORWARD_READABLE_SCHEMA = "mdb-forward-readable-v1"
FORWARD_ANNOTATIONS_SCHEMA = "mdb-forward-annotations-v1"

_ANNOTATION_KEYS = {"id", "scope", "claim", "provenance", "evidence_refs"}
_ANNOTATION_DOCUMENT_KEYS = {"schema", "trace_fingerprint", "annotations"}
_MAX_ANNOTATIONS = 10_000
_MAX_ANNOTATION_TEXT_BYTES = 16 << 10
_MAX_EVIDENCE_REFS = 128

_INFIX = {
    "aten.add.Tensor": "+", "aten.add.Scalar": "+",
    "aten.sub.Tensor": "-", "aten.sub.Scalar": "-",
    "aten.mul.Tensor": "*", "aten.mul.Scalar": "*",
    "aten.div.Tensor": "/", "aten.div.Scalar": "/",
    "aten.floor_divide.default": "//", "aten.remainder.Scalar": "%",
    "aten.matmul.default": "@", "aten.mm.default": "@", "aten.bmm.default": "@",
}
_UNARY = {
    "aten.abs.default": "abs", "aten.cos.default": "cos", "aten.erf.default": "erf",
    "aten.exp.default": "exp", "aten.expm1.default": "expm1", "aten.gelu.default": "gelu",
    "aten.log.default": "log", "aten.log1p.default": "log1p", "aten.neg.default": "neg",
    "aten.relu.default": "relu", "aten.rsqrt.default": "rsqrt", "aten.sigmoid.default": "sigmoid",
    "aten.silu.default": "silu", "aten.softplus.default": "softplus", "aten.sqrt.default": "sqrt",
    "aten.tanh.default": "tanh",
}
_FUNCTIONS = {
    "aten.argmax.default": "argmax", "aten.clamp.default": "clamp",
    "aten.clamp_min.default": "clamp_min", "aten.cat.default": "cat",
    "aten.chunk.default": "chunk", "aten.flatten.using_ints": "flatten",
    "aten.gather.default": "gather", "aten.index.Tensor": "index",
    "aten.layer_norm.default": "layer_norm", "aten.linear.default": "linear",
    "aten.max.dim": "max", "aten.mean.default": "mean", "aten.mean.dim": "mean",
    "aten.native_layer_norm.default": "native_layer_norm", "aten.permute.default": "permute",
    "aten.prod.default": "prod", "aten.reshape.default": "reshape", "aten.select.int": "select",
    "aten.slice.Tensor": "slice", "aten.softmax.int": "softmax", "aten.stack.default": "stack",
    "aten.sum.default": "sum", "aten.sum.dim_IntList": "sum", "aten.t.default": "t",
    "aten.transpose.int": "transpose", "aten.unsqueeze.default": "unsqueeze",
    "aten.view.default": "view", "aten.where.self": "where",
    "aten.group_norm.default": "group_norm", "aten.batch_norm.default": "batch_norm",
}


class ForwardReadableError(ValueError):
    """Raised when a readable view or annotation scope is invalid."""


def _refs(tree: Any) -> list[str]:
    if isinstance(tree, list):
        return [ref for item in tree for ref in _refs(item)]
    if not isinstance(tree, Mapping):
        return []
    if set(tree) == {"$value"}:
        return [tree["$value"]]
    if set(tree) == {"$tuple"}:
        return _refs(tree["$tuple"])
    return [ref for item in tree.values() for ref in _refs(item)]


def _escape_pointer_part(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def _output_refs(
    tree: Any, path: str = "", provenance: str = "structural-observation"
) -> list[tuple[str, str, str]]:
    """Return tensor output references with paths and name provenance."""
    if isinstance(tree, Mapping):
        if set(tree) == {"$value"}:
            return [(tree["$value"], path, provenance)]
        if set(tree) == {"$tuple"}:
            result = []
            for index, item in enumerate(tree["$tuple"]):
                result.extend(_output_refs(item, f"{path}/{index}", provenance))
            return result
        result = []
        for key, item in tree.items():
            result.extend(_output_refs(
                item, f"{path}/{_escape_pointer_part(key)}", "source-name"
            ))
        return result
    if isinstance(tree, list):
        result = []
        for index, item in enumerate(tree):
            result.extend(_output_refs(item, f"{path}/{index}", provenance))
        return result
    return []


def _pointer_path(path: str) -> str:
    return path or "<root>"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "None"
    if type(value) is bool:
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, Mapping):
        if set(value) == {"$value"}:
            return str(value["$value"])
        if set(value) == {"$tuple"}:
            return "(" + ", ".join(_format_scalar(item) for item in value["$tuple"]) + ")"
        if set(value) == {"$float"}:
            return value["$float"]
        if set(value) in ({"$dtype"}, {"$device"}, {"$torch"}):
            return str(next(iter(value.values())))
        return "{" + ", ".join(
            f"{json.dumps(key)}: {_format_scalar(item)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    return repr(value)


def _unpack_arguments(event: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Return positional args, kwargs, and direct named args for recorder/test forms."""
    arguments = event["arguments"]
    if isinstance(arguments, Mapping) and "args" in arguments and "kwargs" in arguments:
        packed = arguments.get("args")
        if isinstance(packed, Mapping) and set(packed) == {"$tuple"}:
            args = list(packed["$tuple"])
        elif isinstance(packed, list):
            args = list(packed)
        elif packed is None:
            args = []
        else:
            args = [packed]
        kwargs = arguments.get("kwargs") or {}
        return args, dict(kwargs), {}
    return [], {}, dict(arguments) if isinstance(arguments, Mapping) else {}


def _argument(event: Mapping[str, Any], index: int, name: str, default: Any = None) -> Any:
    args, kwargs, named = _unpack_arguments(event)
    if name in named:
        return named[name]
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def _call_arguments(event: Mapping[str, Any]) -> list[str]:
    args, kwargs, named = _unpack_arguments(event)
    if named:
        return [f"{key}={_format_scalar(value)}" for key, value in named.items()]
    rendered = [_format_scalar(value) for value in args]
    rendered.extend(f"{key}={_format_scalar(value)}" for key, value in kwargs.items())
    return rendered


def _event_expression(event: Mapping[str, Any]) -> str:
    """Render an exact operator-level formula; unknown operators keep their native name."""
    operator = event["operator"]
    if operator in _INFIX:
        left = _argument(event, 0, "self", event["inputs"][0] if event["inputs"] else "?")
        right = _argument(event, 1, "other", event["inputs"][1] if len(event["inputs"]) > 1 else "?")
        lhs, rhs = _format_scalar(left), _format_scalar(right)
        alpha = _argument(event, 2, "alpha", 1)
        if operator.startswith("aten.add.") and alpha != 1:
            rhs = f"{_format_scalar(alpha)} * {rhs}"
        elif operator.startswith("aten.sub.") and alpha != 1:
            rhs = f"{_format_scalar(alpha)} * {rhs}"
        return f"({lhs} {_INFIX[operator]} {rhs})"
    if operator == "aten.addmm.default":
        bias = _argument(event, 0, "self", "?")
        left = _argument(event, 1, "mat1", "?")
        right = _argument(event, 2, "mat2", "?")
        beta = _argument(event, 3, "beta", 1)
        alpha = _argument(event, 4, "alpha", 1)
        product = f"({_format_scalar(left)} @ {_format_scalar(right)})"
        terms = []
        if beta != 0:
            factor = "" if beta == 1 else f"{_format_scalar(beta)} * "
            terms.append(f"{factor}{_format_scalar(bias)}")
        if alpha != 0:
            factor = "" if alpha == 1 else f"{_format_scalar(alpha)} * "
            terms.append(f"{factor}{product}")
        return "(" + " + ".join(terms or ["0"]) + ")"
    if operator == "aten.unsqueeze.default":
        value = _format_scalar(_argument(event, 0, "self", "?"))
        dim = _argument(event, 1, "dim", "?")
        return f"unsqueeze({value}, {_format_scalar(dim)})"
    if operator == "aten.linear.default":
        args = [_argument(event, index, name, "?") for index, name in enumerate(("input", "weight", "bias"))]
        values = [_format_scalar(value) for value in args[:2]]
        if args[2] is not None:
            values.append(_format_scalar(args[2]))
        return "linear(" + ", ".join(values) + ")"
    if operator in _UNARY:
        name = _UNARY[operator]
        args = _call_arguments(event)
        if name == "neg" and len(args) == 1:
            value = _format_scalar(
                _argument(event, 0, "self", event["inputs"][0] if event["inputs"] else "?")
            )
            return f"(-{value})"
        # Unary ATen operators can still carry meaningful scalar options, such
        # as softplus(beta, threshold) and gelu(approximate). Keep those options
        # in the formula instead of silently rendering a default operation.
        return f"{name}({', '.join(args)})"
    if operator in _FUNCTIONS:
        args = _call_arguments(event)
        return f"{_FUNCTIONS[operator]}({', '.join(args)})"
    return f"{operator}({', '.join(_call_arguments(event))})"


class ForwardReadableProgram:
    """Navigate and render a sealed physical trace without importing a model runtime."""

    def __init__(self, trace: Mapping[str, Any], annotations: Mapping[str, Any] | None = None):
        try:
            verify_trace(trace)
        except ForwardSymbolsError as exc:
            raise ForwardReadableError(str(exc)) from exc
        self.trace = copy.deepcopy(dict(trace))
        self.fingerprint = self.trace["fingerprint"]
        self._events = {event["sequence"]: event for event in self.trace["events"]}
        self._values = self.trace["values"]
        self._producer = {value_id: row["producer"] for value_id, row in self._values.items()}
        self._uses = {value_id: [] for value_id in self._values}
        for event in self.trace["events"]:
            for value_id in event["inputs"]:
                self._uses[value_id].append(event["sequence"])

        self._value_output_paths = {value_id: [] for value_id in self._values}
        self._output_path_provenance: dict[str, str] = {}
        self._event_output_paths = {sequence: [] for sequence in self._events}
        self._output_values: dict[str, list[str]] = {}
        for value_id, path, provenance in _output_refs(self.trace["outputs"]):
            self._output_values.setdefault(path, []).append(value_id)
            self._output_path_provenance[path] = provenance
            pending_values, seen_values, seen_events = [value_id], set(), set()
            while pending_values:
                current = pending_values.pop()
                if current in seen_values:
                    continue
                seen_values.add(current)
                if path not in self._value_output_paths[current]:
                    self._value_output_paths[current].append(path)
                producer = self._producer[current]
                if producer is None or producer in seen_events:
                    continue
                seen_events.add(producer)
                if path not in self._event_output_paths[producer]:
                    self._event_output_paths[producer].append(path)
                pending_values.extend(self._events[producer]["inputs"])

        self._expressions = {sequence: _event_expression(event) for sequence, event in self._events.items()}
        self._annotations = self._validate_annotations(annotations)
        self._annotations_by_scope: dict[tuple[str, Any], list[dict[str, Any]]] = {}
        for annotation in self._annotations:
            scope = annotation["scope"]
            scope_key = (scope["kind"], scope.get("id", scope.get("path")))
            self._annotations_by_scope.setdefault(scope_key, []).append(annotation)
        self._patterns = self._find_patterns()

    def _validate_annotations(self, document: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        if document is None:
            return []
        if not isinstance(document, Mapping) or set(document) != _ANNOTATION_DOCUMENT_KEYS:
            raise ForwardReadableError("annotation document fields differ from mdb-forward-annotations-v1")
        if document["schema"] != FORWARD_ANNOTATIONS_SCHEMA:
            raise ForwardReadableError("unsupported forward annotation schema")
        if document["trace_fingerprint"] != self.fingerprint:
            raise ForwardReadableError("annotations are bound to a different trace fingerprint")
        rows = document["annotations"]
        if not isinstance(rows, list) or len(rows) > _MAX_ANNOTATIONS:
            raise ForwardReadableError("annotations must be a bounded array")
        module_paths = {event["module_path"] for event in self._events.values()}
        output_paths = set(self._output_values)
        checked, seen_ids = [], set()
        for index, raw in enumerate(rows):
            if not isinstance(raw, Mapping) or set(raw) != _ANNOTATION_KEYS:
                raise ForwardReadableError(f"annotation {index} has unsupported fields")
            annotation = copy.deepcopy(dict(raw))
            identifier = annotation["id"]
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen_ids:
                raise ForwardReadableError(f"annotation {index} has an invalid or duplicate id")
            seen_ids.add(identifier)
            if not isinstance(annotation["claim"], str) or not annotation["claim"].strip():
                raise ForwardReadableError(f"annotation {identifier} requires a claim")
            if not isinstance(annotation["provenance"], str) or not annotation["provenance"].strip():
                raise ForwardReadableError(f"annotation {identifier} requires provenance")
            if any(len(annotation[key].encode("utf-8")) > _MAX_ANNOTATION_TEXT_BYTES
                   for key in ("id", "claim", "provenance")):
                raise ForwardReadableError(f"annotation {identifier} text is too large")
            refs = annotation["evidence_refs"]
            if (not isinstance(refs, list) or not refs or len(refs) > _MAX_EVIDENCE_REFS
                    or any(not isinstance(ref, str) or not ref.strip() for ref in refs)):
                raise ForwardReadableError(f"annotation {identifier} requires non-empty evidence_refs")
            if any(len(ref.encode("utf-8")) > _MAX_ANNOTATION_TEXT_BYTES for ref in refs):
                raise ForwardReadableError(f"annotation {identifier} evidence reference is too large")
            scope = annotation["scope"]
            if not isinstance(scope, Mapping) or "kind" not in scope:
                raise ForwardReadableError(f"annotation {identifier} requires a typed scope")
            kind = scope["kind"]
            if kind == "value":
                valid = (set(scope) == {"kind", "id"} and isinstance(scope["id"], str)
                         and scope["id"] in self._values)
            elif kind == "event":
                valid = (set(scope) == {"kind", "id"} and type(scope["id"]) is int
                         and scope["id"] in self._events)
            elif kind == "module":
                valid = (set(scope) == {"kind", "path"} and isinstance(scope["path"], str)
                         and scope["path"] in module_paths)
            elif kind == "output":
                valid = (set(scope) == {"kind", "path"} and isinstance(scope["path"], str)
                         and scope["path"] in output_paths)
            else:
                valid = False
            if not valid:
                raise ForwardReadableError(f"annotation {identifier} has an unknown or malformed scope")
            checked.append(annotation)
        return checked

    def _arg_ref(self, event: Mapping[str, Any], index: int, name: str) -> str | None:
        value = _argument(event, index, name, None)
        return value.get("$value") if isinstance(value, Mapping) and set(value) == {"$value"} else None

    def _event_annotations(self, sequence: int) -> list[dict[str, Any]]:
        event = self._events[sequence]
        entries = list(self._annotations_by_scope.get(("event", sequence), []))
        entries.extend(self._annotations_by_scope.get(("module", event["module_path"]), []))
        for path in self._event_output_paths[sequence]:
            entries.extend(self._annotations_by_scope.get(("output", path), []))
        return self._unique_annotations(entries)

    @staticmethod
    def _unique_annotations(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [copy.deepcopy(item) for item in {item["id"]: item for item in entries}.values()]

    def _local_value_annotations(self, value_id: str) -> list[dict[str, Any]]:
        entries = list(self._annotations_by_scope.get(("value", value_id), []))
        for path in self._value_output_paths[value_id]:
            entries.extend(self._annotations_by_scope.get(("output", path), []))
        return self._unique_annotations(entries)

    def _value_annotations(self, value_id: str) -> list[dict[str, Any]]:
        entries = self._local_value_annotations(value_id)
        producer = self._producer[value_id]
        if producer is not None:
            entries.extend(self._event_annotations(producer))
        return self._unique_annotations(entries)

    def _value_record(self, value_id: str) -> dict[str, Any]:
        row = self._values[value_id]
        producer = self._producer[value_id]
        producer_event = None if producer is None else self._events[producer]
        uses = list(self._uses[value_id])
        return {
            "value_id": value_id,
            "value_type": {"kind": "tensor", "dtype": row["dtype"], "shape": list(row["shape"])},
            "origin": row["origin"],
            "sha256": row["sha256"],
            "preview": list(row["preview"]),
            "producer": producer,
            "producer_operator": None if producer_event is None else producer_event["operator"],
            "module_path": None if producer_event is None else producer_event["module_path"],
            "module_path_provenance": None if producer_event is None else self._module_provenance(producer_event),
            "uses": uses,
            "use_module_paths": [self._events[sequence]["module_path"] for sequence in uses],
            "output_paths": list(self._value_output_paths[value_id]),
            "output_path_provenance": [
                {"path": path, "provenance": self._output_path_provenance[path]}
                for path in self._value_output_paths[value_id]
            ],
            "event_ancestry": self._value_event_ancestry(value_id),
            "expression": None if producer is None else self._expressions[producer],
            "provenance": "structural-observation",
            "annotations": self._value_annotations(value_id),
        }

    @staticmethod
    def _module_provenance(event: Mapping[str, Any]) -> str:
        path = event["module_path"]
        return "source-name" if path and path != "<root>" else "structural-observation"

    def _value_event_ancestry(self, value_id: str) -> list[int]:
        pending = [] if self._producer[value_id] is None else [self._producer[value_id]]
        seen = set()
        while pending:
            sequence = pending.pop()
            if sequence in seen:
                continue
            seen.add(sequence)
            for input_id in self._events[sequence]["inputs"]:
                parent = self._producer[input_id]
                if parent is not None:
                    pending.append(parent)
        return sorted(seen)

    def _event_record(self, sequence: int) -> dict[str, Any]:
        event = self._events[sequence]
        return {
            "sequence": sequence,
            "operator": event["operator"],
            "module_path": event["module_path"],
            "module_path_provenance": self._module_provenance(event),
            "inputs": list(event["inputs"]),
            "outputs": [
                {
                    "value_id": value_id,
                    "value_type": {"kind": "tensor", "dtype": self._values[value_id]["dtype"],
                                   "shape": list(self._values[value_id]["shape"])},
                    "output_paths": list(self._value_output_paths[value_id]),
                    "output_path_provenance": [
                        {"path": path, "provenance": self._output_path_provenance[path]}
                        for path in self._value_output_paths[value_id]
                    ],
                }
                for value_id in event["outputs"]
            ],
            "expression": self._expressions[sequence],
            "expression_provenance": "structural-observation",
            "output_ancestry": list(self._event_output_paths[sequence]),
            "annotations": self._event_annotations(sequence),
        }

    def _event_for(self, value_id: str) -> Mapping[str, Any] | None:
        producer = self._producer[value_id]
        return None if producer is None else self._events[producer]

    @staticmethod
    def _is_operator(event: Mapping[str, Any] | None, *prefixes: str) -> bool:
        return event is not None and any(event["operator"].startswith(prefix) for prefix in prefixes)

    def _softplus_unsqueeze(self, value_id: str) -> tuple[str, str, int] | None:
        event = self._event_for(value_id)
        if not self._is_operator(event, "aten.unsqueeze."):
            return None
        source = self._arg_ref(event, 0, "self")
        if source is None or not self._is_operator(self._event_for(source), "aten.softplus."):
            return None
        return source, value_id, self._producer[value_id]

    def _exp_input(self, value_id: str) -> str | None:
        event = self._event_for(value_id)
        if not self._is_operator(event, "aten.exp."):
            return None
        return self._arg_ref(event, 0, "self")

    def _negative_exp_input(self, value_id: str) -> str | None:
        event = self._event_for(value_id)
        if self._is_operator(event, "aten.neg."):
            source = self._arg_ref(event, 0, "self")
            return None if source is None else self._exp_input(source)
        if event is not None and event["operator"] in {"aten.mul.Scalar", "aten.mul.Tensor"}:
            left = _argument(event, 0, "self", None)
            right = _argument(event, 1, "other", None)
            for candidate, factor in ((left, right), (right, left)):
                if isinstance(factor, (int, float)) and factor == -1 and isinstance(candidate, Mapping):
                    if set(candidate) == {"$value"}:
                        result = self._exp_input(candidate["$value"])
                        if result is not None:
                            return result
        return None

    def _find_patterns(self) -> list[dict[str, Any]]:
        patterns = []
        for sequence, event in self._events.items():
            operator = event["operator"]
            if operator in {"aten.native_layer_norm.default", "aten.layer_norm.default",
                            "aten.group_norm.default", "aten.batch_norm.default"}:
                patterns.append({
                    "kind": "normalization-operator",
                    "operator": operator,
                    "expression": self._expressions[sequence],
                    "events": [sequence], "values": list(event["inputs"]),
                    "outputs": list(event["outputs"]),
                    "provenance": "structural-observation",
                })
            elif operator == "aten.linear.default":
                patterns.append({
                    "kind": "linear-operator",
                    "operator": operator,
                    "expression": self._expressions[sequence],
                    "events": [sequence], "values": list(event["inputs"]),
                    "outputs": list(event["outputs"]),
                    "provenance": "structural-observation",
                })
            elif operator == "aten.addmm.default":
                patterns.append({
                    "kind": "matrix-product-plus-addend",
                    "operator": operator,
                    "expression": self._expressions[sequence],
                    "events": [sequence], "values": list(event["inputs"]),
                    "outputs": list(event["outputs"]),
                    "provenance": "structural-observation",
                })

            if operator in {"aten.matmul.default", "aten.mm.default", "aten.bmm.default"}:
                for weight_index, other_index in ((0, 1), (1, 0)):
                    weight_id = self._arg_ref(event, weight_index, "self" if weight_index == 0 else "other")
                    other_id = self._arg_ref(event, other_index, "other" if other_index == 1 else "self")
                    if weight_id is None or other_id is None:
                        continue
                    weight_event = self._event_for(weight_id)
                    if self._is_operator(weight_event, "aten.softmax.", "aten._softmax."):
                        sequence_list = sorted({self._producer[weight_id], sequence})
                        patterns.append({
                            "kind": "softmax-weighted-matrix-product",
                            "operator": operator,
                            "expression": self._expressions[sequence],
                            "weight_value": weight_id, "other_value": other_id,
                            "events": sequence_list,
                            "values": list(dict.fromkeys([*event["inputs"], *weight_event["inputs"]])),
                            "outputs": list(event["outputs"]),
                            "provenance": "structural-observation",
                        })
                        break

            if operator in {"aten.sum.default", "aten.sum.dim_IntList"}:
                summed_id = self._arg_ref(event, 0, "self")
                multiply = None if summed_id is None else self._event_for(summed_id)
                if self._is_operator(multiply, "aten.mul."):
                    for weight_index, other_index in ((0, 1), (1, 0)):
                        weight_id = self._arg_ref(
                            multiply, weight_index, "self" if weight_index == 0 else "other"
                        )
                        other_id = self._arg_ref(
                            multiply, other_index, "other" if other_index == 1 else "self"
                        )
                        if weight_id is None or other_id is None:
                            continue
                        weight_event = self._event_for(weight_id)
                        if self._is_operator(weight_event, "aten.softmax.", "aten._softmax."):
                            sequences = sorted({self._producer[weight_id], self._producer[summed_id], sequence})
                            patterns.append({
                                "kind": "softmax-weighted-reduction",
                                "operator": operator,
                                "expression": self._expressions[sequence],
                                "weight_value": weight_id, "other_value": other_id,
                                "events": sequences,
                                "values": list(dict.fromkeys([
                                    *event["inputs"], *multiply["inputs"], *weight_event["inputs"]
                                ])),
                                "outputs": list(event["outputs"]),
                                "provenance": "structural-observation",
                            })
                            break

            if operator in {"aten.exp.default", "aten.exp.out"} and event["inputs"]:
                scaled_id = self._arg_ref(event, 0, "self")
                scaled_event = None if scaled_id is None else self._event_for(scaled_id)
                if self._is_operator(scaled_event, "aten.mul."):
                    for time_index, decay_index in ((0, 1), (1, 0)):
                        time_id = self._arg_ref(
                            scaled_event, time_index, "self" if time_index == 0 else "other"
                        )
                        decay_id = self._arg_ref(
                            scaled_event, decay_index, "other" if decay_index == 1 else "self"
                        )
                        if time_id is None or decay_id is None:
                            continue
                        time_pattern = self._softplus_unsqueeze(time_id)
                        inner_exp_input = self._negative_exp_input(decay_id)
                        if time_pattern is None or inner_exp_input is None:
                            continue
                        softplus_id, unsqueeze_id, unsqueeze_sequence = time_pattern
                        softplus_sequence = self._producer[softplus_id]
                        decay_event = self._event_for(decay_id)
                        exp_value = next(
                            (ref for ref in _refs(decay_event["arguments"])
                             if self._exp_input(ref) is not None), None
                        ) if decay_event is not None else None
                        negative_sequence = None if decay_event is None else self._producer[decay_id]
                        exp_sequence = None if exp_value is None else self._producer[exp_value]
                        inner_exp_value = None
                        if exp_value is not None:
                            inner_exp_value = self._exp_input(exp_value)
                        sequences = [sequence, self._producer[scaled_id], unsqueeze_sequence,
                                     softplus_sequence, negative_sequence]
                        if exp_sequence is not None:
                            sequences.append(exp_sequence)
                        sequences = sorted({item for item in sequences if item is not None})
                        pattern_values = list(dict.fromkeys(
                            value_id
                            for item in sequences
                            for value_id in [*self._events[item]["inputs"], *self._events[item]["outputs"]]
                        ))
                        expanded = (
                            f"exp(unsqueeze({softplus_id}, "
                            f"{_format_scalar(_argument(self._events[unsqueeze_sequence], 1, 'dim', '?'))})"
                            f" * (-exp({inner_exp_value or inner_exp_input})))"
                        )
                        patterns.append({
                            "kind": "exp-of-softplus-scaled-negative-exp",
                            "expression": expanded,
                            "events": sequences,
                            "values": pattern_values,
                            "outputs": list(event["outputs"]),
                            "provenance": "structural-observation",
                        })
                        break
        return patterns

    def list_patterns(self) -> list[dict[str, Any]]:
        """Return exact operator/dataflow patterns with their source evidence."""
        return copy.deepcopy(self._patterns)

    def list_values(self, *, scope: str | None = None, producer: int | None = None,
                    use: int | None = None) -> list[dict[str, Any]]:
        """List values, optionally filtered by exact module scope, producer, and/or use event."""
        for label, sequence in (("producer", producer), ("use", use)):
            if sequence is not None and (type(sequence) is not int or sequence not in self._events):
                raise ForwardReadableError(f"{label} must identify a captured event")
        result = []
        for value_id in self._values:
            producer_id = self._producer[value_id]
            uses = self._uses[value_id]
            if producer is not None and producer_id != producer:
                continue
            if use is not None and use not in uses:
                continue
            if scope is not None:
                paths = []
                if producer_id is not None:
                    paths.append(self._events[producer_id]["module_path"])
                paths.extend(self._events[item]["module_path"] for item in uses)
                if scope not in paths:
                    continue
            result.append(self._value_record(value_id))
        return result

    def value(self, value_id: str) -> dict[str, Any]:
        """Inspect one anonymous value with producer, uses, module scope, and output ancestry."""
        if value_id not in self._values:
            raise ForwardReadableError(f"unknown value ID: {value_id}")
        return self._value_record(value_id)

    def event(self, sequence: int) -> dict[str, Any]:
        """Inspect one operation with its exact expression and source module path."""
        if type(sequence) is not int or sequence not in self._events:
            raise ForwardReadableError(f"unknown event: {sequence!r}")
        return self._event_record(sequence)

    def backward_slice(self, *, value_id: str | None = None,
                       output_path: str | None = None) -> dict[str, Any]:
        """Return the producer ancestry of one value or one JSON-pointer output path."""
        if (value_id is None) == (output_path is None):
            raise ForwardReadableError("provide exactly one of value_id or output_path")
        if value_id is not None:
            if value_id not in self._values:
                raise ForwardReadableError(f"unknown value ID: {value_id}")
            targets = [value_id]
        else:
            if output_path not in self._output_values:
                raise ForwardReadableError(f"unknown tensor output path: {output_path!r}")
            targets = list(self._output_values[output_path])
        values, events, pending = set(), set(), list(targets)
        while pending:
            current = pending.pop()
            if current in values:
                continue
            values.add(current)
            producer = self._producer[current]
            if producer is None or producer in events:
                continue
            events.add(producer)
            pending.extend(self._events[producer]["inputs"])
        ordered_values = sorted(values, key=lambda item: (-1 if self._producer[item] is None else self._producer[item], item))
        ordered_events = sorted(events)
        paths = sorted({path for current in values for path in self._value_output_paths[current]})
        return {
            "schema": FORWARD_READABLE_SCHEMA,
            "trace_fingerprint": self.fingerprint,
            "target_values": targets,
            "output_ancestry": paths,
            "values": [self._value_record(item) for item in ordered_values],
            "events": [self._event_record(item) for item in ordered_events],
            "provenance": "structural-observation",
        }

    def program(self) -> dict[str, Any]:
        """Return all values and operations grouped by their recorded module paths."""
        groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for sequence, event in self._events.items():
            groups.setdefault(event["module_path"], []).append(self._event_record(sequence))
        modules = [
            {
                "module_path": path,
                "module_path_provenance": (
                    "source-name" if path and path != "<root>" else "structural-observation"
                ),
                "events": events,
            }
            for path, events in groups.items()
        ]
        roots = [self._value_record(value_id) for value_id in self._values if self._producer[value_id] is None]
        return {
            "schema": FORWARD_READABLE_SCHEMA,
            "trace_fingerprint": self.fingerprint,
            "model_identity": copy.deepcopy(self.trace["model_identity"]),
            "execution": copy.deepcopy(self.trace["execution"]),
            "outputs": copy.deepcopy(self.trace["outputs"]),
            "output_paths": [
                {"path": path, "provenance": self._output_path_provenance[path],
                 "values": list(value_ids)}
                for path, value_ids in self._output_values.items()
            ],
            "roots": roots,
            "modules": modules,
            "values": self.list_values(),
            "patterns": self.list_patterns(),
            "annotations": copy.deepcopy(self._annotations),
            "provenance": "structural-observation",
            "semantic_labels_supplied": False,
        }

    def render(self, *, scope: str | None = None, output_path: str | None = None) -> str:
        """Render a compact SSA-like listing grouped by recorded module path."""
        if output_path is not None:
            sliced = self.backward_slice(output_path=output_path)
            event_ids = {row["sequence"] for row in sliced["events"]}
            value_ids = {row["value_id"] for row in sliced["values"]}
            root_ids = {item for item in value_ids if self._producer[item] is None}
        elif scope is not None:
            rows = self.list_values(scope=scope)
            event_ids = {sequence for sequence, event in self._events.items()
                         if event["module_path"] == scope}
            value_ids = {row["value_id"] for row in rows}
            root_ids = {item for item in value_ids if self._producer[item] is None}
        else:
            event_ids = set(self._events)
            value_ids = set(self._values)
            root_ids = {item for item in value_ids if self._producer[item] is None}

        lines = [
            f"forward readable program: {len(value_ids)} values, {len(event_ids)} operations",
            f"trace fingerprint: {self.fingerprint}",
            "outputs: " + _format_scalar(self.trace["outputs"]),
        ]
        if root_ids:
            lines.append("roots:")
            for value_id in self._values:
                if value_id not in root_ids:
                    continue
                row = self._values[value_id]
                uses = ",".join(str(item) for item in self._uses[value_id]) or "-"
                ancestry = ",".join(_pointer_path(item) for item in self._value_output_paths[value_id]) or "-"
                lines.append(
                    f"  {value_id} = {row['origin']} ; tensor shape={row['shape']} dtype={row['dtype']} "
                    f"uses={uses} output_ancestry={ancestry}"
                )
                lines.extend(self._render_annotations(self._local_value_annotations(value_id), "    "))

        grouped: OrderedDict[str, list[int]] = OrderedDict()
        for sequence, event in self._events.items():
            if sequence in event_ids and (scope is None or event["module_path"] == scope):
                grouped.setdefault(event["module_path"], []).append(sequence)
        for module_path, sequences in grouped.items():
            provenance = "source-name" if module_path and module_path != "<root>" else "structural-observation"
            lines.append(f"module {_pointer_path(module_path)} [{provenance}]")
            for sequence in sequences:
                event = self._events[sequence]
                defs = [value_id for value_id in event["outputs"] if value_id in value_ids]
                lhs = ", ".join(defs)
                expression = self._expressions[sequence]
                rendered = f"{lhs} = {expression}" if lhs else expression
                paths = sorted({path for value_id in defs for path in self._value_output_paths[value_id]})
                ancestry = ",".join(_pointer_path(path) for path in paths) or "-"
                lines.append(f"  event {sequence} {event['operator']}: {rendered} ; output_ancestry={ancestry}")
                for value_id in defs:
                    row = self._values[value_id]
                    lines.append(f"    {value_id}: tensor shape={row['shape']} dtype={row['dtype']}")
                    lines.extend(self._render_annotations(self._local_value_annotations(value_id), "      "))
                lines.extend(self._render_annotations(self._event_annotations(sequence), "    "))
        if not event_ids:
            lines.append("(no operations in this scope)")
        patterns = [pattern for pattern in self._patterns
                    if ((output_path is None and scope is None)
                        or any(value_id in value_ids for value_id in pattern["outputs"])
                        or any(sequence in event_ids for sequence in pattern["events"]))]
        if patterns:
            lines.append("exact structural patterns:")
            for pattern in patterns:
                lines.append(
                    f"  {pattern['kind']}: {pattern['expression']} "
                    f"; events={pattern['events']} provenance={pattern['provenance']}"
                )
        return "\n".join(lines)

    @staticmethod
    def _render_annotations(annotations: list[dict[str, Any]], indent: str) -> list[str]:
        return [
            f"{indent}annotation {item['id']} ({item['provenance']}): {item['claim']} "
            f"; evidence_refs={item['evidence_refs']}"
            for item in annotations
        ]


__all__ = [
    "FORWARD_ANNOTATIONS_SCHEMA", "FORWARD_READABLE_SCHEMA", "ForwardReadableError",
    "ForwardReadableProgram",
]
