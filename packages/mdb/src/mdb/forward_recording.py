"""Observe native ATen execution and replay it from inputs and frozen parameters.

Values receive anonymous SSA identities. No semantic vocabulary, probe, or
intermediate-state oracle is used. This optional module imports PyTorch; the
metadata navigator in ``forward_symbols`` remains framework-free.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


class ForwardRecordingError(ValueError):
    """The observed execution cannot meet the declared replay contract."""


def tensor_descriptor(value):
    with _disable_current_modes():
        data = value.detach().cpu().contiguous()
        raw = data.reshape(-1).view(torch.uint8).numpy().tobytes()
        preview = data.reshape(-1)[:8].tolist()
        preview = [str(item) if isinstance(item, float) and not math.isfinite(item) else item
                   for item in preview]
        return {"shape": list(data.shape), "dtype": str(data.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(), "preview": preview}


def _leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)


@dataclass
class ForwardCapture:
    trace: dict
    initial_values: dict
    observed_values: dict
    output: Any


class _Recorder(TorchDispatchMode):
    def __init__(self, model, arguments, max_operations, forward_call):
        super().__init__()
        self.allowed = {id(value): "input" for value in _leaves(arguments)}
        self.allowed.update({id(value): "parameter" for value in model.parameters()})
        self.allowed.update({id(value): "parameter" for value in model.buffers()})
        self.current, self.references = {}, []
        self.values, self.initial_values, self.observed_values, self.events = {}, {}, {}, []
        self.max_operations, self.forward_call = max_operations, forward_call
        self.modules, self.hooks = [], []
        for name, module in model.named_modules():
            self.hooks.append(module.register_forward_pre_hook(self._enter(name or "<root>")))
            self.hooks.append(module.register_forward_hook(self._leave, always_call=True))

    def _enter(self, name):
        def enter(_module, _args):
            self.modules.append(name)
        return enter

    def _leave(self, _module, _args, _output):
        if self.modules:
            self.modules.pop()

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def remember(self, value, *, producer=None, output_ids=None, literal=False):
        key = id(value)
        if output_ids is not None and key in output_ids:
            return output_ids[key]
        if producer is None and key in self.current:
            return self.current[key]
        if producer is None and key not in self.allowed and not literal:
            raise ForwardRecordingError("unobserved tensor root; only declared inputs/parameters are admitted")
        if producer is None and (value.device.type != "cpu" or not value.is_contiguous()
                                 or value.storage_offset() != 0):
            raise ForwardRecordingError("this replay contract requires dense contiguous CPU roots at storage offset zero")
        identifier = f"v{len(self.values):06d}"
        origin = "operation-output" if producer is not None else self.allowed.get(key, "literal")
        descriptor = {**tensor_descriptor(value), "origin": origin, "producer": producer}
        with _disable_current_modes():
            snapshot = value.detach().clone(memory_format=torch.preserve_format)
        self.values[identifier] = descriptor
        self.observed_values[identifier] = snapshot
        if producer is None:
            self.initial_values[identifier] = snapshot
        self.current[key] = identifier
        self.references.append(value)  # Prevent Python IDs being recycled during capture.
        if output_ids is not None:
            output_ids[key] = identifier
        return identifier

    def encode(self, value, *, producer=None, output_ids=None, literal=False):
        if isinstance(value, torch.Tensor):
            return {"$value": self.remember(value, producer=producer, output_ids=output_ids, literal=literal)}
        if isinstance(value, tuple):
            return {"$tuple": [self.encode(x, producer=producer, output_ids=output_ids, literal=literal)
                               for x in value]}
        if isinstance(value, list):
            return [self.encode(x, producer=producer, output_ids=output_ids, literal=literal) for x in value]
        if isinstance(value, dict):
            return {str(key): self.encode(item, producer=producer, output_ids=output_ids, literal=literal)
                    for key, item in value.items()}
        if isinstance(value, torch.dtype):
            return {"$dtype": str(value).removeprefix("torch.")}
        if isinstance(value, torch.device):
            return {"$device": str(value)}
        if isinstance(value, (torch.layout, torch.memory_format)):
            return {"$torch": str(value).removeprefix("torch.")}
        if value is None or type(value) in (str, int, bool, float):
            if isinstance(value, float) and not math.isfinite(value):
                return {"$float": str(value)}
            return value
        raise ForwardRecordingError(f"unsupported native argument type: {type(value).__name__}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        if len(self.events) >= self.max_operations:
            raise ForwardRecordingError("declared operation bound exceeded")
        if any(argument.alias_info is not None and argument.alias_info.is_write
               for argument in func._schema.arguments):
            raise ForwardRecordingError(f"mutable operator requires an alias-state contract: {func}")
        if torch.Tag.nondeterministic_seeded in func.tags:
            raise ForwardRecordingError(f"random operator requires an RNG closure: {func}")
        sequence = len(self.events)
        encoded = self.encode({"args": args, "kwargs": kwargs or {}},
                              literal=str(func) in {"aten.lift_fresh.default", "aten.lift_fresh_copy.default"})
        result = func(*args, **(kwargs or {}))
        produced = self.encode(result, producer=sequence, output_ids={})
        self.events.append({
            "sequence": sequence, "operator": str(func),
            "module_path": self.modules[-1] if self.modules else "<root>",
            "inputs": _references(encoded), "outputs": _references(produced),
            "arguments": encoded, "result": produced,
            "clock": {"forward_call": self.forward_call, "operation": sequence},
        })
        return result


def _references(value):
    result = []

    def visit(item):
        if isinstance(item, dict):
            if set(item) == {"$value"}:
                if item["$value"] not in result:
                    result.append(item["$value"])
            else:
                for nested in item.values():
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
    visit(value)
    return result


def record_forward(model, args_tuple, kwargs=None, *, model_identity, execution,
                   input_identity, metadata=None, max_operations=8192):
    """Observe one unchanged forward; admission accepts no semantic symbol map."""
    from .forward_symbols import seal_trace

    if any(module.training for module in model.modules()):
        raise ForwardRecordingError("use a frozen model in evaluation mode")
    metadata = dict(metadata or {})
    allowed_metadata = {"forward_call", "program_step", "clock_scope", "source_refs", "source_hashes",
                        "checkpoint_sha256", "case_id"}
    if set(metadata) - allowed_metadata:
        raise ForwardRecordingError("recorder metadata admits only clock and source provenance; no label maps")
    recorder = _Recorder(model, (args_tuple, kwargs or {}), max_operations,
                         int(metadata.get("forward_call", 0)))
    try:
        with torch.no_grad(), recorder:
            output = model(*args_tuple, **(kwargs or {}))
        outputs = recorder.encode(output)
        trace = seal_trace({
            "schema": "mdb-forward-symbols-v1", "model_identity": model_identity,
            "execution": execution, "input_identity": input_identity,
            "values": recorder.values, "events": recorder.events, "outputs": outputs,
            "capture_status": "complete", "semantic_labels_supplied": False,
            "metadata": {**metadata, "observer": "native-TorchDispatchMode",
                         "value_identity": "run-local-SSA", "replay_roots": "inputs-parameters-native-literals-only",
                         "semantic_interpretation": "unassigned"},
        })
        return ForwardCapture(trace, recorder.initial_values, recorder.observed_values, output)
    finally:
        recorder.close()


def _decode(value, environment):
    if isinstance(value, list):
        return [_decode(item, environment) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$value"}:
        return environment[value["$value"]]
    if set(value) == {"$tuple"}:
        return tuple(_decode(item, environment) for item in value["$tuple"])
    if set(value) in ({"$dtype"}, {"$torch"}):
        return getattr(torch, next(iter(value.values())))
    if set(value) == {"$device"}:
        return torch.device(value["$device"])
    if set(value) == {"$float"}:
        return float(value["$float"])
    return {key: _decode(item, environment) for key, item in value.items()}


def _bind(encoded, actual, environment):
    if isinstance(encoded, dict) and set(encoded) == {"$value"}:
        environment[encoded["$value"]] = actual
    elif isinstance(encoded, dict) and set(encoded) == {"$tuple"}:
        if len(encoded["$tuple"]) != len(actual):
            raise ForwardRecordingError("replay tuple geometry changed")
        for left, right in zip(encoded["$tuple"], actual, strict=True):
            _bind(left, right, environment)
    elif isinstance(encoded, dict) and set(encoded) in ({"$dtype"}, {"$torch"}, {"$float"}, {"$device"}):
        if _decode(encoded, environment) != actual:
            raise ForwardRecordingError("native scalar type result changed during replay")
    elif isinstance(encoded, list):
        for left, right in zip(encoded, actual, strict=True):
            _bind(left, right, environment)
    elif isinstance(encoded, dict):
        for key, item in encoded.items():
            _bind(item, actual[key], environment)
    elif encoded != actual:
        raise ForwardRecordingError("native scalar result changed during replay")


class ATenStepper:
    """Execute the observed native operations from roots, one operation per step."""

    def __init__(self, trace, initial_values):
        from .forward_symbols import verify_trace

        verify_trace(trace)
        roots = {key for key, value in trace["values"].items() if value["producer"] is None}
        if set(initial_values) != roots:
            raise ForwardRecordingError("replay admits exactly declared roots; intermediate oracles are forbidden")
        for key, value in initial_values.items():
            descriptor = tensor_descriptor(value)
            if any(descriptor[field] != trace["values"][key][field] for field in ("sha256", "shape", "dtype")):
                raise ForwardRecordingError("replay root bytes differ from the native capture")
        self.trace, self.pc, self.steps = trace, 0, []
        self.environment = {key: value.detach().clone(memory_format=torch.preserve_format)
                            for key, value in initial_values.items()}

    def step(self):
        if self.pc >= len(self.trace["events"]):
            raise StopIteration("forward execution complete")
        event = self.trace["events"][self.pc]
        namespace, name, overload = event["operator"].split(".", 2)
        if namespace not in {"aten", "prims"}:
            raise ForwardRecordingError("replay operator namespace is unsupported")
        operation = getattr(getattr(getattr(torch.ops, namespace), name), overload)
        if torch.Tag.nondeterministic_seeded in operation.tags:
            raise ForwardRecordingError("random replay operation requires an RNG closure")
        if any(argument.alias_info is not None and argument.alias_info.is_write
               for argument in operation._schema.arguments):
            raise ForwardRecordingError("mutable replay operation is unsupported")
        arguments = _decode(event["arguments"], self.environment)
        with torch.no_grad():
            actual = operation(*arguments["args"], **arguments["kwargs"])
        _bind(event["result"], actual, self.environment)
        descriptors = {key: tensor_descriptor(self.environment[key]) for key in event["outputs"]}
        exact = all(all(descriptors[key][field]
                        == self.trace["values"][key][field] for field in ("sha256", "shape", "dtype"))
                    for key in event["outputs"])
        result = {"event": event, "exact_outputs": exact, "pc": self.pc}
        self.steps.append(result)
        self.pc += 1
        return result

    def run(self, *, require_exact=True):
        if require_exact:
            for row in self.steps:
                if not row["exact_outputs"]:
                    raise ForwardRecordingError(f"native operation replay differs at step {row['pc']}")
        while self.pc < len(self.trace["events"]):
            row = self.step()
            if require_exact and not row["exact_outputs"]:
                raise ForwardRecordingError(f"native operation replay differs at step {row['pc']}")
        return _decode(self.trace["outputs"], self.environment)

    @property
    def current(self):
        return None if self.pc == 0 else self.steps[-1]

    def symbols(self, *, readable=False, annotations=None):
        """Read the values computed so far, with no future activation previews."""
        result = {
            "pc": self.pc,
            "event": None if self.current is None else self.current["event"],
            "authority": "live-ATen-replay-environment",
            "semantic_interpretation": "unassigned",
            "values": {
                key: {**tensor_descriptor(value),
                      "producer": self.trace["values"][key]["producer"],
                      "origin": self.trace["values"][key]["origin"]}
                for key, value in self.environment.items()
            },
        }
        if annotations is not None and not readable:
            raise ForwardRecordingError("annotations require the explicit readable view")
        if readable:
            from .forward_readable import ForwardReadableProgram

            program = ForwardReadableProgram(self.trace, annotations)
            structural_keys = ("expression", "producer_operator", "module_path", "module_path_provenance",
                               "uses", "use_module_paths", "output_paths", "output_path_provenance",
                               "annotations")
            for value_id, value in result["values"].items():
                observed = program.value(value_id)
                value["readable"] = {key: observed[key] for key in structural_keys}
            result["interpretation_scope"] = "structure and annotations bind the original trace; numeric descriptors are the current replay values"
        return result
