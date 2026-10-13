"""Caller-owned Transformers and TransformerLens forward/capture adapters.

Imports are lazy: merely selecting these classes requires no model libraries.
No weights are downloaded, no lease is created, and no checkpoint is fabricated.
"""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock

from .backends import ForwardResult, ModelRunner, RunnerCapabilities
from .errors import BenchError


def _snapshot(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


def _edit_tensor(value, edit):
    import torch

    if not isinstance(value, torch.Tensor):
        raise BenchError("this intervention requires a tensor-valued hook")
    private = value.clone()
    if callable(edit):
        result = edit(private)
        if result is None:
            result = private
    else:
        op = edit["op"]
        if op == "suppress":
            result = torch.zeros_like(private)
        elif op == "scale":
            result = private * edit["value"]
        else:
            other = torch.as_tensor(edit["value"], dtype=value.dtype, device=value.device)
            result = private + other if op == "add" else torch.zeros_like(private) + other
    if not isinstance(result, torch.Tensor):
        raise BenchError("intervention must return a native tensor or None")
    if result.shape != value.shape or result.dtype != value.dtype or result.device != value.device:
        raise BenchError("intervention changed shape, dtype or device")
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise BenchError("intervention produced non-finite values")
    return result


def _edit_output(value, edit):
    # HF decoder versions return either a residual tensor or a tuple whose
    # first element is that tensor. Cache/other tuple elements remain untouched.
    if isinstance(value, tuple):
        if not value:
            raise BenchError("cannot intervene on an empty module output")
        return (_edit_tensor(value[0], edit), *value[1:])
    return _edit_tensor(value, edit)


class TransformersRunner(ModelRunner):
    """Wrap a caller-loaded torch module; module output hooks are the addresses.

    Names are ``module:<named_modules path>:output`` (``<root>`` for the root).
    Tensor outputs and the first tensor of tuple outputs support edits. This
    runner exposes no token/layer stepping or durable KV-cache capability.
    """

    def __init__(self, model, *, debugger_adapter=None):
        if not callable(model) or not callable(getattr(model, "named_modules", None)):
            raise BenchError("TransformersRunner requires a caller-loaded torch module")
        self.model = model
        self.debugger_adapter = debugger_adapter
        self._lock = RLock()

    def _modules(self):
        return {
            f"module:{name or '<root>'}:output": module
            for name, module in self.model.named_modules()
        }

    def capabilities(self):
        points = tuple(self._modules())
        return RunnerCapabilities("transformers", points, points, True, True)

    def forward(self, inputs, *, capture=(), interventions=None, **kwargs):
        with self._lock:
            capture, interventions = self.capabilities().validate(capture, interventions)
            captured, fired, handles = {}, set(), []
            modules = self._modules()

            def make_hook(point):
                def hook(module, args, output):
                    if point in interventions:
                        output = _edit_output(output, interventions[point])
                    fired.add(point)
                    if point in capture:
                        captured[point] = _snapshot(output)
                    return output

                return hook

            try:
                for point in dict.fromkeys((*capture, *interventions)):
                    handles.append(modules[point].register_forward_hook(make_hook(point)))
                if isinstance(inputs, Mapping):
                    if set(inputs) & set(kwargs):
                        raise BenchError("input arguments conflict with forward keyword arguments")
                    output = self.model(**dict(inputs), **kwargs)
                else:
                    output = self.model(inputs, **kwargs)
                missing = (set(capture) | set(interventions)) - fired
                if missing:
                    raise BenchError(f"requested hooks did not fire: {sorted(missing)}")
                return ForwardResult(output, captured, {"capture_semantics": "after-intervention"})
            finally:
                for handle in reversed(handles):
                    handle.remove()


class TransformerLensRunner(ModelRunner):
    """Use an existing HookedTransformer, HookedSAETransformer, or v4 bridge.

    Local models use run_with_hooks; declarative-only remote bridges use their
    run_with_cache intervention API. No TransformerLens dependency is imported
    by MDB: the caller constructs its chosen model/bridge and owns its lifecycle.
    """

    def __init__(self, model, *, debugger_adapter=None):
        if not callable(getattr(model, "run_with_hooks", None)):
            raise BenchError("TransformerLensRunner requires a model with run_with_hooks")
        if not isinstance(getattr(model, "hook_dict", None), Mapping):
            raise BenchError("TransformerLens model must expose hook_dict")
        self.model = model
        self.debugger_adapter = debugger_adapter
        self._lock = RLock()

    def _driver(self):
        # BridgeCore v4 binds the documented Driver protocol under _driver.
        # Legacy HookedTransformer has no driver and uses local PyTorch hooks.
        return getattr(self.model, "_driver", None)

    def capabilities(self):
        driver = self._driver()
        points = set(self.model.hook_dict)
        if driver is not None:
            points -= set(getattr(driver, "non_fireable_hook_points", ()))
            supported = getattr(driver, "supported_hook_points", None)
            if supported is not None:
                points &= set(supported)
        callbacks = driver is None or driver.supports("intervention_callbacks")
        points = tuple(sorted(points))
        return RunnerCapabilities("transformerlens", points, points, callbacks, True)

    def forward(self, inputs, *, capture=(), interventions=None, **kwargs):
        with self._lock:
            caps = self.capabilities()
            capture, interventions = caps.validate(capture, interventions)
            reserved = {
                "fwd_hooks",
                "bwd_hooks",
                "reset_hooks_end",
                "names_filter",
                "intervene",
                "return_cache_object",
                "remove_batch_dim",
                "pos_slice",
            }
            if set(kwargs) & reserved:
                raise BenchError("hook lifecycle and capture scope are owned by this runner")
            if not caps.intervention_callbacks:
                # Translate the common scale vocabulary to TL's explicit factor.
                specs = {
                    point: (
                        {"op": "scale", "factor": edit["value"]}
                        if edit["op"] == "scale"
                        else dict(edit)
                    )
                    for point, edit in interventions.items()
                }
                output, cache = self.model.run_with_cache(
                    inputs,
                    names_filter=list(capture),
                    intervene=specs,
                    return_cache_object=False,
                    **kwargs,
                )
                return ForwardResult(
                    output,
                    {point: _snapshot(cache[point]) for point in capture if point in cache},
                    {"capture_semantics": "driver-defined"},
                )
            captured, fired = {}, set()

            def make_hook(point):
                def hook(value, hook):
                    if point in interventions:
                        value = _edit_tensor(value, interventions[point])
                    fired.add(point)
                    if point in capture:
                        captured[point] = _snapshot(value)
                    return value

                return hook

            hooks = [
                (point, make_hook(point)) for point in dict.fromkeys((*capture, *interventions))
            ]
            output = self.model.run_with_hooks(
                inputs, fwd_hooks=hooks, reset_hooks_end=True, **kwargs
            )
            missing = (set(capture) | set(interventions)) - fired
            if missing:
                raise BenchError(f"requested hooks did not fire: {sorted(missing)}")
            return ForwardResult(output, captured, {"capture_semantics": "after-intervention"})
