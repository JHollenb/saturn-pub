"""Capture a genuine native prefix, stopping before a named module body.

The recorded computation is a completed *prefix wrapper*, not a complete model
forward. Its outputs are the boundary arguments and requested observations made
before that boundary. The ordinary forward trace schema and replay API apply.
Torch remains optional until ``record_prefix`` is called.
"""
from __future__ import annotations


class ForwardPrefixError(ValueError):
    pass


class _PrefixAbort(BaseException):
    """Avoid ordinary model ``except Exception`` handlers swallowing the stop."""

    def __init__(self, owner):
        super().__init__("MDB native prefix boundary")
        self.owner = owner


def record_prefix(model, args_tuple, *, stop_before, observe=(), occurrence=0, **kwargs):
    """Return a ``ForwardCapture`` containing only the executed native prefix.

    ``stop_before`` and ``observe`` are exact paths from ``model.named_modules``.
    The first invocation of the stop module is intercepted before its local
    prehooks and body. Only ``occurrence=0`` is supported. Requested observations retain
    every completed invocation in order, without adding tensor-copy operations.

    The output is ``{"boundary": {"args": tuple, "kwargs": dict},
    "observed": {module_path: [outputs, ...]}}``. Recorded paths are prefixed
    with ``native.`` because the original model is a child of the wrapper.
    Remaining keyword arguments are the ordinary ``record_forward`` arguments,
    including model/input/execution identities and optional metadata/budgets.

    A missing stop or observation raises instead of returning a full-forward
    capture. Exceptions from the original model propagate unchanged. All hooks
    added by this function and the recorder are removed on every exit.
    """
    import torch

    from .forward_recording import _leaves, record_forward
    from .forward_symbols import seal_trace

    if type(occurrence) is not int or occurrence != 0:
        raise ForwardPrefixError("only the first stop-module invocation (occurrence=0) is supported")
    if not isinstance(args_tuple, tuple):
        raise ForwardPrefixError("prefix arguments must be an explicit tuple")
    if not isinstance(stop_before, str) or not stop_before:
        raise ForwardPrefixError("stop_before requires an exact non-root module path")
    if isinstance(observe, str):
        raise ForwardPrefixError("observe must be a sequence of exact module paths")
    observed_paths = tuple(observe)
    if any(not isinstance(path, str) or not path for path in observed_paths):
        raise ForwardPrefixError("observation paths must be non-root module paths")
    if len(set(observed_paths)) != len(observed_paths):
        raise ForwardPrefixError("observation paths must be unique")
    if stop_before in observed_paths:
        raise ForwardPrefixError("the stopped module has no completed output to observe")
    modules = dict(model.named_modules())
    if stop_before not in modules or any(path not in modules for path in observed_paths):
        raise ForwardPrefixError("unknown stop or observation module path")
    if any(module.training for module in model.modules()):
        raise ForwardPrefixError("prefix capture requires a model already in evaluation mode")
    supplied_kwargs = kwargs.get("kwargs") or {}
    tensors = [*_leaves((args_tuple, supplied_kwargs)), *model.parameters(), *model.buffers()]
    if any(value.device.type != "cpu" for value in tensors):
        raise ForwardPrefixError("native prefix capture requires CPU inputs and model tensors")

    owner = object()
    observations = {path: [] for path in observed_paths}
    boundary = None
    stopped = False

    class PrefixWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = model
            # All native modules were checked above. Do not recursively change
            # their flags or invoke a caller-defined train()/eval() method.
            self.training = False

        def forward(self, *args, **native_kwargs):
            try:
                self.native(*args, **native_kwargs)
            except _PrefixAbort as abort:
                if abort.owner is not owner:
                    raise
            else:
                raise ForwardPrefixError("requested prefix boundary was not reached")
            if not stopped or boundary is None:
                raise ForwardPrefixError("prefix boundary did not retain its call arguments")
            missing = [path for path, values in observations.items() if not values]
            if missing:
                raise ForwardPrefixError("requested observations did not complete before boundary: " + ", ".join(missing))
            return {"boundary": boundary, "observed": observations}

    def stop(module, args, native_kwargs):
        nonlocal boundary, stopped
        boundary = {"args": args, "kwargs": native_kwargs}
        stopped = True
        raise _PrefixAbort(owner)

    def observer(path):
        def observed(module, args, output):
            observations[path].append(output)
        return observed

    hooks = []
    try:
        hooks.append(modules[stop_before].register_forward_pre_hook(stop, prepend=True, with_kwargs=True))
        for path in observed_paths:
            hooks.append(modules[path].register_forward_hook(observer(path)))
        capture = record_forward(PrefixWrapper(), args_tuple, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    contract = {"schema": "mdb-native-prefix-boundary-v1", "execution_extent": "prefix-wrapper",
                "stop_before": stop_before, "wrapper_stop_before": "native." + stop_before,
                "occurrence": 0, "stop_edge": "before-local-prehooks-and-module-body",
                "observations": {path: {"wrapper_module": "native." + path,
                                        "completed_invocations": len(values)}
                                 for path, values in observations.items()},
                "output_abi": "boundary-args-kwargs-and-completed-module-observations-v1",
                "full_model_outputs_computed": False,
                "completion_scope": "prefix wrapper only; native forward aborted at boundary"}
    body = {key: value for key, value in capture.trace.items() if key != "fingerprint"}
    body["metadata"] = {**body["metadata"], "execution_extent": "prefix-wrapper", "prefix_contract": contract}
    capture.trace = seal_trace(body)
    return capture


__all__ = ["ForwardPrefixError", "record_prefix"]
