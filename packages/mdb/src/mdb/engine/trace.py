"""Passive capture of named ports during a forward the caller already runs (``mdb.trace``).

    with mdb.trace(model, ports=["L1.p", "L1.m", "L0.h"]) as t:
        model(input_ids)                # any forward: batch > 1, sdpa, generate, training
    t["L1.m"]                           # the value native code produced (one traced call)
    t["L1.m", 1]                        # the value of the second call
    t.info("L1.p")                      # state, authority, reason, kind, shape, dtype, axes
    t.record()                          # sealed JSON (<= 64 KiB, no arrays): digests, labels

``mdb.attach`` drives the model itself, one sequence at a time. A trace only watches: it
installs forward hooks at the sites the family's port program names (``PortSpec.site`` and
``PortSpec.hook``, the table the stepping debugger uses) and registers itself with the one
process-wide eager-attention interceptor of ``mdb.engine.ports``. Every hook returns nothing
and the interceptor returns the native result object, so the forward's outputs are unchanged
bit for bit. A port name means what it means in the stepper: on the same native call (one
token on top of the same key/value cache) the captured value equals ``Run.read`` bit for bit.

Calls. One call is one outermost forward of the model's decoder stack (``model.base_model``,
``model.model`` for Qwen) in the thread that entered the ``with`` block; several forwards in
one block give several calls, kept in order. A layer called directly, outside the stack, and
forwards from other threads pass through uncaptured.

Rows. Values keep the batch dimension and every row of the call. Each port's axes come from
the program (``t.info(address)["axes"]``): most ports are ``(batch, row, ...)``, so batch
item ``b``, row ``r`` is ``t[address][b, r]``; ``qr`` is ``(batch, head, row, head_dim)``,
``kr`` is ``(batch, kv_head, src, head_dim)`` and ``s``, ``p`` are ``(batch, head, row, src)``,
so their row is ``t[address][b, :, r]``. ``row`` counts the tokens this call fed in; ``src``
counts every position attention saw (cached ones included).

Labels. A captured port is ``measured``, ``kind: native``. ``s`` (the pre-softmax scores) is
not a native intermediate: it is computed from the captured ``qr`` and ``kr`` and the mask the
native eager call received, by the native expression, and labelled ``derived-exact``,
``kind: replica``; ``softmax(s)`` must equal the native ``p`` bit for bit or ``s`` is refused.

Refusals (state ``refused``, ``authority`` None, a ``reason``; never a wrong tensor):
``qr``, ``kr``, ``s`` and ``p`` when the layer did not run eager attention (sdpa and other
kernels); ``s`` and ``p`` when attention dropout is active; every port of a layer that did not
run in the call; every value after the ``max_bytes`` cap would be exceeded (default 1 GiB,
counted over the captured copies; capture stops there for the rest of the trace).

Captured values are detached copies on the device that produced them (a copy, because a cache
may later be written in place). Hooks and the interceptor registration are removed when the
block exits, also when the forward raises. One trace may be active per process: a second
``trace`` (nested or concurrent) is refused.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from mdb.core.errors import BadArgs, Refused
from mdb.core.pointers import RecordError, check_bounded, seal_digest

from .ports import _acquire, _equality, _release, config_refusal, program_for, tensor_digest

DEFAULT_MAX_BYTES = 1 << 30
SCHEMA = "mdb-trace-v1"
EAGER_PORTS = ("qr", "kr", "s", "p")
_ADDRESS = re.compile(r"L(?P<layer>\d+)\.(?P<port>\w+)")
_LOCK = threading.Lock()
_ACTIVE: list[Trace] = []


@dataclass
class _Call:
    """What one outermost forward of the decoder stack left behind."""

    index: int
    attention: str | None
    values: dict[str, torch.Tensor] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    replica: dict[str, dict[str, Any]] = field(default_factory=dict)
    layers: set[int] = field(default_factory=set)
    eager: set[int] = field(default_factory=set)


class _AttentionTap:
    """The trace's owner entry in the eager-attention interceptor for one layer.

    Implements the two methods the interceptor calls on an owner (``_engaged`` and
    ``_eager``); ``_eager`` returns the native function's own result.
    """

    def __init__(self, trace: Trace, layer: int):
        self.trace = trace
        self.layer = layer

    def _engaged(self) -> bool:
        return self.trace._engaged(self.layer)

    def _eager(self, patch, module, query, key, value, attention_mask, *args, **kwargs):
        trace, layer = self.trace, self.layer
        trace._calls[-1].eager.add(layer)
        trace._capture(layer, "qr", query)
        trace._capture(layer, "kr", key)
        output = patch.original(module, query, key, value, attention_mask, *args, **kwargs)
        weights = output[1]
        if module.training and kwargs.get("dropout", 0.0):
            reason = "attention dropout is active (training mode): p is not softmax(s)"
            trace._refuse(layer, "s", reason)
            trace._refuse(layer, "p", reason)
            return output
        if trace._wants(layer, "s"):
            trace._scores(layer, patch, module, query, key, attention_mask, weights, kwargs)
        trace._capture(layer, "p", weights)
        return output


class Trace:
    """Passive port capture over the forwards run inside one ``with`` block.

    ``ports``: addresses ``L<layer>.<port>`` of the model's port program. ``max_bytes``: cap on
    the bytes of captured copies (default 1 GiB). ``project``: must be None (persistence is not
    built yet). See the module docstring for calls, rows, labels and refusals.
    """

    def __init__(
        self,
        model: Any,
        ports: Iterable[str],
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        project: Any = None,
    ):
        if project is not None:
            raise Refused(
                "mdb.trace does not write to a project; call it without project= (a debugger "
                "attached with a project records cuts and records)"
            )
        if type(max_bytes) is not int or max_bytes <= 0:
            raise BadArgs("max_bytes must be a positive integer")
        config = getattr(model, "config", None)
        backbone = getattr(model, "base_model", None)
        layers = getattr(backbone, "layers", None)
        if config is None or layers is None:
            raise Refused("trace needs a loaded Hugging Face decoder (model.base_model.layers)")
        try:
            self.program = program_for(getattr(config, "model_type", None))
        except ValueError as error:
            raise Refused(str(error)) from None
        refusal = config_refusal(self.program.family, config)
        if refusal is not None:
            raise Refused(refusal)
        if isinstance(ports, str):
            raise BadArgs("ports is a list of addresses, e.g. ['L0.m']")
        self.model = model
        self.max_bytes = max_bytes
        self.ports = tuple(dict.fromkeys(self._parse(address, len(layers)) for address in ports))
        if not self.ports:
            raise BadArgs("name at least one port, e.g. ports=['L0.m']")
        self._backbone = backbone
        self._layers = layers
        self._wanted: dict[int, set[str]] = {}
        for address in self.ports:
            layer, port = self._split(address)
            self._wanted.setdefault(layer, set()).add(port)
        self._calls: list[_Call] = []
        self._held = 0
        self._stopped: str | None = None
        self._handles: list[Any] = []
        self._patches: list[tuple[Any, Any]] = []
        self._thread: int | None = None
        self._depth = 0
        self._layer_depth: dict[int, int] = {}
        self._entered = False

    # --- addresses ----------------------------------------------------------------------

    def _parse(self, address: Any, layers: int) -> str:
        match = _ADDRESS.fullmatch(address.strip()) if isinstance(address, str) else None
        if match is None:
            raise BadArgs(f"trace ports are L<layer>.<port> addresses, got {address!r}")
        layer, port = int(match["layer"]), match["port"]
        if layer >= layers:
            raise BadArgs(f"layer {layer} does not exist in a {layers}-layer model")
        try:
            name = self.program.spec(port).name
        except ValueError as error:
            raise BadArgs(str(error)) from None
        return f"L{layer}.{name}"

    @staticmethod
    def _split(address: str) -> tuple[int, str]:
        match = _ADDRESS.fullmatch(address)
        return int(match["layer"]), match["port"]

    def _wants(self, layer: int, port: str) -> bool:
        return port in self._wanted.get(layer, ())

    # --- install / remove ---------------------------------------------------------------

    def __enter__(self) -> Trace:
        if self._entered:
            raise Refused("a trace is entered once; start a new mdb.trace for more forwards")
        with _LOCK:
            if _ACTIVE:
                raise Refused(
                    "a trace is already active in this process (nested or concurrent traces "
                    "are refused)"
                )
            _ACTIVE.append(self)
        self._entered = True
        self._thread = threading.get_ident()
        try:
            self._install()
        except BaseException:
            self._remove()
            raise
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._remove()
        return False

    def _install(self) -> None:
        self._handles.append(self._backbone.register_forward_pre_hook(self._begin))
        self._handles.append(
            self._backbone.register_forward_hook(self._end, always_call=True)
        )
        modeling = self.program.modeling_module()
        for layer in sorted(self._wanted):
            module = self._layers[layer]
            self._layer_depth[layer] = 0
            self._handles.append(
                module.register_forward_pre_hook(self._enter(layer), with_kwargs=True)
            )
            for port in self.program.ports:
                if port.name not in self._wanted[layer] or not port.site:
                    continue
                if port.hook == "pre":
                    hook = self._pre(layer, port.name)
                    handle = module.get_submodule(port.site).register_forward_pre_hook(
                        hook, with_kwargs=True
                    )
                elif port.hook == "out":
                    hook = self._out(layer, port.name)
                    handle = module.get_submodule(port.site).register_forward_hook(hook)
                else:
                    continue
                self._handles.append(handle)
            self._handles.append(module.register_forward_hook(self._exit(layer), always_call=True))
            if self._wanted[layer] & set(EAGER_PORTS):
                attention = module.self_attn
                patch = _acquire(_AttentionTap(self, layer), modeling, attention)
                self._patches.append((patch, attention))

    def _remove(self) -> None:
        try:
            for handle in self._handles:
                handle.remove()
            for patch, attention in self._patches:
                _release(patch, attention)
        finally:
            self._handles, self._patches = [], []
            self._depth = 0
            self._layer_depth = {layer: 0 for layer in self._layer_depth}
            with _LOCK:
                if self in _ACTIVE:
                    _ACTIVE.remove(self)

    # --- hooks --------------------------------------------------------------------------

    def _mine(self) -> bool:
        return threading.get_ident() == self._thread

    def _engaged(self, layer: int) -> bool:
        """Inside an outermost stack forward and the outermost entry of ``layer``, in the
        thread that entered the trace."""
        return self._mine() and self._depth == 1 and self._layer_depth.get(layer) == 1

    def _begin(self, module: Any, args: tuple[Any, ...]) -> None:
        if not self._mine():
            return
        self._depth += 1
        if self._depth == 1:
            attention = getattr(self.model.config, "_attn_implementation", None)
            self._calls.append(_Call(len(self._calls), attention))

    def _end(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        if self._mine():
            self._depth = max(self._depth - 1, 0)

    def _enter(self, layer: int):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if not self._mine():
                return
            self._layer_depth[layer] += 1
            if self._engaged(layer):
                self._calls[-1].layers.add(layer)
                if self._wants(layer, "h"):
                    self._capture(layer, "h", _first(args, kwargs))

        return hook

    def _exit(self, layer: int):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            if not self._mine():
                return
            if self._engaged(layer) and output is not None and self._wants(layer, "h_out"):
                self._capture(layer, "h_out", output[0] if isinstance(output, tuple) else output)
            self._layer_depth[layer] = max(self._layer_depth[layer] - 1, 0)

        return hook

    def _pre(self, layer: int, port: str):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self._engaged(layer):
                self._capture(layer, port, _first(args, kwargs))

        return hook

    def _out(self, layer: int, port: str):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            if self._engaged(layer):
                self._capture(layer, port, output[0] if isinstance(output, tuple) else output)

        return hook

    # --- capture ------------------------------------------------------------------------

    def _refuse(self, layer: int, port: str, reason: str) -> None:
        if self._wants(layer, port):
            self._calls[-1].refused[f"L{layer}.{port}"] = reason

    def _capture(self, layer: int, port: str, value: Any) -> None:
        if not self._wants(layer, port) or not isinstance(value, torch.Tensor):
            return
        call = self._calls[-1]
        address = f"L{layer}.{port}"
        if self._stopped is None:
            size = value.numel() * value.element_size()
            if self._held + size > self.max_bytes:
                self._stopped = (
                    f"capture stopped: max_bytes={self.max_bytes} would be exceeded at call "
                    f"{call.index}, {address} ({size} bytes on top of {self._held} held); no "
                    "later value of this trace is captured"
                )
            else:
                call.values[address] = value.detach().clone()
                self._held += size
                return
        call.refused[address] = self._stopped

    def _scores(self, layer, patch, module, query, key, attention_mask, weights, kwargs) -> None:
        """``s`` by the native expression on the captured operands, checked against ``p``."""
        with torch.no_grad():
            scaling = kwargs.get("scaling", getattr(module, "scaling", None))
            keys = patch.repeat_kv(key.detach(), module.num_key_value_groups)
            scores = torch.matmul(query.detach(), keys.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            reference = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            check = _equality(reference, weights.detach())
        self._calls[-1].replica[f"L{layer}.s"] = check
        if not check["equal"]:
            self._refuse(
                layer, "s",
                f"replica check failed: max |softmax(s) - p| = {check['max_abs_diff']}",
            )
            return
        self._capture(layer, "s", scores)

    # --- reading ------------------------------------------------------------------------

    @property
    def calls(self) -> int:
        """Number of traced calls (outermost decoder-stack forwards)."""
        return len(self._calls)

    def _call(self, call: int | None) -> _Call | None:
        if call is None:
            if len(self._calls) > 1:
                raise BadArgs(
                    f"{len(self._calls)} calls were traced; name one: t[address, i] or "
                    "t.info(address, call=i)"
                )
            return self._calls[0] if self._calls else None
        if type(call) is not int or not -len(self._calls) <= call < len(self._calls):
            raise BadArgs(f"call {call!r} is out of range ({len(self._calls)} traced)")
        return self._calls[call]

    def _address(self, address: str) -> str:
        name = self._parse(address, len(self._layers))
        if name not in self.ports:
            raise BadArgs(f"{name} was not requested; ports: {', '.join(self.ports)}")
        return name

    def info(self, address: str, call: int | None = None) -> dict[str, Any]:
        """How the value at ``address`` in one call was obtained. Always these keys:
        ``address, call, state, authority, reason, kind, shape, dtype, axes, replica_equal,
        max_abs_diff, max_abs_diff_authority``."""
        name = self._address(address)
        chosen = self._call(call)
        layer, port = self._split(name)
        info: dict[str, Any] = {
            "address": name,
            "call": None if chosen is None else chosen.index,
            "state": "refused",
            "authority": None,
            "reason": None,
            "kind": None,
            "shape": None,
            "dtype": None,
            "axes": list(self.program.spec(port).axes),
            "replica_equal": None,
            "max_abs_diff": None,
            "max_abs_diff_authority": None,
        }
        if chosen is not None and name in chosen.replica:
            check = chosen.replica[name]
            info["replica_equal"] = bool(check["equal"])
            info["max_abs_diff"] = float(check.get("max_abs_diff", 0.0))
            info["max_abs_diff_authority"] = "measured"
        value = None if chosen is None else chosen.values.get(name)
        if value is None:
            info["reason"] = self._reason(chosen, layer, port)
            return info
        replica = port == "s"
        info.update(
            state="resolved",
            authority="derived-exact" if replica else "measured",
            kind="replica" if replica else "native",
            shape=list(value.shape),
            dtype=str(value.dtype),
        )
        return info

    def _reason(self, call: _Call | None, layer: int, port: str) -> str:
        if call is None:
            return "no forward of the model's decoder stack ran inside the trace"
        name = f"L{layer}.{port}"
        if name in call.refused:
            return call.refused[name]
        if layer not in call.layers:
            return f"layer {layer} did not run in call {call.index}"
        if port in EAGER_PORTS and layer not in call.eager:
            if call.attention != "eager":
                return (
                    f"attention implementation is {call.attention!r}: qr, kr, s and p exist "
                    "only inside eager attention (load with attn_implementation='eager')"
                )
            return f"eager attention did not run in layer {layer} during call {call.index}"
        return (
            f"port {port} did not fire in layer {layer} during call {call.index} (the forward "
            "stopped before it)"
        )

    def __getitem__(self, key: str | tuple[str, int]) -> torch.Tensor:
        """``t[address]`` (exactly one call traced) or ``t[address, call]``: the captured copy.
        A refused value raises ``Refused`` with its reason."""
        address, call = key if isinstance(key, tuple) else (key, None)
        info = self.info(address, call)
        if info["state"] != "resolved":
            raise Refused(f"{info['address']}: {info['reason']}")
        return self._calls[info["call"]].values[info["address"]]

    def record(self, calls: Iterable[int] | None = None) -> dict[str, Any]:
        """Sealed JSON of what was traced: per call and address the ``info`` fields plus the
        sha256 of each captured value (``mdb.engine.ports.tensor_digest``), and a ``seal``
        (``seal_digest`` of the rest). ``calls`` picks call indices (default: all). At most
        64 KiB and no arrays; a larger record is refused (pick fewer calls)."""
        indices = range(len(self._calls)) if calls is None else [int(c) for c in calls]
        chosen = [self._call(index) for index in indices]
        entries = []
        for item in chosen:
            values = {}
            for address in self.ports:
                entry = self.info(address, item.index)
                del entry["address"], entry["call"]
                if entry["state"] == "resolved":
                    entry["sha256"] = tensor_digest(item.values[address])
                values[address] = entry
            entries.append(
                {
                    "call": item.index,
                    "attention": item.attention,
                    "layers_run": sorted(item.layers),
                    "values": values,
                }
            )
        parameter = next(self.model.parameters(), None)
        body = {
            "schema": SCHEMA,
            "model": {
                "family": self.program.family,
                "layers": len(self._layers),
                "dtype": None if parameter is None else str(parameter.dtype),
                "device": None if parameter is None else str(parameter.device),
            },
            "ports": list(self.ports),
            "budget": {"max_bytes": self.max_bytes, "held_bytes": self._held},
            "calls_total": len(self._calls),
            "calls": entries,
        }
        try:
            body = check_bounded(body, "trace record")
            return check_bounded({**body, "seal": seal_digest(body)}, "trace record")
        except RecordError as error:
            raise Refused(f"{error}; pick fewer calls with record(calls=...)") from None


def _first(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The module's input tensor: its first positional argument or ``hidden_states``."""
    return args[0] if args else kwargs.get("hidden_states")


def trace(
    model: Any,
    ports: Iterable[str],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    project: Any = None,
) -> Trace:
    """Passive capture of ``ports`` during any forward run inside ``with trace(...) as t:``.

    See the ``mdb.engine.trace`` module docstring. Raises ``BadArgs`` for a malformed or
    unknown address and ``Refused`` for an unsupported model, a non-None ``project`` or a
    second active trace.
    """
    return Trace(model, ports, max_bytes=max_bytes, project=project)
