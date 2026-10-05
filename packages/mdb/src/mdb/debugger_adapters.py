"""Typed debugger adapter contracts for Saturn's native runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from .debugger_algebra import evaluate_algebra
from .debugger_protocol import (
    ActSpec,
    BreakpointSpec,
    DebuggerAddressSpec,
    DebuggerProtocolError,
    DebuggerStructSpec,
    WatchSpec,
)

DEFAULT_ALGEBRA_OPS = (
    "read",
    "ref",
    "value",
    "copy",
    "delta",
    "add",
    "sub",
    "scale",
    "mix",
    "mean",
    "norm",
    "clip",
    "abs",
    "zeros_like",
)


def _address_specs(values: Any) -> tuple[DebuggerAddressSpec, ...]:
    return tuple(
        value if isinstance(value, DebuggerAddressSpec) else DebuggerAddressSpec.from_dict(value)
        for value in (values or ())
    )


def _struct_specs(values: Any) -> tuple[DebuggerStructSpec, ...]:
    return tuple(
        value if isinstance(value, DebuggerStructSpec) else DebuggerStructSpec.from_dict(value)
        for value in (values or ())
    )


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    family: str
    boundaries: tuple[str, ...]
    inspectable_slots: tuple[str, ...]
    writable_slots: tuple[str, ...]
    mutable_closure: tuple[str, ...]
    exact_replay: bool
    state_schema_id: str
    backend: str
    consumer: str = "native"
    boundary_catalog: tuple[Mapping[str, Any], ...] = ()
    address_specs: tuple[DebuggerAddressSpec, ...] = ()
    structs: tuple[DebuggerStructSpec, ...] = ()
    algebra_ops: tuple[str, ...] = DEFAULT_ALGEBRA_OPS
    checkpoint_requires_pause: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "boundaries": list(self.boundaries),
            "inspectable_slots": list(self.inspectable_slots),
            "writable_slots": list(self.writable_slots),
            "mutable_closure": list(self.mutable_closure),
            "exact_replay": self.exact_replay,
            "state_schema_id": self.state_schema_id,
            "backend": self.backend,
            "consumer": self.consumer,
            "boundary_catalog": [dict(item) for item in self.boundary_catalog],
            "address_specs": [item.to_dict() for item in self.address_specs],
            "structs": [item.to_dict() for item in self.structs],
            "algebra_ops": list(self.algebra_ops),
            "checkpoint_requires_pause": self.checkpoint_requires_pause,
        }


class DebuggerAdapter(ABC):
    """Runtime-owned hooks; the control plane never mutates a model directly."""

    capabilities: AdapterCapabilities

    def _runtime_structs(self) -> tuple[DebuggerStructSpec, ...]:
        return tuple(getattr(self, "_registered_structs", ()))

    def _runtime_addresses(self) -> tuple[DebuggerAddressSpec, ...]:
        return tuple(getattr(self, "_registered_addresses", ()))

    def structs(self) -> tuple[DebuggerStructSpec, ...]:
        return self.capabilities.structs + self._runtime_structs()

    def address_specs(self) -> tuple[DebuggerAddressSpec, ...]:
        specs = list(self.capabilities.address_specs)
        for struct in self.structs():
            specs.extend(struct.address_specs)
        specs.extend(self._runtime_addresses())
        return tuple(specs)

    def address_info(self, path: str) -> tuple[DebuggerAddressSpec, dict[str, Any]] | None:
        for spec in self.address_specs():
            fields = spec.match(path)
            if fields is not None:
                return spec, fields
        return None

    def capabilities_payload(self) -> dict[str, Any]:
        payload = self.capabilities.to_dict()
        payload["structs"] = [item.to_dict() for item in self.structs()]
        payload["address_specs"] = [item.to_dict() for item in self.address_specs()]
        return payload

    @abstractmethod
    def status(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def inspect(self, path: str | None = None) -> Mapping[str, Any]: ...

    @abstractmethod
    def checkpoint(self, name: str | None = None) -> Mapping[str, Any]: ...

    @abstractmethod
    def restore(self, cut_id: str) -> Mapping[str, Any]: ...

    @abstractmethod
    def preview_act(self, act: ActSpec) -> Mapping[str, Any]: ...

    @abstractmethod
    def commit_act(self, act_id: str) -> Mapping[str, Any]: ...

    @abstractmethod
    def abort_act(self, act_id: str) -> Mapping[str, Any]: ...

    @abstractmethod
    def replay(self, cut_id: str) -> Mapping[str, Any]: ...

    @abstractmethod
    def step(self, count: int = 1) -> Mapping[str, Any]: ...

    @abstractmethod
    def continue_run(self) -> Mapping[str, Any]: ...

    def validate_breakpoint(self, spec: BreakpointSpec) -> None:
        spec.to_dict()
        if spec.kind not in self.capabilities.boundaries and spec.kind != "if":
            raise DebuggerProtocolError(
                f"adapter {self.capabilities.family} does not expose {spec.kind} boundaries"
            )

    def validate_watch(self, spec: WatchSpec) -> None:
        self.validate_read(spec.path)

    def validate_read(self, path: str | None) -> None:
        if path is None:
            return
        if not isinstance(path, str) or not path.strip():
            raise DebuggerProtocolError("inspect path must be a non-empty string")
        if path not in self.capabilities.inspectable_slots and not any(
            path.startswith(slot + ".") for slot in self.capabilities.inspectable_slots
        ):
            address = self.address_info(path)
            if address is None or not address[0].readable:
                candidates = list(self.capabilities.inspectable_slots)
                candidates.extend(
                    spec.path_template for spec in self.address_specs() if spec.readable
                )
                hint = get_close_matches(path, candidates, n=3, cutoff=0.25)
                suffix = f"; try {', '.join(hint)}" if hint else ""
                raise DebuggerProtocolError(f"path is not inspectable: {path}{suffix}")

    def validate_write(self, act: ActSpec) -> None:
        address = self.address_info(act.path)
        if act.path not in self.capabilities.writable_slots and not any(
            act.path.startswith(slot + ".") for slot in self.capabilities.writable_slots
        ):
            if address is None or not address[0].writable:
                writable = list(self.capabilities.writable_slots)
                writable.extend(
                    spec.path_template for spec in self.address_specs() if spec.writable
                )
                hint = get_close_matches(act.path, writable, n=4, cutoff=0.2)
                suffix = f"; writable addresses: {', '.join(hint)}" if hint else ""
                raise DebuggerProtocolError(
                    f"path is read-only or outside the declared mutable closure: {act.path}{suffix}"
                )
        if address is not None:
            declared, expected = address
            if not declared.writable:
                raise DebuggerProtocolError(
                    f"path is read-only or outside the declared mutable closure: {act.path}"
                )
            if act.struct_name and declared.struct_name != act.struct_name:
                raise DebuggerProtocolError(
                    f"Act struct {act.struct_name!r} does not own logical path {act.path!r}"
                )
            if act.address:
                for key, value in act.address.items():
                    if key in expected and str(expected[key]) != str(value):
                        raise DebuggerProtocolError(
                            f"Act address field {key!r} disagrees with logical path {act.path!r}"
                        )
            source_write = declared.source_write
            if source_write.get("schema") == "saturn-debugger-semantic-binding-v1":
                required_identity = {
                    "semantic_symbol": source_write.get("symbol"),
                    "binding_fingerprint": source_write.get("binding_fingerprint"),
                    "manifest_fingerprint": source_write.get("manifest_fingerprint"),
                    "context_fingerprint": source_write.get("context_fingerprint"),
                }
                for key, expected_value in required_identity.items():
                    observed = act.address.get(key)
                    if expected_value is None or expected_value == "":
                        raise DebuggerProtocolError(
                            f"semantic address declaration is missing {key!r}"
                        )
                    if observed is None or observed == "":
                        raise DebuggerProtocolError(
                            f"semantic Act requires address identity field {key!r}"
                        )
                    if str(observed) != str(expected_value):
                        raise DebuggerProtocolError(
                            f"semantic Act {key!r} does not match the active binding"
                        )
        elif act.struct_name:
            raise DebuggerProtocolError(
                f"Act struct {act.struct_name!r} has no declared logical address: {act.path}"
            )

    def resolve_value(self, path: str) -> Any:
        """Resolve a raw native value for safe worker-side algebra."""

        raise DebuggerProtocolError(
            f"adapter {self.capabilities.family} does not expose raw algebra reads: {path}"
        )

    def evaluate_expression(self, expression: str) -> Any:
        def resolve_read(path: str) -> Any:
            self.validate_read(path)
            return self.resolve_value(path)

        return evaluate_algebra(expression, resolve_read)

    def register_struct(self, spec: DebuggerStructSpec) -> Mapping[str, Any]:
        hook = self._optional_hook("register_struct")
        if hook is None and any(address.writable for address in spec.address_specs):
            raise DebuggerProtocolError(
                "writable custom structs require a native register_struct hook"
            )
        if hook is not None:
            result = hook(spec)
            if not isinstance(result, Mapping):
                raise DebuggerProtocolError("register_struct hook must return an object")
        existing = {item.name for item in self.structs()}
        if spec.name in existing:
            raise DebuggerProtocolError(f"struct is already registered: {spec.name}")
        self._registered_structs = (*self._runtime_structs(), spec)
        return {"registered": True, "struct": spec.to_dict()}

    def register_address(self, spec: DebuggerAddressSpec) -> Mapping[str, Any]:
        hook = self._optional_hook("register_address")
        if hook is None and spec.writable:
            raise DebuggerProtocolError(
                "writable custom addresses require a native register_address hook"
            )
        if hook is not None:
            result = hook(spec)
            if not isinstance(result, Mapping):
                raise DebuggerProtocolError("register_address hook must return an object")
        if any(item.name == spec.name for item in self.address_specs()):
            raise DebuggerProtocolError(f"address is already registered: {spec.name}")
        self._registered_addresses = (*self._runtime_addresses(), spec)
        return {"registered": True, "address": spec.to_dict()}

    def _optional_hook(self, name: str) -> Any:
        callbacks = getattr(self, "_callbacks", {})
        callback = callbacks.get(name) if isinstance(callbacks, Mapping) else None
        if callable(callback):
            return callback
        delegate = getattr(self, "_delegate", None)
        candidate = getattr(delegate, name, None)
        return candidate if callable(candidate) else None

    def optional_operation(self, name: str, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Call an explicitly installed research/debugger hook or refuse.

        Optional hooks are how a runtime can add Future Forest, Thott, xref,
        or native-consumer-specific behavior without making those semantics
        part of the common adapter ABI.
        """

        if name in {"measure_circuit", "replay_circuit"} and self.capabilities.family not in {
            "autoregressive",
            "qwen2",
            "mamba",
        }:
            raise DebuggerProtocolError(
                f"adapter {self.capabilities.family} does not expose debugger operation: {name}"
            )

        hook = self._optional_hook(name)
        if hook is None:
            raise DebuggerProtocolError(
                f"adapter {self.capabilities.family} does not expose debugger operation: {name}"
            )
        result = hook(*args, **kwargs)
        if not isinstance(result, Mapping):
            raise DebuggerProtocolError(f"debugger operation {name} must return an object")
        return result

    def optional_operations(self) -> tuple[str, ...]:
        """Return only optional operations backed by an installed native hook."""

        names = (
            "step_kind",
            "diff",
            "why",
            "xref",
            "reverse_continue",
            "first_divergence",
            "bisect",
            "attribute",
            "promote",
            "measure_circuit",
            "replay_circuit",
            "register_struct",
            "register_address",
        )
        circuit_ops = {"measure_circuit", "replay_circuit"}
        return tuple(
            name
            for name in names
            if self._optional_hook(name) is not None
            and (
                name not in circuit_ops
                or self.capabilities.family in {"autoregressive", "qwen2", "mamba"}
            )
        )


class CallbackDebuggerAdapter(DebuggerAdapter):
    """Small bridge for native workers that already own runtime state.

    Each callback executes inside the worker process.  This keeps tensor/state
    ownership in the native runtime while making the debugger protocol reusable
    across Qwen, Mamba, and FLUX implementations.
    """

    def __init__(self, capabilities: AdapterCapabilities, **callbacks: Any) -> None:
        self.capabilities = capabilities
        self._callbacks = callbacks

    def _call(self, name: str, *args: Any) -> Mapping[str, Any]:
        callback = self._callbacks.get(name)
        if not callable(callback):
            raise DebuggerProtocolError(f"debugger callback is not installed: {name}")
        result = callback(*args)
        if not isinstance(result, Mapping):
            raise DebuggerProtocolError(f"debugger callback {name} must return an object")
        return result

    def resolve_value(self, path: str) -> Any:
        callback = self._callbacks.get("resolve_value")
        if not callable(callback):
            raise DebuggerProtocolError("debugger callback is not installed: resolve_value")
        return callback(path)

    def status(self) -> Mapping[str, Any]:
        return self._call("status")

    def inspect(self, path: str | None = None) -> Mapping[str, Any]:
        return self._call("inspect", path)

    def checkpoint(self, name: str | None = None) -> Mapping[str, Any]:
        return self._call("checkpoint", name)

    def restore(self, cut_id: str) -> Mapping[str, Any]:
        return self._call("restore", cut_id)

    def preview_act(self, act: ActSpec) -> Mapping[str, Any]:
        return self._call("preview_act", act)

    def commit_act(self, act_id: str) -> Mapping[str, Any]:
        return self._call("commit_act", act_id)

    def abort_act(self, act_id: str) -> Mapping[str, Any]:
        return self._call("abort_act", act_id)

    def replay(self, cut_id: str) -> Mapping[str, Any]:
        return self._call("replay", cut_id)

    def step(self, count: int = 1) -> Mapping[str, Any]:
        return self._call("step", count)

    def continue_run(self) -> Mapping[str, Any]:
        return self._call("continue_run")


class AutoregressiveDebuggerAdapter(DebuggerAdapter):
    """Capability declaration for token/layer/KV StateCut runtimes."""

    def __init__(self, **kwargs: Any) -> None:
        boundary_catalog = tuple(
            kwargs.get("boundary_catalog")
            or (
                {
                    "kind": "token",
                    "label": "committed/generated token boundary",
                    "index_semantics": "token cursor after native token commit",
                },
                {
                    "kind": "layer",
                    "label": "decoder layer boundary",
                    "index_semantics": "native decoder layer index",
                },
                {
                    "kind": "sequence",
                    "label": "sequence cursor boundary",
                    "index_semantics": "runtime-defined sequence cursor",
                },
                {
                    "kind": "expert",
                    "label": "expert/router boundary",
                    "index_semantics": "runtime-defined expert dispatch index",
                },
            )
        )
        self.capabilities = AdapterCapabilities(
            family="autoregressive",
            boundaries=("token", "layer", "sequence", "expert"),
            inspectable_slots=(
                "token_ids",
                "attention_mask",
                "kv_cache",
                "logits",
                "cursor",
                "route",
            ),
            writable_slots=("token_ids", "kv_cache", "cursor", "route"),
            mutable_closure=("token_ids", "attention_mask", "kv_cache", "cursor", "route"),
            exact_replay=True,
            state_schema_id="saturn-ar-debugger-state-v1",
            backend=str(kwargs.get("backend", "unknown")),
            consumer="native-decoder",
            boundary_catalog=boundary_catalog,
            address_specs=_address_specs(kwargs.get("address_specs")),
            structs=_struct_specs(kwargs.get("structs")),
            algebra_ops=tuple(kwargs.get("algebra_ops") or DEFAULT_ALGEBRA_OPS),
        )
        self._delegate = kwargs.get("delegate")

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        if self._delegate is None or not hasattr(self._delegate, name):
            raise DebuggerProtocolError(f"autoregressive adapter hook is not installed: {name}")
        return getattr(self._delegate, name)(*args, **kwargs)

    def status(self) -> Mapping[str, Any]:
        return self._call("status")

    def inspect(self, path: str | None = None) -> Mapping[str, Any]:
        return self._call("inspect", path)

    def checkpoint(self, name: str | None = None) -> Mapping[str, Any]:
        return self._call("checkpoint", name)

    def restore(self, cut_id: str) -> Mapping[str, Any]:
        return self._call("restore", cut_id)

    def preview_act(self, act: ActSpec) -> Mapping[str, Any]:
        return self._call("preview_act", act)

    def commit_act(self, act_id: str) -> Mapping[str, Any]:
        return self._call("commit_act", act_id)

    def abort_act(self, act_id: str) -> Mapping[str, Any]:
        return self._call("abort_act", act_id)

    def replay(self, cut_id: str) -> Mapping[str, Any]:
        return self._call("replay", cut_id)

    def step(self, count: int = 1) -> Mapping[str, Any]:
        return self._call("step", count)

    def continue_run(self) -> Mapping[str, Any]:
        return self._call("continue_run")

    def resolve_value(self, path: str) -> Any:
        if self._delegate is None or not hasattr(self._delegate, "resolve_value"):
            raise DebuggerProtocolError(
                f"autoregressive adapter hook is not installed: resolve_value ({path})"
            )
        return self._delegate.resolve_value(path)


class FluxDebuggerAdapter(AutoregressiveDebuggerAdapter):
    """Capability declaration for denoise-step/latent/scheduler runtimes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        conditioner_row_write = bool(kwargs.get("conditioner_row_write", False))
        conditioner_streams = tuple(
            str(value) for value in kwargs.get("conditioner_streams", ("text",))
        )
        conditioner_sites = tuple(str(value) for value in kwargs.get("conditioner_sites", ()))
        conditioner_steps = kwargs.get("conditioner_steps")
        step_descriptor: dict[str, Any] = {"type": "step", "minimum": 0}
        if conditioner_steps is not None:
            step_count = int(conditioner_steps)
            if step_count < 1:
                raise DebuggerProtocolError("conditioner_steps must be positive")
            step_descriptor["maximum"] = step_count - 1
        conditioner_address = DebuggerAddressSpec(
            name="conditioner-row",
            kind="conditioner-row",
            path_template="conditioner[{stream}][{site}][step={step}][row={row}]",
            readable=True,
            writable=conditioner_row_write,
            fields={
                "stream": {"type": "name", "choices": list(conditioner_streams)},
                "site": {
                    "type": "string",
                    **({"choices": list(conditioner_sites)} if conditioner_sites else {}),
                },
                "step": step_descriptor,
                "row": {"type": "row", "minimum": 0},
            },
            boundaries=("step", "layer"),
            invalidates=("conditioner", "native-denoiser-suffix", "decoder"),
            numerical_contract="exact-native-conditioner-row-v1",
        )
        address_specs = _address_specs(kwargs.get("address_specs"))
        if conditioner_row_write or kwargs.get("conditioner_row_inspectable", False):
            address_specs = (conditioner_address, *address_specs)
        structs = _struct_specs(kwargs.get("structs"))
        self.capabilities = AdapterCapabilities(
            family="flux",
            boundaries=("step", "layer", "sequence"),
            inspectable_slots=("latent", "scheduler", "conditioner", "route", "step", "rng"),
            writable_slots=(
                ("latent", "scheduler", "route", "step", "conditioner.row")
                if conditioner_row_write
                else ("latent", "scheduler", "route", "step")
            ),
            mutable_closure=("latent", "scheduler", "conditioner", "route", "step", "rng"),
            exact_replay=bool(kwargs.get("exact_replay", True)),
            state_schema_id=str(
                kwargs.get("state_schema_id", "saturn-flux-debugger-state-v1")
            ),
            backend=kwargs.get("backend", "unknown"),
            consumer="native-denoiser",
            boundary_catalog=tuple(
                kwargs.get("boundary_catalog")
                or (
                    {
                        "kind": "step",
                        "label": "native denoise/scheduler step",
                        "index_semantics": "scheduler step before the next denoiser invocation",
                    },
                    {
                        "kind": "layer",
                        "label": "transformer layer boundary",
                        "index_semantics": "native transformer layer index",
                    },
                    {
                        "kind": "sequence",
                        "label": "sequence/trajectory boundary",
                        "index_semantics": "runtime-defined sequence cursor",
                    },
                )
            ),
            address_specs=address_specs,
            structs=structs,
            algebra_ops=tuple(kwargs.get("algebra_ops") or DEFAULT_ALGEBRA_OPS),
        )
