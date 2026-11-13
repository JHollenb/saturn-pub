"""A bounded, structural debugger for one eager ``torch.fx`` forward graph.

This module is a small bridge between an ordinary PyTorch module and the public
``saturn_pub.Session`` executor.  FX tracing records the Python level graph
without running the caller's numerical inputs.  The resulting graph module is
then executed one original FX node at a time by a Saturn-pub ``Adapter``.

The names emitted here (``v000``, ``v001`` and so on) identify graph values;
they are deliberately not semantic or machine-language labels.  A model's
instruction, reasoning, or ISA meaning requires a separate native consumer.
The adapter accepts only a declared frozen closure and fails closed for known
in-place, stochastic, or state-mutating operations.  FX cannot prove arbitrary
Python side effects that execute during tracing, so unproven graph constants
are rejected rather than called deterministic.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from saturn_pub.cli import Debugger
from saturn_pub.contracts import (
    ExecutionPoint,
    NumericalContract,
    SlotSpec,
    StateAddress,
    SurfaceManifest,
    TransitionSpec,
)
from saturn_pub.core import Adapter, Session
from saturn_pub.symbols import SymbolBinding, SymbolTable
from saturn_pub.values import canonical, clone, describe, digest
from saturn_pub.values import tensor as is_tensor

MAX_GRAPH_NODES = 512
MAX_STATE_BYTES = 256 << 20

_MODULE_STRUCTURAL_ATTRS = frozenset({
    "_buffers",
    "_code",
    "_non_persistent_buffers_set",
    "_parameters",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_create_node_hooks",
    "_deepcopy_hooks",
    "_erase_node_hooks",
    "_graph",
    "_is_full_backward_hook",
    "_lineno_map",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_modules",
    "_prologue_start",
    "_replace_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_tracer_cls",
    "_tracer_extras",
    "training",
    "_version",
    "call_super_init",
    "_compiled_call_impl",
    "_wrapped_call_impl",
    "meta",
    "shape_env",
})


class ForwardDebuggerError(ValueError):
    """The graph cannot be represented as a frozen, exact public session."""


class ForwardDebugger(Debugger):
    """Live CLI view over a :class:`ForwardAdapter` session.

    Base stepping, checkpoint, restore, branching, and observation commands
    remain Saturn-pub ``Debugger`` behavior.  Structural symbol tables are
    refreshed only when ``info symbols`` is requested, so stepping does not
    perform hidden symbol I/O.
    """

    def __init__(self, session: Session) -> None:
        if not isinstance(session.adapter, ForwardAdapter):
            raise TypeError("ForwardDebugger requires a ForwardAdapter session")
        super().__init__(session)
        self._annotations: list[SymbolBinding] = []

    def attach_annotations(self, symbols: SymbolTable) -> None:
        """Attach separately measured read-only labels at their qualified cut.

        These provenance-bearing annotations never enter model execution.
        A declared observation status is not semantic certification.
        """

        if not isinstance(symbols, SymbolTable):
            raise TypeError("annotations require a public SymbolTable")
        for binding in symbols.bindings:
            if binding.status not in {"observed", "validated"}:
                raise ValueError("attach annotations only after recording their observation evidence")
            binding.resolve(self.session, context=self.session.adapter.context)
        indexed = {row.to_dict()["fingerprint"]: row for row in self._annotations}
        indexed.update({row.to_dict()["fingerprint"]: row for row in symbols.bindings})
        self._annotations = list(indexed.values())

    def execute(self, line: str) -> Any:
        command = shlex.split(line)
        if command == ["info", "symbols"]:
            self.symbols = auto_symbol_table(self.session)
            for binding in self._annotations:
                try:
                    binding.resolve(self.session, context=self.session.adapter.context)
                except ValueError:
                    continue  # A different cut/input does not inherit this label.
                self.symbols.bindings.append(binding)
            self.symbol_context = self.session.adapter.context
            return super().execute(line)
        if command == ["info", "forward"]:
            return execution_table(self.session)
        return super().execute(line)


@dataclass(frozen=True)
class _NodeSpec:
    name: str
    op: str
    target: str
    slot: str
    reads: tuple[str, ...]
    users: tuple[str, ...]
    order: int | None
    placeholder: bool = False
    output: bool = False


def _target_text(target: Any) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", getattr(target, "__name__", None))
    if name is not None:
        return f"{module}.{name}" if module else str(name)
    return repr(target)


def _is_node(value: Any) -> bool:
    return hasattr(value, "name") and hasattr(value, "op") and hasattr(value, "users")


def _supported_value(value: Any, *, depth: int = 0) -> bool:
    if depth > 16:
        return False
    if is_tensor(value) or value is None or type(value) in (bool, int, float, str):
        return True
    if isinstance(value, Mapping):
        return all(_supported_value(key, depth=depth + 1) and _supported_value(
            item, depth=depth + 1
        ) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(_supported_value(item, depth=depth + 1) for item in value)
    return False


def _supported_module_attr(value: Any) -> bool:
    """Allow immutable scalar sets used as module configuration metadata."""

    if isinstance(value, (set, frozenset)):
        return all(
            item is None or type(item) in (bool, int, float, str)
            for item in value
        )
    return _supported_value(value)


def _describe_module_attr(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        items = [describe(item) for item in value]
        items.sort(key=canonical)
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": items,
        }
    return describe(value)


def _value_bytes(value: Any, *, depth: int = 0) -> int:
    """Bound the retained forward closure without importing torch eagerly."""

    if depth > 16:
        raise ForwardDebuggerError("FX value nesting exceeds the public closure bound")
    if is_tensor(value):
        try:
            return int(value.numel()) * int(value.element_size())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ForwardDebuggerError("FX tensor lacks a bounded dense geometry") from exc
    if isinstance(value, Mapping):
        return sum(
            _value_bytes(key, depth=depth + 1) + _value_bytes(item, depth=depth + 1)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return sum(_value_bytes(item, depth=depth + 1) for item in value)
    return 0


def _module_state_digest(module: Any) -> str:
    try:
        state = module.state_dict()
        parameters = {
            name: {
                "value": describe(value),
                "requires_grad": bool(getattr(value, "requires_grad", False)),
            }
            for name, value in module.named_parameters()
        }
        # named_buffers includes non-persistent buffers that state_dict omits.
        buffers = {
            name: describe(value)
            for name, value in module.named_buffers()
        }
        attrs: dict[str, Any] = {}
        training: dict[str, bool] = {}
        for prefix, child in module.named_modules():
            training[prefix] = bool(getattr(child, "training", False))
            for name, value in vars(child).items():
                if name in _MODULE_STRUCTURAL_ATTRS:
                    continue
                if not _supported_module_attr(value):
                    path = f"{prefix}.{name}" if prefix else name
                    raise ForwardDebuggerError(
                        f"module attribute {path!r} is outside the supported immutable closure"
                    )
                path = f"{prefix}.{name}" if prefix else name
                attrs[path] = _describe_module_attr(value)
        return digest({
            "state_dict": describe(state),
            "named_parameters": parameters,
            "named_buffers": buffers,
            "module_attributes": attrs,
            "training": training,
        })
    except Exception as exc:  # noqa: BLE001 - unsupported state is a hard boundary
        if isinstance(exc, ForwardDebuggerError):
            raise
        raise ForwardDebuggerError(
            f"module state cannot be sealed for exact replay: {type(exc).__name__}"
        ) from exc


def _node_refs(value: Any, slots: Mapping[str, str]) -> tuple[str, ...]:
    found: list[str] = []

    def visit(item: Any) -> None:
        if _is_node(item):
            slot = slots.get(str(item.name))
            if slot is None:
                raise ForwardDebuggerError(f"FX node reference is outside the traced graph: {item}")
            if slot not in found:
                found.append(slot)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found)


def _map_nodes(value: Any, state: Mapping[str, Any], slots: Mapping[str, str]) -> Any:
    if _is_node(value):
        return state[slots[str(value.name)]]
    if isinstance(value, Mapping):
        return type(value)((_map_nodes(k, state, slots), _map_nodes(v, state, slots))
                           for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(_map_nodes(item, state, slots) for item in value)
    if isinstance(value, list):
        return [_map_nodes(item, state, slots) for item in value]
    return value


def _graph_fingerprint(graph_module: Any, model_identity: str, nodes: Sequence[Any]) -> str:
    rows = []
    for node in nodes:
        rows.append({
            "name": str(node.name),
            "op": str(node.op),
            "target": _target_text(node.target),
            "args": repr(node.args),
            "kwargs": repr(node.kwargs),
        })
    return digest({
        "schema": "mdb-fx-forward-graph-v1",
        "model_identity": model_identity,
        "code": str(getattr(graph_module, "code", "")),
        "nodes": rows,
    })


def _stochastic_name(value: Any) -> bool:
    text = _target_text(value).casefold()
    return any(token in text for token in (
        "dropout", "bernoulli", "multinomial", "rand", "random", "poisson",
        "normal_", "uniform_",
    ))


def _inplace_name(value: Any) -> bool:
    """Recognize in-place suffixes and Python's mutating operator targets."""

    leaf = _target_text(value).casefold().rsplit(".", 1)[-1]
    augmented = {
        "iadd", "isub", "imul", "imatmul", "itruediv", "ifloordiv", "imod",
        "ipow", "ilshift", "irshift", "iand", "ixor", "ior",
    }
    if leaf in augmented or leaf in {f"__{name}__" for name in augmented}:
        return True
    if leaf in {"__setitem__", "__delitem__", "setitem", "delitem"}:
        return True
    return leaf.endswith("_") and not leaf.startswith("__")


def _preflight_graph(graph_module: Any, nodes: Sequence[Any]) -> None:
    for module in graph_module.modules():
        if bool(getattr(module, "training", False)):
            raise ForwardDebuggerError(
                "FX forward requires every module to be in eval mode; refusing a mutable training graph"
            )
        if _stochastic_name(type(module)):
            raise ForwardDebuggerError(
                f"stochastic module is outside the exact debugger closure: {type(module).__name__}"
            )
    for node in nodes:
        if node.op == "call_method" and _inplace_name(node.target):
            raise ForwardDebuggerError(f"in-place FX method is not supported: {node.target}")
        if node.op == "call_function" and _inplace_name(node.target):
            raise ForwardDebuggerError(f"in-place FX function is not supported: {_target_text(node.target)}")
        if node.op == "call_module":
            submodule = graph_module.get_submodule(str(node.target))
            if bool(getattr(submodule, "inplace", False)):
                raise ForwardDebuggerError(
                    f"in-place FX module is not supported: {node.target}"
                )
        if node.op in {"call_function", "call_module"} and _stochastic_name(node.target):
            raise ForwardDebuggerError(
                f"stochastic FX target is outside the exact debugger closure: {_target_text(node.target)}"
            )


def _module_attribute_paths(module: Any) -> frozenset[str]:
    """Return source-owned attrs that FX may legally expose through ``get_attr``."""

    paths = {
        name for name, _value in module.named_parameters()
    }
    paths.update(name for name, _value in module.named_buffers())
    for prefix, child in module.named_modules():
        for name, value in vars(child).items():
            if name in _MODULE_STRUCTURAL_ATTRS or not _supported_module_attr(value):
                continue
            paths.add(f"{prefix}.{name}" if prefix else name)
    return frozenset(paths)


def _reject_unproven_graph_constants(
    graph_module: Any, nodes: Sequence[Any], allowed: frozenset[str]
) -> None:
    for node in nodes:
        if node.op == "get_attr" and str(node.target) not in allowed:
            raise ForwardDebuggerError(
                "FX graph contains an unproven get_attr constant; "
                "trace-time side effects cannot be certified"
            )


def _trace(module: Any, concrete_args: Mapping[str, Any] | None) -> Any:
    try:
        import torch.fx as fx
    except ImportError as exc:  # pragma: no cover - package extra is environment-specific
        raise ForwardDebuggerError("the forward debugger requires the public torch.fx extra") from exc
    if not hasattr(module, "forward") or not callable(module.forward):
        raise ForwardDebuggerError("forward debugger requires a callable module")
    if any(bool(getattr(child, "training", False)) for child in module.modules()):
        raise ForwardDebuggerError(
            "FX forward requires every source module to be in eval mode; refusing a mutable training graph"
        )
    allowed_attrs = _module_attribute_paths(module)
    before = _module_state_digest(module)
    try:
        graph_module = fx.symbolic_trace(module) if concrete_args is None else fx.symbolic_trace(
            module, concrete_args=dict(concrete_args)
        )
    except Exception as exc:  # noqa: BLE001 - tracing failures are explicit graph limits
        raise ForwardDebuggerError(
            f"FX tracing could not produce a bounded graph: {type(exc).__name__}: {exc}"
        ) from exc
    _reject_unproven_graph_constants(
        graph_module, tuple(graph_module.graph.nodes), allowed_attrs
    )
    if _module_state_digest(module) != before:
        raise ForwardDebuggerError("FX tracing mutated module state; exact closure is unavailable")
    # symbolic_trace may create plain container Modules whose default training
    # flag is True even when every source module was evaluated.  This changes
    # only the synthetic graph container flags, after source-state validation.
    graph_module.eval()
    return graph_module


def _bind_inputs(
    graph_module: Any,
    args: Any,
    concrete_args: Mapping[str, Any] | None,
    placeholders: Sequence[Any],
) -> dict[str, Any]:
    if isinstance(args, Mapping):
        supplied = dict(args)
    else:
        if args is None:
            positional = ()
        elif isinstance(args, tuple):
            positional = args
        elif isinstance(args, list):
            positional = tuple(args)
        else:
            positional = (args,)
        if len(positional) > len(placeholders):
            raise ForwardDebuggerError("too many forward inputs for the traced placeholders")
        supplied = {
            str(node.target): value
            for node, value in zip(placeholders, positional, strict=False)
        }
    if concrete_args:
        for key, value in concrete_args.items():
            supplied.setdefault(str(key), value)
    bound: dict[str, Any] = {}
    for node in placeholders:
        name = str(node.target)
        value = supplied.get(name, supplied.get(str(node.name)))
        if name not in supplied and str(node.name) not in supplied:
            # FX gives specialized concrete placeholders names such as ``flag_1``.
            base = name.removesuffix("_1")
            if base in supplied:
                value = supplied[base]
            else:
                raise ForwardDebuggerError(f"no value supplied for FX placeholder {name!r}")
        if not _supported_value(value):
            raise ForwardDebuggerError(f"unsupported input closure value for FX placeholder {name!r}")
        bound[str(node.name)] = clone(value)
    return bound


class ForwardAdapter(Adapter):
    """Public Saturn-pub adapter that executes one original FX node per transition."""

    def __init__(
        self,
        graph_module: Any,
        args: Any,
        *,
        model_identity: str,
        concrete_args: Mapping[str, Any] | None = None,
    ) -> None:
        if type(model_identity) is not str or not model_identity.strip():
            raise ForwardDebuggerError("model_identity must be non-empty text")
        if not hasattr(graph_module, "graph") or not hasattr(graph_module, "get_submodule"):
            raise ForwardDebuggerError("ForwardAdapter requires a torch.fx.GraphModule")
        all_nodes = tuple(graph_module.graph.nodes)
        if not all_nodes or len(all_nodes) > MAX_GRAPH_NODES:
            raise ForwardDebuggerError(
                f"FX graph node count must be between 1 and {MAX_GRAPH_NODES}"
            )
        _preflight_graph(graph_module, all_nodes)
        placeholders = tuple(node for node in all_nodes if node.op == "placeholder")
        self.graph_module = graph_module
        self.model_identity = model_identity
        self._all_nodes = all_nodes
        self._node_slots = {
            str(node.name): f"v{index:03d}"
            for index, node in enumerate(node for node in all_nodes if node.op != "output")
        }
        self._bound_inputs = _bind_inputs(graph_module, args, concrete_args, placeholders)
        self._graph_fingerprint = _graph_fingerprint(graph_module, model_identity, all_nodes)
        self.context = f"fx:{model_identity}:{self._graph_fingerprint}"
        self._initial_module_state = _module_state_digest(graph_module)
        self._node_specs: list[_NodeSpec] = []
        self._steps: list[Any] = []
        step_index = 0
        for node in all_nodes:
            name = str(node.name)
            if node.op == "output":
                reads = _node_refs(node.args, self._node_slots)
                spec = _NodeSpec(name, "output", "output", "output", reads,
                                 (), step_index, output=True)
                self._node_specs.append(spec)
                self._steps.append(node)
                step_index += 1
                continue
            slot = self._node_slots[name]
            reads = _node_refs(node.args, self._node_slots) + _node_refs(
                node.kwargs, self._node_slots
            )
            reads = tuple(dict.fromkeys(reads))
            users = tuple(str(user.name) for user in node.users)
            order = None if node.op == "placeholder" else step_index
            spec = _NodeSpec(
                name, str(node.op), _target_text(node.target), slot, reads, users, order,
                placeholder=node.op == "placeholder",
            )
            self._node_specs.append(spec)
            if node.op != "placeholder":
                self._steps.append(node)
                step_index += 1
        if not self._steps or not self._steps[-1].op == "output":
            raise ForwardDebuggerError("FX graph has no output boundary")
        self._spec_by_name = {spec.name: spec for spec in self._node_specs}
        self.execution = {
            "family": "torch-fx",
            "adapter": "mdb-forward-debugger-v1",
            "operation_level": "native-neural-graph",
            "model_identity": model_identity,
            "graph_fingerprint": self._graph_fingerprint,
            "graph_nodes": len(all_nodes),
            "node_execution": "original-graph-node-call",
            "statefulness": "frozen-module-required-known-drift-checked",
            "trace_side_effect_authority": "known-targets-and-source-owned-attrs",
            "semantic_authority": "unresolved-structural-observation",
        }
        self._slot_names = tuple(
            [spec.slot for spec in self._node_specs if not spec.output]
            + ["output", "output_ready", "cursor", "done"]
        )
        if len(set(self._slot_names)) != len(self._slot_names):
            raise ForwardDebuggerError("FX graph generated duplicate state ports")
        self._initial_state = {slot: None for slot in self._slot_names}
        for name, value in self._bound_inputs.items():
            self._initial_state[self._node_slots[name]] = value
        self._initial_state.update({
            "output": None,
            "output_ready": False,
            "cursor": 0,
            "done": False,
        })
        self.validate(self._initial_state)

    @property
    def graph_fingerprint(self) -> str:
        return self._graph_fingerprint

    @property
    def context_slots(self) -> tuple[str, ...]:
        return tuple(self._node_slots[name] for name in self._bound_inputs)

    @property
    def node_specs(self) -> tuple[_NodeSpec, ...]:
        return tuple(self._node_specs)

    def initial_state(self) -> dict[str, Any]:
        return clone(self._initial_state)

    def _step_node(self, cursor: int) -> Any:
        return self._steps[cursor]

    def _current_spec(self, state: Mapping[str, Any]) -> _NodeSpec:
        cursor = state["cursor"]
        if state["done"] or cursor >= len(self._steps):
            return _NodeSpec("output", "output", "output", "output", (), (), cursor, output=True)
        return self._spec_by_name[str(self._step_node(cursor).name)]

    def _check_module_state(self) -> None:
        current = _module_state_digest(self.graph_module)
        if current != self._initial_module_state:
            raise ForwardDebuggerError(
                "forward node mutated module state; stateful execution is outside the exact closure"
            )

    def boundary(self, state: Mapping[str, Any]) -> str:
        return "complete" if state["done"] else "node"

    def point(self, state: Mapping[str, Any]) -> ExecutionPoint:
        spec = self._current_spec(state)
        if state["done"]:
            return ExecutionPoint(
                "torch-fx", int(state["cursor"]), "complete", operator="output",
                index=int(state["cursor"]), edge="after", next_operation="",
                local_clock=int(state["cursor"]),
            )
        return ExecutionPoint(
            "torch-fx", int(state["cursor"]), "node", operator=spec.op,
            index=int(state["cursor"]), edge="before", next_operation=f"fx.{spec.slot}",
            local_clock=int(state["cursor"]),
        )

    def surface(self, state: Mapping[str, Any]) -> SurfaceManifest:
        slots = []
        for spec in self._node_specs:
            if spec.output:
                continue
            writable = self._computed_writable(spec, state)
            slots.append(SlotSpec(
                spec.slot,
                role="input" if spec.placeholder else "computed",
                writable=writable,
                persistence="authoritative" if spec.placeholder else "derived",
                producer=f"fx.{spec.op}:{spec.target}",
                consumer=", ".join(f"fx.{user}" for user in spec.users),
                schema={
                    "kind": "fx-value",
                    "node": spec.name,
                    "op": spec.op,
                    "writable": writable,
                },
            ))
        slots.extend((
            SlotSpec("output", role="observation", persistence="evidence",
                     producer="fx.output", consumer="caller",
                     schema={"kind": "fx-output", "abi": "native-graph-output"}),
            SlotSpec("output_ready", role="clock", persistence="authoritative",
                     producer="runtime", consumer="runtime", schema={"kind": "bool"}),
            SlotSpec("cursor", role="clock", persistence="authoritative",
                     producer="runtime", consumer="runtime", schema={"kind": "integer"}),
            SlotSpec("done", role="clock", persistence="authoritative",
                     producer="runtime", consumer="runtime", schema={"kind": "bool"}),
        ))
        return SurfaceManifest(
            tuple(slots),
            f"mdb-fx-forward-state-v1:{self._graph_fingerprint[:16]}",
            "torch-fx-original-forward",
            "one-original-node-per-transition",
        )

    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        return self._slot_names

    def validate_execution(self) -> None:
        self._check_module_state()

    def _computed_writable(self, spec: _NodeSpec, state: Mapping[str, Any]) -> bool:
        """Allow interventions only on produced tensors with unconsumed users."""

        if spec.placeholder or spec.output or spec.op == "get_attr":
            return False
        value = state.get(spec.slot)
        if not is_tensor(value) or spec.order is None:
            return False
        cursor = int(state["cursor"])
        if cursor <= spec.order:
            return False
        users = [self._spec_by_name.get(name) for name in spec.users]
        users = [user for user in users if user is not None and user.order is not None]
        return bool(users) and all(user.order >= cursor for user in users)

    def _writable_slots(self, state: Mapping[str, Any]) -> frozenset[str]:
        return frozenset(
            spec.slot
            for spec in self._node_specs
            if self._computed_writable(spec, state)
        )

    def validate(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self._slot_names):
            raise ForwardDebuggerError("FX state keys differ from the declared closure")
        cursor = state["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= len(self._steps):
            raise ForwardDebuggerError("FX cursor is outside the traced node schedule")
        if type(state["done"]) is not bool or state["done"] != (cursor == len(self._steps)):
            raise ForwardDebuggerError("FX done flag disagrees with the node cursor")
        if type(state["output_ready"]) is not bool or state["output_ready"] != state["done"]:
            raise ForwardDebuggerError("FX output readiness disagrees with the node cursor")
        for value in state.values():
            if not _supported_value(value):
                raise ForwardDebuggerError("FX state contains an unsupported value type")
        if _value_bytes(state) > MAX_STATE_BYTES:
            raise ForwardDebuggerError(
                f"FX state exceeds the {MAX_STATE_BYTES} byte retained-closure bound"
            )

    def write(
        self, state: Mapping[str, Any], writes: Mapping[str, Any], invalidates: tuple[str, ...]
    ) -> Mapping[str, Any]:
        writable = self._writable_slots(state)
        if set(writes) - writable:
            refused = sorted(set(writes) - writable)
            raise ForwardDebuggerError(
                f"FX intervention can write only produced tensors with future consumers: {refused}"
            )
        for slot, value in writes.items():
            previous = state[slot]
            if not is_tensor(previous) or not is_tensor(value):
                raise ForwardDebuggerError(f"FX intervention requires tensor replacement for {slot}")
            if previous.shape != value.shape or previous.dtype != value.dtype:
                raise ForwardDebuggerError(f"FX intervention shape or dtype mismatch for {slot}")
            if getattr(previous, "device", None) != getattr(value, "device", None):
                raise ForwardDebuggerError(f"FX intervention device mismatch for {slot}")
        return super().write(state, writes, invalidates)

    def transition(self, state: Mapping[str, Any]) -> TransitionSpec:
        if state["done"]:
            raise ForwardDebuggerError("completed FX session has no next transition")
        spec = self._current_spec(state)
        writes = [spec.slot, "cursor", "done", "output_ready"]
        if spec.output:
            writes.insert(0, "output")
        return TransitionSpec(
            f"fx.{spec.slot}",
            spec.reads,
            tuple(dict.fromkeys(writes)),
            consumer="torch-fx-original-forward",
            footprint="declared",
        )

    def _get_attr(self, target: str) -> Any:
        value = self.graph_module
        for part in target.split("."):
            value = getattr(value, part)
        return value

    def _execute_node(self, node: Any, state: Mapping[str, Any]) -> Any:
        args = _map_nodes(node.args, state, self._node_slots)
        kwargs = _map_nodes(node.kwargs, state, self._node_slots)
        if node.op == "get_attr":
            return self._get_attr(str(node.target))
        if node.op == "call_module":
            return self.graph_module.get_submodule(str(node.target))(*args, **kwargs)
        if node.op == "call_function":
            if not callable(node.target):
                raise ForwardDebuggerError(f"FX function target is not callable: {node.target!r}")
            return node.target(*args, **kwargs)
        if node.op == "call_method":
            if not args:
                raise ForwardDebuggerError(f"FX method has no receiver: {node.target!r}")
            return getattr(args[0], str(node.target))(*args[1:], **kwargs)
        raise ForwardDebuggerError(f"unsupported FX node operation: {node.op}")

    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate_execution()
        self.validate(state)
        if state["done"]:
            raise ForwardDebuggerError("cannot advance a completed FX session")
        cursor = int(state["cursor"])
        node = self._step_node(cursor)
        output_step = node.op == "output"
        result = self._execute_node(node, state) if not output_step else _map_nodes(
            node.args[0], state, self._node_slots
        )
        if not _supported_value(result):
            raise ForwardDebuggerError(
                f"FX node {node.name!r} returned an unsupported value type"
            )
        out = clone(dict(state))
        if output_step:
            out["output"] = result
        else:
            # Adopt the public snapshot layout when a value is produced.
            # Non-dense views (slices/expand/unfold) may become dense on clone;
            # deferring that conversion until the next node changes a sealed
            # undeclared layout during multi-node transactions.
            out[self._node_slots[str(node.name)]] = clone(result)
        out["cursor"] = cursor + 1
        out["done"] = out["cursor"] == len(self._steps)
        out["output_ready"] = out["done"]
        self.validate(out)
        self._check_module_state()
        return out

    def numerical_contract(self) -> NumericalContract:
        return NumericalContract(
            mode="same-program-exact",
            program="torch.fx original graph node calls",
            comparator="typed-content-digest",
        )


def create_forward_session(
    module: Any,
    args: Any,
    *,
    model_identity: str,
    concrete_args: Mapping[str, Any] | None = None,
) -> Session:
    """Trace without numerical inputs, then return a one-node-per-step Session."""

    graph_module = _trace(module, concrete_args)
    adapter = ForwardAdapter(
        graph_module,
        args,
        model_identity=model_identity,
        concrete_args=concrete_args,
    )
    return Session(adapter, adapter.initial_state())


def create_forward_debugger(
    module: Any,
    args: Any,
    *,
    model_identity: str,
    concrete_args: Mapping[str, Any] | None = None,
) -> ForwardDebugger:
    """Create the public Saturn debugger over a traced forward session."""

    return ForwardDebugger(
        create_forward_session(
            module,
            args,
            model_identity=model_identity,
            concrete_args=concrete_args,
        )
    )


def _sample(value: Any, *, depth: int = 0) -> Any:
    """Return a small diagnostic sample without assigning semantic meaning."""

    if depth > 4:
        return "<sample-depth-bound>"
    if is_tensor(value):
        try:
            flat = value.detach().reshape(-1)[:8].cpu().tolist()
            return flat
        except Exception:  # noqa: BLE001 - samples are diagnostic only
            return None
    if type(value) in (bool, int, float, str) or value is None:
        return value
    if isinstance(value, Mapping):
        return [
            {
                "key": _sample(key, depth=depth + 1),
                "value": _sample(item, depth=depth + 1),
            }
            for key, item in list(value.items())[:8]
        ]
    if isinstance(value, (tuple, list)):
        return [_sample(item, depth=depth + 1) for item in value[:8]]
    return None


def auto_symbol_table(session: Session) -> SymbolTable:
    """Bind exact current tensor results under structural ``vNNN`` names."""

    adapter = session.adapter
    if not isinstance(adapter, ForwardAdapter):
        raise TypeError("auto_symbol_table requires a ForwardAdapter session")
    cut = session.capture(retain=True)
    rows: list[SymbolBinding] = []
    for spec in adapter.node_specs:
        if spec.placeholder or spec.output:
            continue
        value = session.read(spec.slot)
        if not is_tensor(value):
            continue
        rows.append(SymbolBinding.bind(
            session,
            spec.slot,
            (StateAddress(spec.slot, role="computed"),),
            context=adapter.context,
            context_slots=adapter.context_slots,
            evidence=(
                f"cas://sha256/{cut.fingerprint}",
                f"fx://graph/{adapter.graph_fingerprint}/node/{spec.name}",
            ),
            claim=(
                f"Observed tensor result for structural FX node {spec.name}; "
                "semantic role and machine-language meaning are unresolved."
            ),
            status="observed",
        ))
    return SymbolTable(tuple(rows))


def execution_table(session: Session) -> dict[str, Any]:
    """Return a bounded structural table of current node values and consumers."""

    adapter = session.adapter
    if not isinstance(adapter, ForwardAdapter):
        raise TypeError("execution_table requires a ForwardAdapter session")
    frame = session.inspect()
    state = {slot: session.read(slot) for slot in adapter.addresses({})}
    cursor = int(state["cursor"])
    rows = []
    for spec in adapter.node_specs:
        if spec.output:
            continue
        value = state[spec.slot]
        available = spec.placeholder or (spec.order is not None and cursor > spec.order)
        rows.append({
            "index": int(spec.order if spec.order is not None else -1),
            "name": spec.name,
            "slot": spec.slot,
            "op": spec.op,
            "target": spec.target,
            "operation_level": "native-neural-graph",
            "writer": f"fx.{spec.op}:{spec.target}",
            "reads": list(spec.reads),
            "consumers": [f"fx.{user}" for user in spec.users],
            "readers": [f"fx.{user}" for user in spec.users],
            "status": "available" if available else "pending",
            "raw": None if not available else {
                "descriptor": describe(value),
                "sample": _sample(value),
            },
            "semantic_role": "unresolved",
            "authority": "structural-fx-observation",
        })
    return {
        "schema": "mdb-fx-execution-table-v1",
        "model_identity": adapter.model_identity,
        "graph_fingerprint": adapter.graph_fingerprint,
        "operation_level": "native-neural-graph",
        "current_point": dict(frame.execution_point),
        "cursor": cursor,
        "done": bool(state["done"]),
        "output": {
            "ready": bool(state["output_ready"]),
            "descriptor": describe(state["output"]) if state["output_ready"] else None,
            "sample": _sample(state["output"]) if state["output_ready"] else None,
            "abi": "native-graph-output",
        },
        "nodes": rows,
        "authority": "read-only-structural-forward-observation",
        "semantic_qualification": "not-established",
        "semantic_inference": "not-performed",
    }


__all__ = [
    "ForwardAdapter",
    "ForwardDebugger",
    "ForwardDebuggerError",
    "MAX_GRAPH_NODES",
    "MAX_STATE_BYTES",
    "auto_symbol_table",
    "create_forward_debugger",
    "create_forward_session",
    "execution_table",
]
