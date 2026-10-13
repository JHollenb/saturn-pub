"""Public, content-addressed execution graphs for native MDB state.

The graph in this module is a history index around native :class:`StateCut`
and Act records.  It deliberately does not execute a model, replay a cut, or
infer a semantic capability.  A graph node points at an already-produced
source, state cut, action/Act receipt, or output; the native Bench and its
consumer remain authoritative.

Two relation spaces are kept separate:

``execution``
    Ordered, typed dependencies used for structural DAG checks.  These are
    the ``parent``, ``continues``, ``reads``, ``writes``, ``derived_from``,
    ``invalidates`` and ``observed_effect`` relations.

``knowledge``
    Descriptive or semantic assertions such as ``supports`` and
    ``contradicts``.  Knowledge edges may be cyclic and never establish an
    executable lineage or a restorable state.

Nodes and edges are immutable values.  Adding a node or edge to a
``StateGraph`` is append-only; ``seal`` creates a hash-bound snapshot and
freezes that graph instance.  A later version must be made with ``fork`` and
retains the parent snapshot identity.  There is intentionally no branch
merge or label-based restore operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

STATE_GRAPH_SCHEMA = "mdb-state-graph-v1"
STATE_GRAPH_SNAPSHOT_SCHEMA = "mdb-state-graph-snapshot-v1"
STATE_GRAPH_EVENT_SCHEMA = "mdb-state-graph-event-v1"
STATE_GRAPH_NODE_SCHEMA = "mdb-state-graph-node-v1"
STATE_GRAPH_EDGE_SCHEMA = "mdb-state-graph-edge-v1"
STATE_GRAPH_BINDING_SCHEMA = "mdb-state-graph-binding-v1"
STATE_GRAPH_EVIDENCE_SCHEMA = "mdb-state-graph-evidence-v1"

NODE_KINDS = frozenset({"source", "state", "action", "output"})
EXECUTION_RELATIONS = frozenset(
    {
        "parent",
        "continues",
        "reads",
        "writes",
        "derived_from",
        "invalidates",
        "observed_effect",
    }
)
KNOWLEDGE_RELATIONS = frozenset(
    {
        "supports",
        "contradicts",
        "describes",
        "hypothesizes",
        "related_to",
        "annotates",
    }
)
EVIDENCE_CLASSES = frozenset({"declared", "observed", "causally-tested", "hypothesis"})


class StateGraphError(ValueError):
    """Raised when a graph record cannot be represented safely."""


class GraphIntegrityError(StateGraphError):
    """Raised when a graph or snapshot violates its identity/lineage contract."""


def _json(value: Any, *, label: str = "value") -> Any:
    """Normalize strict JSON without falling back to process-local ``repr``."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StateGraphError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StateGraphError(f"{label} contains a non-string key")
            result[key] = _json(item, label=f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json(item, label=f"{label}[]") for item in value]
    raise StateGraphError(f"{label} contains unsupported {type(value).__name__}")


def _freeze(value: Any, *, label: str = "value") -> Any:
    value = _json(value, label=label)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item, label=f"{label}.{key}") for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item, label=f"{label}[]") for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _json(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateGraphError(f"{label} must be non-empty text")
    return value.strip()


def _strings(values: Iterable[Any], label: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise StateGraphError(f"{label} must be a sequence of strings")
    result = tuple(_text(value, f"{label}[]") for value in values)
    if len(result) != len(set(result)):
        raise StateGraphError(f"{label} contains duplicate values")
    return result


def _sha(value: Any, label: str) -> str:
    text = _text(value, label).lower().removeprefix("sha256:")
    if text.startswith("cas://sha256/"):
        text = text.removeprefix("cas://sha256/")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise StateGraphError(f"{label} must be a SHA-256 digest")
    return text


def _value_ref(value: Any, *, label: str) -> str:
    """Extract an opaque, content-addressed reference from an existing object."""

    if isinstance(value, str):
        return _text(value, label)
    if isinstance(value, Mapping):
        for key in ("ref", "cut_ref", "state_ref", "source_ref", "output_ref", "uri"):
            if value.get(key) is not None:
                return _text(value[key], f"{label}.{key}")
        for key in ("fingerprint", "cut_fingerprint", "sha256"):
            if value.get(key) is not None:
                return f"cas://sha256/{_sha(value[key], f'{label}.{key}')}"
    for attr in ("ref", "cut_ref", "state_ref", "source_ref", "output_ref"):
        resolved = getattr(value, attr, None)
        if resolved is not None:
            return _text(resolved, f"{label}.{attr}")
    for attr in ("fingerprint", "cut_fingerprint", "sha256"):
        resolved = getattr(value, attr, None)
        if resolved is not None:
            return f"cas://sha256/{_sha(resolved, f'{label}.{attr}')}"
    raise StateGraphError(f"{label} must provide an existing content reference")


def _value_fingerprint(value: Any, *, label: str) -> str | None:
    if isinstance(value, Mapping):
        for key in ("fingerprint", "cut_fingerprint", "sha256"):
            if value.get(key) is not None:
                return _sha(value[key], f"{label}.{key}")
    for attr in ("fingerprint", "cut_fingerprint", "sha256"):
        resolved = getattr(value, attr, None)
        if resolved is not None:
            return _sha(resolved, f"{label}.{attr}")
    text = value if isinstance(value, str) else None
    if text and text.startswith("cas://sha256/"):
        candidate = text.removeprefix("cas://sha256/")
        if len(candidate) == 64 and all(char in "0123456789abcdefABCDEF" for char in candidate):
            return candidate.lower()
    return None


def _node_id(value: Any) -> str:
    return _digest(value)


@dataclass(frozen=True, slots=True)
class NodeBinding:
    """Model/session/clock/version context attached to a graph object.

    Values are intentionally opaque JSON.  A clock may be an integer step, a
    token position, a wall-clock string, or a richer typed object supplied by
    a native capture.  This class does not reinterpret it.
    """

    model: Any = None
    session: Any = None
    clock: Any = None
    version: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema: ClassVar[str] = STATE_GRAPH_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _freeze(self.model, label="binding.model"))
        object.__setattr__(self, "session", _freeze(self.session, label="binding.session"))
        object.__setattr__(self, "clock", _freeze(self.clock, label="binding.clock"))
        object.__setattr__(self, "version", _freeze(self.version, label="binding.version"))
        object.__setattr__(self, "metadata", _freeze(self.metadata, label="binding.metadata"))

    @classmethod
    def from_value(cls, value: "NodeBinding | Mapping[str, Any] | None") -> "NodeBinding":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise StateGraphError("binding must be a NodeBinding or JSON object")
        return cls(
            model=value.get("model", value.get("model_id")),
            session=value.get("session", value.get("session_id")),
            clock=value.get("clock", value.get("time")),
            version=value.get("version", value.get("version_id")),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model": _thaw(self.model),
            "session": _thaw(self.session),
            "clock": _thaw(self.clock),
            "version": _thaw(self.version),
            "metadata": _thaw(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    @property
    def model_id(self) -> Any:
        return _thaw(self.model)

    @property
    def session_id(self) -> Any:
        return _thaw(self.session)


ExecutionBinding = NodeBinding
GraphBinding = NodeBinding


@dataclass(frozen=True, slots=True)
class Evidence:
    """An evidence row retained without reducing it to scalar importance."""

    evidence_class: str = "declared"
    condition: Mapping[str, Any] = field(default_factory=dict)
    sign: Any = None
    dose: Any = None
    effect: Any = None
    receipt_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema: ClassVar[str] = STATE_GRAPH_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        evidence_class = _text(self.evidence_class, "evidence_class").lower().replace("_", "-")
        if evidence_class not in EVIDENCE_CLASSES:
            raise StateGraphError(
                f"unsupported evidence class {evidence_class!r}; expected one of "
                f"{sorted(EVIDENCE_CLASSES)}"
            )
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "condition", _freeze(self.condition, label="evidence.condition"))
        object.__setattr__(self, "sign", _freeze(self.sign, label="evidence.sign"))
        object.__setattr__(self, "dose", _freeze(self.dose, label="evidence.dose"))
        object.__setattr__(self, "effect", _freeze(self.effect, label="evidence.effect"))
        object.__setattr__(self, "receipt_refs", _strings(self.receipt_refs, "receipt_refs"))
        object.__setattr__(self, "metadata", _freeze(self.metadata, label="evidence.metadata"))

    @classmethod
    def from_value(cls, value: "Evidence | Mapping[str, Any] | str | None") -> "Evidence":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(evidence_class=value)
        if not isinstance(value, Mapping):
            raise StateGraphError("evidence must be an Evidence, class string, or JSON object")
        return cls(
            evidence_class=value.get("evidence_class", value.get("class", value.get("class_", "declared"))),
            condition=value.get("condition", {}),
            sign=value.get("sign"),
            dose=value.get("dose"),
            effect=value.get("effect"),
            receipt_refs=tuple(value.get("receipt_refs", value.get("receipts", ())) or ()),
            metadata=value.get("metadata", {}),
        )

    @property
    def class_(self) -> str:
        """Python-safe alias for the serialized ``class`` field."""

        return self.evidence_class

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "class": self.evidence_class,
            "condition": _thaw(self.condition),
            "sign": _thaw(self.sign),
            "dose": _thaw(self.dose),
            "effect": _thaw(self.effect),
            "receipt_refs": list(self.receipt_refs),
            "metadata": _thaw(self.metadata),
        }


def evidence(
    evidence_class: str = "declared",
    *,
    condition: Mapping[str, Any] | None = None,
    sign: Any = None,
    dose: Any = None,
    effect: Any = None,
    receipt_refs: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Evidence:
    """Short constructor retained for in-process panels."""

    return Evidence(
        evidence_class=evidence_class,
        condition=condition or {},
        sign=sign,
        dose=dose,
        effect=effect,
        receipt_refs=tuple(receipt_refs or ()),
        metadata=metadata or {},
    )


def _evidence_tuple(values: Evidence | Mapping[str, Any] | str | Iterable[Any] | None) -> tuple[Evidence, ...]:
    if values is None:
        return ()
    if isinstance(values, (Evidence, Mapping, str)):
        values = (values,)
    return tuple(Evidence.from_value(value) for value in values)


@dataclass(frozen=True, slots=True)
class GraphNode:
    """Immutable content-addressed source, state, action, or output node."""

    kind: str
    content: Mapping[str, Any] = field(default_factory=dict)
    binding: NodeBinding = field(default_factory=NodeBinding)
    evidence: tuple[Evidence, ...] = ()
    labels: tuple[str, ...] = ()
    node_id: str = field(init=False)

    schema: ClassVar[str] = STATE_GRAPH_NODE_SCHEMA

    def __post_init__(self) -> None:
        kind = _text(self.kind, "node.kind").lower()
        if kind not in NODE_KINDS:
            raise StateGraphError(f"unsupported node kind {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "content", _freeze(self.content, label="node.content"))
        object.__setattr__(self, "binding", NodeBinding.from_value(self.binding))
        object.__setattr__(self, "evidence", _evidence_tuple(self.evidence))
        object.__setattr__(self, "labels", _strings(self.labels, "node.labels"))
        object.__setattr__(self, "node_id", _node_id(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "content": _thaw(self.content),
            "binding": self.binding.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "labels": list(self.labels),
        }

    @property
    def id(self) -> str:
        return self.node_id

    @property
    def fingerprint(self) -> str:
        return self.node_id

    @property
    def ref(self) -> str:
        return f"cas://sha256/{self.node_id}"

    @property
    def payload(self) -> Mapping[str, Any]:
        return self.content

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "node_id": self.node_id, "fingerprint": self.node_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphNode":
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise GraphIntegrityError("graph node schema changed")
        result = cls(
            kind=value.get("kind"),
            content=value.get("content", {}),
            binding=value.get("binding", {}),
            evidence=tuple(value.get("evidence", ())),
            labels=tuple(value.get("labels", ())),
        )
        supplied = value.get("node_id", value.get("fingerprint"))
        if supplied != result.node_id:
            raise GraphIntegrityError("graph node content hash changed")
        return result


Node = GraphNode
SourceNode = GraphNode
StateNode = GraphNode
ActionNode = GraphNode
OutputNode = GraphNode


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """Immutable directed edge with an explicit execution/knowledge space."""

    source: str
    target: str
    relation: str
    graph: str = "execution"
    evidence: tuple[Evidence, ...] = ()
    binding: NodeBinding = field(default_factory=NodeBinding)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    edge_id: str = field(init=False)

    schema: ClassVar[str] = STATE_GRAPH_EDGE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _text(self.source, "edge.source"))
        object.__setattr__(self, "target", _text(self.target, "edge.target"))
        relation = _text(self.relation, "edge.relation")
        graph = _text(self.graph, "edge.graph").lower()
        if graph not in {"execution", "knowledge"}:
            raise StateGraphError("edge.graph must be 'execution' or 'knowledge'")
        if graph == "execution" and relation not in EXECUTION_RELATIONS:
            raise StateGraphError(f"unsupported execution relation {relation!r}")
        if graph == "knowledge" and relation not in KNOWLEDGE_RELATIONS:
            raise StateGraphError(f"unsupported knowledge relation {relation!r}")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "evidence", _evidence_tuple(self.evidence))
        object.__setattr__(self, "binding", NodeBinding.from_value(self.binding))
        object.__setattr__(self, "metadata", _freeze(self.metadata, label="edge.metadata"))
        object.__setattr__(self, "edge_id", _digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "graph": self.graph,
            "evidence": [item.to_dict() for item in self.evidence],
            "binding": self.binding.to_dict(),
            "metadata": _thaw(self.metadata),
        }

    @property
    def id(self) -> str:
        return self.edge_id

    @property
    def fingerprint(self) -> str:
        return self.edge_id

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "edge_id": self.edge_id, "fingerprint": self.edge_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphEdge":
        if not isinstance(value, Mapping) or value.get("schema") != cls.schema:
            raise GraphIntegrityError("graph edge schema changed")
        result = cls(
            source=value.get("source"),
            target=value.get("target"),
            relation=value.get("relation"),
            graph=value.get("graph", "execution"),
            evidence=tuple(value.get("evidence", ())),
            binding=value.get("binding", {}),
            metadata=value.get("metadata", {}),
        )
        supplied = value.get("edge_id", value.get("fingerprint"))
        if supplied != result.edge_id:
            raise GraphIntegrityError("graph edge content hash changed")
        return result


@dataclass(frozen=True, slots=True)
class GraphVerification:
    """Structured mechanics result; it carries no scientific verdict."""

    ok: bool
    node_count: int
    edge_count: int
    execution_edge_count: int
    knowledge_edge_count: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "execution_edge_count": self.execution_edge_count,
            "knowledge_edge_count": self.knowledge_edge_count,
            "errors": list(self.errors),
        }

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """Immutable sealed graph snapshot."""

    version: int
    graph_id: str
    binding: NodeBinding
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    events: tuple[Mapping[str, Any], ...]
    parent_snapshot: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    snapshot_id: str = field(init=False)

    schema: ClassVar[str] = STATE_GRAPH_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise StateGraphError("snapshot version must be a non-negative integer")
        object.__setattr__(self, "graph_id", _text(self.graph_id, "graph_id"))
        object.__setattr__(self, "binding", NodeBinding.from_value(self.binding))
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        if any(not isinstance(node, GraphNode) for node in nodes):
            raise StateGraphError("snapshot.nodes must contain GraphNode values")
        if any(not isinstance(edge, GraphEdge) for edge in edges):
            raise StateGraphError("snapshot.edges must contain GraphEdge values")
        if len({node.node_id for node in nodes}) != len(nodes):
            raise GraphIntegrityError("snapshot contains duplicate node identities")
        if len({edge.edge_id for edge in edges}) != len(edges):
            raise GraphIntegrityError("snapshot contains duplicate edge identities")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        frozen_events = []
        for event in self.events:
            if not isinstance(event, Mapping):
                raise StateGraphError("snapshot.events must contain JSON objects")
            frozen_events.append(_freeze(event, label="snapshot.event"))
        object.__setattr__(self, "events", tuple(frozen_events))
        parent = None if self.parent_snapshot is None else _sha(self.parent_snapshot, "parent_snapshot")
        object.__setattr__(self, "parent_snapshot", parent)
        object.__setattr__(self, "metadata", _freeze(self.metadata, label="snapshot.metadata"))
        object.__setattr__(self, "snapshot_id", _digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "graph_id": self.graph_id,
            "binding": self.binding.to_dict(),
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda item: item.node_id)],
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=lambda item: item.edge_id)],
            "events": [_thaw(event) for event in self.events],
            "parent_snapshot": self.parent_snapshot,
            "metadata": _thaw(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return self.snapshot_id

    @property
    def ref(self) -> str:
        return f"cas://sha256/{self.snapshot_id}"

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "snapshot_id": self.snapshot_id, "fingerprint": self.snapshot_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphSnapshot":
        """Decode and verify a serialized snapshot through the public loader."""

        return StateGraph.from_snapshot(value).seal()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def verify(self, *, raise_on_error: bool = True) -> GraphVerification:
        """Verify this sealed snapshot through the public graph checker."""

        return StateGraph.from_snapshot(self).verify(raise_on_error=raise_on_error)

    validate = verify


class StateGraph:
    """Append-only builder and query surface for a versioned execution graph."""

    schema: ClassVar[str] = STATE_GRAPH_SCHEMA

    def __init__(
        self,
        *,
        binding: NodeBinding | Mapping[str, Any] | None = None,
        version: int = 0,
        graph_id: str | None = None,
        parent_snapshot: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise StateGraphError("graph version must be a non-negative integer")
        self.binding = NodeBinding.from_value(binding)
        self.version = version
        self.graph_id = graph_id or _digest(
            {"schema": self.schema, "binding": self.binding.to_dict(), "version": version}
        )
        _text(self.graph_id, "graph_id")
        self.parent_snapshot = None if parent_snapshot is None else _sha(parent_snapshot, "parent_snapshot")
        self.metadata = _freeze(metadata or {}, label="graph.metadata")
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._events: list[Mapping[str, Any]] = []
        self._sealed = False
        self._sealed_snapshot: GraphSnapshot | None = None

    @property
    def nodes(self) -> Mapping[str, GraphNode]:
        return MappingProxyType(self._nodes)

    @property
    def edges(self) -> Mapping[str, GraphEdge]:
        return MappingProxyType(self._edges)

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._events)

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _check_open(self) -> None:
        if self._sealed:
            raise GraphIntegrityError("graph snapshot is sealed; fork it before appending")

    def _append_event(self, kind: str, object_id: str) -> None:
        previous = self._events[-1].get("event_id") if self._events else None
        body = {
            "schema": STATE_GRAPH_EVENT_SCHEMA,
            "sequence": len(self._events),
            "kind": kind,
            "object_id": object_id,
            "previous_event": previous,
        }
        event = {**body, "event_id": _digest(body)}
        self._events.append(_freeze(event, label="event"))

    def add_node(self, node: GraphNode) -> str:
        self._check_open()
        if not isinstance(node, GraphNode):
            raise StateGraphError("add_node requires a GraphNode")
        existing = self._nodes.get(node.node_id)
        if existing is not None:
            if existing != node:
                raise GraphIntegrityError("content-addressed node identity collision")
            return node.node_id
        self._nodes[node.node_id] = node
        self._append_event("node", node.node_id)
        return node.node_id

    def _node_id(self, value: GraphNode | str, *, label: str = "node") -> str:
        node_id = value.node_id if isinstance(value, GraphNode) else _text(value, label)
        if node_id.startswith("cas://sha256/"):
            node_id = _sha(node_id, label)
        if node_id not in self._nodes:
            raise GraphIntegrityError(f"unknown {label} {node_id!r}")
        return node_id

    def add_edge(
        self,
        source: GraphNode | str,
        target: GraphNode | str,
        relation: str,
        *,
        graph: str | None = None,
        evidence: Evidence | Mapping[str, Any] | str | Iterable[Any] | None = None,
        evidence_class: str | None = None,
        binding: NodeBinding | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        condition: Mapping[str, Any] | None = None,
        sign: Any = None,
        dose: Any = None,
        effect: Any = None,
        receipt_refs: Iterable[str] = (),
    ) -> str:
        self._check_open()
        source_id = self._node_id(source, label="edge.source")
        target_id = self._node_id(target, label="edge.target")
        relation = _text(relation, "edge.relation")
        resolved_graph = graph or ("execution" if relation in EXECUTION_RELATIONS else "knowledge")
        if evidence is None:
            evidence = Evidence(
                evidence_class=evidence_class or "declared",
                condition=condition or {},
                sign=sign,
                dose=dose,
                effect=effect,
                receipt_refs=tuple(receipt_refs),
            )
        elif any(value is not None for value in (evidence_class, condition, sign, dose, effect)) or tuple(receipt_refs):
            raise StateGraphError("pass either a complete evidence value or evidence keyword fields")
        edge = GraphEdge(
            source=source_id,
            target=target_id,
            relation=relation,
            graph=resolved_graph,
            evidence=_evidence_tuple(evidence),
            binding=NodeBinding.from_value(binding),
            metadata=metadata or {},
        )
        existing = self._edges.get(edge.edge_id)
        if existing is not None:
            if existing != edge:
                raise GraphIntegrityError("content-addressed edge identity collision")
            return edge.edge_id
        self._edges[edge.edge_id] = edge
        self._append_event("edge", edge.edge_id)
        return edge.edge_id

    def _node(
        self,
        kind: str,
        content: Mapping[str, Any],
        *,
        binding: NodeBinding | Mapping[str, Any] | None = None,
        evidence: Evidence | Mapping[str, Any] | str | Iterable[Any] | None = None,
        labels: Iterable[str] = (),
    ) -> str:
        return self.add_node(
            GraphNode(
                kind=kind,
                content=content,
                binding=NodeBinding.from_value(binding),
                evidence=_evidence_tuple(evidence),
                labels=tuple(labels),
            )
        )

    def add_source(
        self,
        source: Any = None,
        *,
        source_ref: str | None = None,
        binding: NodeBinding | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        evidence: Evidence | Mapping[str, Any] | str | Iterable[Any] | None = None,
        labels: Iterable[str] = (),
    ) -> str:
        candidate = source_ref if source_ref is not None else source
        if candidate is None:
            raise StateGraphError("source requires source_ref or an existing source object")
        reference = _value_ref(candidate, label="source")
        fingerprint = _value_fingerprint(candidate, label="source")
        return self._node(
            "source",
            {"source_ref": reference, "source_fingerprint": fingerprint, "metadata": metadata or {}},
            binding=binding,
            evidence=evidence,
            labels=labels,
        )

    def add_consumer(
        self,
        consumer: Any = None,
        *,
        consumer_ref: str | None = None,
        binding: NodeBinding | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        evidence: Evidence | Mapping[str, Any] | str | Iterable[Any] | None = None,
        labels: Iterable[str] = (),
    ) -> str:
        """Record a native consumer descriptor as a source-side observation.

        Consumers are deliberately source nodes: a consumer descriptor can
        describe an observation or receipt, but it cannot become a restorable
        state.  When a descriptor has no native reference, its strict JSON
        content receives an opaque CAS reference for graph identity only.
        """

        candidate = consumer_ref if consumer_ref is not None else consumer
        if candidate is None:
            raise StateGraphError("consumer requires consumer_ref or a descriptor")
        try:
            reference = _value_ref(candidate, label="consumer")
        except StateGraphError:
            descriptor = _json(candidate, label="consumer")
            reference = f"cas://sha256/{_digest({'consumer': descriptor})}"
        label_values = tuple(labels)
        if "consumer" not in label_values:
            label_values = ("consumer", *label_values)
        return self.add_source(
            source_ref=reference,
            binding=binding,
            metadata={**(metadata or {}), "role": "consumer"},
            evidence=evidence,
            labels=label_values,
        )

    def add_state(
        self,
        state: Any = None,
        *,
        statecut: Any = None,
        cut_ref: str | None = None,
        state_ref: str | None = None,
        fingerprint: str | None = None,
        binding: NodeBinding | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        evidence: Evidence | Mapping[str, Any] | str | Iterable[Any] | None = None,
        labels: Iterable[str] = (),
    ) -> str:
        candidate = statecut if statecut is not None else state
        candidate = state_ref if state_ref is not None else cut_ref if cut_ref is not None else candidate
        if candidate is None:
            raise StateGraphError("state requires a StateCut object or content reference")
        reference = _value_ref(candidate, label="state")
        state_fingerprint = fingerprint or _value_fingerprint(candidate, label="state")
        if state_fingerprint is None:
            raise StateGraphError(
                "state requires a StateCut fingerprint or a cas://sha256 reference; labels cannot authorize restore"
            )
        state_fingerprint = _sha(state_fingerprint, "state.fingerprint")
        return self._node(
            "state",
            {
                "state_ref": reference,
                "state_fingerprint": state_fingerprint,
                "metadata": metadata or {},
            },
            binding=binding,
            evidence=evidence,
            labels=labels,
        )

    def add_output(
        self,
        output: Any = None,
        *,
        output_ref: str | None = None,
        ref: str | None = None,
        binding: NodeBinding | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        evidence: Evidence | Mapping[str, Any] | str | Iterable[Any] | None = None,
        labels: Iterable[str] = (),
    ) -> str:
        candidate = output_ref if output_ref is not None else ref if ref is not None else output
        if candidate is None:
            raise StateGraphError("output requires a value or output_ref")
        if isinstance(candidate, Mapping) and any(
            key in candidate for key in ("ref", "output_ref", "uri", "fingerprint", "sha256")
        ):
            reference = _value_ref(candidate, label="output")
            content = {"output_ref": reference, "metadata": metadata or {}}
        elif isinstance(candidate, str):
            content = {"output_ref": _text(candidate, "output_ref"), "metadata": metadata or {}}
        else:
            content = {"value": candidate, "metadata": metadata or {}}
        return self._node("output", content, binding=binding, evidence=evidence, labels=labels)

    def add_action(
        self,
        action: Any = None,
        *,
        act_ref: str | None = None,
        inputs: Iterable[GraphNode | str] = (),
        output: GraphNode | str | None = None,
        receipt_ref: str | None = None,
        receipt_refs: Iterable[str] = (),
        reads: Iterable[str] = (),
        writes: Iterable[str] = (),
        invalidates: Iterable[str] = (),
        binding: NodeBinding | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        evidence: Evidence | Mapping[str, Any] | str | Iterable[Any] | None = None,
        evidence_class: str | None = None,
        sign: Any = None,
        dose: Any = None,
        effect: Any = None,
        labels: Iterable[str] = (),
    ) -> str:
        input_ids = tuple(self._node_id(item, label="action input") for item in inputs)
        output_id = None if output is None else self._node_id(output, label="action output")
        resolved_receipts = tuple(receipt_refs or ())
        if receipt_ref is not None:
            resolved_receipts = (*resolved_receipts, receipt_ref)
        content = {
            "act_ref": None if act_ref is None else _text(act_ref, "act_ref"),
            "action": action,
            "inputs": list(input_ids),
            "output": output_id,
            "receipt_refs": list(_strings(resolved_receipts, "action.receipt_refs")),
            "reads": list(_strings(reads, "action.reads")),
            "writes": list(_strings(writes, "action.writes")),
            "invalidates": list(_strings(invalidates, "action.invalidates")),
            "metadata": metadata or {},
        }
        if evidence is None:
            evidence = Evidence(
                evidence_class=evidence_class or "declared",
                sign=sign,
                dose=dose,
                effect=effect,
                receipt_refs=resolved_receipts,
            )
        action_id = self._node("action", content, binding=binding, evidence=evidence, labels=labels)
        for input_id in input_ids:
            self.add_edge(input_id, action_id, "reads", evidence=evidence)
        if output_id is not None:
            self.add_edge(action_id, output_id, "writes", evidence=evidence)
        return action_id

    # Explicit aliases make panel code read naturally and keep descriptors
    # independent from the internal GraphNode representation.
    add_source_node = add_source
    add_consumer_node = add_consumer
    add_state_node = add_state
    add_action_node = add_action
    add_output_node = add_output
    record_source = add_source
    record_consumer = add_consumer
    record_cut = add_state
    record_state = add_state
    record_act = add_action
    record_action = add_action
    record_output = add_output
    # Compact names are useful for in-process panel adapters.  They still
    # return immutable content-addressed node/edge identifiers.
    source = add_source
    consumer = add_consumer
    cut = add_state
    state = add_state
    act = add_action
    action = add_action
    output = add_output
    edge = add_edge

    def ingest_bench(
        self,
        bench: Any,
        *,
        include_annotations: bool = True,
        include_events: bool = True,
    ) -> dict[str, Any]:
        """Index an existing public Bench capture without running its backend.

        The method only reads the capture's identity, cut registry, durable
        artifact manifests, events and optional state annotations.  It records
        receipt references exactly as supplied by Bench and never turns a
        label into a restore authority.
        """

        identity = getattr(bench, "identity", {})
        session_id = getattr(bench, "session_id", None)
        model_abi = getattr(bench, "model_abi", None)
        if not isinstance(identity, Mapping):
            identity = {"value": identity}
        cut_nodes: dict[str, str] = {}
        cut_records: dict[str, dict[str, Any]] = {}
        cuts = getattr(bench, "cuts", {})
        if isinstance(cuts, Mapping):
            for cut_ref, raw_meta in cuts.items():
                metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
                artifact = None
                artifacts = getattr(bench, "artifacts", None)
                getter = getattr(artifacts, "get", None)
                if callable(getter):
                    try:
                        candidate = getter(cut_ref)
                        if isinstance(candidate, Mapping):
                            artifact = dict(candidate)
                    except Exception:
                        # An index should remain usable if a referenced payload
                        # was intentionally kept outside the current process.
                        artifact = None
                record = {**metadata, **(artifact or {})}
                cut_records[str(cut_ref)] = record
                cursor = record.get("cursor", {})
                clock = cursor.get("token", cursor.get("step", cursor.get("index"))) if isinstance(cursor, Mapping) else None
                state_id = self.add_state(
                    cut_ref=str(cut_ref),
                    fingerprint=_value_fingerprint(record, label="bench cut") or _value_fingerprint(str(cut_ref), label="bench cut"),
                    binding=NodeBinding(model=model_abi or identity, session=session_id, clock=clock, version=record.get("branch")),
                    metadata={
                        "label": record.get("label"),
                        "branch": record.get("branch"),
                        "cursor": cursor,
                        "source_frame": record.get("source_frame"),
                        "bundle_ref": record.get("bundle_ref"),
                    },
                )
                cut_nodes[str(cut_ref)] = state_id
        for cut_ref, record in cut_records.items():
            parent = record.get("parent_cut")
            if parent in cut_nodes:
                receipt = record.get("receipt_ref")
                self.add_edge(
                    cut_nodes[parent],
                    cut_nodes[cut_ref],
                    "continues",
                    evidence=Evidence("observed", receipt_refs=tuple([receipt] if receipt else [])),
                    binding=NodeBinding(model=model_abi or identity, session=session_id),
                    metadata={"branch": record.get("branch"), "label": record.get("label")},
                )
        action_nodes: dict[str, str] = {}
        if include_events:
            events = getattr(bench, "events", ())
            if isinstance(events, Sequence) and not isinstance(events, (str, bytes, bytearray)):
                for index, raw_event in enumerate(events):
                    if not isinstance(raw_event, Mapping):
                        continue
                    event = dict(raw_event)
                    act_ref = event.get("act_id") or event.get("event_id") or f"bench-event:{index}"
                    parent = event.get("parent") or event.get("parent_cut")
                    inputs = [cut_nodes[parent]] if parent in cut_nodes else []
                    donor = event.get("donor_cut")
                    if donor in cut_nodes:
                        inputs.append(cut_nodes[donor])
                    receipt = event.get("receipt_ref")
                    output = None
                    for cut_ref, record in cut_records.items():
                        if receipt and record.get("receipt_ref") == receipt:
                            output = cut_nodes.get(cut_ref)
                            break
                    event_evidence = Evidence(
                        "observed" if event.get("kind") != "act-rejected" else "causally-tested",
                        condition={"event_kind": event.get("kind", "unknown")},
                        receipt_refs=tuple([receipt] if receipt else []),
                    )
                    action_nodes[str(act_ref)] = self.add_action(
                        action=event,
                        act_ref=str(act_ref),
                        inputs=inputs,
                        output=output,
                        receipt_ref=receipt,
                        binding=NodeBinding(model=model_abi or identity, session=session_id, clock=index),
                        evidence=event_evidence,
                        labels=(str(event.get("kind")),) if event.get("kind") else (),
                    )
        annotation_count = 0
        if include_annotations:
            getter = getattr(bench, "list_state_annotations", None)
            if callable(getter):
                try:
                    annotations = getter()
                except Exception:
                    annotations = ()
                if isinstance(annotations, Mapping):
                    annotations = annotations.get("entries", annotations.get("annotations", ()))
                if isinstance(annotations, Sequence) and not isinstance(annotations, (str, bytes, bytearray)):
                    for annotation in annotations:
                        if not isinstance(annotation, Mapping):
                            continue
                        cut_ref = annotation.get("cut_id") or annotation.get("state_cut_ref")
                        if cut_ref not in cut_nodes:
                            continue
                        digest = annotation.get("fingerprint") or _digest(annotation)
                        source_id = self.add_source(
                            source_ref=(
                                str(annotation.get("annotation_ref"))
                                if annotation.get("annotation_ref")
                                else f"cas://sha256/{digest}"
                            ),
                            metadata={"annotation": annotation},
                            evidence=Evidence(
                                "observed",
                                receipt_refs=tuple(annotation.get("native_act_refs", ()) or ())
                                + tuple(annotation.get("consumer_refs", ()) or ()),
                            ),
                        )
                        self.add_edge(
                            source_id,
                            cut_nodes[cut_ref],
                            "annotates",
                            graph="knowledge",
                            evidence=Evidence("observed"),
                            metadata={"label": annotation.get("label"), "role": annotation.get("role")},
                        )
                        annotation_count += 1
        return {
            "session_id": session_id,
            "model_abi": model_abi,
            "state_nodes": len(cut_nodes),
            "action_nodes": len(action_nodes),
            "annotations": annotation_count,
            "node_ids": dict(cut_nodes),
            "action_ids": dict(action_nodes),
        }

    @classmethod
    def from_bench(
        cls,
        bench: Any,
        *,
        include_annotations: bool = True,
        include_events: bool = True,
        binding: NodeBinding | Mapping[str, Any] | None = None,
    ) -> "StateGraph":
        """Create a graph index from an already-loaded public Bench capture."""

        if binding is None:
            identity = getattr(bench, "identity", {})
            if not isinstance(identity, Mapping):
                identity = {"value": identity}
            binding = NodeBinding(
                model=getattr(bench, "model_abi", None) or identity,
                session=getattr(bench, "session_id", None),
                version=getattr(bench, "lane", None),
            )
        graph = cls(binding=binding, metadata={"source": "mdb.bench"})
        graph.ingest_bench(
            bench,
            include_annotations=include_annotations,
            include_events=include_events,
        )
        return graph

    from_capture = from_bench

    record_bench_capture = ingest_bench

    def query(
        self,
        *,
        kind: str | None = None,
        relation: str | None = None,
        graph: str | None = None,
        evidence_class: str | None = None,
        label: str | None = None,
    ) -> tuple[GraphNode | GraphEdge, ...]:
        """Return immutable matching objects for panel inspection."""

        result: list[GraphNode | GraphEdge] = []
        if relation is None:
            for node in self._nodes.values():
                if kind is not None and node.kind != kind:
                    continue
                if label is not None and label not in node.labels:
                    continue
                if evidence_class is not None and not any(
                    item.evidence_class == evidence_class.replace("_", "-") for item in node.evidence
                ):
                    continue
                result.append(node)
        else:
            for edge in self._edges.values():
                if edge.relation != relation or (graph is not None and edge.graph != graph):
                    continue
                if evidence_class is not None and not any(
                    item.evidence_class == evidence_class.replace("_", "-") for item in edge.evidence
                ):
                    continue
                result.append(edge)
        return tuple(result)

    def lineage(
        self,
        node: GraphNode | str,
        *,
        direction: str = "ancestors",
        include_knowledge: bool = False,
    ) -> tuple[GraphNode, ...]:
        """Return a deterministic traversal of native execution lineage."""

        start = self._node_id(node)
        if direction not in {"ancestors", "descendants"}:
            raise StateGraphError("lineage direction must be ancestors or descendants")
        selected = [
            edge
            for edge in self._edges.values()
            if include_knowledge or edge.graph == "execution"
        ]
        adjacency: dict[str, list[str]] = {}
        for edge in selected:
            source, target = edge.source, edge.target
            left, right = (target, source) if direction == "ancestors" else (source, target)
            adjacency.setdefault(left, []).append(right)
        seen = {start}
        queue: deque[str] = deque(sorted(adjacency.get(start, ())))
        result: list[GraphNode] = []
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            result.append(self._nodes[current])
            queue.extend(sorted(adjacency.get(current, ())))
        return tuple(result)

    def inspect(self, node: GraphNode | str | None = None) -> dict[str, Any]:
        if node is None:
            return {
                "schema": self.schema,
                "graph_id": self.graph_id,
                "version": self.version,
                "sealed": self.sealed,
                "nodes": len(self._nodes),
                "edges": len(self._edges),
                "events": len(self._events),
                "binding": self.binding.to_dict(),
                "parent_snapshot": self.parent_snapshot,
            }
        node_id = self._node_id(node)
        record = self._nodes[node_id].to_dict()
        record["outgoing"] = [edge.to_dict() for edge in self._edges.values() if edge.source == node_id]
        record["incoming"] = [edge.to_dict() for edge in self._edges.values() if edge.target == node_id]
        return record

    def to_dict(self) -> dict[str, Any]:
        """Return the current sealed snapshot as canonical JSON-compatible data."""

        return self.seal().to_dict()

    def _dag_errors(self) -> list[str]:
        execution = [edge for edge in self._edges.values() if edge.graph == "execution"]
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self._nodes}
        incoming_parent: dict[str, int] = {}
        for edge in execution:
            if edge.source not in self._nodes or edge.target not in self._nodes:
                # ``verify`` reports the unknown endpoint below.  Do not let
                # the structural checker turn that integrity finding into a
                # process-level KeyError.
                continue
            adjacency[edge.source].append(edge.target)
            if edge.relation in {"parent", "continues"}:
                incoming_parent[edge.target] = incoming_parent.get(edge.target, 0) + 1
        errors: list[str] = []
        for target, count in incoming_parent.items():
            if count > 1:
                errors.append(
                    f"node {target} has {count} parent/continues inputs; use one Action node with multiple reads"
                )
        color: dict[str, int] = {node_id: 0 for node_id in self._nodes}

        def visit(node_id: str) -> None:
            color[node_id] = 1
            for target in adjacency[node_id]:
                if color[target] == 1:
                    errors.append(f"execution lineage cycle includes {node_id} -> {target}")
                elif color[target] == 0:
                    visit(target)
            color[node_id] = 2

        for node_id in sorted(self._nodes):
            if color[node_id] == 0:
                visit(node_id)
        for edge in execution:
            if edge.source not in self._nodes or edge.target not in self._nodes:
                continue
            source = self._nodes[edge.source]
            target = self._nodes[edge.target]
            if edge.relation == "reads" and target.kind != "action":
                errors.append(f"reads edge target {target.node_id} is not an action")
            if edge.relation == "writes" and source.kind != "action":
                errors.append(f"writes edge source {source.node_id} is not an action")
            if edge.relation == "observed_effect" and source.kind != "action":
                errors.append(f"observed_effect source {source.node_id} is not an action")
        for node in self._nodes.values():
            if node.kind != "action":
                continue
            inputs = tuple(node.content.get("inputs", ()))
            output = node.content.get("output")
            if any(item not in self._nodes for item in inputs):
                errors.append(f"action {node.node_id} references an unknown input")
            if output is not None and output not in self._nodes:
                errors.append(f"action {node.node_id} references an unknown output")
            if output is not None and not any(
                edge.source == node.node_id and edge.target == output and edge.relation == "writes"
                for edge in execution
            ):
                errors.append(f"action {node.node_id} declares output without a writes edge")
        return errors

    def verify(self, *, raise_on_error: bool = True) -> GraphVerification:
        errors: list[str] = []
        try:
            for node_id, node in self._nodes.items():
                if node_id != node.node_id or GraphNode.from_dict(node.to_dict()) != node:
                    errors.append(f"node {node_id} content hash changed")
            for edge_id, edge in self._edges.items():
                if edge_id != edge.edge_id or GraphEdge.from_dict(edge.to_dict()) != edge:
                    errors.append(f"edge {edge_id} content hash changed")
                if edge.source not in self._nodes or edge.target not in self._nodes:
                    errors.append(f"edge {edge_id} references an unknown node")
            previous = None
            for sequence, raw_event in enumerate(self._events):
                event = dict(_thaw(raw_event))
                body = {key: value for key, value in event.items() if key != "event_id"}
                if event.get("sequence") != sequence or event.get("previous_event") != previous:
                    errors.append(f"event {sequence} sequence/chain changed")
                if event.get("schema") != STATE_GRAPH_EVENT_SCHEMA or event.get("event_id") != _digest(body):
                    errors.append(f"event {sequence} hash changed")
                if event.get("kind") not in {"node", "edge"}:
                    errors.append(f"event {sequence} has an unsupported kind")
                if event.get("kind") == "node" and event.get("object_id") not in self._nodes:
                    errors.append(f"event {sequence} references an unknown node")
                if event.get("kind") == "edge" and event.get("object_id") not in self._edges:
                    errors.append(f"event {sequence} references an unknown edge")
                previous = event.get("event_id")
            errors.extend(self._dag_errors())
        except (StateGraphError, TypeError, ValueError) as exc:
            errors.append(str(exc))
        execution_count = sum(edge.graph == "execution" for edge in self._edges.values())
        knowledge_count = sum(edge.graph == "knowledge" for edge in self._edges.values())
        result = GraphVerification(
            ok=not errors,
            node_count=len(self._nodes),
            edge_count=len(self._edges),
            execution_edge_count=execution_count,
            knowledge_edge_count=knowledge_count,
            errors=tuple(dict.fromkeys(errors)),
        )
        if raise_on_error and not result.ok:
            raise GraphIntegrityError("; ".join(result.errors))
        return result

    validate = verify
    verify_structural_dag = verify

    def seal(self) -> GraphSnapshot:
        if self._sealed_snapshot is not None:
            return self._sealed_snapshot
        self.verify()
        snapshot = GraphSnapshot(
            version=self.version,
            graph_id=self.graph_id,
            binding=self.binding,
            nodes=tuple(self._nodes.values()),
            edges=tuple(self._edges.values()),
            events=tuple(self._events),
            parent_snapshot=self.parent_snapshot,
            metadata=self.metadata,
        )
        self._sealed_snapshot = snapshot
        self._sealed = True
        return snapshot

    snapshot = seal

    def fork(self, *, metadata: Mapping[str, Any] | None = None) -> "StateGraph":
        parent = self.seal()
        child = StateGraph(
            binding=parent.binding,
            version=parent.version + 1,
            graph_id=_digest({"parent_snapshot": parent.snapshot_id, "version": parent.version + 1}),
            parent_snapshot=parent.snapshot_id,
            metadata=metadata or self.metadata,
        )
        child._nodes.update({node.node_id: node for node in parent.nodes})
        child._edges.update({edge.edge_id: edge for edge in parent.edges})
        child._events.extend(parent.events)
        return child

    branch = fork

    @classmethod
    def from_snapshot(cls, snapshot: GraphSnapshot | Mapping[str, Any]) -> "StateGraph":
        if isinstance(snapshot, GraphSnapshot):
            resolved = snapshot
        else:
            if not isinstance(snapshot, Mapping) or snapshot.get("schema") != GraphSnapshot.schema:
                raise GraphIntegrityError("graph snapshot schema changed")
            nodes = tuple(GraphNode.from_dict(item) for item in snapshot.get("nodes", ()))
            edges = tuple(GraphEdge.from_dict(item) for item in snapshot.get("edges", ()))
            resolved = GraphSnapshot(
                version=snapshot.get("version"),
                graph_id=snapshot.get("graph_id"),
                binding=snapshot.get("binding", {}),
                nodes=nodes,
                edges=edges,
                events=tuple(snapshot.get("events", ())),
                parent_snapshot=snapshot.get("parent_snapshot"),
                metadata=snapshot.get("metadata", {}),
            )
            supplied = snapshot.get("snapshot_id", snapshot.get("fingerprint"))
            if supplied != resolved.snapshot_id:
                raise GraphIntegrityError("graph snapshot content hash changed")
        graph = cls(
            binding=resolved.binding,
            version=resolved.version,
            graph_id=resolved.graph_id,
            parent_snapshot=resolved.parent_snapshot,
            metadata=resolved.metadata,
        )
        graph._nodes.update({node.node_id: node for node in resolved.nodes})
        graph._edges.update({edge.edge_id: edge for edge in resolved.edges})
        graph._events.extend(resolved.events)
        graph.verify()
        graph._sealed = True
        graph._sealed_snapshot = resolved
        return graph

    reopen = from_snapshot
    from_dict = from_snapshot

    @property
    def event_log(self) -> tuple[Mapping[str, Any], ...]:
        return self.events

    def merge(self, *_: Any, **__: Any) -> None:
        raise GraphIntegrityError(
            "arbitrary branch merges are not supported; create a new Action with explicit inputs"
        )

    def restore(self, *_: Any, **__: Any) -> None:
        raise GraphIntegrityError(
            "StateGraph records lineage only; native Bench/StateCut APIs own restore authority"
        )

    def to_mermaid(self, *, include_knowledge: bool = True) -> str:
        lines = ["flowchart TD"]
        for node in sorted(self._nodes.values(), key=lambda item: item.node_id):
            label = node.labels[0] if node.labels else str(node.content.get("label", node.kind))
            safe = str(label).replace('"', "'")
            lines.append(f'    n_{node.node_id[:12]}["{node.kind}: {safe}"]')
        for edge in sorted(self._edges.values(), key=lambda item: item.edge_id):
            if not include_knowledge and edge.graph == "knowledge":
                continue
            arrow = "-.->" if edge.graph == "knowledge" else "-->"
            lines.append(
                f'    n_{edge.source[:12]} {arrow}|{edge.relation}| n_{edge.target[:12]}'
            )
        return "\n".join(lines) + "\n"

    def export(self, path: str | Path, *, format: str | None = None) -> Path:
        destination = Path(path)
        format = (format or destination.suffix.removeprefix(".") or "json").lower()
        snapshot = self.seal()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if format in {"mmd", "mermaid"}:
            payload = snapshot_to_mermaid(snapshot)
        elif format == "json":
            payload = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n"
        else:
            raise StateGraphError(f"unsupported graph export format {format!r}")
        destination.write_text(payload, encoding="utf-8")
        return destination


def snapshot_to_mermaid(snapshot: GraphSnapshot, *, include_knowledge: bool = True) -> str:
    graph = StateGraph.from_snapshot(snapshot)
    return graph.to_mermaid(include_knowledge=include_knowledge)


def add_source_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_source(*args, **kwargs)


def add_state_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_state(*args, **kwargs)


def add_consumer_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_consumer(*args, **kwargs)


def add_action_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_action(*args, **kwargs)


def add_output_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_output(*args, **kwargs)


def source_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_source(*args, **kwargs)


def state_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_state(*args, **kwargs)


def consumer_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_consumer(*args, **kwargs)


def action_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_action(*args, **kwargs)


def output_node(graph: StateGraph, *args: Any, **kwargs: Any) -> str:
    return graph.add_output(*args, **kwargs)


def _load_snapshot(path: Path) -> GraphSnapshot:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateGraphError(f"cannot read graph snapshot {path}") from exc
    return StateGraph.from_snapshot(raw).seal()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdb graph", description="Inspect and verify MDB execution graph snapshots")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "verify", "lineage", "export", "query"):
        sub.add_parser(name)
    inspect = sub.choices["inspect"]
    inspect.add_argument("snapshot", type=Path)
    inspect.add_argument("node_id", nargs="?")
    verify = sub.choices["verify"]
    verify.add_argument("snapshot", type=Path)
    lineage = sub.choices["lineage"]
    lineage.add_argument("snapshot", type=Path)
    lineage.add_argument("node_id")
    lineage.add_argument("--direction", choices=("ancestors", "descendants"), default="ancestors")
    query = sub.choices["query"]
    query.add_argument("snapshot", type=Path)
    query.add_argument("--kind")
    query.add_argument("--relation")
    query.add_argument("--graph", choices=("execution", "knowledge"))
    export = sub.choices["export"]
    export.add_argument("snapshot", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--format", choices=("json", "mermaid"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    snapshot = _load_snapshot(args.snapshot)
    graph = StateGraph.from_snapshot(snapshot)
    if args.command == "inspect":
        payload = graph.inspect(args.node_id) if args.node_id else graph.inspect()
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "verify":
        print(json.dumps(graph.verify(raise_on_error=False).to_dict(), indent=2, sort_keys=True))
        return 0 if graph.verify(raise_on_error=False).ok else 2
    elif args.command == "lineage":
        print(json.dumps([node.to_dict() for node in graph.lineage(args.node_id, direction=args.direction)], indent=2, sort_keys=True))
    elif args.command == "query":
        print(json.dumps([item.to_dict() for item in graph.query(kind=args.kind, relation=args.relation, graph=args.graph)], indent=2, sort_keys=True))
    elif args.command == "export":
        destination = graph.export(args.destination, format=args.format)
        print(str(destination))
    return 0


__all__ = [
    "ActionNode",
    "EVIDENCE_CLASSES",
    "EXECUTION_RELATIONS",
    "Evidence",
    "ExecutionBinding",
    "GraphBinding",
    "GraphEdge",
    "GraphIntegrityError",
    "GraphNode",
    "GraphSnapshot",
    "GraphVerification",
    "KNOWLEDGE_RELATIONS",
    "Node",
    "NodeBinding",
    "NODE_KINDS",
    "OutputNode",
    "SourceNode",
    "StateGraph",
    "StateGraphError",
    "StateNode",
    "STATE_GRAPH_EDGE_SCHEMA",
    "STATE_GRAPH_EVENT_SCHEMA",
    "STATE_GRAPH_NODE_SCHEMA",
    "STATE_GRAPH_SCHEMA",
    "STATE_GRAPH_SNAPSHOT_SCHEMA",
    "add_action_node",
    "add_consumer_node",
    "add_output_node",
    "add_source_node",
    "add_state_node",
    "action_node",
    "consumer_node",
    "evidence",
    "output_node",
    "snapshot_to_mermaid",
    "source_node",
    "state_node",
    "main",
]
