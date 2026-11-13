"""Content-addressed state graphs: build and verify the ``mdb-state-graph-*-v1`` seals.

Standard library only. This is the read side of the history graph format (nodes, edges, events
and sealed snapshots) that recorded projects store in slices and that a session graph is built in:
the same canonical JSON, the same identities, the same structural checks. It builds plain dicts
(``to_dict`` shapes), never executes anything and holds no model state.

Normalisation follows the format: text fields are stripped, evidence classes are lower-case with
``-``, sequences become lists, mappings need text keys, floats must be finite. Anything else is a
``GraphSealError`` (a ``ValueError``).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping

SCHEMA = "mdb-state-graph-v1"
SNAPSHOT_SCHEMA = "mdb-state-graph-snapshot-v1"
EVENT_SCHEMA = "mdb-state-graph-event-v1"
NODE_SCHEMA = "mdb-state-graph-node-v1"
EDGE_SCHEMA = "mdb-state-graph-edge-v1"
BINDING_SCHEMA = "mdb-state-graph-binding-v1"
EVIDENCE_SCHEMA = "mdb-state-graph-evidence-v1"

NODE_KINDS = frozenset({"source", "state", "action", "output"})
EXECUTION_RELATIONS = frozenset({"parent", "continues", "reads", "writes", "derived_from",
                                 "invalidates", "observed_effect"})
KNOWLEDGE_RELATIONS = frozenset({"supports", "contradicts", "describes", "hypothesizes",
                                 "related_to", "annotates"})
EVIDENCE_CLASSES = frozenset({"declared", "observed", "causally-tested", "hypothesis"})


class GraphSealError(ValueError):
    """A graph value cannot be represented, or a sealed graph does not verify."""


# --------------------------------------------------------------------------- canonical JSON

def plain(value, label="value"):
    """Strict JSON copy: text keys, finite floats, sequences as lists."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraphSealError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GraphSealError(f"{label} contains a non-string key")
            out[key] = plain(item, f"{label}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [plain(item, f"{label}[]") for item in value]
    raise GraphSealError(f"{label} contains unsupported {type(value).__name__}")


def canonical(value) -> bytes:
    return json.dumps(plain(value), ensure_ascii=True, allow_nan=False, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise GraphSealError(f"{label} must be non-empty text")
    return value.strip()


def _strings(values, label):
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise GraphSealError(f"{label} must be a sequence of strings")
    out = [_text(value, f"{label}[]") for value in values]
    if len(out) != len(set(out)):
        raise GraphSealError(f"{label} contains duplicate values")
    return out


def sha(value, label):
    text = _text(value, label).lower().removeprefix("sha256:")
    text = text.removeprefix("cas://sha256/")
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise GraphSealError(f"{label} must be a SHA-256 digest")
    return text


def _ref_fingerprint(text):
    """``source_fingerprint`` of a text reference: the digest of a ``cas://sha256/`` ref."""
    if text.startswith("cas://sha256/"):
        candidate = text.removeprefix("cas://sha256/")
        if len(candidate) == 64 and all(c in "0123456789abcdefABCDEF" for c in candidate):
            return candidate.lower()
    return None


# --------------------------------------------------------------------------- values

def binding(model=None, session=None, clock=None, version=None, metadata=None):
    return {"schema": BINDING_SCHEMA, "model": plain(model, "binding.model"),
            "session": plain(session, "binding.session"), "clock": plain(clock, "binding.clock"),
            "version": plain(version, "binding.version"),
            "metadata": plain({} if metadata is None else metadata, "binding.metadata")}


def binding_from(value):
    """A binding dict from a stored binding mapping (``None`` is the empty binding)."""
    if value is None:
        return binding()
    if not isinstance(value, Mapping):
        raise GraphSealError("binding must be a JSON object")
    return binding(value.get("model", value.get("model_id")),
                   value.get("session", value.get("session_id")),
                   value.get("clock", value.get("time")),
                   value.get("version", value.get("version_id")), value.get("metadata", {}))


def evidence(evidence_class="declared", *, condition=None, sign=None, dose=None, effect=None,
             receipt_refs=(), metadata=None):
    cls = _text(evidence_class, "evidence_class").lower().replace("_", "-")
    if cls not in EVIDENCE_CLASSES:
        raise GraphSealError(f"unsupported evidence class {cls!r}")
    return {"schema": EVIDENCE_SCHEMA, "class": cls,
            "condition": plain({} if condition is None else condition, "evidence.condition"),
            "sign": plain(sign, "evidence.sign"), "dose": plain(dose, "evidence.dose"),
            "effect": plain(effect, "evidence.effect"),
            "receipt_refs": _strings(receipt_refs, "receipt_refs"),
            "metadata": plain({} if metadata is None else metadata, "evidence.metadata")}


def evidence_from(value):
    if value is None:
        return evidence()
    if isinstance(value, str):
        return evidence(value)
    if not isinstance(value, Mapping):
        raise GraphSealError("evidence must be a class string or JSON object")
    return evidence(value.get("evidence_class", value.get("class", value.get("class_", "declared"))),
                    condition=value.get("condition", {}), sign=value.get("sign"),
                    dose=value.get("dose"), effect=value.get("effect"),
                    receipt_refs=tuple(value.get("receipt_refs", value.get("receipts", ())) or ()),
                    metadata=value.get("metadata", {}))


def _evidence_list(values):
    if values is None:
        return []
    if isinstance(values, (Mapping, str)):
        values = (values,)
    return [evidence_from(v) for v in values]


def node(kind, content, *, binding_=None, evidence_=None, labels=()):
    """A node dict (``to_dict`` shape) with its content-addressed ``node_id``."""
    kind = _text(kind, "node.kind").lower()
    if kind not in NODE_KINDS:
        raise GraphSealError(f"unsupported node kind {kind!r}")
    body = {"schema": NODE_SCHEMA, "kind": kind, "content": plain(content, "node.content"),
            "binding": binding_ if binding_ is not None else binding(),
            "evidence": _evidence_list(evidence_), "labels": _strings(labels, "node.labels")}
    identifier = digest(body)
    return {**body, "node_id": identifier, "fingerprint": identifier}


def edge(source, target, relation, *, graph="execution", evidence_=None, binding_=None,
         metadata=None):
    relation = _text(relation, "edge.relation")
    graph = _text(graph, "edge.graph").lower()
    if graph not in {"execution", "knowledge"}:
        raise GraphSealError("edge.graph must be 'execution' or 'knowledge'")
    if graph == "execution" and relation not in EXECUTION_RELATIONS:
        raise GraphSealError(f"unsupported execution relation {relation!r}")
    if graph == "knowledge" and relation not in KNOWLEDGE_RELATIONS:
        raise GraphSealError(f"unsupported knowledge relation {relation!r}")
    body = {"schema": EDGE_SCHEMA, "source": _text(source, "edge.source"),
            "target": _text(target, "edge.target"), "relation": relation, "graph": graph,
            "evidence": _evidence_list(evidence_),
            "binding": binding_ if binding_ is not None else binding(),
            "metadata": plain({} if metadata is None else metadata, "edge.metadata")}
    identifier = digest(body)
    return {**body, "edge_id": identifier, "fingerprint": identifier}


def node_from(value):
    """Re-derive a stored node; raises when its ``node_id`` differs from its content."""
    if not isinstance(value, Mapping) or value.get("schema") != NODE_SCHEMA:
        raise GraphSealError("graph node schema changed")
    out = node(value.get("kind"), value.get("content", {}),
               binding_=binding_from(value.get("binding", {})),
               evidence_=tuple(value.get("evidence", ())), labels=tuple(value.get("labels", ())))
    if value.get("node_id", value.get("fingerprint")) != out["node_id"]:
        raise GraphSealError("graph node content hash changed")
    return out


def edge_from(value):
    if not isinstance(value, Mapping) or value.get("schema") != EDGE_SCHEMA:
        raise GraphSealError("graph edge schema changed")
    out = edge(value.get("source"), value.get("target"), value.get("relation"),
               graph=value.get("graph", "execution"), evidence_=tuple(value.get("evidence", ())),
               binding_=binding_from(value.get("binding", {})), metadata=value.get("metadata", {}))
    if value.get("edge_id", value.get("fingerprint")) != out["edge_id"]:
        raise GraphSealError("graph edge content hash changed")
    return out


# --------------------------------------------------------------------------- structure

def dag_errors(nodes: Mapping[str, Mapping], edges: Iterable[Mapping]) -> list[str]:
    """Structural problems of the execution graph (one parent, acyclic, typed endpoints)."""
    execution = [e for e in edges if e["graph"] == "execution"]
    adjacency = {n: [] for n in nodes}
    incoming = {}
    for e in execution:
        if e["source"] not in nodes or e["target"] not in nodes:
            continue
        adjacency[e["source"]].append(e["target"])
        if e["relation"] in {"parent", "continues"}:
            incoming[e["target"]] = incoming.get(e["target"], 0) + 1
    errors = [f"node {t} has {c} parent/continues inputs; use one Action node with multiple reads"
              for t, c in incoming.items() if c > 1]
    color = {n: 0 for n in nodes}
    for start in sorted(nodes):
        if color[start]:
            continue
        stack = [(start, iter(adjacency[start]))]
        color[start] = 1
        while stack:
            current, children = stack[-1]
            for target in children:
                if color[target] == 1:
                    errors.append(f"execution lineage cycle includes {current} -> {target}")
                elif color[target] == 0:
                    color[target] = 1
                    stack.append((target, iter(adjacency[target])))
                    break
            else:
                color[current] = 2
                stack.pop()
    for e in execution:
        if e["source"] not in nodes or e["target"] not in nodes:
            continue
        if e["relation"] == "reads" and nodes[e["target"]]["kind"] != "action":
            errors.append(f"reads edge target {e['target']} is not an action")
        if e["relation"] in {"writes", "observed_effect"} and nodes[e["source"]]["kind"] != "action":
            errors.append(f"{e['relation']} edge source {e['source']} is not an action")
    for n in nodes.values():
        if n["kind"] != "action":
            continue
        content = n["content"] if isinstance(n["content"], Mapping) else {}
        inputs, output = content.get("inputs") or [], content.get("output")
        if any(not isinstance(i, str) or i not in nodes for i in inputs):
            errors.append(f"action {n['node_id']} references an unknown input")
        if output is not None and output not in nodes:
            errors.append(f"action {n['node_id']} references an unknown output")
        if output is not None and not any(e["source"] == n["node_id"] and e["target"] == output
                                          and e["relation"] == "writes" for e in execution):
            errors.append(f"action {n['node_id']} declares output without a writes edge")
    return list(dict.fromkeys(errors))


def _event_errors(events, nodes, edges):
    errors, previous = [], None
    for sequence, event in enumerate(events):
        body = {k: v for k, v in event.items() if k != "event_id"}
        if event.get("sequence") != sequence or event.get("previous_event") != previous:
            errors.append(f"event {sequence} sequence/chain changed")
        if event.get("schema") != EVENT_SCHEMA or event.get("event_id") != digest(body):
            errors.append(f"event {sequence} hash changed")
        if event.get("kind") not in {"node", "edge"}:
            errors.append(f"event {sequence} has an unsupported kind")
        if event.get("kind") == "node" and event.get("object_id") not in nodes:
            errors.append(f"event {sequence} references an unknown node")
        if event.get("kind") == "edge" and event.get("object_id") not in edges:
            errors.append(f"event {sequence} references an unknown edge")
        previous = event.get("event_id")
    return errors


def _snapshot_body(version, graph_id, binding_, nodes, edges, events, parent, metadata):
    return {"schema": SNAPSHOT_SCHEMA, "version": version, "graph_id": graph_id,
            "binding": binding_, "nodes": [nodes[k] for k in sorted(nodes)],
            "edges": [edges[k] for k in sorted(edges)], "events": events,
            "parent_snapshot": parent, "metadata": metadata}


def verify_snapshot(value) -> dict:
    """Verify a stored sealed snapshot (every node, edge, event and the snapshot id) and return
    ``{"nodes", "edges", "snapshot_id"}``; raises ``GraphSealError`` on any mismatch."""
    if not isinstance(value, Mapping) or value.get("schema") != SNAPSHOT_SCHEMA:
        raise GraphSealError("graph snapshot schema changed")
    nodes_in, edges_in = value.get("nodes", ()), value.get("edges", ())
    nodes = [node_from(item) for item in nodes_in]
    edges = [edge_from(item) for item in edges_in]
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise GraphSealError("snapshot version must be a non-negative integer")
    graph_id = _text(value.get("graph_id"), "graph_id")
    by_node = {n["node_id"]: n for n in nodes}
    by_edge = {e["edge_id"]: e for e in edges}
    if len(by_node) != len(nodes) or len(by_edge) != len(edges):
        raise GraphSealError("snapshot contains duplicate identities")
    events = []
    for event in value.get("events", ()):
        if not isinstance(event, Mapping):
            raise GraphSealError("snapshot.events must contain JSON objects")
        events.append(plain(event, "snapshot.event"))
    parent = value.get("parent_snapshot")
    parent = None if parent is None else sha(parent, "parent_snapshot")
    body = _snapshot_body(version, graph_id, binding_from(value.get("binding", {})), by_node,
                          by_edge, events, parent, plain(value.get("metadata", {}),
                                                         "snapshot.metadata"))
    snapshot_id = digest(body)
    if value.get("snapshot_id", value.get("fingerprint")) != snapshot_id:
        raise GraphSealError("graph snapshot content hash changed")
    errors = [f"edge {e['edge_id']} references an unknown node" for e in edges
              if e["source"] not in by_node or e["target"] not in by_node]
    errors += _event_errors(events, by_node, by_edge) + dag_errors(by_node, edges)
    if errors:
        raise GraphSealError("; ".join(dict.fromkeys(errors)))
    return {"nodes": nodes, "edges": edges, "snapshot_id": snapshot_id}


# --------------------------------------------------------------------------- builder

class GraphBuilder:
    """Append-only graph with the event chain of the format; ``seal`` returns the snapshot dict.

    Only the constructors a recorded project's session graph uses: sources, states, actions,
    outputs and edges between nodes already added.
    """

    def __init__(self, *, metadata=None):
        self.binding = binding()
        self.graph_id = digest({"schema": SCHEMA, "binding": self.binding, "version": 0})
        self.metadata = plain(metadata or {}, "graph.metadata")
        self.nodes, self.edges, self.events = {}, {}, []

    def _event(self, kind, object_id):
        previous = self.events[-1]["event_id"] if self.events else None
        body = {"schema": EVENT_SCHEMA, "sequence": len(self.events), "kind": kind,
                "object_id": object_id, "previous_event": previous}
        self.events.append({**body, "event_id": digest(body)})

    def add_node(self, value):
        identifier = value["node_id"]
        existing = self.nodes.get(identifier)
        if existing is not None:
            if existing != value:
                raise GraphSealError("content-addressed node identity collision")
            return identifier
        self.nodes[identifier] = value
        self._event("node", identifier)
        return identifier

    def _known(self, identifier, label):
        if not isinstance(identifier, str) or identifier not in self.nodes:
            raise GraphSealError(f"unknown {label} {identifier!r}")
        return identifier

    def add_edge(self, source, target, relation, *, graph=None, evidence_=None, metadata=None):
        source = self._known(source, "edge.source")
        target = self._known(target, "edge.target")
        relation = _text(relation, "edge.relation")
        graph = graph or ("execution" if relation in EXECUTION_RELATIONS else "knowledge")
        value = edge(source, target, relation, graph=graph,
                     evidence_=evidence() if evidence_ is None else evidence_, metadata=metadata)
        identifier = value["edge_id"]
        existing = self.edges.get(identifier)
        if existing is not None:
            if existing != value:
                raise GraphSealError("content-addressed edge identity collision")
            return identifier
        self.edges[identifier] = value
        self._event("edge", identifier)
        return identifier

    @staticmethod
    def source_node(source_ref, *, binding_=None, metadata=None, evidence_=None, labels=()):
        ref = _text(source_ref, "source")
        return node("source", {"source_ref": ref, "source_fingerprint": _ref_fingerprint(ref),
                               "metadata": metadata or {}},
                    binding_=binding_, evidence_=evidence_, labels=labels)

    @staticmethod
    def state_node(cut_ref, fingerprint, *, binding_=None, metadata=None):
        return node("state", {"state_ref": _text(cut_ref, "state"),
                              "state_fingerprint": sha(fingerprint, "state.fingerprint"),
                              "metadata": metadata or {}}, binding_=binding_)

    @staticmethod
    def output_node(output_ref, *, binding_=None, metadata=None, evidence_=None, labels=()):
        return node("output", {"output_ref": _text(output_ref, "output_ref"),
                               "metadata": metadata or {}},
                    binding_=binding_, evidence_=evidence_, labels=labels)

    def add_action(self, act_ref, *, inputs=(), output=None, receipt_refs=(), binding_=None,
                   metadata=None, evidence_class="declared"):
        inputs = [self._known(i, "action input") for i in inputs]
        output = None if output is None else self._known(output, "action output")
        content = {"act_ref": _text(act_ref, "act_ref"), "action": None, "inputs": inputs,
                   "output": output, "receipt_refs": _strings(receipt_refs, "action.receipt_refs"),
                   "reads": [], "writes": [], "invalidates": [], "metadata": metadata or {}}
        ev = evidence(evidence_class)
        action = self.add_node(node("action", content, binding_=binding_, evidence_=[ev]))
        for item in inputs:
            self.add_edge(item, action, "reads", evidence_=[ev])
        if output is not None:
            self.add_edge(action, output, "writes", evidence_=[ev])
        return action

    def seal(self):
        errors = dag_errors(self.nodes, self.edges.values())
        if errors:
            raise GraphSealError("; ".join(errors))
        body = _snapshot_body(0, self.graph_id, self.binding, self.nodes, self.edges,
                              list(self.events), None, self.metadata)
        identifier = digest(body)
        return {**body, "snapshot_id": identifier, "fingerprint": identifier}
