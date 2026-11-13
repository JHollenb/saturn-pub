"""Qualified, branch-bound declared dataflow over native cuts and receipts.

Only metadata is read. Port versions are descriptive graph outputs, never new
checkpoints. Missing or ambiguous history stays an explicit frontier.
"""

import argparse
import json
from collections import defaultdict

from saturn_pub.contracts import StateAddress
from saturn_pub.symbols import SymbolBinding
from saturn_pub.values import digest

from .state_graph import StateGraph


def slot_descriptors(cut):
    return {row["key"]["value"]: row["value"] for row in cut["manifest"]["payload"]["items"]}


def bind_live_symbol(adapter, words):
    """The CLI's supplied-name operation uses the original native binder."""

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

    parser = Parser(prog="project symbol-bind", add_help=False)
    parser.add_argument("name")
    parser.add_argument("--port", action="append", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--context-slot", action="append", default=[])
    parser.add_argument("--claim", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--selector", default="[]", help="JSON integer coordinates; one port only")
    parser.add_argument("--role", default="")
    args = parser.parse_args(words)
    selector = json.loads(args.selector)
    if not isinstance(selector, list) or (selector and len(args.port) != 1):
        raise ValueError("selector must be an integer list for one port")
    cut = adapter.session.capture(retain=False)
    binding = SymbolBinding.bind(
        adapter.session,
        args.name,
        tuple(StateAddress(name, tuple(selector), args.role) for name in args.port),
        context=args.context,
        context_slots=tuple(args.context_slot),
        evidence=tuple(args.evidence) or ("cas://sha256/" + cut.fingerprint,),
        claim=args.claim,
    )
    if binding.to_dict()["fingerprint"] not in {
        b.to_dict()["fingerprint"] for b in adapter.symbols.bindings
    }:
        adapter.symbols.bindings.append(binding)
    adapter.symbol_context = args.context
    adapter._persist()
    return {
        "binding": binding.to_dict(),
        "cut_id": cut.fingerprint,
        "support": "native-support-verified",
        "authority": "read-only-debug-info",
    }


class _Slice:
    def __init__(self, project, sid, cut_id, receipt_id=None):
        self.project, self.sid, self.root = project, sid, cut_id
        row = project.session(sid)
        self.model, self.execution = json.loads(row["identity"]), json.loads(row["execution"])
        self.binding = {
            "model": self.model,
            "session": sid,
            "metadata": {"execution": self.execution},
        }
        self.graph = StateGraph(
            binding=self.binding,
            metadata={
                "source": "mdb-project-port-slice",
                "version": 1,
                "authority": "declared-dependency",
                "restorable": False,
            },
        )
        self.retained = set(project.cut_ids(sid))
        self.cuts, self.versions, self.done, self.active = {}, {}, set(), set()
        self.gaps, self.used_receipts = [], set()
        self.producers = defaultdict(list)
        for receipt in project.receipts(sid):
            body = receipt.to_dict()
            if body.get("status") == "refused" or body["operation"] in {"restore", "abort"}:
                continue
            if body.get("model_identity") != self.model or digest(body.get("execution")) != digest(
                self.execution
            ):
                raise ValueError("slice receipt differs from the project execution binding")
            self.producers[body.get("result")].append(body)
        self.receipt_id = receipt_id
        if receipt_id and not any(r["fingerprint"] == receipt_id for r in self.producers[cut_id]):
            raise ValueError("selected receipt does not produce the requested cut")

    def cut(self, cid):
        if cid not in self.retained:
            return None
        if cid not in self.cuts:
            self.cuts[cid] = self.project.cut_descriptor(self.sid, cid)
        return self.cuts[cid]

    def producer(self, cid):
        rows = self.producers[cid]
        if cid == self.root and self.receipt_id:
            rows = [r for r in rows if r["fingerprint"] == self.receipt_id]
        return rows[0] if len(rows) == 1 else None

    def version(self, boundary, slot):
        key = (digest(boundary), slot)
        if key not in self.versions:
            cut = self.cut(boundary.get("cut"))
            if cut and slot not in slot_descriptors(cut):
                raise ValueError(f"receipt references an unavailable slot: {slot}")
            self.versions[key] = self.graph.add_output(
                output_ref="port-version://sha256/" + digest([self.sid, boundary, slot]),
                binding=self.binding,
                evidence="declared",
                labels=(slot,),
                metadata={
                    "boundary": boundary,
                    "slot": slot,
                    "native_cut": boundary.get("cut"),
                    "value": slot_descriptors(cut)[slot] if cut else None,
                    "retained_cut": cut is not None,
                    "restorable": False,
                },
            )
        return self.versions[key]

    def gap(self, target, kind, **detail):
        body = {"kind": kind, "target": target, **detail}
        if body not in self.gaps:
            self.gaps.append(body)
            source = self.graph.add_source(
                source_ref="slice-gap://sha256/" + digest(body), binding=self.binding, metadata=body
            )
            self.graph.add_edge(source, target, "describes", graph="knowledge")

    def copy(self, source, target, *, receipt=None, reason="preserved-slot"):
        left = self.graph.nodes[source].content["metadata"]["value"]
        right = self.graph.nodes[target].content["metadata"]["value"]
        verified = left is not None and right is not None
        if verified and left != right:
            raise ValueError("slice copy differs from sealed native slot descriptors")
        self.graph.add_edge(
            source,
            target,
            "derived_from",
            evidence="observed" if verified else "declared",
            metadata={
                "reason": reason,
                "receipt": receipt,
                "proof": "content-equality" if verified else "declared-preservation",
            },
        )

    def expand(self, boundary, slot, build):
        node = self.version(boundary, slot)
        if node in self.active:
            raise ValueError("cycle in native slice provenance")
        if node not in self.done:
            self.active.add(node)
            try:
                build(node)
                self.done.add(node)
            finally:
                self.active.remove(node)
        return node

    def value(self, cid, slot):
        return self.expand({"cut": cid}, slot, lambda node: self._value(cid, slot, node))

    def _value(self, cid, slot, node):
        cut = self.cut(cid)
        if cut is None:
            self.gap(node, "unretained-cut", cut=cid)
            return
        receipt = self.producer(cid)
        if receipt is None and self.producers[cid]:
            self.gap(
                node,
                "ambiguous-producer",
                cut=cid,
                receipts=[r["fingerprint"] for r in self.producers[cid]],
            )
            return
        if receipt:
            self.used_receipts.add(receipt["fingerprint"])
            if receipt["operation"] == "commit":
                source = self.value(receipt.get("candidate"), slot)
                self.copy(source, node, receipt=receipt["fingerprint"], reason="commit-copy")
                return
            if receipt["operation"] == "apply":
                specs = [
                    {
                        "native_operation": {
                            **receipt["act"],
                            "operation": receipt["act"]["name"],
                            "footprint": "declared",
                        }
                    }
                ]
            elif receipt["operation"] == "continue" and receipt.get("transitions"):
                specs = receipt["transitions"]
                if specs[-1]["after"] != cut["execution_point"]:
                    raise ValueError("receipt end point differs from its retained cut")
                parent = self.cut(receipt["parent"])
                if parent and specs[0]["before"] != parent["execution_point"]:
                    raise ValueError("receipt start point differs from its retained parent")
                if any(a["after"] != b["before"] for a, b in zip(specs, specs[1:])):
                    raise ValueError("receipt transition clocks are discontinuous")
            else:
                self.gap(
                    node,
                    "unsupported-transaction",
                    receipt=receipt["fingerprint"],
                    operation=receipt["operation"],
                )
                return
            self._step(receipt, specs, len(specs), slot, node)
            return
        parent = self.cut(cut["parent"])
        if cut["parent"] is None:
            return  # Retained initial input is a known boundary, not an invented writer.
        if parent:
            left, right = dict(parent["manifest"]), dict(cut["manifest"])
            left.pop("parent")
            right.pop("parent")
            if left == right:
                self.copy(self.value(cut["parent"], slot), node, reason="equal-state-lineage")
                return
        self.gap(node, "unexplained-capture", cut=cid, parent=cut["parent"])

    def boundary(self, receipt, specs, index, slot):
        if index == 0:
            return self.value(receipt["parent"], slot)
        boundary = {
            "receipt": receipt["fingerprint"],
            "transition": index,
            "point": specs[index - 1].get("after"),
        }
        return self.expand(
            boundary, slot, lambda node: self._step(receipt, specs, index, slot, node)
        )

    def _step(self, receipt, specs, index, slot, target):
        step = specs[index - 1]
        spec = step["native_operation"]
        if spec.get("footprint") != "declared":
            self.gap(
                target, "unknown-footprint", receipt=receipt["fingerprint"], transition=index - 1
            )
            return
        touched = set(spec.get("writes", [])) | set(spec.get("invalidates", []))
        if slot not in touched:
            self.copy(
                self.boundary(receipt, specs, index - 1, slot),
                target,
                receipt=receipt["fingerprint"],
            )
            return
        inputs = [self.boundary(receipt, specs, index - 1, name) for name in spec.get("reads", [])]
        action = self.graph.add_action(
            action=spec,
            inputs=inputs,
            binding=self.binding,
            act_ref="occurrence://sha256/" + digest([self.sid, receipt["fingerprint"], index - 1]),
            receipt_refs=("cas://sha256/" + receipt["fingerprint"],),
            evidence="declared",
            labels=(spec["operation"],),
            metadata={
                "receipt": receipt["fingerprint"],
                "transition": index - 1,
                "point": step.get("before"),
                "operation": spec["operation"],
                "transaction": receipt["operation"],
            },
        )
        self.graph.add_edge(
            action,
            target,
            "writes" if slot in spec.get("writes", []) else "invalidates",
            evidence="declared",
        )

    def branch(self, cid, active=None):
        active = set() if active is None else active
        if cid in active:
            raise ValueError("cycle in native branch history")
        cut = self.cut(cid)
        if cut is None:
            return set()
        active.add(cid)
        result = {cid}
        if cut["parent"]:
            result.update(self.branch(cut["parent"], active))
        receipt = self.producer(cid)
        if receipt and receipt["operation"] == "commit":
            result.update(self.branch(receipt.get("candidate"), active))
        active.remove(cid)
        return result


def _resolve(project, sid, cut, symbol, context, binding_id):
    row = project.session(sid)
    context = (
        context if context is not None else json.loads(row["debug_info"]).get("symbol_context", "")
    )
    matches, reasons = [], []
    for body in project.symbols(sid):
        if body["symbol"] != symbol or (binding_id and body["fingerprint"] != binding_id):
            continue
        binding = SymbolBinding.from_dict(body)
        try:
            binding.resolve_metadata(
                model_identity=cut["model_identity"],
                execution=cut["manifest"]["execution"],
                execution_point=cut["execution_point"],
                surface=cut["surface"],
                slots=slot_descriptors(cut),
                context=context,
            )
            matches.append(body)
        except ValueError as error:
            reasons.append(str(error))
    if len(matches) != 1:
        raise ValueError(
            f"symbol {symbol!r} requires one supported binding; found {len(matches)}; "
            + ("; ".join(reasons) or "no binding; select --binding or --context")
        )
    return matches[0]


def _downstream(graph, targets):
    selected = set(targets)
    changed = True
    while changed:
        changed = False
        for edge in graph.edges.values():
            if (
                edge.graph == "execution"
                and edge.source in selected
                and edge.target not in selected
            ):
                selected.add(edge.target)
                changed = True
    side_inputs = {
        n
        for key in selected
        if graph.nodes[key].kind == "action"
        for n in graph.nodes[key].content["inputs"]
    } - selected
    selected.update(side_inputs)
    selected.update(
        edge.source
        for edge in graph.edges.values()
        if edge.graph == "knowledge" and edge.target in selected
    )
    result = StateGraph(binding=graph.binding, metadata=dict(graph.metadata))
    for key in sorted(selected):
        result.add_node(graph.nodes[key])
    for edge in graph.edges.values():
        if edge.source in selected and edge.target in selected:
            result.add_edge(
                edge.source,
                edge.target,
                edge.relation,
                graph=edge.graph,
                evidence=edge.evidence,
                binding=edge.binding,
                metadata=edge.metadata,
            )
    return result, sorted(side_inputs)


def _adjacent_operations(graph, targets, *, downstream=False):
    """Follow value copies to the nearest declared writers or consumers."""
    pending, seen, actions = list(targets), set(), set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        for edge in graph.edges.values():
            if edge.graph != "execution":
                continue
            origin, destination = (
                (edge.source, edge.target) if downstream else (edge.target, edge.source)
            )
            if origin != node:
                continue
            if edge.relation == "derived_from":
                pending.append(destination)
            elif graph.nodes[destination].kind == "action":
                actions.add(destination)
    return [
        {"node_id": key, **graph.nodes[key].to_dict()["content"]["metadata"]}
        for key in sorted(actions)
    ]


def slice_project(
    project,
    sid,
    *,
    port=None,
    symbol=None,
    cut_id=None,
    to_cut=None,
    context=None,
    binding_id=None,
    receipt_id=None,
):
    if bool(port) == bool(symbol):
        raise ValueError("slice requires exactly one port or symbol")
    cut_id = cut_id or project.session(sid)["selected"]
    builder = _Slice(project, sid, cut_id, receipt_id)
    cut = builder.cut(cut_id)
    if cut is None:
        raise ValueError("slice cut is not retained in this project session")
    binding = _resolve(project, sid, cut, symbol, context, binding_id) if symbol else None
    ports = list(dict.fromkeys(loc["slot"] for loc in binding["locations"])) if binding else [port]
    if any(name not in {s["name"] for s in cut["surface"].get("slots", [])} for name in ports):
        raise ValueError("slice port is not declared at the requested cut")
    targets = [builder.value(cut_id, name) for name in ports]
    graph, side_inputs = builder.graph, []
    if to_cut:
        endpoint = builder.cut(to_cut)
        if endpoint is None or cut_id not in builder.branch(to_cut):
            raise ValueError("destination cut is outside the requested branch history")
        for name in slot_descriptors(endpoint):
            builder.value(to_cut, name)
        graph, side_inputs = _downstream(builder.graph, targets)
    if binding:
        source = graph.add_source(
            source_ref="symbol://sha256/" + binding["fingerprint"],
            binding=builder.binding,
            labels=(binding["symbol"],),
            metadata={
                "binding": binding,
                "support": "metadata-support-verified",
                "cut": cut_id,
                "scope": "whole-slot-dependencies; selectors retained in binding",
            },
        )
        for target in targets:
            graph.add_edge(source, target, "describes", graph="knowledge", evidence="declared")
    writers = _adjacent_operations(graph, targets)
    consumers = _adjacent_operations(graph, targets, downstream=True)
    incoming = {e.target for e in graph.edges.values() if e.graph == "execution"}
    frontier = [
        {
            "node_id": n.node_id,
            **n.to_dict()["content"]["metadata"],
            "context_input": n.node_id in side_inputs,
        }
        for n in graph.nodes.values()
        if n.kind == "output" and n.node_id not in incoming
    ]
    snapshot = graph.seal().to_dict()
    result = {
        "schema": "mdb-project-port-slice-v1",
        "session_id": sid,
        "component": f"symbol:{symbol}" if symbol else f"port:{port}",
        "cut_id": cut_id,
        "to_cut": to_cut,
        "view": "declared",
        "direction": "downstream" if to_cut else "upstream",
        "targets": targets,
        "binding": builder.binding,
        "symbol_binding": binding,
        "support": "metadata-support-verified" if binding else "declared-port",
        "scope": "whole-slot; not selector-specific influence",
        "authority": "declared-dependency",
        "causal_effect": "not-established",
        "restorable": False,
        "graph": snapshot,
        "gaps": builder.gaps,
        "complete": not builder.gaps,
        "context_inputs": side_inputs,
        "writers": writers,
        "consumers": consumers,
        "frontier": frontier,
        "evidence": {"cuts": sorted(builder.cuts), "receipts": sorted(builder.used_receipts)},
    }
    identifier = digest(result)
    with project._db() as db:
        db.execute(
            "INSERT OR IGNORE INTO slices VALUES (?, ?, ?)",
            (identifier, sid, project._json(result)),
        )
        saved = db.execute(
            "SELECT body FROM slices WHERE session_id=? AND id=?", (sid, identifier)
        ).fetchone()[0]
        if saved != project._json(result):
            raise ValueError("slice identity collision")
    return {"slice_id": identifier, **result}
