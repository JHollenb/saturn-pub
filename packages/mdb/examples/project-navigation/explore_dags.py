"""Offline DAG experiment; descriptive views, never restorable port checkpoints.

The port DAG intentionally handles only declared, one-transition transactions.
Missing history and wider transactions remain explicit gaps rather than guesses.
"""

import argparse
import json
from collections import Counter, defaultdict
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from saturn_pub.values import digest

from mdb import Project
from mdb.project_navigation import _operations
from mdb.state_graph import StateGraph


def topology(edges):
    parents = defaultdict(set)
    for source, target in edges:
        parents[source]
        parents[target].add(source)
    try:
        order = list(TopologicalSorter(parents).static_order())
        return {"acyclic": True, "nodes": len(order), "cycle": []}
    except CycleError as error:
        return {"acyclic": False, "nodes": len(parents), "cycle": error.args[1]}


def redundant_edges(edges):
    """Count direct edges whose endpoints are also linked by a longer path."""
    adjacency = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
    count = 0
    for source, target in set(edges):
        pending, seen = list(adjacency[source] - {target}), set()
        while pending:
            node = pending.pop()
            if node == target:
                count += 1
                break
            if node not in seen:
                seen.add(node)
                pending.extend(adjacency[node] - seen)
    return count


def slots(cut):
    return {row["key"]["value"]: row["value"] for row in cut["manifest"]["payload"]["items"]}


def inspect(project, sid):
    native = project.graph(sid)
    cuts = {cid: project.cut_descriptor(sid, cid) for cid in project.cut_ids(sid)}
    receipts = [r.to_dict() for r in project.receipts(sid)]
    missing = sorted({r["parent"] for r in receipts if r.get("parent") not in cuts})
    equivalents = defaultdict(list)
    for cid, cut in cuts.items():
        body = dict(cut["manifest"])
        body.pop("parent")
        equivalents[digest(body)].append(cid)
    type_edges = []
    for op in _operations(project, sid):
        spec = op["spec"]
        operation = "operation:" + spec["operation"]
        type_edges.extend(("port:" + port, operation) for port in spec.get("reads", []))
        type_edges.extend(
            (operation, "port:" + port)
            for port in (*spec.get("writes", []), *spec.get("invalidates", []))
        )

    session = project.session(sid)
    binding = {
        "model": json.loads(session["identity"]),
        "session": sid,
        "metadata": {"execution": json.loads(session["execution"])},
    }
    graph = StateGraph(
        binding=binding,
        metadata={
            "source": "versioned-port-dag-experiment",
            "authority": "declared-data-dependency; not semantic proof",
        },
    )
    values, gaps, producers = {}, [], set()
    receipt_results = {r.get("result") for r in receipts if r.get("status") != "refused"}

    def value(cid, slot):
        key = (cid, slot)
        if key not in values:
            descriptor = slots(cuts[cid]).get(slot) if cid in cuts else None
            values[key] = graph.add_output(
                output_ref="port-version://sha256/" + digest([sid, cid, slot]),
                metadata={
                    "native_cut": cid,
                    "slot": slot,
                    "value": descriptor,
                    "retained_cut": cid in cuts,
                    "restorable": False,
                },
                binding=binding,
                labels=(slot,),
                evidence="declared",
            )
        return values[key]

    for receipt in receipts:
        parent, result = receipt.get("parent"), receipt.get("result")
        kind = receipt["operation"]
        if receipt.get("status") == "refused" or kind in {"restore", "abort"}:
            continue
        if result not in cuts:
            gaps.append({"kind": "unretained-result", "receipt": receipt["fingerprint"]})
            continue
        if kind == "commit":
            candidate = receipt.get("candidate")
            if candidate not in cuts:
                gaps.append({"kind": "unretained-candidate", "receipt": receipt["fingerprint"]})
                continue
            for slot, descriptor in slots(cuts[result]).items():
                if slots(cuts[candidate]).get(slot) != descriptor:
                    raise ValueError("commit candidate differs from result")
                graph.add_edge(
                    value(candidate, slot),
                    value(result, slot),
                    "derived_from",
                    evidence="observed",
                    metadata={"receipt": receipt["fingerprint"], "operation": "commit-copy"},
                )
            producers.add(result)
            continue
        if kind == "continue" and len(receipt.get("transitions", [])) == 1:
            spec = receipt["transitions"][0]["native_operation"]
        elif kind == "apply":
            spec = {**receipt["act"], "operation": receipt["act"]["name"], "footprint": "declared"}
        else:
            gaps.append({"kind": "unsupported-transaction", "receipt": receipt["fingerprint"]})
            continue
        if spec.get("footprint") != "declared":
            gaps.append({"kind": "unknown-footprint", "receipt": receipt["fingerprint"]})
            continue
        action = graph.add_action(
            action=spec,
            act_ref="cas://sha256/" + receipt["fingerprint"],
            inputs=[value(parent, slot) for slot in spec.get("reads", [])],
            receipt_refs=("cas://sha256/" + receipt["fingerprint"],),
            binding=binding,
            evidence="declared",
            labels=(spec["operation"],),
        )
        touched = set(spec.get("writes", [])) | set(spec.get("invalidates", []))
        for slot in touched:
            relation = "writes" if slot in spec.get("writes", []) else "invalidates"
            graph.add_edge(action, value(result, slot), relation, evidence="declared")
        if parent in cuts:
            for slot, descriptor in slots(cuts[result]).items():
                if slot not in touched:
                    if slots(cuts[parent]).get(slot) != descriptor:
                        raise ValueError("native preserved slot changed")
                    graph.add_edge(
                        value(parent, slot),
                        value(result, slot),
                        "derived_from",
                        evidence="observed",
                        metadata={"operation": "preserved-slot"},
                    )
        else:
            gaps.append({"kind": "unretained-parent", "receipt": receipt["fingerprint"]})
        producers.add(result)

    # Native forks may capture a new lineage identity with identical state.
    for cid, cut in cuts.items():
        parent = cut["parent"]
        if (
            cid not in producers
            and cid not in receipt_results
            and parent in cuts
            and slots(cuts[parent]) == slots(cut)
        ):
            for slot in slots(cut):
                graph.add_edge(
                    value(parent, slot),
                    value(cid, slot),
                    "derived_from",
                    evidence="observed",
                    metadata={"operation": "equal-state-lineage"},
                )
    graph.verify()
    execution = [(e.source, e.target) for e in native.edges.values() if e.graph == "execution"]
    summary = {
        "session": sid,
        "cuts": len(cuts),
        "receipts": len(receipts),
        "native_graph": {
            "nodes": len(native.nodes),
            "edges": len(native.edges),
            "verified_dag": native.verify().ok,
            "redundant_execution_edges": redundant_edges(execution),
        },
        "component_graph": topology(type_edges),
        "versioned_port_graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "verified_dag": graph.verify().ok,
            "gaps": gaps,
        },
        "missing_receipt_parents": missing,
        "equal_state_different_lineage": [v for v in equivalents.values() if len(v) > 1],
        "operations": dict(Counter(r["operation"] for r in receipts)),
        "replacement_declared_reads": [
            r["act"]["reads"]
            for r in receipts
            if r["operation"] == "apply"
            and r["act"].get("operation", {}).get("id") == "saturn.builtin.replace"
        ],
        "commit_candidate_is_graph_input": [
            bool(
                any(
                    candidate.content.get("state_fingerprint") == r.get("candidate")
                    for action in native.nodes.values()
                    if action.content.get("act_ref") == "cas://sha256/" + r["fingerprint"]
                    for node_id in action.content.get("inputs", [])
                    for candidate in (native.nodes[node_id],)
                )
            )
            for r in receipts
            if r["operation"] == "commit"
        ],
        "committed_hidden_data_ancestors": [
            {
                "commit": r["fingerprint"],
                "slots": sorted(
                    {
                        n.content.get("metadata", {}).get("slot")
                        for n in graph.lineage(values[(r["result"], "hidden")])
                        if n.content.get("metadata", {}).get("slot") is not None
                    }
                ),
                "actions": sorted(
                    {
                        label
                        for n in graph.lineage(values[(r["result"], "hidden")])
                        if n.kind == "action"
                        for label in n.labels
                    }
                ),
            }
            for r in receipts
            if r["operation"] == "commit" and (r["result"], "hidden") in values
        ],
    }
    return summary, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary, graph = inspect(Project(args.project), args.session_id)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    graph.export(args.out / "versioned-ports.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
