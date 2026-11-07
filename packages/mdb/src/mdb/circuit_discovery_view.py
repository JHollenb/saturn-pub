"""Offline adapter for the Qwen circuit-discovery report.

The discovery operation records two different kinds of evidence.  Native read
shortlists are observations of the executed receiver, while intervention rows
are responses measured after a head-local edit.  This module keeps those
provenances separate and translates the report into the typed event format
consumed by :mod:`mdb.temporal_causal_viewer`.

No model or framework is imported here.  The module only validates JSON and
renders the already-recorded report.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .temporal_causal_viewer import (
    TYPED_EVENTS_SCHEMA,
    TemporalViewerError,
    normalize_evidence,
    render_html,
)

DISCOVERY_SCHEMA = "saturn-bench-circuit-discovery-v1"
NATIVE_CONSUMER = "qwen2.native-greedy-continuation"

__all__ = [
    "DISCOVERY_SCHEMA",
    "NATIVE_CONSUMER",
    "typed_events",
    "export_html",
    "main",
]


def _copy_json(value: Any) -> Any:
    """Copy a value through JSON, rejecting non-finite report data."""

    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TemporalViewerError("circuit discovery report must contain finite JSON values") from error


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _axis(value: Any) -> int | str:
    if isinstance(value, bool):
        return "?"
    if isinstance(value, int):
        return value if value >= 0 else "?"
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        number = int(value)
        return number if number >= 0 else "?"
    return "?"


def _refs(value: Any, *, limit: int = 80) -> list[dict[str, str]]:
    """Collect report references without treating equality as a causal link."""

    found: list[dict[str, str]] = []

    def visit(node: Any, path: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                key_text = str(key).lower()
                if isinstance(child, str) and (
                    key_text.endswith(("ref", "sha256", "digest", "cut", "cut-id", "cut_id"))
                    or child.startswith(("cas://", "ar://"))
                ):
                    found.append({"path": child_path, "value": child})
                elif isinstance(child, (Mapping, list)):
                    visit(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                child_path = f"{path}[{index}]"
                if isinstance(child, str) and child.startswith(("cas://", "ar://")):
                    found.append({"path": child_path, "value": child})
                else:
                    visit(child, child_path)

    visit(value, "")
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in found:
        unique[(item["path"], item["value"])] = item
    return list(unique.values())


def _sign(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _report_payload(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the raw report and its optional operation wrapper."""

    copied = _copy_json(report)
    if not isinstance(copied, Mapping):
        raise TemporalViewerError("circuit discovery report must be an object")
    nested = copied.get("report")
    if isinstance(nested, Mapping):
        if nested.get("schema") != DISCOVERY_SCHEMA:
            raise TemporalViewerError(
                f"circuit discovery report schema must be {DISCOVERY_SCHEMA!r}"
            )
        return nested, copied
    if copied.get("schema") != DISCOVERY_SCHEMA:
        raise TemporalViewerError(f"circuit discovery report schema must be {DISCOVERY_SCHEMA!r}")
    return copied, copied


def _list_field(report: Mapping[str, Any], name: str) -> list[Any]:
    value = report.get(name, [])
    if not isinstance(value, list):
        raise TemporalViewerError(f"circuit discovery {name} must be a list")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TemporalViewerError(f"circuit discovery {label} must be an object")
    return value


def _mechanics_valid(report: Mapping[str, Any]) -> bool:
    """Return the recorded mechanics result, without making a scientific gate."""

    return (
        report.get("mechanics_status") == "valid"
        and report.get("instrument_status") == "native-consumers-executed"
        and report.get("selected_parent_restored_exact", True) is True
        and report.get("sibling_selections_preserved", True) is True
    )


def _trajectory_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    rows = value.get("trajectory")
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        # These are the native-consumer receipts written by _run.  Requiring
        # the cut and logits receipt prevents a guessed effect from becoming a
        # causal event.
        if row.get("cut_id") is None or row.get("logits_sha256") is None:
            return False
    return True


def _native_consumer_executed(test: Mapping[str, Any]) -> bool:
    explicit = test.get("native_consumer_executed")
    if explicit is False:
        return False
    act = test.get("act")
    effect = test.get("effect")
    if not isinstance(act, Mapping) or not act.get("receipt_ref"):
        return False
    if test.get("head_locality_exact") is not True:
        return False
    if not _trajectory_valid(test.get("reference")) or not _trajectory_valid(test.get("candidate")):
        return False
    edited_parent = test.get("edited_parent_cut")
    candidate_trajectory = test["candidate"]["trajectory"]
    if edited_parent is not None and candidate_trajectory[0].get("parent_cut") != edited_parent:
        return False
    if not isinstance(effect, Mapping) or not isinstance(effect.get("steps"), list) or not effect["steps"]:
        return False
    alignment = effect.get("alignment")
    return alignment in (None, "plan-offset; generated tokens remain branch-owned")


def _same_parent(test: Mapping[str, Any]) -> bool:
    """Compare the edited parent with the reference row at this receiver."""

    act = test.get("act")
    reference = test.get("reference")
    if not isinstance(act, Mapping) or not isinstance(reference, Mapping):
        return False
    trajectory = reference.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory or not isinstance(trajectory[0], Mapping):
        return False
    edited_parent = act.get("parent_cut")
    reference_parent = trajectory[0].get("parent_cut")
    return edited_parent is not None and edited_parent == reference_parent


def _candidate_branch_id(candidate_id: Any, receiver_offset: Any, index: int) -> str:
    return f"candidate:{candidate_id or f'{receiver_offset}:{index}'}"


def _test_branch_id(test_id: Any, index: int) -> str:
    return f"test:{test_id or index}"


def _effect(step: Mapping[str, Any]) -> tuple[dict[str, Any], float | None]:
    signed_value = step.get("signed_margin_effect")
    if signed_value is None:
        signed_value = step.get("signed_effect")
    signed = _finite(signed_value)
    logit_l2 = _finite(step.get("logit_l2"))
    logit_max_abs = _finite(step.get("logit_max_abs"))
    return (
        {
            "sign": _sign(signed),
            "signed_effect": signed,
            "future_signed": signed,
            "logit_l2": logit_l2,
            "logit_max_abs": logit_max_abs,
            "same_consumed_token": step.get("same_consumed_token"),
            "same_history": step.get("same_history"),
            "same_logits": step.get("same_logits"),
            "same_next_token": step.get("same_next_token"),
        },
        signed,
    )


def _candidate_event(
    candidate: Mapping[str, Any],
    receiver: Mapping[str, Any],
    branch_id: str,
    index: int,
    graph_edge: Mapping[str, Any] | None,
) -> dict[str, Any]:
    offset = _axis(receiver.get("offset"))
    layer = _axis(candidate.get("layer"))
    payload = {
        "kind": "observed-native-read-candidate",
        "receiver_offset": offset,
        "receiver_token_id": receiver.get("receiver_token_id"),
        "candidate": candidate,
        "graph_edge": graph_edge,
        "causal_dependency_tested": False,
        "native_consumer": NATIVE_CONSUMER,
    }
    return {
        "event_id": f"native-read:{candidate.get('id') or 'candidate'}:{offset}:{index}",
        "token_time": offset,
        "layer_depth": layer,
        "event_kind": "observed-native-read",
        "stage": "native-read",
        "node_role": "carrier",
        "branch_id": branch_id,
        "evidence_class": "observation",
        "causal_dependency_tested": False,
        "same_parent": False,
        "native_consumer_executed": True,
        "effect": {
            "sign": "unknown",
            "signed_effect": None,
            "future_signed": None,
            "read_score": _finite(candidate.get("read_score")),
            "group_write_l2": _finite(candidate.get("group_write_l2")),
        },
        "dose": None,
        "native_margin": None,
        "logical_address": None,
        "physical_address": {
            "layer": candidate.get("layer"),
            "kv_head": candidate.get("kv_head"),
            "source_row": candidate.get("source_row"),
            "source_token_id": candidate.get("source_token_id"),
        },
        "memory": {
            "value_sha256": candidate.get("value_sha256"),
            "key_sha256": candidate.get("key_sha256"),
        },
        "payload": payload,
        "refs": _refs(candidate),
        "raw": _copy_json(candidate),
    }


def _coverage_event(coverage: Mapping[str, Any], index: int) -> dict[str, Any]:
    offset = _axis(coverage.get("offset"))
    layer = _axis(coverage.get("layer"))
    geometry = {
        key: _finite(coverage.get(key))
        for key in (
            "attention_reconstruction_l2",
            "attention_reconstruction_max_abs",
            "attention_reconstruction_relative_l2",
        )
    }
    return {
        "event_id": f"native-coverage:{coverage.get('offset', index)}:{coverage.get('layer', '?')}:{index}",
        "token_time": offset,
        "layer_depth": layer,
        "event_kind": "native-read-coverage",
        "stage": "native-read-coverage",
        "node_role": "receiver",
        "branch_id": "baseline",
        "evidence_class": "observation",
        "same_parent": False,
        "native_consumer_executed": True,
        "effect": {"sign": "unknown", **geometry},
        "dose": None,
        "native_margin": None,
        "logical_address": None,
        "physical_address": {"layer": coverage.get("layer"), "receiver_position": coverage.get("receiver_position")},
        "memory": None,
        "payload": {"kind": "native-read-coverage", "coverage": coverage, "native_consumer": NATIVE_CONSUMER},
        "refs": _refs(coverage),
        "raw": _copy_json(coverage),
    }


def _test_events(
    test: Mapping[str, Any],
    index: int,
    branch_id: str,
    report_mechanics_valid: bool,
) -> tuple[list[dict[str, Any]], bool, bool, bool]:
    effect_record = test.get("effect")
    steps = effect_record.get("steps", []) if isinstance(effect_record, Mapping) else []
    if not isinstance(steps, list):
        raise TemporalViewerError(f"circuit discovery test {index} effect steps must be a list")
    same_parent = _same_parent(test)
    native_executed = _native_consumer_executed(test)
    events: list[dict[str, Any]] = []
    rows = steps or [{}]
    for step_index, raw_step in enumerate(rows):
        if not isinstance(raw_step, Mapping):
            raise TemporalViewerError(f"circuit discovery test {index} effect step {step_index} must be an object")
        signed_effect, signed = _effect(raw_step)
        causal = report_mechanics_valid and same_parent and native_executed and signed is not None
        candidate = test.get("candidate")
        candidate_row = candidate if isinstance(candidate, Mapping) else {}
        candidate_rows = candidate_row.get("trajectory")
        receiver_offset = test.get("receiver_offset")
        future_offset = raw_step.get("offset", receiver_offset)
        event = {
            "event_id": f"intervention-response:{test.get('id') or 'test'}:{index}:{step_index}",
            "token_time": _axis(future_offset),
            # The later readout is a future native consumer outcome, not a
            # layer-local observation.  Keep its layer unknown rather than
            # assigning the candidate's layer to the readout.
            "layer_depth": "?",
            "event_kind": "intervention-response",
            "stage": "native-consumer-readout",
            "node_role": "consumer",
            "branch_id": branch_id,
            "evidence_class": "causal_evidence" if causal else "observation",
            "same_parent": same_parent,
            "native_consumer_executed": native_executed,
            "future_offset": _axis(future_offset),
            "native_consumer": NATIVE_CONSUMER,
            "effect": signed_effect,
            "dose": {"value": _finite(test.get("scale")), "source": "circuit-discovery"},
            "native_margin": None,
            "logical_address": None,
            "physical_address": {
                "candidate_layer": candidate_row.get("layer"),
                "candidate_kv_head": candidate_row.get("kv_head"),
                "candidate_source_row": candidate_row.get("source_row"),
                "future_offset": _axis(future_offset),
            },
            "memory": {
                "candidate_value_sha256": candidate_row.get("value_sha256"),
                "candidate_cut_id": candidate_rows[-1].get("cut_id")
                if isinstance(candidate_rows, list) and candidate_rows and isinstance(candidate_rows[-1], Mapping)
                else None,
            },
            "payload": {
                "kind": "measured-intervention-response",
                "test": test,
                "step": raw_step,
                "native_consumer": NATIVE_CONSUMER,
                "recorded_mechanics_valid": report_mechanics_valid,
                "causal_dependency_tested": causal,
            },
            "refs": _refs({"test": test, "step": raw_step}),
            "raw": {"test": _copy_json(test), "step": _copy_json(raw_step)},
        }
        events.append(event)
    return events, same_parent, native_executed, any(event["evidence_class"] == "causal_evidence" for event in events)


def _graph_event(
    edge: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    index: int,
) -> dict[str, Any]:
    kind = str(edge.get("kind") or "graph-edge")
    source = str(edge.get("source") or "")
    target = str(edge.get("target") or "")
    source_node = nodes.get(source, {})
    target_node = nodes.get(target, {})
    source_offset = source_node.get("offset")
    target_offset = target_node.get("offset")
    if source_offset is None and source.startswith("receiver:"):
        source_offset = source.split(":", 1)[1]
    if target_offset is None and target.startswith("receiver:"):
        target_offset = target.split(":", 1)[1]
    if kind == "same-payload-observation":
        token_time = target_offset if target_offset is not None else source_offset
        layer = target_node.get("layer", "?")
        branch_id = f"candidate:{target}" if target else "baseline"
    elif kind == "native-execution-order":
        token_time = source_offset
        layer = "?"
        branch_id = "baseline"
    else:
        token_time = source_offset if source_offset is not None else target_offset
        layer = target_node.get("layer", "?")
        branch_id = "baseline"
    return {
        "event_id": f"graph:{kind}:{index}",
        "token_time": _axis(token_time),
        "layer_depth": _axis(layer),
        "event_kind": kind,
        "stage": "graph-observation",
        "node_role": "relation",
        "branch_id": branch_id,
        "evidence_class": "observation",
        "causal_dependency_tested": False,
        "same_payload_observation": kind == "same-payload-observation",
        "same_parent": False,
        "native_consumer_executed": False,
        "future_offset": _axis(target_offset) if target_offset is not None else None,
        "native_consumer": NATIVE_CONSUMER,
        "effect": None,
        "dose": None,
        "native_margin": None,
        "logical_address": None,
        "physical_address": {"source": source, "target": target},
        "memory": None,
        "payload": {
            "kind": kind,
            "edge": edge,
            "source_node": source_node,
            "target_node": target_node,
            "causal_dependency_tested": edge.get("causal_dependency_tested") is True
            and kind not in {"same-payload-observation", "native-execution-order"},
        },
        "refs": _refs(edge),
        "raw": _copy_json(edge),
    }


def typed_events(report: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a raw discovery report to typed viewer events.

    Only a same-parent intervention with a recorded native continuation and a
    finite signed margin effect receives ``causal_evidence``.  Read rankings,
    equal payloads, and native execution order remain observations even when
    their numeric geometry is available.
    """

    raw_report, wrapper = _report_payload(report)
    receivers = _list_field(raw_report, "receivers")
    tests = _list_field(raw_report, "tests")
    graph = raw_report.get("graph", {})
    if graph is not None and not isinstance(graph, Mapping):
        raise TemporalViewerError("circuit discovery graph must be an object")
    report_mechanics_valid = _mechanics_valid(raw_report)

    graph_edges = graph.get("edges", []) if isinstance(graph, Mapping) else []
    if not isinstance(graph_edges, list):
        raise TemporalViewerError("circuit discovery graph edges must be a list")
    graph_nodes = graph.get("nodes", []) if isinstance(graph, Mapping) else []
    if not isinstance(graph_nodes, list):
        raise TemporalViewerError("circuit discovery graph nodes must be a list")
    nodes: dict[str, Mapping[str, Any]] = {}
    for node_index, node in enumerate(graph_nodes):
        if not isinstance(node, Mapping):
            raise TemporalViewerError(f"circuit discovery graph node {node_index} must be an object")
        if node.get("id") is not None:
            nodes[str(node["id"])] = node
    edges_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for edge_index, edge in enumerate(graph_edges):
        if not isinstance(edge, Mapping):
            raise TemporalViewerError(f"circuit discovery graph edge {edge_index} must be an object")
        source = str(edge.get("source") or "")
        edges_by_source.setdefault(source, []).append(edge)

    branches: list[dict[str, Any]] = [
        {
            "branch_id": "baseline",
            "role": "native",
            "arm_kind": "native",
            "evidence_class": "observation",
            "parent_cut": raw_report.get("parent_cut"),
            "same_parent": True,
            "native_consumer": NATIVE_CONSUMER,
            "instrument_status": raw_report.get("instrument_status"),
            "source": DISCOVERY_SCHEMA,
        }
    ]
    events: list[dict[str, Any]] = []
    candidate_branches: dict[str, dict[str, Any]] = {}

    for receiver_index, raw_receiver in enumerate(receivers):
        receiver = _mapping(raw_receiver, f"receiver {receiver_index}")
        candidates = receiver.get("candidates", [])
        if not isinstance(candidates, list):
            raise TemporalViewerError(f"circuit discovery receiver {receiver_index} candidates must be a list")
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate = _mapping(raw_candidate, f"receiver {receiver_index} candidate {candidate_index}")
            candidate_id = _candidate_branch_id(candidate.get("id"), receiver.get("offset"), candidate_index)
            if candidate_id not in candidate_branches:
                candidate_branches[candidate_id] = {
                    "branch_id": candidate_id,
                    "role": "carrier",
                    "arm_kind": "observed-native-read",
                    "evidence_class": "observation",
                    "parent_cut": candidate.get("parent_cut", receiver.get("parent_cut")),
                    "same_parent": False,
                    "native_consumer": NATIVE_CONSUMER,
                    "candidate_id": candidate.get("id"),
                    "candidate_layer": candidate.get("layer"),
                    "source": DISCOVERY_SCHEMA,
                }
                branches.append(candidate_branches[candidate_id])
            edge = next(
                (
                    item
                    for item in edges_by_source.get(str(candidate.get("id") or ""), [])
                    if item.get("kind") == "observed-native-read"
                ),
                None,
            )
            events.append(_candidate_event(candidate, receiver, candidate_id, candidate_index, edge))
        coverage = receiver.get("coverage", [])
        if not isinstance(coverage, list):
            raise TemporalViewerError(f"circuit discovery receiver {receiver_index} coverage must be a list")
        for coverage_index, raw_coverage in enumerate(coverage):
            coverage_row = _mapping(raw_coverage, f"receiver {receiver_index} coverage {coverage_index}")
            events.append(_coverage_event(coverage_row, coverage_index))

    for test_index, raw_test in enumerate(tests):
        test = _mapping(raw_test, f"test {test_index}")
        branch_id = _test_branch_id(test.get("id"), test_index)
        test_events, same_parent, native_executed, has_causal = _test_events(
            test, test_index, branch_id, report_mechanics_valid
        )
        branches.append(
            {
                "branch_id": branch_id,
                "role": "intervention",
                "arm_kind": "intervention-response",
                "evidence_class": "causal_evidence" if has_causal else "observation",
                "parent_cut": test.get("act", {}).get("parent_cut")
                if isinstance(test.get("act"), Mapping)
                else None,
                "same_parent": same_parent,
                "native_consumer_executed": native_executed,
                "native_consumer": NATIVE_CONSUMER,
                "dose": {"value": _finite(test.get("scale")), "source": "circuit-discovery"},
                "test_id": test.get("id"),
                "source": DISCOVERY_SCHEMA,
            }
        )
        events.extend(test_events)

    # Keep graph relations that are not represented by a candidate or test
    # event visible.  In particular, equality of a payload across receivers and
    # native execution order are explicit observations, never causal edges.
    represented_kinds = {"observed-native-read", "measured-intervention-response"}
    for edge_index, raw_edge in enumerate(graph_edges):
        kind = raw_edge.get("kind")
        if kind in represented_kinds:
            continue
        events.append(_graph_event(raw_edge, nodes, edge_index))

    binding = {
        "model_identity": raw_report.get("model_identity") or {},
        "input_binding": {
            "report_schema": DISCOVERY_SCHEMA,
            "report_name": raw_report.get("name"),
            "session_id": raw_report.get("session_id"),
            "spec_sha256": raw_report.get("spec_sha256"),
            "spec": raw_report.get("spec"),
            "input_authority": raw_report.get("input_authority"),
            "native_consumer": NATIVE_CONSUMER,
            "mechanics_status": raw_report.get("mechanics_status"),
            "instrument_status": raw_report.get("instrument_status"),
            "selected_parent_restored_exact": raw_report.get("selected_parent_restored_exact"),
            "sibling_selections_preserved": raw_report.get("sibling_selections_preserved"),
            "operation": wrapper.get("op") if wrapper is not raw_report else None,
            "report_ref": wrapper.get("report_ref") if wrapper is not raw_report else None,
        },
        "numerical_contract": raw_report.get("numerical_contract"),
        "parent_cut": raw_report.get("parent_cut"),
        "source_schema": TYPED_EVENTS_SCHEMA,
    }
    return _copy_json(
        {
            "schema": TYPED_EVENTS_SCHEMA,
            "binding": binding,
            "branches": branches,
            "events": events,
        }
    )


def export_html(report_path: str | Path, output: str | Path, *, title: str = "Circuit discovery viewer") -> dict[str, Any]:
    """Render one raw discovery report into the existing offline viewer."""

    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemporalViewerError(f"cannot read circuit discovery report {path}: {error}") from error
    payload = typed_events(report)
    bundle = normalize_evidence([payload], sources=[str(path)])
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(bundle, title=title), encoding="utf-8")
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a Qwen circuit-discovery report as offline HTML")
    parser.add_argument("report", help="saturn-bench-circuit-discovery-v1 JSON report")
    parser.add_argument("--output", required=True, help="standalone HTML destination")
    parser.add_argument("--title", default="Circuit discovery viewer")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle = export_html(args.report, args.output, title=args.title)
    except TemporalViewerError as error:
        raise SystemExit(f"circuit discovery view: {error}") from error
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "content_sha256": bundle["content_sha256"],
                "events": len(bundle["events"]),
                "branches": len(bundle["branches"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
