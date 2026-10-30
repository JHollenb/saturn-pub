"""Read-only longitudinal traces for one supplied symbol name.

``SymbolBinding`` is intentionally exact: its execution point, schema, and
context fingerprints are part of the identity.  A symbol can therefore have
several bindings over time without those bindings being silently treated as
the same native location.  This module joins separately sealed lifecycle
snapshots by their supplied symbol name and keeps the binding fingerprints
visible in every row.

The trace is a control-plane index.  It reads retained descriptors, receipts,
and saved declared slices; it never hydrates model payloads or grants write
authority.  A branch label is recorded as supplied evidence.  Native cut
parent ancestry is the only history support reported here, and no field is a
semantic or causal certificate.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from saturn_pub.values import digest

SYMBOL_TRACE_SCHEMA = "mdb-symbol-trace-v1"
TRACE_GRAPH_SCHEMA = "mdb-symbol-trace-graph-v1"


class SymbolTraceError(ValueError):
    """Raised when a longitudinal trace cannot be represented safely."""


def _json_copy(value: Any) -> Any:
    """Return a canonical JSON copy without importing a model framework."""

    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SymbolTraceError("trace metadata is not canonical JSON") from exc


def _identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return None
    return value.removeprefix("cas://sha256/")


def _event_cuts(event: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for key in ("source_cut", "parent_cut", "result_cut"):
        value = event.get(key)
        if value and value not in result:
            result.append(value)
    return tuple(result)


def _event_anchor(event: Mapping[str, Any]) -> str | None:
    for key in ("source_cut", "parent_cut", "result_cut"):
        value = event.get(key)
        if value:
            return value
    return None


def _point_key(point: Mapping[str, Any] | None, cut_id: str) -> tuple[Any, ...]:
    point = point or {}

    def number(name: str) -> int:
        value = point.get(name)
        return value if type(value) is int else 2**62

    return (
        number("logical_step"),
        number("index"),
        str(point.get("phase", "")),
        str(point.get("operator", "")),
        cut_id,
    )


def _frontier_key(row: Mapping[str, Any]) -> str:
    return digest(row)


def _add_frontier(frontier: list[dict[str, Any]], row: Mapping[str, Any]) -> None:
    body = _json_copy(row)
    if not any(_frontier_key(item) == _frontier_key(body) for item in frontier):
        frontier.append(body)


def _normalise_cot(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retain COT references as locators, never as semantic evidence."""

    result: list[dict[str, Any]] = []
    source_cut = _identifier(event.get("source_cut"))
    branch = event.get("branch")
    for raw in event.get("cot_refs") or ():
        if isinstance(raw, Mapping):
            raw_payload = _json_copy(raw)
        else:
            raw_payload = {"reference": _json_copy(raw)}
        cot_cut = _identifier(raw_payload.get("source_cut"))
        cot_branch = raw_payload.get("branch")
        if cot_cut is None:
            locator_status = "unknown"
            locator_reason = "cot source cut is absent"
        elif source_cut is None or cot_cut != source_cut:
            locator_status = "unknown"
            locator_reason = "cot source cut differs from occurrence"
        elif cot_branch is not None and branch is not None and cot_branch != branch:
            locator_status = "unknown"
            locator_reason = "cot branch label differs from occurrence"
        else:
            locator_status = "locator-matched"
            locator_reason = "source cut association only"
        # Keep the imported register payload intact under an explicit raw
        # field, while reserving the outer fields for this trace's authority
        # overlay.  A producer's true semantic/causal flags are evidence of
        # what it supplied, never authority granted by this locator join.
        reserved = {
            "authority",
            "causal_dependency_tested",
            "causal_effect",
            "causal_status",
            "locator_reason",
            "locator_status",
            "semantic_cot_qualified",
            "semantic_status",
        }
        row = {
            key: value for key, value in raw_payload.items() if key not in reserved
        }
        row.update(
            {
                "raw_payload": raw_payload,
                "locator_status": locator_status,
                "locator_reason": locator_reason,
                "authority": "locator-only",
                "semantic_cot_qualified": False,
                "causal_dependency_tested": False,
                "causal_effect": "not-established",
                "semantic_status": (
                    "supplied-unverified"
                    if raw_payload.get("semantic_cot_qualified") is True
                    else "unqualified"
                ),
                "causal_status": "not-established",
            }
        )
        result.append(row)
    return result


def _receipt_summary(
    project: Any,
    session_id: str,
    event: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    model_identity: Any,
    execution_digest: str,
    frontier: list[dict[str, Any]],
) -> dict[str, Any] | None:
    receipt_id = _identifier(event.get("receipt"))
    if receipt_id is None:
        return None
    body = receipts.get(receipt_id)
    if body is None:
        _add_frontier(
            frontier,
            {
                "kind": "unretained-receipt",
                "receipt": receipt_id,
                "event": event.get("fingerprint"),
            },
        )
        return {
            "receipt": receipt_id,
            "support": "unknown",
            "authority": "read-only-evidence-index",
        }
    if body.get("model_identity") != model_identity:
        raise SymbolTraceError("native receipt model identity differs from the project session")
    if digest(body.get("execution")) != execution_digest:
        raise SymbolTraceError("native receipt execution differs from the project session")
    return {
        "receipt": receipt_id,
        "operation": body.get("operation"),
        "parent_cut": body.get("parent"),
        "result_cut": body.get("result"),
        "candidate_cut": body.get("candidate"),
        "status": body.get("status", "completed"),
        "execution_point": body.get("execution_point"),
        "support": "native-identity-verified",
        "authority": "native-receipt-metadata",
    }


def _event_history(
    event: Mapping[str, Any],
    *,
    selected_cut: str | None,
    selected_ancestry: set[str],
    retained: set[str],
    cut_cache: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    cuts = _event_cuts(event)
    unknown = [cut for cut in cuts if cut not in retained]
    if unknown:
        support = "unretained-cut"
    elif not cuts:
        support = "unknown"
    elif selected_cut is not None and not set(cuts).issubset(selected_ancestry):
        support = "outside-selected-ancestry"
    else:
        source = event.get("source_cut") or event.get("parent_cut")
        result = event.get("result_cut")
        if source and result and source != result:
            current = result
            seen: set[str] = set()
            linked = False
            while current and current not in seen:
                if current == source:
                    linked = True
                    break
                seen.add(current)
                descriptor = cut_cache.get(current)
                current = descriptor.get("parent") if descriptor else None
            support = "native-cut-ancestry" if linked else "disconnected-cuts"
        else:
            support = "native-cut-ancestry"
    return {
        "anchor": _event_anchor(event),
        "cuts": list(cuts),
        "selected_cut": selected_cut,
        "support": support,
        "branch_label": event.get("branch"),
        "branch_support": "recorded-label" if event.get("branch") else "absent",
        "unknown_cuts": unknown,
    }


def _slice_join(
    project: Any,
    session_id: str,
    event: Mapping[str, Any],
    binding_fingerprint: str,
    join_row: Mapping[str, Any] | None,
    frontier: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not join_row or not join_row.get("slice_id"):
        return None
    slice_id = join_row["slice_id"]
    try:
        sliced = project.slice_record(session_id, slice_id)
    except Exception as exc:  # retain an explicit frontier, never fabricate a join
        _add_frontier(
            frontier,
            {
                "kind": "unavailable-slice",
                "slice_id": slice_id,
                "event": event.get("fingerprint"),
                "reason": str(exc),
            },
        )
        return {
            "slice_id": slice_id,
            "status": "unknown",
            "support": "unavailable",
            "authority": "read-only-evidence-index",
            "causal_effect": "not-established",
        }
    symbol_binding = sliced.get("symbol_binding") or {}
    if (
        sliced.get("cut_id") != event.get("source_cut")
        or symbol_binding.get("fingerprint") != binding_fingerprint
    ):
        _add_frontier(
            frontier,
            {
                "kind": "slice-binding-mismatch",
                "slice_id": slice_id,
                "event": event.get("fingerprint"),
            },
        )
        return {
            "slice_id": slice_id,
            "status": "unknown",
            "support": "mismatched",
            "authority": "read-only-evidence-index",
            "causal_effect": "not-established",
        }
    return {
        "slice_id": slice_id,
        "cut_id": sliced.get("cut_id"),
        "graph_snapshot": sliced.get("graph", {}).get("snapshot_id"),
        "support": sliced.get("support"),
        "complete": sliced.get("complete"),
        "scope": sliced.get("scope"),
        "authority": sliced.get("authority", "declared-dependency"),
        "causal_effect": sliced.get("causal_effect", "not-established"),
        "status": "joined",
    }


def _merge_invalidations(
    destination: list[dict[str, Any]], lifecycle_id: str, invalidations: Sequence[Any]
) -> None:
    for raw in invalidations:
        if not isinstance(raw, Mapping):
            continue
        body = _json_copy(raw)
        fingerprint = digest(body)
        existing = next(
            (row for row in destination if row.get("invalidation_fingerprint") == fingerprint),
            None,
        )
        if existing is None:
            destination.append(
                {
                    "invalidation_fingerprint": fingerprint,
                    "lifecycle_ids": [lifecycle_id],
                    **body,
                }
            )
        elif lifecycle_id not in existing["lifecycle_ids"]:
            existing["lifecycle_ids"].append(lifecycle_id)
            existing["lifecycle_ids"].sort()


def _merge_event(
    buckets: dict[str, dict[str, Any]],
    event: Mapping[str, Any],
    *,
    kind: str,
    lifecycle_id: str,
    binding_fingerprint: str,
    lifecycle_invalidations: Sequence[Any],
    join_row: Mapping[str, Any] | None,
    project: Any,
    session_id: str,
    selected_cut: str | None,
    selected_ancestry: set[str],
    retained: set[str],
    cut_cache: Mapping[str, Mapping[str, Any] | None],
    frontier: list[dict[str, Any]],
) -> None:
    fingerprint = event.get("fingerprint") or digest(event)
    row = buckets.get(fingerprint)
    if row is None:
        row = _json_copy(event)
        row["event_fingerprint"] = fingerprint
        row["kind"] = kind
        row["binding_fingerprint"] = binding_fingerprint
        row["lifecycle_ids"] = []
        row["snapshot_count"] = 0
        row["history"] = _event_history(
            event,
            selected_cut=selected_cut,
            selected_ancestry=selected_ancestry,
            retained=retained,
            cut_cache=cut_cache,
        )
        row["cot_refs"] = _normalise_cot(event) if kind == "occurrence" else []
        row["slice_joins"] = []
        row["slice_ids"] = []
        row["invalidations"] = []
        buckets[fingerprint] = row
    if lifecycle_id not in row["lifecycle_ids"]:
        row["lifecycle_ids"].append(lifecycle_id)
        row["lifecycle_ids"].sort()
    row["snapshot_count"] += 1
    for unknown_cut in row["history"].get("unknown_cuts", ()):
        _add_frontier(
            frontier,
            {
                "kind": "unretained-cut",
                "cut_id": unknown_cut,
                "event": fingerprint,
            },
        )
    _merge_invalidations(
        row["invalidations"],
        lifecycle_id,
        [item for item in lifecycle_invalidations if item.get("scope") in {
            f"{kind}:{event.get('occurrence_id') or event.get('use_id')}"
        }],
    )
    if kind == "occurrence":
        joined = _slice_join(
            project,
            session_id,
            event,
            binding_fingerprint,
            join_row,
            frontier,
        )
        if joined and joined.get("slice_id") not in row["slice_ids"]:
            row["slice_ids"].append(joined["slice_id"])
            row["slice_joins"].append(joined)


def _lineage(
    project: Any,
    session_id: str,
    cut_id: str,
    retained: set[str],
    cache: dict[str, Mapping[str, Any] | None],
    frontier: list[dict[str, Any]],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    current: str | None = cut_id
    while current:
        if current in seen:
            _add_frontier(frontier, {"kind": "cut-cycle", "cut_id": current})
            break
        seen.add(current)
        result.append(current)
        if current not in retained:
            _add_frontier(
                frontier,
                {"kind": "unretained-cut", "cut_id": current, "reason": "parent ancestry"},
            )
            break
        if current not in cache:
            cache[current] = project.cut_descriptor(session_id, current)
        descriptor = cache[current]
        current = descriptor.get("parent") if descriptor else None
    return result


def _graph(
    *,
    symbol: str,
    binding_groups: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    uses: Sequence[Mapping[str, Any]],
    cuts: Sequence[Mapping[str, Any]],
    frontier: Sequence[Mapping[str, Any]],
    slice_joins: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def node(node_id: str, kind: str, **metadata: Any) -> None:
        nodes.setdefault(node_id, {"id": node_id, "kind": kind, **metadata})

    def edge(source: str, target: str, relation: str, **metadata: Any) -> None:
        body = {"source": source, "target": target, "relation": relation, **metadata}
        edges.setdefault(digest(body), {"edge_id": digest(body), **body})

    symbol_id = f"symbol://sha256/{digest(symbol)}"
    node(
        symbol_id,
        "supplied-symbol",
        symbol=symbol,
        semantic_identity="not-established",
        authority="supplied-label",
    )
    for group in binding_groups:
        fp = group["binding_fingerprint"]
        binding_id = f"binding://sha256/{fp}"
        node(binding_id, "binding", binding_fingerprint=fp)
        edge(
            symbol_id,
            binding_id,
            "supplied-name-association",
            evidence="supplied-label",
            semantic_identity="not-established",
            causal_effect="not-established",
        )

    retained_cut_ids = {row.get("cut_id") for row in cuts}
    for cut in cuts:
        identifier = cut.get("cut_id")
        if not identifier:
            continue
        node(
            f"cut://sha256/{identifier}",
            "cut",
            cut_id=identifier,
            retained=True,
            selected=bool(cut.get("selected")),
        )
    # The native parent descriptor is an observed chronology edge.  A parent
    # outside retained cuts remains a frontier node, rather than being made to
    # look like a fully retained native cut.
    for cut in cuts:
        child = cut.get("cut_id")
        parent = cut.get("parent_cut")
        if not child or not parent:
            continue
        child_id = f"cut://sha256/{child}"
        parent_id = f"cut://sha256/{parent}"
        if parent not in retained_cut_ids:
            node(
                parent_id,
                "frontier-cut",
                cut_id=parent,
                retained=False,
                support="unretained-parent-frontier",
            )
        else:
            node(parent_id, "cut", cut_id=parent, retained=True)
        edge(
            parent_id,
            child_id,
            "native-parent",
            evidence="observed-cut-descriptor",
            support=(
                "native-cut-parent-ancestry"
                if parent in retained_cut_ids
                else "unretained-parent-frontier"
            ),
            causal_effect="not-established",
        )
    # Preserve non-cut frontiers (receipt/slice/cycle diagnostics) in the
    # graph as evidence boundaries too.  The detailed rows remain top-level.
    for item in frontier:
        frontier_id = f"frontier://sha256/{digest(item)}"
        node(frontier_id, "frontier", frontier=_json_copy(item))

    for event in occurrences:
        event_id = event["event_fingerprint"]
        binding_id = f"binding://sha256/{event['binding_fingerprint']}"
        node(f"occurrence://sha256/{event_id}", "occurrence", event_fingerprint=event_id)
        edge(binding_id, f"occurrence://sha256/{event_id}", "defines", evidence="declared")
        for cut in event.get("history", {}).get("cuts", ()):
            cut_id = f"cut://sha256/{cut}"
            node(cut_id, "cut", cut_id=cut)
            edge(f"occurrence://sha256/{event_id}", cut_id, "anchored_at", evidence="observed")
        for joined in event.get("slice_joins", ()):
            if joined.get("slice_id"):
                slice_node = f"slice://sha256/{joined['slice_id']}"
                node(slice_node, "slice", slice_id=joined["slice_id"])
                edge(
                    f"occurrence://sha256/{event_id}",
                    slice_node,
                    "joined_slice",
                    evidence=joined.get("support", "unknown"),
                    causal_effect="not-established",
                )
    for event in uses:
        event_id = event["event_fingerprint"]
        binding_id = f"binding://sha256/{event['binding_fingerprint']}"
        node(f"use://sha256/{event_id}", "use", event_fingerprint=event_id)
        edge(binding_id, f"use://sha256/{event_id}", "used_by", evidence="recorded")
        for cut in event.get("history", {}).get("cuts", ()):
            cut_id = f"cut://sha256/{cut}"
            node(cut_id, "cut", cut_id=cut)
            edge(f"use://sha256/{event_id}", cut_id, "consumer_anchor", evidence="observed")
    for link in links:
        if link.get("definition_id") and link.get("status") == "linked":
            use_fingerprint = link.get("use_fingerprint")
            if not use_fingerprint:
                continue
            edge(
                f"use://sha256/{use_fingerprint}",
                f"occurrence://sha256/{link['definition_fingerprint']}",
                "same_source_binding",
                evidence="exact-source-cut-and-binding",
                causal_effect="not-established",
            )
    return {
        "schema": TRACE_GRAPH_SCHEMA,
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
        "authority": "read-only-evidence-index",
        "causal_effect": "not-established",
    }


def trace_symbol(
    project: Any,
    session_id: str,
    symbol: str,
    *,
    branch: str | None = None,
    cut_id: str | None = None,
) -> dict[str, Any]:
    """Return a sealed, read-only trace for all retained lifecycle snapshots."""

    if not isinstance(symbol, str) or not symbol.strip():
        raise SymbolTraceError("symbol must be non-empty text")
    project.session(session_id)
    retained = set(project.cut_ids(session_id))
    if cut_id is not None and cut_id not in retained:
        raise SymbolTraceError("trace cut is not retained in this project session")

    frontier: list[dict[str, Any]] = []
    cut_cache: dict[str, Mapping[str, Any] | None] = {}
    selected_ancestry = set()
    if cut_id is not None:
        selected_ancestry = set(
            _lineage(project, session_id, cut_id, retained, cut_cache, frontier)
        )
    else:
        for identifier in sorted(retained):
            if identifier not in cut_cache:
                cut_cache[identifier] = project.cut_descriptor(session_id, identifier)
            parent = cut_cache[identifier].get("parent") if cut_cache[identifier] else None
            if parent and parent not in retained:
                _add_frontier(
                    frontier,
                    {"kind": "unretained-cut", "cut_id": parent, "child": identifier},
                )

    session = project.session(session_id)
    model_identity = json.loads(session["identity"])
    execution = json.loads(session["execution"])
    execution_digest = digest(execution)
    native_receipts = {
        receipt.fingerprint: receipt.to_dict() for receipt in project.receipts(session_id)
    }

    records = project.symbol_lifecycles(session_id, symbol=symbol)
    snapshots: list[dict[str, Any]] = []
    invalidations: list[dict[str, Any]] = []
    evidence_refs: set[str] = set()
    groups: dict[str, dict[str, Any]] = {}
    occurrence_buckets: dict[str, dict[str, Any]] = {}
    use_buckets: dict[str, dict[str, Any]] = {}
    filtered_events: list[dict[str, Any]] = []
    receipt_rows: dict[str, dict[str, Any]] = {}

    for record in records:
        lifecycle = record["lifecycle"]
        lifecycle_id = record["lifecycle_id"]
        binding = lifecycle["binding"]
        binding_fingerprint = binding["fingerprint"]
        group = groups.setdefault(
            binding_fingerprint,
            {
                "binding_fingerprint": binding_fingerprint,
                "binding": _json_copy(binding),
                "lifecycle_ids": [],
                "occurrence_fingerprints": [],
                "use_fingerprints": [],
            },
        )
        if lifecycle_id not in group["lifecycle_ids"]:
            group["lifecycle_ids"].append(lifecycle_id)
        snapshot = {
            "lifecycle_id": lifecycle_id,
            "binding_fingerprint": binding_fingerprint,
            "statuses": _json_copy(lifecycle.get("statuses", {})),
            "definition_fingerprints": [
                item.get("fingerprint") for item in lifecycle.get("definitions", ())
            ],
            "use_fingerprints": [
                item.get("fingerprint") for item in lifecycle.get("uses", ())
            ],
            "invalidations": _json_copy(lifecycle.get("invalidations", ())),
            "claim_boundary": lifecycle.get("claim_boundary"),
        }
        snapshots.append(snapshot)
        _merge_invalidations(invalidations, lifecycle_id, lifecycle.get("invalidations", ()))
        for invalidation in lifecycle.get("invalidations", ()):
            if isinstance(invalidation, Mapping):
                evidence_refs.update(
                    item
                    for item in invalidation.get("evidence_refs", ())
                    if isinstance(item, str)
                )

        binding_evidence = binding.get("evidence") or ()
        evidence_refs.update(item for item in binding_evidence if isinstance(item, str))
        join_definitions = {
            item.get("occurrence"): item for item in record.get("joins", {}).get("definitions", ())
        }
        for event in lifecycle.get("definitions", ()):
            event_branch = event.get("branch")
            if branch is not None and event_branch != branch:
                filtered_events.append(
                    {"kind": "occurrence", "event_fingerprint": event.get("fingerprint"), "reason": "branch-filter"}
                )
                continue
            event_cuts = _event_cuts(event)
            if cut_id is not None and (
                not event_cuts or not set(event_cuts).issubset(selected_ancestry)
            ):
                filtered_events.append(
                    {
                        "kind": "occurrence",
                        "event_fingerprint": event.get("fingerprint"),
                        "reason": "outside-selected-ancestry",
                    }
                )
                continue
            _merge_event(
                occurrence_buckets,
                event,
                kind="occurrence",
                lifecycle_id=lifecycle_id,
                binding_fingerprint=binding_fingerprint,
                lifecycle_invalidations=lifecycle.get("invalidations", ()),
                join_row=join_definitions.get(event.get("occurrence_id")),
                project=project,
                session_id=session_id,
                selected_cut=cut_id,
                selected_ancestry=selected_ancestry,
                retained=retained,
                cut_cache=cut_cache,
                frontier=frontier,
            )
            group["occurrence_fingerprints"].append(event.get("fingerprint"))
            evidence_refs.update(item for item in event.get("evidence_refs", ()) if isinstance(item, str))
            evidence_refs.update(item for item in event.get("definition_refs", ()) if isinstance(item, str))
            receipt = _receipt_summary(
                project,
                session_id,
                event,
                native_receipts,
                model_identity,
                execution_digest,
                frontier,
            )
            if receipt:
                receipt_rows[receipt["receipt"]] = receipt
                evidence_refs.add("cas://sha256/" + receipt["receipt"])

        join_uses = {
            item.get("use"): item for item in record.get("joins", {}).get("uses", ())
        }
        for event in lifecycle.get("uses", ()):
            event_branch = event.get("branch")
            if branch is not None and event_branch != branch:
                filtered_events.append(
                    {"kind": "use", "event_fingerprint": event.get("fingerprint"), "reason": "branch-filter"}
                )
                continue
            event_cuts = _event_cuts(event)
            if cut_id is not None and (
                not event_cuts or not set(event_cuts).issubset(selected_ancestry)
            ):
                filtered_events.append(
                    {
                        "kind": "use",
                        "event_fingerprint": event.get("fingerprint"),
                        "reason": "outside-selected-ancestry",
                    }
                )
                continue
            _merge_event(
                use_buckets,
                event,
                kind="use",
                lifecycle_id=lifecycle_id,
                binding_fingerprint=binding_fingerprint,
                lifecycle_invalidations=lifecycle.get("invalidations", ()),
                join_row=join_uses.get(event.get("use_id")),
                project=project,
                session_id=session_id,
                selected_cut=cut_id,
                selected_ancestry=selected_ancestry,
                retained=retained,
                cut_cache=cut_cache,
                frontier=frontier,
            )
            group["use_fingerprints"].append(event.get("fingerprint"))
            evidence_refs.update(item for item in event.get("evidence_refs", ()) if isinstance(item, str))
            evidence_refs.update(item for item in event.get("signed_observations", ()) if isinstance(item, str))
            receipt = _receipt_summary(
                project,
                session_id,
                event,
                native_receipts,
                model_identity,
                execution_digest,
                frontier,
            )
            if receipt:
                receipt_rows[receipt["receipt"]] = receipt
                evidence_refs.add("cas://sha256/" + receipt["receipt"])

    occurrences = sorted(occurrence_buckets.values(), key=lambda row: row["event_fingerprint"])
    uses = sorted(use_buckets.values(), key=lambda row: row["event_fingerprint"])
    for group in groups.values():
        group["lifecycle_ids"].sort()
        group["occurrence_fingerprints"] = sorted(set(group["occurrence_fingerprints"]))
        group["use_fingerprints"] = sorted(set(group["use_fingerprints"]))
    binding_groups = [groups[key] for key in sorted(groups)]

    occurrence_by_key = {
        (row["source_cut"], row["binding_fingerprint"]): [] for row in occurrences
    }
    for row in occurrences:
        occurrence_by_key.setdefault(
            (row.get("source_cut"), row["binding_fingerprint"]), []
        ).append(row)
    links: list[dict[str, Any]] = []
    for use in uses:
        candidates = occurrence_by_key.get(
            (use.get("source_cut"), use["binding_fingerprint"]), []
        )
        if len(candidates) == 1:
            links.append(
                {
                    "use_id": use["use_id"],
                    "use_fingerprint": use["event_fingerprint"],
                    "definition_id": candidates[0]["occurrence_id"],
                    "definition_fingerprint": candidates[0]["event_fingerprint"],
                    "status": "linked",
                    "reason": "exact-source-cut-and-binding",
                    "causal_effect": "not-established",
                }
            )
        elif len(candidates) > 1:
            links.append(
                {
                    "use_id": use["use_id"],
                    "use_fingerprint": use["event_fingerprint"],
                    "definition_ids": sorted(row["occurrence_id"] for row in candidates),
                    "status": "unknown",
                    "reason": "ambiguous-exact-source-binding",
                    "causal_effect": "not-established",
                }
            )
        else:
            links.append(
                {
                    "use_id": use["use_id"],
                    "use_fingerprint": use["event_fingerprint"],
                    "status": "unknown",
                    "reason": "no-exact-source-cut-and-binding-definition",
                    "causal_effect": "not-established",
                }
            )
    link_by_use = {row["use_fingerprint"]: row for row in links}
    for use in uses:
        use["definition_link"] = link_by_use[use["event_fingerprint"]]

    cut_rows: list[dict[str, Any]] = []
    # Chronology is a retained native DAG view even when event rows are
    # filtered to one ancestry.  Hydrate descriptors only (never state pages)
    # so the graph keeps sibling cuts and their observed parent edges visible.
    for identifier in retained:
        if identifier not in cut_cache:
            cut_cache[identifier] = project.cut_descriptor(session_id, identifier)
    for identifier in sorted(retained, key=lambda cid: _point_key((cut_cache.get(cid) or {}).get("execution_point"), cid)):
        descriptor = cut_cache.get(identifier)
        if descriptor is None:
            continue
        parent = descriptor.get("parent")
        if parent and parent not in retained:
            _add_frontier(
                frontier,
                {"kind": "unretained-cut", "cut_id": parent, "child": identifier},
            )
        surface = descriptor.get("surface") or {}
        cut_rows.append(
            {
                "cut_id": identifier,
                "parent_cut": descriptor.get("parent"),
                "boundary": descriptor.get("boundary"),
                "execution_point": descriptor.get("execution_point"),
                "state_schema": surface.get("state_schema"),
                "consumer": surface.get("consumer"),
                "selected": identifier == cut_id,
                "in_selected_ancestry": identifier in selected_ancestry if cut_id else True,
            }
        )

    slice_joins = sorted(
        {
            joined["slice_id"]: joined
            for event in occurrences
            for joined in event.get("slice_joins", ())
            if joined.get("slice_id")
        }.values(),
        key=lambda row: row["slice_id"],
    )
    cot_refs = [
        {
            "event_fingerprint": event["event_fingerprint"],
            "occurrence_id": event["occurrence_id"],
            **cot,
        }
        for event in occurrences
        for cot in event.get("cot_refs", ())
    ]
    signed_observations = [
        {
            "event_fingerprint": event["event_fingerprint"],
            "use_id": event["use_id"],
            "binding_fingerprint": event["binding_fingerprint"],
            "source_cut": event.get("source_cut"),
            "consumer": event.get("consumer"),
            "observation": observation,
            "authority": "signed-observation",
            "causal_effect": "not-established",
        }
        for event in uses
        for observation in event.get("signed_observations", ())
    ]

    chronology = {
        "selected_cut": cut_id,
        "selected_ancestry": _lineage(project, session_id, cut_id, retained, cut_cache, frontier)
        if cut_id is not None
        else [],
        "cuts": cut_rows,
        "receipts": [receipt_rows[key] for key in sorted(receipt_rows)],
        "support": "native-cut-parent-ancestry; partial-retention",
    }
    graph = _graph(
        symbol=symbol,
        binding_groups=binding_groups,
        occurrences=occurrences,
        uses=uses,
        cuts=cut_rows,
        frontier=frontier,
        slice_joins=slice_joins,
        links=links,
    )
    body = {
        "schema": SYMBOL_TRACE_SCHEMA,
        "version": 1,
        "session_id": session_id,
        "symbol": symbol,
        "filters": {"branch": branch, "cut_id": cut_id},
        "binding_groups": binding_groups,
        "occurrences": occurrences,
        "uses": uses,
        "links": links,
        "signed_observations": signed_observations,
        "cot_refs": cot_refs,
        "evidence_refs": sorted(evidence_refs),
        "lifecycle_snapshots": sorted(snapshots, key=lambda row: row["lifecycle_id"]),
        "invalidations": sorted(invalidations, key=lambda row: row["invalidation_fingerprint"]),
        "chronology": chronology,
        "slice_joins": slice_joins,
        "joins": {"slices": slice_joins, "uses": links},
        "coverage": {
            "retained_cuts": len(retained),
            "selected_ancestry_cuts": len(selected_ancestry),
            "binding_groups": len(binding_groups),
            "occurrences": len(occurrences),
            "uses": len(uses),
            "filtered_events": len(filtered_events),
            "frontier_count": len(frontier),
            "scope": "registered lifecycle events on retained native cuts",
            "missing_is_not_absence": True,
        },
        "filtered_events": sorted(
            filtered_events,
            key=lambda row: (str(row.get("kind")), str(row.get("event_fingerprint")), row["reason"]),
        ),
        "frontier": sorted(frontier, key=_frontier_key),
        "graph": graph,
        "authority": "read-only-evidence-trace",
        "causal_effect": "not-established",
        "claim_boundary": (
            "This trace joins supplied symbol names to exact native lifecycle evidence, cuts, "
            "receipts, and declared slices. It does not establish semantic equivalence, "
            "causal necessity, or write authority."
        ),
    }
    body = _json_copy(body)
    return {**body, "fingerprint": digest(body)}


def verify_symbol_trace(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a trace seal without loading native state."""

    if not isinstance(value, Mapping) or value.get("schema") != SYMBOL_TRACE_SCHEMA:
        raise SymbolTraceError("unsupported symbol trace schema")
    declared = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if declared != digest(body):
        raise SymbolTraceError("symbol trace fingerprint mismatch")
    return {
        "schema": SYMBOL_TRACE_SCHEMA,
        "fingerprint": declared,
        "session_id": value.get("session_id"),
        "symbol": value.get("symbol"),
        "passed": True,
    }


def trace_mermaid(trace: Mapping[str, Any]) -> str:
    """Render a compact descriptive view of a sealed trace."""

    verify_symbol_trace(trace)
    lines = ["graph LR"]
    for node in trace.get("graph", {}).get("nodes", ()):
        node_id = "n_" + digest(node["id"])[:12]
        label = str(node.get("kind", "node")) + "\\n" + str(node.get("id", ""))
        lines.append(f'  {node_id}["{label.replace(chr(34), chr(39))}"]')
    for edge in trace.get("graph", {}).get("edges", ()):
        source = "n_" + digest(edge["source"])[:12]
        target = "n_" + digest(edge["target"])[:12]
        relation = str(edge.get("relation", "describes")).replace("|", "/")
        lines.append(f"  {source} -->|{relation}| {target}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace one supplied symbol across sealed MDB lifecycles")
    parser.add_argument("root", type=Path)
    parser.add_argument("symbol")
    parser.add_argument("--session-id")
    parser.add_argument("--cut", dest="cut_id")
    parser.add_argument("--branch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mermaid", action="store_true")
    args = parser.parse_args(argv)

    from .project import Project

    project = Project(args.root)
    session_id = args.session_id
    if session_id is None:
        sessions = project.sessions()
        if len(sessions) != 1:
            parser.error("select --session-id for a project with zero or multiple sessions")
        session_id = sessions[0]["session_id"]
    trace = trace_symbol(
        project,
        session_id,
        args.symbol,
        branch=args.branch,
        cut_id=args.cut_id,
    )
    output = trace_mermaid(trace) if args.mermaid else json.dumps(trace, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


__all__ = [
    "SYMBOL_TRACE_SCHEMA",
    "TRACE_GRAPH_SCHEMA",
    "SymbolTraceError",
    "trace_mermaid",
    "trace_symbol",
    "verify_symbol_trace",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI smoke path
    raise SystemExit(main())
