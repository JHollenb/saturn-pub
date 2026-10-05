"""Join Thott execution traces with temporal capability-route evidence.

This module is deliberately artifact-only.  It does not load a model, retain
activations, or infer a semantic thought from a tensor summary.  It joins the
two existing views into one bounded, typed graph:

    live producer/carrier/consumer events
        + temporal route sites and source-address candidates
        + intervention outcomes and future-forest lineage

The graph is useful for path inspection and graph-algorithm proposals.  Its
edges retain their evidence class so a whole-route consumer closure cannot be
mistaken for sufficiency of one local Q/K/V endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "saturn-thought-route-graph-v1"
CLAIM_BOUNDARY = (
    "Artifact-only typed execution graph. It records observed execution, route trends, "
    "candidate interventions, forecasts, and future lineage; it is not a private chain-of-"
    "thought transcript, semantic oracle, or terminal circuit certificate."
)


def _safe(value: Any) -> Any:
    """Return JSON-safe metadata and fail closed on non-finite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("thought-route graph metadata must be finite")
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe(item) for item in value]
    raise TypeError(f"unsupported graph metadata: {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    """Return a byte-addressed input reference without embedding its contents."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"graph input must be a regular file: {resolved}")
    data = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _node(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    *,
    kind: str,
    modality: str,
    claim_status: str,
    metadata: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any] | str] = (),
) -> None:
    if not node_id:
        raise ValueError("graph node ID must be non-empty")
    row = nodes.get(node_id)
    if row is None:
        nodes[node_id] = {
            "node_id": node_id,
            "kind": str(kind),
            "modality": str(modality),
            "claim_status": str(claim_status),
            "metadata": _safe(metadata or {}),
            "evidence": [_safe(item) for item in evidence],
        }
        return
    if row["kind"] != str(kind) or row["modality"] != str(modality):
        raise ValueError(f"graph node {node_id!r} was redefined with another type")
    row["metadata"].update(_safe(metadata or {}))
    row["evidence"] = _unique_items([*row["evidence"], *[_safe(item) for item in evidence]])
    if row["claim_status"] == "observation" and claim_status != "observation":
        row["claim_status"] = str(claim_status)


def _edge(
    edges: dict[str, dict[str, Any]],
    source: str,
    target: str,
    *,
    kind: str,
    claim_status: str,
    metadata: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any] | str] = (),
) -> str:
    if not source or not target:
        raise ValueError("graph edges require source and target")
    body = {
        "source": source,
        "target": target,
        "kind": str(kind),
        "claim_status": str(claim_status),
        "metadata": _safe(metadata or {}),
        "evidence": [_safe(item) for item in evidence],
    }
    edge_id = f"edge:{_digest(body)[:32]}"
    if edge_id in edges:
        edges[edge_id]["evidence"] = _unique_items([*edges[edge_id]["evidence"], *body["evidence"]])
        return edge_id
    edges[edge_id] = {"edge_id": edge_id, **body}
    return edge_id


def _unique_items(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _site_sort_key(site: str) -> tuple[str, int, str]:
    match = re.match(r"^(.*?)(\d+)$", str(site))
    if match:
        return (match.group(1), int(match.group(2)), str(site))
    return (str(site), -1, str(site))


def _id_part(value: Any) -> str:
    """Make a stable, readable identifier component from captured metadata."""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "unknown"


def _number(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _consumer_claim(metrics: Mapping[str, Any]) -> str:
    """Classify a local intervention without promoting it to a certificate."""

    return_alignment = _number(metrics.get("return_alignment_to_donor"))
    rgb_alignment = _number(metrics.get("rgb_alignment_to_donor"))
    if (
        return_alignment is not None
        and rgb_alignment is not None
        and return_alignment >= 0.75
        and rgb_alignment >= 0.75
    ):
        return "consumer_closed_trend"
    return "causal_candidate"


def _grammar(nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> None:
    roles = ("producer", "carrier", "consumer", "readout", "counterfactual")
    for role in roles:
        _node(
            nodes,
            f"grammar::{role}",
            kind="abstract_role",
            modality="abstract",
            claim_status="design_contract",
            metadata={
                "role": role,
                "meaning": "typed role alignment only; not physical or semantic equivalence",
            },
        )
    for left, right in zip(roles, roles[1:], strict=False):
        _edge(
            edges,
            f"grammar::{left}",
            f"grammar::{right}",
            kind="abstract_program_order",
            claim_status="design_contract",
            metadata={"alignment_scope": "producer-carrier-consumer-readout-future grammar"},
        )


def _add_thott_trace(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    trace: Mapping[str, Any],
) -> list[str]:
    if trace.get("schema") != "saturn-thott-circuit-thought-v1":
        raise ValueError("unsupported Thott circuit trace schema")
    physical_ids: list[str] = []
    for raw in trace.get("nodes", ()):
        if not isinstance(raw, Mapping):
            continue
        raw_id = str(raw.get("node_id") or "")
        if not raw_id:
            continue
        node_id = f"thott::{raw_id}"
        physical_ids.append(node_id)
        role = str(raw.get("role") or "unknown")
        _node(
            nodes,
            node_id,
            kind=role,
            modality="thott",
            claim_status="typed_observation",
            metadata={
                "address": raw.get("address"),
                "layer": raw.get("layer"),
                "position": raw.get("position"),
                "component": raw.get("component"),
                "consumer": raw.get("consumer"),
                "trace_circuit_id": trace.get("circuit_id"),
                **dict(raw.get("metadata") or {}),
            },
            evidence=[
                {"source_schema": trace.get("schema"), "circuit_id": trace.get("circuit_id")}
            ],
        )
        if role in {"producer", "carrier", "consumer", "readout"}:
            _edge(
                edges,
                node_id,
                f"grammar::{role}",
                kind="typed_role_alignment",
                claim_status="design_contract",
                metadata={"alignment_scope": "Thott role to common execution grammar"},
            )

    for raw in trace.get("edges", ()):
        if not isinstance(raw, Mapping):
            continue
        source = f"thott::{raw.get('source_node')}"
        target = f"thott::{raw.get('target_node')}"
        if source not in nodes or target not in nodes:
            continue
        _edge(
            edges,
            source,
            target,
            kind="thott_dataflow",
            claim_status="typed_observation",
            metadata={
                "carrier": raw.get("carrier"),
                "consumer": raw.get("consumer"),
                **dict(raw.get("metadata") or {}),
            },
            evidence=[
                {"source_schema": trace.get("schema"), "circuit_id": trace.get("circuit_id")}
            ],
        )

    previous_event: str | None = None
    for raw in sorted(trace.get("events", ()), key=lambda row: int(row.get("sequence", 0))):
        if not isinstance(raw, Mapping):
            continue
        event_key = str(raw.get("event_key") or raw.get("sequence") or "")
        if not event_key:
            continue
        event_id = f"thott::event::{event_key}"
        event_node = f"thott::{raw.get('node_id')}"
        _node(
            nodes,
            event_id,
            kind="event",
            modality="thott",
            claim_status="typed_observation",
            metadata={
                "sequence": raw.get("sequence"),
                "phase": raw.get("phase"),
                "execution_step": raw.get("execution_step"),
                "stage": raw.get("stage"),
                "summary": raw.get("summary") or {},
                "event_metadata": raw.get("metadata") or {},
            },
            evidence=[{"source_schema": trace.get("schema"), "event_key": event_key}],
        )
        if event_node in nodes:
            _edge(
                edges,
                event_id,
                event_node,
                kind="observes",
                claim_status="typed_observation",
                metadata={"stage": raw.get("stage"), "phase": raw.get("phase")},
            )
        if previous_event is not None:
            _edge(
                edges,
                previous_event,
                event_id,
                kind="temporal_next",
                claim_status="typed_observation",
            )
        previous_event = event_id

    for index, raw in enumerate(trace.get("interventions", ())):
        if not isinstance(raw, Mapping):
            continue
        branch = str(raw.get("branch") or f"branch-{index}")
        intervention_id = f"thott::counterfactual::{index}::{branch}"
        target = f"thott::{raw.get('node_id')}"
        _node(
            nodes,
            intervention_id,
            kind="counterfactual",
            modality="thott",
            claim_status=str(raw.get("status") or "causal_candidate"),
            metadata={
                "name": raw.get("name"),
                "branch": branch,
                "dose": raw.get("dose"),
                "local_effect": raw.get("local_effect") or {},
                "consumer_effect": raw.get("consumer_effect") or {},
                "behavior_effect": raw.get("behavior_effect") or {},
                "repaired": raw.get("repaired"),
            },
            evidence=[{"source_schema": trace.get("schema"), "branch": branch}],
        )
        if target in nodes:
            _edge(
                edges,
                target,
                intervention_id,
                kind="counterfactual_branch",
                claim_status=str(raw.get("status") or "causal_candidate"),
                metadata={"dose": raw.get("dose"), "branch": branch},
            )
        _edge(
            edges,
            intervention_id,
            "grammar::counterfactual",
            kind="future_annotation",
            claim_status="design_contract",
        )
    return physical_ids


def _add_qwen_summary(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    report: Mapping[str, Any] | None,
    raw_manifest: Mapping[str, Any] | None,
) -> list[str]:
    if report is None:
        return []
    configuration = dict(report.get("configuration") or {})
    model = str(configuration.get("model") or "qwen")
    prefix = "qwen::"
    producer = f"{prefix}producer::block_mlp_qkv"
    residual = f"{prefix}carrier::residual_workspace"
    cache = f"{prefix}carrier::normalized_kv_cache"
    attention = f"{prefix}consumer::native_attention"
    o_proj = f"{prefix}consumer::o_proj_and_later_blocks"
    readout = f"{prefix}readout::lm_head"
    continuation = f"{prefix}continuation::visible_tokens"
    rows = (
        (
            producer,
            "producer",
            {"model": model, "captured_surface": configuration.get("captured_surface")},
        ),
        (residual, "carrier", {"model": model, "carrier": "residual workspace"}),
        (cache, "carrier", {"model": model, "carrier": "normalized per-layer K/V cache"}),
        (attention, "consumer", {"model": model, "consumer": "native attention weights"}),
        (o_proj, "consumer", {"model": model, "consumer": "o_proj → later blocks"}),
        (readout, "readout", {"model": model, "consumer": "lm_head logits"}),
        (continuation, "counterfactual", {"model": model, "readout": "visible generated tokens"}),
    )
    for node_id, kind, metadata in rows:
        _node(
            nodes,
            node_id,
            kind=kind,
            modality="autoregressive",
            claim_status="typed_observation",
            metadata={
                **metadata,
                "trace_layers": configuration.get("trace_layers")
                or configuration.get("selected_layers"),
                "report_schema": report.get("schema"),
            },
            evidence=[
                {
                    "source_schema": report.get("schema"),
                    "claim_boundary": report.get("claim_boundary"),
                }
            ],
        )
        if kind in {"producer", "carrier", "consumer", "readout"}:
            _edge(
                edges,
                node_id,
                f"grammar::{kind}",
                kind="typed_role_alignment",
                claim_status="design_contract",
                metadata={"alignment_scope": "Qwen Thott summary to common execution grammar"},
            )
    for source, target, kind in (
        (producer, residual, "writes"),
        (producer, cache, "writes"),
        (cache, attention, "consumed_by"),
        (residual, o_proj, "transported_to"),
        (attention, o_proj, "writes"),
        (o_proj, readout, "consumed_by"),
        (readout, continuation, "continues_to"),
    ):
        _edge(
            edges,
            source,
            target,
            kind=f"qwen_{kind}",
            claim_status="typed_observation",
            metadata={"model": model, "report_schema": report.get("schema")},
        )
    for prompt in configuration.get("prompts", ()):
        if not isinstance(prompt, Mapping):
            continue
        prompt_id = str(prompt.get("id") or "")
        if not prompt_id:
            continue
        task_id = f"{prefix}task::{prompt_id}"
        _node(
            nodes,
            task_id,
            kind="task_variable",
            modality="autoregressive",
            claim_status="external_label",
            metadata={"expected": prompt.get("expected"), "model": model},
            evidence=[{"source_schema": report.get("schema"), "label_source": "prompt_contract"}],
        )
        _edge(
            edges,
            task_id,
            residual,
            kind="context_conditions",
            claim_status="external_label",
            metadata={"meaning": "prompt/task label conditions the observed execution"},
        )
    if raw_manifest is not None:
        artifact_id = f"{prefix}artifact::raw_workspace"
        _node(
            nodes,
            artifact_id,
            kind="artifact",
            modality="autoregressive",
            claim_status="custodied_observation",
            metadata={
                "schema": raw_manifest.get("schema"),
                "tensor_count": raw_manifest.get("tensor_count"),
                "scope": raw_manifest.get("scope"),
            },
        )
        _edge(
            edges,
            artifact_id,
            cache,
            kind="backs_observation",
            claim_status="custodied_observation",
            metadata={"raw_activation_policy": "artifact custody; not semantic evidence by itself"},
        )
    return [producer, residual, cache, attention, o_proj, readout, continuation]


def _gate_rows(confirmation: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for raw in confirmation.get("gates", ()):
        if not isinstance(raw, Mapping):
            continue
        gate_id = str(raw.get("id") or "")
        if "." in gate_id:
            domain, name = gate_id.split(".", 1)
        else:
            domain, name = "global", gate_id
        grouped.setdefault(domain, {})[name] = raw
    return grouped


def _add_flux_routes(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    confirmation: Mapping[str, Any] | None,
    predictors: Mapping[str, Any] | None,
    predictions: Mapping[str, Any] | None,
) -> list[str]:
    if confirmation is None:
        return []
    if confirmation.get("schema") != "saturn-flux-route-confirmation-analysis-v1":
        raise ValueError("unsupported FLUX route confirmation schema")
    prefix = "flux::"
    return_consumer = f"{prefix}consumer::return_register"
    rgb_consumer = f"{prefix}consumer::rgb_render"
    _node(
        nodes,
        return_consumer,
        kind="consumer",
        modality="diffusion",
        claim_status="native_consumer",
        metadata={"consumer": "latent return register"},
        evidence=[{"source_schema": confirmation.get("schema")}],
    )
    _node(
        nodes,
        rgb_consumer,
        kind="readout",
        modality="diffusion",
        claim_status="native_consumer",
        metadata={"consumer": "RGB render / VAE readout"},
        evidence=[{"source_schema": confirmation.get("schema")}],
    )
    _edge(
        edges,
        return_consumer,
        "grammar::consumer",
        kind="typed_role_alignment",
        claim_status="design_contract",
        metadata={"alignment_scope": "FLUX native return consumer"},
    )
    _edge(
        edges,
        rgb_consumer,
        "grammar::readout",
        kind="typed_role_alignment",
        claim_status="design_contract",
        metadata={"alignment_scope": "FLUX RGB readout"},
    )

    site_summary = confirmation.get("site_summary") or {}
    sites = sorted((str(site) for site in site_summary), key=_site_sort_key)
    site_ids: list[str] = []
    for site in sites:
        site_id = f"{prefix}site::{site}"
        site_ids.append(site_id)
        _node(
            nodes,
            site_id,
            kind="route_site",
            modality="diffusion",
            claim_status="temporal_route_observation",
            metadata={"site": site, "execution_axis": "denoising/joint site"},
            evidence=[{"source_schema": confirmation.get("schema")}],
        )
        _edge(
            edges,
            site_id,
            "grammar::carrier",
            kind="typed_role_alignment",
            claim_status="design_contract",
            metadata={"alignment_scope": "FLUX route site as carrier boundary"},
        )
    for left, right in zip(site_ids, site_ids[1:], strict=False):
        _edge(
            edges,
            left,
            right,
            kind="temporal_route",
            claim_status="temporal_route_observation",
            metadata={"axis": "joint-site order"},
        )

    proposals = confirmation.get("route_proposals") or {}
    grouped_gates = _gate_rows(confirmation)
    candidate_ids: dict[str, str] = {}
    for domain, proposal in sorted(proposals.items()):
        if not isinstance(proposal, Mapping):
            continue
        domain = str(domain)
        task_id = f"{prefix}task::{domain}"
        candidate_id = f"{prefix}candidate::{domain}"
        candidate_ids[domain] = candidate_id
        site = str(proposal.get("site") or "")
        qkv = str(proposal.get("qkv") or "")
        _node(
            nodes,
            task_id,
            kind="task_variable",
            modality="diffusion",
            claim_status="external_label",
            metadata={"domain": domain, "label_source": "route panel"},
            evidence=[{"source_schema": confirmation.get("schema")}],
        )
        _node(
            nodes,
            candidate_id,
            kind="candidate_operation",
            modality="diffusion",
            claim_status="structural_candidate",
            metadata={"domain": domain, "site": site, "qkv": qkv},
            evidence=[{"source_schema": confirmation.get("schema"), "proposal": dict(proposal)}],
        )
        _edge(
            edges,
            task_id,
            candidate_id,
            kind="source_address_hypothesis",
            claim_status="structural_candidate",
            metadata={"site": site, "qkv": qkv, "semantic_label_is_external": True},
        )
        if site:
            _edge(
                edges,
                candidate_id,
                f"{prefix}site::{site}",
                kind="targets_route_site",
                claim_status="structural_candidate",
                metadata={"qkv": qkv},
            )
        gates = grouped_gates.get(domain, {})
        gate_ids = (
            "replication",
            "qkv_necessity",
            "qkv_sufficiency",
            "qkv_specificity",
            "dose_response",
        )
        for gate_name in gate_ids:
            gate = gates.get(gate_name)
            if gate is None:
                continue
            gate_id = f"{prefix}gate::{domain}.{gate_name}"
            passed = bool(gate.get("passed"))
            _node(
                nodes,
                gate_id,
                kind="evidence_gate",
                modality="diffusion",
                claim_status="gate_result",
                metadata={
                    "domain": domain,
                    "gate": gate_name,
                    "passed": passed,
                    "observed": gate.get("observed") or {},
                    "rule": gate.get("rule"),
                },
                evidence=[{"source_schema": confirmation.get("schema"), "gate_id": gate.get("id")}],
            )
            _edge(
                edges,
                candidate_id,
                gate_id,
                kind="tested_by",
                claim_status="gate_result",
                metadata={"passed": passed},
            )
        for consumer_id, metric_key in (
            (return_consumer, "mean_endpoint_return_alignment"),
            (rgb_consumer, "mean_endpoint_rgb_alignment"),
        ):
            gate = gates.get("qkv_sufficiency") or {}
            _edge(
                edges,
                candidate_id,
                consumer_id,
                kind="candidate_consumer_test",
                claim_status="causal_candidate"
                if not gate.get("passed")
                else "consumer_closed_candidate",
                metadata={
                    "domain": domain,
                    "consumer": consumer_id,
                    "metric": metric_key,
                    "sufficiency_passed": bool(gate.get("passed")),
                    "observed": gate.get("observed") or {},
                    "interpretation": "local endpoint is not equivalent to whole-route closure",
                },
                evidence=[{"source_schema": confirmation.get("schema"), "gate_id": gate.get("id")}],
            )

    for site in sites:
        site_id = f"{prefix}site::{site}"
        per_domain = site_summary.get(site) or {}
        for domain, metrics in sorted(per_domain.items()):
            if not isinstance(metrics, Mapping):
                continue
            observation_id = f"{prefix}observation::{site}::{domain}"
            return_alignment = metrics.get("mean_return_alignment")
            closure_trend = (
                isinstance(return_alignment, (int, float)) and float(return_alignment) >= 0.75
            )
            _node(
                nodes,
                observation_id,
                kind="route_observation",
                modality="diffusion",
                claim_status="consumer_closed_trend" if closure_trend else "carrier_trend",
                metadata={"site": site, "domain": domain, "metrics": dict(metrics)},
                evidence=[
                    {"source_schema": confirmation.get("schema"), "site": site, "domain": domain}
                ],
            )
            _edge(
                edges,
                site_id,
                observation_id,
                kind="measured_at_route_site",
                claim_status="observed",
                metadata={"domain": domain},
            )
            for consumer_id, metric_key in (
                (return_consumer, "mean_return_alignment"),
                (rgb_consumer, "mean_rgb_alignment"),
            ):
                _edge(
                    edges,
                    observation_id,
                    consumer_id,
                    kind="consumer_closure_observation",
                    claim_status="consumer_closed_trend" if closure_trend else "carrier_trend",
                    metadata={
                        "domain": domain,
                        "site": site,
                        "metric": metric_key,
                        "value": metrics.get(metric_key),
                    },
                )

    whole_gate = next(
        (
            row
            for row in confirmation.get("gates", ())
            if isinstance(row, Mapping) and row.get("id") == "whole_edge_site_closure"
        ),
        None,
    )
    if whole_gate is not None:
        gate_id = f"{prefix}gate::whole_edge_site_closure"
        _node(
            nodes,
            gate_id,
            kind="evidence_gate",
            modality="diffusion",
            claim_status="gate_result",
            metadata={
                "passed": bool(whole_gate.get("passed")),
                "observed": whole_gate.get("observed") or {},
                "rule": whole_gate.get("rule"),
            },
            evidence=[
                {"source_schema": confirmation.get("schema"), "gate_id": whole_gate.get("id")}
            ],
        )
        for site_id in site_ids:
            _edge(
                edges,
                site_id,
                gate_id,
                kind="whole_route_gate",
                claim_status="gate_result",
                metadata={"passed": bool(whole_gate.get("passed")), "scope": "whole text state"},
            )

    if predictors:
        predictor_rows = predictors
        if isinstance(predictors.get("predictors"), Mapping):
            predictor_rows = predictors["predictors"]
        for name, raw in sorted(predictor_rows.items()):
            if (
                not isinstance(raw, Mapping)
                or raw.get("schema") != "saturn-portable-mechanism-forecaster-v1"
            ):
                continue
            predictor_id = f"{prefix}forecaster::{name}"
            _node(
                nodes,
                predictor_id,
                kind="forecaster",
                modality="diffusion",
                claim_status="forecast_only",
                metadata={
                    "features": raw.get("features") or [],
                    "outcome_key": raw.get("outcome_key"),
                    "discovery_groups": raw.get("discovery_groups") or [],
                    "discovery_rows": raw.get("discovery_rows"),
                    "identity_or_clock_features_visible": raw.get(
                        "identity_or_clock_features_visible"
                    ),
                    "fingerprint": raw.get("fingerprint"),
                },
                evidence=[{"source_schema": raw.get("schema"), "predictor": name}],
            )
            if not predictions:
                continue
            prediction_rows = predictions.get(name) if isinstance(predictions, Mapping) else None
            if isinstance(prediction_rows, Mapping) and isinstance(
                prediction_rows.get("rows"), Sequence
            ):
                rows = prediction_rows["rows"]
                prediction_schema = prediction_rows.get("schema")
            else:
                rows = ()
                prediction_schema = None
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                row_id = str(row.get("row_id") or "")
                prediction_id = f"{predictor_id}::prediction::{row_id}"
                _node(
                    nodes,
                    prediction_id,
                    kind="forecast",
                    modality="diffusion",
                    claim_status="forecast_only",
                    metadata={
                        "row_id": row_id,
                        "prediction": row.get("prediction"),
                        "group": row.get("group"),
                        "outcomes_visible": prediction_rows.get("outcomes_visible")
                        if isinstance(prediction_rows, Mapping)
                        else None,
                    },
                    evidence=[{"source_schema": prediction_schema, "predictor": name}],
                )
                _edge(
                    edges,
                    predictor_id,
                    prediction_id,
                    kind="predicts",
                    claim_status="forecast_only",
                )
                parts = row_id.split(":")
                if len(parts) >= 4:
                    candidate_id = candidate_ids.get(parts[1])
                    if candidate_id:
                        _edge(
                            edges,
                            prediction_id,
                            candidate_id,
                            kind="forecasts_candidate",
                            claim_status="forecast_only",
                            metadata={"site": parts[2], "qkv": parts[3], "outcomes_visible": False},
                        )
    return site_ids


def _add_joint_token_route(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Add lexical coordinates and consumer outcomes from the native joint panel.

    The report deliberately contains summaries rather than tensor payloads.  A compressed raw
    QKV sidecar can remain an input reference, while this graph keeps token spans, positions,
    route-site/time coordinates, local QKV summaries, and intervention outcomes as typed edges.
    None of these local observations is promoted to a semantic capability label here.
    """

    if report.get("schema") != "saturn-joint-token-route-capture.v1":
        raise ValueError("unsupported joint token route capture schema")

    prefix = "joint::"
    recipe = dict(report.get("recipe") or {})
    capture_id = (
        f"{prefix}capture::{_id_part(report.get('model') or 'flux')}"
        f"::seed::{_id_part(recipe.get('seed'))}"
    )
    _node(
        nodes,
        capture_id,
        kind="artifact",
        modality="diffusion",
        claim_status="custodied_observation",
        metadata={
            "schema": report.get("schema"),
            "model": report.get("model"),
            "seed": recipe.get("seed"),
            "steps": recipe.get("steps"),
            "resolution": recipe.get("resolution"),
            "route_sites": recipe.get("route_sites") or [],
            "qkv_sites": recipe.get("qkv_sites") or [],
            "qkv_kinds": recipe.get("qkv_kinds") or [],
            "retokenized_variant": recipe.get("retokenized_variant"),
            "tensor_payload_embedded": False,
        },
        evidence=[{"source_schema": report.get("schema")}],
    )

    condition_ids: dict[str, str] = {}
    span_ids: dict[tuple[str, str], str] = {}
    position_ids: dict[tuple[str, str, int], str] = {}
    lexeme_ids: dict[tuple[int, str], str] = {}
    conditions = report.get("conditions") or {}
    if not isinstance(conditions, Mapping):
        raise ValueError("joint token route conditions must be an object")

    for condition, raw_condition in sorted(conditions.items()):
        if not isinstance(raw_condition, Mapping):
            continue
        condition = str(condition)
        condition_id = f"{prefix}condition::{_id_part(condition)}"
        condition_ids[condition] = condition_id
        _node(
            nodes,
            condition_id,
            kind="task_variable",
            modality="diffusion",
            claim_status="typed_observation",
            metadata={
                "condition": condition,
                "prompt": raw_condition.get("prompt"),
                "contrast": raw_condition.get("contrast"),
                "active_len": raw_condition.get("active_len"),
            },
            evidence=[{"source_schema": report.get("schema"), "condition": condition}],
        )
        _edge(
            edges,
            capture_id,
            condition_id,
            kind="captures_condition",
            claim_status="custodied_observation",
            metadata={"condition": condition},
        )
        loci = raw_condition.get("loci") or {}
        if not isinstance(loci, Mapping):
            continue
        for locus, raw_locus in sorted(loci.items()):
            if not isinstance(raw_locus, Mapping):
                continue
            locus = str(locus)
            span_id = f"{prefix}token-span::{_id_part(condition)}::{_id_part(locus)}"
            span_ids[(condition, locus)] = span_id
            positions = [int(value) for value in (raw_locus.get("positions") or [])]
            token_ids = [int(value) for value in (raw_locus.get("token_ids") or [])]
            tokens = [str(value) for value in (raw_locus.get("tokens") or [])]
            _node(
                nodes,
                span_id,
                kind="producer",
                modality="diffusion",
                claim_status="typed_observation",
                metadata={
                    "condition": condition,
                    "locus": locus,
                    "word": raw_locus.get("word"),
                    "role": raw_locus.get("role"),
                    "positions": positions,
                    "token_ids": token_ids,
                    "tokens": tokens,
                },
                evidence=[
                    {
                        "source_schema": report.get("schema"),
                        "condition": condition,
                        "locus": locus,
                    }
                ],
            )
            _edge(
                edges,
                condition_id,
                span_id,
                kind="conditions_source_span",
                claim_status="typed_observation",
                metadata={"locus": locus, "role": raw_locus.get("role")},
            )
            for index, position in enumerate(positions):
                position_id = (
                    f"{prefix}token-position::{_id_part(condition)}"
                    f"::{_id_part(locus)}::{position}"
                )
                position_ids[(condition, locus, position)] = position_id
                token_id = token_ids[index] if index < len(token_ids) else None
                token = tokens[index] if index < len(tokens) else None
                _node(
                    nodes,
                    position_id,
                    kind="token_position",
                    modality="diffusion",
                    claim_status="typed_observation",
                    metadata={
                        "condition": condition,
                        "locus": locus,
                        "position": position,
                        "token_id": token_id,
                        "token": token,
                    },
                    evidence=[
                        {
                            "source_schema": report.get("schema"),
                            "condition": condition,
                            "position": position,
                        }
                    ],
                )
                _edge(
                    edges,
                    span_id,
                    position_id,
                    kind="contains_token_position",
                    claim_status="typed_observation",
                    metadata={"position": position},
                )
                if token_id is not None:
                    lexeme_key = (token_id, str(token))
                    lexeme_id = lexeme_ids.get(lexeme_key)
                    if lexeme_id is None:
                        lexeme_id = (
                            f"{prefix}lexeme::{token_id}::{_id_part(token)}"
                        )
                        lexeme_ids[lexeme_key] = lexeme_id
                        _node(
                            nodes,
                            lexeme_id,
                            kind="task_variable",
                            modality="diffusion",
                            claim_status="typed_observation",
                            metadata={"token_id": token_id, "token": token},
                            evidence=[{"source_schema": report.get("schema")}],
                        )
                    _edge(
                        edges,
                        position_id,
                        lexeme_id,
                        kind="instantiates_lexeme",
                        claim_status="typed_observation",
                        metadata={"token_id": token_id, "token": token},
                    )

    # Preserve the designed lexical contrasts as explicit graph objects.  A contrast is a relation
    # between source spans, not a claim that the spans are semantically identical.
    for condition, raw_condition in sorted(conditions.items()):
        if not isinstance(raw_condition, Mapping):
            continue
        contrast = raw_condition.get("contrast")
        if not isinstance(contrast, Mapping):
            continue
        kind = str(contrast.get("kind") or "contrast")
        locus = str(contrast.get("locus") or "unknown")
        contrast_id = f"{prefix}contrast::{_id_part(kind)}::{_id_part(locus)}"
        _node(
            nodes,
            contrast_id,
            kind="contrast",
            modality="diffusion",
            claim_status="design_contract",
            metadata={
                "kind": kind,
                "locus": locus,
                "condition": str(condition),
                "semantic_equivalence_is_not_assumed": True,
            },
            evidence=[{"source_schema": report.get("schema"), "condition": condition}],
        )
        current_span = span_ids.get((str(condition), locus))
        if current_span:
            _edge(
                edges,
                current_span,
                contrast_id,
                kind="participates_in_contrast",
                claim_status="design_contract",
                metadata={"kind": kind, "locus": locus},
            )
        if kind == "same_token_different_context":
            left_span = span_ids.get(("base", "subject_color"))
            right_span = span_ids.get(("base", "light_color"))
        else:
            left_span = span_ids.get(("base", locus))
            right_span = current_span
        if left_span and right_span and left_span != right_span:
            _edge(
                edges,
                left_span,
                right_span,
                kind="contrast_relation",
                claim_status="observed",
                metadata={"kind": kind, "locus": locus, "semantic_label_is_external": True},
            )
            _edge(
                edges,
                left_span,
                contrast_id,
                kind="participates_in_contrast",
                claim_status="design_contract",
                metadata={"kind": kind, "locus": locus},
            )

    route_site_ids: dict[str, str] = {}
    route_step_ids: dict[tuple[str, int], str] = {}
    route_observation_ids: dict[tuple[str, str, str, int], str] = {}
    profiles = report.get("native_route_profiles") or {}
    if not isinstance(profiles, Mapping):
        profiles = {}
    for condition, raw_streams in sorted(profiles.items()):
        if not isinstance(raw_streams, Mapping):
            continue
        condition = str(condition)
        for stream, raw_sites in sorted(raw_streams.items()):
            if not isinstance(raw_sites, Mapping):
                continue
            stream = str(stream)
            for site, raw_steps in sorted(
                raw_sites.items(), key=lambda item: _site_sort_key(str(item[0]))
            ):
                if not isinstance(raw_steps, Mapping):
                    continue
                site = str(site)
                site_id = route_site_ids.get(site)
                if site_id is None:
                    site_id = f"{prefix}route-site::{_id_part(site)}"
                    route_site_ids[site] = site_id
                    _node(
                        nodes,
                        site_id,
                        kind="route_site",
                        modality="diffusion",
                        claim_status="temporal_route_observation",
                        metadata={"site": site, "source": "joint token route profile"},
                        evidence=[{"source_schema": report.get("schema"), "site": site}],
                    )
                    _edge(
                        edges,
                        site_id,
                        "grammar::carrier",
                        kind="typed_role_alignment",
                        claim_status="design_contract",
                        metadata={"alignment_scope": "joint route site as carrier boundary"},
                    )
                for raw_step, summary in sorted(raw_steps.items(), key=lambda item: int(item[0])):
                    if not isinstance(summary, Mapping):
                        continue
                    step = int(raw_step)
                    step_id = route_step_ids.get((site, step))
                    if step_id is None:
                        step_id = f"{prefix}route-step::{_id_part(site)}::{step}"
                        route_step_ids[(site, step)] = step_id
                        _node(
                            nodes,
                            step_id,
                            kind="route_step",
                            modality="diffusion",
                            claim_status="temporal_route_observation",
                            metadata={"site": site, "step": step},
                            evidence=[{"source_schema": report.get("schema")}],
                        )
                        _edge(
                            edges,
                            site_id,
                            step_id,
                            kind="temporal_step",
                            claim_status="temporal_route_observation",
                            metadata={"site": site, "step": step},
                        )
                        if step > 0 and (site, step - 1) in route_step_ids:
                            _edge(
                                edges,
                                route_step_ids[(site, step - 1)],
                                step_id,
                                kind="temporal_next",
                                claim_status="temporal_route_observation",
                                metadata={"site": site},
                            )
                    observation_id = (
                        f"{prefix}route-observation::{_id_part(condition)}"
                        f"::{_id_part(stream)}::{_id_part(site)}::{step}"
                    )
                    route_observation_ids[(condition, stream, site, step)] = observation_id
                    _node(
                        nodes,
                        observation_id,
                        kind="route_observation",
                        modality="diffusion",
                        claim_status="carrier_trend",
                        metadata={
                            "condition": condition,
                            "stream": stream,
                            "site": site,
                            "step": step,
                            "summary": dict(summary),
                        },
                        evidence=[
                            {
                                "source_schema": report.get("schema"),
                                "condition": condition,
                                "site": site,
                                "step": step,
                            }
                        ],
                    )
                    _edge(
                        edges,
                        step_id,
                        observation_id,
                        kind="measured_at_route_step",
                        claim_status="observed",
                        metadata={"condition": condition, "stream": stream},
                    )
                    condition_id = condition_ids.get(condition)
                    if condition_id:
                        _edge(
                            edges,
                            condition_id,
                            observation_id,
                            kind="condition_route_trajectory",
                            claim_status="observed",
                            metadata={"stream": stream, "site": site, "step": step},
                        )
                    if stream == "text":
                        for (span_condition, locus), span_id in span_ids.items():
                            if span_condition != condition:
                                continue
                            _edge(
                                edges,
                                span_id,
                                observation_id,
                                kind="source_to_route_observation",
                                claim_status="carrier_trend",
                                metadata={"locus": locus, "site": site, "step": step},
                            )
                        for top in summary.get("top_tokens") or []:
                            if not isinstance(top, Mapping):
                                continue
                            position = int(top.get("position", -1))
                            for (
                                span_condition,
                                _locus,
                                span_position,
                            ), position_id in position_ids.items():
                                if span_condition == condition and span_position == position:
                                    _edge(
                                        edges,
                                        position_id,
                                        observation_id,
                                        kind="top_token_in_route_delta",
                                        claim_status="observed",
                                        metadata={"l2": top.get("l2"), "site": site, "step": step},
                                    )

    qkv_projection_ids: dict[tuple[str, str], str] = {}
    qkv_observation_ids: dict[tuple[str, str, str, int], str] = {}
    qkv_rows = report.get("qkv_source_rows") or {}
    if isinstance(qkv_rows, Mapping):
        for condition, raw_sites in sorted(qkv_rows.items()):
            if not isinstance(raw_sites, Mapping):
                continue
            condition = str(condition)
            for site, raw_kinds in sorted(
                raw_sites.items(), key=lambda item: _site_sort_key(str(item[0]))
            ):
                if not isinstance(raw_kinds, Mapping):
                    continue
                site = str(site)
                for kind, raw_steps in sorted(raw_kinds.items()):
                    if not isinstance(raw_steps, Mapping):
                        continue
                    kind = str(kind)
                    projection_key = (site, kind)
                    projection_id = qkv_projection_ids.get(projection_key)
                    if projection_id is None:
                        projection_id = (
                            f"{prefix}qkv-projection::{_id_part(site)}"
                            f"::{_id_part(kind)}"
                        )
                        qkv_projection_ids[projection_key] = projection_id
                        _node(
                            nodes,
                            projection_id,
                            kind="carrier",
                            modality="diffusion",
                            claim_status="typed_observation",
                            metadata={"site": site, "qkv": kind, "instrument": "native text QKV"},
                            evidence=[{"source_schema": report.get("schema")}],
                        )
                    for raw_step, rows in sorted(raw_steps.items(), key=lambda item: int(item[0])):
                        if not isinstance(rows, Mapping):
                            continue
                        step = int(raw_step)
                        observation_id = (
                            f"{prefix}qkv-observation::{_id_part(condition)}"
                            f"::{_id_part(site)}::{_id_part(kind)}::{step}"
                        )
                        qkv_observation_ids[(condition, site, kind, step)] = observation_id
                        _node(
                            nodes,
                            observation_id,
                            kind="route_observation",
                            modality="diffusion",
                            claim_status="typed_observation",
                            metadata={
                                "condition": condition,
                                "site": site,
                                "qkv": kind,
                                "step": step,
                                "row_count": len(rows),
                                "rows": dict(rows),
                                "tensor_payload_embedded": False,
                            },
                            evidence=[
                                {
                                    "source_schema": report.get("schema"),
                                    "condition": condition,
                                    "site": site,
                                    "qkv": kind,
                                    "step": step,
                                }
                            ],
                        )
                        _edge(
                            edges,
                            projection_id,
                            observation_id,
                            kind="emits_qkv_observation",
                            claim_status="typed_observation",
                            metadata={"condition": condition, "site": site, "step": step},
                        )
                        for row_key, raw_row in rows.items():
                            if not isinstance(raw_row, Mapping):
                                continue
                            locus = str(raw_row.get("locus") or str(row_key).split(":", 1)[0])
                            position = int(raw_row.get("position", -1))
                            position_id = position_ids.get((condition, locus, position))
                            if position_id:
                                _edge(
                                    edges,
                                    position_id,
                                    observation_id,
                                    kind="read_by_qkv_projection",
                                    claim_status="observed",
                                    metadata={
                                        "site": site,
                                        "qkv": kind,
                                        "step": step,
                                        "l2": raw_row.get("l2"),
                                        "rms": raw_row.get("rms"),
                                        "fingerprint": raw_row.get("fingerprint"),
                                    },
                                )
                        route_observation_id = route_observation_ids.get(
                            (condition, "text", site, step)
                        )
                        if route_observation_id:
                            _edge(
                                edges,
                                observation_id,
                                route_observation_id,
                                kind="qkv_to_route_observation",
                                claim_status="typed_observation",
                                metadata={
                                    "sufficiency_is_not_inferred": True,
                                    "site": site,
                                    "step": step,
                                },
                            )

    return_consumer = f"{prefix}consumer::return_register"
    rgb_consumer = f"{prefix}consumer::rgb_render"
    for consumer_id, consumer in (
        (return_consumer, "latent return register"),
        (rgb_consumer, "RGB render / VAE readout"),
    ):
        _node(
            nodes,
            consumer_id,
            kind="consumer" if consumer_id == return_consumer else "readout",
            modality="diffusion",
            claim_status="native_consumer",
            metadata={"consumer": consumer, "source": "joint token route capture"},
            evidence=[{"source_schema": report.get("schema")}],
        )

    intervention_ids: list[str] = []

    def add_consumer_edges(
        intervention_id: str,
        metrics: Mapping[str, Any],
        *,
        arm: str,
    ) -> None:
        claim_status = _consumer_claim(metrics)
        for consumer_id, metric_key in (
            (return_consumer, "return_alignment_to_donor"),
            (rgb_consumer, "rgb_alignment_to_donor"),
        ):
            _edge(
                edges,
                intervention_id,
                consumer_id,
                kind="consumer_closure_observation",
                claim_status=claim_status,
                metadata={
                    "arm": arm,
                    "metric": metric_key,
                    "value": metrics.get(metric_key),
                    "consumer_metrics": dict(metrics),
                },
                evidence=[{"source_schema": report.get("schema"), "arm": arm}],
            )

    for index, raw in enumerate(report.get("site_interventions") or []):
        if not isinstance(raw, Mapping):
            continue
        pair = str(raw.get("pair") or "unknown")
        site = str(raw.get("site") or "unknown")
        intervention_id = (
            f"{prefix}site-intervention::{index}::{_id_part(pair)}::{_id_part(site)}"
        )
        metrics = raw.get("consumer") if isinstance(raw.get("consumer"), Mapping) else {}
        _node(
            nodes,
            intervention_id,
            kind="counterfactual",
            modality="diffusion",
            claim_status=_consumer_claim(metrics),
            metadata={
                "arm": "whole_text_state_site_transfer",
                "pair": pair,
                "source_role": raw.get("source_role"),
                "site": site,
                "stream": raw.get("stream"),
                "dose": raw.get("dose"),
                "consumer": dict(metrics),
            },
            evidence=[{"source_schema": report.get("schema"), "index": index}],
        )
        intervention_ids.append(intervention_id)
        condition_id = condition_ids.get(pair)
        site_id = route_site_ids.get(site)
        if condition_id:
            _edge(
                edges,
                condition_id,
                intervention_id,
                kind="source_site_intervention",
                claim_status="causal_candidate",
                metadata={"pair": pair, "site": site},
            )
        if site_id:
            _edge(
                edges,
                site_id,
                intervention_id,
                kind="intervention_at_route_edge",
                claim_status="causal_candidate",
                metadata={"pair": pair, "site": site, "stream": raw.get("stream")},
            )
        add_consumer_edges(intervention_id, metrics, arm="whole_text_state_site_transfer")

    for index, raw in enumerate(report.get("qkv_interventions") or []):
        if not isinstance(raw, Mapping):
            continue
        arm = str(raw.get("arm") or "unknown")
        site = str(raw.get("site") or "unknown")
        qkv_value = str(raw.get("qkv") or "")
        intervention_id = (
            f"{prefix}qkv-intervention::{index}::{_id_part(arm)}::{_id_part(site)}"
        )
        metrics = raw.get("consumer") if isinstance(raw.get("consumer"), Mapping) else {}
        _node(
            nodes,
            intervention_id,
            kind="counterfactual",
            modality="diffusion",
            claim_status=_consumer_claim(metrics),
            metadata={
                "arm": arm,
                "source_pair": raw.get("source_pair"),
                "target_locus": raw.get("target_locus"),
                "site": site,
                "qkv": qkv_value,
                "dose": raw.get("dose"),
                "consumer": dict(metrics),
            },
            evidence=[{"source_schema": report.get("schema"), "index": index}],
        )
        intervention_ids.append(intervention_id)
        source_pair = str(raw.get("source_pair") or "base")
        source_condition_id = condition_ids.get(source_pair)
        if source_condition_id:
            _edge(
                edges,
                source_condition_id,
                intervention_id,
                kind="qkv_source_intervention",
                claim_status="causal_candidate",
                metadata={"arm": arm, "site": site, "qkv": qkv_value},
            )
        target_locus = str(raw.get("target_locus") or "")
        if "<->" in target_locus:
            target_loci = [part for part in target_locus.split("<->") if part]
        else:
            target_loci = [target_locus] if target_locus else []
        for locus in target_loci:
            for position_key, position_id in position_ids.items():
                condition, position_locus, _position = position_key
                if condition == "base" and position_locus == locus:
                    _edge(
                        edges,
                        position_id,
                        intervention_id,
                        kind="targeted_token_position_intervention",
                        claim_status="causal_candidate",
                        metadata={"arm": arm, "site": site, "qkv": qkv_value},
                    )
        kinds = (
            [part for part in qkv_value.split("+") if part]
            if "+" in qkv_value
            else [qkv_value]
        )
        for kind in kinds:
            for step in range(int(recipe.get("steps") or 0)):
                stage_id = qkv_observation_ids.get(("base", site, kind, step))
                if stage_id:
                    _edge(
                        edges,
                        stage_id,
                        intervention_id,
                        kind="qkv_intervention_at_stage",
                        claim_status="causal_candidate",
                        metadata={"arm": arm, "site": site, "qkv": kind, "step": step},
                    )
        add_consumer_edges(intervention_id, metrics, arm=arm)

    return {
        "capture_id": capture_id,
        "condition_ids": list(condition_ids.values()),
        "span_ids": list(span_ids.values()),
        "route_site_ids": list(route_site_ids.values()),
        "consumer_ids": [return_consumer, rgb_consumer],
        "intervention_ids": intervention_ids,
        "counts": {
            "conditions": len(condition_ids),
            "token_spans": len(span_ids),
            "token_positions": len(position_ids),
            "lexemes": len(lexeme_ids),
            "route_sites": len(route_site_ids),
            "route_steps": len(route_step_ids),
            "route_observations": len(route_observation_ids),
            "qkv_observations": len(qkv_observation_ids),
            "interventions": len(intervention_ids),
            "site_interventions": len(report.get("site_interventions") or []),
            "qkv_interventions": len(report.get("qkv_interventions") or []),
        },
    }


def _add_future_forest(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    manifest: Mapping[str, Any] | None,
    head: Mapping[str, Any] | None,
    decision: Mapping[str, Any] | None,
) -> None:
    if manifest is None:
        return
    forest_id = str(manifest.get("forest_id") or "unknown")
    forest_node = f"forest::{forest_id}"
    _node(
        nodes,
        forest_node,
        kind="future_forest",
        modality="control",
        claim_status="future_lineage",
        metadata={"schema": manifest.get("schema"), "storage": manifest.get("storage") or {}},
        evidence=[{"source_schema": manifest.get("schema")}],
    )
    if head:
        node_id = str(head.get("node_id") or "unknown")
        head_node = f"forest::head::{node_id}"
        _node(
            nodes,
            head_node,
            kind="accepted_future",
            modality="control",
            claim_status="future_lineage",
            metadata={
                "generation": head.get("generation"),
                "name": head.get("name"),
                "reason": head.get("reason"),
            },
            evidence=[{"source_schema": head.get("schema"), "node_id": node_id}],
        )
        _edge(edges, forest_node, head_node, kind="accepted_head", claim_status="future_lineage")
    if decision:
        decision_id = str(decision.get("content_sha256") or decision.get("node_id") or "unknown")
        decision_node = f"forest::decision::{decision_id}"
        _node(
            nodes,
            decision_node,
            kind="future_decision",
            modality="control",
            claim_status="future_lineage",
            metadata={
                "decision": decision.get("decision"),
                "accepted_before": decision.get("accepted_before"),
                "accepted_after": decision.get("accepted_after"),
                "observation_ids": decision.get("observation_ids") or [],
                "reason": decision.get("reason"),
            },
            evidence=[
                {"source_schema": decision.get("schema"), "decision": decision.get("decision")}
            ],
        )
        _edge(
            edges,
            forest_node,
            decision_node,
            kind="records_decision",
            claim_status="future_lineage",
        )
        if head:
            _edge(
                edges,
                decision_node,
                f"forest::head::{head.get('node_id') or 'unknown'}",
                kind="promotes_future",
                claim_status="future_lineage",
                metadata={"decision": decision.get("decision")},
            )


def build_thought_route_graph(
    *,
    thott_trace: Mapping[str, Any] | None = None,
    qwen_report: Mapping[str, Any] | None = None,
    qwen_raw_manifest: Mapping[str, Any] | None = None,
    joint_token_report: Mapping[str, Any] | None = None,
    route_confirmation: Mapping[str, Any] | None = None,
    foretelling_predictors: Mapping[str, Any] | None = None,
    foretelling_predictions: Mapping[str, Any] | None = None,
    future_forest_manifest: Mapping[str, Any] | None = None,
    future_forest_head: Mapping[str, Any] | None = None,
    future_forest_decision: Mapping[str, Any] | None = None,
    input_refs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the combined artifact-only graph from retained JSON receipts."""

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    _grammar(nodes, edges)
    thott_ids = _add_thott_trace(nodes, edges, thott_trace) if thott_trace else []
    qwen_ids = _add_qwen_summary(nodes, edges, qwen_report, qwen_raw_manifest)
    flux_ids = _add_flux_routes(
        nodes,
        edges,
        route_confirmation,
        foretelling_predictors,
        foretelling_predictions,
    )
    joint_summary = (
        _add_joint_token_route(nodes, edges, joint_token_report)
        if joint_token_report
        else None
    )
    _add_future_forest(
        nodes,
        edges,
        future_forest_manifest,
        future_forest_head,
        future_forest_decision,
    )

    paths: list[dict[str, Any]] = []
    if thott_ids:
        paths.append(
            {
                "name": "thott_typed_execution",
                "claim_status": "typed_observation",
                "nodes": thott_ids,
                "description": "Thott physical nodes; event chronology is retained separately.",
            }
        )
    if qwen_ids:
        paths.append(
            {
                "name": "qwen_workspace_to_continuation",
                "claim_status": "typed_observation",
                "nodes": qwen_ids,
                "description": (
                    "Qwen shared-workspace summary path; task labels are external prompt labels."
                ),
            }
        )
    if flux_ids:
        paths.append(
            {
                "name": "flux_temporal_route_sites",
                "claim_status": "temporal_route_observation",
                "nodes": flux_ids,
                "description": (
                    "FLUX route-site chronology; consumer edges retain domain-level metrics."
                ),
            }
        )
    if joint_summary:
        paths.append(
            {
                "name": "joint_token_to_consumer",
                "claim_status": "typed_observation",
                "nodes": [
                    joint_summary["capture_id"],
                    *joint_summary["span_ids"],
                    *joint_summary["route_site_ids"],
                    *joint_summary["consumer_ids"],
                ],
                "description": (
                    "Native lexical span/position coordinates joined to route sites, QKV "
                    "observations, interventions, and frozen downstream consumers."
                ),
            }
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "artifact_only",
        "artifact_only": True,
        "certified_causality": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "execution": {
            "model_loads": 0,
            "mrun_leases": 0,
            "tensor_payloads_embedded": False,
            "mode": "retained-artifact-join",
        },
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
        "paths": paths,
        "inputs": {"artifacts": [_safe(item) for item in input_refs]},
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "path_count": len(paths),
            "modalities": sorted({str(row["modality"]) for row in nodes.values()}),
            "node_kinds": {
                kind: sum(row["kind"] == kind for row in nodes.values())
                for kind in sorted({str(row["kind"]) for row in nodes.values()})
            },
            "edge_kinds": {
                kind: sum(row["kind"] == kind for row in edges.values())
                for kind in sorted({str(row["kind"]) for row in edges.values()})
            },
            "claim_statuses": {
                status: sum(
                    row["claim_status"] == status for row in [*nodes.values(), *edges.values()]
                )
                for status in sorted(
                    {str(row["claim_status"]) for row in [*nodes.values(), *edges.values()]}
                )
            },
        },
    }
    if joint_summary:
        result["summary"]["joint_token_route"] = joint_summary["counts"]
    result["content_sha256"] = _digest(result)
    return result


def validate_thought_route_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the graph seal and endpoint references without loading a model."""

    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported thought-route graph schema")
    declared = str(payload.get("content_sha256") or "")
    if len(declared) != 64:
        raise ValueError("thought-route graph has no content seal")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed = _digest(body)
    if computed != declared:
        raise ValueError("thought-route graph content seal mismatch")
    node_ids = [str(row.get("node_id") or "") for row in payload.get("nodes", ())]
    if any(not node_id for node_id in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ValueError("thought-route graph node IDs must be unique and non-empty")
    node_set = set(node_ids)
    for edge in payload.get("edges", ()):
        if edge.get("source") not in node_set or edge.get("target") not in node_set:
            raise ValueError("thought-route graph edge references an unknown node")
    return {
        "schema": SCHEMA,
        "content_sha256": declared,
        "node_count": len(node_ids),
        "edge_count": len(payload.get("edges", ())),
        "endpoint_integrity": True,
    }


def _dot_text(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return text[:180]


def _compact_dot_label(raw: Mapping[str, Any]) -> str:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
    fields = (
        "condition",
        "locus",
        "site",
        "step",
        "stream",
        "qkv",
        "arm",
        "pair",
        "domain",
        "gate",
        "token",
        "position",
    )
    details = [f"{field}={metadata[field]}" for field in fields if field in metadata]
    if not details:
        node_parts = str(raw.get("node_id") or "").split("::")
        details = node_parts[-2:]
    return "\n".join([str(raw.get("kind") or "unknown"), *details[:4]])


def render_dot(payload: Mapping[str, Any], *, compact_labels: bool = False) -> str:
    lines = [
        "digraph ThoughtRouteGraph {",
        "  rankdir=LR;",
        "  graph [fontname=Helvetica, nodesep=0.15, ranksep=0.4];",
        "  node [fontname=Helvetica, fontsize=8];",
        "  edge [fontname=Helvetica, fontsize=7];",
    ]
    shapes = {
        "abstract_role": "box",
        "producer": "ellipse",
        "carrier": "box",
        "consumer": "octagon",
        "readout": "doublecircle",
        "event": "note",
        "counterfactual": "diamond",
        "contrast": "diamond",
        "candidate_operation": "diamond",
        "route_site": "box",
        "route_step": "box",
        "token_position": "box",
        "task_variable": "ellipse",
        "evidence_gate": "note",
        "route_observation": "note",
        "forecaster": "component",
        "forecast": "note",
        "future_forest": "folder",
        "accepted_future": "folder",
        "future_decision": "diamond",
        "artifact": "folder",
    }
    for raw in payload.get("nodes", ()):
        node_id = _dot_text(raw.get("node_id"))
        kind = str(raw.get("kind") or "unknown")
        label = (
            _compact_dot_label(raw)
            if compact_labels
            else f"{raw.get('node_id')}\\n[{kind}]\\n{raw.get('claim_status')}"
        )
        lines.append(
            f'  "{node_id}" [label="{_dot_text(label)}", shape={shapes.get(kind, "box")}];'
        )
    for raw in payload.get("edges", ()):
        source = _dot_text(raw.get("source"))
        target = _dot_text(raw.get("target"))
        label = (
            str(raw.get("kind"))
            if compact_labels
            else f"{raw.get('kind')}\\n{raw.get('claim_status')}"
        )
        lines.append(f'  "{source}" -> "{target}" [label="{_dot_text(label)}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def compact_graph_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a human-sized causal skeleton without deleting the full artifact.

    Forecast rows are useful provenance but create a large fan-out in a first
    visual.  The compact view retains route observations and the two most
    informative gate families: whole-edge closure and local Q/K/V sufficiency.
    """

    kept_nodes: list[Mapping[str, Any]] = []
    kept_ids: set[str] = set()
    for row in payload.get("nodes", ()):
        node_id = str(row.get("node_id") or "")
        kind = str(row.get("kind") or "")
        if kind in {"forecast", "forecaster"}:
            continue
        if node_id.startswith("joint::") and kind in {"route_observation", "route_step"}:
            continue
        if kind == "evidence_gate" and not (
            node_id.endswith("whole_edge_site_closure") or "qkv_sufficiency" in node_id
        ):
            continue
        kept_nodes.append(row)
        kept_ids.add(node_id)
    kept_edges = [
        row
        for row in payload.get("edges", ())
        if str(row.get("source") or "") in kept_ids
        and str(row.get("target") or "") in kept_ids
        and row.get("kind") not in {"predicts", "forecasts_candidate"}
    ]
    compact = {
        "schema": SCHEMA,
        "view": "causal_skeleton",
        "source_content_sha256": payload.get("content_sha256"),
        "claim_boundary": (
            "Compact visualization of the full artifact. Hidden forecast rows remain in the "
            "source graph and are not discarded."
        ),
        "nodes": sorted(kept_nodes, key=lambda row: str(row.get("node_id"))),
        "edges": sorted(kept_edges, key=lambda row: str(row.get("edge_id"))),
        "summary": {"node_count": len(kept_nodes), "edge_count": len(kept_edges)},
    }
    compact["content_sha256"] = _digest(compact)
    return compact


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# ThoughtRouteGraph artifact",
        "",
        "This is an artifact-only join of Thott execution evidence, temporal route evidence, "
        "foretelling, and Future Forest lineage.",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- nodes / edges: `{summary.get('node_count', 0)} / {summary.get('edge_count', 0)}`",
        f"- paths: `{summary.get('path_count', 0)}`",
        f"- modalities: `{', '.join(summary.get('modalities', ()))}`",
        f"- content SHA-256: `{payload.get('content_sha256')}`",
        "",
        payload.get("claim_boundary", ""),
        "",
        "## Combined execution grammar",
        "",
        "```text",
        "task/source hypothesis → address/operation → carrier → native consumer → readout",
        "                                             └→ intervention → future branch",
        "```",
        "",
        "## Paths",
        "",
        "| path | claim status | node count |",
        "|---|---|---:|",
    ]
    for path in payload.get("paths", ()):
        lines.append(
            f"| `{path.get('name')}` | `{path.get('claim_status')}` | "
            f"{len(path.get('nodes', ()))} |"
        )
    joint = summary.get("joint_token_route")
    if isinstance(joint, Mapping):
        lines.extend(
            [
                "",
                "## Lexical route capture",
                "",
                "| measure | count |",
                "|---|---:|",
            ]
        )
        for key in (
            "conditions",
            "token_spans",
            "token_positions",
            "lexemes",
            "route_sites",
            "route_steps",
            "route_observations",
            "qkv_observations",
            "site_interventions",
            "qkv_interventions",
        ):
            if key in joint:
                lines.append(f"| `{key}` | {joint[key]} |")
        lines.extend(
            [
                "",
                "The lexical capture keeps token identity, source position, route site, Q/K/V "
                "projection, intervention arm, and native consumer metrics separate. Local QKV "
                "geometry is not treated as sufficiency without consumer closure.",
            ]
        )
    candidate_nodes = [
        row for row in payload.get("nodes", ()) if row.get("kind") == "candidate_operation"
    ]
    candidate_edges = [
        row for row in payload.get("edges", ()) if row.get("kind") == "candidate_consumer_test"
    ]
    lines.extend(
        [
            "",
            "## Capability-route panel",
            "",
            "| domain | proposed site/QKV | local return alignment | "
            "local RGB alignment | local status |",
            "|---|---|---:|---:|---|",
        ]
    )
    for candidate in sorted(candidate_nodes, key=lambda row: str(row.get("node_id"))):
        metadata = candidate.get("metadata") or {}
        domain = str(metadata.get("domain") or candidate.get("node_id"))
        site_qkv = f"{metadata.get('site', '?')} / {metadata.get('qkv', '?')}"
        local = [row for row in candidate_edges if row.get("source") == candidate.get("node_id")]
        by_target = {str(row.get("target")): row for row in local}
        return_row = by_target.get("flux::consumer::return_register") or {}
        rgb_row = by_target.get("flux::consumer::rgb_render") or {}
        return_value = (
            (return_row.get("metadata") or {})
            .get("observed", {})
            .get("mean_endpoint_return_alignment")
        )
        rgb_value = (
            (rgb_row.get("metadata") or {}).get("observed", {}).get("mean_endpoint_rgb_alignment")
        )
        status = str(return_row.get("claim_status") or "unmeasured")
        lines.append(
            f"| `{domain}` | `{site_qkv}` | "
            f"{return_value if isinstance(return_value, (int, float)) else '—'} | "
            f"{rgb_value if isinstance(rgb_value, (int, float)) else '—'} | `{status}` |"
        )
    whole_gate = next(
        (
            row
            for row in payload.get("nodes", ())
            if row.get("node_id") == "flux::gate::whole_edge_site_closure"
        ),
        None,
    )
    if whole_gate is not None:
        lines.extend(
            [
                "",
                "Whole-route closure gate: "
                f"`{(whole_gate.get('metadata') or {}).get('passed')}`. "
                "This does not promote the local Q/K/V endpoint to sufficiency.",
            ]
        )
    lines.extend(["", "## Evidence-status counts", "", "| status | count |", "|---|---:|"])
    for status, count in sorted((summary.get("claim_statuses") or {}).items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Thott nodes/events remain physical execution evidence.",
            "- Route-site closure remains separate from local Q/K/V sufficiency.",
            "- Forecasts are retained as unopened predictions, not outcomes.",
            "- Future Forest records branch lineage and promotion, not semantic truth.",
            "",
        ]
    )
    return "\n".join(lines)


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"graph input must be a JSON object: {path}")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join Thott and Saturn route evidence into a graph."
    )
    parser.add_argument("--thott-trace", type=Path)
    parser.add_argument("--qwen-report", type=Path)
    parser.add_argument("--qwen-raw-manifest", type=Path)
    parser.add_argument("--joint-token-report", type=Path)
    parser.add_argument("--joint-token-raw", type=Path)
    parser.add_argument("--route-confirmation", type=Path)
    parser.add_argument("--foretelling-predictors", type=Path)
    parser.add_argument("--foretelling-predictions", type=Path)
    parser.add_argument("--future-forest-manifest", type=Path)
    parser.add_argument("--future-forest-head", type=Path)
    parser.add_argument("--future-forest-decision", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--focus-dot", type=Path)
    args = parser.parse_args(argv)
    paths = [
        value
        for value in (
            args.thott_trace,
            args.qwen_report,
            args.qwen_raw_manifest,
            args.joint_token_report,
            args.joint_token_raw,
            args.route_confirmation,
            args.foretelling_predictors,
            args.foretelling_predictions,
            args.future_forest_manifest,
            args.future_forest_head,
            args.future_forest_decision,
        )
        if value is not None
    ]
    result = build_thought_route_graph(
        thott_trace=_load(args.thott_trace),
        qwen_report=_load(args.qwen_report),
        qwen_raw_manifest=_load(args.qwen_raw_manifest),
        joint_token_report=_load(args.joint_token_report),
        route_confirmation=_load(args.route_confirmation),
        foretelling_predictors=_load(args.foretelling_predictors),
        foretelling_predictions=_load(args.foretelling_predictions),
        future_forest_manifest=_load(args.future_forest_manifest),
        future_forest_head=_load(args.future_forest_head),
        future_forest_decision=_load(args.future_forest_decision),
        input_refs=[artifact_ref(path) for path in paths],
    )
    validate_thought_route_graph(result)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "thought-route-graph.json"
    markdown_path = out_dir / "thought-route-graph.md"
    dot_path = out_dir / "thought-route-graph.dot"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    dot_path.write_text(render_dot(result), encoding="utf-8")
    if args.focus_dot:
        focus_path = args.focus_dot.expanduser().resolve()
        focus_path.parent.mkdir(parents=True, exist_ok=True)
        focus_path.write_text(
            render_dot(compact_graph_view(result), compact_labels=True), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(markdown_path),
                "dot": str(dot_path),
                "summary": result["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA",
    "artifact_ref",
    "build_thought_route_graph",
    "compact_graph_view",
    "validate_thought_route_graph",
    "render_dot",
    "render_markdown",
]
