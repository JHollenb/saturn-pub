"""Attach verified attention observations to canonical native symbol records.

Metadata qualification does not hydrate a model or grant writes. The caller
verifies external array bytes; native cut/receipt support is checked here and
again by Project.save_symbol_lifecycle. Semantic names remain supplied.
"""
from __future__ import annotations

from dataclasses import replace
import math

from saturn_pub.contracts import StateAddress
from saturn_pub.symbols import SymbolBinding
from saturn_pub.values import digest

from .attention_port import kv_group
from .project_slice import slot_descriptors
from .symbol_lifecycle import ClockWitness, SymbolLifecycle, SymbolOccurrence, SymbolUse


def retained_binding(project, session_id, cut_id, name, locations, *, context, claim,
                     context_slots=("tokens",), evidence=()):
    """Qualify supplied locations against sealed retained native metadata."""
    cut = project.cut_descriptor(session_id, cut_id)
    point, schema = digest(cut["execution_point"]), cut["surface"]["state_schema"]
    qualified = []
    for loc in locations:
        if loc.execution_point not in (None, point) or loc.state_schema not in (None, schema):
            raise ValueError("supplied location belongs to another native point or schema")
        qualified.append(StateAddress(loc.slot, loc.selector, loc.role, point, schema))
    slots = slot_descriptors(cut)
    binding = SymbolBinding(name, cut["model_identity"], digest(cut["manifest"]["execution"]),
        context, tuple((slot, digest(slots[slot])) for slot in context_slots), tuple(qualified),
        tuple(dict.fromkeys(("cas://sha256/" + cut_id, *evidence))), claim, status="observed")
    binding.resolve_metadata(model_identity=cut["model_identity"], execution=cut["manifest"]["execution"],
        execution_point=cut["execution_point"], surface=cut["surface"], slots=slots, context=context)
    return binding


def definition_at(project, session_id, cut_id, binding, *, metadata=None, definition_refs=(), receipt=None):
    cut = project.cut_descriptor(session_id, cut_id)
    binding.resolve_metadata(model_identity=cut["model_identity"], execution=cut["manifest"]["execution"],
        execution_point=cut["execution_point"], surface=cut["surface"],
        slots=slot_descriptors(cut), context=binding.context)
    ref = "cas://sha256/" + cut_id
    clock = ClockWitness("native-point-" + binding.to_dict()["fingerprint"], "execution-point",
        session_id + "/main", cut["execution_point"], provenance=(ref,), status="observed",
        metadata={"source_cut": cut_id, "model_execution": binding.execution})
    occurrence = SymbolOccurrence.for_binding(binding,
        occurrence_id="definition-" + binding.to_dict()["fingerprint"], session_id=session_id,
        source_cut=cut_id, locations=binding.locations, branch="main", execution=binding.execution,
        receipt=receipt, clock=clock, definition_refs=tuple(definition_refs), evidence_refs=(ref,),
        status="observed", metadata=metadata or {})
    return SymbolLifecycle.define(binding, definitions=(occurrence,), mechanics_status="valid",
        instrument_status="observations-only", trend_status="not-assessed", terminal_status="not-assessed",
        claim_boundary="Verified native metadata and observational attention; supplied semantic names; head necessity untested.")


def attach_attention(project, session_id, lifecycle, *, receipt_id, trace, heads,
                     source_row, output_binding, metadata=None):
    """Join query-head allocation to existing K/V locations and native readout.

    One lifecycle represents one layer at one source boundary. Each query head
    gets its own SymbolUse over the correct grouped-query K/V locations. Raw
    Q/K/V/A/head-return descriptors are retained once at lifecycle level.
    """
    parent = lifecycle.definitions[0].source_cut
    receipts = {r.fingerprint: r.to_dict() for r in project.receipts(session_id)}
    if receipt_id not in receipts:
        raise ValueError("attention receipt is not retained in this native session")
    receipt = receipts[receipt_id]
    if receipt["operation"] != "continue" or receipt["parent"] != parent:
        raise ValueError("attention observation differs from its native parent")
    if not any(loc.slot == "logits" for loc in output_binding.locations):
        raise ValueError("attention output must resolve to the native logits port")
    result = project.cut_descriptor(session_id, receipt["result"])
    output_binding.resolve_metadata(model_identity=result["model_identity"], execution=result["manifest"]["execution"],
        execution_point=result["execution_point"], surface=result["surface"],
        slots=slot_descriptors(result), context=output_binding.context)
    layer = trace["layer"]
    shapes = {key: trace[key]["shape"] for key in ("query", "key", "value", "attention", "head_return")}
    b, count, q, n = shapes["attention"]
    kv = shapes["key"][1]
    width = shapes["query"][-1]
    if (b != 1 or q != 1 or not 0 <= source_row < n or count % kv
            or shapes["query"] != [1, count, 1, width]
            or shapes["key"] != [1, kv, n, width] or shapes["value"] != shapes["key"]
            or shapes["head_return"] != [1, 1, count, width]
            or sorted(h["head"] for h in heads) != list(range(count))):
        raise ValueError("attention observation has inconsistent head/row geometry")
    for key in shapes:
        sha = trace[key]["sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("attention artifact needs an immutable SHA256")
    transitions = [(i, t) for i, t in enumerate(receipt["transitions"])
        if t["before"].get("phase") == "layer" and t["before"].get("index") == layer]
    if len(transitions) != 1:
        raise ValueError("attention layer has no unique native transition")
    index, transition = transitions[0]
    operation = transition["native_operation"]
    if not {"keys", "values"} <= set(operation["reads"]):
        raise ValueError("native attention transition does not declare K/V reads")
    by_location = {(loc.slot, loc.selector): loc for loc in lifecycle.binding.locations}
    trace_ref = digest(trace)
    uses = []
    for row in heads:
        head = row["head"]
        group = kv_group(head, count, kv)
        if row["kv_head"] != group:
            raise ValueError("query head was linked to the wrong KV group")
        selectors = [(slot, (layer, 0, group, source_row)) for slot in ("keys", "values")]
        if any(selector not in by_location for selector in selectors):
            raise ValueError("attention source locations are outside the symbol binding")
        metrics = {key: row[key] for key in ("source_mass", "entropy_nats", "source_value_return_norm")}
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in metrics.values()):
            raise ValueError("attention observation contains invalid numeric evidence")
        use = SymbolUse(f"attention-{receipt_id}-L{layer}-H{head}", lifecycle.binding_fingerprint,
            session_id, operation["consumer"], source_cut=parent, parent_cut=parent,
            result_cut=receipt["result"], branch="main", execution=lifecycle.binding.execution,
            receipt=receipt_id, locations=tuple(by_location[s] for s in selectors),
            status="observed", evidence_refs=("cas://sha256/" + trace["attention"]["sha256"],
                "cas://sha256/" + receipt_id),
            signed_observations=tuple({"observable": k, "value": v, "n": 1} for k, v in metrics.items()),
            metadata={"query_head": head, "kv_head": group, "source_row": source_row,
                "receiver_row": transition["before"]["logical_step"],
                "native_transition_index": index, "native_operation_point": transition["before"],
                "attention_selector": [0, head, 0, source_row], "trace_fingerprint": trace_ref,
                "output_binding": output_binding.to_dict()["fingerprint"],
                "interpretation": "observed attention use; individual causal effect untested"})
        uses.append(use)
    return replace(lifecycle, uses=tuple(uses), metadata={**dict(metadata or {}),
        "attention_trace": trace, "trace_fingerprint": trace_ref,
        "output_binding": output_binding.to_dict()["fingerprint"]})
