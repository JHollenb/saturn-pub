"""Record live public Sessions in MDB's canonical symbol/lifecycle indexes.

Names and claims are supplied by the instrument. Exact addresses, clocks, cuts,
and consumer receipts come from the native Session. This grants no write power.
"""
from __future__ import annotations

from dataclasses import replace

from saturn_pub.symbols import SymbolBinding

from .symbol_lifecycle import ClockWitness, SymbolLifecycle, SymbolOccurrence, SymbolUse


class LiveSymbols:
    """Explicit observations and uses over a persistent SaturnPubDebuggerAdapter."""

    def __init__(self, adapter):
        if adapter.project is None:
            raise ValueError("live symbols require a persistent MDB Project")
        self.adapter = adapter
        self.records = {}

    def observe(self, name, locations, *, context, claim, evidence=(), metadata=None,
                context_slots=("tokens",)):
        adapter = self.adapter
        cut = adapter.checkpoint()["cut_id"]
        ref = "cas://sha256/" + cut
        binding = SymbolBinding.bind(adapter.session, name, tuple(locations), context=context,
            context_slots=tuple(context_slots), evidence=tuple(dict.fromkeys((ref, *evidence))),
            claim=claim, status="observed")
        fingerprint = binding.to_dict()["fingerprint"]
        if fingerprint in self.records:
            return self.records[fingerprint]
        clock = ClockWitness("native-point-" + fingerprint, "execution-point",
            adapter.session_id + "/" + adapter.public.active,
            dict(adapter.session.inspect().execution_point), provenance=(ref,), status="observed",
            metadata={"source_cut": cut, "model_execution": binding.execution})
        occurrence = SymbolOccurrence.for_binding(binding,
            occurrence_id="definition-" + fingerprint, session_id=adapter.session_id,
            source_cut=cut, branch=adapter.public.active, execution=binding.execution,
            locations=binding.locations, clock=clock, status="observed",
            evidence_refs=(ref,), metadata=metadata or {})
        lifecycle = SymbolLifecycle.define(binding, definitions=(occurrence,),
            mechanics_status="valid", instrument_status="observations-only",
            trend_status="not-assessed", terminal_status="not-assessed",
            claim_boundary="Native locations and receipt associations; supplied semantic names.")
        adapter.project.save_symbol_lifecycle(adapter.session_id, lifecycle, include_slices=False)
        if fingerprint not in {row.to_dict()["fingerprint"] for row in adapter.symbols.bindings}:
            adapter.symbols.bindings.append(binding)
        adapter.symbol_context = context
        adapter.checkpoint()
        self.records[fingerprint] = lifecycle
        return lifecycle

    def advance(self, steps, *, symbols=(), observations=()):
        """Advance once, retaining the actual parent/result and canonical uses.

        A stale binding is rejected before execution. Completed native results
        are persisted even if indexing later fails. No inferred dataflow is
        promoted to a measured causal effect by this operation.
        """
        adapter = self.adapter
        current = []
        for lifecycle in symbols:
            key = lifecycle.binding_fingerprint
            lifecycle = self.records.get(key, lifecycle)
            lifecycle.binding.resolve(adapter.session, context=lifecycle.binding.context)
            if lifecycle.definitions[0].branch != adapter.public.active:
                raise ValueError("live symbol belongs to another branch")
            current.append(lifecycle)
        parent = adapter.checkpoint()["cut_id"]
        try:
            receipt = adapter.session.continue_(steps)
        finally:
            result = adapter.checkpoint()["cut_id"]
        body = receipt.to_dict()
        if body["parent"] != parent or body["result"] != result:
            raise ValueError("native symbol use lost parent/result closure")
        consumers = {row["native_operation"]["consumer"] for row in body["transitions"]}
        if len(consumers) != 1:
            raise ValueError("symbol use needs one declared native consumer")
        for lifecycle in current:
            use = SymbolUse("use-" + receipt.fingerprint + "-" + lifecycle.binding_fingerprint,
                lifecycle.binding_fingerprint, adapter.session_id, next(iter(consumers)),
                source_cut=lifecycle.definitions[0].source_cut, parent_cut=parent,
                result_cut=result, branch=adapter.public.active, receipt=receipt.fingerprint,
                execution=lifecycle.binding.execution, locations=lifecycle.binding.locations,
                status="observed", evidence_refs=("cas://sha256/" + receipt.fingerprint,),
                signed_observations=tuple(observations))
            updated = replace(lifecycle, uses=(*lifecycle.uses, use))
            adapter.project.save_symbol_lifecycle(adapter.session_id, updated, include_slices=False)
            self.records[lifecycle.binding_fingerprint] = updated
        return receipt
