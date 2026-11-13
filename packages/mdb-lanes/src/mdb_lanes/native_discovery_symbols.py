"""Anonymous component catalogs and public read-only symbols over native observations."""
from __future__ import annotations

import numpy as np

from .native_discovery import digest, save_json, sealed


def discover_components(model, discovery, *, root, predictor_identity, groups=8, accepted_updates=None):
    """Cluster learned formation/write signatures; examples come only from discovery."""
    import torch
    rows, signatures = [], []
    device = next(model.parameters()).device
    for stage, span in enumerate(model.spans):
        examples = [[] for _ in range(model.config["features"])]
        document_counts = np.zeros(model.config["features"], dtype=int)
        for record, arrays in discovery:
            with torch.inference_mode():
                code = span.encode(torch.tensor(arrays["states"][stage], device=device)).cpu().numpy()
            document_counts += (np.abs(code).max(0) > 0)
            for feature in range(code.shape[-1]):
                for pos in np.argsort(-np.abs(code[:, feature]))[:2]:
                    if code[pos, feature] == 0:
                        continue
                    examples[feature].append({"document": record["document_id"],
                        "position": int(pos), "activation": float(code[pos, feature]),
                        "context": record.get("token_pieces", [])[max(0, pos - 4):pos + 5]})
        encoder = span.encoder.weight.detach().cpu().numpy()
        decoder = span.local_write.weight.detach().cpu().numpy().T
        query = span.query.weight.detach().cpu().numpy().T
        key = span.key.weight.detach().cpu().numpy().T
        for feature in range(len(encoder)):
            parts = [v / max(float(np.linalg.norm(v)), 1e-12)
                     for v in (encoder[feature], decoder[feature], query[feature], key[feature])]
            signatures.append(np.concatenate(parts))
            rows.append({"id": "component_" + digest([predictor_identity, stage, feature])[:16],
                         "stage": stage, "feature": feature,
                         "observed_discovery_documents": int(document_counts[feature]),
                         "examples": sorted(examples[feature], key=lambda e: -abs(e["activation"]))[:6],
                         "description": None, "description_status": "anonymous; no inferred ontology",
                         "relationship_status": "learned computational signature; not causal necessity"})
    x = np.stack(signatures)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    count = min(groups, len(x))
    centers = [x[0]]
    for _ in range(1, count):
        distance = np.min(np.stack([np.square(x - c).sum(-1) for c in centers]), axis=0)
        centers.append(x[int(distance.argmax())])
    centers = np.stack(centers)
    for _ in range(12):
        assignment = (x @ centers.T).argmax(-1)
        for i in range(count):
            if np.any(assignment == i):
                centers[i] = x[assignment == i].mean(0)
                centers[i] /= max(float(np.linalg.norm(centers[i])), 1e-12)
    for row, group in zip(rows, assignment, strict=True):
        row["role_candidate"] = "role_" + digest([predictor_identity, int(group)])[:16]
    result = sealed({"schema": "mdb-native-discovery-v1/components",
        "predictor_identity": predictor_identity, "components": rows,
        "grouping": "spherical clustering of learned encoder/write/query/key signatures",
        "groups": count, "nonempty_groups": int(len(np.unique(assignment))),
        "groups_note": "groups = requested cluster count; nonempty_groups = clusters with at least one component",
        "fit_split": "discovery only", "semantic_ontology": "not supplied",
        "accepted_updates": accepted_updates,
        "representation_status": "initialization-only" if accepted_updates == 0 else "fitted candidate",
        "authority": "predictive component candidates; causal and semantic claims remain separate"})
    save_json(root / "components.json", result)
    return result


def register_observations(record, arrays, prediction, *, root, catalog, max_uses=24, reads=False):
    """Persist real SymbolTable/Lifecycle records over explicitly non-resumable cuts.

    These are captured observation surfaces, not fabricated live Session closures.
    Predicted source relevance is retained beside the actual native source mass.
    Records are built by the shared passive writer in ``mdb.symbol_reads``; the files written
    by default are byte-identical to the pre-consolidation writer. ``reads=True`` additionally
    writes ``reads.json`` (backend-neutral read records, causal status ``unknown``).
    """
    import torch
    from saturn_pub.core import Receipt, StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.symbols import SymbolTable
    from saturn_pub.values import describe
    from saturn_pub.values import digest as public_digest

    from .symbol_lifecycle import ClockWitness
    from .symbol_reads import (
        PassiveSurface,
        Reader,
        ReadSource,
        passive_binding,
        passive_definition,
        passive_lifecycle,
        passive_use,
        read_for_use,
        save_reads,
    )

    root.mkdir(parents=True, exist_ok=False)
    store = LocalStore(root / "store")
    identity = record["binding"]["model_identity"]
    execution = record["binding"]["execution"]
    document = record["document_id"]
    reference = "sha256:" + record["arrays"]["sha256"]
    schema = "mdb-native-observation-state-v1"
    observed_tokens = torch.from_numpy(arrays["tokens"].copy())
    observed_states = [torch.from_numpy(a.copy()) for a in arrays["states"]]
    cuts = []
    for stage, boundary in enumerate(record["boundaries"]):
        point = {"document": document, "boundary_layer": boundary,
                 "capture": record["fingerprint"]}
        cut = StateCut(identity, execution, f"observed-layer:{boundary}",
            cuts[-1].fingerprint if cuts else None,
            {"hidden": observed_states[stage], "tokens": observed_tokens},
            execution_point=point,
            surface={"state_schema": schema, "kind": "passive observation; not a resumable decoder",
                     "slots": [{"name": "hidden", "role": "observed-residual"},
                               {"name": "tokens", "role": "observed-token-ids"}]})
        cut.verify()
        store.save(cut)
        cuts.append(cut)
    token_slot = describe(observed_tokens)
    surfaces = [PassiveSurface.from_cut(cut, session_id=document, slots={
        "tokens": token_slot, "hidden": describe(observed_states[stage])})
        for stage, cut in enumerate(cuts)]
    rows, lifecycles, tables, read_records = [], [], [[] for _ in cuts], []
    feature_lookup = {(r["stage"], r["feature"]): r for r in catalog["components"]}
    stages, heads, length, _ = prediction["routes"].shape
    candidates = [(s, h, p) for s in range(stages) for h in range(heads) for p in range(1, length)]
    # Uniform budget over the full observed geometry; no selected semantic sites.
    chosen = np.linspace(0, len(candidates) - 1, min(max_uses, len(candidates)), dtype=int)
    abstained = 0
    for number, index in enumerate(chosen):
        stage, head, receiver = candidates[index]
        source = int(prediction["routes"][stage, head, receiver].argmax())
        if not np.any(prediction["code"][stage, source]):
            abstained += 1
            continue
        feature = int(np.abs(prediction["code"][stage, source]).argmax())
        component = feature_lookup[stage, feature]
        surface, end = surfaces[stage], cuts[stage + 1]
        # Check exact geometry without granting executable Session authority.
        binding = passive_binding(surface, component["id"],
            (surface.address("hidden", (source,), "observed-residual"),),
            context_slots=(("tokens", public_digest(token_slot)),),
            evidence=(reference, "sha256:" + catalog["fingerprint"]),
            claim="Automatically proposed source component; native location observed; semantic role unresolved")
        receipt = Receipt.make(operation="mdb.native_discovery.observed-span", status="observed",
            parent=surface.cut_ref, result=end.fingerprint,
            capture=record["fingerprint"], source=source, receiver=receiver, head=head,
            claim="native observations plus a previously sealed source forecast; not a causal test")
        store.receipt(receipt)
        clock = ClockWitness("clock_" + str(number), "layer", document,
            record["boundaries"][stage], provenance=(reference,), status="observed")
        occurrence = passive_definition(surface, binding, occurrence_id="definition_" + str(number),
            clock=clock, evidence_refs=(reference,),
            metadata={"component": component["id"], "role_candidate": component["role_candidate"],
                      "source_status": "learned nomination", "surface": "observation-only"})
        observations = {"predicted_source_mass": float(prediction["routes"][stage, head, receiver, source]),
                        "native_source_mass": float(arrays["routes"][stage, head, receiver, source]),
                        "native_top_source": int(arrays["routes"][stage, head, receiver].argmax())}
        span = f"native.layers.{record['boundaries'][stage]}:{record['boundaries'][stage + 1]}"
        use = passive_use(surface, binding, use_id="use_" + str(number), consumer=span,
            result_cut=end.fingerprint, receipt=receipt.fingerprint, evidence_refs=(reference,),
            signed_observations=(observations,))
        lifecycle = passive_lifecycle(binding, definitions=(occurrence,), uses=(use,),
            claim_boundary="Predicted bindings over passive native observations.")
        lifecycles.append(lifecycle.to_dict())
        tables[stage].append(binding)
        rows.append({"component": component["id"], "role_candidate": component["role_candidate"],
                     "stage": stage, "head": head, "receiver": receiver, "source": source,
                     "source_span": arrays["offsets"][source].tolist(), **observations})
        if reads:
            read_records.append(read_for_use(use, lifecycle,
                source=ReadSource(component["id"], (source,), "learned", {
                    "predictor": catalog["predictor_identity"],
                    "catalog": "sha256:" + catalog["fingerprint"],
                    "role_candidate": component["role_candidate"],
                    "score": observations["predicted_source_mass"],
                    "score_kind": "sealed predicted source mass (argmax source)"},
                    binding_fingerprint=lifecycle.binding_fingerprint),
                reader=Reader(stage, head, block_label=span, receivers="token row", receiver_row=receiver),
                mass=observations["native_source_mass"], mass_kind="native attention route mass at source",
                surface="passive-observation", backend="native-discovery",
                evidence=({"kind": "file", "path": record["arrays"]["path"],
                           "sha256": record["arrays"]["sha256"], "array": "routes",
                           "selector": [stage, head, receiver, source]},)))
    for stage, table in enumerate(tables):
        SymbolTable(tuple(table)).save(root / f"symbols-{stage:03d}.json", cuts[stage])
    save_json(root / "lifecycles.json", lifecycles)
    save_json(root / "occurrences.json", rows)
    if reads:
        save_reads(root / "reads.json", read_records, meta={
            "backend": "native-discovery", "document": document, "capture": reference,
            "causal_tests": 0})
    return {"bindings": len(rows), "uses": len(rows), "native_observation_cuts": len(cuts),
            "inactive_feature_abstentions": abstained,
            "semantics": "anonymous learned candidates", "causal_tests": 0}
