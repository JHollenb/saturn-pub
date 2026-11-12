"""One writer for attention-derived symbol reads: shape, causal-status defaults, bounds, FLUX conversion."""
import filecmp
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from saturn_pub import StateAddress

from mdb.attention_ranking import evaluate_top_source
from mdb.native_discovery import save_json
from mdb.native_discovery_symbols import register_observations
from mdb.symbol_reads import (
    UNKNOWN_LABEL,
    CausalResult,
    CausalStatus,
    PassiveSurface,
    Reader,
    ReadSet,
    ReadSource,
    SymbolRead,
    SymbolReadError,
    check_bounded,
    file_pointer,
    load_reads,
    passive_binding,
    passive_definition,
    passive_lifecycle,
    passive_use,
    project_reads,
    read_for_use,
    save_reads,
)

PLAN = "c" * 64
FLUX = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")


def passive_fixture(tmp_path):
    capture = tmp_path / "capture.npz"
    np.savez(capture, tok_w=np.ones((2, 3, 4, 8), np.float16))
    sha = file_pointer(capture)["sha256"]
    surface = PassiveSurface("toy-model", {"dtype": "fp32"},
        {"run": "toy", "capture_sha256": sha}, {"state_schema": "toy-capture-v1", "kind": "passive",
        "slots": [{"name": "text_rows", "role": "text-row"}, {"name": "ids", "role": "ids"}]},
        "passive-capture:sha256:" + sha, "toy/session",
        slots={"text_rows": {"kind": "tensor", "shape": [8]}, "ids": {"kind": "token-ids", "length": 8}})
    from saturn_pub.values import digest
    binding = passive_binding(surface, "toy.word", (surface.address("text_rows", (3,), "text-row"),),
        context_slots=(("ids", digest({"kind": "token-ids", "length": 8})),),
        evidence=("sha256:" + sha,), claim="Supplied toy word.", status="supplied")
    uses = [passive_use(surface, binding, use_id=f"read-h{h}", consumer=f"toy.block1.head{h}",
            signed_observations=({"observable": "write_mass", "value": .1 * h},),
            evidence_refs=("sha256:" + sha,)) for h in range(4)]
    definition = passive_definition(surface, binding, occurrence_id="def", clock=None,
        evidence_refs=("sha256:" + sha,), metadata={"rows": [3]}, status="supplied")
    lifecycle = passive_lifecycle(binding, definitions=(definition,), uses=uses, claim_boundary="toy")
    source = ReadSource("toy:word", (3,), "supplied", {"by": "test"}, lifecycle.binding_fingerprint)
    reads = [read_for_use(u, lifecycle, source=source, reader=Reader(1, h, steps=(0, 1)), mass=.1 * h,
             mass_kind="write mass", surface="passive-observation", backend="toy") for h, u in enumerate(uses)]
    return source, lifecycle, reads, capture


def project_fixture(tmp_path):
    from saturn_pub.adapters.qwen import QwenAdapter

    from mdb.attention_observer import AttentionObserver
    from mdb.project import Project
    from mdb.prompt_qwen import PromptQwenAdapter
    from mdb.saturn_pub_adapter import SaturnPubDebuggerAdapter
    from mdb.symbol_attention import definition_at, retained_binding
    native = PromptQwenAdapter(QwenAdapter.tiny().model).session([5, 7, 11])
    project = Project(tmp_path / "project")
    adapter = SaturnPubDebuggerAdapter(native, project=project, session_id="qwen")
    parent = adapter.checkpoint()["cut_id"]
    with AttentionObserver(native.adapter.model) as observed:
        receipt = native.continue_(5)
    result = adapter.checkpoint()["cut_id"]
    source = retained_binding(project, "qwen", parent, "description.L1",
        tuple(StateAddress(slot, (1, 0, kv, 0)) for slot in ("keys", "values") for kv in range(2)),
        context="example", context_slots=("tokens", "keys", "values"), claim="Supplied source role.")
    output = retained_binding(project, "qwen", result, "output.logits", (StateAddress("logits"),),
        context="example", claim="Observed native readout.")
    lifecycle = definition_at(project, "qwen", parent, source)
    captured = observed.rows[1]
    trace = {"layer": 1, **{key: {"shape": list(captured[key].shape), "sha256": "a" * 64,
        "path": key + ".npy", "dtype": str(captured[key].dtype)} for key in
        ("query", "key", "value", "attention", "head_return")}}
    heads = [{"head": h, "kv_head": h // 2, "source_mass": float(captured["attention"][0, h, 0, 0]),
        "entropy_nats": 1., "source_value_return_norm": .5} for h in range(4)]
    return project, lifecycle, dict(receipt_id=receipt.fingerprint, trace=trace, heads=heads,
                                    source_row=0, output_binding=output)


def keys(value):
    if isinstance(value, dict):
        return {k: keys(v) for k, v in value.items() if k not in ("provenance", "metadata", "evidence")}
    return type(value).__name__ if value is not None else None


def test_project_and_passive_variants_give_the_same_record_shape(tmp_path):
    project, lifecycle, attach = project_fixture(tmp_path)
    source = ReadSource("description.L1", (0,), "supplied", {"by": "test"})
    joined, project_side = project_reads(project, "qwen", lifecycle, source=source, step=0, **attach)
    assert len(project_side) == 4 and {r.reader.kv_group for r in project_side} == {0, 1}
    assert all(r.binding_fingerprint == joined.binding_fingerprint for r in project_side)
    _, _, passive_side, _ = passive_fixture(tmp_path)
    a, b = project_side[0].to_dict(), passive_side[0].to_dict()
    assert set(a) == set(b) and set(a["source"]) == set(b["source"]) and set(a["reader"]) == set(b["reader"])
    assert set(a["causal"]) == set(b["causal"]) == {"status", "label", "results"}
    for r in (*project_side, *passive_side):
        assert r.causal.status == "unknown" and r.causal.label == UNKNOWN_LABEL
        assert SymbolRead.from_dict(r.to_dict()) == r
    # The Project-backed canonical records are unchanged: still saved through the Project.
    saved = project.save_symbol_lifecycle("qwen", joined, include_slices=False)
    assert len(saved["joins"]["uses"]) == 4


def test_causal_status_defaults_to_unknown_and_results_need_provenance():
    assert CausalStatus().status == "unknown" and CausalStatus().label == UNKNOWN_LABEL
    with pytest.raises(SymbolReadError, match="needs a control"):
        CausalResult("cut", "no-renormalisation", "supported", PLAN, "job-1", effect=.5)
    with pytest.raises(SymbolReadError, match="cannot carry"):
        CausalResult("cut", "no-renormalisation", "control_missing", PLAN, "job-1", control={"arm": "x"})
    with pytest.raises(SymbolReadError, match="verdict"):
        CausalResult("cut", "no-renormalisation", "causal", PLAN, "job-1")
    with pytest.raises(SymbolReadError, match="sha256"):
        CausalResult("cut", "no-renormalisation", "unresolved", "not-a-hash", "job-1")
    with pytest.raises(SymbolReadError, match="names the run"):
        CausalResult("keep", "renormalising", "unresolved", PLAN, "")
    result = CausalResult("cut", "renormalising", "supported", PLAN, "job-1", effect=.4,
                          control={"arm": "z:rand", "effect": 0.}, floor=.01)
    status = CausalStatus((result,))
    assert status.status == "intervened" and status.label == "cut (renormalising): supported"
    assert CausalStatus.from_dict(status.to_dict()) == status


def test_set_level_status_is_not_propagated_to_member_reads(tmp_path):
    source, lifecycle, reads, capture = passive_fixture(tmp_path)
    members = [(1, h) for h in range(4)]
    read_set = ReadSet("own4", source, ReadSet.membership(members, file_pointer(capture)),
                       {"kind": "nominated"})
    reads = [replace(r, sets=(read_set.set_id,)) for r in reads]
    read_set = replace(read_set, member_reads=tuple(r.fingerprint for r in reads)).with_result(
        CausalResult("cut", "no-renormalisation", "supported", PLAN, "job-1", effect=.6,
                     control={"arm": "z:rand4", "effect": .01}))
    assert read_set.set_id == ReadSet("own4", source, read_set.members, {"kind": "nominated"}).set_id
    save_reads(tmp_path / "reads.json", reads, (read_set,))
    loaded = load_reads(tmp_path / "reads.json")
    (saved_set,) = loaded["sets"]
    assert saved_set.causal.results[0].verdict == "supported"
    assert all(r.causal.status == "unknown" and r.causal.label == UNKNOWN_LABEL for r in loaded["reads"])
    assert set(saved_set.member_reads) == {r.fingerprint for r in loaded["reads"]}
    with pytest.raises(SymbolReadError, match="already exists"):
        save_reads(tmp_path / "reads.json", reads, (read_set,))


def test_records_refuse_arrays_and_long_sequences(tmp_path):
    source, lifecycle, reads, capture = passive_fixture(tmp_path)
    with pytest.raises(SymbolReadError, match="array or tensor"):
        replace(reads[0], metadata={"values": np.zeros(3)})
    with pytest.raises(SymbolReadError, match="by pointer"):
        replace(reads[0], metadata={"values": [0.0] * 65})
    with pytest.raises(SymbolReadError, match="raw bytes"):
        check_bounded({"blob": b"weights"})
    with pytest.raises(SymbolReadError, match="by pointer"):
        ReadSource("long", tuple(range(65)), "supplied", {"by": "test"})
    surface_use = lifecycle.uses[0]
    with pytest.raises(SymbolReadError, match="array or tensor"):
        passive_use(PassiveSurface("m", {}, {}, {"state_schema": "s"}, "ref", "sess"), lifecycle.binding,
                    use_id="x", consumer="c", signed_observations=({"value": np.ones(2)},),
                    evidence_refs=surface_use.evidence_refs)
    # Large memberships are held by pointer + digest, never inline.
    big = ReadSet("own240", source, ReadSet.membership([(b, h) for b in range(10) for h in range(24)],
                  file_pointer(capture)), {"kind": "nominated"})
    assert big.members["count"] == 240 and "units" not in big.members
    record = big.to_dict()
    assert max(len(v) for v in _sequences(record)) <= 64
    pointer = file_pointer(capture, array="tok_w", selector=[None, 1, 2, [3]])
    assert set(pointer) == {"kind", "path", "sha256", "array", "selector"}


def _sequences(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _sequences(v)
    elif isinstance(value, list):
        yield value
        for v in value:
            yield from _sequences(v)


def test_nominated_role_carries_ranking_score_and_trivial_baselines():
    with pytest.raises(SymbolReadError, match="ranking and score"):
        ReadSource("s", (2,), "nominated", {"ranking": "x"})
    with pytest.raises(SymbolReadError, match="trivial baselines"):
        ReadSource("s", (2,), "nominated", {"ranking": "x", "score": .5, "evaluation": {}})
    evaluation = evaluate_top_source([2, 1], [2, 1], visible=[3, 3])
    nominated = ReadSource("s", (2,), "nominated", {"ranking": "x", "score": .5, "evaluation": evaluation})
    assert nominated.assignment == "nominated"
    with pytest.raises(SymbolReadError, match="predictor"):
        ReadSource("s", (2,), "learned", {})
    with pytest.raises(SymbolReadError, match="role assignment"):
        ReadSource("s", (2,), "guessed", {"by": "x"})


def native_inputs():
    rng = np.random.default_rng(3)
    stages, heads, length, width, features = 2, 2, 6, 4, 3
    routes = rng.random((stages, heads, length, length)).astype(np.float32)
    routes /= routes.sum(-1, keepdims=True)
    code = rng.random((stages, length, features)).astype(np.float32)
    code[0, 2] = 0
    record = {"binding": {"model_identity": "tiny", "execution": {"device": "cpu"}}, "document_id": "doc",
              "arrays": {"sha256": "a" * 64, "path": "doc.npz"}, "boundaries": [0, 1, 2], "fingerprint": "f" * 64}
    arrays = {"tokens": np.arange(length, dtype=np.int64),
              "states": [rng.random((length, width)).astype(np.float32) for _ in range(stages + 1)],
              "routes": routes, "offsets": np.stack([np.arange(length), np.arange(length) + 1], 1)}
    prediction = {"routes": routes[:, :, ::-1, ::-1].copy(), "code": code}
    catalog = {"fingerprint": "b" * 64, "predictor_identity": "pred",
               "components": [{"stage": s, "feature": f, "id": f"component_{s}{f}", "role_candidate": f"role_{f}"}
                              for s in range(stages) for f in range(features)]}
    return record, arrays, prediction, catalog


def test_register_observations_delegates_and_stays_byte_identical(tmp_path):
    inputs = native_inputs()
    legacy_register_observations(*inputs[:3], root=tmp_path / "legacy", catalog=inputs[3])
    counts = register_observations(*inputs[:3], root=tmp_path / "new", catalog=inputs[3])
    old = sorted(p.relative_to(tmp_path / "legacy") for p in (tmp_path / "legacy").rglob("*") if p.is_file())
    new = sorted(p.relative_to(tmp_path / "new") for p in (tmp_path / "new").rglob("*") if p.is_file())
    assert old == new and len(old) > 10
    assert all(filecmp.cmp(tmp_path / "legacy" / p, tmp_path / "new" / p, shallow=False) for p in old)
    assert counts["bindings"] > 0
    register_observations(*inputs[:3], root=tmp_path / "with-reads", catalog=inputs[3], reads=True)
    loaded = load_reads(tmp_path / "with-reads" / "reads.json")
    assert len(loaded["reads"]) == counts["bindings"]
    assert all(r.source.assignment == "learned" and r.causal.status == "unknown"
               and r.surface == "passive-observation" for r in loaded["reads"])
    lifecycles = json.loads((tmp_path / "with-reads" / "lifecycles.json").read_text())
    assert {r.lifecycle_fingerprint for r in loaded["reads"]} == {row["fingerprint"] for row in lifecycles}


@pytest.mark.skipif(not (FLUX / "runs/full-a/result/capture.cat.4242.npz").exists(),
                    reason="saved FLUX read-graph experiment not on this machine")
def test_flux_converter_on_saved_files(tmp_path):
    path = Path(__file__).resolve().parents[1] / "qualification" / "flux_read_graph_symbols.py"
    spec = importlib.util.spec_from_file_location("flux_read_graph_symbols", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.convert(FLUX, tmp_path / "flux")
    assert summary["bindings"] == 6 and summary["reads"] == 360 == summary["uses"]
    assert all(summary["checks"].values())
    loaded = load_reads(tmp_path / "flux" / "reads.json")
    assert len(loaded["reads"]) == 360 and len(loaded["sets"]) == summary["read_sets"]
    assert all(r.causal.status == "unknown" for r in loaded["reads"])
    by_fp = {r.fingerprint: r for r in loaded["reads"]}
    intervened = [s for s in loaded["sets"] if s.causal.results]
    assert intervened and all(by_fp[f].causal.status == "unknown" for s in intervened for f in s.member_reads)
    for s in loaded["sets"]:
        for r in s.causal.results:
            if s.source.name in ("cat:snow", "car:rain"):
                assert r.verdict == "unresolved" and r.effect is None and r.effect_per_seed is None
            assert len(r.plan_sha256) == 64 and r.run_id.startswith("job-")
    # Captures are referenced, not copied.
    assert not list((tmp_path / "flux").glob("*.npz"))
    assert all(p["kind"] == "file" and len(p["sha256"]) == 64 for p in loaded["meta"]["inputs"].values())
    html = (tmp_path / "flux" / "symbol-reads.html").read_text()
    assert "unknown (observed only" in html and "http" not in html and "control_missing" in html


# Frozen copy of register_observations at b7318c7 (unchanged since be8c719; pre-consolidation),
# the byte-compatibility oracle.
def legacy_register_observations(record, arrays, prediction, *, root, catalog, max_uses=24):
    """Persist real SymbolTable/Lifecycle records over explicitly non-resumable cuts.

    These are captured observation surfaces, not fabricated live Session closures.
    Predicted source relevance is retained beside the actual native source mass.
    """
    import torch
    from saturn_pub.contracts import StateAddress
    from saturn_pub.core import Receipt, StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.symbols import SymbolBinding, SymbolTable
    from saturn_pub.values import describe
    from saturn_pub.values import digest as public_digest

    from mdb.symbol_lifecycle import ClockWitness, SymbolLifecycle, SymbolOccurrence, SymbolUse

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
    rows, lifecycles, tables = [], [], [[] for _ in cuts]
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
        cut, end = cuts[stage], cuts[stage + 1]
        point = public_digest(dict(cut.execution_point))
        location = StateAddress("hidden", (source,), "observed-residual", point, schema)
        binding = SymbolBinding(component["id"], identity, public_digest(execution),
            document, (("tokens", public_digest(describe(observed_tokens))),), (location,),
            (reference, "sha256:" + catalog["fingerprint"]),
            "Automatically proposed source component; native location observed; semantic role unresolved",
            status="observed")
        # Check exact geometry without granting executable Session authority.
        binding.resolve_metadata(model_identity=identity, execution=execution, context=document,
            slots={"tokens": describe(observed_tokens), "hidden": describe(observed_states[stage])},
            surface=dict(cut.surface), execution_point=dict(cut.execution_point))
        receipt = Receipt.make(operation="mdb.native_discovery.observed-span", status="observed",
            parent=cut.fingerprint, result=end.fingerprint,
            capture=record["fingerprint"], source=source, receiver=receiver, head=head,
            claim="native observations plus a previously sealed source forecast; not a causal test")
        store.receipt(receipt)
        clock = ClockWitness("clock_" + str(number), "layer", document,
            record["boundaries"][stage], provenance=(reference,), status="observed")
        occurrence = SymbolOccurrence.for_binding(binding, occurrence_id="definition_" + str(number),
            session_id=document, source_cut=cut.fingerprint, locations=binding.locations,
            branch="ordinary-execution", execution=binding.execution, clock=clock,
            evidence_refs=(reference,), status="observed",
            metadata={"component": component["id"], "role_candidate": component["role_candidate"],
                      "source_status": "learned nomination", "surface": "observation-only"})
        observations = {"predicted_source_mass": float(prediction["routes"][stage, head, receiver, source]),
                        "native_source_mass": float(arrays["routes"][stage, head, receiver, source]),
                        "native_top_source": int(arrays["routes"][stage, head, receiver].argmax())}
        use = SymbolUse("use_" + str(number), binding.to_dict()["fingerprint"], document,
            f"native.layers.{record['boundaries'][stage]}:{record['boundaries'][stage + 1]}",
            source_cut=cut.fingerprint, parent_cut=cut.fingerprint, result_cut=end.fingerprint,
            branch="ordinary-execution", execution=binding.execution, receipt=receipt.fingerprint,
            locations=binding.locations, evidence_refs=(reference,), status="observed",
            signed_observations=(observations,))
        lifecycle = SymbolLifecycle.define(binding, definitions=(occurrence,), uses=(use,),
            mechanics_status="valid", instrument_status="observations-only", trend_status="not-assessed",
            terminal_status="not-assessed", claim_boundary="Predicted bindings over passive native observations.")
        lifecycles.append(lifecycle.to_dict())
        tables[stage].append(binding)
        rows.append({"component": component["id"], "role_candidate": component["role_candidate"],
                     "stage": stage, "head": head, "receiver": receiver, "source": source,
                     "source_span": arrays["offsets"][source].tolist(), **observations})
    for stage, table in enumerate(tables):
        SymbolTable(tuple(table)).save(root / f"symbols-{stage:03d}.json", cuts[stage])
    save_json(root / "lifecycles.json", lifecycles)
    save_json(root / "occurrences.json", rows)
    return {"bindings": len(rows), "uses": len(rows), "native_observation_cuts": len(cuts),
            "inactive_feature_abstentions": abstained,
            "semantics": "anonymous learned candidates", "causal_tests": 0}
