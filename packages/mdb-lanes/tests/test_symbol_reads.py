"""One writer for attention-derived symbol reads: shape, causal-status defaults, bounds, FLUX conversion."""
import filecmp
import importlib.util
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from saturn_pub import StateAddress

from mdb_lanes.attention_ranking import evaluate_top_source
from mdb_lanes.native_discovery import save_json
from mdb_lanes.native_discovery_symbols import register_observations
from mdb_lanes.symbol_reads import (
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
    declare_root,
    file_pointer,
    load_reads,
    model_identity,
    parameter_pointer,
    passive_binding,
    passive_definition,
    passive_lifecycle,
    passive_use,
    project_reads,
    read_for_use,
    resolve_pointer,
    save_reads,
    verdict_rule,
    verify_pointer,
)

PLAN = "c" * 64
FLUX = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")
HAS_FLUX = (FLUX / "runs/full-a/result/capture.cat.4242.npz").exists()


def control(**kw):
    """A valid counted control: in-run, structured, judged clean."""
    return {"arm": "z:rand4", "run_id": "job-1", "in_run": True, "matched_on": ["set_size"],
            "judged_by": "picture", "judgement": "clean", "effect": .01, **kw}


def result(verdict="set_only", intervention="cut", run_id="job-1", **kw):
    """A passing set result; with nothing ruled out it is the weaker ``set_only`` claim."""
    scopes = {"cut": "needed_set", "keep": "sufficient_set"}
    base = dict(scope="set", basis="score", effect=.6, floor=.01, floor_source="in-run sham",
                floor_incomplete=False, control=control(), contrast_id="toy",
                claim_scope=scopes[intervention] if verdict == "set_only" else
                "identity" if verdict == "supported" else None)
    base.update(kw)
    return CausalResult(intervention, "no-renormalisation", verdict, PLAN, run_id, **base)


def passive_fixture(tmp_path):
    capture = tmp_path / "capture.npz"
    np.savez(capture, tok_w=np.ones((2, 3, 4, 8), np.float16))
    root = declare_root("toy", tmp_path)
    sha = file_pointer(capture, root=root)["digest"]["sha256"]
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

    from mdb_lanes.attention_observer import AttentionObserver
    from mdb_lanes.project import Project
    from mdb_lanes.prompt_qwen import PromptQwenAdapter
    from mdb_lanes.saturn_pub_adapter import SaturnPubDebuggerAdapter
    from mdb_lanes.symbol_attention import definition_at, retained_binding
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
        result(control=None)
    with pytest.raises(SymbolReadError, match="cannot carry"):
        result("control_missing")
    with pytest.raises(SymbolReadError, match="verdict"):
        result("causal")
    with pytest.raises(SymbolReadError, match="sha256"):
        CausalResult("cut", "no-renormalisation", "unresolved", "not-a-hash", "job-1", scope="set",
                     basis="none", contrast_id="toy")
    with pytest.raises(SymbolReadError, match="names the run"):
        CausalResult("keep", "renormalising", "unresolved", PLAN, "", scope="set", basis="none", contrast_id="toy")
    ok = result()
    status = CausalStatus((ok,))
    assert status.status == "intervened" and status.label == "cut (no-renormalisation): set_only"
    assert CausalStatus.from_dict(status.to_dict()) == status
    assert ok.claim == "necessary" and ok.remaining_alternatives == ("generic_damage", "mass", "set_size")


def test_writer_refuses_untyped_or_unbacked_supported_results(tmp_path):
    """Audit finding 4: every probe.py case that the old writer accepted is refused now."""
    with pytest.raises(SymbolReadError, match="numeric floor"):
        result(floor=None, floor_source=None)                       # supported with floor=None
    with pytest.raises(SymbolReadError, match="arm|matched_on|run id|in_run"):
        result(control={"anything": 1})                             # junk control
    with pytest.raises(SymbolReadError, match="finite number"):
        result(effect="huge")                                       # effect is a string
    with pytest.raises(SymbolReadError, match="exceed its floor"):
        result(effect=.001, floor=.5)                               # effect below floor
    with pytest.raises(SymbolReadError, match="same run"):
        result(control=control(in_run=False))                       # other-job control counted
    with pytest.raises(SymbolReadError, match="own run id"):
        result(control=control(run_id="job-other"))
    for bad in (" ", " job-1", ""):
        with pytest.raises(SymbolReadError, match="names the run"):
            result("unresolved", run_id=bad, control=None, basis="none")
    with pytest.raises(SymbolReadError, match="judged clean"):
        result(control=control(judgement="mixed"))
    with pytest.raises(SymbolReadError, match="audit_statement"):
        result(basis="audit_statement")
    with pytest.raises(SymbolReadError, match="picture calls"):
        result(basis="picture_only", effect=None, floor=None, floor_source=None)
    picture_only = result(basis="picture_only", effect=None, floor=None, floor_source=None,
                          picture={"calls": ["N", "N"], "levels": [3, 3]})
    assert picture_only.verdict == "set_only"
    with pytest.raises(SymbolReadError, match="scope"):
        CausalResult("cut", "no-renormalisation", "unresolved", PLAN, "job-1", basis="none")
    with pytest.raises(SymbolReadError, match="contrast"):
        result(contrast_id=None)
    with pytest.raises(SymbolReadError, match="claims 'necessary'"):
        result(claim="sufficient")
    # Alternatives: only a clean in-run record that says so rules one out; identity needs all.
    with pytest.raises(SymbolReadError, match="not backed"):
        result(ruled_out=("set_size",))
    with pytest.raises(SymbolReadError, match="size >= "):
        result(control=control(rules_out=["set_size"], size=10, set_size=60), ruled_out=("set_size",))
    sized = result(control=control(rules_out=["set_size"], size=60, set_size=60), ruled_out=("set_size",))
    assert sized.remaining_alternatives == ("generic_damage", "mass") and sized.claim_scope == "needed_set"
    with pytest.raises(SymbolReadError, match="rules out mass only"):
        result(control=control(rules_out=["mass"], mass_held=[.1], set_mass_held=[.5]), ruled_out=("mass",))
    with pytest.raises(SymbolReadError, match="claim_scope 'needed_set'"):
        result(claim_scope="identity")
    with pytest.raises(SymbolReadError, match="only on supported / set_only"):
        result("not_supported", claim_scope="needed_set")
    # One vocabulary with attention_panels: supported = nothing remains; set_only = something remains.
    with pytest.raises(SymbolReadError, match="'set_only'"):
        result("supported", claim_scope="needed_set")
    with pytest.raises(SymbolReadError, match="'supported'"):
        result("supported")
    full = control(rules_out=["generic_damage", "mass", "set_size"], matched_on=["set_size", "mass"],
                   size=60, set_size=60)
    identity = result("supported", control=full, ruled_out=("generic_damage", "mass", "set_size"))
    assert identity.claim_scope == "identity" and identity.remaining_alternatives == ()
    with pytest.raises(SymbolReadError, match="with none remaining"):
        result("set_only", control=full, ruled_out=("generic_damage", "mass", "set_size"), claim_scope="needed_set")
    # Plan hash verifiable against a file.
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    from mdb_lanes.symbol_reads import file_sha256
    good = CausalResult("cut", "no-renormalisation", "unresolved", file_sha256(plan), "job-1",
                        scope="set", basis="none", contrast_id="toy")
    assert good.verify_plan(plan)
    with pytest.raises(SymbolReadError, match="does not hash"):
        result().verify_plan(plan)
    # Keep: separates from its control by more than the floor.
    with pytest.raises(SymbolReadError, match="separate from its control"):
        result(intervention="keep", effect=.05, claim_scope="sufficient_set", control=control(effect=.04))


def test_set_scope_results_never_attach_to_reads(tmp_path):
    """Audit finding 4: SymbolRead.with_causal refuses a set result; ReadSet refuses a read result."""
    source, lifecycle, reads, capture = passive_fixture(tmp_path)
    with pytest.raises(SymbolReadError, match="member read"):
        reads[0].with_causal(result())
    read_result = result("not_supported", scope="read", control=None, effect=.0)
    attached = reads[0].with_causal(read_result)
    assert attached.causal.results[0].scope == "read"
    read_set = ReadSet("own4", source, ReadSet.membership([(1, h) for h in range(4)],
                       file_pointer(capture, root=declare_root("toy", tmp_path))), {"kind": "ranked"})
    with pytest.raises(SymbolReadError, match="belongs on its SymbolRead"):
        read_set.with_result(read_result)
    with pytest.raises(SymbolReadError, match="'set'-scope"):
        replace(read_set, causal=CausalStatus((read_result,)))
    with pytest.raises(SymbolReadError, match="'read'-scope"):
        replace(reads[0], causal=CausalStatus((result(),)))


def test_set_level_status_is_not_propagated_to_member_reads(tmp_path):
    source, lifecycle, reads, capture = passive_fixture(tmp_path)
    members = [(1, h) for h in range(4)]
    root = declare_root("toy", tmp_path)
    read_set = ReadSet("own4", source, ReadSet.membership(members, file_pointer(capture, root=root)),
                       {"kind": "ranked"})
    reads = [replace(r, sets=(read_set.set_id,)) for r in reads]
    read_set = replace(read_set, member_reads=tuple(r.fingerprint for r in reads)).with_result(result())
    assert read_set.set_id == ReadSet("own4", source, read_set.members, {"kind": "ranked"}).set_id
    save_reads(tmp_path / "reads.json", reads, (read_set,))
    loaded = load_reads(tmp_path / "reads.json")
    (saved_set,) = loaded["sets"]
    assert saved_set.causal.results[0].verdict == "set_only"
    assert all(r.causal.status == "unknown" and r.causal.label == UNKNOWN_LABEL for r in loaded["reads"])
    assert set(saved_set.member_reads) == {r.fingerprint for r in loaded["reads"]}
    with pytest.raises(SymbolReadError, match="already exists"):
        save_reads(tmp_path / "reads.json", reads, (read_set,))


def test_set_selection_kind_is_not_a_role_assignment(tmp_path):
    """Audit finding 7: a set is 'ranked', never 'nominated' (that names a source role)."""
    source, _, _, capture = passive_fixture(tmp_path)
    pointer = file_pointer(capture, root=declare_root("toy", tmp_path))
    with pytest.raises(SymbolReadError, match="use 'ranked'"):
        ReadSet("own", source, ReadSet.membership([(1, 1)], pointer), {"kind": "nominated"})
    with pytest.raises(SymbolReadError, match="selection kind"):
        ReadSet("own", source, ReadSet.membership([(1, 1)], pointer), {})


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
    root = declare_root("toy", tmp_path)
    big = ReadSet("own240", source, ReadSet.membership([(b, h) for b in range(10) for h in range(24)],
                  file_pointer(capture, root=root)), {"kind": "ranked"})
    assert big.members["count"] == 240 and "units" not in big.members
    record = big.to_dict()
    assert max(len(v) for v in _sequences(record)) <= 64
    pointer = file_pointer(capture, root=root, array="tok_w", selector=[None, 1, 2, [3]])
    assert set(pointer) == {"kind", "locator", "digest", "array", "selector"}


def test_size_budget_is_recursive_and_covers_every_record_and_meta(tmp_path):
    """Audit finding 5: nested lists, passive use/definition metadata and meta are bounded."""
    source, lifecycle, reads, capture = passive_fixture(tmp_path)
    nested = [[0.1234567] * 48 for _ in range(48)]                 # 2304 floats, 35 KB: passed before
    with pytest.raises(SymbolReadError, match="nested"):
        replace(reads[0], metadata={"grid": nested})
    with pytest.raises(SymbolReadError, match="nested"):
        check_bounded({"a": [[0.1234567] * 60] * 60})
    surface = PassiveSurface("m", {}, {}, {"state_schema": "s"}, "ref", "sess")
    with pytest.raises(SymbolReadError, match="nested"):
        passive_use(surface, lifecycle.binding, use_id="x", consumer="c", signed_observations=(),
                    evidence_refs=(), metadata={"grid": nested})
    with pytest.raises(SymbolReadError, match="nested"):
        passive_definition(surface, lifecycle.binding, occurrence_id="d", clock=None, evidence_refs=(),
                           metadata={"grid": nested})
    wide = {str(i): [0.123456789] * 64 for i in range(256)}          # 16k values, flat lists
    with pytest.raises(SymbolReadError, match="values in one record|bytes"):
        check_bounded(wide)
    with pytest.raises(SymbolReadError, match="bytes"):
        check_bounded({str(i): "x" * 4000 for i in range(20)})       # 80 KB of text, few values
    with pytest.raises(SymbolReadError, match="nested|bytes|values"):
        save_reads(tmp_path / "meta.json", reads, meta={"m": [[[0.5] * 60] * 60] * 10})
    with pytest.raises(SymbolReadError, match="bytes"):
        save_reads(tmp_path / "meta2.json", reads, meta={str(i): "y" * 4000 for i in range(20)})
    assert not (tmp_path / "meta.json").exists() and not (tmp_path / "meta2.json").exists()
    # numpy scalars convert like Python scalars, np.bool_ included; arrays (also 0-d) do not.
    assert check_bounded({"a": np.bool_(True), "b": np.float32(1.5), "c": np.int64(3)}) == \
        {"a": True, "b": 1.5, "c": 3}
    assert type(check_bounded(np.bool_(False))) is bool
    with pytest.raises(SymbolReadError, match="array or tensor"):
        check_bounded({"a": np.array(True)})


def test_pointers_split_locator_and_digest_and_names_carry_identity_status(tmp_path):
    """Audit finding 6: moving a file changes the root declaration, never the reference."""
    first = tmp_path / "a"
    (first / "runs").mkdir(parents=True)
    (first / "runs" / "plan.json").write_text('{"x": 1}')
    pointer = file_pointer(first / "runs" / "plan.json", root=declare_root("exp", first), role="plan")
    assert pointer["locator"] == {"root": "exp", "path": "runs/plan.json"}
    assert set(pointer["digest"]) == {"sha256", "size"} and pointer["digest"]["size"] == 8
    assert str(tmp_path) not in json.dumps(pointer)
    moved = tmp_path / "b"
    shutil.move(first, moved)
    assert file_pointer(moved / "runs" / "plan.json", root=declare_root("exp", moved), role="plan") == pointer
    assert verify_pointer(pointer, {"exp": moved}) == resolve_pointer(pointer, {"exp": moved})
    (moved / "runs" / "plan.json").write_text('{"x": 2}')
    with pytest.raises(SymbolReadError, match="digest"):
        verify_pointer(pointer, {"exp": moved})
    with pytest.raises(SymbolReadError, match="undeclared root"):
        resolve_pointer(pointer, {})
    with pytest.raises(SymbolReadError, match="not under"):
        file_pointer(tmp_path / "elsewhere.json", root=declare_root("exp", moved))
    # A bare model name is not an identity.
    unresolved = parameter_pointer("flux2-klein-4b-distilled", "w")
    assert unresolved["model"]["identity_status"] == "unresolved" and unresolved["model"]["resolve_by"]
    assert parameter_pointer("m", "w", checkpoint_sha256="d" * 64)["model"]["identity_status"] == "verified"
    with pytest.raises(SymbolReadError, match="resolve"):
        model_identity("flux2-klein-4b-distilled")
    source, lifecycle, reads, _ = passive_fixture(tmp_path)
    named = replace(reads[0], model=model_identity("toy-model", resolve_by="checkpoint sha256"))
    assert named.to_dict()["model"]["identity_status"] == "unresolved"
    with pytest.raises(SymbolReadError, match="identity_status"):
        replace(reads[0], model={"name": "toy-model"})


def test_verdict_rule_is_one_sealed_record_named_by_every_result(tmp_path):
    """Audit finding 3: results name the rule id + sha256 declared once in meta."""
    source, _, reads, capture = passive_fixture(tmp_path)
    rule = verdict_rule("toy-rule-v1", {"levels": {"Y": 0, "N": 3}, "steps": ["picture decides"]})
    root = declare_root("toy", tmp_path)
    members = ReadSet.membership([(1, 0)], file_pointer(capture, root=root))
    named = ReadSet("s", source, members, {"kind": "ranked"}).with_result(
        result(rule_id=rule["id"], rule_sha256=rule["sha256"]))
    save_reads(tmp_path / "ok.json", (), (named,), meta={"verdict_rule": rule})
    assert load_reads(tmp_path / "ok.json")["meta"]["verdict_rule"]["sha256"] == rule["sha256"]
    other = ReadSet("s", source, members, {"kind": "ranked"}).with_result(
        result(rule_id="other", rule_sha256="e" * 64))
    with pytest.raises(SymbolReadError, match="other than meta.verdict_rule"):
        save_reads(tmp_path / "bad.json", (), (other,), meta={"verdict_rule": rule})
    with pytest.raises(SymbolReadError, match="does not match its rule"):
        save_reads(tmp_path / "bad2.json", (), (named,), meta={"verdict_rule": {**rule, "rule": {"x": 1}}})


def test_qwen_view_causal_line_comes_from_the_data(tmp_path):
    """Audit finding 7: the Qwen view's causal-status line is data, not a constant in the page."""
    from mdb_lanes.symbol_attention_view import write_view
    head = {"head": 0, "kv_head": 0, "allocation": [1.0], "source_mass": 1.0, "entropy_nats": 0.,
            "source_value_return_norm": 0., "head_return_norm": 0., "token_labels": ["t"]}
    row = {"session_id": "s", "arm": "native", "step": 0, "layer": 0, "binding": {}, "lifecycle_id": "l",
           "definition": {}, "output": {}, "heads": [head], "uses": [{}]}
    (tmp_path / "linked-symbols.json").write_text(json.dumps([row]))
    (tmp_path / "formation-symbols.json").write_text("[]")
    (tmp_path / "summary.json").write_text("{}")
    write_view(tmp_path)
    html = (tmp_path / "symbols-attention.html").read_text()
    page, data = html.split("const data=", 1)[1].split(";\n", 1)[1], html.split("const data=", 1)[1]
    assert "Causal status: <strong>'+esc(r.causal_label)+'</strong>" in page
    assert '"causal_label": "' + UNKNOWN_LABEL + '"' in data
    assert UNKNOWN_LABEL not in page


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


def flux_module():
    path = Path(__file__).resolve().parents[1] / "qualification" / "flux_read_graph_symbols.py"
    spec = importlib.util.spec_from_file_location("flux_read_graph_symbols", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flux_verdict_rule_is_symmetric_and_reference_controls_never_count():
    """Audit finding 1 (cause): a Y- control is a weak effect, as it is for the set; screens
    can only downgrade; with no counted control the verdict is control_missing, never supported."""
    m = flux_module()
    pics = {"calls": ["P", "P"], "levels": [2, 2]}
    weak = {"arm": "z:rand120", "levels": [1, 0]}
    assert m.decide("cut", "night", pics, dict(weak), [])[0] == "unresolved"     # was 'supported'
    assert m.decide("cut", "night", pics, {"arm": "z:r", "levels": [0, 0]}, [])[0] == "supported"
    assert m.decide("cut", "night", pics, {"arm": "z:r", "levels": [2, 3]}, [])[0] == "not_supported"
    assert m.decide("cut", "red", pics, None, [])[0] == "control_missing"
    assert m.decide("cut", "red", pics, None, [{"arm": "z:rand240", "levels": [2, 0]}])[0] == "unresolved"
    assert m.decide("cut", "red", pics, None, [{"arm": "z:rand240", "levels": [0, 0]}])[0] == "control_missing"
    carried = {"calls": ["Y", "Y"], "levels": [0, 0]}
    assert m.decide("keep", "snow", carried, None, [{"arm": "z:keep-nounI", "levels": [1, 1]}])[:2] == \
        ("unresolved", "picture_only")
    assert m.decide("cut", "snow", carried, None, [])[:2] == ("not_supported", "picture_only")
    assert m.decide("cut", "red", None, None, [], m.X1_OVERRIDE)[:2] == ("not_supported", "audit_statement")
    assert m.judge_control("cut", [1, 1], [1, 0]) == m.judge_control("cut", [2, 2], [1, 0]) == "mixed"
    rule = m.VERDICT_RULE
    assert rule["rule"]["parameters"]["reference_cut_clean_max_abs_gone"] == .15
    assert rule["rule"]["parameters"]["reference_keep_lost_min_gone"] == .7
    assert {(c["run"], c["intervention"], c["set"]) for c in rule["rule"]["counted_controls"]} == \
        set(m.COUNTED_CONTROLS)
    assert not any(k[1] == "keep" for k in m.COUNTED_CONTROLS)   # no random keep arm was run in-run
    with pytest.raises(TypeError):
        m.convert(FLUX)                                           # the output directory is required


@pytest.fixture(scope="module")
def flux_out(tmp_path_factory):
    if not HAS_FLUX:
        pytest.skip("saved FLUX read-graph experiment not on this machine")
    out = tmp_path_factory.mktemp("flux") / "flux"
    summary = flux_module().convert(FLUX, out)
    loaded = load_reads(out / "reads.json")
    results = {(s.source.name, s.name, r.intervention, r.cut_type): r for s in loaded["sets"] for r in s.causal.results}
    return summary, out, loaded, results


# MEASURED on the saved experiment (2026-11-12): status x contrast after the audit-8 fixes and the
# set_only vocabulary (rule v3): the 19 former `supported` (all needed_set) are `set_only`.
FLUX_TABLE = {"ball": {"control_missing": 6, "not_supported": 16, "set_only": 5, "unresolved": 5},
              "colour-words": {"control_missing": 16, "not_supported": 33, "set_only": 13, "unresolved": 2},
              "night": {"control_missing": 2, "not_supported": 20, "set_only": 1, "unresolved": 9},
              "rain": {"control_missing": 3, "not_supported": 12},
              "snow": {"control_missing": 6, "not_supported": 19, "unresolved": 7}}


def test_flux_converter_on_saved_files(flux_out):
    summary, out, loaded, results = flux_out
    assert summary["bindings"] == 6 and summary["reads"] == 360 == summary["uses"]
    assert all(summary["checks"].values())
    assert len(loaded["reads"]) == 360 and len(loaded["sets"]) == summary["read_sets"] == 114
    assert all(r.causal.status == "unknown" for r in loaded["reads"])
    by_fp = {r.fingerprint: r for r in loaded["reads"]}
    intervened = [s for s in loaded["sets"] if s.causal.results]
    assert intervened and all(by_fp[f].causal.status == "unknown" for s in intervened for f in s.member_reads)
    assert not list(out.glob("*.npz"))
    assert len(results) == summary["set_results"] == 175
    assert all(len(r.plan_sha256) == 64 and r.run_id.startswith("job-") for r in results.values())


def test_flux_supported_rests_only_on_in_run_matched_controls(flux_out):
    """Audit finding 1: the 17 results that rested on another job's control, on the other pool's keep,
    or on a Y- control are no longer supported; snow/rain keep their picture result."""
    summary, out, loaded, results = flux_out
    assert summary["set_results_by_contrast_verdict"] == FLUX_TABLE
    assert summary["set_results_by_verdict"] == {"control_missing": 33, "not_supported": 100,
                                                 "set_only": 19, "unresolved": 23}
    for r in results.values():
        assert r.verdict != "supported"                       # nothing reaches identity
        if r.verdict == "set_only":
            assert r.control["in_run"] is True and r.control["run_id"] == r.run_id
            assert r.control["judgement"] == "clean" and r.basis == "picture"
        for ref in r.reference_controls:
            assert ref["in_run"] or ref["verdict_effect"] == "none"
    # 14 that rested on the other-job Addendum 4 arm: control_missing, arm kept as a reference.
    for key in [("cat:red", "headfree:own60", "cut"), ("cat:red", "headfree:own60", "keep"),
                ("cat:black", "headfree:own60", "cut"), ("cat:ball", "headfree:own60", "keep"),
                ("cat:red", "headx:color", "keep"), ("cat:black", "headx:light", "keep"),
                ("cat:ball", "headx:scene", "keep")]:
        r = results[(*key, "no-renormalisation")]
        assert r.verdict == "control_missing" and r.control is None
        other_job = [c for c in r.reference_controls if not c["in_run"]]
        assert len(other_job) == 1 and other_job[0]["arm"] in ("z:rand60a", "z:keep-rand60a")
        assert other_job[0]["run_id"] != r.run_id and other_job[0]["judged_by"] == "score"
    for word in ("red", "black"):
        r = results[(f"cat:{word}", "headfree:modI", "keep", "no-renormalisation")]
        assert r.verdict == "control_missing"
        assert [c["arm"] for c in r.reference_controls] == ["z:keep-nounI"]
    # night: Y- on the control on one seed is a weak effect there too (symmetric rule).
    assert results[("car:night", "headfree:own120", "cut", "no-renormalisation")].verdict == "unresolved"
    night60 = results[("car:night", "headfree:own60", "cut", "no-renormalisation")]
    assert night60.verdict == "unresolved" and "rand240" in night60.reason
    for name in ("headx:color", "headx:light"):
        assert results[("car:night", name, "cut", "no-renormalisation")].verdict == "unresolved"
    # snow and rain: picture result kept, numbers dropped.
    snow_rain = {k: r for k, r in results.items() if k[0] in ("cat:snow", "car:rain")}
    assert len(snow_rain) == 47
    assert all(r.effect is None and r.effect_per_seed is None and r.floor is None for r in snow_rain.values())
    cuts_yy = [r for k, r in snow_rain.items() if k[2] == "cut" and r.picture and max(r.picture["levels"]) == 0]
    assert len(cuts_yy) == 28 and all(r.verdict == "not_supported" and r.basis == "picture_only" for r in cuts_yy)
    no_picture = [r for r in snow_rain.values() if r.picture is None]
    assert len(no_picture) == 4 and all(r.verdict == "unresolved" for r in no_picture)
    keeps = Counter(r.verdict for k, r in snow_rain.items() if k[2] == "keep")
    assert keeps == {"control_missing": 9, "unresolved": 3, "not_supported": 3}
    for key in [("cat:snow", "headfree:modI"), ("cat:snow", "headfree:nounI"), ("cat:snow", "headx:object")]:
        assert results[(*key, "keep", "no-renormalisation")].verdict == "unresolved"
    for r in snow_rain.values():
        for c in (r.control, *r.reference_controls):
            assert c is None or ("effect" not in c and "effect_per_seed" not in c)


def test_flux_scope_contrasts_and_claim_vocabulary(flux_out):
    """Audit finding 2: results carry their contrast and claim scope; the view shows scope, controls,
    mass, borrowed floors and the L1 caveat next to each verdict."""
    summary, out, loaded, results = flux_out
    m = flux_module()
    scope = loaded["meta"]["scope"]
    assert (scope["prompts"], scope["seeds"], scope["independent_contrasts"]) == (2, 2, 3)
    assert scope["source_to_contrast"] == m.CONTRASTS
    assert summary["scope"]["independent_contrast_ids"] == ["colour-words", "ball", "night"]
    per_contrast = Counter()
    for v in summary["set_results_by_contrast_verdict"].values():
        per_contrast.update(v)
    assert dict(per_contrast) == summary["set_results_by_verdict"]
    for (source, _, kind, _), r in results.items():
        assert r.scope == "set" and r.contrast_id == m.CONTRASTS[source]
        assert r.claim == {"cut": "necessary", "keep": "sufficient"}[kind]
        assert r.claim_scope != "identity" and r.remaining_alternatives
        assert (r.claim_scope is not None) == (r.verdict == "set_only")
        assert isinstance(r.floor_incomplete, bool)
        if r.run_id == loaded["meta"]["jobs"]["headfree"] and r.floor is not None:
            assert r.floor_source.startswith("borrowed:") and r.floor_incomplete
        if r.verdict == "set_only":
            assert set(r.control["matched_on"]) >= {"set_size"} or "other_pool" in r.control["matched_on"]
            assert r.control["mass_held"] and r.control["set_mass_held"]
    red120 = results[("cat:red", "headfree:own120", "cut", "no-renormalisation")]
    assert red120.control["arm"] == "z:rand120" and red120.control["mass_held"][0] < .2 < .7 < red120.control["set_mass_held"][0]
    assert "not attributable to the word's own heads" in red120.metadata["caveat"]
    # mass is ruled out only where a clean control holds at least the set's mass (night mod vs noun).
    massed = [k for k, r in results.items() if "mass" in r.ruled_out]
    assert massed == [("car:night", "headfree:mod", "cut", "no-renormalisation")]
    html = (out / "symbol-reads.html").read_text()
    for text in ('class="scope"', "independent contrasts", "Contrast: colour-words", "Contrast: night",
                 "size-matched only", "borrowed:", "not attributable to the word", "OTHER JOB", "mass kind",
                 "identity_status", "unresolved", "word mass held", "needed_set", "set_only (weaker)",
                 "v set_only"):
        assert text in html, text
    assert html.index('class="scope"') < html.index("Read sets (")
    assert "http" not in html


def test_flux_verdict_rule_recorded_once_and_x1_marked(flux_out):
    """Audit finding 3."""
    summary, out, loaded, results = flux_out
    from mdb_lanes.symbol_reads import seal_digest
    rule = loaded["meta"]["verdict_rule"]
    assert rule["sha256"] == seal_digest(rule["rule"]) == summary["verdict_rule"]["sha256"]
    assert all((r.rule_id, r.rule_sha256) == (rule["id"], rule["sha256"]) for r in results.values())
    x1 = [(k, r) for k, r in results.items() if r.basis == "audit_statement"]
    assert len(x1) == 16 and all(r.verdict == "not_supported" and k[1].startswith("headx:")
                                 and k[1].endswith("X") and k[0] != "cat:snow" for k, r in x1)
    for text in ("counted_controls", "screens", "other_job_references", "x1_override", "parameters"):
        assert text in rule["rule"]


def test_flux_pointers_are_locator_plus_digest_and_identity_unresolved(flux_out):
    """Audit finding 6."""
    summary, out, loaded, results = flux_out
    meta = loaded["meta"]
    assert meta["model"]["identity_status"] == "unresolved" and meta["model"]["resolve_by"]
    assert all(r.model["identity_status"] == "unresolved" for r in loaded["reads"])
    lifecycles = json.loads((out / "lifecycles.json").read_text())
    assert all(l["metadata"]["model"]["identity_status"] == "unresolved" for l in lifecycles)
    roots = meta["roots"]
    assert list(roots) == ["flux-read-graph"]
    for pointer in meta["inputs"].values():
        assert set(pointer) >= {"locator", "digest"} and "path" not in pointer
        verify_pointer(pointer, roots)
    text = (out / "reads.json").read_text()
    assert text.count(str(FLUX)) == 1                     # only the root declaration


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

    from mdb_lanes.symbol_lifecycle import ClockWitness, SymbolLifecycle, SymbolOccurrence, SymbolUse

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


# ------------------------------------------------------------------ panel verdict -> set result

def _panel_outcomes(panel, item, effect, control, *, floor=0.01, keep=None, keep_random=None, allcut=1.0):
    out = {}
    for a in panel.arms:
        if a.item != item or a.role == "floor":
            continue
        is_keep = a.kind == "keep"
        if a.role == "test":
            out[a.arm] = keep if is_keep else effect
        elif a.control == "random":
            out[a.arm] = keep_random if is_keep else control
        elif a.control == "other_item":
            out[a.arm] = 0.9 if is_keep else control
        elif a.kind == "all_cut":
            out[a.arm] = allcut
    for a in panel.arms:
        if a.item == item and a.role == "floor" and a.floor_of in out:
            out[a.arm] = out[a.floor_of] if a.kind == "floor_cut_recompute" else floor
    pairs = {}
    from mdb_lanes.attention_panels import twin_pairs
    for key in twin_pairs(panel):
        a, b = key.split("|")
        if a in out and b in out:
            pairs[key] = abs(out[a] - out[b])
    return out, pairs


def _panel_verdicts(panel, item, **kw):
    from mdb_lanes import attention_panels as P
    out, pairs = _panel_outcomes(panel, item, **kw)
    rec = P.results_record(panel, {"c0": out}, pair_distances={"c0": pairs}, metric="toy distance")
    # equal per-item mass: the other-item control cuts the same mass share (since 2026-11-13 an
    # other-item control of unknown mass share rules nothing out)
    equal = {it: {(b, h): 1.0 for b in range(4) for h in range(3)} for it in ("red", "ball")}
    return {(v["item"], v["unit_set"]): v for v in P.verdicts(panel, rec, item_mass=equal)}


def test_panel_verdict_converts_to_set_results_without_re_deriving():
    from mdb_lanes import attention_panels as P
    from mdb_lanes.symbol_reads import set_results_from_panel_verdict
    universe = [(b, h) for b in range(4) for h in range(3)]

    def panel(**kw):
        base = dict(items=("red", "ball"), universe=universe, sets={"red": {"s": [(0, 0)]}, "ball": {"s": [(1, 1)]}},
                    contexts=("c0",), cut_sets=("s",), random=P.RandomControls(n=1, seed=0),
                    references=("condition", "all_cut"))
        base.update(kw)
        return P.read_panel(P.ReadPanelSpec(**base))

    # set_only: random of unknown mass beaten + other item beaten -> mass remains
    p = panel()
    v = _panel_verdicts(p, "red", effect=.5, control=.05)[("red", "s")]
    assert v["status"] == "set_only"
    results = set_results_from_panel_verdict(p, v, run_id="job-x", contrast_id="toy", effect_units="toy distance")
    assert [r.cut_type for r in results] == ["no-renormalisation", "renormalising"]
    for r in results:
        assert r.verdict == "set_only" and r.claim_scope == "needed_set" and r.scope == "set"
        assert r.remaining_alternatives == ("mass",) and r.ruled_out == ("generic_damage", "set_size")
        assert r.plan_sha256 == p.sha256() and r.control["judgement"] == "clean"
        assert r.control["run_id"] == "job-x" and r.control["in_run"]
        assert r.effect == .5 and r.floor == pytest.approx(.01) and r.claim_text.startswith("WEAKER CLAIM")
        assert r.metadata["panel"]["cut_type_status"] == "set_only"
    # supported (identity): mass-matched disjoint random + other item beaten
    flat = tuple(1.0 for _ in universe)
    pi = panel(random=P.RandomControls(n=1, mass=flat, mass_tolerance=.1, disjoint=True, seed=0))
    v = _panel_verdicts(pi, "red", effect=.5, control=.05)[("red", "s")]
    assert v["status"] == "supported"
    for r in set_results_from_panel_verdict(pi, v, run_id="job-x", contrast_id="toy"):
        assert r.verdict == "supported" and r.claim_scope == "identity" and r.remaining_alternatives == ()
        assert "mass" in r.control["matched_on"]
    # not_supported: the random control did as much damage; no counted control, refuter recorded
    v = _panel_verdicts(p, "red", effect=.5, control=.9)[("red", "s")]
    assert v["status"] == "not_supported"
    for r in set_results_from_panel_verdict(p, v, run_id="job-x", contrast_id="toy"):
        assert r.verdict == "not_supported" and r.control is None and r.claim_scope is None
        refuter = [c for c in r.reference_controls if c["control_kind"] == "random"][0]
        assert refuter["judgement"] == "same" and refuter["verdict_effect"] == "downgrade_only"
    # keep: set_only sufficient_set; the counted control is the beaten random keep
    pk = panel(sets={"red": {"s": [(0, 0), (1, 1)]}, "ball": {"s": [(2, 2), (3, 1)]}}, cut_sets=(), keep_sets=("s",))
    v = _panel_verdicts(pk, "red", effect=0, control=0, keep=.1, keep_random=.8)[("red", "keep-s")]
    assert v["status"] == "set_only" and v["claim_scope"] == "sufficient_set"
    for r in set_results_from_panel_verdict(pk, v, run_id="job-x", contrast_id="toy"):
        assert r.intervention == "keep" and r.verdict == "set_only" and r.claim_scope == "sufficient_set"
        assert r.control["control_kind"] == "random" and r.control["effect"] == .8 and r.effect == .1
        assert r.control["size"] == r.control["set_size"] == 2
        assert any(c["control_kind"] == "other_item" and c["judgement"] is None for c in r.reference_controls)
