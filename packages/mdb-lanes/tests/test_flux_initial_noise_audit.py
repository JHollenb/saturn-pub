"""Own-input/clock/prospective-history regressions; no trained model or weights."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_initial_noise.py"))


def counters(native, acts):
    return {"adapter_steps": native, "acts": acts, "receipt_emissions": native + acts,
        "images": acts, "vae_calls": acts}


def remaining(origin):
    return {str(i): dict.fromkeys(A["TARGETS"], 0) for i in range(origin, 4)}


def test_two_seals_have_distinct_monotonic_histories_not_reset_counters():
    zero = remaining(0)
    matched_zero = remaining(2)
    full = A["guard_contract"]("full", counters(0, 0), counters(0, 1), zero, zero)
    matched = A["guard_contract"]("matched", counters(56, 1), counters(56, 2), matched_zero, matched_zero)
    assert full["earlier_native_transitions"] == 0
    assert matched["earlier_native_transitions"] == 56
    with pytest.raises(ValueError, match="erased historical"):
        A["guard_contract"]("matched", counters(0, 0), counters(0, 1), zero, zero)
    changed = copy.deepcopy(zero)
    changed["0"]["joint0"] = 1
    with pytest.raises(ValueError, match="future target"):
        A["guard_contract"]("full", counters(0, 0), counters(0, 1), zero, changed)


def test_future_image_must_precede_reference_not_just_follow_a_seal():
    zero = remaining(2)
    with pytest.raises(ValueError, match="pre-reference image"):
        A["guard_contract"]("matched", counters(56, 1), counters(57, 2), zero, zero)
    with pytest.raises(ValueError, match="seal erased"):
        A["guard_contract"]("full", counters(1, 0), counters(1, 1), remaining(0), remaining(0))


def test_geometry_and_every_native_clock_includes_four_commits():
    assert A["geometry"](0)["transitions"] == 112
    assert A["geometry"](2)["transitions"] == 56
    assert A["point_after"](0) == (0, "joint", 0)
    assert A["point_after"](5) == (0, "single", 0)
    assert A["point_after"](25) == (0, "readout", None)
    assert A["point_after"](26) == (0, "scheduler", None)
    assert [A["point_after"](i) for i in (27, 55, 83, 111)] == [(i, "project", None) for i in range(1, 5)]
    for bad in (-1, 112, True):
        with pytest.raises(ValueError):
            A["point_after"](bad)


def synthetic_pass(logical=0):
    identity = A["MATH"]["identity"]
    latent = np.zeros((1, 2, 2), dtype=np.float32)
    conditioning = np.zeros((1, 1, 2), dtype=np.float32)
    context = {key: np.ones((1, 2), dtype=np.float32) for key in ("temb", "img_mod", "txt_mod", "single_mod")}
    context["rope"] = (np.ones((3, 2), np.float32), np.zeros((3, 2), np.float32))
    image, text = np.ones_like(latent), np.ones_like(conditioning)
    prediction = {"logical_step": logical, "chain_length": 28, "input_latent_descriptor": identity(latent), "project": {"image": image, "text": text,
        "context": context, "input_latent_descriptor": identity(latent), "conditioning_descriptor": identity(conditioning)},
        "context_input_descriptors": A["PREV"]["context_descriptor"](context), "joints": {}, "single_steps": [],
        "next_state_contract": {"clock": {"step": logical + 1, "phase": "project", "cursor": 0},
            "reset": dict.fromkeys(("context", "image", "text", "noise_prediction")), "carried_state": {"ids": "own"}}}
    events = [("project", None, {"image": image, "text": text, "context": context})]
    for index in range(5):
        next_image, next_text = image + 1, text + 1
        prediction["joints"][str(index)] = {"image": next_image, "text": next_text,
            "input_image_descriptor": identity(image), "input_text_descriptor": identity(text)}
        events.append(("joint", index, {"image": next_image, "text": next_text}))
        image, text = next_image, next_text
    current = np.concatenate((text, image), axis=1)
    for index in range(20):
        following = current + 1
        prediction["single_steps"].append({"block_index": index, "step": index, "input_descriptor": identity(current),
            "prediction": {"input_descriptor": identity(current),
            "output": following, "output_descriptor": identity(following)}})
        events.append(("single", index, {"output": following}))
        current = following
    noise = current[:, 1:]
    prediction["readout"] = {"input_descriptor": identity(noise), "noise": noise, "noise_descriptor": identity(noise)}
    prediction["commit"] = {"input_descriptor": identity(latent), "noise_descriptor": identity(noise),
        "latent": noise, "latent_descriptor": identity(noise)}
    events += [("readout", None, {"noise": noise}), ("commit", None, {"latent": noise})]
    native = []
    for index, (kind, _, value) in enumerate(events):
        fields = (value if kind in ("project", "joint", "commit") else
            {"text": value["output"][:, :1], "image": value["output"][:, 1:]} if kind == "single" else
            {"noise_prediction": value["noise"]})
        native.append({"cut": "native-" + str(index), "point": {"local": index}, "values": fields})
    segment = {"boundaries": [None] * (28 * logical) + native}
    return prediction, events, segment, latent, conditioning


def run_synthetic(monkeypatch, prediction=None, *, logical=0, perturb_independent=False):
    original, events, segment, latent, conditioning = synthetic_pass(logical)
    prediction = original if prediction is None else prediction
    globals_ = A["audit_pass"].__globals__
    monkeypatch.setitem(globals_, "boundary", lambda e, record: record["values"])

    def independent(*args):
        for kind, index, value in events:
            if perturb_independent and kind == "commit":
                value = {"latent": value["latent"] + np.float32(1000)}
            yield kind, index, value

    monkeypatch.setitem(globals_["TWO"], "independent_pass_events", independent)
    return A["audit_pass"](None, {"carried_state": {"ids": "own"}}, prediction, {}, latent, segment,
        logical, conditioning, np.zeros((2, 1), np.float32), np.zeros((1, 1), np.float32), latent)


def test_complete_pass_checks_own_joins_at_initial_and_matched_origin(monkeypatch):
    for logical in (0, 2, 3):
        result, _ = run_synthetic(monkeypatch, logical=logical)
        assert result["logical_step"] == logical and len(result["transitions"]) == 28


@pytest.mark.parametrize("change", ["joint", "single", "project", "commit", "clock", "carried", "truncated"])
def test_stale_or_borrowed_state_is_mechanical_failure(monkeypatch, change):
    prediction = copy.deepcopy(synthetic_pass()[0])
    identity = A["MATH"]["identity"]
    foreign = identity(np.full((1, 2, 2), 7, np.float32))
    if change == "joint":
        prediction["joints"]["2"]["input_image_descriptor"] = foreign
    elif change == "single":
        prediction["single_steps"][4]["prediction"]["input_descriptor"] = foreign
    elif change == "project":
        prediction["project"]["input_latent_descriptor"] = foreign
    elif change == "commit":
        prediction["commit"]["noise_descriptor"] = foreign
    elif change == "clock":
        prediction["next_state_contract"]["clock"]["step"] = 4
    elif change == "carried":
        prediction["next_state_contract"]["carried_state"] = {"ids": "future"}
    else:
        prediction["single_steps"].pop()
    with pytest.raises(ValueError):
        run_synthetic(monkeypatch, prediction)


def test_large_independent_numerical_mismatch_is_retained_not_quality_gate(monkeypatch):
    result, independent = run_synthetic(monkeypatch, perturb_independent=True)
    assert len(result["transitions"]) == 28
    assert float(np.min(independent)) > 1000
    assert result["transitions"][-1]["fields"]["latent"]["sealed_minus_independent"]["max_abs"] == 1000


def test_each_sealed_history_contains_only_its_observed_origin_and_actual_clock_counts():
    binding = {"input_cut": "initial", "origins": {"full": {"cut": "actual-Z0"}, "matched": {"cut": "actual-Z2"}},
        "logical_steps": [0, 1, 2, 3], "bank": {"weights": "source-pinned"}}
    snapshots = {history: {"counts": counters(n, acts), "target_calls": A["target_profile"](origin),
        "target_calls_by_step": {str(i): A["target_profile"](int(i < origin)) for i in range(4)},
        "clock_qualified_targets_remaining": remaining(origin)}
        for history, origin, n, acts in (("full", 0, 0, 0), ("matched", 2, 56, 1))}
    full = A["history_binding"](binding, "full", snapshots["full"])
    matched = A["history_binding"](binding, "matched", snapshots["matched"])
    assert "origins" not in full and "actual-Z2" not in repr(full)
    assert full["origin"]["cut"] == "actual-Z0" and matched["origin"]["cut"] == "actual-Z2"
    assert matched["input_cut"] == "initial"  # Shared program identity is not the matched origin.
    assert matched["logical_steps"] == [2, 3] and matched["counts_before_seal"]["adapter_steps"] == 56


def test_independent_pass_to_pass_state_never_reconditions_on_sealed_or_native_future(monkeypatch):
    globals_ = A["audit_history"].__globals__
    times = np.arange(4, dtype=np.float32)
    sigmas = np.arange(4, -1, -1, dtype=np.float32)
    latent = np.zeros((1, 2, 2), np.float32)
    source = {"latent": latent, "conditioning": np.zeros((1, 1, 2), np.float32),
        "img_ids": np.zeros((2, 4), np.int64), "txt_ids": np.zeros((1, 4), np.int64), "timesteps": times, "sigmas": sigmas}
    monkeypatch.setitem(globals_, "check_origin", lambda *args: source)
    incoming = []

    def independent_pass(e, binding, predicted, schedule, computed, segment, logical, c, ids, tids, sealed):
        incoming.append((logical, computed.copy(), sealed.copy()))
        return {"logical_step": logical}, computed + 1

    monkeypatch.setitem(globals_, "audit_pass", independent_pass)
    binding = {"origins": {"full": {}}, "pass_schedules": {str(i):
        {"timestep": times[i], "sigma": sigmas[i], "sigma_next": sigmas[i + 1]} for i in range(4)}}
    class Reader:
        def array(self, ref):
            return ref
    forecast = {"origin_logical_step": 0, "chain_length": 112, "passes":
        [{"commit": {"latent": np.full_like(latent, 100 + i)}} for i in range(4)]}
    _, final = A["audit_history"](Reader(), binding, forecast, {"native_future": "must not be read"}, origin=0)
    assert [float(np.min(v[1])) for v in incoming] == [0, 1, 2, 3]
    assert [float(np.min(v[2])) for v in incoming] == [0, 100, 101, 102]
    assert np.array_equal(final, np.full_like(latent, 4))


@pytest.mark.parametrize("change", ["old-seed", "future-origin", "old-horizon", "boolean-census"])
def test_historical_or_wrong_future_metadata_cannot_qualify_fresh_plan(change):
    source = {"context": {"seed": 113, "id": "p1-s113"}, "execution": {
        "adapter": "mdb-flux-final-latent-closure-v1", "device": {"type": "cuda"}, "dtype": "torch.bfloat16"},
        "checkpoint_files": {"weights": "sha"}}
    source["checkpoint_sha256"] = A["BASE"]["canonical_hash"](source["checkpoint_files"])
    plan = {"context": copy.deepcopy(source["context"]), "logical_steps": [0, 1, 2, 3], "matched_logical_steps": [2, 3],
        "block_indices": list(range(20)), "selector": {"logical_step": 0, "phase": "project", "index": None},
        "matched_selector": {"logical_step": 2, "phase": "project", "index": None},
        "branch_execution_order": list(A["ORDER"]), "expected_checkpoint_sha256": source["checkpoint_sha256"]}
    summary = {"counts": dict(A["COUNTS"])}
    A["validate_plan"](plan, source, summary)
    if change == "old-seed":
        plan["context"]["seed"] = 109
    elif change == "future-origin":
        plan["selector"]["logical_step"] = 2
    elif change == "old-horizon":
        plan["logical_steps"] = [2, 3]
    else:
        summary["counts"]["model_loads"] = True
    with pytest.raises(ValueError):
        A["validate_plan"](plan, source, summary)


def test_cli_entry_is_after_every_audit_definition():
    syntax = ast.parse((HERE / "analyze_flux_initial_noise.py").read_text())
    guards = [node for node in syntax.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)]
    assert len(guards) == 1
    assert guards[-1].lineno > max(node.lineno for node in syntax.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))


@pytest.mark.parametrize("history,limit", [("full", 3 << 30), ("matched", 1536 << 20)])
def test_saved_verifier_receives_declared_large_geometry_and_exact_seal(tmp_path, monkeypatch, history, limit):
    calls = []
    class ReachedDeclaredBound(Exception):
        pass
    def geometry(root, *, expected_seal_sha256=None, max_bytes=1 << 30):
        assert (1 << 30) + 1 <= max_bytes, "offline transaction geometry exceeds bound"
        calls.append((root, expected_seal_sha256, max_bytes))
        raise ReachedDeclaredBound
    monkeypatch.setitem(A["verify_transaction"].__globals__["H"], "verify_transaction", geometry)
    summary = {"seals": {history: {"seal_sha256": "b" * 64}}}
    with pytest.raises(ReachedDeclaredBound):
        A["verify_transaction"](tmp_path, summary, {}, history, max_bytes=limit)
    assert calls == [(tmp_path / "transaction" / history, "b" * 64, limit)]


@pytest.fixture
def typed_int64_origin(tmp_path):
    """A real public typed CAS with tensor values, without any model instance."""
    import torch
    from saturn_pub import StateCut
    from saturn_pub.store import LocalStore

    from mdb_lanes.tensor_archive import write_tensor_archive

    ids = torch.tensor([-(2 ** 63), 2 ** 63 - 1, 2 ** 53 + 1, -(2 ** 53 + 1), 0, 1, 2, 3], dtype=torch.int64).reshape(1, 2, 4)
    payload = {"latent": torch.zeros((1, 2, 2), dtype=torch.bfloat16),
        "conditioning": torch.zeros((1, 1, 2), dtype=torch.bfloat16), "img_ids": ids,
        "txt_ids": ids[:, :1].clone(), "timesteps": torch.tensor([1000., 750., 500., 250.]),
        "sigmas": torch.tensor([1., .75, .5, .25, 0.]), "step": 0, "phase": "project", "cursor": 0,
        **dict.fromkeys(("context", "image", "text", "noise_prediction"))}
    point = {"logical_step": 0, "phase": "project", "index": None}
    cut = StateCut("a" * 64, {"fixture": "tensor-only"}, "fixture.project", None, payload, execution_point=point, surface={})
    LocalStore(tmp_path / "typed-store").save(cut)
    origin = {"cut": cut.fingerprint, "point": point,
        "inputs": {key: write_tensor_archive(tmp_path / (key + ".npy"), payload[key]) for key in A["SLOTS"]}}
    return A["Evidence"](tmp_path, tmp_path / "unused-parameter-cache", {}), origin, payload


def test_modern_int64_ids_remain_lossless_and_match_actual_typed_cut(typed_int64_origin):
    e, origin, payload = typed_int64_origin
    result = A["check_origin"](e, origin, 0)
    for key in ("img_ids", "txt_ids"):
        assert origin["inputs"][key]["encoding"] == "native-npy-v1"
        assert result[key].dtype == np.dtype("int64")
        assert result[key].tobytes() == payload[key].numpy().tobytes()
        assert A["exact"](result[key], e.slot(origin["cut"], key))
        assert not result[key].flags.writeable
    assert result["img_ids"][0, 0, 2] == 2 ** 53 + 1


@pytest.mark.parametrize("corruption", ["file-hash", "native-hash", "dtype", "shape", "valid-foreign-ids", "unknown-encoding"])
def test_modern_id_corruption_or_valid_foreign_ids_cannot_pass_typed_join(typed_int64_origin, corruption):
    import torch

    from mdb_lanes.tensor_archive import write_tensor_archive

    e, original, payload = typed_int64_origin
    origin = copy.deepcopy(original)
    record = origin["inputs"]["img_ids"]
    if corruption == "file-hash":
        record["sha256"] = "0" * 64
    elif corruption == "native-hash":
        record["descriptor"]["sha256"] = "0" * 64
    elif corruption == "dtype":
        record["descriptor"]["dtype"] = "torch.float32"
    elif corruption == "shape":
        record["stored_shape"] = [1, 4, 2]
    elif corruption == "unknown-encoding":
        record["encoding"] = "unqualified-id-conversion"
    else:
        changed = payload["img_ids"].clone()
        changed[0, 1, 0] = torch.tensor(987654321, dtype=torch.int64)
        origin["inputs"]["img_ids"] = write_tensor_archive(e.data / "foreign-ids.npy", changed)
    with pytest.raises(ValueError):
        A["check_origin"](e, origin, 0)


def test_unencoded_legacy_float_witness_fallback_remains_supported(typed_int64_origin):
    e, origin, payload = typed_int64_origin
    record = copy.deepcopy(origin["inputs"]["timesteps"])
    del record["encoding"]
    actual = e.array(record)
    assert actual.dtype == np.float32 and actual.tobytes() == payload["timesteps"].numpy().tobytes()


@pytest.fixture
def correction_fixture(tmp_path, monkeypatch):
    run, helpers = tmp_path / "run", tmp_path / "corrected"
    frozen = run / "independent-math-freeze"
    frozen.mkdir(parents=True)
    helpers.mkdir()
    own = "analyze_flux_initial_noise.py"
    (frozen / own).write_text("original immutable reader")
    (helpers / own).write_text("corrected reader")
    (frozen / "math.py").write_text("unchanged mathematics")
    (helpers / "math.py").write_text("unchanged mathematics")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cohort = {p.name: sha(p) for p in frozen.iterdir()}
    request, job = {"request_sha256": "a" * 64, "independent_math_sources": cohort}, {"job_id": "job-fixture"}
    globals_ = A["audit_source_authority"].__globals__
    monkeypatch.setitem(globals_, "HERE", helpers)
    # Pin-validation tests isolate the separately tested AST comparison.
    monkeypatch.setitem(globals_, "reader_ast_proof", lambda *_: {"fixture": "unchanged"})
    def reference(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": sha(path)}
    failed = {key: reference(key + ".json", {"output": key}) for key in ("stdout", "stderr")}
    failed["receipt"] = reference("failed-receipt.json", {"returncode": 1, "model_calls": 0,
        "source_before_sha256": cohort[own], "source_after_sha256": cohort[own],
        "stdout_sha256": failed["stdout"]["sha256"], "stderr_sha256": failed["stderr"]["sha256"],
        "command": [str(frozen / own), str(run)]})
    record = {"schema": "mdb-flux-initial-noise-reader-correction-v1", "job_id": job["job_id"],
        "request_sha256": request["request_sha256"], "original_auditor_sha256": cohort[own],
        "corrected_auditor_sha256": sha(helpers / own), "original_cohort_sha256s": cohort,
        "model_calls": 0, "producer_files_changed": False, "scope": "lossless-modern-native-reader-only",
        "failed_attempt": failed,
        "qualification": reference("qualification.json", {"returncode": 0, "analyzer_sha256": sha(helpers / own)}),
        "ast_proof": {"fixture": "unchanged"}}
    path = tmp_path / "correction.json"
    path.write_text(json.dumps(record))
    return run, helpers, request, job, record, path


def test_reader_correction_is_explicit_and_leaves_original_cohort_strict(correction_fixture):
    run, _, request, job, _, path = correction_fixture
    with pytest.raises(ValueError, match="explicit correction"):
        A["audit_source_authority"](run, request, job, None)
    result = A["audit_source_authority"](run, request, job, path)
    assert result["record"]["model_calls"] == 0 and "post-outcome" in result["status"]


@pytest.mark.parametrize("field", ["original_auditor_sha256", "corrected_auditor_sha256", "request_sha256", "job_id"])
def test_wrong_correction_identity_refuses(correction_fixture, field):
    run, _, request, job, record, path = correction_fixture
    record[field] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity/scope"):
        A["audit_source_authority"](run, request, job, path)


@pytest.mark.parametrize("change", ["frozen", "unrelated", "failed-reference", "proof"])
def test_correction_never_authorizes_other_source_or_evidence_changes(correction_fixture, change):
    run, helpers, request, job, record, path = correction_fixture
    if change == "frozen":
        (run / "independent-math-freeze/math.py").write_text("changed")
    elif change == "unrelated":
        (helpers / "math.py").write_text("changed")
    elif change == "failed-reference":
        Path(record["failed_attempt"]["stderr"]["path"]).write_text("rewritten failure")
    else:
        record["ast_proof"] = {"fake": "claim"}
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        A["audit_source_authority"](run, request, job, path)


def test_ast_proof_rejects_changed_math_or_non_source_custody_control():
    old = '''class Evidence:
    def array(self, record):
        return record
def equation(value):
    return value * value
def custody(run):
    for name, digest in request["independent_math_sources"].items():
        require(sha(run / "independent-math-freeze" / name) == sha(HERE / name) == digest, "changed")
    return run
def analyze(run, *, parameter_root=None, checkpoint=None, ssh=None):
    data, request, job, collection, inventory = custody(run)
    return {"scope": "fixture"}
'''
    new = '''class Evidence:
    def array(self, record):
        return load_lossless(record)
def equation(value):
    return value * value
def reader_ast_proof(a, b):
    pass
def audit_source_authority(a, b, c, d):
    pass
def custody(run, *, audit_correction=None):
    audit_source_authority(run, request, job, audit_correction)
    return run
def analyze(run, *, parameter_root=None, checkpoint=None, ssh=None, audit_correction=None):
    data, request, job, collection, inventory = custody(run, audit_correction=audit_correction)
    authority = audit_source_authority(run, request, job, audit_correction)
    return {"scope": "fixture", "preoutcome_auditor_sha256": request["independent_math_sources"]["analyze_flux_initial_noise.py"], "audit_correction": authority}
'''
    assert A["reader_ast_proof"](old, new)["analyze_control_unchanged"]
    with pytest.raises(ValueError, match="equation/control AST"):
        A["reader_ast_proof"](old, new.replace("value * value", "value * value * value"))
    with pytest.raises(ValueError, match="custody correction"):
        A["reader_ast_proof"](old, new.replace("    return run", "    return altered"))
    with pytest.raises(ValueError, match="top-level"):
        A["reader_ast_proof"](old, "MATH = changed_authority\n" + new)
