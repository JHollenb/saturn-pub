"""Own-input/clock/prospective-history regressions; no trained model or weights."""
from __future__ import annotations

import ast
import copy
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
