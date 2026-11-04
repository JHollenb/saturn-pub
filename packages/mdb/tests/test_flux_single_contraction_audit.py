"""Model-free checks for contraction evidence and hostile custody mutations."""
from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_single_contraction.py"))
QUANTIZE = runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py"))["quantize_bf16"]


@pytest.mark.parametrize("selection", ["native", "A", "M", "AM"])
def test_only_declared_image_feature_partition_can_change(selection):
    original = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    donor = original + 100
    used = original.copy()
    if selection in {"A", "AM"}:
        used[:, 2:, :2] = donor[:, 2:, :2]
    if selection in {"M", "AM"}:
        used[:, 2:, 2:] = donor[:, 2:, 2:]
    assert all(A["feature_selection"](original, used, donor, text_rows=2, attention_width=2,
                                       selection=selection).values())
    used[0, 0, 0] += 1
    with pytest.raises(ValueError, match="undeclared recipient partition"):
        A["feature_selection"](original, used, donor, text_rows=2, attention_width=2, selection=selection)


def test_feature_recipe_cannot_hide_changed_nonselected_mlp():
    original = np.zeros((1, 3, 4), np.float32)
    donor = np.ones_like(original)
    used = original.copy()
    used[:, 1:, :2] = 1
    used[0, 1, 3] = 1
    with pytest.raises(ValueError, match="undeclared recipient partition"):
        A["feature_selection"](original, used, donor, text_rows=1, attention_width=2, selection="A")


def test_contraction_identity_uses_actual_rounded_update_and_retains_amplification():
    low_r = np.array([0, 0], np.float32)
    high_r = np.array([3, 4], np.float32)
    low_y = np.array([1, 0], np.float32)
    high_y = np.array([1, 2], np.float32)
    result = A["contraction_identity"](low_r, high_r, low_y, high_y)
    assert result["cross_term"] == -34
    assert result["update_energy"] == 13
    assert result["observed_squared_distance_change"] == -21
    assert result["identity_residual"] == result["vector_closure"]["l2"] == 0
    assert result["signed_l2_change"] == -3
    result = A["contraction_identity"](low_r, high_r, low_y, np.array([7, 8], np.float32))
    assert result["signed_l2_change"] > 0  # a missed contraction remains an observation
    assert "No attention/MLP attribution" in result["scope"]


def test_sampled_projection_has_independent_dense_equation_and_explicit_rounds():
    low = np.array([[[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]]], np.float32)
    high = low + 8
    weights = np.array([[1, 0, 1, 0], [0, 1, 0, -1]], np.float32)
    residual = np.full((1, 3, 2), 2, np.float32)
    gate = np.full((1, 1, 2), .5, np.float32)
    result = A["sampled_projection"](low, high, weights, np.array([1, 2], np.float32), residual, gate,
        text_rows=1, attention_width=2, selection="A", query_indices=[0, 1], output_indices=[0, 1], quantize=QUANTIZE)
    # A uses high attention but low MLP: [9,10,3,4] and [13,14,7,8].
    expected = np.array([[[13, 8], [21, 8]]], np.float64)
    assert np.array_equal(result["projection_fp64"], expected)
    assert np.array_equal(result["updated_bf16"], np.array([[[8.5, 6], [12.5, 6]]], np.float32))
    with pytest.raises(ValueError, match="operands"):
        A["sampled_projection"](low, high, weights, None, residual, gate,
            text_rows=1, attention_width=2, selection="A", query_indices=[0, 0], output_indices=[0, 1], quantize=QUANTIZE)


def test_factorial_vector_and_endpoint_misses_are_not_distance_algebra_or_gates():
    result = A["factorial_interaction"](np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([2, 3]))
    assert result["l2"] == np.sqrt(5)
    comparisons = {}
    for name, value in [("zero", 4), ("A.image", 3), ("M.image", 6), ("AM.image", 5), ("text.text", 4)]:
        comparisons[name] = {"text": {"latent": {"base_l2": value}, "rgb": {"base_mae": value / 2}}}
    trends = A["endpoint_trends"](comparisons)
    assert trends["text_direction"]["A.image"]["latent"] == -1
    assert trends["image_only_relative_to_low"]["M.image"]["latent"] == 2
    assert "no quality or monotonicity gate" in trends["scope"]


def test_local_counterfactual_identity_separates_precision_and_residual_terms():
    values = [np.array(x, np.float32) for x in ([0, 0], [3, 4], [1, 2], [8, 9], [2, 2], [1, 4], [2, 5])]
    result = A["local_vector_account"](*values)
    assert result["identity_residual"]["l2"] == 0
    assert result["vectors"]["j"]["l2"] == 1
    assert result["vectors"]["epsilon"]["l2"] == 3
    assert "local j reflects precision/rounding" in result["scope"]


def inventory_fixture(tmp_path):
    targets = {"input-store/cuts/cut.json": "typed-store/cuts/cut.json", "prior-source.json": "source.json"}
    source_files, manifest = {}, {}
    for target, original in targets.items():
        path = tmp_path / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(target.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_files[target] = {"prior_path": original, "sha256": digest, "bytes": path.stat().st_size}
        manifest[original] = digest
    selection = {"files": {"input-store/cuts/cut.json": source_files["input-store/cuts/cut.json"]["sha256"]}, "source_files": source_files}
    return selection, manifest


def test_imported_bytes_must_join_pinned_original_manifest(tmp_path):
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    selection, manifest = inventory_fixture(tmp_path)
    assert A["verify_source_inventory"](tmp_path, selection, manifest, h)["original_manifest_joined_files"] == 2
    path = tmp_path / "prior-source.json"
    path.write_bytes(b"tamper with self-consistent replacement digest")
    selection["source_files"][path.name].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="original producer manifest"):
        A["verify_source_inventory"](tmp_path, selection, manifest, h)


def test_extra_cas_and_unsafe_original_path_are_rejected(tmp_path):
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    selection, manifest = inventory_fixture(tmp_path)
    (tmp_path / "input-store/extra").write_bytes(b"x")
    with pytest.raises(ValueError, match="extra or missing"):
        A["verify_source_inventory"](tmp_path, selection, manifest, h)
    (tmp_path / "input-store/extra").unlink()
    selection["source_files"]["prior-source.json"]["prior_path"] = "../source.json"
    with pytest.raises(ValueError, match="unsafe original"):
        A["verify_source_inventory"](tmp_path, selection, manifest, h)


def test_native_byte_closure_cannot_drop_component_or_rename_colliding_source():
    assert A["native_implementation_closure"]({"/new/a.py": "a"}, {"/old/a.py": "a"}) == {"a.py": "a"}
    with pytest.raises(ValueError, match="byte closure"):
        A["native_implementation_closure"]({"/new/a.py": "a"}, {"/old/a.py": "a", "/old/b.py": "b"})
    with pytest.raises(ValueError, match="ambiguous"):
        A["native_implementation_closure"]({"/one/a.py": "a", "/two/a.py": "b"}, {"a.py": "a"})


def timing_fixture():
    counts = {**A["COUNTS"], "model_loads": 0, "adapter_steps": 6, "candidate_calls": 0, "receipt_emissions": 6,
              "acts": 0, "images": 0, "vae_calls": 0}
    events = [{"kind": "hypotheses_sealed", "counts": dict(counts), "blocked_native_entries": 0}]
    for name in ("zero", "text"):
        events.extend([{"kind": "candidate_start", "candidate": name}, {"kind": "candidate_complete", "candidate": name}])
    events.append({"kind": "projection_forecast_sealed", "counts": {**counts, "candidate_calls": 2}})
    for name in ("A", "M", "AM"):
        events.extend([{"kind": "candidate_start", "candidate": name}, {"kind": "candidate_complete", "candidate": name}])
    return [{"sequence": i, **event} for i, event in enumerate(events)]


def test_timing_allows_declared_source_replay_but_retains_early_target_miss():
    events = timing_fixture()
    assert A["projection_timing"](events)["qualified"]
    events[0]["counts"]["candidate_calls"] = 1
    result = A["projection_timing"](events)
    assert not result["qualified"] and not result["checks"]["main_after_only_six_allowed_replay_steps"]
    events = timing_fixture()
    events[5], events[6] = events[6], events[5]
    events = [{**event, "sequence": i} for i, event in enumerate(events)]
    assert not A["projection_timing"](events)["qualified"]


def test_prior_boundary_selection_uses_next_frontier_not_list_position():
    entries = [{"point": {"logical_step": 3, "phase": "single", "index": i}, "cut": str(i)} for i in (2, 3, 6, 7)]
    audit = {"trajectory": {"references": {"zero": entries}}}
    assert A["prior_reference"](audit, "zero", 6)["cut"] == "6"
    entries.append(entries[2])
    with pytest.raises(ValueError, match="ambiguous"):
        A["prior_reference"](audit, "zero", 6)


def test_prior_first_boundary_is_authenticated_by_prefix_not_suffix_only():
    audit = {"paired_source_checks": {"prefixes": {"zero": {"receipts": ["prefix"]}}},
             "native_paths": {"zero": {"receipts": ["suffix1", "suffix2"]}}}
    known = A["prior_boundary_receipts"](audit, "zero")
    assert known == {"prefix", "suffix1", "suffix2"}
    assert "foreign" not in known


def test_current_custody_verified_once_without_inventing_child_report(tmp_path):
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    data = tmp_path / "extracted/flux-single-contraction-result"
    data.mkdir(parents=True)
    tx = ForecastTransaction(data / "transaction", model_identity="fixture", input_identity="fixture-input",
                             source_identity="fixture-source", frontier={"kind": "architectural"})
    with tx:
        tx.predict("contraction_hypotheses", lambda _: {"predictions": {"fixture": True}})
        seal = tx.seal()
        tx.execute("native", lambda _: {"fixture": True})
    plan = {"single_index": 6, "selector": {"logical_step": 3, "phase": "single", "index": 6}}
    request = {"config": plan, "request_sha256": "current-request", "source_hashes": {}}
    source = {"job_id": "current-job", "request_sha256": "current-request", "source_hashes": {}, "model_identity": "fixture"}
    summary = {"job_id": "current-job", "request_sha256": "current-request", "model_identity": "fixture",
        "counts": {**A["COUNTS"], "model_loads": 0}, "seal": seal,
        "transaction_verification": verify_transaction(data / "transaction", expected_seal_sha256=seal["seal_sha256"])}
    summary["transaction_verification"]["identity_scope"] = "local-hash-consistency"
    save(data / "plan.json", plan)
    save(data / "summary.json", summary)
    outer = {**summary, "counts": dict(A["COUNTS"]), "panel_summary_sha256": A["sha_file"](data / "summary.json")}
    for name, value in [("outer-summary.json", outer), ("source.json", source),
        ("plan-frozen.json", {"request_sha256": "current-request", "plan_sha256": A["sha_file"](data / "plan.json"),
                              "model_loads": 0, "adapter_steps": 0, "candidate_calls": 0})]:
        save(data / name, value)
    save(tmp_path / "request.json", request)
    save(tmp_path / "job.json", {"job_id": "current-job", "state": "succeeded"})
    envelope = {"job_id": "current-job", "request_sha256": "current-request", "archive_root": "flux-single-contraction-result", "report": outer}
    save(tmp_path / "report.json", envelope)
    calls = []
    def verify(root, *, limits):
        calls.append(root)
        return {"mechanics_status": "verified", "job_id": "current-job", "request_sha256": "current-request"}
    result = A["current_parent"](tmp_path, custody_verify=verify)
    assert calls == [tmp_path.resolve()]
    assert result["summary"]["counts"]["model_loads"] == 0 and result["outer"]["counts"]["model_loads"] == 1
    assert result["transaction"]["identity_scope"] == "externally-pinned"
    assert result["saved_worker_verification"]["identity_scope"] == "local-hash-consistency"
    envelope["job_id"] = "fabricated-child"
    save(tmp_path / "report.json", envelope)
    with pytest.raises(ValueError, match="custody/job/archive"):
        A["current_parent"](tmp_path, custody_verify=verify)


def test_transaction_scope_upgrade_excludes_no_transaction_facts():
    saved = {"schema": "mdb-forecast-offline-verification-v1", "verified": True, "seal_sha256": "seal",
             "event_head": "head", "events": 7, "phase": "closed", "frontiers": ["source_binding"],
             "forecasts": ["sampled_projection"], "identity_scope": "local-hash-consistency"}
    actual = {**saved, "identity_scope": "externally-pinned"}
    assert A["transaction_verifications_match"](actual, saved)
    for key, changed in [("seal_sha256", "other"), ("event_head", "other"), ("events", 8),
                         ("verified", False), ("phase", "errored"), ("forecasts", []), ("frontiers", [])]:
        assert not A["transaction_verifications_match"]({**actual, key: changed}, saved)
    assert not A["transaction_verifications_match"](actual, {**saved, "identity_scope": "unknown"})
    assert not A["transaction_verifications_match"](saved, saved)
    assert not A["transaction_verifications_match"]({**actual, "extra": True}, saved)


@pytest.mark.parametrize("roles", [("zero", "text"), ("text", "zero")])
def test_physical_census_accepts_either_complete_replay_order_not_reordered_steps(tmp_path, roles):
    replay = {role: {"receipts": [role + str(i) for i in range(3)], "native_steps": 3} for role in roles}
    prefix = {role: {"receipts": ["p" + role], "native_steps": 1} for role in roles}
    acts = [0, 0, 2, 1, 1, 1, 1, 2, 2]
    paths = {name: {"receipts": [name + str(i) for i in range(15 + n)], "native_steps": 15, "acts": n}
             for name, n in zip(A["ROW_IDS"], acts)}
    plan = {"branches": [{"id": name} for name in A["ROW_IDS"]]}
    pairs = [("replay." + role, rid) for role in roles for rid in replay[role]["receipts"]]
    pairs += [("prefix." + role, rid) for role in ("zero", "text") for rid in prefix[role]["receipts"]]
    pairs += [(name, rid) for name in A["ROW_IDS"] for rid in paths[name]["receipts"]]
    def write(pairs):
        (tmp_path / "receipt-emissions.jsonl").write_text("".join(json.dumps({"sequence": i, "branch": branch, "receipt": rid}) + "\n"
            for i, (branch, rid) in enumerate(pairs)))
    write(pairs)
    result = A["emission_census"](SimpleNamespace(data=tmp_path), replay, prefix, paths, {"counts": A["COUNTS"]}, plan)
    assert result["source_replay_role_order"] == ["replay." + role for role in roles]
    pairs[0], pairs[1] = pairs[1], pairs[0]
    write(pairs)
    with pytest.raises(ValueError, match="physical receipt"):
        A["emission_census"](SimpleNamespace(data=tmp_path), replay, prefix, paths, {"counts": A["COUNTS"]}, plan)


def test_auditor_is_framework_free():
    code = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common'}:raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1])
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(HERE / "analyze_flux_single_contraction.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
