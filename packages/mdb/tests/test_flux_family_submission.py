"""Finite resident-family admission and offline analysis mechanics, no models."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mdb.job_support
import mdb.payload

QUALIFICATION = Path(__file__).parents[1] / "qualification"


def frozen_plan(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"model": "flux2-klein-4b-distilled", "model_path": "/explicit/pinned/model",
                                 "expected_checkpoint_sha256": "a" * 64}))
    module = runpy.run_path(str(QUALIFICATION / "submit_flux_family.py"))
    plan = module["make_plan"](prior)
    path = tmp_path / "campaign.json"
    module["save"](path, plan)
    return module, path, plan


def test_one_resident_job_keeps_all_rows_offline_wheels_and_physical_memory_margin(tmp_path, monkeypatch):
    module, path, plan = frozen_plan(tmp_path)
    staged, calls = {}, []

    def stage(_path, **kwargs):
        staged.update(kwargs)

    def launch(command, **kwargs):
        calls.append(kwargs)
        assert "--offline" in command
        assert kwargs["ram_mb"] == 18432 and kwargs["vram_mb"] == 13000 and kwargs["cpu_threads"] == 4
        assert kwargs["timeout_s"] == 3300 and kwargs["est_wall_s"] == 1800
        assert kwargs["preflight"] is True and kwargs["retry_on_kill"] is False
        assert kwargs["gpu"] and kwargs["needs"] == {"cuda": True} and kwargs["prefer_host"] == "beast"
        request = kwargs["request"]
        assert request == staged["request"]
        assert json.loads(staged["config_bytes"]) == staged["config"] == plan
        assert request["batch_size"] == 1 and request["dtype"] == "bfloat16"
        contract = request["scheduling_contract"]
        assert contract["model_loads"] == 1 and contract["logical_branches"] == 47 and contract["max_local_work_items"] == 64
        assert set(module["WORKERS"][1:]) <= {name for _, name in staged["additional_files"]}
        assert len(kwargs["note"]) <= 500
        memory = request["resource_exception"]
        from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

        assert memory["ram_kill_factor"] == RAM_KILL_FACTOR
        assert memory["ram_kill_ceiling_mb"] == kill_ceiling_mb(18432) < memory["reference_host_available_mb"]
        unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
        assert hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"]
        return "job-tiny-no-real-submission"

    monkeypatch.setattr(mdb.payload, "stage_worker", stage)
    monkeypatch.setattr(mdb.payload, "worker_command", lambda name: ["uv", "run", "--offline", "--with", "./public.whl", "python", name])
    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", launch)
    result = module["stage"](tmp_path / "run", path)
    assert result["rows"] == 47 and result["model_loads"] == len(calls) == 1
    assert json.loads((tmp_path / "run/submission.json").read_text())["job_id"] == result["job_id"]


def test_frozen_rows_cannot_be_silently_pruned_before_admission(tmp_path, monkeypatch):
    module, path, plan = frozen_plan(tmp_path)
    altered = copy.deepcopy(plan)
    altered["rows"].pop()
    path.write_text(json.dumps(altered))
    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", lambda *_args, **_kwargs: pytest.fail("invalid campaign launched"))
    with pytest.raises(ValueError, match="geometry"):
        module["stage"](tmp_path / "run", path)
    assert not (tmp_path / "run").exists()


def test_ram_kill_ceiling_not_just_reservation_must_fit():
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    resource = runpy.run_path(str(QUALIFICATION / "submit_flux_family.py"))["resource_envelope"]
    assert resource()["ram_kill_factor"] == RAM_KILL_FACTOR
    assert resource()["ram_kill_ceiling_mb"] == kill_ceiling_mb(18432)
    assert resource(kill_factor=1.3)["ram_kill_ceiling_mb"] == pytest.approx(23961.6)
    with pytest.raises(ValueError, match="kill ceiling"):
        resource(55000, kill_factor=1.3)


def test_direct_collection_uses_exact_job_and_declared_finite_custody_limits(tmp_path, monkeypatch):
    module, _, plan = frozen_plan(tmp_path)
    (tmp_path / "request.json").write_text(json.dumps({"config": plan}))
    (tmp_path / "submission.json").write_text(json.dumps({"job_id": "job-exact"}))
    calls = []

    def collect(root, job, limits):
        calls.append((root, job, limits))
        return {"collection_status": "verified"}

    monkeypatch.setitem(module["collect"].__globals__, "runpy", SimpleNamespace(run_path=lambda _path: {"collect": collect}))
    assert module["collect"](tmp_path, "job-exact")["collection_status"] == "verified"
    assert calls == [(tmp_path, "job-exact", plan["limits"])]
    with pytest.raises(ValueError, match="exact submitted"):
        module["collect"](tmp_path, "job-retry")
    assert len(calls) == 1


def test_all_six_panels_and_holdout_references_remain_in_frozen_plan(tmp_path):
    _, _, plan = frozen_plan(tmp_path)
    assert len(plan["histories"]) == 9 and len(plan["prompts"]) == 8
    assert len(plan["rows"]) == len({row["id"] for row in plan["rows"]}) == 47
    assert sum(row["split"] == "development" for row in plan["rows"]) == 36
    assert sum(row["split"] == "heldout" for row in plan["rows"]) == 11
    assert {panel for row in plan["rows"] for panel in row["panels"]} == set(range(1, 7))
    assert {row["id"] for row in plan["rows"] if row["kind"] == "reference" and row["split"] == "heldout"} == {
        "ref.novel.base", "ref.novel.relation", "ref.triple.base", "ref.triple.relation"}


def test_both_resolvers_ignore_heldout_outcomes_and_keep_far_case_abstention():
    rules = runpy.run_path(str(QUALIFICATION / "flux_family_plan.py"))
    analyzer = runpy.run_path(str(QUALIFICATION / "analyze_flux_family.py"))
    keys = set(rules["FEATURES"]) | set(rules["RECIPIENT_FEATURES"])
    pairs = [{"id": name, "split": split} for name, split in
             (("dev-a", "development"), ("dev-z", "development"), ("near", "heldout"), ("far", "heldout"))]
    signatures = {name: dict.fromkeys(keys, value) for name, value in
                  (("dev-a", 0.), ("dev-z", 2.), ("near", 1.), ("far", 100.))}
    targets = {"dev-a": {"R": 1., "A": .5, "RA": -.2}, "dev-z": {"R": -.3, "A": .9, "RA": .7},
               "near": {"R": -999., "A": -999., "RA": -999.}, "far": {"R": 999., "A": 999., "RA": 999.}}
    first = analyzer["resolver_forecasts"](pairs, signatures, targets, rules)
    targets["near"] = targets["far"] = {"R": 12345., "A": -12345., "RA": None}
    second = analyzer["resolver_forecasts"](pairs, signatures, targets, rules)
    assert first == second
    for field in ("model", "recipient_only_model"):
        assert [sample["pair_id"] for sample in first[field]["samples"]] == ["dev-a", "dev-z"]
    for field in ("predictions", "recipient_only_predictions"):
        assert first[field]["near"]["nearest_pair"] == "dev-a"  # exact distance tie uses pair ID
        assert first[field]["near"]["predictions"] == targets["dev-a"]
        assert first[field]["far"]["status"] == "abstain"
        assert first[field]["far"]["predictions"] is None
        assert first[field]["far"]["candidate_predictions"] == targets["dev-z"]


def test_operand_selection_changes_only_declared_text_keys_and_heads():
    select = runpy.run_path(str(QUALIFICATION / "analyze_flux_family.py"))["selected_qkv"]
    base = {key: np.arange(1 * 5 * 3 * 2, dtype=np.float32).reshape(1, 5, 3, 2) + offset
            for key, offset in (("q", 0), ("k", 100), ("v", 200))}
    changed = {key: value + 1000 for key, value in base.items()}
    original = {key: value.copy() for key, value in changed.items()}
    result = select(base, changed, 2, {"k_text": "base", "v_text": "base", "heads": [1], "text_positions": [0]})
    for key in result:
        expected = original[key].copy()
        if key in {"k", "v"}:
            expected[:, 0, 1] = base[key][:, 0, 1]
        np.testing.assert_array_equal(result[key], expected)
        np.testing.assert_array_equal(changed[key], original[key])
        assert not np.shares_memory(result[key], changed[key])


def test_prefix_features_exclude_text_and_preserve_signed_alignment():
    analyzer = runpy.run_path(str(QUALIFICATION / "analyze_flux_family.py"))
    signature = analyzer["prefix_signature"]
    residual = np.array([[[999., 999.], [3., 4.], [0., 0.]]], dtype=np.float32)
    a = {"residual": residual, "operands": np.concatenate([residual, residual], axis=-1)}
    b = {key: -value for key, value in a.items()}
    values = signature(a, b, 2, 1)
    assert values["R_base_rms"] == 2.5
    assert values["R_delta_rms"] == 5. and values["R_symmetric_relative_rms_delta"] == 1.
    assert values["R_base_changed_cosine"] == -1. and values["delta_R_A_cosine"] == 1.
    a["residual"][:, 0] = -12345.
    a["operands"][:, 0] = 54321.
    assert signature(a, b, 2, 1) == values
    zero = {key: np.zeros_like(value) for key, value in a.items()}
    null = signature(zero, zero, 2, 1)
    assert null["R_base_changed_either_zero"] == 1. and null["R_symmetric_relative_rms_delta"] == 0.
    changed_reduction = dict(values, R_base_rms=values["R_base_rms"] + .01)
    diagnostic = analyzer["feature_diagnostics"](values, changed_reduction)
    assert diagnostic["max_abs_recompute_delta"] > .009
    assert diagnostic["recorded"] == values  # even a large finite difference remains visible, not gated
    with pytest.raises(ValueError, match="nonfinite"):
        analyzer["feature_diagnostics"](values, dict(values, R_base_rms=float("nan")))


def test_hybrid_natural_diagnostics_retain_signed_and_geometry_differences():
    diagnostic = runpy.run_path(str(QUALIFICATION / "analyze_flux_family.py"))["array_diagnostic"]
    natural = np.array([0, 250], dtype=np.uint8)
    hybrid = np.array([255, 0], dtype=np.uint8)
    difference = diagnostic(hybrid, natural)
    assert difference["exact"] is False and difference["max_abs"] == 255.
    assert difference["signed_mean"] == 2.5  # subtraction must not wrap unsigned pixels
    assert diagnostic(natural, natural)["exact"] is True
    assert diagnostic(np.zeros((2, 3)), np.zeros((3, 2)))["status"] == "different-geometry"
