"""Synthetic saved-array regressions for full-readout audit claim boundaries."""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

QUALIFICATION = Path(__file__).parents[1] / "qualification"


@pytest.fixture
def saved_assay(tmp_path, monkeypatch):
    """Mock already-verified base custody; exercise new head/readout checks."""
    module = runpy.run_path(str(QUALIFICATION / "analyze_qwen_full_readout.py"))
    base_module = runpy.run_path(str(QUALIFICATION / "analyze_qwen_mlp_forecast.py"))
    helper = dict(base_module["helpers"]())
    data = tmp_path / "extracted/qwen-mlp-result"
    data.mkdir(parents=True)
    tx = data / "transaction"
    weight_sha, files_sha, source_sha = "a" * 64, "b" * 64, "c" * 64
    binding = {"native_parameter": "lm_head.weight", "descriptor": {"shape": [3, 2], "dtype": "torch.float32", "sha256": weight_sha},
        "matrix_serialized": False, "copied_before_prefixes": True, "readonly_copy_sha256": weight_sha,
        "checkpoint_weights_sha256": weight_sha, "checkpoint_files_sha256": files_sha,
        "extraction_recipe": {"callable_source_sha256": source_sha, "parameter_name": "lm_head.weight", "shape": [3, 2],
                              "token_id_row_order": {"kind": "identity", "start": 0, "stop": 3, "step": 1}},
        "final_rmsnorm_contract": {"module": "model.norm", "type": "fixture.Qwen2RMSNorm", "bias": None,
            "axis": -1, "mean_square_divisor": 2, "weight_shape": [2], "weight_dtype": "torch.float32", "epsilon": .01}}
    proof = {"descriptor": copy.deepcopy(binding["descriptor"]), "exact_preseal_binding": True,
             "readonly_copy_sha256": weight_sha, "checkpoint_weights_sha256": weight_sha}
    documents = {data / "full-head-binding.json": binding,
                 **{data / ("full-head-" + stage + ".json"): copy.deepcopy(proof)
                    for stage in ("pre-prediction", "at-release", "post-execution")}}
    expected = np.array([[[3., 2., 1.]]], dtype=np.float64)
    predicted = {"logits": expected, "token_id": 0, "runner_up_token_id": 1, "normalizer": 1., "top2gap": 1.}
    full = {"a": np.array([3., 2., 1.]), "b": np.zeros(3), "vocab_size": 3, "rms_epsilon": .01,
        "max_chunk_rows": 2, "doses": {"1": predicted}, "envelope": {"domain": [0., 1.],
            "segments": [{"left": 0., "right": 1., "token_id": 0}], "breakpoints": [
                {"dose": 0., "token_id": 0, "tied_token_ids": [0]}, {"dose": 1., "token_id": 0, "tied_token_ids": [0]}]}}
    forecast = {"full_vocabulary": {"prompt": full}, "rows": {"prompt": {"cut": np.zeros((1, 1, 2)),
                "rms_epsilon": .01, "doses": {"1": {"logits": expected[:, :, [0, 2]]}}}}}
    observed = {"prompt_id": "prompt", "arm": "native", "dose": 1.,
                "full_vocabulary_logits": expected.astype(np.float32), "selected_logits": expected[:, :, [0, 2]].astype(np.float32),
                "full_vocab_greedy_token": 0, "full_vocab_runner_up_token": 1, "full_vocab_top_two_margin": 1.,
                "greedy_text": "observed only", "generated_text": "observed future", "generated_token_ids": [0, 2]}
    base = {"mechanics_status": "valid", "source": {"job_id": "synthetic-saved-arrays"},
        "model_identity": {"weights_sha256": weight_sha, "checkpoint_files_sha256": files_sha},
        "execution": {"backend": "cpu", "dtype": "float32"}, "candidate_ids": [0, 2],
        "counts": {}, "prefixes": {}, "pre_release_counters": {}, "mechanics_errors": [], "checkpoint_files": {},
        "installed_model_sources": {}, "native_weight_binding": {}, "rows": {"prompt.native": {"continuation_check": {}}}, "limits": []}
    run = {"data": data, "transaction": tx, "tree": SimpleNamespace(decode=lambda value: value),
           "plan": {"full_vocabulary_readout": True}, "request": {"source_hashes": {"qwen_mlp_forecast_campaign.py": source_sha}}}
    documents.update({tx / "forecasts/campaign.json": {"predictions": forecast},
                      tx / "observations/native.json": {"value": {"rows": {"prompt.native": observed}}},
                      tx / "frontiers/full_head.json": {"value": binding}})
    helper["source_run"] = lambda *_args: run
    helper["read"] = lambda path: documents[path]
    base_module["audit"] = lambda _root: base
    base_module["helpers"] = lambda: helper
    monkeypatch.setitem(module["audit"].__globals__, "runpy", SimpleNamespace(run_path=lambda _path: base_module))

    def invoke():
        # These four plain JSON files are also retained reference/hash evidence.
        for path, value in documents.items():
            if path.parent == data:
                path.write_text(json.dumps(value, sort_keys=True))
        return module["audit"](tmp_path)

    return SimpleNamespace(audit=invoke, binding=binding, observed=observed, full=full,
                           release=documents[data / "full-head-at-release.json"])


def test_consistent_synthetic_saved_assay_is_mechanically_valid(saved_assay):
    result = saved_assay.audit()
    assert result["mechanics_status"] == "valid"
    assert result["scientific_summary"]["full_vocab_argmax_matches"] == 1


def test_changed_at_release_readonly_copy_is_rejected(saved_assay):
    saved_assay.release["readonly_copy_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="at-release"):
        saved_assay.audit()


def test_changed_at_release_native_parameter_is_rejected(saved_assay):
    saved_assay.release["descriptor"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="at-release"):
        saved_assay.audit()


@pytest.mark.parametrize("field,value", [("bias", 0.), ("axis", 0), ("epsilon", 0.), ("mean_square_divisor", 3)])
def test_changed_rmsnorm_contract_cannot_silently_change_the_law(saved_assay, field, value):
    saved_assay.binding["final_rmsnorm_contract"][field] = value
    with pytest.raises(ValueError, match="RMSNorm contract"):
        saved_assay.audit()


def test_consistent_actual_top_one_counterexample_is_scientific_evidence(saved_assay):
    actual = np.array([[[-4., 9., 1.]]], dtype=np.float32)
    saved_assay.observed.update(full_vocabulary_logits=actual, selected_logits=actual[:, :, [0, 2]],
                                full_vocab_greedy_token=1, full_vocab_runner_up_token=2, full_vocab_top_two_margin=8.)
    result = saved_assay.audit()
    row = result["rows"]["prompt.native"]
    assert result["mechanics_status"] == "valid" and not row["mechanics_errors"]
    assert not row["full_vocab_argmax_matches"]
    assert row["actual_token_id"] == 1 and row["forecast_token_id"] == 0
    assert row["full_logit_comparison"]["max_abs"] == 7.
    assert result["scientific_summary"]["full_vocab_argmax_matches"] == 0
    assert result["terminal_status"] == "unassessed"


def test_inconsistent_stored_forecast_winner_is_an_instrument_error(saved_assay):
    saved_assay.full["doses"]["1"]["token_id"] = 2
    result = saved_assay.audit()
    assert result["mechanics_status"] == "instrument-errors-retained"
    assert "declared full forecast winner/runner-up/normalizer differs" in result["rows"]["prompt.native"]["mechanics_errors"]
