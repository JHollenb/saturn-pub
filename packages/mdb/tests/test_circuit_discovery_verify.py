"""Offline custody checks for the native circuit-discovery report."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchSession
from mdb.bench_circuit_discovery_verify import _close_structure, verify_artifact
from mdb.bench_verify import verify


def _spec():
    return {
        "plan": [
            {"kind": "input", "token_ids": [7]},
            {"kind": "generate", "count": 1},
            {"kind": "input", "token_ids": [8]},
            {"kind": "generate", "count": 1},
        ],
        "receiver_offsets": [0, 2],
        "candidates_per_receiver": 1,
        "scales": [0.0, 0.5],
        "observable": {"kind": "token_margin", "token_id": 9, "against_token_id": 11},
    }


def test_wide_vocabulary_cuda_cpu_norm_reduction_calibration():
    # Frozen trained-Qwen CUDA norm versus CPU reconstruction of the same
    # byte-identical BF16 logits. Only this derived norm has the wider budget.
    _close_structure({"logit_l2": 43.136619567871094},
                     {"logit_l2": 43.13589096069336}, label="effect")
    with pytest.raises(ValueError, match="logit_l2 changed"):
        _close_structure({"logit_l2": 44.0}, {"logit_l2": 43.13589096069336}, label="effect")
    with pytest.raises(ValueError, match="logit_max_abs changed"):
        _close_structure({"logit_max_abs": 43.136619567871094},
                         {"logit_max_abs": 43.13589096069336}, label="effect")


class _ArtifactOverlay:
    """Read-only artifact view for testing a resealed but tampered object."""

    def __init__(self, base, overrides):
        self.base = base
        self.overrides = overrides

    def get(self, reference):
        if reference in self.overrides:
            return deepcopy(self.overrides[reference])
        return self.base.get(reference)


@pytest.fixture(scope="module")
def discovery_bundle(tmp_path_factory):
    torch.manual_seed(604)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attn_implementation="eager",
        )
    ).eval().to(torch.bfloat16)
    root = tmp_path_factory.mktemp("circuit-discovery") / "bench"
    bench = BenchSession.from_resident(
        model,
        root=root,
        continuation_capacity=16,
        attention_backend="native-eager",
    )
    bench.prefill([1, 2, 3])
    report = bench.discover_circuit(_spec())["report"]
    exported = bench.export(tmp_path_factory.mktemp("circuit-discovery-export") / "export")
    manifest = bench._manifest()
    try:
        yield {
            "bench": bench,
            "model": deepcopy(model),
            "report": deepcopy(report),
            "manifest": manifest,
            "exported": exported["root"],
        }
    finally:
        bench.close()


def _verify(bundle, report=None, *, artifacts=None):
    bench = bundle["bench"]
    return verify_artifact(
        report if report is not None else bundle["report"],
        artifacts or bench.artifacts,
        bundle["manifest"],
        bench.vm.artifacts,
    )


def _discovery_count(result):
    counts = result["counts"]
    return counts.get("circuit_discovery_artifacts", counts.get("circuit_discoveries", 0))


def test_frozen_report_verifies_with_signed_effects(discovery_bundle):
    assert _verify(discovery_bundle) is True

    # A newly serialized report still has to agree with the immutable logits,
    # even when the modified effect looks like a plausible signed observation.
    changed = deepcopy(discovery_bundle["report"])
    changed["tests"][0]["effect"]["steps"][0]["signed_margin_effect"] += 1.0
    with pytest.raises(ValueError, match="intervention effect"):
        _verify(discovery_bundle, changed)


def test_export_fresh_resume_and_offline_verify(discovery_bundle):
    resumed = BenchSession.resume(deepcopy(discovery_bundle["model"]), root=discovery_bundle["exported"])
    try:
        assert resumed.branch.cut
    finally:
        resumed.close()
    assert _discovery_count(verify(discovery_bundle["exported"])) == 1


def test_wrong_source_hash_and_false_cot_flag_are_refused(discovery_bundle):
    changed = deepcopy(discovery_bundle["report"])
    changed["instrument_identity"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source custody"):
        _verify(discovery_bundle, changed)

    changed = deepcopy(discovery_bundle["report"])
    changed["semantic_cot_qualified"] = True
    with pytest.raises(ValueError, match="semantic CoT"):
        _verify(discovery_bundle, changed)


def test_head_locality_payload_and_hash_tampering_are_refused(discovery_bundle):
    changed = deepcopy(discovery_bundle["report"])
    locality = changed["tests"][0]["head_locality"]
    locality["untouched_heads_after_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="head locality"):
        _verify(discovery_bundle, changed)

    changed = deepcopy(discovery_bundle["report"])
    locality = changed["tests"][0]["head_locality"]
    locality["after_ref"] = locality["before_ref"]
    with pytest.raises(ValueError, match="selected-head payload"):
        _verify(discovery_bundle, changed)


def test_query_history_and_graph_forgery_are_refused(discovery_bundle):
    changed = deepcopy(discovery_bundle["report"])
    changed["receivers"][0]["coverage"][0]["key_history_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="route binding"):
        _verify(discovery_bundle, changed)

    changed = deepcopy(discovery_bundle["report"])
    changed["graph"]["edges"].append({"source": "forged", "target": "receiver:0"})
    with pytest.raises(ValueError, match="derived from frozen observations"):
        _verify(discovery_bundle, changed)


def test_local_prediction_target_reference_and_formula_are_refused(discovery_bundle):
    changed = deepcopy(discovery_bundle["report"])
    changed["tests"][0]["local_prediction"]["actual_read"]["parent_cut"] = (
        changed["receivers"][0]["parent_cut"]
    )
    with pytest.raises(ValueError, match="route binding"):
        _verify(discovery_bundle, changed)

    changed = deepcopy(discovery_bundle["report"])
    changed["tests"][0]["local_prediction"]["prediction_ref"] = (
        changed["tests"][1]["local_prediction"]["prediction_ref"]
    )
    with pytest.raises(ValueError, match="artifact binding"):
        _verify(discovery_bundle, changed)

    changed = deepcopy(discovery_bundle["report"])
    prediction_ref = changed["tests"][0]["local_prediction"]["prediction_ref"]
    prediction = discovery_bundle["bench"].artifacts.get(prediction_ref)
    tampered_prediction = deepcopy(prediction)
    tampered_prediction["predicted_output_ref"] = tampered_prediction["baseline_output_ref"]
    overlay = _ArtifactOverlay(discovery_bundle["bench"].artifacts, {prediction_ref: tampered_prediction})
    with pytest.raises(ValueError, match="saved local prediction"):
        _verify(discovery_bundle, changed, artifacts=overlay)
