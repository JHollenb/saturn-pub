"""Tiny native qualification of the complete predicted-state/KV bridge."""
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb_lanes.bench_verify import verify


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receiver_free_span_installs_own_hidden_and_cache_before_future(tmp_path, monkeypatch):
    root = Path(__file__).parents[1] / "qualification"
    composition = load(root / "read_composition_20261107/worker.py", "composition_support")
    monkeypatch.setitem(sys.modules, "composition_support", composition)
    old = load(root / "read_weight_second_order_20261107/worker.py", "second_order_support")
    monkeypatch.setitem(sys.modules, "second_order_support", old)
    worker = load(root / "read_weight_closure_20261110/worker.py", "read_weight_closure_worker")
    torch.manual_seed(911)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=24,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="eager",
    )).eval().to(torch.bfloat16).requires_grad_(False)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {" is": [3], " x": [9], " y": [11], "prompt": [1,2,3,4],
                    "first": [5,3,7], "second": [6,3,8]}[text]

        def decode(self, ids):
            return str(ids[0])

        def get_vocab(self):
            return {str(i): i for i in range(32)}

    case = {"id":"tiny-closure", "prompt":"prompt", "followups":["first","second"],
            "objects":["x","y"], "owners":["A","B"], "previous_owner":"B", "split":"development"}
    budget = {"used":0, "limit":120}
    result = worker.run_case(model, Tokenizer(), SimpleNamespace(model_id="tiny-qwen",revision="seed911"),
                             None, case, tmp_path, budget)
    assert budget["used"] == result["physical_token_steps"]
    assert result["second_world_source_cut"] == result["first"]["arms"]["midpoint"]["end_cut"]
    seals = [e for e in result["prediction_events"] if e["kind"] == "all-predictions-sealed"]
    starts = [e for e in result["prediction_events"] if e["kind"] == "edited-receiver-started"]
    assert len(seals) == 2 and len(starts) == 8
    for i, seal in enumerate(seals):
        assert all(e["physical_steps"] >= seal["physical_steps"] for e in starts[i*4:i*4+4])
        assert all(e["forecast_ref"] == seal["ref"] for e in starts[i*4:i*4+4])
    for bundle in (result["first"],result["second_candidate_world"]):
        assert not bundle["native_A_assisted"]
        assert bundle["forecast"]["native_module_entries"] == 0
        assert bundle["functional_zero_closure_exact"]
        assert bundle["functional_AB_native_closure_exact"]
        assert bundle["functional_AB_full_logit_signature_exact"]
        for method in worker.CONSUMERS:
            arm = bundle["arms"][method]
            assert [r["layer"] for r in arm["span_replacements"]] == [19,20,21,22]
            assert arm["replaced_layer_transitions"] == 4
            assert arm["native_layer_transitions"] == 3*24-4
            assert bundle["local_errors"][method]["injected_exit_exact"]
            assert bundle["local_errors"][method]["injected_current_k_exact"]
            assert arm["trajectory"][1]["authority"] == "native-greedy"
        for dose in bundle["dose_rows"]:
            assert dose["functional_logp_rmse"] == 0.
            assert dose["local_source_error"]["exact"]
    assert verify(tmp_path/case["id"])["mechanics_status"] == "valid"
    audit = load(root / "read_weight_closure_20261110/verify.py", "closure_saved_audit")
    monkeypatch.setattr(audit, "NUM_LAYERS", 24)
    monkeypatch.setattr(audit, "CASE_SPLITS", {"tiny-closure": "development"})
    assert audit._verify_case(result, tmp_path/case["id"])["mechanics_status"] == "valid"
    changed = deepcopy(result)
    changed["second_world_source_cut"] = result["first"]["arms"]["AB"]["end_cut"]
    with pytest.raises(ValueError, match="midpoint branch"):
        audit._verify_case(changed, tmp_path/case["id"])
