"""Tiny native proof of the sealed second-order read-weight boundary."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench_verify import verify
from mdb.read_program import FrozenReadProgram


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_second_order_predictions_seal_before_edited_receivers_and_recur_on_own_history(
    tmp_path, monkeypatch
):
    root = Path(__file__).parents[1] / "qualification"
    support = load(root / "read_composition_20261107/worker.py", "composition_support")
    monkeypatch.setitem(sys.modules, "composition_support", support)
    worker = load(root / "read_weight_second_order_20261107/worker.py", "weight_second_order_worker")

    torch.manual_seed(723)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=24,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attn_implementation="eager",
        )
    ).eval().to(torch.bfloat16).requires_grad_(False)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {
                " is": [3],
                " x": [9],
                " y": [11],
                "prompt": [1, 2, 3, 4],
                "first": [5, 3, 7],
                "second": [6, 3, 8],
            }[text]

        def decode(self, ids):
            return str(ids[0])

        def get_vocab(self):
            return {str(index): index for index in range(32)}

    case = {
        "id": "tiny-second-order-history",
        "prompt": "prompt",
        "followups": ["first", "second"],
        "objects": ["x", "y"],
        "owners": ["A", "B"],
        "previous_owner": "B",
    }
    budget = {"used": 0, "limit": 160}
    result = worker.run_case(
        model,
        Tokenizer(),
        SimpleNamespace(model_id="tiny-qwen", revision="seed723"),
        None,
        case,
        tmp_path,
        budget,
    )
    assert budget["used"] == result["physical_token_steps"]
    first = result["first"]
    second = result["second_candidate_world"]
    assert result["second_world_source_cut"] == first["arms"]["second-order"]["end_cut"]
    assert second["sites"]["A"]["source_row"] == result["second_answer_span"][0]
    assert second["sites"]["A"]["source_row"] > first["sites"]["A"]["source_row"]

    for bundle in (first, second):
        assert bundle["native_module_entries_during_forecast"] == 0
        assert bundle["weight_forecast"]["source_replay"]["exact"]
        assert bundle["weight_forecast"]["taylor_program"]["schema"] == "mdb-smooth-qwen-taylor-program-v1"
        assert bundle["weight_forecast"]["taylor_program"]["cost"]["native_forward_calls"] == 0
        forecast = bundle["weight_forecast"]
        assert forecast["primary_float32_equivalent_bytes"] == 4 * forecast["serving_tensor_elements"]
        assert forecast["serving_tensor_elements"] > 0
        assert [float(row["dose"]) for row in bundle["dose_rows"]] == list(worker.DOSES)
        for method in worker.CONSUMERS:
            assert bundle["local_errors"][method]["injected_mlp_exact"]
            assert bundle["local_errors"][method]["injected_exit_exact"]
            assert bundle["arms"][method]["trajectory"][1]["authority"] == "native-greedy"
        assert bundle["old_keys22_unchanged"]
        assert bundle["B_source_value_unchanged"]

    events = result["prediction_events"]
    seals = [event for event in events if event["kind"] == "weights-sealed"]
    edited = [event for event in events if event["kind"] == "edited-receiver-started"]
    assert len(seals) == 2 and len(edited) == 8
    for index, seal in enumerate(seals):
        assert all(
            seal["physical_steps"] <= event["physical_steps"]
            for event in edited[index * 4:index * 4 + 4]
        )
        assert seal["ref"] == edited[index * 4]["weight_ref"]

    assert verify(tmp_path / case["id"])["mechanics_status"] == "valid"
    auditor = load(root / "read_weight_second_order_20261107/verify.py", "read_weight_second_order_auditor")
    checked = auditor._verify_case(result, tmp_path / case["id"], FrozenReadProgram.standard())
    assert checked["mechanics_status"] == checked["instrument_status"] == "valid"
    assert any("nested second-derivative compiler" in item for item in checked["limitations"])

    changed = copy.deepcopy(result)
    changed["prediction_events"][0]["physical_steps"] += 1000
    with pytest.raises(ValueError):
        auditor._verify_case(changed, tmp_path / case["id"], FrozenReadProgram.standard())

    changed = copy.deepcopy(result)
    changed["first"]["dose_rows"][4]["methods"]["second-order"]["log_change_rmse"] += 1
    with pytest.raises(ValueError):
        auditor._verify_case(changed, tmp_path / case["id"], FrozenReadProgram.standard())

    changed = copy.deepcopy(result)
    changed["first"]["arms"]["second-order"]["trajectory"][1]["consumed_token_id"] += 1
    with pytest.raises(ValueError, match="history|token"):
        auditor._verify_case(changed, tmp_path / case["id"], FrozenReadProgram.standard())
