"""Tiny-model mechanics for the prospective two-read composition assay."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb_lanes.bench import BenchSession
from mdb_lanes.bench_verify import verify
from mdb_lanes.read_program import FrozenReadProgram


def load_qualification(name="worker"):
    path = Path(__file__).parents[1] / f"qualification/read_composition_20261107/{name}.py"
    spec = importlib.util.spec_from_file_location(f"read_composition_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_composed_write_is_sealed_consumed_and_reinstalled(tmp_path):
    worker = load_qualification()
    torch.manual_seed(711)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=24,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="eager",
    )).eval().to(torch.bfloat16).requires_grad_(False)

    class Tokenizer:
        def decode(self, ids):
            return str(ids[0])

    bench = BenchSession.from_resident(model, root=tmp_path / "session", attention_backend="native-eager",
                                       continuation_capacity=16)
    try:
        parent = bench.prefill([1, 2, 3, 4])
        runner = worker.Runner(bench, Tokenizer())
        sites = {"A": {"layer": 19, "kv_head": 0, "source_row": 1},
                 "B": {"layer": 22, "kv_head": 0, "source_row": 3},
                 "wrong": {"layer": 19, "kv_head": 0, "source_row": 0}}
        result = worker.bundle(runner, parent, sites, 7, [9, 11])
        assert result["A_vs_AB_query_keys_current_value_exact"]
        assert result["old_keys22_unchanged"]
        assert result["prediction_record"]["physical_steps_at_seal"] == 18
        assert runner.events[-1]["kind"] == "native-AB-started"
        assert runner.events[-1]["physical_steps"] == 24
        assert runner.token_steps == 27
        assert bench.branch.cut == parent
        for name in ("conditional", "stale"):
            assert result["local_errors"][name]["injected_mlp_exact"]
            assert result["local_errors"][name]["injected_exit_exact"]
            assert result["arms"][name]["trajectory"][1]["authority"] == "native-greedy"
        # Identical parent/Act payloads are CAS-deduplicated across arms.
        assert verify(bench.root)["counts"]["memory_writes"] == 6
    finally:
        bench.close()


def test_role_program_recurs_on_each_owned_history(tmp_path):
    worker = load_qualification()
    torch.manual_seed(712)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=24,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attn_implementation="eager",
    )).eval().to(torch.bfloat16).requires_grad_(False)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return {" is": [3], " x": [9], " y": [11], "prompt": [1, 2, 3, 4],
                    "first": [5, 3, 7], "second": [6, 3, 8]}[text]

        def get_vocab(self):
            return {str(index): index for index in range(32)}

        def decode(self, ids):
            return str(ids[0])

    case = {"id": "tiny-owned-history", "prompt": "prompt", "followups": ["first", "second"],
            "objects": ["x", "y"], "owners": ["A", "B"], "previous_owner": "B"}
    budget = {"used": 0, "limit": 100}
    result = worker.run_case(model, Tokenizer(), SimpleNamespace(model_id="tiny-qwen", revision="seed712"),
                             None, case, tmp_path, FrozenReadProgram.standard(), budget)
    assert set(result["recurrent_policies"]) == {"00", "A", "B", "AB"}
    assert result["second_AB_world"]["parent_cut"] != result["first"]["parent_cut"]
    first_source = result["first"]["sites"]["A"]["source_row"]
    for policy in result["recurrent_policies"].values():
        assert policy["resolved_sites"]["A"]["source_row"] > first_source
        assert policy["resolved_sites"]["A"]["source_row"] == policy["preceding_answer_span"][0]
        assert policy["outcome"]["trajectory"][1]["authority"] == "native-greedy"
    assert budget["used"] == result["physical_token_steps"]
    assert verify(tmp_path / case["id"])["counts"]["memory_writes"] > 6
    auditor = load_qualification("verify")
    audit = auditor.verify_case(result, tmp_path / case["id"], program=FrozenReadProgram.standard())
    assert audit["mechanics_status"] == audit["instrument_status"] == "valid"
    assert audit["terminal_status"] == "not-assessed"
    changed = copy.deepcopy(result)
    changed["first"]["arms"]["conditional"]["trajectory"][1]["consumed_token_id"] += 1
    with pytest.raises(ValueError, match="token history changed"):
        auditor.verify_case(changed, tmp_path / case["id"], program=FrozenReadProgram.standard())
    changed = copy.deepcopy(result)
    changed["first"]["sites"]["A"]["source_row"] += 1
    with pytest.raises(ValueError, match="source|row"):
        auditor.verify_case(changed, tmp_path / case["id"], program=FrozenReadProgram.standard())
