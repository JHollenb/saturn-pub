"""Native-eager parity for the functional L19-to-L22 Qwen2 transport."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb_lanes.bench import BenchSession
from mdb_lanes.read_weight_native import NativeQwenReadTransport, ReadWeightNativeError


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _weight_worker(monkeypatch):
    root = Path(__file__).parents[1] / "qualification"
    support = _load(root / "read_composition_20261107/worker.py", "composition_support")
    monkeypatch.setitem(sys.modules, "composition_support", support)
    return _load(root / "read_weight_transport_20261107/worker.py", "read_weight_worker")


def _forbid_original_forwards(model):
    handles = []
    entered = []

    def refuse(module, _args):
        entered.append(type(module).__name__)
        raise AssertionError(f"native module forward entered: {type(module).__name__}")

    for module in model.modules():
        handles.append(module.register_forward_pre_hook(refuse))
    return handles, entered


def _assert_matches_capture(tape, captured):
    for layer in range(19, 23):
        for part in ("input", "attention", "mlp", "exit"):
            assert torch.equal(tape[f"{part}{layer}"], captured[f"{part}{layer}"]), (part, layer)
    for layer in (20, 21, 22):
        for part in ("q", "k", "v"):
            assert torch.equal(tape[f"{part}{layer}"], captured[f"{part}{layer}"]), (part, layer)
        assert torch.equal(tape[f"current_k{layer}"], captured[f"k{layer}"][:, :, -1:])
        assert torch.equal(tape[f"current_v{layer}"], captured[f"current_v{layer}"])
    for part in ("scores22", "weights22"):
        assert torch.equal(tape[part], captured[part])


def test_native_eager_functional_transport_matches_zero_a_and_direct_ab(tmp_path, monkeypatch):
    worker = _weight_worker(monkeypatch)
    torch.manual_seed(729)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attn_implementation="eager",
    )).eval().to(torch.bfloat16).requires_grad_(False)

    class Tokenizer:
        def decode(self, ids):
            return str(ids[0])

    bench = BenchSession.from_resident(
        model,
        root=tmp_path / "session",
        attention_backend="native-eager",
        continuation_capacity=8,
    )
    try:
        parent = bench.prefill([1, 2, 3, 4])
        runner = worker.Runner(bench, Tokenizer())
        sites = {
            "A": {"layer": 19, "kv_head": 0, "source_row": 1},
            "B": {"layer": 22, "kv_head": 0, "source_row": 1},
        }
        _, zero = runner.arm("00", parent, sites, 7, [9, 11], capture=True, count=1)
        _, native_a = runner.arm("A", parent, sites, 7, [9, 11], edits=("A",), capture=True, count=1)
        _, native_ab = runner.arm("AB", parent, sites, 7, [9, 11],
                                  edits=("A", "B"), capture=True, count=1)

        source_zero = worker.local_source_attention(model, zero, sites["A"], dose=0.0)
        source_a = worker.local_source_attention(model, zero, sites["A"], dose=1.0)
        assert torch.equal(source_zero, zero["attention19"])
        assert torch.equal(source_a, native_a["attention19"])

        transport = NativeQwenReadTransport(
            model,
            incoming=zero["input19"],
            history={
                layer: (zero[f"k{layer}"][:, :, :-1], zero[f"v{layer}"][:, :, :-1])
                for layer in (20, 21, 22)
            },
            cos=zero["cos22"],
            sin=zero["sin22"],
        )
        assert transport.cost["logical_dense_linears_per_evaluate"] == 24
        assert transport.cost["native_module_forwards"] == 0

        handles, entered = _forbid_original_forwards(model)
        try:
            predicted_zero = transport.evaluate(source_zero)
            predicted_a = transport.evaluate(source_a)
            predicted_ab = transport.evaluate(
                source_a,
                value_edit={"kv_head": 0, "source_row": 1, "scale": 0.0},
            )
        finally:
            for handle in handles:
                handle.remove()
        assert entered == []

        _assert_matches_capture(predicted_zero, zero)
        _assert_matches_capture(predicted_a, native_a)
        _assert_matches_capture(predicted_ab, native_ab)
        assert torch.equal(predicted_a["input22"], predicted_ab["input22"])
        assert torch.equal(predicted_a["q22"], predicted_ab["q22"])
        assert torch.equal(predicted_a["k22"], predicted_ab["k22"])
        assert torch.equal(predicted_a["weights22"], predicted_ab["weights22"])
        assert not torch.equal(predicted_a["v22"], predicted_ab["v22"])

        invalid = source_a.clone()
        invalid[0, 0, 0] = float("nan")
        with pytest.raises(ReadWeightNativeError, match="finite"):
            transport.evaluate(invalid)
        with pytest.raises(ReadWeightNativeError, match="scale"):
            transport.evaluate(source_a, value_edit={"kv_head": 0, "source_row": 1, "scale": -0.1})

        attention = model.model.layers[22].self_attn
        original_weight = attention.o_proj.weight
        replacement_weight = original_weight.detach().clone()
        replacement_weight[0, 0] += 0.125
        attention.o_proj.weight = torch.nn.Parameter(
            replacement_weight, requires_grad=False
        )
        with pytest.raises(ReadWeightNativeError, match="weights changed"):
            transport.evaluate(source_a)
        attention.o_proj.weight = original_weight

        norm = model.model.layers[22].input_layernorm
        original_epsilon = norm.variance_epsilon
        norm.variance_epsilon = original_epsilon * 2
        with pytest.raises(ReadWeightNativeError, match="execution policy changed"):
            transport.evaluate(source_a)
        norm.variance_epsilon = original_epsilon

        original_scaling = attention.scaling
        attention.scaling = original_scaling * 1.25
        with pytest.raises(ReadWeightNativeError, match="execution policy changed"):
            transport.evaluate(source_a)
        attention.scaling = original_scaling

        original_backend = model.config._attn_implementation
        model.config._attn_implementation = "sdpa"
        with pytest.raises(ReadWeightNativeError, match="config changed"):
            transport.evaluate(source_a)
        model.config._attn_implementation = original_backend

        with torch.no_grad():
            attention.o_proj.weight[0, 0].add_(1)
        with pytest.raises(ReadWeightNativeError, match="weights changed"):
            transport.evaluate(source_a)
    finally:
        bench.close()
