"""Synthetic CPU Qwen2 mechanics; no pretrained checkpoint or model claims."""
from __future__ import annotations

import copy
import runpy
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

from mdb.computation_protocol import SPEC_SCHEMA, compile_protocol, run_graph

MATH = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/qwen_family_math.py"))
FunctionalQwen, QwenMathError = MATH["FunctionalQwen"], MATH["QwenMathError"]
Q = torch.tensor([[2, 7, 11, 17, 5]])
P = torch.tensor([[2, 9, 13, 17, 5]])
CUT = 1


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(1907)
    config = Qwen2Config(vocab_size=47, hidden_size=32, intermediate_size=56,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=64, rope_parameters={"rope_type": "default", "rope_theta": 500000.})
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


def snapshot(cache):
    return tuple((layer.keys.detach().clone(), layer.values.detach().clone()) for layer in cache.layers)


def assert_cache(left, right):
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            for x, y in zip(a, b, strict=True):
                torch.testing.assert_close(x, y, atol=2e-7, rtol=5e-6)


@contextmanager
def forbid_module_calls(monkeypatch):
    with monkeypatch.context() as local:
        def forbidden(*_args, **_kwargs):
            pytest.fail("functional forecast called nn.Module")
        local.setattr(torch.nn.Module, "_call_impl", forbidden)
        yield


def native(model, ids, monkeypatch, *, family=None, factor=None, donor=None):
    """Independent native module/hook reference, including post-update KV policy."""
    captured, handles = {}, []
    original = modeling_qwen2.eager_attention_forward

    def attention(module, q, k, v, attention_mask, scaling, dropout=0., **kwargs):
        captured[(module.layer_idx, "attention")] = {
            "q": q.detach().clone(), "k": k.detach().clone(), "v": v.detach().clone(),
            "mask": attention_mask.detach().clone(), "scaling": scaling,
            "kv_groups": module.num_key_value_groups}
        result, weights = original(module, q, k, v, attention_mask, scaling, dropout=dropout, **kwargs)
        if family == "attention" and module.layer_idx == CUT:
            own = {"q": q, "k": k, "v": v}
            for key in (("q", "k") if factor == "route" else ("v",)):
                own[key] = donor[(CUT, "attention")][key]
            tail, _ = original(module, own["q"][:, :, -1:, :], own["k"], own["v"],
                               attention_mask[:, :, -1:, :], scaling, dropout=dropout, **kwargs)
            result = result.clone()
            result[:, -1:, :, :] = tail
        return result, weights

    def projection_hook(layer, name):
        def hook(_module, _args, value):
            captured[(layer, name)] = value.detach().clone()
            if family == "mlp" and layer == CUT and name == factor:
                result = value.clone()
                result[:, -1:, :] = donor[(layer, name)][:, -1:, :]
                return result
        return hook

    def block_hook(layer):
        def hook(_module, args, output):
            captured[(layer, "block")] = {"input": args[0].detach().clone(), "output": output.detach().clone()}
        return hook

    for index, block in enumerate(model.model.layers):
        handles.append(block.mlp.gate_proj.register_forward_hook(projection_hook(index, "gate")))
        handles.append(block.mlp.up_proj.register_forward_hook(projection_hook(index, "up")))
        handles.append(block.register_forward_hook(block_hook(index)))
    try:
        with monkeypatch.context() as local, torch.no_grad():
            local.setattr(modeling_qwen2, "eager_attention_forward", attention)
            result = model(ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    return result, captured


def native_greedy(model, initial, count):
    current, ids, rows = initial, [], []
    cache = copy.deepcopy(initial.past_key_values)
    with torch.no_grad():
        for index in range(count):
            logits = current.logits[:, -1, :].clone()
            token = logits.argmax(-1, keepdim=True)
            ids.append(token)
            rows.append(logits)
            if index + 1 < count:
                current = model(token, past_key_values=cache, use_cache=True)
    return torch.cat(ids, dim=1), rows, snapshot(cache)


def test_full_prefill_and_own_greedy_cache_without_module_calls(model, monkeypatch):
    expected, captured = native(model, Q, monkeypatch)
    expected_cache = snapshot(expected.past_key_values)
    tokens, logits, future_cache = native_greedy(model, expected, 4)
    decoder = FunctionalQwen(model.state_dict(), model.config)
    with forbid_module_calls(monkeypatch):
        actual = decoder.forward(Q, retain_layers=range(3))
        future = decoder.greedy(initial=actual, max_new_tokens=4)
    torch.testing.assert_close(actual["logits"], expected.logits, atol=2e-7, rtol=5e-6)
    assert_cache(actual["cache"], expected_cache)
    assert_cache(future["cache"], future_cache)
    assert torch.equal(future["tokens"], tokens)
    for row, target in zip(future["steps"], logits, strict=True):
        torch.testing.assert_close(row["logits"], target, atol=2e-7, rtol=5e-6)
    assert [row["cache_length"] for row in future["steps"]] == [5, 6, 7, 8]
    for layer in range(3):
        ports = actual["boundaries"][layer]
        for name in ("q", "k", "v"):
            torch.testing.assert_close(ports["attention"]["original_" + name], captured[(layer, "attention")][name],
                                       atol=2e-7, rtol=5e-6)
        for name in ("gate", "up"):
            torch.testing.assert_close(ports["mlp"]["original_" + name], captured[(layer, name)], atol=2e-7, rtol=5e-6)
        assert future["cache"][layer][0].data_ptr() != actual["cache"][layer][0].data_ptr()
    assert_cache(actual["cache"], expected_cache)


@pytest.mark.parametrize("factor", ["route", "value"])
def test_attention_last_query_intervention_and_receiver_kv(model, monkeypatch, factor):
    _, donor = native(model, P, monkeypatch)
    baseline, _ = native(model, Q, monkeypatch)
    target, _ = native(model, Q, monkeypatch, family="attention", factor=factor, donor=donor)
    expected_tokens, expected_logits, expected_cache = native_greedy(model, target, 3)
    decoder = FunctionalQwen(model.state_dict(), model.config)
    with forbid_module_calls(monkeypatch):
        source = decoder.forward(P, retain_layers=(CUT,))["boundaries"][CUT]["attention"]

        def intervention(layer, stage, _ports):
            if layer == CUT and stage == "attention":
                names = ("q", "k") if factor == "route" else ("v",)
                return {**{name: source["original_" + name] for name in names}, "tail_only": True}

        actual = decoder.forward(Q, intervene=intervention, retain_layers=(CUT,))
        future = decoder.greedy(initial=actual, max_new_tokens=3)
    torch.testing.assert_close(actual["logits"], target.logits, atol=2e-7, rtol=5e-6)
    torch.testing.assert_close(actual["logits"][:, :-1, :], baseline.logits[:, :-1, :], atol=0, rtol=0)
    assert_cache((actual["cache"][CUT],), (snapshot(baseline.past_key_values)[CUT],))
    assert_cache(future["cache"], expected_cache)
    assert torch.equal(future["tokens"], expected_tokens)
    for row, expected in zip(future["steps"], expected_logits, strict=True):
        torch.testing.assert_close(row["logits"], expected, atol=2e-7, rtol=5e-6)
    attn = actual["boundaries"][CUT]["attention"]
    assert attn["primitive_return"].shape[-2] == 1 and attn["head"].shape[-2] == Q.shape[1]
    assert torch.equal(attn["head"][:, :, :-1, :], attn["natural_head"][:, :, :-1, :])


@pytest.mark.parametrize("factor", ["gate", "up"])
def test_swiglu_last_row_intervention_recomputes_downstream(model, monkeypatch, factor):
    _, donor = native(model, P, monkeypatch)
    target, _ = native(model, Q, monkeypatch, family="mlp", factor=factor, donor=donor)
    expected_tokens, expected_logits, expected_cache = native_greedy(model, target, 3)
    decoder = FunctionalQwen(model.state_dict(), model.config)
    with forbid_module_calls(monkeypatch):
        source = decoder.forward(P, retain_layers=(CUT,))["boundaries"][CUT]["mlp"]

        def intervention(layer, stage, ports):
            if layer == CUT and stage == "mlp":
                value = ports[factor].clone()
                value[:, -1:, :] = source["original_" + factor][:, -1:, :]
                return {factor: value}

        actual = decoder.forward(Q, intervene=intervention, retain_layers=(CUT,))
        future = decoder.greedy(initial=actual, max_new_tokens=3)
    torch.testing.assert_close(actual["logits"], target.logits, atol=2e-7, rtol=5e-6)
    assert_cache(actual["cache"], snapshot(target.past_key_values))
    assert_cache(future["cache"], expected_cache)
    assert torch.equal(future["tokens"], expected_tokens)
    for row, expected in zip(future["steps"], expected_logits, strict=True):
        torch.testing.assert_close(row["logits"], expected, atol=2e-7, rtol=5e-6)
    ports = actual["boundaries"][CUT]["mlp"]
    assert torch.equal(ports[factor][:, :-1, :], ports["original_" + factor][:, :-1, :])


def test_resume_genuine_layer_input_and_private_partially_filled_cache(model, monkeypatch):
    expected, native_ports = native(model, Q, monkeypatch)
    native_cache = snapshot(expected.past_key_values)
    supplied = tuple(native_cache[index] if index < CUT else None for index in range(3))
    before = MATH["clone_cache"](supplied)
    decoder = FunctionalQwen(model.state_dict(), model.config)
    with forbid_module_calls(monkeypatch):
        prefix = decoder.forward(Q, stop_layer=CUT)
        resumed = decoder.forward(hidden_states=native_ports[(CUT, "block")]["input"], start_layer=CUT, cache=supplied)
        own_resume = decoder.forward(hidden_states=prefix["hidden_states"], start_layer=CUT, cache=prefix["cache"])
    assert prefix["logits"] is None and prefix["cache"][CUT] is None
    torch.testing.assert_close(resumed["logits"], expected.logits, atol=2e-7, rtol=5e-6)
    torch.testing.assert_close(own_resume["logits"], expected.logits, atol=2e-7, rtol=5e-6)
    assert_cache(resumed["cache"], native_cache)
    assert_cache(supplied, before)
    assert resumed["cache"][0][0].data_ptr() != supplied[0][0].data_ptr()


def graph_plan(kind):
    if kind == "attention":
        names, factors = ("q", "k", "v"), [{"id": "route", "ports": ["q", "k"]}, {"id": "value", "ports": ["v"]}]
        nodes = [{"id": "result", "op": "sdpa", "inputs": ["q", "k", "v"]}]
    else:
        names, factors = ("G", "U"), [{"id": "gate", "ports": ["G"]}, {"id": "value", "ports": ["U"]}]
        nodes = [{"id": "activated", "op": "silu", "inputs": ["G"]},
                 {"id": "result", "op": "multiply", "inputs": ["activated", "U"]}]
    return compile_protocol({"schema": SPEC_SCHEMA, "family_id": "tiny-qwen-" + kind,
        "model_identity": "synthetic-only", "endpoint_adapter": "original-fp32-qwen-equations",
        "source_policy": "synthetic paired operands", "ports": {name: {"shape": ["B", "T", "D"],
            "dtype": "torch.float32", "stage": "consumed"} for name in names}, "factors": factors,
        "nodes": nodes, "output": "result", "max_contexts": 4, "receiver": {"residual": "Q"},
        "clock": {"layer": CUT}, "continuation": {"consumer": "own greedy", "transitions": 3},
        "metric": {"observable": "full logits", "units": "logit", "reference": "native",
                   "scope": "tiny mechanics", "direction": "minimize"}})


def test_public_generated_graph_callbacks_execute_both_primitive_families(model, monkeypatch):
    decoder = FunctionalQwen(model.state_dict(), model.config)
    calls = []
    plans = {stage: graph_plan(stage) for stage in ("attention", "mlp")}

    def intervention(layer, stage, ports):
        if layer == CUT:
            return {"q": ports["q"], "tail_only": True} if stage == "attention" else {"gate": ports["gate"]}

    def kernel(layer, stage, ports):
        calls.append((layer, stage))
        if stage == "attention":
            source = {key: ports[key] for key in ("q", "k", "v")}
            kernels = {"sdpa": lambda q, k, v: MATH["eager_attention"](
                q, k, v, ports["mask"], ports["scaling"], ports["kv_groups"])}
            bindings = {"route": "Q", "value": "Q"}
        else:
            source = {"G": ports["gate"], "U": ports["up"]}
            kernels, bindings = {"silu": torch.nn.functional.silu, "multiply": torch.mul}, {"gate": "Q", "value": "Q"}
        return run_graph(plans[stage], sources={"P": source, "Q": source}, bindings=bindings, kernels=kernels)

    with forbid_module_calls(monkeypatch):
        baseline = decoder.forward(Q)
        generated = decoder.forward(Q, intervene=intervention, graph_kernel=kernel)
    assert calls == [(CUT, "attention"), (CUT, "mlp")]
    torch.testing.assert_close(generated["logits"], baseline["logits"], atol=2e-7, rtol=5e-6)


def test_hostile_profiles_cache_and_intervention_geometry_are_rejected(model):
    decoder = FunctionalQwen(model.state_dict(), model.config)
    config = copy.deepcopy(model.config)
    config.rope_parameters["rope_type"] = "linear"
    with pytest.raises(QwenMathError, match="default RoPE"):
        FunctionalQwen(model.state_dict(), config)
    with pytest.raises(QwenMathError, match="device profile"):
        FunctionalQwen({key: value.double() for key, value in model.state_dict().items()}, model.config)
    with pytest.raises(QwenMathError, match="cache layer count"):
        decoder.forward(Q, cache=())
    with pytest.raises(QwenMathError, match="intervention shape"):
        decoder.forward(Q, intervene=lambda _layer, stage, ports: {"q": ports["q"][:, :, -1:, :]} if stage == "attention" else None)
    with pytest.raises(QwenMathError, match="unknown intervention port"):
        decoder.forward(Q, intervene=lambda *_args: {"stored_KV": torch.zeros(1)})
    with pytest.raises(QwenMathError, match="all-attended"):
        decoder.forward(Q, attention_mask=torch.tensor([[0, 1, 1, 1, 1]]))
