"""Tiny installed native Qwen3 layer, conditional observed-frontier mechanics."""
from __future__ import annotations

import hashlib
import inspect
import json
import runpy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

S = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_row_sign_shortcut.py"))


def fixture():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng():
        torch.manual_seed(315)
        config = Qwen3Config(vocab_size=23, hidden_size=6, intermediate_size=12, num_hidden_layers=5,
            num_attention_heads=4, num_key_value_heads=2, head_dim=2, max_position_embeddings=16,
            rope_theta=1_000_000, attention_bias=False, attention_dropout=0., use_cache=False)
        config._attn_implementation = "sdpa"
        layer = Qwen3DecoderLayer(config, layer_idx=2).to(torch.bfloat16).eval().requires_grad_(False)
        hidden = torch.randn(1, 6, 6, dtype=torch.bfloat16)
        rotary = Qwen3RotaryEmbedding(config=config, device=torch.device("cpu"))
        positions = torch.arange(6, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            cos, sin = rotary(hidden, positions)
    context = dict(attention_mask=None, position_ids=positions, position_embeddings=(cos, sin), use_cache=False)
    permitted = torch.ones(6, 6, dtype=torch.bool).tril()
    return layer, hidden, context, permitted


def native(layer, hidden, context):
    ports, calls, handles = {}, [], []
    original = F.scaled_dot_product_attention

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        ports["head"] = out.detach().clone()
        mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        calls.append(dict(mask_none=mask is None, is_causal=kwargs.get("is_causal", False),
            enable_gqa=kwargs.get("enable_gqa", False), QKV=[list(value.shape) for value in args[:3]]))
        return out

    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, args: ports.update(R=args[0].detach().clone())))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args: ports.update(product=args[0].detach().clone())))
    for name, module in (("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj), ("M", layer.mlp.down_proj)):
        handles.append(module.register_forward_hook(lambda _m, _a, out, name=name: ports.update({name: out.detach().clone()})))
    F.scaled_dot_product_attention = capture
    try:
        with torch.inference_mode():
            ports["Hnext"] = layer(hidden, **context).detach().clone()
    finally:
        F.scaled_dot_product_attention = original
        for handle in reversed(handles):
            handle.remove()
    return ports, calls


def profile(ports, weight, signs, permitted, *, head=None, axis=None):
    def digest(tensor):
        return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast("B")).hexdigest()

    return S["make_profile"]({name: ports[name] for name in ("R", "gate", "up")}, weight, signs, permitted,
        source_identity=dict(native_qwen=hashlib.sha256(Path(inspect.getfile(Qwen3DecoderLayer)).read_bytes()).hexdigest(),
            native_sdpa_interface=hashlib.sha256(Path(inspect.getfile(sdpa_attention_forward)).read_bytes()).hexdigest()),
        frontier_kind="observed-positive-ports", attention_profile=dict(backend="sdpa", gqa_policy="enable_gqa", query_rows="all-attended",
            mask_identity=digest(permitted), positions_identity="1" * 64, rope_identity="2" * 64),
        positive_head=head, head_query_axis=axis)


def test_three_shortcuts_before_three_native_targets_on_one_actual_component(tmp_path, monkeypatch):
    layer, hidden, context, permitted = fixture()
    plus, positive_calls = native(layer, hidden, context)
    assert positive_calls == [dict(mask_none=True, is_causal=True, enable_gqa=True, QKV=[[1, 4, 6, 2], [1, 2, 6, 2], [1, 2, 6, 2]])]
    signs = dict(baseline=torch.ones(6, dtype=torch.bfloat16), global_negative=-torch.ones(6, dtype=torch.bfloat16),
                 prefix_negative=torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.bfloat16))
    predictions = {}
    attention, linear = F.scaled_dot_product_attention, F.linear
    counted = dict(linear=0, attention=0)

    def deny_attention(*_args, **_kwargs):
        counted["attention"] += 1
        raise AssertionError("shortcut attention entry")

    def count_linear(*args, **kwargs):
        counted["linear"] += 1
        return linear(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(F, "scaled_dot_product_attention", deny_attention)
        scoped.setattr(F, "linear", count_linear)
        for case, s in signs.items():
            bound = profile(plus, layer.mlp.down_proj.weight, s, permitted, head=plus["head"], axis=2)
            predictions[case] = S["predict_row_sign"]({name: plus[name] for name in ("R", "gate", "up")}, layer.mlp.down_proj.weight,
                s, permitted, profile=bound, positive_head=plus["head"], head_query_axis=2)
    assert F.scaled_dot_product_attention is attention and counted == dict(linear=3, attention=0)
    assert predictions["prefix_negative"]["support_closed"].tolist() == [True, True, True, False, False, False]
    observations = {}
    for case, s in signs.items():
        actual, receipts = native(layer, s.view(1, -1, 1) * hidden, context)
        assert len(receipts) == 1 and receipts[0]["mask_none"] and receipts[0]["is_causal"] and receipts[0]["enable_gqa"]
        predicted = predictions[case]
        closed = predicted["support_closed"]
        fields = {}
        for name in ("product", "M", "Hnext"):
            delta = predicted["ports"][name].double() - actual[name].double()
            fields[name] = dict(rms=float(delta.square().mean().sqrt()), closed_rms=float(delta[:, closed].square().mean().sqrt()),
                mixed_rms=None if bool(closed.all()) else float(delta[:, ~closed].square().mean().sqrt()),
                predicted=S["base"]()["tensor_descriptor"](predicted["ports"][name]), actual=S["base"]()["tensor_descriptor"](actual[name]))
            assert torch.equal(predicted["ports"][name][:, closed], actual[name][:, closed])
        assert torch.equal(predicted["ports"]["headhat"][:, :, closed], actual["head"][:, :, closed])
        observations[case] = dict(fields=fields, counts=predicted["counts"], support=predicted["support"], receipts=receipts)
    assert observations["prefix_negative"]["fields"]["Hnext"]["mixed_rms"] > 0
    (tmp_path / "actual-component-observations.json").write_text(json.dumps(dict(
        scope="Tiny installed observed-positive frontier only; one source-positive and three target component calls, no full encoder/LM or pretrained weights",
        observations=observations, shortcut_primitive_calls=counted), sort_keys=True, indent=2))


def test_mask_support_uses_full_allowed_keys_and_preserves_empty_queries_as_diagnostic():
    signs = torch.tensor([-1, -1, 1, 1], dtype=torch.bfloat16)
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    assert S["support_closed"](signs, causal).tolist() == [True, True, False, False]
    assert S["support_closed"](signs, torch.eye(4, dtype=torch.bool)).tolist() == [True] * 4
    causal[0] = False
    assert S["support_closed"](signs, causal).tolist() == [False, True, False, False]


def test_signed_zero_and_explicit_head_query_axes():
    ports = dict(R=torch.tensor([[[0., -0.], [-0., 0.]]], dtype=torch.bfloat16),
        gate=torch.tensor([[[1., -2., 0.], [2., 3., -0.]]], dtype=torch.bfloat16),
        up=torch.ones(1, 2, 3, dtype=torch.bfloat16))
    weight = torch.eye(2, 3, dtype=torch.bfloat16)
    signs = torch.tensor([-1, 1], dtype=torch.bfloat16)
    allowed = torch.ones(2, 2, dtype=torch.bool).tril()
    head = torch.arange(12, dtype=torch.bfloat16).view(1, 2, 2, 3)
    bound = profile(ports, weight, signs, allowed, head=head, axis=1)
    result = S["predict_row_sign"](ports, weight, signs, allowed, profile=bound, positive_head=head, head_query_axis=1)
    assert torch.signbit(result["ports"]["Rhat"]).tolist() == [[[True, False], [True, False]]]
    assert torch.equal(result["ports"]["headhat"][:, 0], -head[:, 0])
    with pytest.raises(ValueError, match="query axis changed"):
        S["predict_row_sign"](ports, weight, signs, allowed, profile=bound, positive_head=head, head_query_axis=2)


def test_input_integrity_mask_tamper_and_invalid_signs():
    layer, hidden, context, permitted = fixture()
    ports, _ = native(layer, hidden, context)
    signs = torch.ones(6, dtype=torch.bfloat16)
    bound = profile(ports, layer.mlp.down_proj.weight, signs, permitted)
    permitted[0, 1] = True
    with pytest.raises(ValueError, match="permitted-mask identity"):
        S["predict_row_sign"]({name: ports[name] for name in ("R", "gate", "up")}, layer.mlp.down_proj.weight, signs, permitted, profile=bound)
    signs[0] = 0
    with pytest.raises(ValueError, match="exact BF16"):
        S["support_closed"](signs, permitted)
    with pytest.raises(ValueError, match="bool permitted"):
        S["support_closed"](torch.ones(6, dtype=torch.bfloat16), permitted.float())


def test_module_guard_refuses_hostile_primitive_and_restores_global_hooks(monkeypatch):
    layer, hidden, context, permitted = fixture()
    ports, _ = native(layer, hidden, context)
    positive = {name: ports[name] for name in ("R", "gate", "up")}
    signs = torch.ones(6, dtype=torch.bfloat16)
    bound = profile(ports, layer.mlp.down_proj.weight, signs, permitted)
    before = len(torch.nn.modules.module._global_forward_pre_hooks)
    with monkeypatch.context() as scoped:
        scoped.setattr(F, "silu", lambda _value: layer(hidden, **context))
        with pytest.raises(RuntimeError, match="native module forward forbidden"):
            S["predict_row_sign"](positive, layer.mlp.down_proj.weight, signs, permitted, profile=bound)
    assert len(torch.nn.modules.module._global_forward_pre_hooks) == before
    assert S["predict_row_sign"](positive, layer.mlp.down_proj.weight, signs, permitted, profile=bound)["counts"]["module_forwards"] == 0
