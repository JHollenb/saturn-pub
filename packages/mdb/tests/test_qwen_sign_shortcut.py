"""Actual installed Qwen3 positive-frontier/negative-block tiny mechanics."""
from __future__ import annotations

import hashlib
import inspect
import json
import runpy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

S = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_sign_shortcut.py"))


def context_profile(mask, kwargs, layer):
    def digest(value):
        values = value if isinstance(value, (list, tuple)) else (value,)
        h = hashlib.sha256()
        for child in values:
            raw = child.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            h.update(memoryview(raw).cast("B"))
        return h.hexdigest()

    padding = bool((mask == 0).any())
    return dict(backend="sdpa", gqa_policy="repeat_kv" if padding else "enable_gqa",
        query_rows="partially-padded" if padding else "all-attended", mask_identity=digest(mask),
        positions_identity=digest(kwargs["position_ids"]), rope_identity=digest(kwargs["position_embeddings"]))


def fixture(padded):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng():
        torch.manual_seed(315)
        config = Qwen3Config(vocab_size=23, hidden_size=6, intermediate_size=12, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=2, max_position_embeddings=16,
            rope_theta=1_000_000, attention_bias=False, attention_dropout=0., pad_token_id=0, use_cache=False)
        model = Qwen3ForCausalLM(config).eval().to(torch.bfloat16).requires_grad_(False)
        model.config._attn_implementation = "sdpa"
        model.model.rotary_emb = type(model.model.rotary_emb)(config=model.config, device=torch.device("cpu"))
    ids = torch.tensor([[1, 3, 5, 7, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]]) if padded else torch.ones_like(ids)
    return model, ids, mask


def positive_frontier(model, ids, mask):
    layer = model.model.layers[0]
    ports, context, handles = {}, {}, []

    def entry(_module, args, kwargs):
        ports["h"] = args[0].detach().clone()
        context.update(kwargs)

    handles.append(layer.register_forward_pre_hook(entry, with_kwargs=True))
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
        lambda _m, args: ports.update(R=args[0].detach().clone())))
    for name, module in (("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj), ("M", layer.mlp.down_proj)):
        handles.append(module.register_forward_hook(lambda _m, _a, out, name=name: ports.update({name: out.detach().clone()})))
    handles.append(layer.register_forward_hook(lambda _m, _a, out: ports.update(Hnext=out.detach().clone())))
    try:
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for handle in reversed(handles):
            handle.remove()
    assert (context["attention_mask"] is not None) == bool((mask == 0).any())
    return ports, context


def native_negative(layer, hidden, context):
    ports, handles = {}, []
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
        lambda _m, args: ports.update(R=args[0].detach().clone())))
    for name, module in (("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj), ("M", layer.mlp.down_proj)):
        handles.append(module.register_forward_hook(lambda _m, _a, out, name=name: ports.update({name: out.detach().clone()})))
    try:
        with torch.inference_mode():
            ports["Hnext"] = layer(hidden, **context).detach().clone()
    finally:
        for handle in reversed(handles):
            handle.remove()
    return ports


def profile(positive, layer, mask, context):
    source_path = Path(inspect.getfile(type(layer)))
    return S["make_profile"]({name: positive[name] for name in S["INPUTS"]}, layer.mlp.down_proj.weight,
        frontier_kind="observed-positive-ports", source_identity={
            "installed_model_source": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "synthetic_down_weight": S["tensor_descriptor"](layer.mlp.down_proj.weight)["bf16_le_sha256"]},
        attention_profile=context_profile(mask, context, layer))


@pytest.mark.parametrize("padded", [True, False])
def test_actual_native_block_conditional_parity_no_attention_or_modules_in_shortcut(tmp_path, monkeypatch, padded):
    model, ids, mask = fixture(padded)
    layer = model.model.layers[0]
    plus, context = positive_frontier(model, ids, mask)
    inputs = {name: plus[name] for name in S["INPUTS"]}
    bound = profile(plus, layer, mask, context)
    linear, silu, attention = F.linear, F.silu, F.scaled_dot_product_attention
    calls = dict(linear=0, SiLU=0, attention=0, module=0)

    def counted_linear(*args, **kwargs):
        calls["linear"] += 1
        return linear(*args, **kwargs)

    def counted_silu(*args, **kwargs):
        calls["SiLU"] += 1
        return silu(*args, **kwargs)

    def reject_attention(*args, **kwargs):
        calls["attention"] += 1
        raise AssertionError("shortcut attempted attention")

    handle = layer.register_forward_pre_hook(lambda *_args: calls.update(module=calls["module"] + 1))
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(F, "linear", counted_linear)
            scoped.setattr(F, "silu", counted_silu)
            scoped.setattr(F, "scaled_dot_product_attention", reject_attention)
            result = S["predict_negative"](inputs, layer.mlp.down_proj.weight, profile=bound)
    finally:
        handle.remove()
    assert F.scaled_dot_product_attention is attention
    assert calls == dict(linear=1, SiLU=1, attention=0, module=0)
    negative = native_negative(layer, -plus["h"], context)
    premises = {name: torch.equal(negative[name], -plus[name]) for name in S["INPUTS"]}
    assert all(premises.values())  # Synthetic mechanism qualification, not a production gate.
    for name in ("M", "Hnext"):
        assert torch.equal(result["ports"][name], negative[name])
    assert result["profile"]["attention_profile"]["query_rows"] == ("partially-padded" if padded else "all-attended")
    assert result["counts"]["gate_projection"] == result["counts"]["up_projection"] == result["counts"]["attention"] == 0
    assert result["comparison_max_elements"] == sum(result["ports"][key].numel() for key in ("product", "M", "Hnext"))
    (tmp_path / "actual-component-proof.json").write_text(json.dumps(dict(
        scope="Synthetic installed Qwen3 component conditional frontier only", padded=padded, source=bound["source_identity"],
        plus={key: S["tensor_descriptor"](value) for key, value in inputs.items()},
        negative={key: S["tensor_descriptor"](value) for key, value in negative.items()},
        shortcut=result["output_descriptors"], counts=result["counts"], premise_numeric=premises), indent=2, sort_keys=True))


def test_deliberately_wrong_upstream_premise_is_a_measured_miss_not_helper_gate(tmp_path):
    model, ids, mask = fixture(True)
    layer = model.model.layers[0]
    plus, context = positive_frontier(model, ids, mask)
    result = S["predict_negative"]({key: plus[key] for key in S["INPUTS"]}, layer.mlp.down_proj.weight,
        profile=profile(plus, layer, mask, context))
    wrong = -plus["h"]
    wrong[..., 0] += .5  # Deliberately not a complete global sign intervention.
    actual = native_negative(layer, wrong, context)
    delta = actual["Hnext"].double() - result["ports"]["Hnext"].double()
    report = dict(status="observation-only", helper_returned=True, premise="deliberately violated in synthetic target",
        orientation="actual-negative-consumer-minus-shortcut", rms=float(delta.square().mean().sqrt()),
        signed_mean=float(delta.mean()), maxabs=float(delta.abs().max()))
    assert report["rms"] > 0 and result["status"].startswith("computed")
    (tmp_path / "wrong-premise-observation.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def tensor_fixture():
    positive = dict(R=torch.tensor([[[0., -0., 1.]]], dtype=torch.bfloat16),
                    gate=torch.tensor([[[1., -2., 3., 0.]]], dtype=torch.bfloat16),
                    up=torch.tensor([[[4., 2., -1., -0.]]], dtype=torch.bfloat16))
    weight = torch.eye(3, 4, dtype=torch.bfloat16)
    attention = dict(backend="sdpa", gqa_policy="repeat_kv", query_rows="partially-padded",
        mask_identity="1" * 64, positions_identity="2" * 64, rope_identity="3" * 64)
    bound = S["make_profile"](positive, weight, source_identity={"synthetic_source": "4" * 64},
                            frontier_kind="own-functional-positive-ports", attention_profile=attention)
    return positive, weight, bound


def test_signed_zero_is_preserved_and_profile_inputs_are_unchanged():
    positive, weight, bound = tensor_fixture()
    before = {name: S["tensor_descriptor"](value) for name, value in positive.items()}
    result = S["predict_negative"](positive, weight, profile=bound)
    assert bool(torch.signbit(result["ports"]["negative_R"][0, 0, 0]))
    assert not bool(torch.signbit(result["ports"]["negative_R"][0, 0, 1]))
    assert before == {name: result["input_descriptors"][name] for name in positive}
    assert all(torch.equal(value, result["ports"][name]) for name, value in {"negative_R": -positive["R"],
        "negative_gate": -positive["gate"], "negative_up": -positive["up"]}.items())


def test_exact_positive_keys_dtype_layout_finite_and_runtime_profile_guards():
    positive, weight, bound = tensor_fixture()
    with pytest.raises(ValueError, match="exact positive"):
        S["predict_negative"]({**positive, "negative_target": positive["R"]}, weight, profile=bound)
    with pytest.raises(ValueError, match="BF16"):
        S["predict_negative"](positive, weight.float(), profile=bound)
    wrong = json.loads(json.dumps(bound))
    wrong["operands"]["gate"]["stride"] = [1, 1, 1]
    with pytest.raises(ValueError, match="layout differs"):
        S["predict_negative"](positive, weight, profile=wrong)
    wrong = json.loads(json.dumps(bound))
    wrong["execution"]["deterministic_algorithms"] = not wrong["execution"]["deterministic_algorithms"]
    with pytest.raises(ValueError, match="primitive policy"):
        S["predict_negative"](positive, weight, profile=wrong)
    bad = {**positive, "gate": positive["gate"].clone()}
    bad["gate"][0, 0, 0] = float("nan")
    bad_profile = S["make_profile"](bad, weight, source_identity=bound["source_identity"],
        frontier_kind=bound["frontier_kind"], attention_profile=bound["attention_profile"])
    with pytest.raises(ValueError, match="nonfinite input"):
        S["predict_negative"](bad, weight, profile=bad_profile)


def test_global_module_guard_restores_on_forbidden_call(monkeypatch):
    positive, weight, bound = tensor_fixture()
    module = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    original = F.silu

    def forbidden(value):
        return module(value)

    with monkeypatch.context() as scoped:
        scoped.setattr(F, "silu", forbidden)
        with pytest.raises(RuntimeError, match="module forward forbidden"):
            S["predict_negative"](positive, weight, profile=bound)
    assert F.silu is original
    assert module(positive["gate"]).shape == positive["gate"].shape
    assert S["predict_negative"](positive, weight, profile=bound)["counts"]["module_forwards"] == 0


def test_production_geometry_comparison_budget_without_large_allocation():
    budget = S["comparison_element_budget"](dict(product=[1, 512, 9728], M=[1, 512, 2560], Hnext=[1, 512, 2560]))
    assert budget == 7_602_176 and budget > 1_000_000
    with pytest.raises(ValueError, match="comparison shape"):
        S["comparison_element_budget"](dict(Hnext=[1, 0, 2560]))
