"""Meaningful mechanics for byte-scoped restores and autonomous cached futures."""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

QUAL = Path(__file__).resolve().parents[1] / "qualification"
RESTORE = runpy.run_path(str(QUAL / "qwen_cache_restoration.py"))
MATH = runpy.run_path(str(QUAL / "qwen_family_math.py"))


def cache_pair():
    torch.manual_seed(135)
    q = tuple(tuple(torch.randn(1, 2, 3, 4) for _ in range(2)) for _ in range(4))
    e = tuple(tuple(value.clone() for value in pair) for pair in q)
    for pair in e[2:]:
        pair[0][:, :, -1, :] += .25
        pair[1][:, :, -1, :] -= .5
    return q, e


def test_typed_factorial_scope_and_private_ownership():
    q, e = cache_pair()
    rows = RESTORE["support"](q, e, 2)
    assert [row["changed_rows"] for row in rows["rows"]] == [[], [], [], [], [2], [2], [2], [2]]
    corners, frames = RESTORE["corners"](q, e, 2, 42)
    for role, cache in corners.items():
        assert frames["corners"][role]["receipt"]["input_frame_id"] == frames["parent"]["frame_id"]
        for layer, pair in enumerate(cache):
            for component, value in enumerate(pair):
                expected = (e if role[component] == "E" and layer >= 2 else q)[layer][component]
                assert RESTORE["bit_equal"](value, expected)
                assert value.data_ptr() != q[layer][component].data_ptr()
                assert value.data_ptr() != e[layer][component].data_ptr()
    corners["EE"][2][0].zero_()
    assert RESTORE["bit_equal"](corners["EQ"][2][0], e[2][0])
    assert RESTORE["bit_equal"](corners["QQ"][2][0], q[2][0])


@pytest.mark.parametrize("kind", ["prefix", "lower-layer", "nonfinite", "signed-zero"])
def test_support_rejects_uninterpretable_source(kind):
    q, e = cache_pair()
    if kind == "prefix":
        e[2][0][0, 0, 0, 0] += 1
    elif kind == "lower-layer":
        e[0][1][0, 0, -1, 0] += 1
    elif kind == "nonfinite":
        e[3][0][0, 0, -1, 0] = float("nan")
    else:
        q[0][0][0, 0, 0, 0] = 0.
        e[0][0][0, 0, 0, 0] = -0.
    with pytest.raises(ValueError, match="support|finite"):
        RESTORE["support"](q, e, 2)


def test_all_four_corners_forecast_own_cached_native_futures():
    torch.manual_seed(981)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=14, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0., tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    engine = MATH["FunctionalQwen"](dict(model.named_parameters(remove_duplicate=False)), model.config)
    with torch.no_grad():
        pref = model(torch.tensor([[3, 7, 12]]), use_cache=True)
    q = tuple((layer.keys.clone(), layer.values.clone()) for layer in pref.past_key_values.layers)
    e = tuple(tuple(value.clone() for value in pair) for pair in q)
    e[13][0][:, :, -1, :] += torch.randn_like(e[13][0][:, :, -1, :]) * .3
    e[13][1][:, :, -1, :] += torch.randn_like(e[13][1][:, :, -1, :]) * .3
    corners, _ = RESTORE["corners"](q, e, 13, 9)
    before = MATH["clone_cache"](q)
    forecasts = {}
    forbidden = [module.register_forward_pre_hook(
        lambda module, args: (_ for _ in ()).throw(RuntimeError("native forbidden"))) for module in model.modules()]
    calls = dict(functional=0, native=0)

    def functional_step(ids, cache):
        ports = {}

        def capture(layer, stage, values):
            if stage in ("attention", "block"):
                key, field = ("a", "attention_return") if stage == "attention" else ("r", "output")
                ports[f"{key}{layer:02d}"] = values[field].clone()

        result = engine.forward(ids, cache=cache, capture=capture, logits_to_keep=1)
        result["ports"] = ports
        calls["functional"] += 1
        return result

    def retain(step, result, ids, token):
        return dict(step=step, logits=result["logits"].clone(), cache=MATH["clone_cache"](result["cache"]),
            ports=result["ports"], input=ids.clone(), token=token)

    try:
        with torch.inference_mode():
            for role, cache in corners.items():
                forecasts[role] = RESTORE["future"](functional_step, cache, 9, 5, retain)
    finally:
        for handle in forbidden:
            handle.remove()
    assert calls == dict(functional=20, native=0)
    captured = {}
    hooks = []

    def capture_native(name, value):
        captured[name] = value.detach().clone()

    for layer, module in enumerate(model.model.layers):
        hooks.append(module.self_attn.o_proj.register_forward_hook(
            lambda module, args, value, name=f"a{layer:02d}": capture_native(name, value)))
        hooks.append(module.register_forward_hook(
            lambda module, args, value, name=f"r{layer:02d}": capture_native(
                name, value[0] if isinstance(value, tuple) else value)))
    try:
        with torch.inference_mode():
            for role, cache in corners.items():
                state, incoming = RESTORE["native_cache"](model, cache), torch.tensor([[9]])
                for step, predicted in enumerate(forecasts[role]):
                    captured.clear()
                    actual = model(incoming, past_key_values=state, use_cache=True, logits_to_keep=1)
                    calls["native"] += 1
                    state = actual.past_key_values
                    assert RESTORE["bit_equal"](predicted["logits"], actual.logits)
                    assert torch.equal(predicted["input"], incoming)
                    for index, pair in enumerate(predicted["cache"]):
                        assert pair[0].shape[-2] == 4 + step
                        assert RESTORE["bit_equal"](pair[0], state.layers[index].keys)
                        assert RESTORE["bit_equal"](pair[1], state.layers[index].values)
                    assert set(predicted["ports"]) == set(captured)
                    for name, value in captured.items():
                        assert RESTORE["bit_equal"](predicted["ports"][name], value)
                    native_token = int(actual.logits[0, -1].argmax())
                    assert predicted["token"] == native_token
                    incoming = torch.tensor([[native_token]])
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == dict(functional=20, native=20)
    for old, current in zip(before, q, strict=True):
        assert RESTORE["bit_equal"](old[0], current[0])
        assert RESTORE["bit_equal"](old[1], current[1])
