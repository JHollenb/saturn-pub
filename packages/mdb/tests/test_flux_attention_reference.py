"""Tiny native and scalar mechanics for the entry-only attention reference."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.models.transformers import transformer_flux2 as native

SOURCE = Path(__file__).parents[1] / "qualification" / "flux_attention_reference.py"
HELPER = runpy.run_path(str(SOURCE))


def widened(value):
    return value.detach().float().cpu().numpy().copy()


def fixture():
    torch.manual_seed(814)
    block = native.Flux2SingleTransformerBlock(8, 2, 4, mlp_ratio=2, bias=True).eval().to(torch.bfloat16)
    with torch.no_grad():
        block.attn.norm_q.weight.copy_((torch.rand(4) + .5).to(torch.bfloat16))
        block.attn.norm_k.weight.copy_((torch.rand(4) + .5).to(torch.bfloat16))
    residual = torch.randn(1, 5, 8).to(torch.bfloat16)
    mod = (torch.randn(1, 24) * .15).to(torch.bfloat16)
    phase = torch.arange(10).reshape(5, 2).float() * .13
    cos, sin = phase.cos().repeat_interleave(2, -1), phase.sin().repeat_interleave(2, -1)
    args = (widened(residual), widened(mod), widened(block.attn.to_qkv_mlp_proj.weight[:24]),
        widened(block.attn.to_qkv_mlp_proj.bias[:24]))
    kwargs = {"text_rows": 2, "heads": 2, "norm_q_weight": widened(block.attn.norm_q.weight),
        "norm_k_weight": widened(block.attn.norm_k.weight), "layernorm_eps": block.norm.eps,
        "qk_eps": block.attn.norm_q.eps, "rope_cos": widened(cos), "rope_sin": widened(sin),
        "query_indices": [0, 2], "query_chunk": 2, "projection_chunk": 3}
    return block, residual, mod, (cos, sin), args, kwargs


def test_current_native_modules_blocked_then_observed_parity(monkeypatch, record_property):
    block, residual, mod, rope, args, kwargs = fixture()
    def forbidden(*values):
        raise AssertionError("current native target entered during prediction")
    guards = [module.register_forward_pre_hook(forbidden) for module in block.modules()]
    try:
        prediction = HELPER["predict_attention_field"](*args, **kwargs)
    finally:
        for guard in guards:
            guard.remove()
    observed = {}
    original = native.dispatch_attention_fn
    def dispatch(query, key, value, **options):
        observed.update(q_rope=widened(query), k_rope=widened(key), v=widened(value))
        result = original(query, key, value, **options)
        observed["attention"] = widened(result.flatten(2, 3)[:, 2:])
        return result
    monkeypatch.setattr(native, "dispatch_attention_fn", dispatch)
    handles = [block.norm.register_forward_hook(lambda m, a, v: observed.update(layernorm=widened(v))),
        block.attn.norm_q.register_forward_hook(lambda m, a, v: observed.update(q_norm=widened(v))),
        block.attn.norm_k.register_forward_hook(lambda m, a, v: observed.update(k_norm=widened(v)))]
    try:
        with torch.inference_mode():
            block(hidden_states=residual, encoder_hidden_states=None, temb_mod=mod, image_rotary_emb=rope)
    finally:
        for handle in handles:
            handle.remove()
    errors = {"attention": float(np.max(np.abs(prediction["attention"] - observed["attention"])))}
    for name in ("layernorm", "q_norm", "k_norm", "q_rope", "k_rope", "v"):
        expected = observed[name][:, [2, 4]]
        errors[name] = float(np.max(np.abs(prediction["sampled"][name] - expected)))
    record_property("tiny_cpu_bf16_reference_max_abs", json.dumps(errors, sort_keys=True))
    # Engineering mechanics bound for this fixed tiny CPU fixture, not a science gate.
    assert max(errors.values()) < .04, errors
    assert prediction["attention"].shape == (1, 3, 8)
    assert prediction["sampled"]["key_indices"] == [0, 1, 2, 4]
    assert np.allclose(prediction["sampled"]["text_mass"] + prediction["sampled"]["image_mass"], 1)


def test_zero_qk_uses_all_text_and_image_values_and_bias():
    _, _, _, _, args, kwargs = fixture()
    residual, mod, weight, bias = (value.copy() for value in args)
    mod.fill(0)
    weight.fill(0)
    bias.fill(0)
    weight[16:] = np.eye(8, dtype=np.float32)
    bias[16:] = np.float32(.25)
    result = HELPER["predict_attention_field"](residual, mod, weight, bias, **kwargs)
    # Q=K=0: exact uniform softmax over all5 keys, independent of RoPE.
    x = torch.from_numpy(residual)
    with torch.inference_mode():
        norm = torch.nn.functional.layer_norm(x.to(torch.bfloat16), (8,), eps=kwargs["layernorm_eps"])
        values = (norm + .25).to(torch.bfloat16).float().numpy()
    expected = torch.from_numpy(values.astype(np.float64).mean(axis=1, keepdims=True)).to(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(result["attention"], np.repeat(expected, 3, axis=1))
    np.testing.assert_allclose(result["sampled"]["text_mass"], .4)
    changed = residual.copy()
    changed[:, 0] *= -1
    altered = HELPER["predict_attention_field"](changed, mod, weight, bias, **kwargs)
    assert not np.array_equal(altered["attention"], result["attention"])


def test_chunks_immutable_payload_and_source_nonmutation():
    _, _, _, _, args, kwargs = fixture()
    before = [value.tobytes() for value in args]
    first = HELPER["predict_attention_field"](*args, **kwargs)
    second = HELPER["predict_attention_field"](*args, **{**kwargs, "query_chunk": 1, "projection_chunk": 24})
    np.testing.assert_array_equal(first["attention"], second["attention"])
    assert all(value.tobytes() == original for value, original in zip(args, before, strict=True))
    assert not np.any(first["attention"].view(np.uint32) & 0xffff)
    with pytest.raises(ValueError):
        first["attention"].setflags(write=True)
    lowering = runpy.run_path(str(SOURCE.with_name("flux_mlp_replacement.py")))["prepare_mlp_payload"]
    reference = torch.empty_strided((1, 3, 8), (55, 16, 1), dtype=torch.bfloat16)
    payload = lowering(first["attention"], reference)
    assert payload.stride() == reference.stride()
    np.testing.assert_array_equal(widened(payload), first["attention"])


@pytest.mark.parametrize("bad", ["rope", "dtype", "epsilon", "query", "head", "bits"])
def test_invalid_source_contracts(bad):
    _, _, _, _, args, kwargs = fixture()
    if bad == "rope":
        kwargs["rope_cos"] = kwargs["rope_cos"][:, :2]
    elif bad == "dtype":
        args = (args[0].astype(np.float64), *args[1:])
    elif bad == "epsilon":
        kwargs["qk_eps"] = 0
    elif bad == "query":
        kwargs["query_indices"] = [3]
    elif bad == "head":
        kwargs["heads"] = 8
    else:
        args[0][0, 0, 0] = np.float32(.123456)
    with pytest.raises(ValueError):
        HELPER["predict_attention_field"](*args, **kwargs)


def test_predictor_import_path_blocks_all_model_frameworks(tmp_path):
    _, _, _, _, args, kwargs = fixture()
    path = tmp_path / "inputs.npz"
    np.savez(path, **{f"arg{i}": value for i, value in enumerate(args)}, **kwargs)
    script = """
import importlib.abc,runpy,sys,numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','diffusers','transformers','saturn','saturn_pub','mdb'}:
   raise ImportError('model dependency forbidden:'+name)
sys.meta_path.insert(0,Block())
data=np.load(sys.argv[2]);keys=dict(data);args=[keys.pop('arg'+str(i)) for i in range(4)]
for k in ('text_rows','heads','query_chunk','projection_chunk'):keys[k]=int(keys[k])
for k in ('layernorm_eps','qk_eps'):keys[k]=float(keys[k])
keys['query_indices']=keys['query_indices'].tolist()
result=runpy.run_path(sys.argv[1])['predict_attention_field'](*args,**keys)
assert result['attention'].shape==(1,3,8)
print('numpy-only prediction passed')
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(SOURCE), str(path)], capture_output=True, text=True, check=True)
    assert "numpy-only prediction passed" in result.stdout
