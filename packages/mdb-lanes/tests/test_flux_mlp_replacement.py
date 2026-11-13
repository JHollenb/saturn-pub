"""Full-field reference arithmetic, exact sealed cast and native consumption."""
from __future__ import annotations

import math
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from saturn_pub import Session

from mdb_lanes.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1]
q = runpy.run_path(str(HERE / "qualification/flux_mlp_replacement.py"))
tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))


def bfloat(value):
    return torch.as_tensor(np.array(value, copy=True)).float().to(torch.bfloat16).float().numpy()


def test_full_field_matches_scalar_bf16_reference_and_chunk_partition_is_lossless():
    rng = np.random.default_rng(91)
    x = bfloat(rng.normal(size=(1, 4, 5)))
    mod = bfloat(rng.normal(size=(1, 15)))
    weight = bfloat(rng.normal(size=(14, 5)))
    bias = bfloat(rng.normal(size=14))
    originals = [value.copy() for value in (x, mod, weight, bias)]
    for value in (x, mod, weight, bias):
        value.setflags(write=False)
    results = [q["predict_mlp_field"](x, mod, weight, bias, eps=1e-6, hidden_width=5, mlp_width=7,
        neuron_chunk=chunk, query_indices=[0, 3], neuron_indices=[0, 6]) for chunk in (1, 3, 7, 256)]
    for result in results[1:]:
        np.testing.assert_array_equal(result["activation"], results[0]["activation"])
    expected = np.empty((1, 4, 7), dtype=np.float32)
    for row in range(4):
        values = [float(value) for value in x[0, row]]
        mean = math.fsum(values) / 5
        variance = math.fsum((value - mean) ** 2 for value in values) / 5
        norm = bfloat([(value - mean) / math.sqrt(variance + 1e-6) for value in values])
        factor = bfloat(1 + mod[0, 5:10].astype(np.float64))
        product = bfloat(factor * norm)
        affine = bfloat(product.astype(np.float64) + mod[0, :5].astype(np.float64))
        uv = bfloat([math.fsum(float(a) * float(b) for a, b in zip(affine, coefficients, strict=True)) + float(offset)
            for coefficients, offset in zip(weight, bias, strict=True)])
        silu = bfloat([float(value) / (1 + math.exp(-float(value))) for value in uv[:7]])
        expected[0, row] = bfloat(silu * uv[7:])
    np.testing.assert_array_equal(results[0]["activation"], expected)
    sampled = results[0]["sampled"]
    np.testing.assert_array_equal(sampled["activation"], expected[:, [0, 3]][:, :, [0, 6]])
    assert sampled["layernorm"].shape == sampled["modulated"].shape == (1, 2, 5)
    for result in results:
        assert result["activation"].dtype == np.float32 and not result["activation"].flags.writeable
        assert not np.any(result["activation"].view(np.uint32) & np.uint32(0xffff))
        with pytest.raises(ValueError):
            result["activation"].setflags(write=True)
    for actual, original in zip((x, mod, weight, bias), originals, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_native_mlp_payload_footprint_identity_roll_and_typed_future(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copyfile(HERE / "qualification/flux_single_contraction.py", "flux_single_contraction.py")
    legacy = runpy.run_path("flux_single_contraction.py")
    model, adapter, session = tiny["tiny_session"]()
    n = NativeCutNavigator(session, max_steps=100)
    n.until(logical_step=3, phase="single", index=6)
    cut = n.capture()
    block = model.single_transformer_blocks[6]
    native = legacy["capture_candidate"](block, cut)
    h, i, t = native["attention_width"], block.attn.mlp_hidden_dim, native["text_rows"]
    fused = block.attn.to_qkv_mlp_proj
    weight = fused.weight[3 * h:3 * h + 2 * i].detach().float().numpy().copy()
    bias = fused.bias[3 * h:3 * h + 2 * i].detach().float().numpy().copy() if fused.bias is not None else None
    result = q["predict_mlp_field"](cut.payload["image"].float().numpy(), cut.payload["context"]["single_mod"].float().numpy(),
        weight, bias, eps=block.norm.eps, neuron_chunk=5, query_indices=[0, 3], neuron_indices=[0, i - 1])
    reference = native["ports"]["original_concat"][:, t:, h:]
    prediction = q["prepare_mlp_payload"](result["activation"], reference)
    assert prediction.stride() == reference.stride()
    predicted = q["capture_replacement"](block, cut, native, prediction)
    assert all(predicted["replacement_controls"].values())
    np.testing.assert_array_equal(predicted["ports"]["used_concat"][:, t:, h:].float().numpy(), result["activation"])
    measured = q["prepare_mlp_payload"](reference.float().numpy(), reference)
    identity = q["capture_replacement"](block, cut, native, measured)
    assert torch.equal(identity["image"], native["image"]) and torch.equal(identity["text"], native["text"])
    rolled = q["prepare_mlp_payload"](result["activation"], reference, roll=2)
    changed = q["capture_replacement"](block, cut, native, rolled)
    np.testing.assert_array_equal(changed["ports"]["used_concat"][:, t:, h:].float().numpy(), np.roll(result["activation"], 2, axis=1))
    assert torch.equal(changed["ports"]["used_concat"][:, t:, :h], native["ports"]["original_concat"][:, t:, :h])
    assert torch.equal(changed["text"], native["text"])
    # Public typed install and native suffix consume an exact identity payload.
    from saturn_pub.store import LocalStore
    baseline = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=20)
    baseline.step()
    parent = baseline.capture()
    edited = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=20)
    act = legacy["install_port"](edited, "image", identity["image"], name="measured-MLP-identity", store=LocalStore(tmp_path / "store"))
    assert act["undeclared_slots_exact"]
    baseline.until(logical_step=4, phase="project")
    edited.until(logical_step=4, phase="project")
    assert torch.equal(edited.session.read("latent"), baseline.session.read("latent"))
    assert not block.attn.to_out._forward_hooks and not block.attn.to_out._forward_pre_hooks


def test_payload_cast_and_source_geometry_reject_unsafe_or_unsealed_values():
    reference = torch.empty_strided((1, 4, 7), (80, 10, 1), dtype=torch.bfloat16)
    valid = np.zeros((1, 4, 7), dtype=np.float32)
    assert q["prepare_mlp_payload"](valid, reference).stride() == reference.stride()
    for value, roll in ((valid.astype(np.float64), 0), (valid[:, :, :6], 0), (valid, -1), (valid, 4)):
        with pytest.raises(ValueError):
            q["prepare_mlp_payload"](value, reference, roll=roll)
    bad = valid.copy()
    bad[0, 0, 0] = np.nextafter(np.float32(1), np.float32(2))
    with pytest.raises(ValueError, match="BF16"):
        q["prepare_mlp_payload"](bad, reference)
    with pytest.raises(ValueError):
        q["predict_mlp_field"](np.zeros((1, 4, 5), np.float32), np.zeros((1, 15), np.float32), np.zeros((14, 5), np.float32),
            eps=1e-6, neuron_chunk=0, query_indices=[0], neuron_indices=[0])


def test_numpy_predictor_is_framework_free_and_branch_recipes_are_fixed():
    assert [row["id"] for row in q["BRANCHES"]] == ["zero", "text", "sham", "low.predicted", "M.measured", "M.predicted", "M.measured.roll", "M.predicted.roll", "high.predicted"]
    assert [row["roll"] for row in q["BRANCHES"]].count(512) == 2
    code = """import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn_pub','saturn'}:raise ImportError('framework blocked')
sys.meta_path.insert(0,Block())
import numpy as np
q=runpy.run_path(sys.argv[1])
result=q['predict_mlp_field'](np.ones((1,2,3),np.float32),np.zeros((1,9),np.float32),np.ones((10,3),np.float32),eps=1e-6,query_indices=[0],neuron_indices=[0])
assert result['activation'].shape==(1,2,5) and result['activation'].dtype==np.float32
assert not result['activation'].flags.writeable and 'torch' not in sys.modules
print('numpy-only')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(HERE / "qualification/flux_mlp_replacement.py")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "numpy-only"
