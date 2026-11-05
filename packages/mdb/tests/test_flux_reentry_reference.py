"""Tiny explicit project/joint fixtures, never trained weights or real models."""
from __future__ import annotations

import ast
import hashlib
import inspect
import runpy
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "qualification/flux_reentry_reference.py"
R = runpy.run_path(str(PATH))


def b(value):
    return R["bf16"](np.asarray(value, np.float32))


def wide(value):
    return value.detach().float().cpu().numpy().copy()


def parameters(module):
    return {key: wide(value) for key, value in module.named_parameters()}


def tiny_model():
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    torch.manual_seed(812)
    return Flux2Transformer2DModel(in_channels=4, out_channels=4, num_layers=1,
        num_single_layers=1, attention_head_dim=4, num_attention_heads=2,
        joint_attention_dim=6, timestep_guidance_channels=8, mlp_ratio=2,
        axes_dims_rope=(2, 2), guidance_embeds=False).eval().to(torch.bfloat16)


def operands():
    torch = pytest.importorskip("torch")
    z = torch.tensor([[[.25, -.5, 1., 2.], [-1., .5, .75, -.25], [.5, 1., -.5, .25]]], dtype=torch.bfloat16)
    c = torch.tensor([[[1., -.5, .25, -.25, .5, -1.], [-.5, 1., -.25, .5, .25, 1.]]], dtype=torch.bfloat16)
    img_ids = torch.tensor([[[0, 0], [0, 1], [1, 0]]], dtype=torch.int64)
    txt_ids = torch.tensor([[[0, 0], [1, 1]]], dtype=torch.int64)
    return z, c, img_ids, txt_ids


def test_predictor_has_no_framework_or_dynamic_native_imports():
    tree = ast.parse(PATH.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.startswith(("torch", "diffusers", "saturn", "runpy", "importlib")) for name in imports)


def test_exact_installed_native_source_contract():
    import diffusers.models.embeddings as embeddings
    import diffusers.models.transformers.transformer_flux2 as transformer
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    pins = R["SOURCE_CONTRACT"]
    for key, value in (("transformer", transformer.Flux2TransformerBlock),
                       ("embeddings", embeddings.Timesteps), ("adapter", Flux2KleinAdapter)):
        assert sha(Path(inspect.getfile(value)).read_bytes()) == pins[key + "_module_sha256"]
    for value in (transformer.Flux2TransformerBlock, transformer.Flux2AttnProcessor, transformer.Flux2PosEmbed,
                  embeddings.get_timestep_embedding, embeddings.get_1d_rotary_pos_embed, embeddings.apply_rotary_emb):
        assert sha(inspect.getsource(value).encode()) == pins[value.__name__ + ".source_sha256"]


def test_adapter_timestep_roundtrip_and_flip_order_against_native():
    torch = pytest.importorskip("torch")
    from diffusers.models.embeddings import Timesteps

    for scalar in (0., 1., 166.4853057861328, 501.1):
        result = R["predict_timestep"](np.float32(scalar), 8)
        timestep = torch.tensor([scalar], dtype=torch.float32).to(torch.bfloat16)
        roundtrip = (timestep / 1000).to(torch.bfloat16) * 1000
        assert np.array_equal(result["roundtrip_timestep"], wide(roundtrip))
        expected = wide(Timesteps(8, True, 0)(roundtrip).to(torch.bfloat16))
        assert np.array_equal(result["sinusoidal_bf16"], expected)


def test_rope_uses_f64_angles_and_text_before_image():
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

    _, _, image, text = operands()
    actual = Flux2PosEmbed(theta=2000, axes_dim=[2, 2])
    native_i, native_t = actual(image[0]), actual(text[0])
    predicted = R["predict_rope"](image.numpy(), text.numpy(), axes_dims_rope=[2, 2])
    for index in (0, 1):
        expected = np.concatenate((wide(native_t[index]), wide(native_i[index])), axis=0)
        assert np.array_equal(predicted[index], expected)


def test_complete_project_matches_tiny_native_and_does_not_invoke_modules():
    torch = pytest.importorskip("torch")
    model = tiny_model()
    z, c, img_ids, txt_ids = operands()
    time = torch.tensor([166.4853057861328], dtype=torch.float32)
    t = time.to(torch.bfloat16)
    native_temb = model.time_guidance_embed((t / 1000).to(torch.bfloat16) * 1000, None)
    expected = {"temb": native_temb, "img_mod": model.double_stream_modulation_img(native_temb),
        "txt_mod": model.double_stream_modulation_txt(native_temb), "single_mod": model.single_stream_modulation(native_temb)}
    expected_i, expected_t = model.x_embedder(z), model.context_embedder(c)
    weights = {key: value for key, value in parameters(model).items()
        if any(key.startswith(name + ".") for name in R["PROJECT_LINEAR_NAMES"])}
    snapshots = {key: value.tobytes() for key, value in weights.items()}
    hook = torch.nn.modules.module.register_module_forward_pre_hook(lambda *args: (_ for _ in ()).throw(AssertionError("native call")))
    try:
        predicted = R["predict_project"](wide(z), wide(c), img_ids.numpy(), txt_ids.numpy(), time.numpy(), weights,
            axes_dims_rope=[2, 2], timestep_channels=8, projection_chunk=3)
    finally:
        hook.remove()
    assert np.array_equal(predicted["image"], wide(expected_i))
    assert np.array_equal(predicted["text"], wide(expected_t))
    for name, value in expected.items():
        # FP64 linear sums and native BF16 accumulation may differ by one word.
        np.testing.assert_allclose(predicted["context"][name], wide(value), rtol=.02, atol=.001)
    assert all(weights[key].tobytes() == raw for key, raw in snapshots.items())
    assert not predicted["image"].flags.writeable
    with pytest.raises(ValueError):
        predicted["image"].setflags(write=True)
    # Equal total token count cannot authorize swapping image/text positions.
    with pytest.raises(ValueError, match="complete-state geometry"):
        R["predict_project"](wide(z), wide(c), txt_ids.numpy(), img_ids.numpy(), time.numpy(), weights,
            axes_dims_rope=[2, 2], timestep_channels=8)


def test_full_joint_native_tiny_both_streams_and_chunk_independence():
    torch = pytest.importorskip("torch")
    model = tiny_model()
    z, c, img_ids, txt_ids = operands()
    t = torch.tensor([167.], dtype=torch.bfloat16)
    temb = model.time_guidance_embed(t, None)
    image, text = model.x_embedder(z), model.context_embedder(c)
    img_mod, txt_mod = model.double_stream_modulation_img(temb), model.double_stream_modulation_txt(temb)
    ri, rt = model.pos_embed(img_ids[0]), model.pos_embed(txt_ids[0])
    rope = tuple(torch.cat((rt[j], ri[j]), dim=0) for j in (0, 1))
    actual_t, actual_i = model.transformer_blocks[0](image, text, img_mod, txt_mod, rope)
    context = {"img_mod": wide(img_mod), "txt_mod": wide(txt_mod), "rope": tuple(wide(v) for v in rope)}
    weights = parameters(model.transformer_blocks[0])
    source_bytes = {key: value.tobytes() for key, value in weights.items()}
    result = R["predict_joint"](wide(image), wide(text), context, weights, heads=2, query_chunk=1, projection_chunk=3)
    other = R["predict_joint"](wide(image), wide(text), context, weights, heads=2, query_chunk=5, projection_chunk=8)
    assert result["image"].tobytes() == other["image"].tobytes()
    assert result["text"].tobytes() == other["text"].tobytes()
    np.testing.assert_allclose(result["image"], wide(actual_i), rtol=.02, atol=.015625)
    np.testing.assert_allclose(result["text"], wide(actual_t), rtol=.02, atol=.015625)
    assert all(weights[key].tobytes() == raw for key, raw in source_bytes.items())


def test_joint_uniform_attention_receiving_mlp_hand_calculation():
    # Q/K=0 yields uniform weights over every T/I position. Distinct stream
    # values, output projections, gates and receiving MLP expose ordering errors.
    width = 2
    weights = {}
    for name in R["JOINT_LINEAR_NAMES"]:
        weights[name + ".weight"] = b(np.eye(width))
    for name in ("attn.to_q", "attn.to_k", "attn.add_q_proj", "attn.add_k_proj"):
        weights[name + ".weight"] = b(np.zeros((2, 2)))
    for name in R["JOINT_NORM_NAMES"]:
        weights[name] = b(np.ones(2))
    for name in ("ff", "ff_context"):
        weights[name + ".linear_in.weight"] = b(np.zeros((4, 2)))
        weights[name + ".linear_in.bias"] = b([0, 0, 1, 1])
        weights[name + ".linear_out.weight"] = b(np.eye(2))
        weights[name + ".linear_out.bias"] = b([.25, -.5])
    image, text = b([[[1., -1.], [2., -2.]]]), b([[[-1., 1.]]])
    mod = b(np.zeros((1, 12)))
    mod[..., 4:6] = 1  # attention gate
    mod[..., 10:12] = .5  # sequential MLP gate
    context = {"img_mod": mod, "txt_mod": mod, "rope": (np.ones((3, 2), np.float32), np.zeros((3, 2), np.float32))}
    result = R["predict_joint"](image, text, context, weights, heads=1)
    # Native population LN values round to (+/-1); three-token average is 1/3.
    attention = b([1/3, -1/3])
    expected_i = b(b(image + attention) + b([.125, -.25]))
    expected_t = b(b(text + attention) + b([.125, -.25]))
    assert result["image"].tobytes() == expected_i.tobytes()
    assert result["text"].tobytes() == expected_t.tobytes()


def test_receiving_mlp_recomputes_after_attention_sign_flip():
    weights = {name + ".weight": b(np.eye(2)) for name in R["JOINT_LINEAR_NAMES"]}
    for name in ("attn.to_q", "attn.to_k", "attn.add_q_proj", "attn.add_k_proj"):
        weights[name + ".weight"] = b(np.zeros((2, 2)))
    for name in R["JOINT_NORM_NAMES"]:
        weights[name] = b(np.ones(2))
    for name in ("ff", "ff_context"):
        weights[name + ".linear_in.weight"] = b(np.concatenate((np.eye(2), np.zeros((2, 2))), axis=0))
        weights[name + ".linear_in.bias"] = b([0, 0, 1, 1])
    image, text = b([[[.125, -.125]]]), b([[[-2., 2.], [-2., 2.]]])
    mod = b(np.zeros((1, 12)))
    mod[..., 4:6] = mod[..., 10:12] = 1
    context = {"img_mod": mod, "txt_mod": mod, "rope": (np.ones((3, 2), np.float32), np.zeros((3, 2), np.float32))}
    actual = R["predict_joint"](image, text, context, weights, heads=1)
    incoming_norm = b([[[1., -1.]]])
    receiving_norm = b([[[-1., 1.]]])
    assert not np.array_equal(incoming_norm, receiving_norm)
    assert np.array_equal(actual["sampled"]["image"]["mlp_input"], receiving_norm)
    after_attention = b(image + b([-1/3, 1/3]))
    expected_mlp = b(np.array([-1., 1.]) / (1 + np.exp(-np.array([-1., 1.]))))
    assert actual["image"].tobytes() == b(after_attention + expected_mlp).tobytes()


@pytest.mark.parametrize("bad", [np.array([np.nan], np.float32), np.array([1.], np.float64), np.ones((2,), np.float32)])
def test_invalid_timestep_is_rejected(bad):
    with pytest.raises(ValueError):
        R["predict_timestep"](bad, 8)


def test_invalid_joint_operand_geometry_and_non_bf16_rejected():
    model = tiny_model()
    weights = parameters(model.transformer_blocks[0])
    context = {"img_mod": b(np.zeros((1, 48))), "txt_mod": b(np.zeros((1, 48))),
        "rope": (np.ones((5, 4), np.float32), np.zeros((5, 4), np.float32))}
    image, text = b(np.ones((1, 3, 8))), b(np.ones((1, 2, 8)))
    image[0, 0, 0] = np.float32(1.001)
    with pytest.raises(ValueError, match="BF16"):
        R["predict_joint"](image, text, context, weights, heads=2)
    assert "torch" not in inspect.getsource(R["predict_project"])
