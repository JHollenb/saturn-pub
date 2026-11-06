"""Complete image/text and chunked-attention engineering mechanics; tiny tensors."""
from pathlib import Path
import runpy

import numpy as np
import pytest

C = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/flux_state_closure.py"))


def test_chunked_attention_keeps_all_keys_for_every_query():
    rng = np.random.default_rng(90)
    q, k, v = [rng.normal(size=(1, 2, 7, 4)) for _ in range(3)]
    L = C["helper"]("flux_attention_state_law.py")
    B = C["helper"]("flux_mlp_state_law.py")["quantize"]
    full, _ = L["dense"](q, k, v)
    assert np.array_equal(C["attention"](q, k, v, rows=2), B(full))


def test_complete_forecast_has_both_streams_and_composes_without_observed_inputs():
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    torch.manual_seed(101)
    block = native.Flux2TransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4, mlp_ratio=3.).eval().to(torch.bfloat16)
    # Nonzero FF computations on both streams, with exactly known row-local
    # normalization. Attention writes are disabled at the native modulation gate.
    image = torch.tensor([[[-1., 1.]*4]*4], dtype=torch.bfloat16)
    text = -image[:, :2].clone()
    mod = torch.zeros(1, 48, dtype=torch.bfloat16)
    mod[:, 40:] = 1
    state = {k:v.float().numpy() for k,v in dict(image=image,text=text,img_mod=mod,txt_mod=mod,cos=torch.ones(6,4),sin=torch.zeros(6,4)).items()}
    weights = {n:p.detach().float().numpy() for n,p in block.named_parameters()}
    p1 = C["forecast"](state, weights, [0,3], [0,1], heads=2, chunk_rows=2)
    with torch.no_grad():
        nt, ni = block(hidden_states=image,encoder_hidden_states=text,temb_mod_img=mod,temb_mod_txt=mod,image_rotary_emb=(torch.ones(6,4),torch.zeros(6,4)))
    assert p1["image"].shape == (1,4,8) and p1["text"].shape == (1,2,8)
    assert np.array_equal(p1["image"],ni.float().numpy())
    assert np.array_equal(p1["text"],nt.float().numpy())
    assert not np.array_equal(p1["image"],state["image"])
    assert not np.array_equal(p1["text"],state["text"])
    p2 = C["forecast"]({**state,"image":p1["image"],"text":p1["text"]},weights,[0,3],[0,1],heads=2,chunk_rows=2)
    assert np.isfinite(p2["image"]).all() and np.isfinite(p2["text"]).all()


def test_quarter_state_is_fixed_before_execution_and_preserves_other_inputs():
    arrays = {k:np.zeros((1,2,8),np.float32) for k in ("image","text_base","text_changed")}
    arrays["text_changed"][:] = 4
    arrays.update({k:np.ones((2,4),np.float32) for k in ("img_mod","txt_mod","cos","sin")})
    cases = C["states"](arrays)
    assert np.all(cases["quarter"]["text"] == 1) and np.all(cases["midpoint"]["text"] == 2)
    assert all(cases[n]["image"] is arrays["image"] for n in cases)


def test_full_text_and_image_attention_use_joint_values_and_both_output_projections():
    torch = pytest.importorskip("torch")
    native = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    block = native.Flux2TransformerBlock(dim=8,num_attention_heads=2,attention_head_dim=4,mlp_ratio=3.).eval().to(torch.bfloat16)
    with torch.no_grad():
        for module in (block.attn.to_q,block.attn.to_k,block.attn.add_q_proj,block.attn.add_k_proj):
            module.weight.zero_()
        eye = torch.eye(8,dtype=torch.bfloat16)
        block.attn.to_v.weight.copy_(eye)
        block.attn.add_v_proj.weight.copy_(2*eye)
        block.attn.to_out[0].weight.copy_(eye)
        block.attn.to_add_out.weight.copy_(eye)
    image = torch.tensor([[[-1.,1.]*4]*4],dtype=torch.bfloat16)
    text = image.clone()
    mod = torch.zeros(1,48,dtype=torch.bfloat16);mod[:,16:24]=1
    cos,sin = torch.ones(8,4),torch.zeros(8,4)
    state={k:v.float().numpy() for k,v in dict(image=image,text=text,img_mod=mod,txt_mod=mod,cos=cos,sin=sin).items()}
    weights={n:p.detach().float().numpy() for n,p in block.named_parameters()}
    predicted=C["forecast"](state,weights,[0,3],[1,2],heads=2,chunk_rows=3)
    with torch.no_grad():
        nt,ni=block(hidden_states=image,encoder_hidden_states=text,temb_mod_img=mod,temb_mod_txt=mod,image_rotary_emb=(cos,sin))
    assert np.array_equal(predicted["image"],ni.float().numpy())
    assert np.array_equal(predicted["text"],nt.float().numpy())
    assert np.array_equal(predicted["image"],2.5*image.float().numpy())
    assert np.array_equal(predicted["text"],2.5*text.float().numpy())
