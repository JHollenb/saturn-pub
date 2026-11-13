"""Text-query reference parity and separate residual/attention/MLP writers."""
import itertools
import runpy
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).parents[1]
helper=runpy.run_path(str(HERE/"qualification/flux_text_program.py"))
a=runpy.run_path(str(HERE/"qualification/flux_attention_reference.py"))["predict_attention_field"]
m=runpy.run_path(str(HERE/"qualification/flux_mlp_replacement.py"))["predict_mlp_field"]


def test_text_attention_packing_matches_native_original_order(monkeypatch):
    fixture=runpy.run_path(str(HERE/"tests/test_flux_attention_reference.py"))
    block,residual,mod,rope,args,kw=fixture["fixture"]();wide=fixture["widened"]
    predictions=helper["predict_text_fields"](wide(residual[:,:2]),wide(residual[:,2:]),wide(mod),qkv_weight=args[2],qkv_bias=args[3],
        mlp_weight=wide(block.attn.to_qkv_mlp_proj.weight[24:]),mlp_bias=wide(block.attn.to_qkv_mlp_proj.bias[24:]),norm_q=kw["norm_q_weight"],norm_k=kw["norm_k_weight"],
        eps=kw["layernorm_eps"],qk_eps=kw["qk_eps"],heads=2,rope_cos=kw["rope_cos"],rope_sin=kw["rope_sin"],attention_predictor=a,mlp_predictor=m)
    observed={}
    hook=block.attn.to_out.register_forward_pre_hook(lambda module,inputs:observed.update(concat=inputs[0].detach().clone()))
    try:
        with torch.inference_mode():block(hidden_states=residual,encoder_hidden_states=None,temb_mod=mod,image_rotary_emb=rope)
    finally:hook.remove()
    native=wide(observed["concat"][:,:2]);assert predictions["A"].shape==(1,2,8) and predictions["M"].shape==(1,2,16)
    # Fixed tiny CPU fixture bound, not a scientific semantic gate.
    assert np.max(np.abs(predictions["A"]-native[:,:,:8]))<.04
    assert np.max(np.abs(predictions["M"]-native[:,:,8:]))<.04


def test_text_RAM_cache_matches_all_eight_full_block_combinations():
    from saturn_pub import Act,Session
    from mdb_lanes.native_navigation import NativeCutNavigator
    model,adapter,session=runpy.run_path(str(HERE/"tests/test_flux_single_contexts.py"))["tiny_session"]()
    nav=NativeCutNavigator(session,max_steps=112);nav.until(logical_step=3,phase="single",index=8);low=nav.capture()
    variant=Session.from_cut(adapter,low);text=low.payload["text"];text[:,:,::3]+=.25;variant.apply(Act.replace("text",text,name="text-program"));high=variant.capture()
    block=model.single_transformer_blocks[8];capture=runpy.run_path(str(HERE/"qualification/flux_single_contraction.py"))["capture_candidate"]
    sources=[capture(block,cut,single_index=8) for cut in (low,high)];t,h=sources[0]["text_rows"],sources[0]["attention_width"]
    for r,ai,mi in itertools.product((0,1),repeat=3):
        attention=sources[ai]["ports"]["original_concat"][:,:t,:h];mlp=sources[mi]["ports"]["original_concat"][:,:t,h:]
        predicted=helper["text_exit"](block,sources[1],residual=(low,high)[r].payload["text"],attention=attention,mlp=mlp)
        checked=helper["check_text_override"](block,(low,high)[r],attention=attention,mlp=mlp,single_index=8)
        assert torch.equal(predicted,checked)
