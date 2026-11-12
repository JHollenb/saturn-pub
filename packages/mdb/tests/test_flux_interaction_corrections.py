"""Correction corners and native cached consumer mechanics."""
import runpy
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).parents[1]
algebra=runpy.run_path(str(HERE/"qualification/flux_interaction_corrections.py"))["correction_fields"]


def test_opposing_terms_keep_distinct_correction_corners_without_mutating_sources():
    fields={k:np.full((1,4,8),v,np.float32) for k,v in (("base",1),("color",2),("shape",3),("additive",5),("joint",4))}
    copies={k:v.copy() for k,v in fields.items()};corners,terms=algebra(fields)
    np.testing.assert_array_equal(corners["r0c0"],4)
    np.testing.assert_array_equal(corners["r0c1"],5)
    np.testing.assert_array_equal(corners["r1c0"],3)
    np.testing.assert_array_equal(corners["r1c1"],4)
    np.testing.assert_array_equal(terms["consumer"],-terms["representation"])
    for k,v in copies.items():np.testing.assert_array_equal(v,fields[k])


def test_round_once_retains_additive_and_joint_corners():
    import torch
    generator=torch.Generator().manual_seed(44)
    fields={k:torch.randn((1,7,16),generator=generator).to(torch.bfloat16).float().numpy() for k in ("base","color","shape","additive","joint")}
    corners,_=algebra(fields)
    np.testing.assert_array_equal(corners["r0c1"],fields["additive"])
    np.testing.assert_array_equal(corners["r1c1"],fields["joint"])
    assert all(not np.any(v.view(np.uint32)&65535) for v in corners.values())


def test_incomplete_nonfinite_or_non_bf16_sources_are_refused():
    fields={k:np.ones((1,4,8),np.float32) for k in ("base","color","shape","additive","joint")}
    with pytest.raises(ValueError):algebra({k:v for k,v in fields.items() if k!="joint"})
    fields["joint"]=np.full((1,4,8),np.nan,np.float32)
    with pytest.raises(ValueError):algebra(fields)
    fields["joint"]=np.full((1,4,8),.10001,np.float32)
    with pytest.raises(ValueError):algebra(fields)


def test_fixed_image_attention_projection_keeps_receiving_native_text():
    import torch
    from saturn_pub import Session
    from mdb.native_navigation import NativeCutNavigator
    tiny=runpy.run_path(str(HERE/"tests/test_flux_single_contexts.py"))
    model,adapter,session=tiny["tiny_session"]()
    nav=NativeCutNavigator(session,max_steps=112);nav.until(logical_step=3,phase="single",index=7);base=nav.capture()
    capture=runpy.run_path(str(HERE/"qualification/flux_single_contraction.py"))["capture_candidate"]
    from saturn_pub import Act
    variant=Session.from_cut(adapter,base);variant.apply(Act.replace("text",base.payload["text"]+.125,name="changed-program"));joint=variant.capture()
    block=model.single_transformer_blocks[7];fixed=capture(block,base,single_index=7);target=capture(block,joint,single_index=7)
    t,h=fixed["text_rows"],fixed["attention_width"]
    operand=fixed["ports"]["original_concat"].clone();operand[:,t:,:h]=target["ports"]["original_concat"][:,t:,:h]
    with torch.inference_mode():cached=(fixed["ports"]["residual"]+fixed["ports"]["gate"]*block.attn.to_out(operand))[:,t:]
    for cut,recipient in ((base,fixed),(joint,target)):
        actual=capture(block,cut,single_index=7,selection="A",source={"concat":operand,"parent":"sealed-prediction-identity","text_rows":t,"attention_width":h})
        assert torch.equal(actual["image"],cached)
        assert torch.equal(actual["text"],recipient["text"])
