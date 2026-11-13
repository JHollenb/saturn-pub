"""Factorial arithmetic and unchanged native port observation on tiny tensors."""
from pathlib import Path
import runpy

import numpy as np
import pytest

Q = Path(__file__).resolve().parents[1]/"qualification"
C = runpy.run_path(str(Q/"flux_stream_factorial.py"))


def test_corners_replace_only_the_selected_streams():
    native=dict(image=np.array([1.]),text=np.array([2.]),context=np.array([7.]))
    predicted=dict(image=np.array([3.]),text=np.array([5.]))
    for corner in ("00","10","01","11"):
        value=C["corner_state"](native,predicted,corner)
        assert value["context"] is native["context"]
        assert value["image"] is (predicted["image"] if corner[0]=="1" else native["image"])
        assert value["text"] is (predicted["text"] if corner[1]=="1" else native["text"])
    with pytest.raises(ValueError):C["corner_state"](native,predicted,"20")


def test_factorial_retains_signed_interaction_and_cancellation():
    values={k:np.array([v],np.float64) for k,v in dict(zip(("00","10","01","11"),(0.,3.,4.,2.))).items()}
    c=C["decompose"](values)
    assert c["image"].item()==3 and c["text"].item()==4
    assert c["interaction"].item()==-5 and c["total"].item()==2
    assert np.array_equal(c["closure"],np.zeros(1))


def test_observer_records_actual_ports_without_changing_native_values():
    torch=pytest.importorskip("torch")
    native=pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    Observer=runpy.run_path(str(Q/"flux_native_joint_ports.py"))["NativeJointPorts"]
    torch.manual_seed(19)
    block=native.Flux2TransformerBlock(dim=8,num_attention_heads=2,attention_head_dim=4,mlp_ratio=3.).eval().to(torch.bfloat16)
    image=torch.randn(1,4,8,dtype=torch.bfloat16);text=torch.randn(1,4,8,dtype=torch.bfloat16)
    mod=torch.randn(1,48,dtype=torch.bfloat16);cos,sin=torch.ones(8,4),torch.zeros(8,4)
    args=dict(hidden_states=image,encoder_hidden_states=text,temb_mod_img=mod,temb_mod_txt=mod,image_rotary_emb=(cos,sin))
    original=native.dispatch_attention_fn
    with torch.inference_mode():expected_text,expected_image=block(**args)
    counts=dict.fromkeys(("native_blocks","native_attention","native_image_ff","native_text_ff","native_linears","native_sdpa"),0)
    observer=Observer(block,native,counts,[0,3],[1,2],image_rows=4)
    with observer,torch.inference_mode():
        observer.begin();actual_text,actual_image=block(**args)
        fields,ports=observer.finish(dict(image=actual_image,text=actual_text))
    assert native.dispatch_attention_fn is original
    assert torch.equal(actual_image,expected_image) and torch.equal(actual_text,expected_text)
    assert len(fields)==len(ports)==24
    assert np.array_equal(fields["image"],actual_image.float().numpy())
    assert np.array_equal(fields["text"],actual_text.float().numpy())
    assert fields["image.q"].shape==fields["text.q"].shape==(1,2,2,4)
    assert counts==dict(native_blocks=1,native_attention=1,native_image_ff=1,native_text_ff=1,native_linears=12,native_sdpa=1)
