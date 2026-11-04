"""Tiny native/full-algebra qualification; no real checkpoint or model job."""
from __future__ import annotations

import hashlib
import math
import runpy
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock

SOURCE = Path(__file__).parents[1] / "qualification/flux_transition_reference.py"
REFERENCE = runpy.run_path(str(SOURCE))


def controlled_block(*, bias=True):
    """Two-dimensional native block with uniform attention and constant MLP."""
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        block = Flux2SingleTransformerBlock(dim=2,num_attention_heads=1,attention_head_dim=2,
            mlp_ratio=1.5,eps=3.,bias=bias).eval().to(torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        block.attn.norm_q.weight.copy_(torch.tensor([1.,2.]))
        block.attn.norm_k.weight.copy_(torch.tensor([2.,1.]))
        # Q=K=0 gives exact uniform attention over BOTH source positions.
        block.attn.to_qkv_mlp_proj.weight[4:6].copy_(torch.eye(2))
        block.attn.to_out.weight[:,:2].copy_(torch.eye(2))
        if bias:
            block.attn.to_qkv_mlp_proj.bias[4:6].copy_(torch.tensor([.25,-.5]))
            block.attn.to_qkv_mlp_proj.bias[6:9].copy_(torch.tensor([1.,-1.,.5]))
            block.attn.to_qkv_mlp_proj.bias[9:12].copy_(torch.tensor([2.,1.,-2.]))
            block.attn.to_out.weight[0,2]=.25
            block.attn.to_out.weight[1,3]=.5
            block.attn.to_out.bias.copy_(torch.tensor([.125,.25]))
    return block


def arrays(block):
    def wide(value):
        return None if value is None else value.detach().float().numpy().copy()
    return {"qkv_mlp_weight":wide(block.attn.to_qkv_mlp_proj.weight),
        "qkv_mlp_bias":wide(block.attn.to_qkv_mlp_proj.bias),"to_out_weight":wide(block.attn.to_out.weight),
        "to_out_bias":wide(block.attn.to_out.bias),"norm_q_weight":wide(block.attn.norm_q.weight),
        "norm_k_weight":wide(block.attn.norm_k.weight)}


def inputs():
    return (np.array([[[1.,-1.],[-1.,1.]]],np.float32),
        np.array([[0.,0.,0.,0.,.5,.5]],np.float32),
        {"heads":1,"layernorm_eps":3.,"qk_eps":3.,"rope_cos":np.ones((2,2),np.float32),
         "rope_sin":np.zeros((2,2),np.float32),"text_rows":1,"sample_indices":[0,1],
         "query_chunk":1,"projection_chunk":2,"neuron_chunk":2})


def digest(values):
    return [None if value is None else hashlib.sha256(value.tobytes()).hexdigest() for value in values]


def test_complete_transition_matches_independent_scalar_algebra_and_native_both_streams():
    block=controlled_block()
    residual,mod,profile=inputs()
    parameters=arrays(block)
    sources=[residual,mod,*parameters.values(),profile["rope_cos"],profile["rope_sin"]]
    before=digest(sources)
    calls=[]
    hook=block.register_forward_pre_hook(lambda module,args:calls.append(True))
    try:
        predicted=REFERENCE["predict_transition"](residual,mod,**parameters,**profile)
        assert not calls
        with torch.inference_mode():
            text,image=block(hidden_states=torch.from_numpy(residual[:,1:]).to(torch.bfloat16),encoder_hidden_states=torch.from_numpy(residual[:,:1]).to(torch.bfloat16),
                temb_mod=torch.from_numpy(mod).to(torch.bfloat16),image_rotary_emb=(torch.ones(2,2),torch.zeros(2,2)),
                split_hidden_states=True,text_seq_len=1)
    finally:
        hook.remove()
    assert len(calls)==1 and digest(sources)==before
    # LN([1,-1],eps3)=[.5,-.5]; uniform A is mean(V)=[.25,-.5].
    # M0=B(B(SiLU(1))*2)=1.4609375; M1=B(SiLU(-1))=-.26953125.
    # Whole native projection includes bias before B; ties round to even.
    expected_attention=np.array([[[.25,-.5],[.25,-.5]]],np.float32)
    expected_projection=np.array([[[.7421875,-.384765625],[.7421875,-.384765625]]],np.float32)
    expected_write=np.array([[[.37109375,-.1923828125],[.37109375,-.1923828125]]],np.float32)
    expected_output=np.array([[[1.375,-1.1953125],[-.62890625,.80859375]]],np.float32)
    np.testing.assert_array_equal(predicted["attention"],expected_attention)
    np.testing.assert_array_equal(predicted["projection"],expected_projection)
    np.testing.assert_array_equal(predicted["write"],expected_write)
    np.testing.assert_array_equal(predicted["output"],expected_output)
    np.testing.assert_array_equal(predicted["output"],torch.cat((text,image),dim=1).float().numpy())
    np.testing.assert_array_equal(predicted["text"],text.float().numpy())
    np.testing.assert_array_equal(predicted["image"],image.float().numpy())
    for value in (predicted["output"],predicted["attention"],predicted["text"],predicted["image"]):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_no_bias_full_output_survives_zero_text_split_and_chunk_changes():
    block=controlled_block(bias=False)
    residual,mod,profile=inputs()
    residual[:]=np.array([1.,-1.],np.float32)
    parameters=arrays(block)
    first=REFERENCE["predict_transition"](residual,mod,**parameters,**profile)
    other=REFERENCE["predict_transition"](residual,mod,**parameters,
        **{**profile,"text_rows":0,"query_chunk":2,"projection_chunk":3,"neuron_chunk":3,"retain_fields":False})
    expected=np.array([[[1.25,-1.25],[1.25,-1.25]]],np.float32)
    np.testing.assert_array_equal(first["output"],expected)
    np.testing.assert_array_equal(other["output"],expected)
    assert other["text"].shape==(1,0,2) and other["image"].shape==expected.shape
    assert not {"attention","mlp","projection","write"}.intersection(other)


def test_all_query_nonuniform_attention_rms_rope_and_output_match_scalar_oracle():
    def bfloat(value):
        word=struct.unpack("<I",struct.pack("<f",value))[0]
        upper,lower=word>>16,word&65535
        if lower>32768 or (lower==32768 and upper%2):upper+=1
        return struct.unpack("<f",struct.pack("<I",upper<<16))[0]
    block=controlled_block(bias=False)
    with torch.no_grad():
        block.attn.to_qkv_mlp_proj.weight[:2].copy_(torch.eye(2))
        block.attn.to_qkv_mlp_proj.weight[2:4].copy_(torch.eye(2))
    residual,mod,profile=inputs()
    profile["rope_cos"]=np.array([[1.,1.],[0.,0.]],np.float32)
    profile["rope_sin"]=np.array([[0.,0.],[1.,1.]],np.float32)
    prediction=REFERENCE["predict_transition"](residual,mod,**arrays(block),**profile)
    source=[[.5,-.5],[-.5,.5]]
    q=[[bfloat(x*gamma/math.sqrt(3.25)) for x,gamma in zip(row,[1,2],strict=True)] for row in source]
    k=[[bfloat(x*gamma/math.sqrt(3.25)) for x,gamma in zip(row,[2,1],strict=True)] for row in source]
    q[1],k[1]=[-q[1][1],q[1][0]],[-k[1][1],k[1][0]]
    expected=[]
    for query in q:
        logits=[sum(x*y for x,y in zip(query,key,strict=True))/math.sqrt(2) for key in k]
        exps=[math.exp(value-max(logits)) for value in logits]
        probability=[value/sum(exps) for value in exps]
        expected.append([bfloat(sum(probability[j]*source[j][i] for j in range(2))) for i in range(2)])
    expected=np.array([expected],np.float32)
    np.testing.assert_array_equal(prediction["attention"],expected)
    with torch.inference_mode():
        actual=block(hidden_states=torch.from_numpy(residual).to(torch.bfloat16),encoder_hidden_states=None,
            temb_mod=torch.from_numpy(mod).to(torch.bfloat16),
            image_rotary_emb=(torch.from_numpy(profile["rope_cos"]),torch.from_numpy(profile["rope_sin"])))
    np.testing.assert_array_equal(prediction["output"],actual.float().numpy())


def test_two_native_blocks_chain_own_output_without_native_intermediate_input():
    first,second=controlled_block(),controlled_block(bias=False)
    with torch.no_grad():
        second.attn.to_out.weight.zero_()
    residual,mod,profile=inputs()
    blocks=[{**arrays(first),"block_index":6},{**arrays(second),"block_index":7}]
    before=digest([residual,mod,*[value for block in blocks for key,value in block.items() if key!="block_index"]])
    predicted=REFERENCE["predict_chain"](residual,mod,blocks,**profile,retain_fields=False)
    stage0,stage1=predicted["steps"]
    assert stage1["input_source"]=="previous predicted full output"
    assert stage1["input_descriptor"]==stage0["prediction"]["output_descriptor"]
    assert [step["block_index"] for step in predicted["steps"]]==[6,7]
    assert stage1["prediction"]["sampled"]["input"].tobytes()==stage0["prediction"]["output"].tobytes()
    with torch.inference_mode():
        native=torch.from_numpy(residual).to(torch.bfloat16)
        for block in (first,second):
            native=block(hidden_states=native,encoder_hidden_states=None,temb_mod=torch.from_numpy(mod).to(torch.bfloat16),
                image_rotary_emb=(torch.ones(2,2),torch.zeros(2,2)))
    np.testing.assert_array_equal(predicted["output"],native.float().numpy())
    assert before==digest([residual,mod,*[value for block in blocks for key,value in block.items() if key!="block_index"]])
    with pytest.raises(ValueError,match="parameter inventory"):
        REFERENCE["predict_chain"](residual,mod,[{**blocks[0],"native_intermediate":residual}],**profile)


@pytest.mark.parametrize("changed",["state_bits","output_shape","rope_shape","gamma_shape","sample"])
def test_invalid_source_contract_rejected_without_native_calls(changed):
    residual,mod,profile=inputs()
    parameters=arrays(controlled_block())
    if changed=="state_bits":residual[0,0,0]=np.float32(1.001)
    if changed=="output_shape":parameters["to_out_weight"]=parameters["to_out_weight"][:,:-1]
    if changed=="rope_shape":profile["rope_cos"]=profile["rope_cos"][:1]
    if changed=="gamma_shape":parameters["norm_q_weight"]=np.ones(1,np.float32)
    if changed=="sample":profile["sample_indices"]=[0,2]
    with pytest.raises(ValueError):
        REFERENCE["predict_transition"](residual,mod,**parameters,**profile)


def test_pinned_dependency_refuses_source_identity_drift(monkeypatch):
    residual,mod,profile=inputs()
    parameters=arrays(controlled_block())
    monkeypatch.setitem(REFERENCE["predict_transition"].__globals__,"MLP_HELPER_SHA256","0"*64)
    with pytest.raises(ValueError,match="dependency source differs"):
        REFERENCE["predict_transition"](residual,mod,**parameters,**profile)


def test_full_reference_computes_under_framework_import_blocker():
    code='''import importlib.abc,runpy,sys,numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common','common_harness'}:raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
m=runpy.run_path(sys.argv[1]);r=np.array([[[1.,-1.],[-1.,1.]]],np.float32)
p=m['predict_transition'](r,np.zeros((1,6),np.float32),qkv_mlp_weight=np.zeros((8,2),np.float32),to_out_weight=np.zeros((2,3),np.float32),norm_q_weight=np.ones(2,np.float32),norm_k_weight=np.ones(2,np.float32),heads=1,layernorm_eps=1e-6,qk_eps=1e-6,rope_cos=np.ones((2,2),np.float32),rope_sin=np.zeros((2,2),np.float32),retain_fields=False)
assert np.array_equal(p['output'],r) and not p['output'].flags.writeable
'''
    result=subprocess.run([sys.executable,"-I","-c",code,str(SOURCE)],capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr
