"""Text-side prediction and native RAM writer cuts without changing old ABIs."""
from __future__ import annotations


def predict_text_fields(text,image,mod,*,qkv_weight,qkv_bias,mlp_weight,mlp_bias,norm_q,norm_k,
                        eps,qk_eps,heads,rope_cos,rope_sin,attention_predictor,mlp_predictor):
    """Use entry states and actual weights only; no native target returns.

    The existing complete attention reference returns a trailing query bank.
    Permute [T;I] to [I;T], moving RoPE operands with their rows. Unmasked
    self-attention is equivariant to this simultaneous row/key permutation.
    Thus the trailing bank is original T attending to every original key.
    FP64 key-reduction order differs; retain native errors, not exactness claims.
    """
    import numpy as np
    if (text.ndim!=3 or image.ndim!=3 or text.shape[0]!=1 or image.shape[0]!=1
            or text.shape[2]!=image.shape[2] or text.shape[1]<2):raise ValueError("text/image geometry differs")
    t=text.shape[1];n=image.shape[1];queries=list(dict.fromkeys((0,min(31,t-1),min(127,t-1),t-1)))
    packed=np.concatenate((image,text),axis=1)
    cos=np.concatenate((rope_cos[t:],rope_cos[:t]),axis=0);sin=np.concatenate((rope_sin[t:],rope_sin[:t]),axis=0)
    a=attention_predictor(packed,mod,qkv_weight,qkv_bias,text_rows=n,heads=heads,norm_q_weight=norm_q,norm_k_weight=norm_k,
        layernorm_eps=eps,qk_eps=qk_eps,rope_cos=cos,rope_sin=sin,query_indices=queries)
    m=mlp_predictor(text,mod,mlp_weight,mlp_bias,eps=eps,query_indices=queries,
        neuron_indices=list(range(0,mlp_weight.shape[0]//2,max(1,(mlp_weight.shape[0]//2)//32))))
    return {"A":a["attention"],"M":m["activation"],"attention_metadata":{"heads":heads,"text_queries":t,"all_keys":t+n,
        "packing":"Original [T;I] and matchingRoPE permuted [I;T], trailingTqueries, all keys retained. Original text_mass is packed image_mass; no semantic mass attribution.",
        "scope":"Reference arithmetic conditional on observed entry; not exact native CUDA kernels or raw-prompt future prediction."}}


def text_write(block,source,*,attention,mlp):
    """Native fixed-geometry projection/gate; arbitrary named text operands."""
    import torch
    t,h=source["text_rows"],source["attention_width"]
    cat=source["ports"]["original_concat"].clone()
    if attention.shape!=(1,t,h) or mlp.shape!=cat[:,:t,h:].shape:raise ValueError("text writer geometry differs")
    cat[:,:t,:h]=attention;cat[:,:t,h:]=mlp
    with torch.inference_mode():
        projected=block.attn.to_out(cat)
        return source["ports"]["gate"]*projected[:,:t]


def text_exit(block,source,*,residual,attention,mlp):
    return residual+text_write(block,source,attention=attention,mlp=mlp)


def check_text_override(block,cut,*,attention,mlp,single_index):
    """Actual full-block text-A/M override; residual follows declared entry."""
    import torch
    t,h=cut.payload["text"].shape[1],cut.payload["image"].shape[-1]
    if cut.execution_point["index"]!=single_index:raise ValueError("wrong declared block")
    def before(module,args):
        cat=args[0].clone();cat[:,:t,:h]=attention;cat[:,:t,h:]=mlp;return (cat,)
    handle=block.attn.to_out.register_forward_pre_hook(before)
    try:
        with torch.inference_mode():
            value=block(hidden_states=torch.cat((cut.payload["text"],cut.payload["image"]),dim=1),encoder_hidden_states=None,
                temb_mod=cut.payload["context"]["single_mod"],image_rotary_emb=cut.payload["context"]["rope"],joint_attention_kwargs=None)
    finally:handle.remove()
    return value[:,:t]


__all__=["predict_text_fields","text_write","text_exit","check_text_override"]
