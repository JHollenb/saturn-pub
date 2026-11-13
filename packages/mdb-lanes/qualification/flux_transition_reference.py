"""Framework-free complete FLUX2 single-block transition reference.

All concatenated text/image positions are both queries and keys. Predictions
use only incoming state/modulation/RoPE and actual parameters. No native layer,
current-block activation or teacher-forced intermediate is an accepted input.
Explicit BF16 boundaries define a reference stream, not CUDA kernel emulation.
"""
from __future__ import annotations

import hashlib
import runpy
from functools import lru_cache
from pathlib import Path

MLP_HELPER_SHA256 = "5ee9ba9d67c44e947bb6983a8bdec7b1e9335aaf28f0ffdd8ce71ab559767f8a"
PARAMETER_KEYS = frozenset({"qkv_mlp_weight", "qkv_mlp_bias", "to_out_weight", "to_out_bias",
                            "norm_q_weight", "norm_k_weight"})


@lru_cache(maxsize=1)
def _cached_mlp():
    return runpy.run_path(str(Path(__file__).with_name("flux_mlp_replacement.py")))


def _mlp():
    path = Path(__file__).with_name("flux_mlp_replacement.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != MLP_HELPER_SHA256:
        raise ValueError("immutable full-MLP reference dependency source differs")
    return _cached_mlp()


def _readonly(value):
    import numpy as np

    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _bf16(value):
    import numpy as np

    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite transition reference intermediate")
    bits = value.view(np.uint32)
    result = ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BF16 transition reference overflow")
    return result


def _descriptor(value):
    import numpy as np

    value = np.ascontiguousarray(value)
    bits = (value.view(np.uint32) >> 16).astype("<u2")
    return {"shape": list(value.shape), "dtype": "float32-exact-bf16-widening",
            "fp32_le_sha256": hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest(),
            "bf16_le_sha256": hashlib.sha256(bits.tobytes()).hexdigest()}


def _sources(residual, single_mod, parameters, *, heads, layernorm_eps, qk_eps, rope_cos, rope_sin,
             query_chunk, projection_chunk, neuron_chunk, retain_fields, text_rows, sample_indices):
    import numpy as np

    if (not isinstance(residual, np.ndarray) or residual.ndim != 3 or residual.shape[0] != 1
            or min(residual.shape[1:]) < 1 or type(heads) is not int or heads < 1):
        raise ValueError("complete transition requires one nonempty concatenated register")
    _, length, width = residual.shape
    if width % heads or (width // heads) % 2:
        raise ValueError("transition requires divisible heads and even RoPE head width")
    weight = parameters["qkv_mlp_weight"]
    if (not isinstance(weight, np.ndarray) or weight.ndim != 2 or weight.shape[1] != width
            or weight.shape[0] <= 3*width or (weight.shape[0]-3*width) % 2):
        raise ValueError("complete fused QKV/paired-MLP weight shape differs")
    mlp_width = (weight.shape[0]-3*width)//2
    shapes = {"qkv_mlp_weight": (3*width+2*mlp_width,width), "qkv_mlp_bias": (3*width+2*mlp_width,),
              "to_out_weight": (width,width+mlp_width), "to_out_bias": (width,),
              "norm_q_weight": (width//heads,), "norm_k_weight": (width//heads,)}
    for name, value in parameters.items():
        if value is None and name in {"qkv_mlp_bias", "to_out_bias"}:
            continue
        if not isinstance(value,np.ndarray) or value.shape != shapes[name]:
            raise ValueError("transition native parameter shape differs: " + name)
    if (not isinstance(single_mod,np.ndarray) or single_mod.shape not in {(1,3*width),(1,1,3*width)}
            or any(not isinstance(value,np.ndarray) or value.shape != (length,width//heads) for value in (rope_cos,rope_sin))
            or any(type(eps) not in {int,float} or not np.isfinite(eps) or eps<=0 for eps in (layernorm_eps,qk_eps))
            or any(type(chunk) is not int or not 1<=chunk<=4096 for chunk in (query_chunk,projection_chunk,neuron_chunk))
            or type(retain_fields) is not bool
            or (text_rows is not None and (type(text_rows) is not int or not 0<=text_rows<=length))):
        raise ValueError("transition modulation/RoPE/epsilon/chunk/split profile differs")
    if sample_indices is None:
        sample_indices = np.unique(np.linspace(0,length-1,min(8,length)).astype(np.int64)).tolist()
    else:
        sample_indices = list(sample_indices)
    if (not sample_indices or len(set(sample_indices)) != len(sample_indices)
            or any(type(index) is not int or not 0<=index<length for index in sample_indices)):
        raise ValueError("transition sample coordinates differ")
    bf16 = [residual,single_mod,*[value for value in parameters.values() if value is not None]]
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in [*bf16,rope_cos,rope_sin]):
        raise ValueError("transition operands must be finite FP32 native widenings")
    if any(np.any(np.ascontiguousarray(value).view(np.uint32) & np.uint32(65535)) for value in bf16):
        raise ValueError("transition BF16 operands must preserve exact widened native bits")
    return length,width,mlp_width,sample_indices


def predict_transition(residual, single_mod, *, qkv_mlp_weight, qkv_mlp_bias=None, to_out_weight,
        to_out_bias=None, norm_q_weight, norm_k_weight, heads, layernorm_eps, qk_eps, rope_cos, rope_sin,
        query_chunk=64, projection_chunk=256, neuron_chunk=256, retain_fields=True,
        text_rows=None, sample_indices=None):
    """Calculate the complete all-query native block reference, including I/T.

    Fused weight rows are Q,K,V,u,v; output weight columns are attention,MLP.
    No masks, KV cache, dropout, guidance or FP16 clipping are supported. The
    optional text split never restricts the query/key set. Parameters/state
    are exact FP32 widenings of native BF16; RoPE is actual upstream FP32.
    """
    import numpy as np

    parameters = {"qkv_mlp_weight":qkv_mlp_weight,"qkv_mlp_bias":qkv_mlp_bias,
        "to_out_weight":to_out_weight,"to_out_bias":to_out_bias,
        "norm_q_weight":norm_q_weight,"norm_k_weight":norm_k_weight}
    length,width,mlp_width,indices = _sources(residual,single_mod,parameters,heads=heads,
        layernorm_eps=layernorm_eps,qk_eps=qk_eps,rope_cos=rope_cos,rope_sin=rope_sin,
        query_chunk=query_chunk,projection_chunk=projection_chunk,neuron_chunk=neuron_chunk,
        retain_fields=retain_fields,text_rows=text_rows,sample_indices=sample_indices)
    helper = _mlp()
    neurons = np.unique(np.linspace(0,mlp_width-1,min(32,mlp_width)).astype(np.int64)).tolist()
    # This pinned pointwise MLP is mathematically valid at every token row;
    # its original image-side label does not change the rowwise equations.
    mlp_prediction = helper["predict_mlp_field"](residual,single_mod,qkv_mlp_weight[3*width:],
        None if qkv_mlp_bias is None else qkv_mlp_bias[3*width:],eps=layernorm_eps,
        hidden_width=width,mlp_width=mlp_width,neuron_chunk=neuron_chunk,
        query_indices=indices,neuron_indices=neurons)
    mlp = mlp_prediction["activation"]
    wide = residual.astype(np.float64)
    centered = wide-wide.mean(axis=-1,keepdims=True)
    normalized = _bf16(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+layernorm_eps))
    del wide,centered
    shift,scale,gate = np.split(single_mod.reshape(1,1,3*width),3,axis=-1)
    factor = _bf16(1+scale)
    modulated = _bf16(_bf16(factor*normalized).astype(np.float64)+shift.astype(np.float64))
    mod64 = modulated.astype(np.float64)
    qkv = np.empty((1,length,3*width),np.float32)
    for first in range(0,3*width,projection_chunk):
        last = min(first+projection_chunk,3*width)
        projected = mod64 @ qkv_mlp_weight[first:last].astype(np.float64).T
        if qkv_mlp_bias is not None:
            projected += qkv_mlp_bias[first:last].astype(np.float64)
        qkv[:,:,first:last] = _bf16(projected)
    del mod64
    dim = width//heads
    q,k,v = [value.reshape(1,length,heads,dim) for value in np.split(qkv,3,axis=-1)]

    def rms(value, gamma):
        value = value.astype(np.float64)
        return _bf16(value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+qk_eps)*gamma.astype(np.float64))

    def rotate(value):
        rotated = np.empty_like(value)
        rotated[...,0::2],rotated[...,1::2] = -value[...,1::2],value[...,0::2]
        # Separate FP32 multiplications/addition as qualified apply_rotary_emb.
        return _bf16(value*rope_cos[None,:,None,:]+rotated*rope_sin[None,:,None,:])

    q_norm,k_norm = rms(q,norm_q_weight),rms(k,norm_k_weight)
    q_rope,k_rope = rotate(q_norm),rotate(k_norm)
    keys = k_rope.astype(np.float64).transpose(0,2,3,1)
    values = v.astype(np.float64).transpose(0,2,1,3)
    attention = np.empty((1,length,width),np.float32)
    for first in range(0,length,query_chunk):
        last = min(first+query_chunk,length)
        queries = q_rope[:,first:last].astype(np.float64).transpose(0,2,1,3)
        logits = np.matmul(queries,keys)/np.sqrt(float(dim))
        logits -= logits.max(axis=-1,keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=-1,keepdims=True)
        attended = np.matmul(probability,values).transpose(0,2,1,3).reshape(1,last-first,width)
        attention[:,first:last] = _bf16(attended)
    # All-query A and M are already BF16 values. Their concatenation introduces
    # no rounding. Widen only output-weight row tiles, not the complete weight.
    concat64 = np.empty((1,length,width+mlp_width),np.float64)
    concat64[:,:,:width],concat64[:,:,width:] = attention,mlp
    projection = np.empty((1,length,width),np.float32)
    for first in range(0,width,projection_chunk):
        last = min(first+projection_chunk,width)
        projected = concat64 @ to_out_weight[first:last].astype(np.float64).T
        if to_out_bias is not None:
            projected += to_out_bias[first:last].astype(np.float64)
        projection[:,:,first:last] = _bf16(projected)
    del concat64
    write = _bf16(gate*projection)
    output = _readonly(_bf16(residual+write))
    sampled = {"indices":indices,"neuron_indices":neurons,
        **{name:_readonly(value[:,indices]) for name,value in (("input",residual),("layernorm",normalized),
            ("modulated",modulated),("q_raw",q),("k_raw",k),("v",v),("q_norm",q_norm),("k_norm",k_norm),
            ("q_rope",q_rope),("k_rope",k_rope),("attention",attention),("projection",projection),("write",write),("output",output))},
        "u":mlp_prediction["sampled"]["u"],"v_mlp":mlp_prediction["sampled"]["v"],
        "silu_u":mlp_prediction["sampled"]["silu_u"],"mlp":mlp_prediction["sampled"]["activation"],
        "gate":_readonly(gate)}
    result = {"output":output,"output_descriptor":_descriptor(output),"input_descriptor":_descriptor(residual),
        "sampled":sampled,"hidden_width":width,"mlp_width":mlp_width,"heads":heads,"head_dim":dim,
        "sequence_length":length,"text_rows":text_rows,"layernorm_eps":float(layernorm_eps),"qk_eps":float(qk_eps),
        "query_chunk":query_chunk,"projection_chunk":projection_chunk,"neuron_chunk":neuron_chunk,"retain_fields":retain_fields,
        "dependency_sources":{"flux_mlp_replacement.py":MLP_HELPER_SHA256},
        "arithmetic":"B is FP64/FP32→FP32→RNE BF16 widened FP32. Population LayerNorm FP64→B; B(1+scale); B(product); B(+shift). Full fused Q,K,V and paired u/v FP64 sums+bias→B. Learned Q/K RMSNorm FP64 with actual gamma/eps→B. Interleaved RoPE with separate FP32 multiplies/add→B. ALL queries over complete K/V, unmasked noncausal FP64 QK/softmax/AV→B(A). B(SiLU(Bu)); B(product)=M. Actual whole [A,M] output projection FP64 sums+bias→B(P); B(gate*P)=write; B(incoming residual+write)=output. BF16 profile has no FP16 clipping.",
        "scope":"Complete concatenated text/image transition conditional on incoming state/modulation, all native parameters and upstream RoPE. No native current-target activations. FP64 reference reductions are not exact native CUDA/SDPA emulation. Pinned pointwise MLP helper is applied to all token rows, not just image rows."}
    if retain_fields:
        result.update(attention=_readonly(attention),mlp=mlp,projection=_readonly(projection),write=_readonly(write))
    if text_rows is not None:
        result.update(text=output[:,:text_rows],image=output[:,text_rows:])
    return result


def predict_chain(residual, single_mod, blocks, *, heads, layernorm_eps, qk_eps, rope_cos, rope_sin,
        query_chunk=64, projection_chunk=256, neuron_chunk=256, retain_fields=True,
        text_rows=None, sample_indices=None):
    """Consume each previous complete prediction as the next block's input.

    Ordered block mappings contain the six parameter keys and optional
    ``block_index`` metadata. There is no native intermediate-state argument.
    Common modulation, RoPE, head count and normalization profile are upstream
    inputs shared by the qualified block6→7 path, not regenerated by the chain.
    """
    if not isinstance(blocks,(tuple,list)) or not blocks:
        raise ValueError("transition chain requires ordered parameter blocks")
    current,steps = residual,[]
    for index,block in enumerate(blocks):
        if not isinstance(block,dict) or set(block) - PARAMETER_KEYS - {"block_index"} or not PARAMETER_KEYS.issubset(block):
            raise ValueError("transition chain block parameter inventory differs")
        prediction = predict_transition(current,single_mod,**{key:block[key] for key in PARAMETER_KEYS},heads=heads,
            layernorm_eps=layernorm_eps,qk_eps=qk_eps,rope_cos=rope_cos,rope_sin=rope_sin,
            query_chunk=query_chunk,projection_chunk=projection_chunk,neuron_chunk=neuron_chunk,
            retain_fields=retain_fields,text_rows=text_rows,sample_indices=sample_indices)
        if steps and prediction["input_descriptor"] != steps[-1]["prediction"]["output_descriptor"]:
            raise ValueError("transition chain failed own-predicted-state binding")
        steps.append({"step":index,"block_index":block.get("block_index"),
            "input_source":"caller pre-entry residual" if index==0 else "previous predicted full output",
            "input_descriptor":prediction["input_descriptor"],"prediction":prediction})
        current = prediction["output"]
    result = {"output":current,"output_descriptor":steps[-1]["prediction"]["output_descriptor"],
        "input_descriptor":steps[0]["input_descriptor"],"steps":steps,"chain_length":len(steps),
        "scope":"Every later block consumes the previous predicted complete register. No teacher forcing, native intermediate, correction, model/module call or independently predicted raw-prompt future."}
    if text_rows is not None:
        result.update(text=current[:,:text_rows],image=current[:,text_rows:])
    return result


__all__ = ["MLP_HELPER_SHA256", "predict_chain", "predict_transition"]
