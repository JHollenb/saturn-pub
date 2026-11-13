"""Own-state predictions with a frozen absolute affine arithmetic surrogate.

No Torch, learned module, native attention or target-state observations execute
here. Anchors are explicit calibration, not reconstructed numerical internals.
"""
import time
import runpy
from pathlib import Path
import numpy as np

AR=runpy.run_path(str(Path(__file__).with_name("flux_arithmetic_reference.py")))


def absolute_affine(value,base_output,weight,policy,bf16,*,bias=None,mutable_rows=None):
    """Recompute a declared support; all other returns are unchanged cache hits."""
    result=base_output.copy() if mutable_rows is not None else np.empty((*value.shape[:-1],weight.shape[0]),np.float32)
    operand=value[:,mutable_rows] if mutable_rows is not None else value
    for first in range(0,weight.shape[0],256):
        sl=slice(first,first+256); predicted=bf16(AR['affine_wide'](operand,weight[sl],policy,None if bias is None else bias[sl]))
        if mutable_rows is None: result[...,sl]=predicted
        else: result[:,mutable_rows,sl]=predicted
    return result
from mdb_lanes.attention_partition import prepare_attention_partition


def affine_delta(value,base_input,base_output,weight,bf16,*,chunk=256):
    """B(y0 + (x1-x0) W^T); never quantize the input difference."""
    if value.shape!=base_input.shape or base_output.shape[:-1]!=value.shape[:-1] or weight.shape!=(base_output.shape[-1],value.shape[-1]): raise ValueError('affine response geometry differs')
    x=value.reshape(-1,value.shape[-1]); x0=base_input.reshape(x.shape)
    result=base_output.copy().reshape(-1,base_output.shape[-1]); rows=np.flatnonzero(np.any(x!=x0,axis=-1))
    if len(rows):
        delta=x[rows].astype(np.float64)-x0[rows].astype(np.float64)
        for first in range(0,weight.shape[0],chunk):
            sl=slice(first,first+chunk)
            result[rows,sl]=bf16(base_output.reshape(result.shape)[rows,sl].astype(np.float64)+delta@weight[sl].astype(np.float64).T)
    return result.reshape(base_output.shape),int(len(rows))


def normalized_delta(value,base_input,base_output,law,bf16):
    """Only changed normalization vectors use their finite wide-law response."""
    if value.shape!=base_input.shape or value.shape!=base_output.shape: raise ValueError('normalization response geometry differs')
    x=value.reshape(-1,value.shape[-1]); x0=base_input.reshape(x.shape); y=base_output.copy().reshape(x.shape)
    rows=np.flatnonzero(np.any(x!=x0,axis=-1))
    if len(rows): y[rows]=bf16(y[rows].astype(float)+law(x[rows].astype(float))-law(x0[rows].astype(float)))
    return y.reshape(value.shape)


def prepare(image,text,context,weights,stages,reference,*,mutable_rows=(4,11),eps=1e-6,absolute_ports=(),affine_policy=None):
    started=time.monotonic(); B=reference['bf16']; width=image.shape[-1]
    for role,value in [('image',image),('text',text)]:
        expected=B(value+B(np.split(context['img_mod' if role=='image' else 'txt_mod'].reshape(1,1,6*width),6,axis=-1)[2]*stages[role+'.attention']))
        if not np.array_equal(expected,stages[role+'.after_attention']): raise ValueError('native anchor residual contract differs')
    q,k,v=[np.concatenate([stages[role+'.'+label] for role in ('text','image')],axis=1).transpose(0,2,1,3) for label in ('q_positioned','k_positioned','v_projection')]
    partition=prepare_attention_partition(q,k,v,list(mutable_rows))
    ix=list(mutable_rows)
    A0,_=partition.evaluate(k[:,:,ix],v[:,:,ix],query_rows=q[:,:,ix],query_indices=ix)
    return dict(image=image,text=text,context=context,weights=weights,stages=stages,reference=reference,mutable_rows=ix,eps=eps,partition=partition,math_attention_anchor=A0,seconds=time.monotonic()-started,absolute_ports=tuple(absolute_ports),affine_policy=affine_policy)


def predict(cache,text,*,observer=None):
    def see(key,value):
        if observer is not None: observer(key,value)
    started=time.monotonic(); s=cache['stages']; w=cache['weights']; r=cache['reference']; B=r['bf16']; I=cache['image']; T0=cache['text']; context=cache['context']; ix=cache['mutable_rows']; width=I.shape[-1]; heads,dim=s['image.q_projection'].shape[-2:]
    other=[i for i in range(text.shape[1]) if i not in ix]
    if text.shape!=T0.shape or not np.array_equal(text[:,other],T0[:,other]): raise ValueError('undeclared text-row edit')
    work={}; mods={role:np.split(context[key].reshape(1,1,6*width),6,axis=-1) for role,key in [('image','img_mod'),('text','txt_mod')]}
    def ln_law(x):
        centered=x-x.mean(axis=-1,keepdims=True)
        return centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+cache['eps'])
    def ln(value,base_input,base_output): return normalized_delta(value,base_input,base_output,ln_law,B)
    def linear(value,base_input,base_output,name):
        if name in cache['absolute_ports']:
            support=ix if name in ('attn.add_q_proj','attn.add_k_proj','attn.add_v_proj') else None
            result=absolute_affine(value,base_output,w[name+'.weight'],cache['affine_policy'],B,bias=w.get(name+'.bias'),mutable_rows=support)
            rows=len(support) if support is not None else int(np.prod(value.shape[:-1])); method='absolute-frozen-numerical-law-with-declared-support'
        else:
            result,rows=affine_delta(value,base_input,base_output,w[name+'.weight'],B); method='native-anchor-wide-delta'
        work[name]=dict(rows_calculated=rows,input_width=value.shape[-1],output_width=base_output.shape[-1],method=method); return result
    text_norm=ln(text,T0,s['text.norm1']); mod=mods['text']
    text_mod=B(B(text_norm*B(1+mod[1]))+mod[0]); text_qkv=[]; see('text.modulated',text_mod)
    for label,name in [('q','add_q_proj'),('k','add_k_proj'),('v','add_v_proj')]:
        value=linear(text_mod,s['text.modulated'],s['text.'+label+'_projection'].reshape(text.shape),'attn.'+name).reshape(1,text.shape[1],heads,dim)
        see('text.'+label+'_projection',value)
        if label!='v':
            gamma=w['attn.norm_added_'+label+'.weight'].astype(float)
            def rms_law(x): return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+cache['eps'])*gamma
            value=normalized_delta(value,s['text.'+label+'_projection'],s['text.'+label+'_normalized'],rms_law,B)
            see('text.'+label+'_normalized',value)
            cos,sin=[a[:text.shape[1]][None,:,None,:] for a in context['rope']]; rotated=np.empty_like(value)
            rotated[...,0::2],rotated[...,1::2]=-value[...,1::2],value[...,0::2]
            value=B(np.add(np.multiply(value,cos,dtype=np.float32),np.multiply(rotated,sin,dtype=np.float32),dtype=np.float32))
        if label!='v': see('text.'+label+'_positioned',value)
        text_qkv.append(value.transpose(0,2,1,3))
    q,k,v=text_qkv
    A,geometry=cache['partition'].evaluate(k[:,:,ix],v[:,:,ix],query_rows=q[:,:,ix],query_indices=ix)
    native_A=s['attention.raw']; attention=B(native_A.astype(float)+(A-cache['math_attention_anchor']).transpose(0,2,1,3).reshape(native_A.shape))
    see('attention.raw',attention)
    result={}
    for role,residual,sl,out,ff in [('image',I,slice(text.shape[1],None),'attn.to_out.0','ff'),('text',text,slice(0,text.shape[1]),'attn.to_add_out','ff_context')]:
        mod=mods[role]; raw=attention[:,sl]; raw0=native_A[:,sl]
        a=linear(raw,raw0,s[role+'.attention'],out); updated=B(residual+B(mod[2]*a))
        see(role+'.after_attention',updated)
        normalized=ln(updated,s[role+'.after_attention'],s[role+'.norm2']); ff_input=B(B(normalized*B(1+mod[4]))+mod[3])
        paired=linear(ff_input,s[role+'.mlp_input'],s[role+'.mlp_paired'],ff+'.linear_in'); u,value=np.split(paired,2,axis=-1)
        activation=B(r['_silu'](u)*value); mlp=linear(activation,s[role+'.mlp_activation'],s[role+'.mlp'],ff+'.linear_out')
        result[role]=dict(attention=a,mlp_input=ff_input,mlp=mlp,exit=B(updated+B(mod[5]*mlp)))
    return dict(fields=result,seconds=time.monotonic()-started,affine_work=work,attention_work=geometry,
        scope='Frozen absolute affine numerical schedule on declared ports; historical native normalization/attention anchors and unchanged image prefix remain supplied. All candidate operands are self-calculated. Zero-edit native equality is an observation, not an absolute-surrogate mechanics requirement.')
