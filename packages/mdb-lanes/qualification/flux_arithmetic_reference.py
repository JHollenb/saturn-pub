"""Small, explicit numerical hypotheses; not descriptions of CUDA internals."""
import numpy as np


def bf16(value, rounding='rne'):
    value=np.asarray(value,np.float32); bits=value.view(np.uint32)
    if rounding=='rne': bits=bits+np.uint32(0x7fff)+((bits>>16)&1)
    elif rounding!='rtz': raise ValueError('unknown rounding')
    return ((bits>>16)<<16).view(np.float32)


def affine_family():
    return [dict(id='wide',kind='wide'),dict(id='fp32',kind='fp32')]+[
        dict(id=f'{kind}.{chunk}.{rounding}',kind=kind,chunk=chunk,rounding=rounding)
        for kind in ('carry','tree') for chunk in (128,256,512,1024) for rounding in ('rne','rtz')]


def affine_wide(x,weight,policy,bias=None):
    """Return the pre-final-cast hypothesis, preserving a finite response law."""
    x,weight=np.asarray(x),np.asarray(weight); kind=policy['kind']
    if x.shape[-1]!=weight.shape[-1]: raise ValueError('affine dimensions differ')
    if kind in ('wide','fp32'):
        dtype=np.float64 if kind=='wide' else np.float32
        out=x.astype(dtype)@weight.astype(dtype).T
    elif kind in ('carry','tree'):
        parts=[]; out=np.zeros((*x.shape[:-1],weight.shape[0]),np.float64)
        for first in range(0,x.shape[-1],policy['chunk']):
            sl=slice(first,first+policy['chunk']); part=x[...,sl].astype(float)@weight[:,sl].astype(float).T
            if kind=='carry': out=bf16(out+part,policy['rounding']).astype(float)
            else: parts.append(bf16(part,policy['rounding']).astype(float))
        if kind=='tree':
            while len(parts)>1:
                parts=[parts[i]+parts[i+1] if i+1<len(parts) else parts[i] for i in range(0,len(parts),2)]
            out=parts[0]
    else: raise ValueError('unknown reduction family')
    return out if bias is None else out+np.asarray(bias,dtype=out.dtype)


def attention_family():
    return [dict(id='wide',kind='dense',precision='float64',probability='wide'),
            dict(id='fp32',kind='dense',precision='float32',probability='wide'),
            dict(id='normalized.bf16',kind='dense',precision='float32',probability='normalized'),
            dict(id='unnormalized.bf16',kind='dense',precision='float32',probability='unnormalized')]+[
        dict(id=f'online.{tile}.{probability}',kind='online',precision='float32',probability=probability,tile=tile)
        for tile in (64,128,256) for probability in ('wide','unnormalized')]


def attention_wide(q,k,v,policy):
    """Q/K/V axes B,H,N,D; sampled queries keep all original keys/values."""
    dtype=np.dtype(policy['precision']); q,k,v=[np.asarray(a,dtype=dtype) for a in (q,k,v)]
    scale=np.asarray(1/np.sqrt(q.shape[-1]),dtype=dtype)
    if policy['kind']=='dense':
        score=(q@k.swapaxes(-1,-2))*scale; score-=score.max(-1,keepdims=True)
        p=np.exp(score); denominator=p.sum(-1,keepdims=True,dtype=dtype)
        if policy['probability']=='normalized': return bf16(p/denominator).astype(dtype)@v
        if policy['probability']=='unnormalized': p=bf16(p).astype(dtype)
        return (p@v)/denominator
    if policy['kind']!='online': raise ValueError('unknown attention family')
    shape=(*q.shape[:-1],1); maximum=np.full(shape,-np.inf,dtype=dtype); denominator=np.zeros(shape,dtype=dtype)
    output=np.zeros((*q.shape[:-1],v.shape[-1]),dtype=dtype)
    for first in range(0,k.shape[-2],policy['tile']):
        sl=slice(first,first+policy['tile']); score=(q@k[...,sl,:].swapaxes(-1,-2))*scale
        new_maximum=np.maximum(maximum,score.max(-1,keepdims=True)); alpha=np.exp(maximum-new_maximum)
        p=np.exp(score-new_maximum); denominator=denominator*alpha+p.sum(-1,keepdims=True,dtype=dtype)
        probabilities=bf16(p).astype(dtype) if policy['probability']=='unnormalized' else p
        output=output*alpha+probabilities@v[...,sl,:]; maximum=new_maximum
    return output/denominator


def metrics(predicted,native):
    from mdb_lanes.response_closure import measure_error
    p,n=np.asarray(predicted,np.float32),np.asarray(native,np.float32)
    def ordered(a):
        bits=a.view(np.uint32)>>16
        return (bits&0x7fff).astype(np.int32)*np.where(bits&0x8000,-1,1)
    ulp=ordered(p)-ordered(n); result=measure_error(p,n)
    return dict(**result,exact_fraction=float(np.mean(p==n)),signed_ulp_mean=float(ulp.mean()),mean_abs_ulp=float(np.abs(ulp).mean()),
        ulp_histogram={str(i):int(np.count_nonzero(ulp==i)) for i in (-2,-1,0,1,2)},other_ulp=int(np.count_nonzero(np.abs(ulp)>2)))
