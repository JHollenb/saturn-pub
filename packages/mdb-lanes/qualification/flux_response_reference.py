"""A reusable changed-row joint-block predictor over qualified numeric primitives."""
import time
import numpy as np
from mdb_lanes.attention_partition import prepare_attention_partition,dense_attention


def qk_rms(value,weight,reference,*,eps=1e-6,rounding='legacy'):
    if rounding not in {'legacy','staged'}:raise ValueError('unknown QK normalization contract')
    wide=value.astype(np.float64)
    normalized=wide/np.sqrt(np.mean(wide*wide,axis=-1,keepdims=True)+eps)
    if rounding=='staged':normalized=reference['bf16'](normalized)
    return reference['bf16'](normalized*weight.astype(np.float64))


def prepare(image,text,context,weights,reference,*,heads,mutable_rows=(4,11),eps=1e-6,qk_rounding='legacy',observer=None):
    started=time.monotonic();r=reference;r['_weights'](weights,r['JOINT_LINEAR_NAMES'],r['JOINT_NORM_NAMES'])
    width=image.shape[-1];dim=width//heads
    mods=[np.split(context[k].reshape(1,1,6*width),6,axis=-1)for k in('img_mod','txt_mod')]
    def qkv(value,role,indices):
        mi=mods[0 if role=='image'else 1]
        normalized=r['_modulate'](value,mi[0],mi[1],eps)
        if observer:
            observer(role+'.norm1',r['_ln'](value,eps));observer(role+'.modulated',normalized)
        names=('to_q','to_k','to_v')if role=='image'else('add_q_proj','add_k_proj','add_v_proj')
        norms=('norm_q','norm_k')if role=='image'else('norm_added_q','norm_added_k')
        q,k,v=[r['_linear'](normalized,weights,'attn.'+n,256).reshape(1,value.shape[1],heads,dim)for n in names]
        if observer:
            for label,x in zip(('q','k','v'),(q,k,v)):observer(role+'.'+label+'_projection',x)
        def rms(x,name):return qk_rms(x,weights['attn.'+name+'.weight'],r,eps=eps,rounding=qk_rounding)
        q,k=rms(q,norms[0]),rms(k,norms[1]);cos,sin=[x[indices]for x in context['rope']]
        if observer:observer(role+'.q_normalized',q);observer(role+'.k_normalized',k)
        def rotate(x):
            rotated=np.empty_like(x);rotated[...,0::2],rotated[...,1::2]=-x[...,1::2],x[...,0::2]
            return r['bf16'](np.add(np.multiply(x,cos[None,:,None,:],dtype=np.float32),np.multiply(rotated,sin[None,:,None,:],dtype=np.float32),dtype=np.float32))
        q,k=rotate(q),rotate(k)
        if observer:observer(role+'.q_positioned',q);observer(role+'.k_positioned',k)
        return q,k,v
    tqkv=qkv(text,'text',list(range(text.shape[1])))
    iqkv=qkv(image,'image',list(range(text.shape[1],text.shape[1]+image.shape[1])))
    all_qkv=[np.concatenate((tqkv[j],iqkv[j]),axis=1).transpose(0,2,1,3)for j in range(3)]
    partition=prepare_attention_partition(*all_qkv,list(mutable_rows))
    return dict(image=image,text=text,context=context,weights=weights,reference=r,heads=heads,mods=mods,qkv=qkv,
        partition=partition,mutable_rows=tuple(mutable_rows),eps=eps,seconds=time.monotonic()-started,observer=observer,qk_rounding=qk_rounding)


def predict(cache,text,*,compare_dense=True):
    """Only declared T rows may differ; receiving I and context are cache-bound."""
    started=time.monotonic();r=cache['reference'];B=r['bf16'];w=cache['weights'];ix=list(cache['mutable_rows'])
    other=[i for i in range(text.shape[1])if i not in ix]
    if text.shape!=cache['text'].shape or not np.array_equal(text[:,other],cache['text'][:,other]):raise ValueError('undeclared working text change')
    q,k,v=cache['qkv'](text[:,ix],'text',ix)
    q,k,v=[x.transpose(0,2,1,3)for x in(q,k,v)]
    attention,work=cache['partition'].evaluate(k,v,query_rows=q,query_indices=ix)
    dense_report=None;dense_seconds=0.
    if compare_dense:
        full_q=cache['partition'].q.copy();full_k=cache['partition'].k.copy();full_v=cache['partition'].v.copy()
        full_q[:,:,ix]=q;full_k[:,:,ix]=k;full_v[:,:,ix]=v
        dense_started=time.monotonic();dense=dense_attention(full_q,full_k,full_v);dense_seconds=time.monotonic()-dense_started
        dense_report=dict(max_abs=float(np.max(np.abs(dense-attention))),bf16_exact=bool(np.array_equal(B(dense),B(attention))),
            scope='Independent dense attention over the same independently calculated Q/K/V; no native target operands')
    attention=B(attention.transpose(0,2,1,3).reshape(1,-1,cache['image'].shape[-1]))
    if cache['observer']:cache['observer']('attention.raw',attention)
    result={}
    for role,residual,raw,mod,out,ff in[
        ('image',cache['image'],attention[:,text.shape[1]:],cache['mods'][0],'attn.to_out.0','ff'),
        ('text',text,attention[:,:text.shape[1]],cache['mods'][1],'attn.to_add_out','ff_context')]:
        a=r['_linear'](raw,w,out,256);updated=B(residual+B(mod[2]*a))
        ff_input=r['_modulate'](updated,mod[3],mod[4],cache['eps'])
        paired=r['_linear'](ff_input,w,ff+'.linear_in',256);u,v_mlp=np.split(paired,2,axis=-1)
        silu=r['_silu'](u);activation=B(silu*v_mlp);mlp=r['_linear'](activation,w,ff+'.linear_out',256)
        result[role]=dict(attention=a,mlp_input=ff_input,mlp=mlp,exit=B(updated+B(mod[5]*mlp)))
        if cache['observer']:
            cache['observer'](role+'.after_attention',updated)
            for label,value in [('norm2',r['_ln'](updated,cache['eps'])),('mlp_paired',paired),('mlp_silu',silu),('mlp_activation',activation)]:cache['observer'](role+'.'+label,value)
            for label,value in result[role].items():cache['observer'](role+'.'+label,value)
    return dict(fields=result,work=work,dense_comparison=dense_report,dense_validation_seconds=dense_seconds,
        seconds=time.monotonic()-started,scope='Independent weights/input/context; partitioned attention then receiving-state MLP. No new native module inputs/returns used.')
