"""Observed-operand microscope for the text reader's projection/residual/MLP.

This is post-hoc numerical diagnosis, not a new independent forecast. Reference
and reconstituted native operations retain the full original tensor geometry.
"""
import runpy
from pathlib import Path

H=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
R=runpy.run_path(str(Path(__file__).with_name('flux_reentry_reference.py')))


def math_tail(head,text,mod,weights,*,eps,after=None):
    import numpy as np
    b=R['bf16'];m=np.split(mod.reshape(1,1,-1),6,axis=-1);fields={}
    if after is None:
        fields['projection']=R['_linear'](head,weights,'attn.to_add_out',256)
        after=b(text+b(m[2]*fields['projection']))
    fields['after_attention']=after
    fields['norm']=R['_ln'](after,eps)
    fields['mlp_input']=b(b(fields['norm']*b(1+m[4]))+m[3])
    fields['paired']=R['_linear'](fields['mlp_input'],weights,'ff_context.linear_in',256)
    u,v=np.split(fields['paired'],2,axis=-1)
    fields['activation']=b(R['_silu'](u)*v)
    fields['mlp']=R['_linear'](fields['activation'],weights,'ff_context.linear_out',256)
    fields['text']=b(after+b(m[5]*fields['mlp']))
    return fields


def math_local(actual,text,mod,weights,*,eps):
    """Each operator gets its own actual input; no error is carried between ops."""
    import numpy as np
    b=R['bf16'];m=np.split(mod.reshape(1,1,-1),6,axis=-1)
    u,v=np.split(actual['paired'],2,axis=-1)
    return dict(projection=R['_linear'](actual['head'],weights,'attn.to_add_out',256),
        after_attention=b(text+b(m[2]*actual['projection'])),norm=R['_ln'](actual['after_attention'],eps),
        mlp_input=b(b(actual['norm']*b(1+m[4]))+m[3]),
        paired=R['_linear'](actual['mlp_input'],weights,'ff_context.linear_in',256),
        activation=b(R['_silu'](u)*v),mlp=R['_linear'](actual['activation'],weights,'ff_context.linear_out',256),
        text=b(actual['after_attention']+b(m[5]*actual['mlp'])))


def make_modules(weights,*,device,eps):
    import torch
    from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward
    width=weights['attn.to_add_out.weight'].shape[0]
    hidden=weights['ff_context.linear_out.weight'].shape[1]
    bias=weights.get('attn.to_add_out.bias') is not None
    with torch.device('meta'):
        projection=torch.nn.Linear(width,width,bias=bias)
        norm=torch.nn.LayerNorm(width,eps=eps,elementwise_affine=False)
        ff=Flux2FeedForward(dim=width,inner_dim=hidden,bias=weights.get('ff_context.linear_in.bias') is not None)
    def state(prefix):
        return {k.removeprefix(prefix):torch.from_numpy(v.copy()).to(device,torch.bfloat16)
            for k,v in weights.items() if k.startswith(prefix) and v is not None}
    projection.load_state_dict(state('attn.to_add_out.'),assign=True)
    ff.load_state_dict(state('ff_context.'),assign=True)
    for m in (projection,norm,ff):m.eval().requires_grad_(False)
    return dict(projection=projection,norm=norm,ff=ff)


def native_tail(head,text,mod,modules,*,after=None,counts=None):
    """Original Linear/LayerNorm/Flux2FeedForward, with observation-only hooks."""
    import torch
    fields=dict(head=head);m=torch.chunk(mod.reshape(1,1,-1),6,dim=-1)
    handles=[]
    for name,module,kind in [('projection',modules['projection'],'linear'),('norm',modules['norm'],'layernorm'),
            ('paired',modules['ff'].linear_in,'linear'),('activation',modules['ff'].act_fn,'swiglu'),
            ('mlp',modules['ff'].linear_out,'linear')]:
        def hook(module,args,value,name=name,kind=kind):
            fields[name]=value
            if counts is not None:counts[kind]+=1
        handles.append(module.register_forward_hook(hook))
    try:
        if after is None:
            fields['projection']=modules['projection'](head)
            after=text+m[2]*fields['projection']
        fields['after_attention']=after
        normalized=modules['norm'](after)
        fields['mlp_input']=normalized*(1+m[4])+m[3]
        mlp=modules['ff'](fields['mlp_input'])
        fields['text']=after+m[5]*mlp
        return fields
    finally:
        for handle in handles:handle.remove()


def wide(fields):
    return {k:v.detach().float().cpu().numpy().copy() for k,v in fields.items()}


def native_local_affines(actual,modules,counts):
    """Change the reduction policy while keeping each matrix's input fixed."""
    handles=[]
    def called(m,a,v):counts['linear']+=1
    for m in (modules['projection'],modules['ff'].linear_in,modules['ff'].linear_out):
        handles.append(m.register_forward_hook(called))
    try:
        return dict(projection=modules['projection'](actual['head']),
            paired=modules['ff'].linear_in(actual['mlp_input']),mlp=modules['ff'].linear_out(actual['activation']))
    finally:
        for handle in handles:handle.remove()
