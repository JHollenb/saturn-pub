"""Observed-operand microscope for the text reader's projection/residual/MLP.

This is post-hoc numerical diagnosis, not a new independent forecast. Reference
and reconstituted native operations retain the full original tensor geometry.
"""
import runpy
from pathlib import Path

H=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
R=runpy.run_path(str(Path(__file__).with_name('flux_reentry_reference.py')))


def configure_eager_determinism():
    """Strict eager determinism via the public interface, without compiler setup."""
    import torch
    torch.set_deterministic_debug_mode('error')
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError('strict eager determinism did not apply')


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
    from torch.nn import functional as f
    width=weights['attn.to_add_out.weight'].shape[0]
    class PinnedLinear(torch.nn.Module):
        def __init__(self,name):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.from_numpy(weights[name+'.weight'].copy()).to(device,torch.bfloat16),requires_grad=False)
            bias=weights.get(name+'.bias')
            self.bias=None if bias is None else torch.nn.Parameter(torch.from_numpy(bias.copy()).to(device,torch.bfloat16),requires_grad=False)
        def forward(self,x):return f.linear(x,self.weight,self.bias)
    class SwiGLU(torch.nn.Module):
        def forward(self,x):
            a,b=x.chunk(2,dim=-1)
            return f.silu(a)*b
    class FeedForward(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear_in=PinnedLinear('ff_context.linear_in')
            self.act_fn=SwiGLU();self.linear_out=PinnedLinear('ff_context.linear_out')
        def forward(self,x):return self.linear_out(self.act_fn(self.linear_in(x)))
    projection=PinnedLinear('attn.to_add_out')
    norm=torch.nn.LayerNorm(width,eps=eps,elementwise_affine=False)
    ff=FeedForward()
    for m in (projection,norm,ff):m.eval().requires_grad_(False)
    if any(getattr(child,'_compiled_call_impl',None) is not None for module in (projection,norm,ff) for child in module.modules()):
        raise ValueError('this primitive recipe is eager-only')
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
