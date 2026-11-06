"""Reconstituted component parity against an actual tiny native joint block."""
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from saturn_pub import Session

Q=Path(__file__).parents[1]/'qualification'
C=runpy.run_path(str(Q/'flux_reader_arithmetic.py'))


def test_lazy_archive_comparison_preserves_frozen_measurements(tmp_path):
    audit=runpy.run_path(str(Q/'analyze_flux_reader_arithmetic.py'))
    rng=np.random.default_rng(8);b=C['R']['bf16']
    a=b(rng.normal(size=(1,7,8)).astype(np.float32));p=b(a+.03125)
    ar,pr=[C['H']['array'](tmp_path,x) for x in (a,p)]
    ref=audit['ArrayReference'];loader=C['H']['load_array']
    assert audit['measure'](a,p)==audit['measure'](ref(loader,tmp_path,ar),ref(loader,tmp_path,pr))
    assert np.array_equal(a.astype(np.float64),ref(loader,tmp_path,ar).astype(np.float64))


def test_strict_eager_determinism_does_not_import_compiler():
    code="import runpy,sys,torch; c=runpy.run_path("+repr(str(Q/'flux_reader_arithmetic.py'))+"); c['configure_eager_determinism'](); assert torch.are_deterministic_algorithms_enabled(); assert not torch.is_deterministic_algorithms_warn_only_enabled(); assert 'torch._dynamo' not in sys.modules and 'torch._inductor' not in sys.modules"
    subprocess.run([sys.executable,'-I','-B','-c',code],check=True,capture_output=True,text=True)


def test_actual_native_block_matches_reconstituted_tail_and_local_residuals():
    fixture=runpy.run_path(str(Path(__file__).with_name('test_flux_prompt_factorial.py')))
    model,adapter,sessions=fixture['tiny']();model.requires_grad_(False)
    block=model.transformer_blocks[0];session=sessions['c0-o0'];session.continue_(85)
    parent=session.capture(retain=False);p=parent.payload;captured={}
    def see_head(module,args):captured['head']=args[0].clone()
    def see_after(module,args):captured['after_attention']=args[0].clone()
    def see_mlp(module,args,value):captured['mlp']=value.clone()
    handles=[block.attn.to_add_out.register_forward_pre_hook(see_head),
        block.norm2_context.register_forward_pre_hook(see_after),block.ff_context.register_forward_hook(see_mlp)]
    try:
        target=Session.from_cut(adapter,parent);target.continue_();captured['text']=target.read('text')
    finally:
        for handle in handles:handle.remove()
    weights={k:v.detach().float().numpy().copy() for k,v in block.state_dict().items()
        if k.startswith(('attn.to_add_out.','ff_context.linear_in.','ff_context.linear_out.'))}
    modules=C['make_modules'](weights,device='cpu',eps=1e-6);counts=dict(linear=0,layernorm=0,swiglu=0)
    with torch.inference_mode():
        native=C['native_tail'](captured['head'],p['text'],p['context']['txt_mod'],modules,counts=counts)
        for key in ('after_attention','mlp','text'):assert torch.equal(native[key],captured[key])
        forced=C['native_tail'](captured['head'],p['text'],p['context']['txt_mod'],modules,
            after=captured['after_attention'],counts=counts)
        assert torch.equal(forced['text'],captured['text'])
    assert counts==dict(linear=5,layernorm=2,swiglu=2)
    with torch.inference_mode():
        repeated=C['native_local_affines'](native,modules,counts)
        for field in repeated:assert torch.equal(repeated[field],native[field])
    assert counts['linear']==8
    observed=C['wide'](native);text=p['text'].float().numpy();mod=p['context']['txt_mod'].float().numpy()
    local=C['math_local'](observed,text,mod,weights,eps=1e-6)
    for key in ('after_attention','mlp_input','text'):assert np.array_equal(local[key],observed[key])
    math=C['math_tail'](observed['head'],text,mod,weights,eps=1e-6)
    assert set(math)==set(observed)-{'head'}
    for module in modules.values():
        for child in module.modules():assert not child._forward_hooks
