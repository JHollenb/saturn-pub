import runpy
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(HERE/'qualification/flux_receiver_cross.py'))


def test_fixed_correction_signed_doses_and_untouched_ports():
    native=torch.arange(32,dtype=torch.float32).reshape(1,4,8).to(torch.bfloat16)
    target=(native[:,1:,:2].float()+.75).to(torch.bfloat16)
    delta=target.float()-native[:,1:,:2].float()
    for dose in H['DOSES']:
        cat=H['compose'](native,delta,text_rows=1,width=2,dose=dose)
        assert torch.equal(cat[:,:1],native[:,:1]) and torch.equal(cat[:,1:,2:],native[:,1:,2:])
        assert torch.equal(cat[:,1:,:2],(native[:,1:,:2].float()+dose*delta).to(native.dtype))
        if dose==1:assert torch.equal(cat[:,1:,:2],target)
    rows=H['cases'](('60666','60667'))
    assert len(rows)==len({r['id'] for r in rows})==20
    assert sum(r['predicted'] for r in rows)==2
    assert H['census'](('60666','60667'))['native_transitions']==60


def test_fixed_correction_matches_native_block_substitution():
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'))
    from mdb.native_navigation import NativeCutNavigator
    model,adapter,session=tiny['tiny_session']();nav=NativeCutNavigator(session,max_steps=112)
    nav.until(logical_step=3,phase='single',index=0);cut=nav.session.capture(retain=False)
    capture=runpy.run_path(str(HERE/'qualification/flux_single_contraction.py'))['capture_candidate']
    p=capture(model.single_transformer_blocks[0],cut,logical_step=3,single_index=0)
    cat=p['ports']['original_concat'];t,w=p['text_rows'],p['attention_width']
    delta=torch.full_like(cat[:,t:,:w],.125,dtype=torch.float32)
    for dose in (-1.,1.):
        mixed=H['compose'](cat,delta,text_rows=t,width=w,dose=dose)
        with torch.inference_mode():P=model.single_transformer_blocks[0].attn.to_out(mixed)
        actual=capture(model.single_transformer_blocks[0],cut,logical_step=3,single_index=0,selection='A',source={'parent':'fixed-correction-test','concat':mixed,'text_rows':t,'attention_width':w})
        assert torch.equal(actual['image'],cut.payload['image']+p['ports']['gate']*P[:,t:])
        assert torch.equal(actual['text'],p['text'])
