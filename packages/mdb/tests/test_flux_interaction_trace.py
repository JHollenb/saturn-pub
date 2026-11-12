import runpy
from pathlib import Path
import torch
from saturn_pub import Session
from mdb.native_navigation import NativeCutNavigator
HERE=Path(__file__).resolve().parents[1];H=runpy.run_path(str(HERE/'qualification/flux_interaction_trace.py'))


def test_interaction_witnesses_and_live_state_rules():
    v={k:torch.full((1,4,4),x,dtype=torch.bfloat16) for k,x in [('00',1),('10',2),('01',3),('11',7)]}
    row,c=H['measure'](v,torch.zeros((1,4,4),dtype=torch.float64))
    assert torch.equal(c,torch.full_like(c,3)) and row['l2']==12 and row['write_interaction_l2']==12
    assert row['source_contrast_projection']==.5 and row['interaction_witnesses']==[3]*16
    assert H['live_slots']({'phase':'project'})==['latent']
    assert H['live_slots']({'phase':'scheduler'})==['latent','noise_prediction']
    assert H['census']()['native_transitions']==448
    assert not H['candidate']({'logical_step':3,'phase':'single','index':19})


def test_interleaved_native_four_corners_match_serial_and_retain_only_requested_history():
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'));model,adapter,session=tiny['tiny_session']();n=NativeCutNavigator(session,max_steps=112);n.until(logical_step=2,phase='project');root=n.session.capture(retain=False)
    roots={}
    for b,offset in zip(H['BITS'],(0,.1,.2,.4)):
        state=root.payload;state['latent']+=offset;roots[b]=Session(adapter,state,parent=root.fingerprint).capture(retain=False)
    serial={}
    for b,c in roots.items():
        nav=NativeCutNavigator(Session.from_cut(adapter,c),max_steps=7)
        for _ in range(7):nav.step()
        serial[b]=nav.session.capture(retain=False)
    nav={b:NativeCutNavigator(Session.from_cut(adapter,c),max_steps=7) for b,c in roots.items()};capture=runpy.run_path(str(HERE/'qualification/flux_single_contraction.py'))['capture_candidate']
    parents={}
    for i in range(7):
        for b,n in nav.items():
            if i==6:parents[b]=n.session.capture(retain=False)
            n.step()
    for b,n in nav.items():
        actual=n.session.capture(retain=False);assert actual.fingerprint==serial[b].fingerprint and not n.session.history
        ports=capture(model.single_transformer_blocks[0],parents[b],logical_step=2,single_index=0)
        assert torch.equal(ports['image'],actual.payload['image']) and torch.equal(ports['text'],actual.payload['text'])
