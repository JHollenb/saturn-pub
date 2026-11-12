import runpy
from pathlib import Path
import torch
from saturn_pub import Session,Act
from mdb.native_navigation import NativeCutNavigator

HERE=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(HERE/'qualification/flux_suffix_context.py'))


def test_late_latent_is_selected_by_native_parent_not_a_late_latent_write():
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'));model,adapter,session=tiny['tiny_session']();nav=NativeCutNavigator(session,max_steps=112);nav.until(logical_step=3,phase='single',index=19);a=nav.session.capture(retain=False)
    state=a.payload;state['latent']+=.25;state['text']+=.125;b=Session(adapter,state,parent=a.fingerprint).capture(retain=False)
    assert H['shared_closure']({'60666':a,'60667':b}) and len(H['cases']())==16
    values=[]
    for cut in (a,b):
        n=NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=3)
        n.apply(Act.replace('image',a.payload['image']));n.apply(Act.replace('text',a.payload['text']))
        n.step();image=n.session.capture(retain=False).payload['image'];n.step();velocity=n.session.capture(retain=False).payload['noise_prediction'];n.step();values.append((image,velocity,n.session.capture(retain=False).payload['latent']))
    assert torch.equal(values[0][0],values[1][0]) and torch.equal(values[0][1],values[1][1])
    assert not torch.equal(values[0][2],values[1][2])


def test_parallel_image_mlp_is_text_independent_at_fixed_image_and_context():
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'));model,adapter,session=tiny['tiny_session']();nav=NativeCutNavigator(session,max_steps=112);nav.until(logical_step=3,phase='single',index=19);a=nav.session.capture(retain=False)
    state=a.payload;state['text']+=.125;b=Session(adapter,state,parent=a.fingerprint).capture(retain=False)
    capture=runpy.run_path(str(HERE/'qualification/flux_single_contraction.py'))['capture_candidate']
    x,y=[capture(model.single_transformer_blocks[19],cut,logical_step=3,single_index=19) for cut in (a,b)];t,w=x['text_rows'],x['attention_width']
    assert torch.equal(x['ports']['original_concat'][:,t:,w:],y['ports']['original_concat'][:,t:,w:])
    assert torch.equal(x['ports']['gate'],y['ports']['gate'])
    assert not torch.equal(x['ports']['original_concat'][:,t:,:w],y['ports']['original_concat'][:,t:,:w])
