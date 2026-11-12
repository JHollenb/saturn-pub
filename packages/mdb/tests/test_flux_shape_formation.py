import runpy
from pathlib import Path
from types import SimpleNamespace
import torch
from saturn_pub import Session
from saturn_pub.store import LocalStore
from mdb.native_navigation import NativeCutNavigator
HERE=Path(__file__).resolve().parents[1]
H=runpy.run_path(str(HERE/'qualification/flux_shape_formation.py'))


def test_finite_clamp_rescue_census_and_footprints():
    assert H['census']()==dict(loads=1,encodes=0,noise_draws=0,native_transitions=1068,acts=47,receipts=1115,images=14)
    sources={role:{i:SimpleNamespace(payload={'image':(role,i,'I'),'text':(role,i,'T')}) for i in range(8)} for role in ('base','shape')}
    counts={}
    for r in H['recipes']():
        slots=[H['edits'](r,i,sources) for i in range(r['entry'],8)]
        counts[r['id']]=sum(len(v) for v in slots)
        if r['mode'].startswith('Tclamp'):
            assert all(set(v)=={'text'} for v in slots[:-1]) and not slots[-1]
    assert sum(counts.values())+14==47
    recipe=next(v for v in H['recipes']() if v['id']=='Tclamp0to5.rescue6')
    assert H['edits'](recipe,5,sources)=={'text':('base',5,'T')}
    assert H['edits'](recipe,6,sources)=={'text':('shape',6,'T')}
    assert H['census'](fresh=True)==dict(loads=1,encodes=0,noise_draws=1,native_transitions=1108,acts=49,receipts=1157,images=14)


def test_typed_native_clamp_rescue_leaves_unwritten_stream_exact(tmp_path):
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'))
    _model,adapter,session=tiny['tiny_session']();n=NativeCutNavigator(session,max_steps=112)
    n.until(logical_step=1,phase='single',index=0);root=n.capture()
    high=NativeCutNavigator(Session.from_cut(adapter,root),max_steps=7)
    from saturn_pub import Act
    high.apply(Act.replace('text',root.payload['text']+.2,name='source'))
    sources={'base':{},'shape':{}}
    for role,nav in [('base',n),('shape',high)]:
        for i in range(8):
            sources[role][i]=nav.capture()
            if i<7:nav.step()
    restore=runpy.run_path(str(HERE/'qualification/flux_scene_dependencies.py'))['restore_slots'];store=LocalStore(tmp_path/'store')
    for recipe in H['recipes']():
        cut=sources[recipe['role']][recipe['entry']];nav=NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=7)
        for i in range(recipe['entry'],8):
            selected=H['edits'](recipe,i,sources)
            if selected:
                before=nav.capture();restore(nav,selected,label=recipe['id']+str(i),store=store);after=nav.capture()
                for slot in ('image','text'):
                    assert torch.equal(after.payload[slot],selected[slot] if slot in selected else before.payload[slot])
            if i<7:nav.step()
        assert nav.point['index']==7 and nav.point['logical_step']==1


def test_fresh_session_from_saved_schedule_has_full_step0_prefix():
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'))
    _model,adapter,session=tiny['tiny_session']();n=NativeCutNavigator(session,max_steps=112)
    n.until(logical_step=1,phase='single',index=0);p=n.capture().payload
    latent=torch.randn(p['latent'].shape,dtype=p['latent'].dtype,generator=torch.Generator().manual_seed(60667))
    fresh=adapter.session(latent=latent,conditioning=p['conditioning'],img_ids=p['img_ids'],txt_ids=p['txt_ids'],timesteps=p['timesteps'],sigmas=p['sigmas'])
    nav=NativeCutNavigator(fresh,max_steps=28)
    assert nav.point['logical_step']==0 and nav.point['phase']=='project'
    for _ in range(28):nav.step()
    assert nav.point['logical_step']==1 and nav.point['phase']=='project'
    assert not torch.equal(nav.capture().payload['latent'],p['latent'])
