import runpy
from pathlib import Path
import numpy as np
import torch
from saturn_pub import Act,Session
from saturn_pub.store import LocalStore
from mdb.native_navigation import NativeCutNavigator

HERE=Path(__file__).resolve().parents[1]
HELPER=runpy.run_path(str(HERE/'qualification/flux_step1_writers.py'))
TINY=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'))


def test_finite_step1_panel_census():
    assert HELPER['census']()==dict(loads=1,encodes=0,noise_draws=0,native_transitions=1646,acts=44,receipts=1690,images=22,candidate_blocks=5)


def test_full_predicted_joint_exits_match_actual_frozen_write_substitutions():
    model,adapter,session=TINY['tiny_session']();n=NativeCutNavigator(session,max_steps=112)
    n.until(logical_step=1,phase='joint',index=0);base=n.capture()
    source=NativeCutNavigator(Session.from_cut(adapter,base),max_steps=1)
    source.apply(Act.replace('text',base.payload['text']*1.125,name='sourceT'));shape=source.capture()
    candidate=runpy.run_path(str(HERE/'qualification/flux_joint_writes.py'))['joint_candidate'];block=model.transformer_blocks[0]
    captures={k:candidate(block,c) for k,c in [('base',base),('shape',shape)]}
    ports={k:{'residual':c.payload['image'].float().numpy(),**{p:v.float().numpy() for p,v in captures[k]['ports'].items()}} for k,c in [('base',base),('shape',shape)]}
    predicted=HELPER['predict_exits'](ports['base'],ports['shape'])
    np.testing.assert_array_equal(predicted['00'],captures['base']['image'].float().numpy())
    np.testing.assert_array_equal(predicted['11'],captures['shape']['image'].float().numpy())
    for bits in ('10','01','11'):
        actual=candidate(block,base,attention=captures['shape' if bits[0]=='1' else 'base']['ports']['attention'],mlp=captures['shape' if bits[1]=='1' else 'base']['ports']['mlp'])
        np.testing.assert_array_equal(predicted[bits],actual['image'].float().numpy())
        assert torch.equal(actual['text'],captures['base']['text'])


def test_pulse_reset_happens_before_step2_project_and_matches_manual_native(tmp_path):
    _model,adapter,session=TINY['tiny_session']();n=NativeCutNavigator(session,max_steps=112)
    n.until(logical_step=1,phase='project');root=n.capture();base=root.payload['conditioning'];changed=base+.25
    restore=runpy.run_path(str(HERE/'qualification/flux_scene_dependencies.py'))['restore_slots'];store=LocalStore(tmp_path/'store')
    manual=NativeCutNavigator(Session.from_cut(adapter,root),max_steps=84)
    restore(manual,{'conditioning':changed},label='source',store=store)
    for _ in range(28):manual.step()
    restore(manual,{'conditioning':base},label='reset',store=store)
    for _ in range(56):manual.step()
    auto=NativeCutNavigator(Session.from_cut(adapter,root),max_steps=84);restore(auto,{'conditioning':changed},label='source',store=store)
    at=[]
    def apply(nav,value):
        at.append(nav.point.copy());return restore(nav,{'conditioning':value},label='reset',store=store)
    actions=HELPER['run_reset_suffix'](auto,reset_conditioning=base,apply=apply)
    assert len(actions)==1 and at[0]['logical_step']==2 and at[0]['phase']=='project'
    assert torch.equal(auto.session.read('latent'),manual.session.read('latent'))
