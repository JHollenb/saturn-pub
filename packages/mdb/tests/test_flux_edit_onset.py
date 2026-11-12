import runpy
from pathlib import Path
import pytest
import torch
from saturn_pub import Session
from saturn_pub.store import LocalStore
from mdb.native_navigation import NativeCutNavigator

HERE=Path(__file__).resolve().parents[1]
HELPER=runpy.run_path(str(HERE/'qualification/flux_edit_onset.py'))


def test_bounded_physical_census_refuses_geometry_drift():
    assert HELPER['census']()==dict(loads=1,encodes=0,noise_draws=0,native_transitions=2128,acts=33,receipts=2161,images=25)
    with pytest.raises(ValueError):HELPER['census']((1,2,3))


def test_repeated_canonical_receipts_are_physical_emissions_not_collisions():
    audit=runpy.run_path(str(HERE/'qualification/analyze_flux_edit_onset.py'))
    receipt={'fingerprint':'same','operation':'continue','parent':'A','result':'B'}
    emitted=[receipt,dict(receipt)]
    indexed=audit['index_receipts'](emitted)
    assert len(emitted)==2 and len(indexed)==1
    with pytest.raises(ValueError,match='identity collision'):
        audit['index_receipts']([receipt,{**receipt,'result':'different'}])


def test_native_project_read_repeats_and_text_shams_stack_exactly(tmp_path):
    tiny=runpy.run_path(str(HERE/'tests/test_flux_single_contexts.py'))
    _model,adapter,session=tiny['tiny_session']();root=session.capture()
    native=NativeCutNavigator(session,max_steps=112)
    fields=[]
    HELPER['run_suffix'](native,on_project=lambda n,k,r:fields.append(n.session.read('text').clone()))
    assert len(fields)==4 and all(torch.equal(v,fields[0]) for v in fields)
    restore=runpy.run_path(str(HERE/'qualification/flux_scene_dependencies.py'))['restore_slots']
    nav=NativeCutNavigator(Session.from_cut(adapter,root),max_steps=112)
    calls=[];boundaries=[];store=LocalStore(tmp_path/'store')
    def replace(n,k,r):
        assert r.to_dict()['operation']=='continue'
        restore(n,{'text':fields[0]},label='stacked-sham',store=store)
        calls.append(k)
    HELPER['run_suffix'](nav,on_project=replace,on_boundary=lambda n:boundaries.append(n.point['logical_step']))
    assert calls==[0,1,2,3] and boundaries==[0,1,2,3,4]
    assert torch.equal(nav.session.read('latent'),native.session.read('latent'))
