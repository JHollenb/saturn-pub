import runpy
from pathlib import Path
import numpy as np


def test_complete_changed_row_update_matches_separate_dense_joint_reference():
    here=Path(__file__).resolve().parents[1]/'qualification'
    r=runpy.run_path(str(here/'flux_reentry_reference.py'));p=runpy.run_path(str(here/'flux_response_reference.py'))
    rng=np.random.default_rng(2026);B=r['bf16'];width=8;heads=2;hidden=12
    image=B(rng.normal(size=(1,3,width)));text=B(rng.normal(size=(1,5,width)))
    context=dict(img_mod=B(rng.normal(size=(1,6*width))),txt_mod=B(rng.normal(size=(1,6*width))),
        rope=(np.ones((8,4),np.float32),np.zeros((8,4),np.float32)))
    weights={}
    for name in r['JOINT_LINEAR_NAMES']:
        shape=(2*hidden,width)if name.endswith('linear_in')else(width,hidden)if name.endswith('linear_out')else(width,width)
        weights[name+'.weight']=B(rng.normal(size=shape)*.1)
    for name in r['JOINT_NORM_NAMES']:weights[name]=B(np.ones((4,)))
    cache=p['prepare'](image,text,context,weights,r,heads=heads,mutable_rows=(1,3))
    edited=text.copy();edited[:,1]=B(edited[:,1]+.3);edited[:,3]=B(edited[:,3]-.7)
    delta=p['predict'](cache,edited);dense=r['predict_joint'](image,edited,context,weights,heads=heads)
    for role in('image','text'):np.testing.assert_array_equal(delta['fields'][role]['exit'],dense[role])
    assert delta['dense_comparison']['bf16_exact']


def test_finite_campaign_counts_input_acts_and_shared_clock_futures():
    here=Path(__file__).resolve().parents[1]/'qualification'
    campaign=runpy.run_path(str(here/'flux_response_campaign.py'))
    cases=campaign['cases']()
    assert len(cases)==5 and sum(c['role']=='historical control'for c in cases)==1
    assert campaign['expected_counts'](5)==dict(loads=1,encodes=0,noise_draws=0,native_transitions=535,acts=20,receipts=45,images=10)


def test_resident_entrypoint_reaches_worker_setup_before_cli(monkeypatch):
    import os,subprocess,sys
    here=Path(__file__).resolve().parents[1]/'qualification'
    env=dict(os.environ,SATURN_RESIDENT_WORKER='1');env.pop('SATURN_RUN_CONFIG',None);env.pop('SATURN_DEBUG_REQUEST',None)
    result=subprocess.run([sys.executable,'-I',str(here/'flux_response_campaign.py')],cwd=here,env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'SATURN_RUN_CONFIG'in result.stderr
    assert 'arguments are required'not in result.stderr
