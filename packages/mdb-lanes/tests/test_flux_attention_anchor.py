"""Mechanics of attention-boundary calibration, without a model."""
import runpy
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]/'qualification'
WORKER=runpy.run_path(str(HERE/'flux_attention_anchor.py'))
B=runpy.run_path(str(HERE/'flux_arithmetic_reference.py'))['bf16']


def test_absolute_and_anchored_baselines_are_distinct():
    x0=np.array([[1.003]],np.float64);x1=np.array([[1.005]],np.float64);n0=B(x0)
    zero=WORKER['candidates'](x0,x0,n0,B);new=WORKER['candidates'](x1,x0,n0,B)
    np.testing.assert_array_equal(zero['anchored'],n0)
    np.testing.assert_array_equal(new['absolute'],[[1.0078125]])
    np.testing.assert_array_equal(new['anchored'],[[1.]])


def test_native_calibration_is_not_corrected_baseline():
    x0=np.array([[1.]],float);n0=np.array([[1.0078125]],np.float32)
    values=WORKER['candidates'](x0,x0,n0,B)
    np.testing.assert_array_equal(values['anchored'],n0)
    np.testing.assert_array_equal(values['absolute'],[[1.]])
    # Installing P1-N0 and measuring P1-P0 answer different questions.
    assert float(values['absolute'][0,0]-n0[0,0])!=0
    assert float(values['absolute'][0,0]-B(x0)[0,0])==0


def test_receiving_norm_choice_changes_own_mlp_input_only_when_selected():
    law=runpy.run_path(str(HERE/'flux_receiver_norm_prediction.py'));r=runpy.run_path(str(HERE/'flux_reentry_reference.py'))
    rng=np.random.default_rng(51);i=B(rng.normal(size=(1,2,4)));t=B(rng.normal(size=(1,12,4)));s={};w={}
    context=dict(img_mod=np.zeros((1,24),np.float32),txt_mod=np.zeros((1,24),np.float32),rope=(np.ones((14,4),np.float32),np.zeros((14,4),np.float32)))
    for role,x,ff in [('image',i,'ff'),('text',t,'ff_context')]:
        s[role+'.norm1']=r['_ln'](x,1e-6);s[role+'.modulated']=s[role+'.norm1'];s[role+'.norm2']=np.full_like(x,77)
        for key in ('attention','mlp','mlp_input','mlp_activation'):s[role+'.'+key]=np.zeros_like(x)
        s[role+'.after_attention']=x;s[role+'.mlp_paired']=np.zeros((*x.shape[:-1],8),np.float32)
        for key in ('q_projection','k_projection','v_projection','q_normalized','k_normalized','q_positioned','k_positioned'):s[role+'.'+key]=np.zeros((*x.shape[:-1],1,4),np.float32)
        w[ff+'.linear_in.weight']=np.zeros((8,4),np.float32);w[ff+'.linear_out.weight']=np.zeros((4,4),np.float32)
    for p in ('attn.add_q_proj','attn.add_k_proj','attn.add_v_proj','attn.to_add_out','attn.to_out.0'):w[p+'.weight']=np.zeros((4,4),np.float32)
    for q in ('q','k'):w['attn.norm_added_'+q+'.weight']=np.ones(4,np.float32)
    s['attention.raw']=np.zeros((1,14,4),np.float32)
    anchor=law['predict'](law['prepare'](i,t,context,w,s,r,receiver_norm='anchored'),t)
    direct=law['predict'](law['prepare'](i,t,context,w,s,r,receiver_norm='absolute'),t)
    np.testing.assert_array_equal(anchor['fields']['image']['mlp_input'],np.full_like(i,77))
    np.testing.assert_array_equal(direct['fields']['image']['mlp_input'],r['_ln'](i,1e-6))
    np.testing.assert_array_equal(anchor['fields']['image']['exit'],direct['fields']['image']['exit'])


def test_better_endpoint_can_have_worse_edit_from_baseline_error():
    audit=runpy.run_path(str(HERE/'analyze_flux_receiver_norm.py'))['error_transport']
    old=audit([0.],[1.25],[0.],[1.]);new=audit([.5],[1.125],[0.],[1.])
    assert new['mean_square']['new']<old['mean_square']['new']
    assert new['mean_square']['response']>old['mean_square']['response']
    p=new['mean_square']
    assert p['response']==p['reconstructed']==.140625
    assert p['twice_cross_moment']==.125
