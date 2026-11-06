"""A color-swapped source must never overwrite the receiver's text ports."""
import runpy
from pathlib import Path

import numpy as np

Q=Path(__file__).parents[1]/'qualification'
C=runpy.run_path(str(Q/'flux_reader_route_future.py'))
B=runpy.run_path(str(Q/'flux_reentry_reference.py'))['bf16']
F=runpy.run_path(str(Q/'flux_reader_fresh.py'))
A=runpy.run_path(str(Q/'flux_reader_fresh_audit.py'))


def test_cross_context_band_keeps_receiver_text_and_nonselected_image_rows():
    base={k:np.ones((1,1026,2,2),np.float32) for k in ('k','v')}
    donor={k:np.full_like(v,8.) for k,v in base.items()}
    result=C['ports'](base,donor,dict(kind='band-cross',band=2),B,nt=2)
    for k in result:
        assert np.all(result[k][:,:514]==1)
        assert np.all(result[k][:,514:770]==8)
        assert np.all(result[k][:,770:]==1)


def test_own_value_scaling_requires_no_donor_and_keeps_keys_text_other_heads():
    base={k:np.ones((1,10,2,2),np.float32) for k in ('k','v')}
    result=C['ports'](base,None,dict(kind='own-v',head=1,scale=1.0625),B,nt=2)
    assert np.array_equal(result['k'],base['k'])
    assert np.all(result['v'][:,:2]==1) and np.all(result['v'][:,2:,0]==1)
    assert np.all(result['v'][:,2:,1]==1.0625)
    assert np.all(base['v']==1)


def test_baseline_color_masks_abstain_on_white_and_separate_yellow_green():
    masks=F['semantic_masks'](np.array([[[50,150,30],[180,180,40],[255,255,255]]]))
    assert masks['green'].tolist()==[[True,False,False]]
    assert masks['yellow'].tolist()==[[False,True,False]]


def test_region_effect_preserves_sign_and_empty_mask_status():
    base=np.array([[[50,150,30],[180,180,40]]]);changed=np.array([[[150,150,30],[100,180,40]]])
    green=np.array([[True,False]]);yellow=~green
    assert A['region_effect'](base,changed,green)['signed_green_score']<0
    assert A['region_effect'](base,changed,yellow)['signed_green_score']>0
    empty=A['region_effect'](base,changed,np.zeros_like(green))
    assert empty['mean_energy'] is None and empty['status']=='instrument-incomplete-empty-mask'


def test_fresh_evidence_publication_binds_the_numerical_execution(tmp_path):
    from mdb.equation_comparison import EquationCatalog, compare_equations
    context=dict(model='test-model',clock=dict(step=3,block=0))
    profile=dict(dtype='bfloat16',device='test-device',bf16_reduced_reduction=True)
    item=EquationCatalog(tmp_path).record(law_id='test-reader',
        binding=A['catalog_binding'](context,{'r':dict(parent='parent-cut')},dict(request_sha256='source-request'),profile),
        report=compare_equations({'x':[1.,2.]},{'x':[1.,2.]}))
    assert item['binding']['execution']==profile


def test_attention_stream_split_preserves_the_entire_other_stream():
    import torch
    base=torch.ones((1,6,2,2),dtype=torch.bfloat16);changed=torch.full_like(base,8)
    for kind in ('band-text','band-image','value-text','value-image'):
        result=C['output_route'](base,changed,kind,2)
        assert torch.equal(result[:,:2],changed[:,:2] if C['split_stream'](kind)=='text' else base[:,:2])
        assert torch.equal(result[:,2:],base[:,2:] if C['split_stream'](kind)=='text' else changed[:,2:])
    assert torch.all(base==1) and torch.all(changed==8)


def test_single_operand_band_does_not_leak_into_the_other_operand():
    base={k:np.ones((1,1026,2,2),np.float32) for k in ('k','v')}
    donor={k:np.full_like(v,8.) for k,v in base.items()}
    for kind,selected,protected in (('band-key','k','v'),('band-value','v','k'),('value-text','v','k'),('value-image','v','k')):
        result=C['ports'](base,donor,dict(kind=kind,band=2),B,nt=2)
        assert np.array_equal(result[protected],base[protected])
        assert np.all(result[selected][:,:514]==1) and np.all(result[selected][:,514:770]==8) and np.all(result[selected][:,770:]==1)


def test_operand_prediction_retains_counterexample_and_mask_abstention():
    def rows(k,v):return {name:dict(regions=dict(green=dict(signed_green_score=x))) for name,x in (('keys-only',k),('values-only',v))}
    miss=A['operand_hypothesis'](rows(-.2,.1))
    assert miss['values_larger_matches'] is False and miss['values_sign_matches'] is False
    unknown=A['operand_hypothesis'](rows(None,None))
    assert unknown['values_larger_matches'] is None and unknown['values_sign_matches'] is None


def test_stream_prediction_retains_failed_dominance_and_sign_separately():
    rows={name:dict(regions=dict(green=dict(signed_green_score=x))) for name,x in (('text-only',-.2),('image-only',.1))}
    miss=A['stream_hypothesis'](rows)
    assert miss['image_larger_matches'] is False and miss['image_sign_matches'] is False
