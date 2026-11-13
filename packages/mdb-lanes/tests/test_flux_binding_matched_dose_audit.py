"""Independent saved-only dose and consumer-axis arithmetic."""
from pathlib import Path
import runpy
import numpy as np

HERE=Path(__file__).parents[1]/'qualification'
H=runpy.run_path(str(HERE/'flux_binding_matched_dose_audit.py'))
BF=H['helpers']()


def test_exact_signed_zero_endpoints_and_delivered_norm():
    b=np.array([[[0,0x3f80]]],np.uint16);s=np.array([[[0x8000,0x4000]]],np.uint16)
    np.testing.assert_array_equal(H['mixed_rows'](b,s,[0],0,BF),b)
    np.testing.assert_array_equal(H['mixed_rows'](b,s,[0],1,BF),s)
    assert H['delta_metrics'](b,s,BF)['l2']==1.
    assert H['delta_metrics'](b,s,BF)['different_bf16_elements']==2


def test_consumer_axis_preserves_signed_and_orthogonal_responses():
    base=np.zeros((2,4,3),np.uint8);source=base.copy();source[:,:,0]=10
    image=base.copy();image[:,:,0]=5;image[:,:,1]=3
    result=H['rgb_response'](image,base,base,source)['whole']
    assert result['signed_increment_on_natural_axis']==.5
    assert np.isclose(result['axis_orthogonal_response_rms_0_255'],np.sqrt(3))
    reverse=H['rgb_response'](base,image,base,source)['whole']
    assert reverse['signed_increment_on_natural_axis']==-.5
    empty=H['rgb_response'](image,base,base,base)['whole']
    assert empty['signed_increment_on_natural_axis'] is None


def test_projected_text_dose_full_field_norm_gain_and_changed_rows():
    sham=np.zeros((1,4,2),np.uint16)
    actual=BF['round_bf16'](np.array([[[0,0],[3,4],[0,0],[0,0]]],np.float32))
    result=H['projected_text_dose'](sham,actual,2.,BF)
    assert result['l2']==5.
    assert np.isclose(result['rms'],5/np.sqrt(8))
    assert result['projection_gain_l2']==2.5
    assert result['bit_changed_token_rows']==result['numerically_changed_token_rows']==1
    assert result['token_rows']==4 and result['elements']==8


def test_projected_text_dose_zero_gain_and_signed_zero_are_explicit():
    sham=np.zeros((1,4,2),np.uint16)
    result=H['projected_text_dose'](sham,sham,0.,BF)
    assert result['projection_gain_l2'] is None
    assert result['gain_status']=='undefined-zero-conditioning-delta'
    assert result['l2']==0. and result['numerically_changed_token_rows']==0
    signed_zero=sham.copy();signed_zero[0,2,1]=0x8000
    result=H['projected_text_dose'](sham,signed_zero,1.,BF)
    assert result['bit_changed_token_rows']==1
    assert result['numerically_changed_token_rows']==0
    assert result['l2']==result['projection_gain_l2']==0.
