"""Delivered BF16 dose matching, endpoints and mutation closure; no model."""
import runpy
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).parents[1]/'qualification'
H = runpy.run_path(str(HERE/'flux_binding_matched_dose.py'))
BF = runpy.run_path(str(HERE/'flux_binding_generalization_audit.py'))['round_bf16']


def test_original_endpoints_preserve_negative_zero_bits():
    b=np.array([[[0,0x3f80]]],dtype=np.uint16)
    s=np.array([[[0x8000,0x4000]]],dtype=np.uint16)
    np.testing.assert_array_equal(H['interpolate_bits'](b,s,0.),b)
    np.testing.assert_array_equal(H['interpolate_bits'](b,s,1.),s)
    assert H['interpolate_bits'](b,s,.5)[0,0,1] == 0x3fc0


def test_equal_dose_pairs_from_different_natural_strengths_and_no_mutation():
    b=BF(np.zeros((1,2,3),np.float32));a=BF(np.full((1,2,3),2.,np.float32));c=BF(np.full((1,2,3),.5,np.float32))
    originals=[x.tobytes() for x in (b,a,c)]
    panel={r['arm']:r for r in H['dose_panel'](b,a,b,c)}
    assert tuple(panel)==H['ARMS']
    for low,high in [('A.half-context','B.half'),('A.context','B.full'),('B.attribute-norm','A.full')]:
        assert panel[low]['landed_delta_l2']==panel[high]['landed_delta_l2']
        assert panel[low]['match_rel_error']==0
    # Several amplitudes land on the same BF16 plateau; actual values, rather
    # than the unrounded nominal amplitude4, establish this exact dose match.
    np.testing.assert_array_equal(panel['B.attribute-norm']['value'],a)
    assert panel['B.attribute-norm']['extrapolation']
    assert not panel['A.context']['extrapolation']
    assert [x.tobytes() for x in (b,a,c)]==originals


def test_matching_quantized_delivered_norm_not_nominal_scale():
    base=BF(np.array([[[100.,1.]]],np.float32));source=BF(np.array([[[101.,2.]]],np.float32))
    target=H['delta_norm'](base,H['interpolate_bits'](base,source,.3))
    alpha,value,landed=H['match_norm'](base,source,target)
    assert landed==target==H['delta_norm'](base,value)
    assert alpha>=0


def test_zero_direction_and_nonfinite_strength_refuse():
    b=BF(np.zeros((1,2,3),np.float32))
    with pytest.raises(ValueError):H['match_norm'](b,b,1.)
    with pytest.raises(ValueError):H['interpolate_bits'](b,b,float('nan'))
    assert H['expected_counts']()['native_transitions']==64*56
    assert H['expected_counts']()['receipts']==64*(56+2)
