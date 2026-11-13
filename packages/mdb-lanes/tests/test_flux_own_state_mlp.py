"""Exercise real finite-write counterexamples before a saved-state experiment."""
import runpy
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]/'qualification'


def instruments():
    return runpy.run_path(str(HERE/'flux_own_state_mlp_math.py')), runpy.run_path(str(HERE/'flux_reentry_reference.py'))


def tensor(value):
    return np.array([[[value]]],np.float32)


def test_better_mlp_can_remove_residual_error_cancellation():
    h,r=instruments();B=r['bf16'];gate=tensor(1)
    zero=h['writeback'](tensor(0),gate,tensor(0),B)
    native=h['writeback'](tensor(2),gate,tensor(-1),B)
    old=h['writeback'](tensor(2.5),gate,tensor(-1.5),B)
    new=h['writeback'](tensor(2.5),gate,tensor(-1),B)
    assert abs(float(new['mlp'].item()-native['mlp'].item())) < abs(float(old['mlp'].item()-native['mlp'].item()))
    assert old['exit'].item()==native['exit'].item() and new['exit'].item()!=native['exit'].item()
    before=h['transport'](zero,native,zero,old)['prediction_error'];after=h['transport'](zero,native,zero,new)['prediction_error']
    assert before['total_mse']==0 and after['total_mse']==.25
    assert before['twice_signed_cross_moments']['after_attention × time_gated_mlp']==-.5
    assert before['identity_residual_rms']==after['identity_residual_rms']==0


def test_difference_of_addition_defects_is_needed_for_finite_response():
    h,r=instruments();B=r['bf16'];gate=tensor(1)
    n0=h['writeback'](tensor(256),gate,tensor(1),B);n1=h['writeback'](tensor(256),gate,tensor(2),B)
    terms,total=h['differences'](n0,n1)
    assert terms['time_gated_mlp'].item()==1 and terms['finite_addition'].item()==1 and total.item()==2
    p0=h['writeback'](tensor(255),gate,tensor(1),B);p1=n1
    error=h['transport'](n0,n1,p0,p1)['prediction_error']
    assert error['total_mse']==0 and error['component_mse']['after_attention']==1
    assert error['component_mse']['finite_addition']==1
    assert error['twice_signed_cross_moments']['after_attention × finite_addition']==-2


def test_extracted_old_equation_matches_frozen_qualified_subprogram():
    h,r=instruments();law=runpy.run_path(str(HERE/'flux_receiver_norm_prediction.py'));old=runpy.run_path(str(HERE/'flux_hybrid_mlp_cut.py'))['mlp'];B=r['bf16']
    x0=B(np.arange(8,dtype=np.float32).reshape(1,2,4)/16);x=B(x0+.125)
    W1=B(np.arange(24,dtype=np.float32).reshape(6,4)/32);W2=B(np.arange(12,dtype=np.float32).reshape(4,3)/24)
    z0=r['_linear'](x0,{'w.weight':W1},'w',256);u,v=np.split(z0,2,axis=-1);a0=B(r['_silu'](u)*v);y0=r['_linear'](a0,{'w.weight':W2},'w',256)
    stages=dict(mlp_input=x0,mlp_paired=z0,mlp_activation=a0,mlp=y0)
    actual=h['mlp'](x,stages,{'in':W1,'out':W2},'anchored',r,law)
    expected=old(x,{'image.'+k:v for k,v in stages.items()},{'ff.linear_in.weight':W1,'ff.linear_out.weight':W2},'image',r,law)
    assert np.array_equal(actual,expected)


def test_wrong_output_space_and_gate_shape_are_refused():
    h,r=instruments()
    with pytest.raises(ValueError,match='geometry'):
        h['writeback'](np.ones((1,2,3)),np.ones(3),np.ones((1,2,3)),r['bf16'])
    with pytest.raises(ValueError,match='same-space'):
        h['power_account']({'first':np.ones((2,4))},np.ones((2,3)))


def test_exact_control_distinguishes_numeric_equality_from_signed_zero():
    h,_=instruments();positive=np.array([0.],np.float32);negative=np.array([-0.],np.float32)
    assert np.array_equal(positive,negative)
    result=h['equality'](positive,negative)
    assert result['numeric_exact'] and not result['raw_bits_exact'] and result['signed_zero_differences']==1
    assert not h['same_bits'](positive,positive.astype(np.float64))
