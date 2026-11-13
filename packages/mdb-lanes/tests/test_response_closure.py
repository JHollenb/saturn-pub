import numpy as np
import pytest
from mdb_lanes.response_closure import audit_response,audit_factorial


def test_constant_defect_cancels_but_changing_defect_does_not():
    n0=np.array([2.,8.]); n1=np.array([3.,10.]); p0=n0+4; p1=n1+4
    anchored=n0+p1-p0
    r=audit_response(n0,n1,p0,p1,candidates={'anchored':anchored,'copy':n0})
    assert r['reference_endpoint']['rms']==4
    assert r['reference_response']['rms']==0
    assert r['candidates']['anchored']['endpoint']['rms']==0
    assert r['candidates']['copy']['error_over_native_effect']==1
    p1=p1+np.array([1.,-2.]); recast=np.round((n0+p1-p0)*2)/2
    r=audit_response(n0,n1,p0,p1,candidates={'anchored':recast})
    assert r['reference_response']['rms']>0
    assert r['candidates']['anchored']['anchoring_identity_residual_rms']==0


def test_factorial_reveals_rounding_interaction_in_an_affine_law():
    wide={k:np.array([v]) for k,v in {'00':.25,'01':.5,'10':.5,'11':.75}.items()}
    native={k:np.round(v) for k,v in wide.items()}
    r=audit_factorial(native,wide)
    assert r['contrasts']['receiver_by_dose']['reference_rms']==1
    assert r['contrasts']['receiver_by_dose']['relative_rms']==1


def test_zero_effect_is_undefined_relative_error_and_invalid_arrays_are_rejected():
    r=audit_response([0],[0],[1],[1],candidates={'copy':[0]})
    assert r['reference_response']['relative_rms'] is None
    assert r['candidates']['copy']['error_over_native_effect'] is None
    with pytest.raises(ValueError,match='finite'):audit_response([0],[np.nan],[0],[0])
    with pytest.raises(ValueError,match='same-shaped'):audit_response([0],[0,1],[0],[0])
    with pytest.raises(ValueError,match='four explicit'):audit_factorial({}, {})


def test_error_power_retains_signed_covariance_and_final_rounding():
    n0=np.array([2.,8.]); n1=np.array([3.25,10.25]); p0=n0+np.array([.5,-.5]); p1=n1+np.array([1.,-.25])
    c=np.round(n0+(p1-p0)); row=audit_response(n0,n1,p0,p1,candidates={'rounded':c})
    d=row['defect_power']; assert d['change_mse']==d['anchor_mse']+d['target_mse']-d['signed_cross_twice']
    c=row['candidates']['rounded']; d=c['candidate_power']
    assert d['modification_mse']>0
    assert d['error_mse']==d['defect_change_mse']+d['modification_mse']+d['signed_cross_twice']
    assert c['anchoring_identity_residual_rms']==0
