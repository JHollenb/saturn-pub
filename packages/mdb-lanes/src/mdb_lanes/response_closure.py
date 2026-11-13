"""Finite-response audits without scientific pass/fail thresholds.

NumPy is loaded only on use. Arrays must already be source-bound by the caller.
These measurements distinguish endpoint accuracy, edit accuracy and baseline
copying; they do not certify semantic closure or automatically diagnose a cause.
"""


def _arrays(*values):
    import numpy as np
    arrays=[np.asarray(v,dtype=np.float64) for v in values]
    if not arrays or any(a.shape!=arrays[0].shape or not a.size or not np.isfinite(a).all() for a in arrays):
        raise ValueError('same-shaped finite nonempty response arrays required')
    return arrays


def _rms(value):
    import numpy as np
    return float(np.sqrt(np.mean(value*value)))


def measure_error(predicted,reference):
    """Signed-array comparison; relative error is None for a zero reference."""
    import numpy as np
    p,n=_arrays(predicted,reference); d=p-n; nr=_rms(n); pr=_rms(p); error=_rms(d)
    return dict(rms=error,reference_rms=nr,relative_rms=None if nr==0 else error/nr,
        max_abs=float(np.max(np.abs(d))),different=int(np.count_nonzero(d)),scalars=int(n.size),
        cosine=None if nr==0 or pr==0 else float(np.mean(p*n)/(nr*pr)))


def audit_response(native_anchor,native_target,reference_anchor,reference_target,*,candidates=None):
    """Audit a same-receiver finite response and optional stored target forecasts.

    Anchoring N0+P1-P0 has exactly the original reference response error. For a
    supplied candidate C, rho=C-(N0+P1-P0) includes any final recast or other
    modification; only a genuine anchored candidate makes rho a rounding term.
    No candidate is assumed to be anchored merely from its name.
    """
    import numpy as np
    n0,n1,p0,p1=_arrays(native_anchor,native_target,reference_anchor,reference_target)
    effect=n1-n0; response=p1-p0; b0=p0-n0; b1=p1-n1; unrounded=n0+response
    result=dict(schema='mdb-response-audit-v1',native_effect_rms=_rms(effect),
        reference_endpoint=measure_error(p1,n1),reference_response=measure_error(response,effect),
        copy_anchor=measure_error(n0,n1),defect_change=measure_error(b1,b0),
        defect_power=dict(anchor_mse=float(np.mean(b0*b0)),target_mse=float(np.mean(b1*b1)),
            signed_cross_twice=float(2*np.mean(b0*b1)),change_mse=float(np.mean((b1-b0)**2))),candidates={})
    for name,value in (candidates or {}).items():
        c,_=_arrays(value,n1); rho=c-unrounded; residual=c-n1
        result['candidates'][name]=dict(endpoint=measure_error(c,n1),
            error_over_native_effect=None if _rms(effect)==0 else _rms(residual)/_rms(effect),
            modification_from_unrounded_anchor_rms=_rms(rho),
            candidate_power=dict(error_mse=float(np.mean(residual*residual)),defect_change_mse=float(np.mean((b1-b0)**2)),
                modification_mse=float(np.mean(rho*rho)),signed_cross_twice=float(2*np.mean((b1-b0)*rho))),
            anchoring_identity_residual_rms=_rms(residual-((b1-b0)+rho)))
    return result


def audit_factorial(native_corners,predicted_corners):
    """Measure a receiver-by-edit four-corner response.

    Keys 00,01,10,11 mean receiver then dose (0=anchor, 1=new dose).
    All corners must describe the same output space and execution boundary.
    Mixed differences include rounding effects; they are not semantic labels.
    """
    keys=('00','01','10','11')
    if set(native_corners)!=set(keys) or set(predicted_corners)!=set(keys): raise ValueError('four explicit receiver/dose corners required')
    arrays=_arrays(*[native_corners[k] for k in keys],*[predicted_corners[k] for k in keys]); n=dict(zip(keys,arrays[:4])); p=dict(zip(keys,arrays[4:]))
    forms=dict(receiver0_dose={'01':1,'00':-1},receiver1_dose={'11':1,'10':-1},
        new_dose_receivers={'11':1,'01':-1},anchor_receivers={'10':1,'00':-1},
        receiver_by_dose={'11':1,'10':-1,'01':-1,'00':1})
    return dict(schema='mdb-response-factorial-audit-v1',contrasts={name:measure_error(
        sum(p[k]*sign for k,sign in coeff.items()),sum(n[k]*sign for k,sign in coeff.items())) for name,coeff in forms.items()})
