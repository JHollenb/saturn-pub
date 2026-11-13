"""Pulse continuation and explicit sequential BF16 joint image-write equations."""
CUTS={'after_joint0':('joint',1),'after_joint4':('single',0),'after_single6':('single',7),'after_single18':('single',19)}


def census():
    remaining=(82,78,71,59)
    transitions=4*28+4*56+2*56+3*sum(remaining)+4*82
    acts=4+4+4*4+12+4+4
    return dict(loads=1,encodes=0,noise_draws=0,native_transitions=transitions,acts=acts,
                receipts=transitions+acts,images=22,candidate_blocks=5)


def run_reset_suffix(nav, *, reset_conditioning, apply, on_boundary=None):
    actions=[];reset=False
    while nav.point['logical_step']<4:
        if nav.point['phase']=='project':
            if reset_conditioning is not None and not reset and nav.point['logical_step']==2:
                actions.extend(apply(nav,reset_conditioning));reset=True
            if on_boundary is not None:on_boundary(nav)
        nav.step()
    if on_boundary is not None:on_boundary(nav)
    return actions


def predict_exits(base,shape):
    import numpy as np
    import runpy
    from pathlib import Path
    B=runpy.run_path(str(Path(__file__).with_name('flux_conditioning_program.py')))['bf16']
    names=('residual','attention','mlp','gate_attention','gate_mlp')
    if any(not np.isfinite(v[n]).all() for v in (base,shape) for n in names):raise ValueError('Nonfinite port')
    if not all(np.array_equal(base[n],shape[n]) for n in ('residual','gate_attention','gate_mlp')):raise ValueError('Matched residual and gates required')
    result={}
    for a,m in ((0,0),(1,0),(0,1),(1,1)):
        R=base['residual'].astype(np.float64)
        A=(shape if a else base)['attention'].astype(np.float64)
        M=(shape if m else base)['mlp'].astype(np.float64)
        ga,gm=[base[n].astype(np.float64) for n in ('gate_attention','gate_mlp')]
        result[str(a)+str(m)]=B(B(R+B(ga*A)).astype(np.float64)+B(gm*M))
    return result
