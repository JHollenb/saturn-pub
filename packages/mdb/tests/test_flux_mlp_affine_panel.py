"""A transported baseline defect must remain visible in both endpoint laws."""
import runpy
from pathlib import Path
import numpy as np
import pytest


def test_absolute_and_anchored_have_their_own_baselines():
    here=Path(__file__).resolve().parents[1]/'qualification';panel=runpy.run_path(str(here/'flux_mlp_affine_panel.py'));r=runpy.run_path(str(here/'flux_reentry_reference.py'));law=runpy.run_path(str(here/'flux_receiver_norm_prediction.py'))
    x0=np.ones((1,2,2),np.float32);x1=2*x0;y0=np.full_like(x0,1.25);W=np.eye(2,dtype=np.float32)
    p={method:[panel['affine'](x,x0,y0,W,method,r,law) for x in (x0,x1)] for method in panel['METHODS']}
    assert np.array_equal(p['anchored'][0],y0) and np.array_equal(p['absolute'][0],x0)
    assert np.array_equal(p['anchored'][1],x1+.25) and np.array_equal(p['absolute'][1],x1)
    # Both finite responses are one here, despite different baseline/target errors.
    assert np.array_equal(p['anchored'][1]-p['anchored'][0],p['absolute'][1]-p['absolute'][0])
    with pytest.raises(ValueError,match='predeclared'):
        panel['affine'](x1,x0,y0,W,'auto-fit',r,law)
