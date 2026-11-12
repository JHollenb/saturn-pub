"""Distinct RMSNorm implementations have distinct numerical contracts."""
import runpy
from pathlib import Path
import numpy as np


def test_distinct_rms_implementations_have_distinct_rounding_contracts():
    import torch
    from diffusers.models.normalization import RMSNorm
    here=Path(__file__).resolve().parents[1]/'qualification'
    reference=runpy.run_path(str(here/'flux_reentry_reference.py'))
    norm=runpy.run_path(str(here/'flux_response_reference.py'))['qk_rms'];B=reference['bf16']
    rng=np.random.default_rng(81);x=B(rng.normal(size=(1,37,4,128)));w=B(rng.uniform(.5,1.5,size=128))
    wrapper=RMSNorm(128,eps=1e-6).to(torch.bfloat16)
    native=torch.nn.RMSNorm(128,eps=1e-6).to(torch.bfloat16)
    with torch.no_grad():
        wrapper.weight.copy_(torch.from_numpy(w));native.weight.copy_(torch.from_numpy(w))
        observed=wrapper(torch.from_numpy(x).to(torch.bfloat16)).float().numpy()
        fused=native(torch.from_numpy(x).to(torch.bfloat16)).float().numpy()
    staged=norm(x,w,reference,rounding='staged');legacy=norm(x,w,reference,rounding='legacy')
    np.testing.assert_array_equal(staged,observed)
    np.testing.assert_array_equal(legacy,fused)
    assert np.count_nonzero(legacy!=observed)>x.size//10


def test_identity_rms_weight_and_legacy_default_remain_unchanged():
    here=Path(__file__).resolve().parents[1]/'qualification'
    r=runpy.run_path(str(here/'flux_reentry_reference.py'));norm=runpy.run_path(str(here/'flux_response_reference.py'))['qk_rms'];x=r['bf16'](np.arange(16,dtype=np.float32).reshape(1,2,8));w=np.ones(8,np.float32)
    np.testing.assert_array_equal(norm(x,w,r),norm(x,w,r,rounding='legacy'))
    np.testing.assert_array_equal(norm(x,w,r,rounding='staged'),norm(x,w,r))
