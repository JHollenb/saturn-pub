import runpy
from pathlib import Path
import torch


def test_declared_receiving_latent_doses_preserve_endpoints_without_randomness():
    h=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_latent_neighborhood.py'))
    a=torch.tensor([.0625,1.,-2.],dtype=torch.bfloat16);b=torch.tensor([1e-10,-1.,2.],dtype=torch.bfloat16)
    assert torch.equal(h['mix'](a,b,0),a) and torch.equal(h['mix'](a,b,1),b)
    assert h['ALPHAS']==(0.,.05,.1,.25,.5,1.)
    for alpha in h['ALPHAS']:
        v=h['mix'](a,b,alpha);assert v.dtype==a.dtype and torch.isfinite(v).all()
