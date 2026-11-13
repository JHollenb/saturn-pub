"""Qualify source extraction, original gate order and observation-only ports."""
import hashlib
import importlib.metadata
import runpy
from pathlib import Path

import pytest


def core():return runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/flux_mlp_primitive_cut.py'))


def test_original_source_component_and_port_geometry():
    torch=pytest.importorskip('torch');h=core();path=Path(importlib.metadata.distribution('diffusers').locate_file('diffusers/models/transformers/transformer_flux2.py'));cls,hashes=h['extract_classes'](path,hashlib.sha256(path.read_bytes()).hexdigest())
    assert set(hashes)=={'Flux2FeedForward','Flux2SwiGLU'}
    with torch.device('meta'):module=cls(4,inner_dim=3,bias=False)
    W1=torch.arange(24,dtype=torch.float32).reshape(6,4)/32;W2=torch.arange(12,dtype=torch.float32).reshape(4,3)/24
    module.load_state_dict({'linear_in.weight':W1,'linear_out.weight':W2},assign=True);module.eval().requires_grad_(False);x=torch.arange(8,dtype=torch.float32).reshape(1,2,4)/16;counts=dict(mlp=0,linear=0,silu=0,product=0)
    with torch.inference_mode():fields,layouts=h['capture'](module,x,counts)
    z=torch.nn.functional.linear(x,W1);u,v=z.chunk(2,dim=-1);s=torch.nn.functional.silu(u);a=s*v;y=torch.nn.functional.linear(a,W2)
    for key,value in dict(paired=z,silu=s,activation=a,mlp=y).items():assert torch.equal(fields[key],value)
    assert counts==dict(mlp=1,linear=2,silu=1,product=1)
    assert layouts['chunks']['u']['stride']==[12,6,1] and layouts['chunks']['v']['storage_offset']==3
    assert not any(child._forward_hooks for child in module.modules())


def test_source_drift_refuses_before_class_execution(tmp_path):
    path=tmp_path/'module.py';path.write_text('raise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError,match='source changed'):core()['extract_classes'](path,'0'*64)


def test_output_space_error_identity_preserves_signed_cancellation():
    import numpy as np
    h=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/analyze_flux_mlp_primitives.py'))
    # A large first term and opposing local output term can leave a small total.
    C=np.array([1.,2.,3.]);Z=np.array([4.,5.,6.]);A=np.array([3.,4.,5.]);Y=np.array([0.,1.,2.])
    terms,error=h['substitutions'](dict(C=C,after_native_paired=Z,after_native_activation=A),Y);result=h['signed_error'](terms,error)
    assert result['identity_residual_rms']==0 and result['summed_power']==result['total_mse']==1
    assert result['twice_signed_cross_moments']['first_affine_propagation × local_output_affine']==-18
    assert sum(result['component_mse'].values())>result['total_mse']


def test_error_terms_cannot_mix_primitive_spaces():
    import numpy as np
    h=runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification/analyze_flux_mlp_primitives.py'))
    with pytest.raises(ValueError,match='same-shaped'):
        h['signed_error'](dict(paired=np.zeros((2,6)),output=np.zeros((2,3))),np.zeros((2,3)))
