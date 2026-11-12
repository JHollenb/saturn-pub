"""Finite paired writes include rounding and require actual source coordinates."""
import hashlib
import runpy
from pathlib import Path
import numpy as np
import pytest


def helper(name):return runpy.run_path(str(Path(__file__).resolve().parents[1]/'qualification'/name))


def test_finite_write_alignment_can_reverse_a_local_gain():
    m=helper('flux_consumer_trace_math.py');power=helper('flux_own_state_mlp_math.py')['power_account']
    before={('half','native'):np.array([0.]),('edit','native'):np.array([1.]),('half','old'):np.array([.2]),('edit','old'):np.array([1.3])}
    after={**before,('edit','old'):np.array([1.1])}
    result=m['finite_write'](before,after,'edit','old',power)
    assert result['identity_residual_rms']==0
    assert result['twice_signed_cross_moments']['carried_response_error × finite_write_response_error']<0
    with pytest.raises(ValueError,match='coordinate'):m['finite_write'](before,{k:np.repeat(v,2) for k,v in after.items()},'edit','old',power)


def test_euler_rounding_remainder_can_cancel_noise_error():
    m=helper('flux_consumer_trace_math.py');power=helper('flux_own_state_mlp_math.py')['power_account'];B=helper('flux_reentry_reference.py')['bf16']
    noise={('half','native'):np.array([0.],np.float32),('edit','native'):np.array([.015625],np.float32),('half','old'):np.array([0.],np.float32),('edit','old'):np.array([.01171875],np.float32)}
    inputs={key:np.array([1.],np.float32) for key in noise};outputs={key:B(inputs[key]+.5*value) for key,value in noise.items()}
    result=m['euler_account'](inputs,noise,outputs,.5,'edit','old',power)
    assert result['total_mse']==0 and result['identity_residual_rms']==0
    assert result['component_mse']['step_scaled_noise']>0 and result['component_mse']['finite_scheduler']>0


def test_native_parameter_recipe_restores_original_bytes_and_rejects_mutation(tmp_path):
    torch=pytest.importorskip('torch');from safetensors.torch import save_file
    from mdb.parameter_artifacts import restore
    model=torch.nn.Module();model.transformer_blocks=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False).to(torch.bfloat16) for _ in range(3)])
    model.unused=torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)
    root=tmp_path/'checkpoint';(root/'transformer').mkdir(parents=True);path=root/'transformer/diffusion_pytorch_model.safetensors'
    save_file(dict(model.named_parameters()),str(path));inventory={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()}
    payload=tmp_path/'payload';payload.mkdir();(payload/'retrieve_weights.py').write_text('# standalone test source\n')
    output=tmp_path/'out';output.mkdir();summary=helper('flux_trace_parameter_recipes.py')['recipes'](model,root,inventory,output,payload)
    import json
    manifest=json.loads((output/'parameter-artifacts.json').read_text());assert summary['tensors']==1 and summary['persisted_parameter_bytes']==0
    audited=helper('analyze_flux_consumer_trace.py')['audit_parameters'](output,dict(model_path=str(root)))
    assert audited['tensors']==1 and audited['exact_recipe_hashes']
    restored=tmp_path/'restored';restore(manifest,restored,root)
    item=manifest['artifacts'][0];assert hashlib.sha256((restored/item['path']).read_bytes()).hexdigest()==item['sha256']
    with torch.no_grad():model.transformer_blocks[2].weight.add_(1)
    with pytest.raises(ValueError,match='differs from original'):helper('flux_trace_parameter_recipes.py')['recipes'](model,root,inventory,output,payload)
