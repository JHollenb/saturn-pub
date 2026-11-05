"""Tiny native and separate scalar readout/Euler fixtures; no trained models."""
from __future__ import annotations

import builtins
import hashlib
import inspect
import math
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.normalization import AdaLayerNormContinuous

PATH = Path(__file__).parents[1]/'qualification/flux_step_boundary_reference.py'
R = runpy.run_path(str(PATH))


def b16(value):
    bits = int(np.asarray(value, np.float32).view(np.uint32))
    return np.asarray((bits + 32767 + ((bits >> 16) & 1)) & 0xffff0000, np.uint32).view(np.float32).item()


def source():
    image = np.array([[[-1., 1., 0., 2.], [1., -1., 1., -1.], [0., 1., 2., -1.]]], np.float32)
    temb = np.array([[0., .5, -.25, 1.]], np.float32)
    weights = {'modulation_weight': (np.arange(32).reshape(8, 4)%5-2).astype(np.float32)*.25,
        'modulation_bias': np.arange(8).astype(np.float32)*.125-.5,
        'projection_weight': np.array([[1., -.5, .25, 0.], [-.25, .5, 0., 1.]], np.float32),
        'projection_bias': np.array([.125, -.25], np.float32)}
    return image, temb, weights


def scalar_readout(image, temb, weights):
    # No vectorized helper calls: modulation SCALE,SHIFT order is explicit.
    activated = [b16(float(x)/(1 + math.exp(-float(x)))) for x in temb[0]]
    modulation = [b16(sum(float(w)*v for w, v in zip(row, activated))+float(bias))
                  for row, bias in zip(weights['modulation_weight'], weights['modulation_bias'])]
    scale, shift = modulation[:4], modulation[4:]
    result = []
    for row in image[0]:
        mean = sum(float(x) for x in row)/4
        variance = sum((float(x)-mean)**2 for x in row)/4
        norm = [b16((float(x)-mean)/math.sqrt(variance+3)) for x in row]
        modulated = [b16(b16(norm[j]*b16(1+scale[j]))+shift[j]) for j in range(4)]
        result.append([b16(sum(float(w)*v for w, v in zip(wrow, modulated))+float(bias))
                       for wrow, bias in zip(weights['projection_weight'], weights['projection_bias'])])
    return np.asarray([result], np.float32)


def test_complete_readout_matches_separate_scalar_construction_and_chunking():
    image, temb, weights = source()
    result = R['predict_readout'](image, temb, **weights, eps=3., neuron_chunk=1, sample_indices=[0, 2], retain_fields=True)
    alternate = R['predict_readout'](image, temb, **weights, eps=3., neuron_chunk=4)
    assert result['noise'].tobytes() == scalar_readout(image, temb, weights).tobytes() == alternate['noise'].tobytes()
    assert result['sampled']['noise'].tobytes() == result['noise'][:, [0, 2]].tobytes()
    assert result['noise'].shape == (1, 3, 2)


def test_tiny_actual_adalncontinuous_and_projection_native_parity():
    image, temb, weights = source()
    norm = AdaLayerNormContinuous(4, 4, elementwise_affine=False, eps=3., bias=True).to(torch.bfloat16).eval()
    projection = torch.nn.Linear(4, 2, bias=True).to(torch.bfloat16).eval()
    with torch.no_grad():
        for target, value in ((norm.linear.weight, weights['modulation_weight']), (norm.linear.bias, weights['modulation_bias']),
                              (projection.weight, weights['projection_weight']), (projection.bias, weights['projection_bias'])):
            target.copy_(torch.from_numpy(value).to(torch.bfloat16))
        actual_norm = norm(torch.from_numpy(image).to(torch.bfloat16), torch.from_numpy(temb).to(torch.bfloat16))
        actual = projection(actual_norm).float().numpy()
    result = R['predict_readout'](image, temb, **weights, eps=3., retain_fields=True)
    assert result['modulated'].tobytes() == actual_norm.float().numpy().tobytes()
    assert result['noise'].tobytes() == actual.tobytes()


def test_native_flow_scalar_product_rounds_dt_before_product():
    noise = np.array([[[.001983642578125, 1., -2.], [.0019683837890625, .5, 4.]]], np.float32)
    latent = np.zeros_like(noise)
    sigma, following = np.float32(.75), np.float32(.41666666)
    scheduler = FlowMatchEulerDiscreteScheduler(stochastic_sampling=False)
    scheduler.sigmas = torch.tensor([sigma, following, 0.], dtype=torch.float32)
    scheduler.timesteps = scheduler.sigmas[:-1]*1000
    scheduler.set_begin_index(0)
    actual = scheduler.step(torch.from_numpy(noise).to(torch.bfloat16), scheduler.timesteps[0],
                            torch.from_numpy(latent).to(torch.bfloat16), return_dict=False)[0]
    result = R['predict_flow_commit'](latent, noise, sigma, following)
    assert result['latent'].tobytes() == actual.float().numpy().tobytes()
    naive = torch.from_numpy(latent)+(torch.tensor(following)-torch.tensor(sigma))*torch.from_numpy(noise)
    assert naive.to(torch.bfloat16).float().numpy().tobytes() != result['latent'].tobytes()
    assert result['dt_fp32'] != result['dt_bf16']
    assert scheduler.step_index == 1


def test_native_hostile_scalar_fixture_discriminates_wrong_wide_product():
    qualification = R['qualify_torch_scalar_promotion'](torch, device='cpu')
    assert qualification['qualified'] and qualification['model_calls'] == 0
    assert all(c['native_vs_bf16_dt_mismatches'] == 0 and c['native_vs_fp32_dt_mismatches'] > 0 for c in qualification['cases'])


def test_primary_callable_source_pins_match_installed_039_profile():
    for key, value in (('AdaLayerNormContinuous', AdaLayerNormContinuous.forward),
                       ('FlowMatchEulerDiscreteScheduler', FlowMatchEulerDiscreteScheduler.step)):
        name = key + ('.forward_source_sha256' if key == 'AdaLayerNormContinuous' else '.step_source_sha256')
        assert hashlib.sha256(inspect.getsource(value).encode()).hexdigest() == R['SOURCE_CONTRACT'][name]


def test_inputs_unchanged_and_outputs_cannot_reenable_writes():
    image, temb, weights = source()
    arrays = [image, temb, *weights.values()]
    original = [v.tobytes() for v in arrays]
    result = R['predict_readout'](image, temb, **weights, eps=3.)
    assert original == [v.tobytes() for v in arrays]
    with pytest.raises(ValueError):
        result['noise'].setflags(write=True)
    with pytest.raises(ValueError):
        result['sampled']['scale'].setflags(write=True)
    result['source_contract']['modulation_order'].append('accidental metadata mutation')
    later = R['predict_readout'](image, temb, **weights, eps=3.)
    assert later['source_contract']['modulation_order'] == ['scale', 'shift']


@pytest.mark.parametrize('kind', ['image_nonbf16', 'nonfinite_weight', 'bias_geometry', 'sample_index', 'epsilon'])
def test_readout_contract_rejects_invalid_sources(kind):
    image, temb, weights = source()
    kwargs = {'eps': 3.}
    if kind == 'image_nonbf16':
        image[0, 0, 0] = .1234567
    elif kind == 'nonfinite_weight':
        weights['modulation_weight'][0, 0] = np.inf
    elif kind == 'bias_geometry':
        weights['projection_bias'] = np.zeros(3, np.float32)
    elif kind == 'sample_index':
        kwargs['sample_indices'] = [3]
    else:
        kwargs['eps'] = 0.
    with pytest.raises(ValueError):
        R['predict_readout'](image, temb, **weights, **kwargs)


@pytest.mark.parametrize('sigma', [np.array([.5], np.float32), 1, float('nan')])
def test_flow_requires_finite_scalar_schedule(sigma):
    value = np.ones((1, 1, 2), np.float32)
    with pytest.raises(ValueError):
        R['predict_flow_commit'](value, value, sigma, .25)


def test_prediction_functions_are_framework_free(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'diffusers', 'saturn_pub', 'mdb'}:
            raise AssertionError('native/framework import in predictor: '+name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    helper = runpy.run_path(str(PATH))
    image, temb, weights = source()
    result = helper['predict_readout'](image, temb, **weights, eps=3.)
    helper['predict_flow_commit'](np.zeros_like(result['noise']), result['noise'], .75, .5)
