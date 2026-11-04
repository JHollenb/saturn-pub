"""Small independent algebra and hostile chain-provenance fixtures; no models."""
from __future__ import annotations

import ast
import copy
import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_transition_chain.py"))


def test_command_line_entry_runs_only_after_all_audit_functions_are_defined():
    module = ast.parse((HERE/'analyze_flux_transition_chain.py').read_text())
    guards = [node for node in module.body if isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)]
    assert len(guards) == 1
    assert max(node.end_lineno for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))) < guards[0].lineno


def b16(x):
    bits = int(np.asarray(x, np.float32).view(np.uint32))
    return np.asarray((bits + 32767 + ((bits >> 16) & 1)) & 0xffff0000, np.uint32).view(np.float32).item()


def specimen():
    r = np.array([[[-1, 1, 0, 2], [1, -1, 1, -1], [0, 1, 2, -1]]], np.float32)
    mod = np.array([[.25, -.5, .5, 0, 0, 1, -.5, .25, .5, -.25, 1, 0]], np.float32)
    block = {"qkv_mlp_weight": (np.arange(72).reshape(18, 4) % 7 - 3).astype(np.float32) * .25,
             "qkv_mlp_bias": np.full(18, .125, np.float32),
             "to_out_weight": (np.arange(28).reshape(4, 7) % 5 - 2).astype(np.float32) * .25,
             "to_out_bias": np.array([.125, -.25, .5, -.125], np.float32),
             "norm_q_weight": np.array([1, .5], np.float32),
             "norm_k_weight": np.array([1.5, 1], np.float32)}
    kwargs = dict(heads=2, layernorm_eps=3., qk_eps=.5,
                  rope_cos=np.ones((3, 2), np.float32), rope_sin=np.zeros((3, 2), np.float32))
    return r, mod, block, kwargs


def scalar_transition(r, mod, block):
    # Every matrix row, token, head and softmax reduction is a scalar loop.
    fused = []
    for token in r[0]:
        mean = sum(float(x) for x in token) / 4
        variance = sum((float(x) - mean) ** 2 for x in token) / 4
        norm = [b16((float(x) - mean) / math.sqrt(variance + 3)) for x in token]
        affine = [b16(b16(b16(1 + float(mod[0, 4 + j])) * norm[j]) + float(mod[0, j])) for j in range(4)]
        fused.append([b16(sum(float(w) * x for w, x in zip(row, affine)) + float(bias))
                      for row, bias in zip(block['qkv_mlp_weight'], block['qkv_mlp_bias'])])
    q, k = [], []
    for index, target in ((0, q), (4, k)):
        gamma = block['norm_q_weight' if index == 0 else 'norm_k_weight']
        for row in fused:
            normalized = []
            for h in range(2):
                bank = row[index + 2*h:index + 2*h + 2]
                denom = math.sqrt(sum(x*x for x in bank) / 2 + .5)
                normalized.extend(b16(x / denom * float(gamma[j])) for j, x in enumerate(bank))
            target.append(normalized)
    result = []
    for i, row in enumerate(fused):
        attn = []
        for h in range(2):
            scores = [sum(q[i][2*h+j] * k[t][2*h+j] for j in range(2)) / math.sqrt(2) for t in range(3)]
            exps = [math.exp(s-max(scores)) for s in scores]
            probs = [v/sum(exps) for v in exps]
            attn.extend(b16(sum(probs[t] * fused[t][8+2*h+j] for t in range(3))) for j in range(2))
        mlp = [b16(b16(row[12+j] / (1 + math.exp(-row[12+j]))) * row[15+j]) for j in range(3)]
        concat = attn + mlp
        projection = [b16(sum(float(w)*x for w, x in zip(wrow, concat)) + float(bias))
                      for wrow, bias in zip(block['to_out_weight'], block['to_out_bias'])]
        result.append([b16(float(r[0, i, j]) + b16(float(mod[0, 8+j]) * projection[j])) for j in range(4)])
    return np.asarray([result], np.float32)


def test_full_transition_all_tokens_matches_separate_scalar_construction():
    r, mod, block, kwargs = specimen()
    actual = A['independent_transition'](r, mod, block, **kwargs, query_chunk=1, row_chunk=3)
    assert A['exact_bits'](actual, scalar_transition(r, mod, block))
    assert A['exact_bits'](actual, A['independent_transition'](r, mod, block, **kwargs, query_chunk=3, row_chunk=18))
    assert not A['exact_bits'](actual[:, :1], r[:, :1])  # text query output is predicted too


@pytest.mark.parametrize('kind', ['nonfinite', 'nonbf16', 'wrong_projection', 'invalid_chunk'])
def test_invalid_source_contracts_refused(kind):
    r, mod, block, kwargs = specimen()
    if kind == 'nonfinite':
        block['norm_q_weight'][0] = np.nan
    elif kind == 'nonbf16':
        r[0, 0, 0] = np.float32(.1234567)
    elif kind == 'wrong_projection':
        block['to_out_weight'] = block['to_out_weight'][:, :6]
    else:
        kwargs['query_chunk'] = 0
    with pytest.raises(ValueError):
        A['independent_transition'](r, mod, block, **kwargs)


def test_full_vector_decomposition_retains_opposed_errors_and_no_error_floor():
    native = np.zeros((1, 2, 2), np.float32)
    propagated = np.ones_like(native) * 2
    predicted = np.ones_like(native)
    originals = [copy.copy(x) for x in (native, propagated, predicted)]
    result = A['vector_decomposition'](predicted, propagated, native)
    assert result['closure_max_abs'] == 0
    assert result['local']['signed_mean'] == -1 and result['carried']['signed_mean'] == 2
    assert result['local_dot_carried'] == -8
    assert all(np.array_equal(x, y) for x, y in zip((native, propagated, predicted), originals))


def saved_chain():
    r, mod, block, kwargs = specimen()
    steps, current = [], r
    for index in range(2):
        output = A['independent_transition'](current, mod, block, **kwargs)
        descriptor = A['tensor_identity'](current)
        prediction = {'output': output, 'input_descriptor': descriptor,
            'output_descriptor': A['tensor_identity'](output), 'text_rows': 1,
            'text': output[:, :1].copy(), 'image': output[:, 1:].copy(),
            'sampled': {'indices': [0, 2], 'input': current[:, [0, 2]].copy(), 'output': output[:, [0, 2]].copy()}}
        steps.append({'step': index, 'block_index': index+6,
            'input_source': 'caller pre-entry residual' if index == 0 else 'previous predicted full output',
            'input_descriptor': descriptor, 'prediction': prediction})
        current = output
    return r, {'chain_length': 2, 'input_descriptor': A['tensor_identity'](r), 'steps': steps,
        'output': current, 'output_descriptor': A['tensor_identity'](current),
        'text': current[:, :1].copy(), 'image': current[:, 1:].copy()}


def test_saved_chain_carries_both_full_streams_and_exact_output_alias():
    r, chain = saved_chain()
    outputs = A['verify_predicted_chain'](chain, r, text_rows=1, sample_indices=[0, 2])
    assert A['exact_bits'](outputs[0][:, [0, 2]], chain['steps'][1]['prediction']['sampled']['input'])
    assert A['exact_bits'](outputs[1], chain['output'])
    assert not A['exact_bits'](outputs[0][:, :1], r[:, :1])


@pytest.mark.parametrize('corruption', ['stale_source', 'native_intermediate', 'stale_text', 'final_alias', 'wrong_block', 'digest'])
def test_saved_chain_refuses_teacher_forcing_or_broken_full_state_bindings(corruption):
    r, chain = saved_chain()
    second = chain['steps'][1]
    if corruption == 'stale_source':
        second['input_source'] = 'native current block6 output'
    elif corruption == 'native_intermediate':
        # A plausible finite native intermediate is still forbidden as input7.
        native = A['quantize'](r + np.float32(.5))
        second['input_descriptor'] = second['prediction']['input_descriptor'] = A['tensor_identity'](native)
        second['prediction']['sampled']['input'] = native[:, [0, 2]].copy()
    elif corruption == 'stale_text':
        chain['steps'][0]['prediction']['text'] = r[:, :1].copy()
    elif corruption == 'final_alias':
        chain['output'] = chain['steps'][0]['prediction']['output']
    elif corruption == 'wrong_block':
        second['block_index'] = 6
    else:
        second['prediction']['output_descriptor']['fp32_le_sha256'] = '0'*64
    with pytest.raises(ValueError):
        A['verify_predicted_chain'](chain, r, text_rows=1, sample_indices=[0, 2])


def test_zero_error_decomposition_remains_measured_zero():
    value = np.ones((1, 1, 2), np.float32)
    result = A['vector_decomposition'](value, value, value)
    assert result['local']['rms'] == result['carried']['rms'] == result['total']['rms'] == 0


def test_expected_cursor_advance_keeps_data_fixed_and_clock_witness_is_checked():
    scalar = lambda value: {'kind': 'scalar', 'value': value}
    point = {'logical_step': 3, 'phase': 'single', 'index': 7, 'local_clock': 7}
    fields = {'step': scalar(3), 'phase': scalar('single'), 'cursor': scalar(7),
              'conditioning': scalar('source-bound-C'), 'context': scalar('source-bound-time'),
              'image': scalar('I'), 'text': scalar('T')}
    payload = {'kind': 'mapping', 'items': [{'key': scalar(k), 'value': v} for k, v in fields.items()]}
    record = {'point': point, 'clock_state': {'step': 3, 'phase': 'single', 'cursor': 7},
        'non_stream_state': {**payload, 'items': [x for x in payload['items'] if x['key']['value'] not in {'image', 'text', 'cursor'}]}}
    initial = {**fields, 'cursor': scalar(6)}
    e = SimpleNamespace(fields=lambda cut: fields if cut == 'after' else initial,
        point=lambda _: point, typed={'unpack': lambda x: x['value']}, store={'cuts': {'after': {'payload': payload}}})
    A['verify_clock_and_data'](e, 'after', record)
    assert A['stable_fields'](e, 'after') == A['stable_fields'](e, 'before')
    record['clock_state']['cursor'] = 6
    with pytest.raises(ValueError, match='clock-state'):
        A['verify_clock_and_data'](e, 'after', record)


@pytest.mark.parametrize('corruption', ['target_entry', 'branch_before_seal'])
def test_custodial_timeline_rejects_native_entry_or_consumption_before_seal(tmp_path, corruption):
    before = {**A['COUNTS'], 'adapter_steps': 108, 'acts': 4, 'receipt_emissions': 112, 'images': 0, 'vae_calls': 0}
    seal = {'seal_sha256': 'a'*64}
    events = [{'kind': 'prediction_start'}, {'kind': 'transition_chain_sealed', 'seal': seal,
        'counts': before, 'target_entries_before_seal': {'6': 0, '7': 0}, 'blocked_entries': 0}]
    events += [{'kind': 'branch_complete', 'row': name} for name in A['ORDER']]
    parent = {'data': tmp_path, 'binding': {'counts_before_seal': before},
        'summary': {'seal': seal, 'counts': A['COUNTS'], 'target_calls': {'6': 2, '7': 4}}}
    if corruption == 'target_entry':
        events[1]['target_entries_before_seal']['7'] = 1
    else:
        events.insert(1, events.pop(2))
    (tmp_path/'timeline.jsonl').write_text(''.join(json.dumps({**v, 'sequence': i, 'seconds': i})+'\n' for i, v in enumerate(events)))
    with pytest.raises(ValueError, match='seal'):
        A['audit_timing'](parent, {'counts': A['COUNTS']})
