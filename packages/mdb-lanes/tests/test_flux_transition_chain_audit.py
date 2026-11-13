"""Small independent algebra and hostile chain-provenance fixtures; no models."""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
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


def saved_chain(length=2):
    r, mod, block, kwargs = specimen()
    steps, current = [], r
    for index in range(length):
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
    return r, {'chain_length': length, 'input_descriptor': A['tensor_identity'](r), 'steps': steps,
        'output': current, 'output_descriptor': A['tensor_identity'](current),
        'text': current[:, :1].copy(), 'image': current[:, 1:].copy()}


def test_saved_chain_carries_both_full_streams_and_exact_output_alias():
    r, chain = saved_chain()
    outputs = A['verify_predicted_chain'](chain, r, text_rows=1, sample_indices=[0, 2])
    assert A['exact_bits'](outputs[0][:, [0, 2]], chain['steps'][1]['prediction']['sampled']['input'])
    assert A['exact_bits'](outputs[1], chain['output'])
    assert not A['exact_bits'](outputs[0][:, :1], r[:, :1])


def test_three_block_chain_has_no_native_refresh_and_default_remains_two_blocks():
    r, chain = saved_chain(3)
    outputs = A['verify_predicted_chain'](chain, r, text_rows=1, sample_indices=[0, 2], expected_block_indices=[6, 7, 8])
    assert len(outputs) == 3
    assert A['exact_bits'](outputs[1][:, [0, 2]], chain['steps'][2]['prediction']['sampled']['input'])
    with pytest.raises(ValueError, match='initial binding'):
        A['verify_predicted_chain'](chain, r, text_rows=1, sample_indices=[0, 2])
    chain['steps'][2]['input_descriptor'] = chain['steps'][0]['input_descriptor']
    with pytest.raises(ValueError, match='intermediate input'):
        A['verify_predicted_chain'](chain, r, text_rows=1, sample_indices=[0, 2], expected_block_indices=[6, 7, 8])


def test_geometry_is_independently_bounded_and_preserves_legacy_counts():
    old = A['derive_geometry']({'block_indices': [6, 7]})
    new = A['derive_geometry']({'block_indices': [6, 7, 8]})
    assert old['context'] == A['CONTEXT'] and old['branches'] == A['BRANCHES'] and old['order'] == A['ORDER']
    assert old['counts'] == A['COUNTS'] and old['panel_counts'] == A['PANEL_COUNTS']
    assert (new['context']['seed'], len(new['branches']), new['suffix_steps'], new['observer_boundaries'], new['common_key']) == (83, 9, 13, 14, 'common_post8')
    assert (new['counts']['adapter_steps'], new['counts']['acts'], new['counts']['receipt_emissions'], new['counts']['vae_calls']) == (237, 18, 255, 9)
    assert (new['panel_counts']['adapter_steps'], new['panel_counts']['acts'], new['panel_counts']['receipt_emissions']) == (129, 14, 143)
    assert new['targets'] == {'6': 2, '7': 4, '8': 6}
    assert new['order'] == ['low.native', 'high.native', 'low.sham', 'low.pred1', 'low.pred2', 'low.pred3', 'high.pred1', 'high.pred2', 'high.pred3']


@pytest.mark.parametrize('indices', [[7, 8], [6, 8], [6, 7, 9], [6, 7, 8, 9], [6., 7.]])
def test_unsupported_or_ambiguous_horizons_rejected(indices):
    with pytest.raises(ValueError, match='horizon'):
        A['derive_geometry']({'block_indices': indices})


def test_three_term_transport_identity_preserves_cancelled_text_and_image_effects():
    native = np.zeros((1, 2, 2), np.float32)
    bridge1 = np.array([[[2., -2.], [0., 0.]]], np.float32)
    bridge2 = np.array([[[1., -1.], [4., -4.]]], np.float32)
    predicted = np.array([[[0., 0.], [3., -3.]]], np.float32)
    result = A['three_term_decomposition'](predicted, bridge2, bridge1, native)
    text = A['three_term_decomposition'](predicted[:, :1], bridge2[:, :1], bridge1[:, :1], native[:, :1])
    image = A['three_term_decomposition'](predicted[:, 1:], bridge2[:, 1:], bridge1[:, 1:], native[:, 1:])
    assert result['closure_max_abs'] == text['closure_max_abs'] == image['closure_max_abs'] == 0
    assert len(result['pairwise_dot_products']) == 3
    assert text['total']['l2'] == 0 and all(v['l2'] > 0 for v in text['terms'].values())
    assert image['total']['l2'] > 0 and image['terms']['transported_block6_difference']['l2'] == 0
    assert text['pairwise_dot_products']['local_block8_dot_transported_block6_difference'] < 0
    assert 'native8 applied to an error vector' in result['scope']


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


def test_observer_full_reference_metric_abi_includes_signed_metrics_and_mae():
    actual = np.array([[[1., -3.]]], np.float32)
    reference = np.array([[[0., 1.]]], np.float32)
    metrics = A['retained_reference_metrics'](actual, reference)
    assert set(('l2', 'rms', 'mae', 'signed_mean', 'max_abs')) <= set(metrics)
    assert metrics['mae'] == 2.5 and metrics['signed_mean'] == -1.5
    inherited = A['field_error'](actual, reference)
    assert {k: metrics[k] for k in inherited} == inherited


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


@pytest.mark.parametrize('corruption', ['success', 'rerun', 'archive', 'payload', 'request'])
def test_failed_recovery_cannot_rewrite_success_or_payload_identity(corruption):
    request = {'request_sha256': 'a'*64}
    job = {'job_id': 'job-tiny', 'state': 'failed', 'result': {'returncode': 1, 'executed_payload': {'sha256': 'b'*64}}}
    recovery = {'schema': 'mdb-complete-result-failed-publication-recovery-v1', 'job_id': 'job-tiny',
        'scheduler_status': 'failed', 'model_calls_during_recovery': 0, 'original_job_and_log_preserved': True,
        'archive_sha256': None, 'request_sha256': 'a'*64, 'executed_payload_sha256': 'b'*64}
    A['verify_failed_recovery_identity'](recovery, job, request)
    if corruption == 'success':
        job['state'] = 'succeeded'
    elif corruption == 'rerun':
        recovery['model_calls_during_recovery'] = 1
    elif corruption == 'archive':
        recovery['archive_sha256'] = 'c'*64
    elif corruption == 'payload':
        recovery['executed_payload_sha256'] = 'c'*64
    else:
        recovery['request_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='failed job'):
        A['verify_failed_recovery_identity'](recovery, job, request)


def recovery_inventory(tmp_path):
    files = {'summary.json': '{}\n', 'transaction/seal.json': '{}\n', 'failure.json': '{"error":"transport"}\n'}
    for name, content in files.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'files.json').write_text(json.dumps({name: digest(tmp_path/name) for name in files}))
    inventory = {p.relative_to(tmp_path).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                 for p in tmp_path.rglob('*') if p.is_file()}
    before = {'schema': 'mdb-authenticated-remote-inventory-v1', 'files': inventory,
        'file_count': len(inventory), 'expanded_bytes': sum(v['bytes'] for v in inventory.values()),
        'max_member_bytes': max(v['bytes'] for v in inventory.values()),
        'manifest_sha256': digest(tmp_path/'files.json'), 'summary_sha256': digest(tmp_path/'summary.json'),
        'seal_sha256': digest(tmp_path/'transaction/seal.json')}
    limits = {'max_member_bytes': 4096, 'max_expanded_bytes': 16384, 'max_files': 20}
    return before, limits


def test_recovery_exact_manifest_includes_failed_export_metadata(tmp_path):
    before, limits = recovery_inventory(tmp_path)
    actual = A['verify_recovered_inventory'](tmp_path, before, copy.deepcopy(before), limits)
    assert set(actual) == {'summary.json', 'transaction/seal.json', 'failure.json', 'files.json'}


@pytest.mark.parametrize('corruption', ['changed_remote', 'extra', 'missing', 'tamper', 'symlink', 'member_bound'])
def test_recovery_rejects_partial_changed_aliased_or_unbounded_bytes(tmp_path, corruption):
    before, limits = recovery_inventory(tmp_path)
    after = copy.deepcopy(before)
    if corruption == 'changed_remote':
        after['summary_sha256'] = 'f'*64
    elif corruption == 'extra':
        (tmp_path/'unrecorded.npy').write_bytes(b'not-admitted')
    elif corruption == 'missing':
        (tmp_path/'summary.json').unlink()
    elif corruption == 'tamper':
        (tmp_path/'summary.json').write_text('{"new":1}')
    elif corruption == 'symlink':
        (tmp_path/'alias').symlink_to(tmp_path/'summary.json')
    else:
        limits['max_member_bytes'] = 3
    with pytest.raises(ValueError):
        A['verify_recovered_inventory'](tmp_path, before, after, limits)


def test_complete_failed_transport_recovery_keeps_failed_status_and_no_fake_report(tmp_path):
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    root = tmp_path/'run-02'
    data = root/'extracted/flux-transition-chain-result'
    data.mkdir(parents=True)
    save = lambda p, v: p.write_text(json.dumps(v, sort_keys=True)+'\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    request = {'source_hashes': {}, 'config': {}}
    request['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(root/'request.json', request)
    job = {'job_id': 'job-tiny', 'state': 'failed', 'config': {'request_sha256': request['request_sha256']},
        'result': {'returncode': 1, 'executed_payload': {'sha256': 'b'*64}}}
    save(root/'job.json', job)
    with ForecastTransaction(data/'transaction', model_identity='tiny', input_identity='tiny', source_identity='tiny',
                             frontier={'kind': 'architectural'}) as tx:
        tx.predict('tiny', lambda _: {'predictions': {'value': np.array([1.], np.float32)}})
        seal = tx.seal()
        tx.execute('native', lambda _: {'fixture_only': True})
    save(data/'transaction-report.json', tx.report())
    summary = {'job_id': 'job-tiny', 'request_sha256': request['request_sha256'], 'counts': A['COUNTS'],
        'panel_counts': A['PANEL_COUNTS'], 'seal': seal, 'transaction_verification': verify_transaction(data/'transaction')}
    save(data/'summary.json', summary)
    save(data/'failure.json', {'job_id': 'job-tiny', 'request_sha256': request['request_sha256'],
        'mechanics_status': 'errored', 'error': 'CustodyError: archive member exceeds byte bound'})
    save(data/'files.json', {p.relative_to(data).as_posix(): sha(p) for p in data.rglob('*') if p.is_file()})
    inventory = {p.relative_to(data).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in data.rglob('*') if p.is_file()}
    remote = {'schema': 'mdb-authenticated-remote-inventory-v1', 'job_id': 'job-tiny',
        'remote_path': '/remote/job-tiny/flux-transition-chain-result/', 'files': inventory,
        'file_count': len(inventory), 'expanded_bytes': sum(v['bytes'] for v in inventory.values()),
        'max_member_bytes': max(v['bytes'] for v in inventory.values()), 'manifest_sha256': sha(data/'files.json'),
        'summary_sha256': sha(data/'summary.json'), 'seal_sha256': sha(data/'transaction/seal.json')}
    for name in ('recovery-remote-before.json', 'recovery-remote-after.json'):
        save(root/name, remote)
    save(root/'recovery-local-inventory.json', {k: remote[k] for k in ('files', 'file_count', 'expanded_bytes')})
    marker = 'MDB_FLUX_TRANSITION_CHAIN_PHASE '
    events = [{'kind': 'transition_chain_sealed', 'sequence': 0, 'seal': seal},
        {'kind': 'complete', 'sequence': 1, 'counts': A['COUNTS']}]
    raw = (''.join(marker+json.dumps(v)+'\n' for v in events)+'CustodyError: archive member exceeds byte bound\n').encode()
    compressed = gzip.compress(raw, mtime=0)
    (root/'worker.log.gz').write_bytes(compressed)
    save(root/'log-custody.json', {'schema': 'mdb-collector-log-custody-v1', 'storage': 'gzip',
        'compressed_bytes': len(compressed), 'uncompressed_bytes': len(raw),
        'compressed_sha256': hashlib.sha256(compressed).hexdigest(), 'uncompressed_sha256': hashlib.sha256(raw).hexdigest()})
    for name in ('recover_result.py', 'remote_inventory.py'):
        (tmp_path/name).write_text('# fixture procedure; no model\n')
    recovery = {'schema': 'mdb-complete-result-failed-publication-recovery-v1', 'job_id': 'job-tiny',
        'scheduler_status': 'failed', 'model_calls_during_recovery': 0, 'original_job_and_log_preserved': True,
        'archive_sha256': None, 'request_sha256': request['request_sha256'], 'executed_payload_sha256': 'b'*64,
        'worker_report_emitted': False, 'remote_path': remote['remote_path'],
        'recovery_limits': {'max_member_bytes': 256 << 20, 'max_expanded_bytes': 3072 << 20, 'max_files': 20000}}
    for key, path in (('remote_before_sha256', root/'recovery-remote-before.json'),
        ('remote_after_sha256', root/'recovery-remote-after.json'), ('local_inventory_sha256', root/'recovery-local-inventory.json'),
        ('remote_manifest_sha256', data/'files.json'), ('remote_summary_sha256', data/'summary.json'),
        ('remote_seal_sha256', data/'transaction/seal.json'), ('recovery_script_sha256', tmp_path/'recover_result.py'),
        ('remote_inventory_script_sha256', tmp_path/'remote_inventory.py')):
        recovery[key] = sha(path)
    save(root/'recovery.json', recovery)
    h = runpy.run_path(str(HERE/'summarize_context_window.py'))
    custody, recovered = A['recovered_custody'](root, h, request, job)
    assert recovered == summary and custody['mechanics_status'] == 'verified'
    assert custody['scheduler_status'] == job['state'] == 'failed'
    assert custody['archive_sha256'] is None and custody['worker_report_emitted'] is False
    assert not (root/'report.json').exists()
