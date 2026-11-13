"""Audit captured precision stages without executing any model components."""
import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from mdb.edit_search import file_digest
from mdb.forecast_transaction import verify_transaction, load_forecast_tree
from mdb.tensor_archive import load_tensor_archive


STAGES = ('norm1', 'modulated', 'q_projection', 'k_projection', 'v_projection',
          'q_normalized', 'k_normalized', 'q_positioned', 'k_positioned',
          'attention', 'after_attention', 'norm2', 'mlp_input', 'mlp_paired',
          'mlp_silu', 'mlp_activation', 'mlp', 'exit')


def error(predicted, native):
    if predicted.shape != native.shape:
        raise ValueError('comparison shape differs')
    ps = predicted.reshape(-1); ns = native.reshape(-1)
    ss = pp = nn = dot = 0.; maximum = 0.; different = 0
    for first in range(0, ns.size, 262144):
        p = ps[first:first+262144].astype(np.float64)
        n = ns[first:first+262144].astype(np.float64)
        if not np.isfinite(p).all() or not np.isfinite(n).all():
            raise ValueError('non-finite comparison')
        d = p-n
        ss += float(d@d); pp += float(p@p); nn += float(n@n); dot += float(p@n)
        maximum = max(maximum, float(np.max(np.abs(d))))
        different += int(np.count_nonzero(d))
    return dict(exact=different == 0, different=different, scalars=int(ns.size),
                rms=float(np.sqrt(ss/ns.size)), reference_rms=float(np.sqrt(nn/ns.size)),
                relative_rms=None if nn == 0 else float(np.sqrt(ss/nn)), max_abs=maximum,
                cosine=None if nn == 0 or pp == 0 else dot/np.sqrt(pp*nn))


def analyze(run, output):
    here = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text())
    custody = runpy.run_path(str(here/'flux_family_custody.py'))['verify'](run)
    data = run/'extracted/binding-minimal-result'
    request = read(run/'request.json'); config = read(data/'plan.json')
    summary = read(data/'summary.json'); forecasts = read(data/'forecasts.json')
    native = read(data/'native.json'); conditional = read(data/'conditional.json')
    worker = runpy.run_path(str(run/'payload/flux_precision_cut.py'))
    if config != request['config'] or summary != read(run/'report.json')['report']:
        raise ValueError('request/report binding differs')
    for name, digest in request['source_hashes'].items():
        if file_digest(run/'payload'/name) != digest:
            raise ValueError('payload source differs')
    for name, digest in read(run/'payload/payload-seal.json')['wheels'].items():
        if file_digest(run/'payload'/name) != digest:
            raise ValueError('installed wheel differs')
    tx = verify_transaction(data/'transaction', expected_seal_sha256=summary['seal']['seal_sha256'])
    if not tx['verified'] or tx['phase'] != 'closed':
        raise ValueError('transaction not closed and verified')
    tree = runpy.run_path(str(here/'analyze_flux_eye_program.py'))['json_tree']
    frozen = tree(load_forecast_tree(data/'transaction', read(data/'transaction/forecasts/local_precision.json')['predictions']))
    expected = worker['expected_counts']()
    if frozen['forecasts'] != forecasts or frozen['entries_at_seal'] != dict(blocks=0, dispatch=0):
        raise ValueError('forecast or pre-target seal differs')
    if frozen['counts_at_seal'] != dict(expected, loads=1, native_transitions=0, acts=0, receipts=0):
        raise ValueError('unexpected work before seal')
    if summary['counts'] != expected or summary['entries'] != dict(blocks=2, dispatch=2):
        raise ValueError('physical census differs')
    receipts = read(data/'receipts.json')
    if len(receipts) != 6 or [r['operation'] for r in receipts] != ['apply', 'apply', 'continue']*2:
        raise ValueError('native receipt schedule differs')
    typed = runpy.run_path(str(here/'analyze_flux_mlp_forecast.py'))
    store = typed['audit_store'](data/'store')
    source_store = typed['audit_store'](data/'source/store')
    readslot = runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']
    def array(record):
        return load_tensor_archive(data, record)
    def payload(fp, which):
        return (store if which == 'store' else source_store)['cuts'][fp]['payload']
    keys = {role+'.'+stage for role in ('image', 'text') for stage in STAGES} | {'attention.raw'}
    ordered = [role+'.'+stage for stage in STAGES for role in ('image', 'text')]
    ordered.insert(18, 'attention.raw')
    reference = runpy.run_path(str(run/'payload/flux_reentry_reference.py'))
    B = reference['bf16']
    local = {}; probes = {}; contrast = {}; pointwise = {}
    for name in ('half', 'half.image1'):
        for kind in ('entry', 'exit'):
            if payload(native[name][kind], 'store') != payload(config['history'][name][kind], 'source'):
                raise ValueError('historical full-state replay differs')
        for r in receipts[0:3] if name == 'half' else receipts[3:6]:
            if r != read(data/'store/receipts'/(r['fingerprint']+'.json')):
                raise ValueError('durable receipt differs')
        if set(native[name]['stages']) != keys:
            raise ValueError('incomplete native port set')
        for role in ('image', 'text'):
            if not np.array_equal(array(native[name]['stages'][role+'.exit']), readslot(data/'store', native[name]['exit'], role)):
                raise ValueError('exit port does not match typed cut')
        local[name] = {}
        for mode in ('legacy', 'staged'):
            if set(forecasts[name][mode]['stages']) != keys:
                raise ValueError('incomplete forecast port set')
            local[name][mode] = {key: error(array(forecasts[name][mode]['stages'][key]), array(native[name]['stages'][key])) for key in ordered}
            for role in ('image', 'text'):
                for field in ('attention', 'mlp_input', 'mlp', 'exit'):
                    if forecasts[name][mode]['fields'][role][field] != forecasts[name][mode]['stages'][role+'.'+field]:
                        raise ValueError('forecast port/field differs')
        probes[name] = {}
        for key, probe in conditional[name].items():
            measured = error(array(probe['output']), array(probe['native']))
            original = probe['error']
            for field in ('exact', 'different', 'scalars', 'max_abs'):
                if original[field] != measured[field]:
                    raise ValueError('conditional probe metadata differs')
            probes[name][key] = measured
        # Labelled diagnostic restoration: use observed operands to identify
        # operator contracts, never count these as independent predictions.
        pointwise[name] = {}
        context = readslot(data/'store', native[name]['entry'], 'context')
        text_rows = array(native[name]['stages']['text.norm1']).shape[1]
        for role in ('image', 'text'):
            stage = lambda key: array(native[name]['stages'][role+'.'+key])
            x = readslot(data/'store', native[name]['entry'], role)
            mod = np.split(context['img_mod' if role == 'image' else 'txt_mod'].reshape(1, 1, -1), 6, axis=-1)
            calculations = {
                'modulated': B(B(stage('norm1')*B(1+mod[1]))+mod[0]),
                'after_attention': B(x+B(mod[2]*stage('attention'))),
                'norm2': reference['_ln'](stage('after_attention'), 1e-6),
                'mlp_input': B(B(stage('norm2')*B(1+mod[4]))+mod[3]),
                'mlp_silu': reference['_silu'](np.split(stage('mlp_paired'), 2, axis=-1)[0]),
                'mlp_activation': B(stage('mlp_silu')*np.split(stage('mlp_paired'), 2, axis=-1)[1]),
                'exit': B(stage('after_attention')+B(mod[5]*stage('mlp'))),
            }
            sl = slice(0, text_rows) if role == 'text' else slice(text_rows, None)
            cos, sin = [a[sl][None, :, None, :] for a in context['rope']]
            for label in ('q', 'k'):
                v = stage(label+'_normalized'); rotated = np.empty_like(v)
                rotated[..., 0::2], rotated[..., 1::2] = -v[..., 1::2], v[..., 0::2]
                calculations[label+'_positioned'] = B(np.add(np.multiply(v, cos, dtype=np.float32), np.multiply(rotated, sin, dtype=np.float32), dtype=np.float32))
            pointwise[name][role] = {key:error(value, stage(key)) for key,value in calculations.items()}
    for mode in ('legacy', 'staged'):
        contrast[mode] = {}
        for key in ordered:
            p = array(forecasts['half.image1'][mode]['stages'][key])-array(forecasts['half'][mode]['stages'][key])
            n = array(native['half.image1']['stages'][key])-array(native['half']['stages'][key])
            contrast[mode][key] = error(p, n)
    timeline = [json.loads(line) for line in (data/'timeline.jsonl').read_text().splitlines()]
    if [r['kind'] for r in timeline] != ['calculated']*4+['sealed', 'target_captured', 'conditional_probes_complete', 'target_captured', 'conditional_probes_complete']:
        raise ValueError('unexpected execution chronology')
    result = dict(schema='mdb-precision-cut-audit-v1', mechanics_status='verified',
                  instrument_status='37 full native stages per case; exact historical input/output replay; observer parity and cleanup tested on tiny native block',
                  terminal_status='unassessed', custody=custody, transaction=tx,
                  auditor_source_sha256=file_digest(Path(__file__)), counts=expected,
                  native_contract=read(data/'native-contract.json'), stage_order=ordered,
                  local_errors=local, receiver_contrast=contrast, native_input_assisted_rms_probes=probes,
                  native_input_assisted_pointwise_probes=pointwise,
                  first_nonexact_stage={name:{mode:next((k for k in ordered if not local[name][mode][k]['exact']), None) for mode in ('legacy', 'staged')} for name in local},
                  scope='Historical diagnostic inputs and targets, newly observed intermediate ports. Candidate extra rounding sealed before these ports, not a held-out prediction. Conditional probes consume native projection operands and are diagnostic only.',
                  new_native_calls_in_audit=0, worker_seconds=summary['worker_seconds'])
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    return dict(mechanics_status='verified', first_nonexact=result['first_nonexact_stage'],
                receiver_exit_errors={mode:{role:contrast[mode][role+'.exit']['relative_rms'] for role in ('image', 'text')} for mode in contrast},
                native_input_rms_errors={name:{key:value['relative_rms'] for key,value in values.items()} for name,values in probes.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); print(json.dumps(analyze(a.run.resolve(), a.output.resolve()), sort_keys=True))
