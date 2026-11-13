"""Saved-only integrity and finite-response audit; no model/component calls."""
import argparse
import json
import runpy
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from mdb_lanes.edit_search import file_digest as sha
from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
from mdb_lanes.response_closure import audit_response,audit_factorial,measure_error
from mdb_lanes.tensor_archive import load_tensor_archive


def json_tree(value):
    if isinstance(value,Mapping): return {k:json_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [json_tree(v) for v in value]
    return value


def analyze(collection,precision,output):
    here=Path(__file__).resolve().parent; read=lambda p:json.loads(p.read_text())
    custody=runpy.run_path(str(here/'flux_family_custody.py'))['verify'](collection)
    data=collection/'extracted/binding-minimal-result'; request=read(collection/'request.json')
    config=read(data/'plan.json'); summary=read(data/'summary.json'); forecasts=read(data/'forecasts.json'); native=read(data/'native.json')
    if config!=request['config'] or summary!=read(collection/'report.json')['report']: raise ValueError('report/request mismatch')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('payload hash mismatch')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest: raise ValueError('payload wheel mismatch')
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256'])
    if not tx['verified'] or tx['phase']!='closed': raise ValueError('transaction not verified/closed')
    frozen=json_tree(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/response_closure.json')['predictions']))
    expected=dict(loads=1,native_transitions=2,acts=4,receipts=6,images=0,encodes=0,noise_draws=0)
    if frozen['forecasts']!=forecasts or frozen['entries_at_seal']!=dict(blocks=0) or frozen['counts_at_seal']!=dict(expected,native_transitions=0,acts=0,receipts=0): raise ValueError('pretarget seal mismatch')
    if summary['counts']!=expected or summary['entries']!=dict(blocks=2): raise ValueError('execution census mismatch')
    receipts=read(data/'receipts.json')
    if len(receipts)!=6 or [r['operation'] for r in receipts]!=['apply','apply','continue']*2: raise ValueError('receipt schedule mismatch')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py')); stores={name:typed['audit_store'](data/name) for name in ('store','source/store')}
    slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']; B=typed['quantize_bf16']; arrays_verified=set()
    def arr(record):
        value=load_tensor_archive(data,record); arrays_verified.add(record['path']); return value
    def payload(root,fp): return stores[root]['cuts'][fp]['payload']
    # Native historical anchors previously replayed exactly in the precision run.
    if read(precision/'verification-v1.json')['mechanics_status']!='verified': raise ValueError('historical precision study not verified')
    prior=precision/'collection-01/extracted/binding-minimal-result'; prior_native=read(prior/'native.json'); prior_forecasts=read(prior/'forecasts.json')
    base=config['parents']['base']; edited=config['parents']['edited']; T0=slot(data/'source/store',base,'text'); T1=slot(data/'source/store',edited,'text')
    local={}; corners={}; saved={}
    for index,spec in enumerate(config['cases']):
        name=spec['id']; anchor=spec['anchor']; n=native[name]; fields=forecasts[name]['fields']; saved[name]={}; local[name]={}
        if spec['dose']!=.375 or spec['receiver'] not in ('base','edited'): raise ValueError('unexpected fresh state')
        historical=config['history'][anchor]
        if historical['exit']!=prior_native[anchor]['exit'] or historical['entry']!=prior_native[anchor]['entry']: raise ValueError('historical policy replay join mismatch')
        chain=receipts[3*index:3*index+3]; parent=base
        for receipt in chain:
            if receipt['parent']!=parent or receipt!=read(data/'store/receipts'/(receipt['fingerprint']+'.json')): raise ValueError('durable lineage mismatch')
            parent=receipt['result']
        if chain[1]['result']!=n['entry'] or parent!=n['exit']: raise ValueError('native cut lineage mismatch')
        expected_text=T0.copy(); expected_text[:,[4,11]]=B(T0[:,[4,11]].astype(float)+spec['dose']*(T1[:,[4,11]].astype(float)-T0[:,[4,11]]))
        if not np.array_equal(slot(data/'store',n['entry'],'text'),expected_text): raise ValueError('new text construction mismatch')
        if not np.array_equal(slot(data/'store',n['entry'],'image'),slot(data/'source/store',config['parents'][spec['receiver']],'image')): raise ValueError('receiving image mismatch')
        def entries(p): return {v['key']['value']:v['value'] for v in p['items']}
        old,new=entries(payload('source/store',historical['entry'])),entries(payload('store',n['entry']))
        if any(old[k]!=new[k] for k in old if k not in {'image','text'}): raise ValueError('receiving closure mismatch')
        for role in ('image','text'):
            f=fields[role]; values={k:arr(v) for k,v in f.items() if isinstance(v,dict) and 'path' in v}; N1=arr(n['fields'][role]); N0=values['native_anchor']; P0=values['reference_anchor']; P1=values['absolute']
            if not np.array_equal(N1,slot(data/'store',n['exit'],role)) or not np.array_equal(N0,slot(data/'source/store',historical['exit'],role)): raise ValueError('native array/cut mismatch')
            if not np.array_equal(P0,load_tensor_archive(prior,prior_forecasts[anchor]['legacy']['fields'][role]['exit'])): raise ValueError('reference anchor did not reproduce original math')
            unrounded=N0.astype(float)+P1.astype(float)-P0.astype(float)
            if not np.array_equal(values['anchored'],B(unrounded)) or not np.array_equal(values['rounding_residual'],values['anchored'].astype(float)-unrounded): raise ValueError('anchoring/recast construction mismatch')
            if not np.array_equal(values['zero_response'],N0): raise ValueError('copy foil mismatch')
            modes=('absolute','anchored','zero_response','cross_receiver')
            local[name][role]=audit_response(N0,N1,P0,P1,candidates={k:values[k] for k in modes})
            saved[name][role]=dict(**values,native=N1)
    for name,other in zip(saved,reversed(saved)):
        for role in ('image','text'):
            f=saved[name][role]; g=saved[other][role]; crossed=f['native_anchor'].astype(float)+g['absolute'].astype(float)-g['reference_anchor'].astype(float)
            if not np.array_equal(f['cross_receiver'],B(crossed)) or not np.array_equal(f['cross_rounding_residual'],f['cross_receiver'].astype(float)-crossed): raise ValueError('cross-receiver foil mismatch')
    names=list(saved); factorial={}
    for role in ('image','text'):
        a,b=[saved[name][role] for name in names]; n=dict(zip(('00','01','10','11'),(a['native_anchor'],a['native'],b['native_anchor'],b['native'])))
        factorial[role]={}
        for mode in ('absolute','anchored','zero_response','cross_receiver'):
            p0=a['reference_anchor'] if mode=='absolute' else a['native_anchor']; p2=b['reference_anchor'] if mode=='absolute' else b['native_anchor']
            p=dict(zip(('00','01','10','11'),(p0,a[mode],p2,b[mode])))
            factorial[role][mode]=audit_factorial(n,p)
    timeline=[json.loads(line) for line in (data/'timeline.jsonl').read_text().splitlines()]
    kinds=[r['kind'] for r in timeline]; seal_index=kinds.index('sealed')
    if any(r['counts']['native_transitions'] or r['entries']['blocks'] for r in timeline[:seal_index+1]) or kinds.count('target_captured')!=2: raise ValueError('native target chronology mismatch')
    contract=read(data/'numerical-contract.json')
    if not contract['bf16_reduced_precision_reduction'] or contract['tf32'] or not contract['deterministic']: raise ValueError('numerical contract mismatch')
    result=dict(schema='mdb-response-closure-study-audit-v1',mechanics_status='verified',instrument_status='two complete native exits; no image/semantic consumer in this assay',terminal_status='unassessed',
        custody=custody,transaction=tx,counts=expected,external_array_files_verified=len(arrays_verified),local=local,factorial=factorial,numerical_contract=contract,
        worker_seconds=summary['worker_seconds'],auditor_source_sha256=sha(Path(__file__)),audit_module_sha256=sha(Path(__import__('mdb_lanes.response_closure',fromlist=['x']).__file__)),new_native_calls_in_audit=0,
        scope='Two previously unexecuted 3/8 internal text states, two known receivers/half-dose native anchors. P/A/copy/cross sealed before targets. Anchoring preserves unrounded dose/mixed errors; only stored recast changes them. No new semantic or wall-time speedup claim.')
    with output.open('x') as f: json.dump(result,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    return dict(mechanics_status='verified',local={name:{role:{mode:v['endpoint']['relative_rms'] for mode,v in row['candidates'].items()} for role,row in roles.items()} for name,roles in local.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('collection',type=Path); p.add_argument('precision',type=Path); p.add_argument('output',type=Path); a=p.parse_args()
    print(json.dumps(analyze(a.collection.resolve(),a.precision.resolve(),a.output.resolve()),sort_keys=True))
