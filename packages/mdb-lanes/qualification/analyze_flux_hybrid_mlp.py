"""Saved-array scoring of conditional MLP outputs; no parameter or model execution."""
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np
from mdb_lanes.tensor_archive import load_tensor_archive
from mdb_lanes.response_closure import audit_response,measure_error


def split(predicted,conditional,native):
    p,c,n=[np.asarray(v,float) for v in (predicted,conditional,native)];a=p-c;b=c-n;e=p-n
    return dict(inherited_operand_mse=float(np.mean(a*a)),conditional_law_mse=float(np.mean(b*b)),twice_signed_cross_moment=float(2*np.mean(a*b)),total_mse=float(np.mean(e*e)),identity_residual_rms=float(np.sqrt(np.mean((e-a-b)**2))),
        scope='Signed error decomposition under a conditional diagnostic. Cross term can cancel; components are not positive causal shares or image effects.')


def analyze(out,data,remote):
    read=lambda p:json.loads(p.read_text());records=read(out/'conditional.json');reconstruction=read(out/'reconstruction.json');plan,native,forecasts,baselines=[read(data/n) for n in ('plan.json','native.json','forecasts.json','baselines.json')];bits=runpy.run_path(str(Path(__file__).with_name('flux_attention_anchor.py')))['load_bits'];reports={};arrays=set()
    if set(records['anchors'])!={'half','half.image1'} or set(records['cases'])!={s['id'] for s in plan['cases']} or len(reconstruction)!=24:raise ValueError('MLP cohort differs')
    def arr(root,record):
        name=(root/record['path']).relative_to(data).as_posix()
        if remote['files'][name]['sha256']!=record['sha256']:raise ValueError('source array outside verified cohort')
        return load_tensor_archive(root,record)
    for by_role in [*records['anchors'].values(),*records['cases'].values()]:
        if set(by_role)!={'image','text'}:raise ValueError('conditional role census differs')
        for record in by_role.values():bits(out,record);arrays.add(record['path'])
    if {p.name for p in out.glob('*.npy')}!=arrays:raise ValueError('conditional archive census differs')
    for spec in plan['cases']:
        name=spec['id'];anchor=spec['anchor'];reports[name]={}
        for role in ('image','text'):
            c0=bits(out,records['anchors'][anchor][role]);c1=bits(out,records['cases'][name][role]);n0=arr(data/'source/prefixes',plan['anchors'][anchor]['stages'][role+'.mlp']);n1=arr(data,native[name]['stages'][role+'.mlp']);reports[name][role]=dict(conditional=audit_response(n0,n1,c0,c1),conditional_baseline=measure_error(c0,n0),programs={})
            for mode in ('control','hybrid'):
                p0=arr(data,baselines[anchor][mode]['fields'][role]['mlp']);p1=arr(data,forecasts[name][mode]['fields'][role]['mlp']);reports[name][role]['programs'][mode]=dict(endpoint=split(p1,c1,n1),response=split(p1.astype(float)-p0.astype(float),c1.astype(float)-c0.astype(float),n1.astype(float)-n0.astype(float)))
    order=['row11seven_eighths','row4seven_eighths','row11seven_eighths.image1','row4seven_eighths.image1'];mixed={}
    def contrast(values):return values[3].astype(float)-values[2].astype(float)-values[1].astype(float)+values[0].astype(float)
    for role in ('image','text'):
        nc=contrast([arr(data,native[name]['stages'][role+'.mlp']) for name in order]);cc=contrast([bits(out,records['cases'][name][role]) for name in order]);mixed[role]=dict(conditional=measure_error(cc,nc),programs={})
        for mode in ('control','hybrid'):
            pc=contrast([arr(data,forecasts[name][mode]['fields'][role]['mlp']) for name in order]);mixed[role]['programs'][mode]=split(pc,cc,nc)
    qualified=all(reconstruction.values())
    result=dict(schema='mdb-hybrid-mlp-audit-v1',mechanics_status='verified' if qualified else 'instrument-broken-owned-reconstruction',reports=reports,mixed=mixed,owned_reconstructions=reconstruction,logical_conditional_arrays=12,unique_arrays=len(arrays),source_custody=remote['verification'],auditor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_native_calls=0,scope='Actual target MLPinput supplied only to conditional diagnostic. Split endpoint, paired response and receiver-by-pattern mixed contrast; no fresh prediction or native intervention. Cross moments retained; no additive positive importance claim.')
    with (out/'audit-v1.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return result
