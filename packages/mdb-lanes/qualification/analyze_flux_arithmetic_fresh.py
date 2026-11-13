"""Verify prospective arithmetic forecasts and split the raw-attention defect."""
import argparse
import json
import runpy
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from mdb_lanes.edit_search import file_digest as sha
from mdb_lanes.tensor_archive import load_tensor_archive
from mdb_lanes.forecast_transaction import verify_transaction,load_forecast_tree
from mdb_lanes.response_closure import measure_error,audit_factorial


def plain(x):
    if isinstance(x,Mapping): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x


def compare(p,n):
    p,n=np.asarray(p,float),np.asarray(n,float); value=measure_error(p,n); pr=float(np.sqrt(np.mean(p*p)))
    return dict(**value,predicted_rms=pr,predicted_over_native_norm=None if value['reference_rms']==0 else pr/value['reference_rms'])


def split_error(predicted,own_native,true_native):
    p,o,n=[np.asarray(x,float) for x in (predicted,own_native,true_native)]; local=p-o; inherited=o-n; total=p-n
    if not np.array_equal(local+inherited,total): raise ValueError('error split identity differs')
    power=lambda x:float(np.mean(x*x)); cross=float(2*np.mean(local*inherited))
    return dict(total=compare(p,n),local_defect_rms=float(np.sqrt(power(local))),inherited_operand_rms=float(np.sqrt(power(inherited))),
        mean_square=dict(total=power(total),local=power(local),inherited=power(inherited),twice_signed_cross=cross,reconstructed=power(local)+power(inherited)+cross))


def analyze(collection,precision,transfer_audit,output):
    here=Path(__file__).resolve().parent; read=lambda p:json.loads(p.read_text()); verify=runpy.run_path(str(here/'flux_family_custody.py'))['verify']
    custody=dict(fresh=verify(collection),precision=verify(precision)); data=collection/'extracted/binding-minimal-result'; source=precision/'extracted/binding-minimal-result'
    request=read(collection/'request.json'); c=request['config']; summary=read(data/'summary.json')
    if c!=read(data/'plan.json') or summary!=read(collection/'report.json')['report'] or sha(transfer_audit)!=c['source_transfer_audit_sha256']: raise ValueError('plan/report/transfer binding differs')
    qualification=read(transfer_audit)
    if qualification['mechanics_status']!='verified' or qualification['policy']!=c['policy']: raise ValueError('selected qualification/policy differs')
    if any(qualification['custody']['precision'][key]!=custody['precision'][key] for key in ('job_id','archive_sha256','request_sha256')): raise ValueError('transfer source calibration differs')
    for n,digest in request['source_hashes'].items():
        if sha(collection/'payload'/n)!=digest: raise ValueError('worker source differs')
    for n,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/n)!=digest: raise ValueError('installed wheel differs')
    for name,digest in c['inputs'].items():
        if sha(source/name)!=digest: raise ValueError('source operand differs')
    worker=runpy.run_path(str(collection/'payload/flux_arithmetic_fresh.py'))
    if summary['counts']!=worker['counts_expected']() or summary['entries']!=dict(blocks=4,dispatch=4): raise ValueError('physical work differs')
    profile=read(data/'native-profile.json')
    if profile['bf16_reduced'] is not True or profile['tf32'] is not False: raise ValueError('numerical profile differs')
    tx=verify_transaction(data/'transaction',expected_seal_sha256=summary['seal']['seal_sha256'])
    if not tx['verified'] or tx['phase']!='closed': raise ValueError('transaction not verified/closed')
    frozen=plain(load_forecast_tree(data/'transaction',read(data/'transaction/forecasts/complete_writes.json')['predictions']))
    forecasts=read(data/'forecasts.json'); zeros=read(data/'zero-measurements.json'); native=read(data/'native.json'); controls=read(data/'controls.json'); receipts=read(data/'receipts.json')
    if frozen['forecasts']!=forecasts or frozen['zeros']!=zeros or frozen['entries_at_seal']!=dict(blocks=0,dispatch=0): raise ValueError('forecast changed or native entries before seal')
    if frozen['counts_at_seal']!=dict.fromkeys(worker['counts_expected'](),0)|dict(loads=1): raise ValueError('unexpected preseal native work')
    timeline=[json.loads(line) for line in (data/'timeline.jsonl').read_text().splitlines()]; seal_at=next(i for i,row in enumerate(timeline) if row['kind']=='sealed')
    if sum(row['kind']=='predicted' for row in timeline[:seal_at])!=4 or any(row['entries']['blocks'] or row['entries']['dispatch'] for row in timeline[:seal_at+1]): raise ValueError('target chronology differs')
    typed=runpy.run_path(str(here/'analyze_flux_mlp_forecast.py')); store=typed['audit_store'](data/'store'); slot=runpy.run_path(str(here/'analyze_flux_binding_minimal.py'))['read_slot']; B=typed['quantize_bf16']; public=runpy.run_path(str(here/'summarize_context_window.py'))['public_receipt']
    if len(receipts)!=12 or sum(r['body'].get('steps',0) for r in receipts)!=4: raise ValueError('receipt census differs')
    for row in receipts:
        public(row['body'])
        if read(data/'store/receipts'/(row['body']['fingerprint']+'.json'))!=row['body']: raise ValueError('durable receipt differs')
    def fields(fp):return {x['key']['value']:x['value'] for x in store['cuts'][fp]['payload']['items']}
    prior=read(source/'native.json'); arrays=set()
    def array(record):
        value=load_tensor_archive(data,record)
        if not np.isfinite(value).all() or not np.array_equal(value,B(value)): raise ValueError('nonfinite/nonBF16 archive')
        arrays.add(record['path']); return value
    T0,T1=[slot(data/'store',c['parents'][key],'text') for key in ('base','edited')]; state={}; qkv={}; attention={}; exit_n={}; exit_p={}; raw={}; raw_n={}; raw_o={}; raw_g={}
    for spec in c['cases']:
        name,anchor=spec['id'],spec['anchor']; n=native[name]; ctrl=controls[name]; chain=[r['body'] for r in receipts if r['case']==name]; parent=c['parents']['base']
        for row in chain:
            if row['parent']!=parent: raise ValueError('native branch disconnected')
            parent=row['result']
        if parent!=n['exit'] or [r['operation'] for r in chain]!=['apply','apply','continue'] or chain[2]['parent']!=n['entry'] or chain[2]['steps']!=1: raise ValueError('numerical exit not actual target successor')
        point=typed['unpack'](store['cuts'][n['exit']]['execution_point'])
        if any(point[k]!=v for k,v in dict(logical_step=2,phase='joint',index=2).items()): raise ValueError('exit clock differs')
        if any(store['cuts'][n['entry']][key]!=store['cuts'][c['parents']['base']][key] for key in ('model_identity','execution_point')): raise ValueError('entry model/clock differs')
        basefields=fields(c['parents']['base']); entryfields=fields(n['entry'])
        if any(basefields[k]!=entryfields[k] for k in basefields if k not in {'image','text'}): raise ValueError('native closure changed')
        expected=T0.copy()
        for index,dose in zip((4,11),spec['row_doses']): expected[:,index]=B(T0[:,index].astype(float)+dose*(T1[:,index].astype(float)-T0[:,index].astype(float)))
        if not np.array_equal(expected,slot(data/'store',n['entry'],'text')) or not np.array_equal(slot(source/'store',prior[anchor]['entry'],'image'),slot(data/'store',n['entry'],'image')): raise ValueError('fresh input differs')
        if len(n['stages'])!=19 or len(n['unchanged_image_prefix'])!=8 or any(record!=prior[anchor]['stages'][key] for key,record in n['unchanged_image_prefix'].items()): raise ValueError('retained native port coverage differs')
        if ctrl['dispatch_metadata']!=n['dispatch_metadata'] or ctrl['dispatch_arguments']!=n['dispatch_arguments']: raise ValueError('control attention contract differs')
        for role in ('image','text'):
            if not typed['matches'](array(n['stages'][role+'.exit']),fields(n['exit'])[role],bf16=True): raise ValueError('native array disconnected from typed exit')
        state[name]={}; exit_n[name]={}; exit_p[name]={}
        for mode,forecast in forecasts[name].items():
            state[name][mode]={}; exit_p[name][mode]={}
            for role,records in forecast['fields'].items():
                state[name][mode][role]={}
                for field,record in records.items():
                    p=array(record); actual=array(n['stages'][role+'.'+field]); n0=load_tensor_archive(source,prior[anchor]['stages'][role+'.'+field])
                    state[name][mode][role][field]=dict(endpoint=compare(p,actual),installed_edit=compare(p.astype(float)-n0.astype(float),actual.astype(float)-n0.astype(float)))
                    if field=='exit':exit_n[name][role]=actual;exit_p[name][mode][role]=p
        qkv[name]={}; new=forecasts[name]['arithmetic']; other=[i for i in range(512) if i not in (4,11)]
        for key in ('modulated','q_projection','k_projection','v_projection','q_normalized','k_normalized','q_positioned','k_positioned'):
            p=array(new['stages']['text.'+key]); actual=array(n['stages']['text.'+key]); qkv[name][key]=dict(full=compare(p,actual),changed_rows=compare(p[:,[4,11]],actual[:,[4,11]]))
            if key in ('q_positioned','k_positioned','v_projection') and not np.array_equal(p[:,other],load_tensor_archive(source,prior[anchor]['stages']['text.'+key])[:,other]): raise ValueError('fallback effective operands differ from partition')
        for label in ('q','k','v'):
            p=array(ctrl['conditional_qkv'][label]); actual=array(n['stages']['text.'+label+'_projection']); qkv[name][label+'_conditional_same_input']=dict(full=compare(p,actual),changed_rows=compare(p[:,[4,11]],actual[:,[4,11]]),exact_fraction=float(np.mean(p==actual)))
        raw[name]=array(new['stages']['attention.raw']); raw_n[name]=array(n['stages']['attention.raw']); raw_o[name]=array(ctrl['native_own_attention']); raw_g[name]=array(ctrl['wide_attention_native_operands'])
        attention[name]=dict(split=split_error(raw[name],raw_o[name],raw_n[name]),wide_on_true_operands=compare(raw_g[name],raw_n[name]))
    order=['row11five_eighths','row4five_eighths','row11five_eighths.image1','row4five_eighths.image1']; coeff=dict(zip(order,(1,-1,-1,1)))
    mix=lambda values:sum(values[name].astype(float)*sign for name,sign in coeff.items()); factorial={}
    for mode in ('local','arithmetic'):
        factorial[mode]={role:audit_factorial(dict(zip(('00','01','10','11'),(exit_n[name][role] for name in order))),dict(zip(('00','01','10','11'),(exit_p[name][mode][role] for name in order)))) for role in ('image','text')}
    mixed=dict(split=split_error(mix(raw),mix(raw_o),mix(raw_n)),wide_on_true_operands=compare(mix(raw_g),mix(raw_n)),coefficients=coeff,
        native_rms=float(np.sqrt(np.mean(mix(raw_n)**2))),own_native_rms=float(np.sqrt(np.mean(mix(raw_o)**2))),prediction_rms=float(np.sqrt(np.mean(mix(raw)**2))))
    result=dict(schema='mdb-fresh-affine-arithmetic-audit-v1',mechanics_status='verified',instrument_status='Complete forecast seals, source operands, native typed lineage and same-operand attention-control contract verified; scalar primitive counts source-instrumented',terminal_status='unassessed',
        custody=custody,transaction=tx,counts=summary['counts'],state=state,qkv=qkv,attention=attention,mixed_attention=mixed,factorial=factorial,zeros=zeros,arrays_verified=len(arrays),worker_seconds=summary['worker_seconds'],source_sha256=sha(Path(__file__)),new_native_calls_in_audit=0,
        limitations='Installed edit compares P1−N0 with N1−N0; arithmetic baseline P0 is not exact and only its error measurements were retained, so this is not a mathematical P1−P0 response audit. Four native attention controls are explicit native diagnostics. Native true-operand ports never enter preseal forecasts. No new image/semantic or speed claim.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',state={n:{m:{r:records['exit']['installed_edit']['relative_rms'] for r,records in rows.items()} for m,rows in modes.items()} for n,modes in state.items()},mixed=mixed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','precision','transfer_audit','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.collection.resolve(),a.precision.resolve(),a.transfer_audit.resolve(),a.output.resolve()),sort_keys=True))
