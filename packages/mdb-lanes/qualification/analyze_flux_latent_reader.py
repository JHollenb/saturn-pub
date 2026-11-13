"""Frozen saved-only audit of latent closure and prospective text-reader writes."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def audit(root,output):
    import numpy as np
    from PIL import Image
    from saturn_pub.values import canonical

    from mdb_lanes.forecast_transaction import verify_transaction
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')))
    read,sha,load=h['read'],h['sha'],h['load_array']
    manifest=read(root/'files.json')
    if any(sha(root/n)!=v for n,v in manifest.items()):raise ValueError('changed evidence file')
    s,c,r=[read(root/n) for n in ('summary.json','plan.json','request.json')]
    cohort=read(root/'cohort.json');prefix=read(root/'prefixes.json');closure=read(root/'closure.json');readers=read(root/'readers.json')
    known=read(root/'known-calibrations.json')['records']
    if r['config']!=c or r['request_sha256']!=s['request_sha256'] or sha(root/'cohort.json')!=s['cohort_sha256']:
        raise ValueError('request/cohort binding differs')
    expected=dict(prefix_stops=4,mathematical_text_predictions=8,native_transitions=440,completed_blocks=400,
        block_entries=404,native_sdpa=400,latent_acts=8,images=16,model_loads=1,encodes=0,noise_draws=0)
    if s['counts']!=expected:raise ValueError('physical census differs')
    if cohort['counts']['native_sdpa'] or cohort['counts']['completed_blocks'] or cohort['counts']['mathematical_text_predictions']!=8:
        raise ValueError('native target preceded complete forecast cohort')
    if prefix['counts']['prefix_stops']!=4 or prefix['counts']['native_sdpa']:raise ValueError('prefix consumer completed')
    def metric(a,b):
        a,b=np.asarray(a,np.float64),np.asarray(b,np.float64);d=a-b
        rms=lambda x:float(np.sqrt(np.mean(x*x)))
        an,bn=float(np.linalg.norm(a)),float(np.linalg.norm(b))
        return dict(rms=rms(d),mae=float(np.mean(abs(d))),max_abs=float(abs(d).max()),
            actual_rms=rms(a),reference_rms=rms(b),relative_to_actual_rms=rms(d)/rms(a) if rms(a) else None,
            cosine=float(np.vdot(a,b)/(an*bn)) if an and bn else None,mean_signed=float(d.mean()),exact=bool(np.array_equal(a,b)))
    errors={};factorial={};identities={};controls={};futures={};pred={};actual={}
    def rgb(record):
        if sha(root/record['path'])!=record['sha256']:raise ValueError('PNG identity differs')
        return np.asarray(Image.open(root/record['path']).convert('RGB'),np.float64)
    for receiver in ('left','right'):
        b,d=[{k:load(root,v) for k,v in prefix['operands'][receiver+'.'+name].items()} for name in ('base','delta')]
        nt=b['q'].shape[1]
        for name in ('base','delta','restore','transplant'):
            key=receiver+'.'+name;row=closure[key];target=closure[row['target']]
            if len(row['records'])!=28:raise ValueError('incomplete closure')
            equal=row['final_payload']==target['final_payload'] and row['image']['sha256']==target['image']['sha256']
            equal=equal and all(x['outputs']==y['outputs'] for x,y in zip(row['records'],target['records']))
            controls[key]=dict(full_future_exact=equal,target=row['target'],known_png_exact=row['image']['sha256']==known[row['target']]['image']['sha256'])
        for ki,vi in ((0,0),(1,0),(0,1),(1,1)):
            key=f'{receiver}.k{ki}v{vi}';row=readers[key]
            verify_transaction(root/'transactions'/key,expected_seal_sha256=cohort['seals'][key]['seal_sha256'])
            if len(row['records'])!=27:raise ValueError('incomplete reader continuation')
            k,v=b['k'].copy(),b['v'].copy()
            if ki:k[:,nt:]=d['k'][:,nt:]
            if vi:v[:,nt:]=d['v'][:,nt:]
            footprint=row['port_intervention']
            for name,value in [('q_text',b['q']),('key_after',k),('value_after',v)]:
                if hashlib.sha256(value.tobytes()).hexdigest()!=footprint[name+'_fp32_sha256']:
                    raise ValueError('actual port intervention differs from declared frozen operands')
            pred[key]={f:load(root,a) for f,a in cohort['predictions'][key].items()}
            actual[key]={f:load(root,a) for f,a in row['fields'].items()}
            errors[key]={f:metric(actual[key][f],pred[key][f]) for f in actual[key]}
            if ki==vi:
                target=receiver+('.delta' if ki else '.base')
                identities[key]=metric(actual[key]['text'],load(root,known[target]['text']))
            futures[key]=dict(rgb_vs_baseline=metric(rgb(row['image']),rgb(closure[receiver+'.base']['image'])),
                text_at_joint4=metric(load(root,row['records'][4]['outputs']['text']['archive']),
                    load(root,closure[receiver+'.base']['records'][5]['outputs']['text']['archive'])),
                latent_vs_baseline=metric(load(root,row['records'][-1]['outputs']['latent']['archive']),
                    load(root,closure[receiver+'.base']['records'][-1]['outputs']['latent']['archive'])))
        factorial[receiver]={}
        for field in ('head','after_attention','mlp','text'):
            corners=lambda values:[values[f'{receiver}.k{k}v{v}'][field].astype(np.float64) for k,v in ((0,0),(1,0),(0,1),(1,1))]
            p,a=corners(pred),corners(actual)
            contrasts=lambda x:dict(key=x[1]-x[0],value=x[2]-x[0],interaction=x[3]-x[1]-x[2]+x[0],total=x[3]-x[0])
            pc,ac=contrasts(p),contrasts(a)
            factorial[receiver][field]={n:metric(ac[n],pc[n]) for n in ac}
    receipts=[json.loads(line) for line in (root/'receipts.jsonl').read_text().splitlines()]
    if len(receipts)!=452:raise ValueError('receipt census differs')
    for row in receipts:
        receipt=dict(row['receipt']);fp=receipt.pop('fingerprint')
        if hashlib.sha256(canonical(receipt)).hexdigest()!=fp:raise ValueError('receipt content seal differs')
    result=dict(schema='mdb-latent-reader-audit-v1',mechanics_status='verified',instrument_status='applicable',
        trend_status='observed',terminal_status='unassessed',job=s['job_id'],counts=s['counts'],scope=c['scope'],
        errors=errors,factorial=factorial,controls=controls,text_identity_observations=identities,observed_futures=futures,
        receipt_rows=len(receipts),native_manifest_sha256=sha(root/'files.json'),auditor_sha256=sha(Path(__file__)),new_model_calls=0,
        limitations=['Observed native QKV prefix; not raw-prompt prediction.',
            'Numerical forecasts end at first text exit; subsequent images are observed native futures.',
            'Two known contexts, one seed, one clock; semantic object labels remain unassigned.',
            'Closure controls reproduce complete live state; they are not independent semantic predictions.'])
    h['save'](output,result)
    return dict(job=s['job_id'],mechanics_status='verified',controls=controls,
        text_relative_errors={k:v['text']['relative_to_actual_rms'] for k,v in errors.items()},
        text_identity_exact={k:v['exact'] for k,v in identities.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.root.resolve(),a.output.resolve()),sort_keys=True))
