"""Saved-only numerical microscope audit; never turns a miss into a null."""
import argparse
import json
import runpy
from pathlib import Path


def measure(actual,predicted):
    import numpy as np
    a,p=np.asarray(actual,np.float64),np.asarray(predicted,np.float64)
    if a.shape!=p.shape:raise ValueError('comparison geometry differs')
    d=a-p;rms=lambda x:float(np.sqrt(np.mean(x*x)));an,pn=np.linalg.norm(a),np.linalg.norm(p)
    return dict(rms=rms(d),actual_rms=rms(a),predicted_rms=rms(p),relative_to_actual_rms=rms(d)/rms(a) if rms(a) else None,
        cosine=float(np.vdot(a,p)/(an*pn)) if an and pn else None,max_abs=float(abs(d).max()),
        mean_signed=float(d.mean()),exact_fraction=float(np.mean(a==p)),exact=bool(np.array_equal(a,p)))


class ArrayReference:
    """Lazy, checksum-verified archive access; never retains tensor payloads."""
    def __init__(self,loader,root,record):self.loader,self.root,self.record=loader,root,record
    def __array__(self,dtype=None,copy=None):
        value=self.loader(self.root,self.record)
        return value.astype(dtype,copy=False) if dtype is not None else value
    def astype(self,dtype,**kwargs):return self.loader(self.root,self.record).astype(dtype,**kwargs)


def audit(root,source,output,*,lazy_arrays=False):
    import numpy as np
    h=runpy.run_path(str(Path(__file__).with_name('flux_source_receiver.py')));read,sha,load=h['read'],h['sha'],h['load_array']
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('output evidence changed')
    c,s,r=[read(root/n) for n in ('plan.json','summary.json','request.json')]
    if r['config']!=c or r['request_sha256']!=s['request_sha256'] or sha(source/'files.json')!=c['source_files_sha256']:
        raise ValueError('source/request binding differs')
    for n,v in c['inputs'].items():
        if sha(source/n)!=v['sha256'] or (source/n).stat().st_size!=v['bytes']:raise ValueError('source input changed')
    if s['counts']!=c['planned_counts']:raise ValueError('execution census differs')
    cache={}
    raw_load=load
    def load(where,record):
        key=(record['sha256'],tuple(record['shape']),record['encoding'])
        if key not in cache:cache[key]=ArrayReference(raw_load,where,record) if lazy_arrays else raw_load(where,record)
        return cache[key]
    cases=read(root/'cases.json');parity={};errors={};values={};source_final={}
    for key,row in cases.items():
        actual={f:load(source,a) for f,a in row['source_observed'].items()}
        forecasts={f:load(source,a) for f,a in row['source_forecast'].items()}
        native={lane:{f:load(root,a) for f,a in fields.items()} for lane,fields in row['native'].items()}
        math={lane:{f:load(root,a) for f,a in fields.items()} for lane,fields in row['math'].items()}
        parity[key]={f:measure(actual[f],native['native-input'][f]) for f in ('after_attention','mlp','text')}
        for field in parity[key]:
            parity[key][field]['archive_bytes_exact']=row['source_observed'][field]['sha256']==row['native']['native-input'][field]['sha256']
        source_final[key]=measure(actual['text'],load(root,row['source_final']))
        errors[key]={}
        for prefix,lanes in [('native',native),('math',math)]:
            for lane,fields in lanes.items():
                reference=native['native-input'] if lane=='local' else {**native['native-input'],**actual}
                errors[key][prefix+'.'+lane]={f:measure(reference[f],p) for f,p in fields.items() if f in reference}
        errors[key]['previous-reference']={f:measure(actual[f],v) for f,v in forecasts.items()}
        errors[key]['math.local-vs-opposite']={f:measure(native['local-opposite'][f],math['local'][f]) for f in native['local-opposite']}
        values[key]=dict(observed=actual,previous=forecasts,
            **{'native.'+k:v for k,v in native.items()},**{'math.'+k:v for k,v in math.items()})
    effects={}
    for receiver in ('left','right'):
        effects[receiver]={}
        for lane in values[receiver+'.k0v0']:
            effects[receiver][lane]={}
            for field in values[receiver+'.k0v0'][lane]:
                if field in ('head','after_attention','mlp','text') and lane!='math.local':
                    targets={key:values[key]['observed'][field] for key in values if key.startswith(receiver+'.')}
                else:targets={key:values[key]['native.native-input'][field] for key in values if key.startswith(receiver+'.')}
                predicted={key:values[key][lane][field] for key in targets}
                def contrasts(v):
                    b,k,z,d=[v[f'{receiver}.k{ki}v{vi}'].astype(np.float64) for ki,vi in ((0,0),(1,0),(0,1),(1,1))]
                    return dict(key=k-b,value=z-b,interaction=d-k-z+b,total=d-b)
                ac,pc=contrasts(targets),contrasts(predicted)
                effects[receiver][lane][field]={name:measure(ac[name],pc[name]) for name in ac}
        effects[receiver]['math.local-vs-opposite']={}
        for field in ('projection','paired','mlp'):
            def contrast(lane):
                b,k,z,d=[values[f'{receiver}.k{ki}v{vi}'][lane][field].astype(np.float64) for ki,vi in ((0,0),(1,0),(0,1),(1,1))]
                return dict(key=k-b,value=z-b,interaction=d-k-z+b,total=d-b)
            ac,pc=contrast('native.local-opposite'),contrast('math.local')
            effects[receiver]['math.local-vs-opposite'][field]={name:measure(ac[name],pc[name]) for name in ac}
    qualified=all(p['archive_bytes_exact'] for row in parity.values() for p in row.values())
    result=dict(schema='mdb-reader-arithmetic-audit-v1',job=s['job_id'],mechanics_status='verified',
        instrument_status='component-replay-qualified' if qualified else 'component-replay-not-qualified',
        trend_status='observed',terminal_status='unassessed',counts=s['counts'],scope=c['scope'],native_contract=read(root/'native-contract.json'),
        original_computation_job=s.get('original_computation_job',s['job_id']),publication_job=s['job_id'],
        new_execution_counts=s.get('new_execution_counts'),array_loading='lazy' if lazy_arrays else 'eager',
        native_parity=parity,source_final_residual=source_final,errors=errors,effects=effects,
        output_manifest_sha256=sha(root/'files.json'),source_manifest_sha256=sha(source/'files.json'),auditor_sha256=sha(Path(__file__)),new_model_calls=0,
        limitations=['All targets are already observed; this is diagnosis, not new prospective validation.',
            'No attention kernel executes here. Earlier native head discrepancy remains outside this replay.',
            'Local operator errors and errors carried from earlier predictions are distinct measurements.',
            'Two known contexts at one clock/seed; semantic interpretation and generality unassessed.'])
    h['save'](output,result)
    return dict(job=s['job_id'],mechanics_status='verified',instrument_status=result['instrument_status'],
        full_source_parity=qualified,source_final_exact=all(x['exact'] for x in source_final.values()),
        text_total_effect_errors={receiver:{lane:fields['text']['total']['relative_to_actual_rms'] for lane,fields in effects[receiver].items() if 'text' in fields} for receiver in effects})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(audit(a.root.resolve(),a.source.resolve(),a.output.resolve()),sort_keys=True))
