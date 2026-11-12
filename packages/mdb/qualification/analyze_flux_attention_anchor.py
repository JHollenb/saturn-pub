"""Independently join and compare retained attention-boundary predictions."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive
from mdb.response_closure import measure_error,audit_factorial


def analyze(collection,precision,fresh,normalization,source_audit,output):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());verify=runpy.run_path(str(here/'flux_family_custody.py'))['verify'];custody=verify(collection)
    data=collection/'extracted/binding-minimal-result';roots={k:p/'extracted/binding-minimal-result' for k,p in [('prefix',precision),('fresh',fresh),('normalization',normalization)]};request=read(collection/'request.json');c=request['config'];summary=read(data/'summary.json')
    if summary!=read(collection/'report.json')['report'] or c!=read(data/'plan.json'):raise ValueError('report/plan mismatch')
    if sha(source_audit)!=c['source_normalization_audit_sha256'] or read(source_audit)['mechanics_status']!='verified':raise ValueError('source audit mismatch')
    qualified=read(source_audit);witness=read(collection/'payload/source-qualification.json')
    if witness['audit_sha256']!=sha(source_audit):raise ValueError('source qualification witness differs')
    for key,known in [('normalization',qualified['custody']),('fresh',qualified['source_custody']['fresh']),('precision',qualified['source_custody']['precision'])]:
        if any(witness['custody'][key][k]!=known[k] for k in ('job_id','request_sha256','archive_sha256')):raise ValueError('source custody binding differs')
    for key,files in c['inputs'].items():
        inventory=read(roots[key]/'files.json')
        for name,digest in files.items():
            if inventory[name]!=digest or sha(roots[key]/name)!=digest:raise ValueError('consumed source differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('sealed worker source differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('installed wheel differs')
    helper=runpy.run_path(str(collection/'payload/flux_attention_anchor.py'));read_bits=helper['load_bits'];native=read(roots['prefix']/'native.json');actual=read(roots['fresh']/'native.json');predictions=read(data/'predictions.json');arrays=set();global_reports={};streams={};pvalues={};nvalues={};zero={}
    if set(predictions['baselines'])!={'half','half.image1'} or len(predictions['cases'])!=4:raise ValueError('array census differs')
    def array(item):arrays.add(item['path']);return read_bits(data,item)
    for anchor,records in predictions['baselines'].items():
        zero[anchor]={m:array(v) for m,v in records.items()};n0=load_tensor_archive(roots['prefix'],native[anchor]['stages']['attention.raw'])
        if not np.array_equal(zero[anchor]['anchored'],n0):raise ValueError('baseline anchor identity differs')
    for case in c['cases']:
        name,anchor=case['id'],case['anchor'];n=load_tensor_archive(roots['fresh'],actual[name]['stages']['attention.raw']);n0=load_tensor_archive(roots['prefix'],native[anchor]['stages']['attention.raw']);nvalues[name]=n;global_reports[name]={};streams[name]={}
        for scope,methods in predictions['cases'][name].items():
            global_reports[name][scope]={};streams[name][scope]={};pvalues.setdefault(scope,{})[name]={}
            for method,item in methods.items():
                p=array(item);p0=zero[anchor][method];pvalues[scope][name][method]=p
                def metrics(sl):
                    x,y,b,b0=p[:,sl].astype(float),n[:,sl].astype(float),p0[:,sl].astype(float),n0[:,sl].astype(float)
                    return dict(endpoint=measure_error(x,y),installed_edit=measure_error(x-b0,y-b0),boundary_response=measure_error(x-b,y-b0))
                global_reports[name][scope][method]=metrics(slice(None));streams[name][scope][method]={role:metrics(sl) for role,sl in [('text',slice(0,512)),('image',slice(512,None))]}
    order=['row11five_eighths','row4five_eighths','row11five_eighths.image1','row4five_eighths.image1'];keys=('00','01','10','11');factorial={};stream_factorial={}
    for scope in pvalues:
        factorial[scope]={};stream_factorial[scope]={}
        for method in ('absolute','anchored'):
            factorial[scope][method]=audit_factorial(dict(zip(keys,(nvalues[n] for n in order))),dict(zip(keys,(pvalues[scope][n][method] for n in order))))
            stream_factorial[scope][method]={role:audit_factorial(dict(zip(keys,(nvalues[n][:,sl] for n in order))),dict(zip(keys,(pvalues[scope][n][method][:,sl] for n in order)))) for role,sl in [('text',slice(0,512)),('image',slice(512,None))]}
    if global_reports!=summary['reports'] or factorial!=summary['factorial']:raise ValueError('recomputed metrics differ')
    if len(arrays)!=20 or summary['counts']!=dict(model_loads=0,native_calls=0,weights_loaded=0,states=4,methods=2,operand_scopes=2,predictions=16,baselines=4):raise ValueError('execution census differs')
    result=dict(schema='mdb-attention-anchor-development-audit-v1',mechanics_status='verified',instrument_status='Source/custody bindings,20arrays, native-anchor equality and all retained-array comparisons verified; law execution source-qualified',terminal_status='unassessed',custody=custody,source_audit_sha256=sha(source_audit),reports=global_reports,streams=streams,factorial=factorial,stream_factorial=stream_factorial,counts=summary['counts'],arrays_verified=len(arrays),worker_seconds=summary['seconds'],source_sha256=sha(Path(__file__)),new_native_calls_in_audit=0,
        limitations='Known-state development. Own corrected QKV is independently calculated but still imperfect; conditional-native-operands is explicitly target assisted. Baseline is the supplied native calibration QKV, not corrected-prefix full-program P0. Previous native-own controls have different operands and are not reused. No fresh native controls, full downstream forecast or semantic/image/speed qualification.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',arrays_verified=len(arrays),reports=global_reports,mixed={s:{m:v['contrasts']['receiver_by_dose']['relative_rms'] for m,v in methods.items()} for s,methods in factorial.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','precision','fresh','normalization','source_audit','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(analyze(*[getattr(a,n).resolve() for n in ('collection','precision','fresh','normalization','source_audit','output')]),sort_keys=True))
