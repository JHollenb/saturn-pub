"""Rebuild saved-development normalization comparisons from retained arrays."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive
from mdb.response_closure import measure_error,audit_factorial


def analyze(collection,precision,fresh,fresh_audit,output):
    read=lambda p:json.loads(p.read_text()); here=Path(__file__).resolve().parent
    verify=runpy.run_path(str(here/'flux_family_custody.py'))['verify']; custody=verify(collection); source_custody=dict(fresh=verify(fresh),precision=verify(precision))
    source=precision/'extracted/binding-minimal-result'; target=fresh/'extracted/binding-minimal-result'; data=collection/'extracted/binding-minimal-result'
    request=read(collection/'request.json'); c=request['config']; summary=read(data/'summary.json'); prior=read(fresh_audit)
    if prior['mechanics_status']!='verified' or sha(fresh_audit)!=c['source_fresh_audit_sha256']:raise ValueError('fresh audit binding differs')
    for key,record in source_custody.items():
        if any(record[k]!=prior['custody'][key][k] for k in ('job_id','archive_sha256','request_sha256')):raise ValueError('source custody differs from audited source')
    if summary!=read(collection/'report.json')['report'] or read(data/'plan.json')!=c:raise ValueError('plan/report differs')
    for key,root in [('prefix',source),('fresh',target)]:
        inventory=read(root/'files.json')
        for name,digest in c['inputs'][key].items():
            if inventory[name]!=digest or sha(root/name)!=digest:raise ValueError('bound source differs')
    if read(target/'summary.json')!=read(fresh/'report.json')['report']:raise ValueError('fresh source summary differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('worker source differs')
    native=read(source/'native.json'); newnative=read(target/'native.json'); original=read(target/'forecasts.json'); predictions=read(data/'predictions.json'); arrays=set()
    def array(item):
        p=data/item['path']
        if sha(p)!=item['sha256'] or item['encoding']!='bf16-u16-npy':raise ValueError('array identity differs')
        bits=np.load(p,allow_pickle=False)
        if bits.dtype!=np.dtype('<u2') or list(bits.shape)!=item['shape']:raise ValueError('array schema differs')
        value=(bits.astype(np.uint32)<<16).view(np.float32)
        if not np.isfinite(value).all():raise ValueError('nonfinite array')
        arrays.add(item['path']);return value
    reports={}; exits={}; observed={}; cached={}
    for spec in c['cases']:
        name,anchor=spec['id'],spec['anchor'];reports[name]={};observed[name]={}
        for mode,records in predictions['cases'][name].items():
            reports[name][mode]=dict(stages={},exits={});exits.setdefault(mode,{})[name]={}
            for key,item in records['stages'].items():
                p=array(item);n=load_tensor_archive(target,newnative[name]['stages'][key])[:,[4,11]]
                reports[name][mode]['stages'][key]={**measure_error(p,n),'different':int(np.count_nonzero(p!=n)),'values':p.size}
            for role,item in records['exits'].items():
                p=array(item);p0=array(predictions['baselines'][anchor][mode][role]);n=load_tensor_archive(target,newnative[name]['stages'][role+'.exit']);n0=load_tensor_archive(source,native[anchor]['stages'][role+'.exit'])
                if mode=='arithmetic' and not np.array_equal(p,load_tensor_archive(target,original[name][mode]['fields'][role]['exit'])):raise ValueError('control algorithm changed from frozen prospective forecast')
                reports[name][mode]['exits'][role]=dict(endpoint=measure_error(p,n),installed_edit=measure_error(p.astype(float)-n0.astype(float),n.astype(float)-n0.astype(float)),mathematical_response=measure_error(p.astype(float)-p0.astype(float),n.astype(float)-n0.astype(float)))
                exits[mode][name][role]=p;observed[name][role]=n
    order=['row11five_eighths','row4five_eighths','row11five_eighths.image1','row4five_eighths.image1']
    factorial={mode:{role:audit_factorial(dict(zip(('00','01','10','11'),(observed[n][role] for n in order))),dict(zip(('00','01','10','11'),(exits[mode][n][role] for n in order)))) for role in ('image','text')} for mode in exits}
    if reports!=summary['reports'] or factorial!=summary['factorial']:raise ValueError('recomputed measurements differ')
    result=dict(schema='mdb-absolute-normalization-development-audit-v1',mechanics_status='verified',instrument_status='Custody, source hashes, prospective arithmetic-control equality and all retained-array comparisons verified',trend_status='see per-case signed comparisons',terminal_status='unassessed',custody=custody,source_custody=source_custody,source_fresh_audit_sha256=sha(fresh_audit),reports=reports,factorial=factorial,counts=summary['counts'],arrays_verified=len(arrays),source_sha256=sha(Path(__file__)),new_native_calls_in_audit=0,limitations='Post-hoc fixed-law development on known outcomes; four states have only two unique text inputs. No fresh-state, image, object/relationship, universal arithmetic or speed qualification. Mathematical P1-P0 and installed P1-N0 effects reported separately.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',arrays_verified=len(arrays),state={n:{m:{r:v['mathematical_response']['relative_rms'] for r,v in rec['exits'].items()} for m,rec in modes.items()} for n,modes in reports.items()},factorial=factorial)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('collection','precision','fresh','fresh_audit','output'):p.add_argument(n,type=Path)
    a=p.parse_args();print(json.dumps(analyze(*[getattr(a,n).resolve() for n in ('collection','precision','fresh','fresh_audit','output')]),sort_keys=True))
