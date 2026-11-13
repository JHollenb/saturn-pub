"""Audit full-block attention-law comparisons and join prior boundary predictions."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive
from mdb.response_closure import measure_error,audit_factorial


def analyze(collection,precision,fresh,normalization,normalization_audit,boundary,boundary_audit,output):
    read=lambda p:json.loads(p.read_text()); here=Path(__file__).resolve().parent
    verify=runpy.run_path(str(here/'flux_family_custody.py'))['verify']; custody=verify(collection); source_custody=dict(fresh=verify(fresh),precision=verify(precision),normalization=verify(normalization),boundary=verify(boundary))
    source=precision/'extracted/binding-minimal-result'; target=fresh/'extracted/binding-minimal-result'; data=collection/'extracted/binding-minimal-result'
    request=read(collection/'request.json'); c=request['config']; summary=read(data/'summary.json'); prior=read(normalization_audit);attention=read(boundary_audit)
    if attention['mechanics_status']!='verified' or sha(boundary_audit)!=c['source_attention_audit_sha256'] or prior['mechanics_status']!='verified':raise ValueError('source audit binding differs')
    for key,record in source_custody.items():
        known=prior['source_custody'][key] if key in ('fresh','precision') else prior['custody'] if key=='normalization' else attention['custody']
        if any(record[k]!=known[k] for k in ('job_id','archive_sha256','request_sha256')):raise ValueError('source custody differs from audited source')
    if attention['source_audit_sha256']!=sha(normalization_audit):raise ValueError('attention and normalization audits are disconnected')
    if summary!=read(collection/'report.json')['report'] or read(data/'plan.json')!=c:raise ValueError('plan/report differs')
    for key,root in [('prefix',source),('fresh',target)]:
        inventory=read(root/'files.json')
        for name,digest in c['inputs'][key].items():
            if inventory[name]!=digest or sha(root/name)!=digest:raise ValueError('bound source differs')
    if read(target/'summary.json')!=read(fresh/'report.json')['report']:raise ValueError('fresh source summary differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('worker source differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('installed wheel differs')
    native=read(source/'native.json'); newnative=read(target/'native.json'); original=read(normalization/'extracted/binding-minimal-result/predictions.json'); raw=read(boundary/'extracted/binding-minimal-result/predictions.json'); predictions=read(data/'predictions.json'); arrays=set()
    read_bits=runpy.run_path(str(boundary/'payload/flux_attention_anchor.py'))['load_bits']
    expected_stages={'attention.raw',*['text.'+key for key in ('modulated','q_projection','k_projection','v_projection','q_normalized','k_normalized','q_positioned','k_positioned','after_attention')]}
    if len(c['cases'])!=4 or set(predictions['cases'])!={x['id'] for x in c['cases']} or set(predictions['baselines'])!={'half','half.image1'}:raise ValueError('case/anchor census differs')
    for modes in predictions['baselines'].values():
        if set(modes)!={'absolute','anchored'} or any(set(roles)!={'image','text'} for roles in modes.values()):raise ValueError('baseline census differs')
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
        if set(predictions['cases'][name])!={'absolute','anchored'}:raise ValueError('method census differs')
        for mode,records in predictions['cases'][name].items():
            if set(records['stages'])!=expected_stages or set(records['exits'])!={'image','text'}:raise ValueError('retained port census differs')
            reports[name][mode]=dict(stages={},exits={});exits.setdefault(mode,{})[name]={}
            for key,item in records['stages'].items():
                p=array(item);n=load_tensor_archive(target,newnative[name]['stages'][key])
                if key=='attention.raw':
                    if not np.array_equal(p,read_bits(boundary/'extracted/binding-minimal-result',raw['cases'][name]['own'][mode])):raise ValueError('full candidate raw attention differs from boundary calculation')
                else:n=n[:,[4,11]]
                reports[name][mode]['stages'][key]={**measure_error(p,n),'different':int(np.count_nonzero(p!=n)),'values':p.size}
            for role,item in records['exits'].items():
                p=array(item);p0=array(predictions['baselines'][anchor][mode][role]);n=load_tensor_archive(target,newnative[name]['stages'][role+'.exit']);n0=load_tensor_archive(source,native[anchor]['stages'][role+'.exit'])
                if mode=='anchored':
                    if not np.array_equal(p,read_bits(normalization/'extracted/binding-minimal-result',original['cases'][name]['absolute_ln_rms']['exits'][role])) or not np.array_equal(p0,read_bits(normalization/'extracted/binding-minimal-result',original['baselines'][anchor]['absolute_ln_rms'][role])):raise ValueError('anchored program changed from original normalization diagnostic')
                reports[name][mode]['exits'][role]=dict(endpoint=measure_error(p,n),installed_edit=measure_error(p.astype(float)-n0.astype(float),n.astype(float)-n0.astype(float)),mathematical_response=measure_error(p.astype(float)-p0.astype(float),n.astype(float)-n0.astype(float)))
                exits[mode][name][role]=p;observed[name][role]=n
    order=['row11five_eighths','row4five_eighths','row11five_eighths.image1','row4five_eighths.image1']
    factorial={mode:{role:audit_factorial(dict(zip(('00','01','10','11'),(observed[n][role] for n in order))),dict(zip(('00','01','10','11'),(exits[mode][n][role] for n in order)))) for role in ('image','text')} for mode in exits}
    if reports!=summary['reports'] or factorial!=summary['factorial']:raise ValueError('recomputed measurements differ')
    if len(arrays)!=104 or summary['counts']!=dict(model_loads=0,native_calls=0,variants=2,states=4,predictions=8,baselines=4):raise ValueError('array/execution census differs')
    result=dict(schema='mdb-attention-law-development-audit-v1',mechanics_status='verified',instrument_status='Custody, source hashes, prior full-control and raw-attention equality and all retained-array comparisons verified',trend_status='see per-case signed comparisons',terminal_status='unassessed',custody=custody,source_custody=source_custody,source_attention_audit_sha256=sha(boundary_audit),source_normalization_audit_sha256=sha(normalization_audit),reports=reports,factorial=factorial,counts=summary['counts'],arrays_verified=len(arrays),source_sha256=sha(Path(__file__)),new_native_calls_in_audit=0,limitations='Post-hoc fixed attention-law development on four known states, two unique text inputs. Both programs calculate their own full P0/P1 exits; boundary-only baseline is distinct. No new native control or unseen-state, semantic/image or speed qualification. Historical native calibration and weights remain supplied; temporary weights deleted.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',arrays_verified=len(arrays),state={n:{m:{r:v['mathematical_response']['relative_rms'] for r,v in rec['exits'].items()} for m,rec in modes.items()} for n,modes in reports.items()},mixed={m:{r:v['contrasts']['receiver_by_dose']['relative_rms'] for r,v in roles.items()} for m,roles in factorial.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('collection','precision','fresh','normalization','normalization_audit','boundary','boundary_audit','output'):p.add_argument(n,type=Path)
    a=p.parse_args();print(json.dumps(analyze(*[getattr(a,n).resolve() for n in ('collection','precision','fresh','normalization','normalization_audit','boundary','boundary_audit','output')]),sort_keys=True))
