"""Audit receiver-law factorial and retain exact baseline-error covariance."""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
from mdb.edit_search import file_digest as sha
from mdb.tensor_archive import load_tensor_archive
from mdb.response_closure import measure_error,audit_factorial,audit_response


def error_transport(p0,p1,n0,n1):
    p0,p1,n0,n1=[np.asarray(x,float) for x in (p0,p1,n0,n1)];e0=p0-n0;e1=p1-n1;response=(p1-p0)-(n1-n0)
    if not np.array_equal(e1-e0,response):raise ValueError('error subtraction identity differs')
    public=audit_response(n0,n1,p0,p1);power=public['defect_power']
    return dict(baseline_endpoint=measure_error(p0,n0),mean_error=dict(baseline=float(np.mean(e0)),new=float(np.mean(e1))),error_cosine=public['defect_change']['cosine'],mean_square=dict(baseline=power['anchor_mse'],new=power['target_mse'],twice_cross_moment=power['signed_cross_twice'],response=power['change_mse'],reconstructed=power['anchor_mse']+power['target_mse']-power['signed_cross_twice']))


def analyze(collection,precision,fresh,full,full_audit,boundary_audit,output):
    here=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());verify=runpy.run_path(str(here/'flux_family_custody.py'))['verify'];custody=verify(collection);prior_custody=verify(full)
    data=collection/'extracted/binding-minimal-result';source=precision/'extracted/binding-minimal-result';target=fresh/'extracted/binding-minimal-result';prior_root=full/'extracted/binding-minimal-result';request=read(collection/'request.json');c=request['config'];summary=read(data/'summary.json');qualified=read(full_audit)
    if qualified['mechanics_status']!='verified' or read(boundary_audit)['mechanics_status']!='verified' or sha(boundary_audit)!=c['source_attention_audit_sha256'] or qualified['source_attention_audit_sha256']!=sha(boundary_audit):raise ValueError('qualification chain differs')
    if any(prior_custody[k]!=qualified['custody'][k] for k in ('job_id','request_sha256','archive_sha256')):raise ValueError('prior full custody differs')
    prior_plan=read(prior_root/'plan.json')
    for key in ('cases','policy','absolute_ports','inputs'):
        if c[key]!=prior_plan[key]:raise ValueError('controlled sources or laws changed')
    if summary!=read(collection/'report.json')['report'] or c!=read(data/'plan.json'):raise ValueError('plan/report differs')
    for key,root in [('prefix',source),('fresh',target)]:
        inventory=read(root/'files.json')
        for name,digest in c['inputs'][key].items():
            if inventory[name]!=digest or sha(root/name)!=digest:raise ValueError('consumed source differs')
    for name,digest in request['source_hashes'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('worker source differs')
    for name,digest in read(collection/'payload/payload-seal.json')['wheels'].items():
        if sha(collection/'payload'/name)!=digest:raise ValueError('installed wheel differs')
    helper=runpy.run_path(str(here/'flux_attention_anchor.py'));bits=helper['load_bits'];r=runpy.run_path(str(collection/'payload/flux_reentry_reference.py'));B=r['bf16'];slot=runpy.run_path(str(collection/'payload/analyze_flux_binding_minimal.py'))['read_slot']
    predictions=read(data/'predictions.json');old=read(prior_root/'predictions.json');native=read(source/'native.json');newnative=read(target/'native.json');modes={a+'.'+n for a in ('anchored','absolute') for n in ('anchored','absolute')};roles={'image','text'};fields={'attention','mlp_input','mlp','exit'}
    stages={'attention.raw','image.after_attention','text.after_attention',*['text.'+x for x in ('modulated','q_projection','k_projection','v_projection','q_normalized','k_normalized','q_positioned','k_positioned')]};arrays=set();occurrences=0
    def array(item):
        nonlocal occurrences
        occurrences+=1;arrays.add(item['path']);return bits(data,item)
    if len(c['cases'])!=4 or set(predictions['cases'])!={x['id'] for x in c['cases']} or set(predictions['baselines'])!={'half','half.image1'}:raise ValueError('case/anchor census differs')
    # Validate every descriptor occurrence once; later reads do not inflate the census.
    for by_mode in [*predictions['baselines'].values(),*predictions['cases'].values()]:
        if set(by_mode)!=modes:raise ValueError('mode census differs')
        for record in by_mode.values():
            table=record.get('fields',record)
            if set(table)!=roles or any(set(v)!=fields for v in table.values()):raise ValueError('writer field census differs')
            for values in table.values():
                for item in values.values():array(item)
            if 'stages' in record:
                if set(record['stages'])!=stages:raise ValueError('observer port census differs')
                for item in record['stages'].values():array(item)
    npys={p.name for p in data.glob('*.npy')}
    if occurrences!=368 or arrays!=npys or summary['counts']!=dict(model_loads=0,native_calls=0,variants=4,states=4,predictions=16,baselines=8):raise ValueError('array/reference/execution census differs')
    reports={};covariances={};exits={};observed={};law_interaction={}
    for spec in c['cases']:
        name,anchor=spec['id'],spec['anchor'];reports[name]={};covariances[name]={};observed[name]={};residual={k:slot(source/'store',native[anchor]['entry'],k) for k in ('image','text')};context=slot(source/'store',native[anchor]['entry'],'context')
        gates={role:np.split(context['img_mod' if role=='image' else 'txt_mod'].reshape(1,1,-1),6,axis=-1)[2] for role in roles}
        for mode in modes:
            attention_mode,norm_mode=mode.split('.');records=predictions['cases'][name][mode];baseline=predictions['baselines'][anchor][mode];reports[name][mode]=dict(stages={},writers={},exits={});covariances[name][mode]={};exits.setdefault(mode,{})[name]={}
            opposite=predictions['cases'][name][attention_mode+'.'+('absolute' if norm_mode=='anchored' else 'anchored')]
            if records['stages']!=opposite['stages']:raise ValueError('receiving norm changed upstream stages')
            common=predictions['cases'][name]['anchored.anchored']['stages']
            if any(item!=common[key] for key,item in records['stages'].items() if key.startswith('text.') and key!='text.after_attention'):raise ValueError('attention law changed text prefix')
            for key,item in records['stages'].items():
                p=bits(data,item);n=load_tensor_archive(target,newnative[name]['stages'][key])
                if key.startswith('text.') and key!='text.after_attention':n=n[:,[4,11]]
                reports[name][mode]['stages'][key]={**measure_error(p,n),'different':int(np.count_nonzero(p!=n)),'values':p.size}
                oldkey=old['cases'][name][attention_mode]['stages'].get(key)
                if oldkey is not None:
                    current=p[:,[4,11]] if key=='text.after_attention' else p
                    if not np.array_equal(current,bits(prior_root,oldkey)):raise ValueError('upstream operand or write changed with receiving norm')
            for role in roles:
                reports[name][mode]['writers'][role]={};covariances[name][mode][role]={}
                for field in fields:
                    p=bits(data,records['fields'][role][field]);p0=bits(data,baseline[role][field]);n=load_tensor_archive(target,newnative[name]['stages'][role+'.'+field]);n0=load_tensor_archive(source,native[anchor]['stages'][role+'.'+field])
                    reports[name][mode]['writers'][role][field]=dict(endpoint=measure_error(p,n),mathematical_response=measure_error(p.astype(float)-p0.astype(float),n.astype(float)-n0.astype(float)))
                    covariances[name][mode][role][field]=error_transport(p0,p,n0,n)
                    if field=='exit':
                        reports[name][mode]['exits'][role]=dict(endpoint=measure_error(p,n),installed_edit=measure_error(p.astype(float)-n0.astype(float),n.astype(float)-n0.astype(float)),mathematical_response=measure_error(p.astype(float)-p0.astype(float),n.astype(float)-n0.astype(float)))
                        exits[mode][name][role]=p;observed[name][role]=n
                        if norm_mode=='anchored' and (not np.array_equal(p,bits(prior_root,old['cases'][name][attention_mode]['exits'][role])) or not np.array_equal(p0,bits(prior_root,old['baselines'][anchor][attention_mode][role]))):raise ValueError('prior complete P0/P1 control changed')
                    if field=='attention':
                        other_baseline=predictions['baselines'][anchor][attention_mode+'.'+('absolute' if norm_mode=='anchored' else 'anchored')]
                        if records['fields'][role][field]!=opposite['fields'][role][field] or baseline[role][field]!=other_baseline[role][field]:raise ValueError('receiving norm changed projected attention')
                p0u=B(residual[role]+B(gates[role]*bits(data,baseline[role]['attention'])));p1u=bits(data,records['stages'][role+'.after_attention']);n0u=load_tensor_archive(source,native[anchor]['stages'][role+'.after_attention']);n1u=load_tensor_archive(target,newnative[name]['stages'][role+'.after_attention'])
                covariances[name][mode][role]['after_attention']=error_transport(p0u,p1u,n0u,n1u)
        law_interaction[name]={}
        for role in roles:
            coeff={'absolute.absolute':1,'absolute.anchored':-1,'anchored.absolute':-1,'anchored.anchored':1}
            p1=sum(exits[m][name][role].astype(float)*sign for m,sign in coeff.items());p0=sum(bits(data,predictions['baselines'][anchor][m][role]['exit']).astype(float)*sign for m,sign in coeff.items())
            law_interaction[name][role]={key:dict(rms=float(np.sqrt(np.mean(value*value))),max_abs=float(np.max(np.abs(value)))) for key,value in [('endpoint_P1',p1),('baseline_P0',p0),('response_P1_minus_P0',p1-p0)]}
            law_interaction[name][role]['scope']='Numerical attention-law x receiving-LN-law interaction; baseline, endpoint and finite response retained separately. Not the native receiver x input-pattern contrast.'
    order=['row11five_eighths','row4five_eighths','row11five_eighths.image1','row4five_eighths.image1'];keys=('00','01','10','11')
    factorial={mode:{role:audit_factorial(dict(zip(keys,(observed[n][role] for n in order))),dict(zip(keys,(exits[mode][n][role] for n in order)))) for role in roles} for mode in modes}
    if reports!=summary['reports'] or factorial!=summary['factorial']:raise ValueError('recomputed metrics differ')
    result=dict(schema='mdb-receiver-norm-development-audit-v1',mechanics_status='verified',instrument_status='Source/custody bindings,368logical references, exact upstream invariance, prior P0/P1 controls and all writer/error-transport comparisons verified',terminal_status='unassessed',custody=custody,source_full_audit_sha256=sha(full_audit),source_boundary_audit_sha256=sha(boundary_audit),counts=summary['counts'],logical_array_references=occurrences,unique_arrays_verified=len(arrays),reports=reports,error_transport=covariances,factorial=factorial,law_interaction=law_interaction,worker_seconds=summary['seconds'],source_sha256=sha(Path(__file__)),new_native_calls_in_audit=0,
        limitations='Fixed-law development on four already observed states/two distinct text inputs. No fitted coefficient or target-operand substitution; native historical calibration remains supplied. All countertrends retained. No unseen-state, semantic/image or speed qualification.')
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(mechanics_status='verified',logical_array_references=occurrences,unique_arrays_verified=len(arrays),state={n:{m:{r:values['mathematical_response']['relative_rms'] for r,values in rec['exits'].items()} for m,rec in table.items()} for n,table in reports.items()},mixed={m:{r:v['contrasts']['receiver_by_dose']['relative_rms'] for r,v in values.items()} for m,values in factorial.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','precision','fresh','full','full_audit','boundary_audit','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(analyze(*[getattr(a,n).resolve() for n in ('collection','precision','fresh','full','full_audit','boundary_audit','output')]),sort_keys=True))
