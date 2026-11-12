"""Saved attention-boundary comparison with explicit native-QKV calibration."""
import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path


def candidates(wide,wide0,native0,bf16):
    """Both laws share f(X0_native); no corrected-prefix baseline is inferred."""
    return dict(absolute=bf16(wide),anchored=bf16(native0.astype(float)+(wide-wide0)))


def load_bits(root,item):
    import numpy as np
    path=root/item['path'];bits=np.load(path,allow_pickle=False)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256'] or item['encoding']!='bf16-u16-npy' or bits.dtype!=np.dtype('<u2') or list(bits.shape)!=item['shape']:raise ValueError('BF16 array binding differs')
    value=(bits.astype(np.uint32)<<16).view(np.float32)
    if not np.isfinite(value).all():raise ValueError('nonfinite BF16 evidence')
    return value


def worker():
    payload=Path.cwd();request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text());c=request['config'];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('worker source changed')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
    import numpy as np
    from mdb.tensor_archive import load_tensor_archive
    from mdb.attention_partition import prepare_attention_partition
    from mdb.response_closure import measure_error,audit_factorial
    roots={k:Path(v) for k,v in c['input_roots'].items()}
    for key,items in c['inputs'].items():
        if any(sha(roots[key]/n)!=v for n,v in items.items()):raise ValueError('source evidence changed')
    source,target,norm=roots['prefix'],roots['fresh'],roots['normalization'];native=read(source/'native.json');fresh=read(target/'native.json');owned=read(norm/'predictions.json')
    r=runpy.run_path(str(payload/'flux_arithmetic_reference.py'));B=r['bf16'];ix=c['mutable_rows'];labels=('q_positioned','k_positioned','v_projection')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    save=lambda p,x:p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n');save(out/'plan.json',c);save(out/'request.json',request)
    def array(name,value):
        p=out/(name+'.npy');value=np.ascontiguousarray(value,np.float32)
        if not np.isfinite(value).all() or np.any(value.view(np.uint32)&65535):raise ValueError('nonfinite/nonBF16 prediction')
        with p.open('xb') as f:np.save(f,(value.view(np.uint32)>>16).astype('<u2'),allow_pickle=False)
        return dict(path=p.name,sha256=sha(p),encoding='bf16-u16-npy',shape=list(value.shape))
    baselines={};predictions={};reports={};targets={};values={};n0s={}
    for anchor in ('half','half.image1'):
        stages=native[anchor]['stages'];base=[np.concatenate([load_tensor_archive(source,stages[role+'.'+label]) for role in ('text','image')],axis=1).transpose(0,2,1,3) for label in labels]
        q0,k0,v0=base
        if any(list(x.shape)!=c['qkv_shape'] for x in base):raise ValueError('source attention geometry differs')
        partition=prepare_attention_partition(q0,k0,v0,ix);wide0,_=partition.evaluate(k0[:,:,ix],v0[:,:,ix],query_rows=q0[:,:,ix],query_indices=ix)
        native0=load_tensor_archive(source,stages['attention.raw']);shape=native0.shape;flatten=lambda x:x.transpose(0,2,1,3).reshape(shape);wide0=flatten(wide0)
        p0=candidates(wide0,wide0,native0,B);baselines[anchor]={method:array(anchor+'.'+method+'.P0',p) for method,p in p0.items()}
        if not np.array_equal(p0['anchored'],native0):raise ValueError('anchor identity differs')
        for case in [x for x in c['cases'] if x['anchor']==anchor]:
            name=case['id'];predictions[name]={};reports[name]={};targets[name]=load_tensor_archive(target,fresh[name]['stages']['attention.raw']);n0s[name]=native0
            for scope in ('own','conditional_native_operands'):
                if scope=='own':
                    replacement=[load_bits(norm,owned['cases'][name]['absolute_ln_rms']['stages']['text.'+label]).transpose(0,2,1,3) for label in labels]
                else:
                    text=[load_tensor_archive(target,fresh[name]['stages']['text.'+label]).transpose(0,2,1,3) for label in labels]
                    other=[j for j in range(text[0].shape[-2]) if j not in ix]
                    if any(not np.array_equal(x[:,:,other],old[:,:,other]) for x,old in zip(text,base)):raise ValueError('conditional unchanged text complement differs')
                    replacement=[x[:,:,ix] for x in text]
                q,k,v=replacement;wide,geometry=partition.evaluate(k,v,query_rows=q,query_indices=ix);calculated=candidates(flatten(wide),wide0,native0,B)
                predictions[name][scope]={};reports[name][scope]={};values.setdefault(scope,{})[name]={}
                for method,p in calculated.items():
                    predictions[name][scope][method]=array(name+'.'+scope+'.'+method,p);values[scope][name][method]=p;n=targets[name]
                    reports[name][scope][method]=dict(endpoint=measure_error(p,n),installed_edit=measure_error(p.astype(float)-native0.astype(float),n.astype(float)-native0.astype(float)),boundary_response=measure_error(p.astype(float)-p0[method].astype(float),n.astype(float)-native0.astype(float)))
                print('MDB_ATTENTION_ANCHOR '+json.dumps(dict(case=name,scope=scope,seconds=time.monotonic()-started,reports=reports[name][scope]),sort_keys=True),flush=True)
            del wide
        del partition,base,wide0
    order=['row11five_eighths','row4five_eighths','row11five_eighths.image1','row4five_eighths.image1'];keys=('00','01','10','11')
    factorial={scope:{method:audit_factorial(dict(zip(keys,(targets[n] for n in order))),dict(zip(keys,(values[scope][n][method] for n in order)))) for method in ('absolute','anchored')} for scope in values}
    save(out/'predictions.json',dict(cases=predictions,baselines=baselines))
    if any(n in sys.modules for n in ('torch','diffusers','transformers')):raise ValueError('model framework imported')
    summary=dict(schema='mdb-attention-anchor-development-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),seconds=time.monotonic()-started,reports=reports,factorial=factorial,
        counts=dict(model_loads=0,native_calls=0,weights_loaded=0,states=4,methods=2,operand_scopes=2,predictions=16,baselines=4),mechanics_status='pending-independent-audit',terminal_status='unassessed',
        scope='Saved-case development. Own QKV comes from prior independent LN+RMS calculation on changed rows and explicit native unchanged complement. Conditional-native-operands is a separate target-assisted diagnostic. P0 at this attention boundary uses supplied native calibration X0, not a reconstructed corrected-prefix P0. No new native controls, complete downstream writes or images.')
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,precision,fresh,normalization,audit):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text());verify=runpy.run_path(str(here/'flux_family_custody.py'))['verify']
    checked={k:verify(p) for k,p in [('precision',precision),('fresh',fresh),('normalization',normalization)]};qualified=read(audit)
    if qualified['mechanics_status']!='verified':raise ValueError('normalization source not verified')
    for key,actual in checked.items():
        prior=qualified['custody'] if key=='normalization' else qualified['source_custody'][key]
        if any(actual[k]!=prior[k] for k in ('job_id','archive_sha256','request_sha256')):raise ValueError('source audit custody mismatch')
    roots={k:p/'extracted/binding-minimal-result' for k,p in [('prefix',precision),('fresh',fresh),('normalization',normalization)]};remote={k:read(p/'summary.json')['output_root']+'/binding-minimal-result' for k,p in roots.items()};inputs={k:{} for k in roots}
    native=read(roots['prefix']/'native.json');new=read(roots['fresh']/'native.json');owned=read(roots['normalization']/'predictions.json');labels=('q_positioned','k_positioned','v_projection')
    names={'prefix':{'native.json','summary.json'},'fresh':{'native.json','summary.json'},'normalization':{'predictions.json','summary.json'}}
    for anchor in ('half','half.image1'):
        for key in ['attention.raw',*[role+'.'+label for role in ('text','image') for label in labels]]:names['prefix'].add(native[anchor]['stages'][key]['path'])
    cases=read(fresh/'request.json')['config']['cases']
    for case in cases:
        name=case['id']
        for key in ['attention.raw',*['text.'+label for label in labels]]:names['fresh'].add(new[name]['stages'][key]['path'])
        for label in labels:names['normalization'].add(owned['cases'][name]['absolute_ln_rms']['stages']['text.'+label]['path'])
    for key,files in names.items():
        inventory=read(roots[key]/'files.json')
        for name in files:
            if sha(roots[key]/name)!=inventory[name]:raise ValueError('input inventory differs')
            inputs[key][name]=inventory[name]
    prior=read(fresh/'request.json');h['bind_environment'](prior['execution_environment']);run.mkdir(parents=True,exist_ok=False)
    c=dict(input_roots=remote,inputs=inputs,cases=cases,mutable_rows=[4,11],qkv_shape=[1,24,1536,128],clock=dict(step=2,joint=1),checkpoint_sha256=prior['config']['checkpoint_sha256'],source_normalization_audit_sha256=sha(audit),output_scratch='/mnt/big/mdb/research',directory_custody=True,scope='Attention-boundary calibration comparison on saved states, not full-program forecast;0weights/0native calls.')
    save(run/'plan.json',c);qualification=run/'source-qualification.json';save(qualification,dict(audit_sha256=sha(audit),custody=checked))
    extras=[(here/n,n) for n in ('flux_family_custody.py','flux_arithmetic_reference.py')]+[(qualification,qualification.name)];pre=run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[*remote.values(),c['output_scratch'],'binding-minimal-result']));extras.append((pre,pre.name))
    request=dict(schema='mdb-attention-anchor-development-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',execution_environment=prior['execution_environment'],source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=20,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=20,extra=dict(task_family='analysis'),checkpoint_policy='2native-QKV calibration caches;4corrected-own states plus4explicit conditional-native-operand diagnostics;2fixedattentionlaws,20arrays.0native/weights/images.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or source drift')
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-attention-anchor-development',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=300,retry_on_kill=False,preflight=True,note='Saved attention boundary: anchored vs absolute wide,4corrected-own QKV states +4clearlylabelled native-operand diagnostics.0weights/0model/native/image calls.2cachedpartitions,4CPUthreads,under90s/300souter,20BF16arrays,no retry,intent-first.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
        for n in ('precision','fresh','normalization','audit'):p.add_argument(n,type=Path,nargs='?')
        p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(*[getattr(a,n).resolve() for n in ('run','precision','fresh','normalization','audit')]),sort_keys=True))
