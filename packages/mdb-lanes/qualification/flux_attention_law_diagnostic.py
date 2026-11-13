"""Saved full-block comparison of two attention laws on a corrected prefix."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
from pathlib import Path


def worker():
    payload=Path.cwd(); request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text()); c=request['config']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); read=lambda p:json.loads(p.read_text())
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('source changed')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
    import numpy as np
    from mdb_lanes.parameter_artifacts import restore,validate_manifest
    from mdb_lanes.tensor_archive import load_tensor_archive
    from mdb_lanes.response_closure import measure_error,audit_factorial
    roots={k:Path(v) for k,v in c['input_roots'].items()}
    for key,items in c['inputs'].items():
        if any(sha(roots[key]/n)!=v for n,v in items.items()): raise ValueError('source operand changed')
    source,target=roots['prefix'],roots['fresh']; manifest=validate_manifest(read(source/'parameter-artifacts.json'))
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out=scratch/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    save=lambda p,x:p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
    save(out/'plan.json',c); save(out/'request.json',request); save(out/'parameter-artifacts.json',manifest); shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    def array(name,value):
        if not np.isfinite(value).all() or np.any(np.ascontiguousarray(value).view(np.uint32)&65535): raise ValueError('expected finite BF16')
        p=out/(name+'.npy'); bits=(np.ascontiguousarray(value).view(np.uint32)>>16).astype('<u2')
        with p.open('xb') as f:np.save(f,bits,allow_pickle=False)
        return dict(path=p.name,sha256=sha(p),shape=list(value.shape),encoding='bf16-u16-npy')
    r=runpy.run_path(str(payload/'flux_reentry_reference.py')); law=runpy.run_path(str(payload/'flux_attention_law_prediction.py')); slot=runpy.run_path(str(payload/'analyze_flux_binding_minimal.py'))['read_slot']
    native=read(source/'native.json'); fresh=read(target/'native.json'); plan=read(source/'plan.json')
    T0,T1=[slot(source/'source/store',plan['parents'][k],'text') for k in ('base','edited')]
    variants={'anchored':['text.norm1','text.qk_rms'],'absolute':['text.norm1','text.qk_rms']}; predictions={}; baselines={}; reports={}; exits={}; actuals={}
    with tempfile.TemporaryDirectory(prefix='absolute-norm-weights-',dir=scratch) as tmp:
        save(out/'weight-restoration.json',restore(manifest,tmp,manifest['defaults']['checkpoint_root'])); weights={}
        for item in manifest['artifacts']:
            bits=np.load(Path(tmp)/item['path'],allow_pickle=False); value=(bits.astype(np.uint32)<<16).view(np.float32)
            for port in item['parameter_ports']:weights[port]=value
        for anchor in ('half','half.image1'):
            stages={k:load_tensor_archive(source,v) for k,v in native[anchor]['stages'].items()}; entry=native[anchor]['entry']
            I,T,context=[slot(source/'store',entry,k) for k in ('image','text','context')]; baselines[anchor]={}
            for mode,norms in variants.items():
                cache=law['prepare'](I,T,context,weights,stages,r,absolute_ports=c['absolute_ports'],affine_policy=c['policy'],absolute_norms=norms,attention_mode=mode)
                zero=law['predict'](cache,T); baselines[anchor][mode]={role:array(anchor+'.'+mode+'.zero.'+role,zero['fields'][role]['exit']) for role in ('image','text')}
                for case in [x for x in c['cases'] if x['anchor']==anchor]:
                    name=case['id']; text=T0.copy()
                    for row,dose in zip((4,11),case['row_doses']):text[:,row]=r['bf16'](T0[:,row].astype(float)+dose*(T1[:,row].astype(float)-T0[:,row].astype(float)))
                    seen={}; value=law['predict'](cache,text,observer=lambda k,v:seen.__setitem__(k,v.copy()) if k.startswith('text.') or k=='attention.raw' else None)
                    predictions.setdefault(name,{})[mode]=dict(exits={role:array(name+'.'+mode+'.'+role,value['fields'][role]['exit']) for role in ('image','text')},stages={key:array(name+'.'+mode+'.'+key,(x if key=='attention.raw' else x[:,[4,11]])) for key,x in seen.items()},seconds=value['seconds'])
                    reports.setdefault(name,{})[mode]={'stages':{},'exits':{}}; exits.setdefault(mode,{})[name]={}; actuals.setdefault(name,{})
                    for key,x in seen.items():
                        n=load_tensor_archive(target,fresh[name]['stages'][key]); p=x
                        if key!='attention.raw':n=n[:,[4,11]];p=p[:,[4,11]]
                        reports[name][mode]['stages'][key]={**measure_error(p,n),'different':int(np.count_nonzero(p!=n)),'values':p.size}
                    for role in ('image','text'):
                        p=value['fields'][role]['exit']; n=load_tensor_archive(target,fresh[name]['stages'][role+'.exit']); n0=stages[role+'.exit']; p0=zero['fields'][role]['exit']
                        reports[name][mode]['exits'][role]=dict(endpoint=measure_error(p,n),installed_edit=measure_error(p.astype(float)-n0.astype(float),n.astype(float)-n0.astype(float)),mathematical_response=measure_error(p.astype(float)-p0.astype(float),n.astype(float)-n0.astype(float)))
                        exits[mode][name][role]=p; actuals[name][role]=n
                    print('MDB_ATTENTION_LAW '+json.dumps(dict(case=name,mode=mode,seconds=time.monotonic()-started,exits=reports[name][mode]['exits'])),flush=True)
                del cache,zero
    order=['row11five_eighths','row4five_eighths','row11five_eighths.image1','row4five_eighths.image1']
    factorial={mode:{role:audit_factorial(dict(zip(('00','01','10','11'),(actuals[n][role] for n in order))),dict(zip(('00','01','10','11'),(exits[mode][n][role] for n in order)))) for role in ('image','text')} for mode in variants}
    save(out/'predictions.json',dict(cases=predictions,baselines=baselines))
    if any(n in sys.modules for n in ('torch','diffusers','transformers')): raise ValueError('model framework imported')
    summary=dict(schema='mdb-attention-law-development-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),reports=reports,factorial=factorial,seconds=time.monotonic()-started,
        counts=dict(model_loads=0,native_calls=0,variants=2,states=4,predictions=8,baselines=4),mechanics_status='pending-independent-audit',terminal_status='unassessed',scope='Post-hoc full-block development on four known target states. Corrected LN/RMS and affine laws fixed; attention anchored versus absolute only. All downstream operations consume own predicted operands, not target ports. Full independently calculated P0/P1 exits retained. Zero native/framework/image calls; historical calibration remains supplied.')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,precision,fresh,boundary,audit):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    verify=runpy.run_path(str(here/'flux_family_custody.py'))['verify']; verify(precision); verify(fresh); checked=verify(boundary); qualified=read(audit)
    if qualified['mechanics_status']!='verified' or any(qualified['custody'][k]!=checked[k] for k in ('job_id','archive_sha256','request_sha256')):raise ValueError('verified attention-boundary audit required')
    boundary_plan=read(boundary/'request.json')['config']
    for key,collection in [('prefix',precision),('fresh',fresh)]:
        if sha(collection/'extracted/binding-minimal-result/summary.json')!=boundary_plan['inputs'][key]['summary.json']:raise ValueError('boundary audit has different native sources')
    roots={key:root/'extracted/binding-minimal-result' for key,root in [('prefix',precision),('fresh',fresh)]}; inputs={}; remote={}
    for key,root in roots.items():
        inputs[key]={}; remote[key]=read(root/'summary.json')['output_root']+'/binding-minimal-result'
        for name,digest in read(root/'files.json').items():
            if name.startswith(('arrays/','store/','source/store/')) or name in ('native.json','plan.json','parameter-artifacts.json','summary.json'):
                if sha(root/name)!=digest:raise ValueError('input inventory differs')
                inputs[key][name]=digest
    prior=read(fresh/'request.json'); h['bind_environment'](prior['execution_environment'])
    c=dict(input_roots=remote,inputs=inputs,output_scratch='/mnt/big/mdb/research',directory_custody=True,cases=prior['config']['cases'],absolute_ports=prior['config']['absolute_ports'],policy=prior['config']['policy'],source_attention_audit_sha256=sha(audit),scope='Post-hoc full-block attention-law comparison; zero native/model/image calls.')
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    extras=[(here/n,n) for n in ('flux_family_custody.py','flux_reentry_reference.py','flux_arithmetic_reference.py','flux_attention_law_prediction.py','analyze_flux_binding_minimal.py')]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py')]
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[*remote.values(),c['output_scratch'],'binding-minimal-result',read(roots['prefix']/'parameter-artifacts.json')['defaults']['checkpoint_root']]))
    extras.append((pre,pre.name)); request=dict(schema='mdb-attention-law-development-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',execution_environment=prior['execution_environment'],source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=12,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=12,extra=dict(task_family='analysis'),checkpoint_policy='Two known receivers, two fixed attention variants, four known targets. Corrected own prefix, downstream own-state equations;0native/framework calls; retain full P0/P1 exits and raw attention.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or source changed')
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-attention-law-development',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,note='Saved full-block NumPy development:2fixed attentionlaws on corrected LN/RMS/affine prefix,2receivers/8predictions/4baselines.0native/model/image calls,4CPUthreads,temporaryweightsdeleted;fullBF16 P0/P1 andrawA,under3minutes/600souter,no retries,intent-first.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
        for n in ('precision','fresh','boundary','audit'):p.add_argument(n,type=Path,nargs='?')
        p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.precision.resolve(),a.fresh.resolve(),a.boundary.resolve(),a.audit.resolve()),sort_keys=True))
