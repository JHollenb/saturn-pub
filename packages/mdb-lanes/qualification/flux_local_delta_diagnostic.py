"""Model-free saved-case development test of primitive finite responses."""
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
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()): raise ValueError('worker source differs')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
    import numpy as np
    from mdb_lanes.tensor_archive import load_tensor_archive
    from mdb_lanes.parameter_artifacts import restore,validate_manifest
    from mdb_lanes.response_closure import measure_error,audit_factorial
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out=scratch/'binding-minimal-result'; out.mkdir(parents=True,exist_ok=False)
    save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n'); save(out/'plan.json',c); save(out/'request.json',request); started=time.monotonic()
    def event(kind,**kw): print('MDB_LOCAL_DELTA '+json.dumps(dict(kind=kind,seconds=time.monotonic()-started,**kw),sort_keys=True),flush=True)
    for key,items in c['inputs'].items():
        for name,item in items.items():
            p=Path(c['input_roots'][key])/name
            if sha(p)!=item['sha256'] or p.stat().st_size!=item['bytes']: raise ValueError('source evidence differs')
    source=Path(c['input_roots']['prefix']); target=Path(c['input_roots']['targets'])
    manifest=validate_manifest(read(source/'parameter-artifacts.json')); save(out/'parameter-artifacts.json',manifest); shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    r=runpy.run_path(str(payload/'flux_reentry_reference.py')); local=runpy.run_path(str(payload/'flux_local_delta_reference.py')); slot=runpy.run_path(str(payload/'analyze_flux_binding_minimal.py'))['read_slot']
    native=read(source/'native.json'); target_native=read(target/'native.json'); prior=read(target/'forecasts.json'); old_forecasts=read(source/'forecasts.json')
    source_plan=read(source/'plan.json'); T0=slot(source/'source/store',source_plan['parents']['base'],'text'); T1=slot(source/'source/store',source_plan['parents']['edited'],'text')
    reports={}; predicted={}; observed={}; anchors={}; outputs={}
    def array(name,value):
        p=out/'arrays'/(name+'.npy'); p.parent.mkdir(exist_ok=True)
        with p.open('xb') as f: np.save(f,value,allow_pickle=False)
        return dict(path=p.relative_to(out).as_posix(),sha256=sha(p),shape=list(value.shape),dtype=str(value.dtype))
    with tempfile.TemporaryDirectory(prefix='local-delta-weights-',dir=scratch) as tmp:
        save(out/'weight-restoration.json',restore(manifest,tmp,manifest['defaults']['checkpoint_root'])); weights={}
        for item in manifest['artifacts']:
            bits=np.load(Path(tmp)/item['path'],allow_pickle=False); weights[item['parameter_ports'][0]]=(bits.astype(np.uint32)<<16).view(np.float32)
        event('parameters_ready',ports=len(weights))
        for name,anchor in [('three_eighths','half'),('three_eighths.image1','half.image1')]:
            stage={k:load_tensor_archive(source,v) for k,v in native[anchor]['stages'].items()}; fp=native[anchor]['entry']
            I=slot(source/'store',fp,'image'); T=slot(source/'store',fp,'text'); context=slot(source/'store',fp,'context')
            cache=local['prepare'](I,T,context,weights,stage,r)
            zero=local['predict'](cache,T)
            if any(not np.array_equal(zero['fields'][role]['exit'],stage[role+'.exit']) for role in ('image','text')): raise ValueError('zero-edit native anchor closure differs')
            event('zero_edit_exact',case=name)
            newtext=T0.copy(); newtext[:,[4,11]]=r['bf16'](T0[:,[4,11]].astype(float)+.375*(T1[:,[4,11]].astype(float)-T0[:,[4,11]]))
            new=local['predict'](cache,newtext); outputs[name]=dict(fields={role:{k:array(name+'.'+role+'.'+k,v) for k,v in values.items()} for role,values in new['fields'].items()},cache_seconds=cache['seconds'],predict_seconds=new['seconds'],affine_work=new['affine_work'],attention_work=new['attention_work'])
            reports[name]={}; predicted[name]={}; observed[name]={}; anchors[name]={}
            for role in ('image','text'):
                y=new['fields'][role]['exit']; n=load_tensor_archive(target,target_native[name]['fields'][role]); n0=stage[role+'.exit']; p1=load_tensor_archive(target,prior[name]['fields'][role]['absolute']); p0=load_tensor_archive(source,old_forecasts[anchor]['legacy']['fields'][role]['exit'])
                reports[name][role]=dict(endpoint=measure_error(y,n),dose_response=measure_error(y.astype(float)-n0.astype(float),n.astype(float)-n0.astype(float)),old_absolute_endpoint=measure_error(p1,n),old_reference_response=measure_error(p1.astype(float)-p0.astype(float),n.astype(float)-n0.astype(float)))
                predicted[name][role]=y; observed[name][role]=n; anchors[name][role]=n0
            del cache,stage,zero,new
            event('case_complete',case=name,reports=reports[name])
    names=list(predicted); factorial={}
    for role in ('image','text'):
        a,b=names; n=dict(zip(('00','01','10','11'),(anchors[a][role],observed[a][role],anchors[b][role],observed[b][role])))
        p=dict(zip(('00','01','10','11'),(anchors[a][role],predicted[a][role],anchors[b][role],predicted[b][role])))
        factorial[role]=audit_factorial(n,p)
    save(out/'forecasts.json',outputs)
    if any(x in sys.modules for x in ('torch','diffusers','transformers')): raise ValueError('model framework imported in mathematical development assay')
    summary=dict(schema='mdb-local-delta-diagnostic-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),reports=reports,factorial=factorial,
        model_loads=0,native_forwards=0,native_primitives=0,zero_edit_native_closure=True,seconds=time.monotonic()-started,mechanics_status='pending-independent-audit',terminal_status='unassessed',
        scope='Post-hoc algorithm-development test on known3/8 targets. Historical native half-dose ports explicitly calibrate local laws. Changed computations NumPy-only; no target intermediate used. Not fresh-state or semantic evidence.')
    save(out/'summary.json',summary); save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,precision,response):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    if read(precision/'verification-v1.json')['mechanics_status']!='verified' or read(response/'verification-v1.json')['mechanics_status']!='verified': raise ValueError('verified source studies required')
    roots={key:study/'collection-01/extracted/binding-minimal-result' for key,study in [('prefix',precision),('targets',response)]}; inputs={key:{} for key in roots}; remote={}
    for key,root in roots.items():
        inventory=read(root/'files.json'); remote[key]=read(root/'summary.json')['output_root']+'/binding-minimal-result'
        for name,digest in inventory.items():
            # Retain hashes of consumed arrays and typed source states; no copies.
            if name.startswith(('arrays/','store/','source/store/')) or name in ('native.json','forecasts.json','plan.json','parameter-artifacts.json','summary.json'):
                p=root/name
                if sha(p)!=digest: raise ValueError('source inventory differs')
                inputs[key][name]=dict(sha256=digest,bytes=p.stat().st_size)
    prior=read(precision/'collection-01/request.json'); h['bind_environment'](prior['execution_environment'])
    c=dict(input_roots=remote,inputs=inputs,output_scratch='/mnt/big/mdb/research',directory_custody=True,algorithm='native-primitive-anchors-plus-wide-finite-response-v1',
        scope='Saved known-case development; no held-out target or semantic claims. Zero model/framework/native calls.')
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',c); extras=[(here/n,n) for n in ('flux_family_custody.py','flux_reentry_reference.py','flux_local_delta_reference.py','analyze_flux_binding_minimal.py')]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py')]
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[*remote.values(),c['output_scratch'],'binding-minimal-result',read(roots['prefix']/'parameter-artifacts.json')['defaults']['checkpoint_root']]))
    extras.append((pre,pre.name)); request=dict(schema='mdb-local-delta-diagnostic-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',execution_environment=prior['execution_environment'],
        source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=4,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=4,extra=dict(task_family='analysis'),checkpoint_policy='Two receiver caches, zero-edit checks and two known3/8 calculations; all changed math NumPy. No native model/primitives/image calls.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request); stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted or source changed')
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-flux-local-delta-diagnostic',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=300,retry_on_kill=False,preflight=True,
        note='NumPy-only saved-case local finite-response development:2native half-dose calibration banks, zero-edit closure and2known3/8 states. No Torch imports/model loads/native primitives/images. Temporary exact checkpoint hydration deleted; CPU4BLASthreads; expectedunder90s/300souter,no retry. Intent-first resource sizing.')
    h['save'](run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1': worker()
    else:
        p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('precision',type=Path,nargs='?'); p.add_argument('response',type=Path,nargs='?'); p.add_argument('--submit',action='store_true'); a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.precision.resolve(),a.response.resolve()),sort_keys=True))
