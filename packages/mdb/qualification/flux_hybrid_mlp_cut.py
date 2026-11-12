"""Separate inherited receiving-state error from conditional MLP-law discrepancy."""
import argparse
import hashlib
import io
import json
import os
import runpy
import sys
import tempfile
import time
from pathlib import Path


def mlp(value,stages,weights,role,reference,law):
    import numpy as np
    ff='ff' if role=='image' else 'ff_context';B=reference['bf16']
    paired,_=law['affine_delta'](value,stages[role+'.mlp_input'],stages[role+'.mlp_paired'],weights[ff+'.linear_in.weight'],B)
    u,v=np.split(paired,2,axis=-1);activation=B(reference['_silu'](u)*v)
    result,_=law['affine_delta'](activation,stages[role+'.mlp_activation'],stages[role+'.mlp'],weights[ff+'.linear_out.weight'],B)
    return result


def worker():
    payload=Path.cwd();request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text());c=request['config'];h=runpy.run_path(str(payload/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text())
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('frozen source changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    from mdb.parameter_artifacts import restore,validate_manifest
    from mdb.tensor_archive import load_tensor_archive
    collection=Path(c['remote_collection']);remote=read(collection/'remote-custody.json');data=collection/'extracted/binding-minimal-result'
    if sha(collection/'remote-custody.json')!=c['remote_record_sha256'] or sha(Path(c['source_audit']))!=c['source_audit_sha256']:raise ValueError('source custody/audit changed')
    source_audit=read(Path(c['source_audit']))
    if source_audit['mechanics_status']!='verified' or source_audit['custody']!=remote['verification']:raise ValueError('verified source cohort differs')
    for name in ('plan.json','native.json','forecasts.json','baselines.json','parameter-artifacts.json'):
        if sha(data/name)!=remote['files'][name]['sha256']:raise ValueError('source record changed')
    plan,native,forecasts,baselines=[read(data/n) for n in ('plan.json','native.json','forecasts.json','baselines.json')]
    r=runpy.run_path(str(payload/'flux_reentry_reference.py'));law=runpy.run_path(str(payload/'flux_receiver_norm_prediction.py'))
    prior=read(collection/'request.json')
    if any(request['source_hashes'][name]!=prior['source_hashes'][name] for name in ('flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py')):raise ValueError('conditional subprogram differs from forecast')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();save(out/'plan.json',c);save(out/'request.json',request)
    manifest=validate_manifest(read(data/'parameter-artifacts.json'));save(out/'parameter-artifacts.json',manifest)
    import shutil
    shutil.copyfile(payload/'retrieve_weights.py',out/'retrieve_weights.py')
    arrays={}
    def array(value):
        value=np.ascontiguousarray(value,np.float32)
        if not np.isfinite(value).all() or np.any(value.view(np.uint32)&65535):raise ValueError('expected finite BF16')
        bits=(value.view(np.uint32)>>16).astype('<u2');buffer=io.BytesIO();np.save(buffer,bits,allow_pickle=False);raw=buffer.getvalue();digest=hashlib.sha256(raw).hexdigest();p=out/(digest+'.npy')
        if digest not in arrays:
            with p.open('xb') as f:f.write(raw)
            arrays[digest]=dict(path=p.name,sha256=digest,shape=list(value.shape),encoding='bf16-u16-npy')
        return arrays[digest]
    def arr(root,record):
        name=(root/record['path']).relative_to(data).as_posix()
        if remote['files'][name]['sha256']!=record['sha256']:raise ValueError('source array not in verified cohort')
        return load_tensor_archive(root,record)
    conditional={};anchors={};reconstruction={};calls=0
    with tempfile.TemporaryDirectory(prefix='hybrid-mlp-constants-',dir=scratch) as tmp:
        save(out/'weight-restoration.json',restore(manifest,tmp,manifest['defaults']['checkpoint_root']));weights={}
        for item in manifest['artifacts']:
            ports=[p for p in item['parameter_ports'] if p in {'ff.linear_in.weight','ff.linear_out.weight','ff_context.linear_in.weight','ff_context.linear_out.weight'}]
            if ports:
                bits=np.load(Path(tmp)/item['path'],allow_pickle=False);value=(bits.astype(np.uint32)<<16).view(np.float32)
                for port in ports:weights[port]=value
        for anchor in ('half','half.image1'):
            stages={role+'.'+field:arr(data/'source/prefixes',plan['anchors'][anchor]['stages'][role+'.'+field]) for role in ('image','text') for field in ('mlp_input','mlp_paired','mlp_activation','mlp')};anchors[anchor]={}
            for role in ('image','text'):
                C0=mlp(stages[role+'.mlp_input'],stages,weights,role,r,law);calls+=1;anchors[anchor][role]=array(C0)
                for mode in ('control','hybrid'):
                    record=baselines[anchor][mode]['fields'][role];actual=mlp(arr(data,record['mlp_input']),stages,weights,role,r,law);calls+=1
                    reconstruction[anchor+'.P0.'+role+'.'+mode]=bool(np.array_equal(actual,arr(data,record['mlp'])))
                for spec in [s for s in plan['cases'] if s['anchor']==anchor]:
                    name=spec['id'];C1=mlp(arr(data,native[name]['stages'][role+'.mlp_input']),stages,weights,role,r,law);calls+=1;conditional.setdefault(name,{})[role]=array(C1)
                    for mode in ('control','hybrid'):
                        record=forecasts[name][mode]['fields'][role];actual=mlp(arr(data,record['mlp_input']),stages,weights,role,r,law);calls+=1
                        reconstruction[name+'.P1.'+role+'.'+mode]=bool(np.array_equal(actual,arr(data,record['mlp'])))
            print('MDB_MLP_CUT '+json.dumps(dict(anchor=anchor,seconds=time.monotonic()-started,mlp_calculations=calls)),flush=True)
    save(out/'conditional.json',dict(anchors=anchors,cases=conditional));save(out/'reconstruction.json',reconstruction)
    audit=runpy.run_path(str(payload/'analyze_flux_hybrid_mlp.py'))['analyze'](out,data,remote)
    if any(n in sys.modules for n in ('torch','diffusers','transformers')):raise ValueError('diagnostic imported model framework')
    if calls!=36 or len(reconstruction)!=24 or len(conditional)!=4 or len(anchors)!=2:raise ValueError('diagnostic census differs')
    summary=dict(schema='mdb-hybrid-mlp-cut-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),seconds=time.monotonic()-started,counts=dict(model_loads=0,native_calls=0,images=0,mlp_calculations=calls,conditional_outputs=12,owned_reconstructions=24),mechanics_status=audit['mechanics_status'],instrument_status='Saved target MLP inputs supplied to conditional diagnostic. Own forecast writes rebuilt separately; no new predictions or image futures.',terminal_status='unassessed',source_custody=remote['verification'])
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,pointer,projection,native_run):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text());record=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer)
    prior=read(native_run/'request.json');h['bind_environment'](prior['execution_environment']);audit=read(projection/'audit-v1.json')
    if audit['mechanics_status']!='verified' or audit['custody']!=record['verification']:raise ValueError('verified fresh comparison required')
    run.mkdir(parents=True,exist_ok=False);source_root=read(projection/'summary.json')['output_root']+'/binding-minimal-result'
    c=dict(remote_collection=record['collection'],remote_record_sha256=read(pointer)['record_sha256'],source_audit=source_root+'/audit-v1.json',source_audit_sha256=sha(projection/'audit-v1.json'),output_scratch='/mnt/big/mdb/research',directory_custody=True,
        hypothesis='Conditional true-input MLP law separates inherited receiving-input propagation from remaining MLP numerical-law discrepancy. C shared across control/hybrid. Preserve endpoint and paired-response split with signed cross terms; no positive attribution percentages.',scope='Post-hoc diagnostic on4known targets. Actual native target MLPinputs explicitly supplied toC1; not independent prediction, native causal restoration or new image test. Rebuild24owned P0/P1 writes using unchanged law; retain12C0/C1outputs and all errors.0native/model/framework/image calls. Temporary original parameters restored by verified recipes then removed.')
    save(run/'plan.json',c);extras=[(here/n,n) for n in ('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py','flux_receiver_norm_prediction.py','flux_arithmetic_reference.py','flux_attention_anchor.py','analyze_flux_hybrid_mlp.py')]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['remote_collection'],source_root,c['output_scratch'],'binding-minimal-result',prior['config']['model_path']]))
    extras.append((pre,pre.name));request=dict(schema='mdb-hybrid-mlp-cut-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',execution_environment=prior['execution_environment'],source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=36,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=36,extra=dict(task_family='analysis'),checkpoint_policy='Known-input conditional MLP diagnostic,12fullC0/C1outputs plus24exactownedMLP reconstructions;0nativecalls/images. Preserve signed endpoint/response cross terms; no target input in prediction claims.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras);return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or source changed')
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-flux-hybrid-mlp-cut',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,note='Savedinput MLPsplit:12C0/C1outputs+24ownedP0/P1exactreconstructions,36MLPevals. Conditionalnativeinputsexplicitdiagnostic,0native/model/framework/images.4CPUthreads,temporarycheckpointweightsremoved,expected2-4min/600outer,noretries,intent-first.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
        for name in ('pointer','projection','native_run'):p.add_argument(name,type=Path,nargs='?')
        p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.pointer.resolve(),a.projection.resolve(),a.native_run.resolve()),sort_keys=True))
