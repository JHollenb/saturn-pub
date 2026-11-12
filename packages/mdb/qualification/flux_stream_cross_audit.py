"""Near-data CPU audit and compact publication of the J2 stream corners."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def worker():
    payload=Path.cwd();request=read(os.environ['SATURN_DEBUG_REQUEST']);c=request['config']
    if read(os.environ['SATURN_RUN_CONFIG'])!=c or any(sha(payload/n)!=digest for n,digest in request['source_hashes'].items()):raise ValueError('audit source changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    collection=Path(c['remote_collection']);remote=read(collection/'remote-custody.json')
    if sha(collection/'remote-custody.json')!=c['remote_record_sha256'] or remote['verification']['request_sha256']!=c['native_request_sha256']:raise ValueError('native source differs')
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
    save(out/'request.json',request);save(out/'plan.json',c)
    audit=runpy.run_path(str(payload/'analyze_flux_stream_cross.py'))['analyze'](collection,out/'audit-v1.json')
    if audit['custody']!=remote['verification']:raise ValueError('complete native custody differs')
    data=collection/'extracted/binding-minimal-result';copied={}
    def copy(p,name):
        q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        if sha(p)!=sha(q):raise ValueError('compact projection copy differs')
        copied[name]=dict(sha256=sha(q),bytes=q.stat().st_size,remote_path=str(p))
    copy(collection/'remote-custody.json','remote-custody.json')
    for name in ('request.json','plan.json','summary.json','source-binding.json','consumed-source.json','rows.json','receipts.json','timeline.jsonl','parameter-artifacts.json','retrieve_weights.py','native-contract.json'):copy(data/name,'native-proof/'+name)
    save(out/'projection.json',dict(schema='mdb-compact-scientific-projection-v1',remote_collection=str(collection),remote_record_sha256=c['remote_record_sha256'],native_request_sha256=c['native_request_sha256'],copied=copied,
        scope='Partial scalar/observation/provenance projection. All raw states/arrays are remote references. Full current custody verified remotely; previous full source qualification inherited with fresh consumed-subset hashes.'))
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>32*1024**2:raise ValueError('compact projection exceeded bound')
    if any(n in sys.modules for n in ('torch','diffusers','transformers')):raise ValueError('saved-only CPU audit imported model framework')
    summary=dict(schema='mdb-stream-cross-remote-audit-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(out.parent),scientific_custody=audit['custody'],audit_sha256=sha(out/'audit-v1.json'),projection_sha256=sha(out/'projection.json'),
        seconds=time.monotonic()-started,mechanics_status='verified',instrument_status=audit['instrument_status'],terminal_status='unassessed',shams=audit['shams'],counts=dict(model_loads=0,native_calls=0,**audit['measurement_counts']),
        response_errors={label:{role:{case:{corner:value['reference_response']['relative_rms'] for corner,value in measured['corners'].items()} for case,measured in cases.items()} for role,cases in roles.items()} for label,roles in audit['measurements'].items()})
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,pointer,native_run):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));save=h['save'];record=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer)
    previous=read(native_run/'request.json');h['bind_environment'](previous['execution_environment'])
    if previous['request_sha256']!=record['verification']['request_sha256'] or read(native_run/'submission.json')['job_id']!=record['verification']['job_id']:raise ValueError('native job differs')
    run.mkdir(parents=True,exist_ok=False);c=dict(remote_collection=record['collection'],remote_record_sha256=read(pointer)['record_sha256'],native_request_sha256=previous['request_sha256'],output_scratch='/mnt/big/mdb/research',directory_custody=True);save(run/'plan.json',c)
    names=('flux_binding_minimal.py','flux_family_custody.py','analyze_flux_stream_cross.py','analyze_flux_consumer_trace.py','analyze_flux_mlp_forecast.py','analyze_flux_binding_minimal.py','summarize_context_window.py','flux_qualified_source.py','flux_stream_cross_math.py','flux_consumer_trace_math.py','flux_own_state_mlp_math.py')
    extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[name for _,name in extras],pre.name],path_token_allowlist=[c['remote_collection'],previous['config']['source_collection'],str(Path(previous['config']['source_audit']).parent),previous['config']['model_path'],c['output_scratch'],'binding-minimal-result']));extras.append((pre,pre.name))
    request=dict(schema='mdb-stream-cross-audit-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-audit-of-BF16',execution_environment=previous['execution_environment'],source_hashes={name:sha(p) for p,name in [(Path(__file__),Path(__file__).name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=64,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=64,extra=dict(task_family='analysis'),
            checkpoint_policy='Full current custody/typedActs/36receiptjoins/3shams/130original parameter recipes. Previous fullsource qualification inherited plus fresh allconsumedsubset hashes.64ownbaseline responses/16factorial/16finiteattentionwriteaccounts.0model/framework/imagecalls; compactprojectiononly.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=read(run/'request.json');h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=digest for n,digest in r['source_hashes'].items()):raise ValueError('changed or submitted audit')
    cmd=worker_command(Path(__file__).name);cmd.insert(1,'CUDA_VISIBLE_DEVICES=');cmd.insert(cmd.index('python')+1,'-B')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-flux-J2-stream-cross-audit',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,
        note='CPU independentJ2streamfactorial: fullcurrentcustody/typedActs/source/36receiptjoins/3shams/130originalparameterrecipes. Inheritedfullsourceaudit plusfreshconsumedsubsetverification.64responses/16factorial/16finitewriteaccounts.0native/framework/VRAM/images. Intentfirstsizing,expectedunder3min/600outer,noretries; compactscalarprojection.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('pointer',type=Path,nargs='?');p.add_argument('native_run',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.pointer.resolve(),a.native_run.resolve()),sort_keys=True))
