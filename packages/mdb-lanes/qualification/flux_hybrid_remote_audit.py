"""Audit full remote scientific custody; publish a small, explicitly partial local projection."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


def worker():
    payload=Path.cwd();request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text());c=request['config'];read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    if any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('auditor source differs')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    collection=Path(c['remote_collection']);raw=(collection/'remote-custody.json').read_bytes();remote=read(collection/'remote-custody.json')
    if hashlib.sha256(raw).hexdigest()!=c['remote_record_sha256'] or remote['verification']['job_id']!=c['native_job_id'] or remote['verification']['request_sha256']!=c['native_request_sha256']:raise ValueError('remote custody binding differs')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
    save(out/'request.json',request);save(out/'plan.json',c)
    audit=runpy.run_path(str(payload/'analyze_flux_hybrid_future.py'))['analyze'](collection,out/'audit-v1.json');verified=read(out/'audit-v1.json')
    if verified['custody']!=remote['verification'] or verified['custody']['archive_sha256']!=c['native_archive_sha256']:raise ValueError('scientific audit custody differs')
    publication=runpy.run_path(str(payload/'publish_flux_hybrid_future.py'))['publish'](collection,out/'audit-v1.json',out/'index-v1')
    data=collection/'extracted/binding-minimal-result';copied={}
    def copy(p,name):
        q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);digest=sha(q)
        if digest!=sha(p):raise ValueError('projection changed bytes')
        copied[name]=dict(sha256=digest,bytes=q.stat().st_size,remote_path=str(p))
    copy(collection/'remote-custody.json','remote-custody.json')
    for name in ('request.json','plan.json','summary.json','rows.json','receipts.json','native.json','forecasts.json','baselines.json','image-forecasts.json','timeline.jsonl','parameter-artifacts.json','retrieve_weights.py'):
        copy(data/name,'native-proof/'+name)
    for directory in ('write-transaction','image-transaction'):
        for p in sorted((data/directory).rglob('*')):
            if p.is_file():copy(p,'native-proof/'+p.relative_to(data).as_posix())
    for p in sorted(data.glob('*.png')):copy(p,'images/'+p.name)
    save(out/'projection.json',dict(schema='mdb-compact-scientific-projection-v1',remote_collection=str(collection),remote_record_sha256=c['remote_record_sha256'],native_job_id=c['native_job_id'],native_request_sha256=c['native_request_sha256'],archive_sha256=c['native_archive_sha256'],copied=copied,
        scope='Partial scientific projection: reports,14images,source and transaction metadata only. Full strict legacy custody and every inventoried tensor remain on Beast. References in native-proof point into remote evidence and are not complete local arrays/stores. Use authenticated --retrieve for exact members; the audit ran against the full remote collection.'))
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>128*1024**2:raise ValueError('compact projection exceeded declared bound')
    if any(n in sys.modules for n in ('torch','diffusers','transformers')):raise ValueError('audit imported model framework')
    summary=dict(schema='mdb-hybrid-remote-audit-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),scientific_native_job_id=c['native_job_id'],scientific_custody=verified['custody'],audit_sha256=sha(out/'audit-v1.json'),projection_sha256=sha(out/'projection.json'),seconds=time.monotonic()-started,mechanics_status='verified',instrument_status=verified['instrument_status'],terminal_status='unassessed',counts=dict(model_loads=0,native_calls=0,scientific_images=14),primary=audit['primary'],publication=publication)
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,pointer,native_run):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text())
    record=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer);v=read(pointer);prior=read(native_run/'request.json')
    if record['verification']['request_sha256']!=prior['request_sha256'] or record['verification']['job_id']!=read(native_run/'submission.json')['job_id']:raise ValueError('native audit source mismatch')
    h['bind_environment'](prior['execution_environment']);run.mkdir(parents=True,exist_ok=False)
    c=dict(remote_collection=record['collection'],remote_record_sha256=v['record_sha256'],native_job_id=record['verification']['job_id'],native_request_sha256=prior['request_sha256'],native_archive_sha256=record['verification']['archive_sha256'],output_scratch='/mnt/big/mdb/research',directory_custody=True,scope='Independent audit on full verified remote evidence. Partial report/image projection only; no native model computation.')
    save(run/'plan.json',c)
    names=('analyze_flux_hybrid_future.py','flux_family_custody.py','analyze_flux_mlp_forecast.py','analyze_flux_binding_minimal.py','summarize_context_window.py','publish_flux_hybrid_future.py')
    extras=[(here/n,n) for n in names]+[(run/'plan.json','plan.json'),(pointer,'source-remote-pointer.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['remote_collection'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name));request=dict(schema='mdb-hybrid-remote-audit-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-audit-of-BF16',execution_environment=prior['execution_environment'],source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=14,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=14,extra=dict(task_family='analysis'),checkpoint_policy='Verify full remote archive/log/source/wheels/typedstores/receipts/two seals; independently recompute responses/RGB/mixed effects.0native/frameworkcalls. Publish<128MB report/image projection; all raw evidence retained remote.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras);return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or changed stage')
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-flux-hybrid-independent-audit',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,
        note='Independent CPU audit of full remote scientific custody.14images/4states/2frozenprograms;pairedresponse/errorpower/RGB/lineage/seals verified,0native/frameworkcalls.Small<128MBreport/imageprojection;fullrawstaysBeast.Expected2-4min,600souter,noretries,intent-first.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('pointer',type=Path,nargs='?');p.add_argument('native_run',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.pointer.resolve(),a.native_run.resolve()),sort_keys=True))
