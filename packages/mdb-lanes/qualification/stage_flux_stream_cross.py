"""Freeze the missing J2 stream corners with matched sham continuations."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,study):
    from mdb_lanes import parameter_artifacts
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));save,sha=h['save'],h['sha'];read=lambda p:json.loads(p.read_text())
    proof=read(study/'verification-v1.json');previous=read(study/'run-02/request.json');pointer=read(study/'native-remote-pointer-v1.json');local=study/'audit/collection-01/extracted/binding-minimal-result';audit=read(local/'audit-v1.json')
    if proof['mechanics_status']!='verified' or proof['remote_scientific_custody']!=pointer['record']['verification'] or audit['custody']!=pointer['record']['verification'] or any(v!='qualified' for v in audit['replay_status'].values()):raise ValueError('qualified source trace required')
    auditjob=read(study/'audit-job-v1.json')['job_id'];traces=read(local/'native-proof/traces.json');by={(x['id'],x['mode']):x for x in traces};base=previous['config'];h['bind_environment'](previous['execution_environment'])
    c=dict(schema='mdb-stream-cross-plan-v1',source_collection=pointer['record']['collection'],source_record_sha256=pointer['record_sha256'],source_audit='/mnt/big/mdb/research/'+auditjob+'/binding-minimal-result/audit-v1.json',source_audit_sha256=sha(local/'audit-v1.json'),
        origins={name:{mode:by[name,mode]['origin'] for mode in ('native','old','new')} for name in ('half','row4heavy','row11heavy')},
        branches=[dict(id=name,corner=corner,slot=slot) for name in ('half','row4heavy','row11heavy') for corner,slot in [('00','image'),('10','image'),('01','text')]],
        **{key:base[key] for key in ('model','model_path','checkpoint_sha256')},output_scratch='/mnt/big/mdb/research',worker_wall_s=600,directory_custody=True,
        scope='Causal old/new image-by-text input corners at step2/J2 on the previously observed fresh receiver. Six missing crossings plus3old/old sham controls; each condition retains its own half baseline. Native J2 attention/exit, projected noise and commit3 measured. Not an independent prediction, new semantic address, new prompt/seed or better-editor claim.0VAE/images.',
        hypotheses=dict(architectures=['Incoming image alone accounts for the diagonal change.','Incoming text alone accounts for the diagonal change.','Both finite stream changes and/or their interaction contribute.'],
            account='For R_ab=Y_ab(case)-Y_ab(half), A=R10-R00, B=R01-R00, H=R11-R10-R01+R00. Revised error=(R00-native)+A+B+H. Retain signed powers/cross, per-corner errors and finite attention writes U-X.',
            source_contract='Inherit prior full archive/store qualification and rehash every consumed source file against that authenticated inventory. Do not claim a new full archive verification.',
            controls='Three same-slot old/old sham Acts must reproduce prior J2 image/text attention archives and complete J2exit/noise/commit payloads. Failures remain per-case instrument diagnostics; retain all completed branches.'))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c);worker=here/'flux_stream_cross.py'
    names=('flux_binding_minimal.py','flux_family_custody.py','flux_hybrid_future.py','flux_trace_parameter_recipes.py','flux_qualified_source.py')
    extras=[(here/name,name) for name in names]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[name for _,name in extras],pre.name],path_token_allowlist=[c['source_collection'],str(Path(c['source_audit']).parent),c['model_path'],c['output_scratch'],'binding-minimal-result']));extras.append((pre,pre.name))
    request=dict(schema='mdb-stream-cross-request-v1',config=c,model=c['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,needs=dict(cuda=True,host='beast'),execution_environment=previous['execution_environment'],source_hashes={name:sha(p) for p,name in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=225,logical_branches=9,logical_checkpoint_replays=3,max_local_work_items=234,checkpoint_policy='9branches×25native=225steps/207blocks;9Acts/36receipts.6missing streamcorners+3oldoldshams. CaptureonlyJ2midpoint/exit/readoutnoise/commit.0VAE/images/prefixes/encodes/noise. One sequentialB1CUDAlease,noretries.'),
        resource_exception=dict(ram_mb=13312,vram_mb=9500,cpu_threads=4,disk_gb=6,est_wall_s=180,calibration_job='job-8f0f405ff25e',measured_peak_rss_mb=7813.1,measured_peak_vram_mb=8212.,
            reason='Sameoriginal model/CUDA B1/native225step geometry measured177.95s/7813MBRSS/8212MBVRAM. Preserve13GBsuccessful setup reservation per advisor;9.5GBVRAM headroom. Earlier6GBkilled beforebranches. Sparsecapture reduces2.375GB fulltrace evidence; expected~400MB. One model/9branches,180sestimate/600worker/900outer,noretries.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'],expected_counts=runpy.run_path(str(worker))['counts']())


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=digest for n,digest in r['source_hashes'].items()):raise ValueError('changed or submitted cross')
    job=guarded_mrun_launch(worker_command('flux_stream_cross.py'),request=r,model=r['model'],task='forward',experiment='mdb-flux-J2-stream-cross',payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=900,retry_on_kill=False,preflight=True,
        **{k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')},
        note='J2image×text:6missingold/newcorners+3oldoldshams withownhalfbaselines.225native transitions/207blocks/9Acts/36receipts. J2midpoint/exit+noise+commit,0VAE/images/prefix/encodes/noise. OriginalB1profile,13GBRAM/9.5GBVRAM calibrated7813/8212MB;180sestimate/600worker/900outer,noretries. Sourceinheritedfullcustody+freshconsumedsubsetchecks;allcountertrendsretained.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('study',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.study.resolve()),sort_keys=True))
