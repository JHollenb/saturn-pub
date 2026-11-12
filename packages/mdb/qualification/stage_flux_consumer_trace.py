"""Freeze a nine-branch, 25-step native consumer microscope from audited saved exits."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,study):
    from mdb import parameter_artifacts
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));save,sha=h['save'],h['sha'];read=lambda p:json.loads(p.read_text())
    proof=read(study/'verification-v1.json');original=read(study/'run-02/request.json');pointer=read(study/'native-remote-pointer-v1.json')
    if proof['mechanics_status']!='verified' or proof['remote_scientific_custody']!=pointer['record']['verification']:raise ValueError('verified source futures required')
    local=study/'audit/collection-01/extracted/binding-minimal-result';audit=read(local/'audit-v1.json')
    if audit['custody']!=pointer['record']['verification'] or audit['independent_forecast_field_comparisons']!=72:raise ValueError('original independent replay not joined')
    auditjob=read(study/'audit-job-v1.json')['job_id'];base=original['config'];h['bind_environment'](original['execution_environment'])
    c=dict(schema='mdb-consumer-trace-plan-v1',source_collection=pointer['record']['collection'],source_record_sha256=pointer['record_sha256'],
        source_audit='/mnt/big/mdb/research/'+auditjob+'/binding-minimal-result/audit-v1.json',source_audit_sha256=sha(local/'audit-v1.json'),
        branches=[{key:row[key] for key in ('id','mode','entry','commits')} for row in read(local/'native-proof/rows.json')],
        **{key:base[key] for key in ('model','model_path','checkpoint_sha256')},output_scratch='/mnt/big/mdb/research',worker_wall_s=600,directory_custody=True,
        scope='Retrospective native microscope of nine already observed own-state futures, not a fresh prediction. Replay J1 exits through next commit; full-payload closure to archived commits qualifies each row. All 23 block boundaries,3 joint attention midpoints,readout norm/noise and Euler retained. No new intervention, image, prompt, prefix, noise,calibration or model-law choice.',
        hypotheses=dict(localization='Locate first observed image/text paired-response ranking change, retaining all points and cases. An isolated change or large write term is an observational lead, not an accuracy gate or causal percentage.',
            recurrence='Use actual finite post-minus-pre writes in the same coordinates; retain carried/write powers and signed crossmoment. Stop residual recurrence at readout coordinate changes.',
            euler='Retain actual incoming latent/noise/sigmas/outgoinglatent. Verify whether incoming latent and schedules are common; account for h*noise response and finite scheduler remainder separately.',
            continuation='Intermediate writes are qualified as the old image futures only when complete next-commit payloads replay exactly. Per-row mismatches remain evidence and do not erase other captures.'))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    worker=here/'flux_consumer_trace.py';names=('flux_binding_minimal.py','flux_family_custody.py','flux_hybrid_future.py','flux_trace_parameter_recipes.py')
    extras=[(here/name,name) for name in names]+[(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[name for _,name in extras],pre.name],path_token_allowlist=[c['source_collection'],str(Path(c['source_audit']).parent),c['model_path'],c['output_scratch'],'binding-minimal-result']));extras.append((pre,pre.name))
    request=dict(schema='mdb-consumer-trace-request-v1',config=c,model=c['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,needs=dict(cuda=True,host='beast'),execution_environment=original['execution_environment'],source_hashes={name:sha(p) for p,name in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=225,logical_branches=9,logical_checkpoint_replays=9,max_local_work_items=225,
            checkpoint_policy='Nine saved J1-exits replay25native steps each;225receipts/207blocks,27jointattention midpoints,9norm+9projection+9Euler,0Acts/VAE/images/prefixes/noise/encodes. Sequential branches, typed CAS state at every endpoint; complete archived commit closure per row.'),
        resource_exception=dict(ram_mb=9216,vram_mb=9500,cpu_threads=4,disk_gb=10,est_wall_s=240,calibration_job='job-8f0f405ff25e',measured_peak_rss_mb=7813.1,measured_peak_vram_mb=8212.,
            reason='Completed exact9branch/225step trace job-8f0f405ff25e measured177.95s/7813.1MBRSS/8212MBVRAM under13GBRAM/10.5GBVRAM. Future9GBRAM/9.5GBVRAM retain measured setup headroom. Earlier6GBattempt killed during setup at7145MB; historical4.1GBsource measurement was not applicable sizing for this setup. Sequential branches,noVAEcalls,2.37GB typed evidence, parameterrecipesonly.240sestimate/600worker/900outer,noretries. Executed13GBrequest remains immutable.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=worker,config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'],expected_counts=runpy.run_path(str(worker))['counts']())


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=digest for n,digest in r['source_hashes'].items()):raise ValueError('changed or submitted microscope')
    job=guarded_mrun_launch(worker_command('flux_consumer_trace.py'),request=r,model=r['model'],task='forward',experiment='mdb-flux-consumer-microscope',payload=run/'payload',gpu=True,needs=r['needs'],prefer_host='beast',timeout_s=900,retry_on_kill=False,preflight=True,
        **{key:r['resource_exception'][key] for key in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')},
        note='Nine exact saved J1-exits through step2 remainder:225individual transitions/207blocks/225receipts,27jointattentionmidpoints+readoutnorm/noise+Euler.0Acts/images/VAE/prefix/encoder/noise. Per-row fullpayload closure to archivedcommit3. Exactcompleted job-8f0f405ff25e measured178s/7813MBRSS/8212MBVRAM; future9GBRAM/9.5GBVRAM retain setup headroom.240sestimate/600worker/900outer,noretries. Typed arrays nearBeast,compactauditprojection later.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('study',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.study.resolve()),sort_keys=True))
