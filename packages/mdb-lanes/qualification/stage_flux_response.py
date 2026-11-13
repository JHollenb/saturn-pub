"""Stage one finite resident response-law assay from audited carrier prefixes."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.execution_environment import intent
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent;helper=runpy.run_path(str(here/'flux_binding_minimal.py'));save,sha=helper['save'],helper['sha']
    read=lambda p:json.loads(p.read_text())
    data=source/'collection-01/extracted/binding-minimal-result';a=read(source/'audit-v1.json');old=read(data/'plan.json');rows={r['id']:r for r in read(data/'rows.json')}
    if a['mechanics_status']!='verified':raise ValueError('audited observed prefix required')
    helper['bind_environment'](read(source/'run-01/request.json')['execution_environment'])
    parents={role:rows[name]['marks']['joint0_exit']for role,name in(('base','control'),('edited','positive'))}
    names=set();manifest=read(data/'files.json')
    for fp in parents.values():
        name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name)
        names.update('store/pages/'+v+'.json'for v in cut['slots'].values());names.update('store/blobs/'+v+'.safetensors'for v in cut['physical_blobs'])
    files={}
    for name in sorted(names):
        if sha(data/name)!=manifest[name]:raise ValueError('source cut CAS differs')
        files[name]=dict(bytes=(data/name).stat().st_size,sha256=manifest[name])
    worker=here/'flux_response_campaign.py';code=runpy.run_path(str(worker))
    config=dict(schema='mdb-response-plan-v1',**{k:old[k]for k in('model','model_path','checkpoint_sha256')},parents=parents,output_scratch='/mnt/big/mdb/research',worker_wall_s=900,
        input_root=read(data/'summary.json')['output_root']+'/binding-minimal-result',input_files=files,cases=code['cases'](),directory_custody=True,
        baseline_png_sha256=rows['control']['png_sha256'],source_audit_sha256=sha(source/'audit-v1.json'),
        hypotheses=dict(local='Partitioned attention plus recomputed receiving MLP predicts complete joint1 I/T. Cache and all variant outputs calculated from weights and observed input before any target block.',
            continuation='Predicted-write futures should closely follow corresponding native futures; report signed/full discrepancies without an arbitrary pass threshold.',
            categorical={c['id']:c['categorical']for c in code['cases']()},signed_donor_effect={c['id']:'positive vs matched base'for c in code['cases']()if c['categorical']is None},
            scope='One checkpoint/seed/clock. New working-row compositions, dose and receiver; no new prompt/encoder prediction. Baseline known. New predicted PNG precedes its corresponding native joint1 reference; shared baseline supplies only clock/unchanged closure.'))
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config)
    extras=[(here/n,n)for n in('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py','flux_response_reference.py','flux_response_validate.py')]
    extras.extend([(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')])
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=['binding-minimal-result','input-data',config['input_root'],config['model_path'],config['output_scratch']]))
    extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request=dict(schema='mdb-response-request-v1',config=config,model=config['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        needs=dict(cuda=True,host='beast'),execution_environment=intent(),config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p)for p,n in[(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=10,logical_branches=10,logical_checkpoint_replays=10,max_local_work_items=10,
            checkpoint_policy='Five independent calculated joint1 writes; ten serial native suffix/VAE futures in one resident lease; no encodes/noise/retries.'),
        resource_exception=dict(ram_mb=13000,vram_mb=10500,disk_gb=2,calibration_job='job-ec8810b4b5d4',measured_native_peak_rss_mb=8172.8,measured_native_peak_vram_mb=9274,
            reason='Same resident noencoder B1 native lane, plus bounded CPU joint1 FP32 weights, two attention caches and sequential FP64 workspaces (under3GBextra from matrix geometry).13GBRAM/10.5GBVRAM; NVMe output. CPU increment is a declared sizing allowance; three measured native VRAM peaks9246–9274MB.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'],counts=code['expected_counts'](5))


def submit(run,source=None):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    here=Path(__file__).resolve().parent;helper=runpy.run_path(str(here/'flux_binding_minimal.py'));r=json.loads((run/'request.json').read_text())
    helper['bind_environment'](r['execution_environment'])
    if(run/'submission.json').exists()or any(helper['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already admitted or mutated stage')
    cmd=worker_command('flux_response_campaign.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    resource=r['resource_exception']
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-flux-response-program',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],
        ram_mb=resource['ram_mb'],vram_mb=resource['vram_mb'],disk_gb=resource['disk_gb'],cpu_threads=4,est_wall_s=600,timeout_s=1500,retry_on_kill=False,preflight=True,priority=70,
        note='Klein BF16 B1: pure NumPy partitioned attention + MLP predicts5joint1states before any target.10serial RGB futures,535native transitions,20Acts,45grouped receipts,0encodes/noise,1load. New predicted images precede their native references. NVMe output; same commit grouping qualified by prepared job326e9b. CPUcache/validation measured separately. Stop900worker/1500outer; no retries.')
    helper['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job),run=str(run))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('source',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps((submit if a.submit else stage)(a.run.resolve(),a.source.resolve()),sort_keys=True))
