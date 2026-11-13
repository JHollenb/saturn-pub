"""Admit a finite continuation using forecasts preserved from a timeout."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,source):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text())
    data=source/'extracted/binding-minimal-result';old=read(data/'plan.json');original=read(source/'request.json');recovery=read(data/'recovery.json');inventory=recovery['original_inventory']
    if read(source/'job.json')['state']!='timeout'or read(source/'forensic-custody.json')['new_model_calls']!=0:raise ValueError('terminal forensic source required')
    h['bind_environment'](original['execution_environment'])
    exit=read(data/'ports-base.json')['exit'];names={'forecasts.json','timeline.jsonl','parameter-artifacts.json','retrieve_weights.py','ports-base.json','completed-base.native.json'}
    names.update(n for n in inventory if n.startswith(('arrays/','transaction/')))
    for fp in [*old['parents'].values(),exit]:
        name='store/cuts/'+fp+'.json';names.add(name);cut=read(data/name)
        names.update('store/pages/'+v+'.json'for v in cut['slots'].values());names.update('store/blobs/'+v+'.safetensors'for v in cut['physical_blobs'])
    files={}
    for name in sorted(names):
        item=inventory[name]
        if sha(data/name)!=item['sha256']or(data/name).stat().st_size!=item['bytes']:raise ValueError('forensic source changed')
        files[name]=item
    seal=next(json.loads(l)['seal']for l in(data/'timeline.jsonl').read_text().splitlines()if json.loads(l)['kind']=='sealed')
    config=dict(old,schema='mdb-response-prepared-plan-v1',input_root=read(source/'forensic-custody.json')['remote_snapshot'],input_files=files,
        output_scratch='/mnt/big/mdb/research',control_exit=exit,forecast_seal_sha256=seal['seal_sha256'],source_job=recovery['report']['job_id'],source_state='timeout',
        source_inventory_sha256=sha(data/'recovery.json'),worker_wall_s=2700,
        continuation_scope='No forecasts recalculated. Reuse completed base.native; execute9remaining images, predicted before corresponding new native target. Group25/28 native operations into atomic continue(n) at committed boundaries; every native transition/footprint still validated, no intermediate full cuts promised. Original partial predicted attempt remains evidence. NVMe output store avoids spinning-disk small-file pressure.')
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);worker=here/'flux_response_validate.py'
    extras=[(here/n,n)for n in('flux_binding_minimal.py','flux_family_custody.py')]+[(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=['binding-minimal-result','source',config['input_root'],config['model_path'],config['output_scratch']]))
    extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request=dict(schema='mdb-response-prepared-request-v1',config=config,model=config['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        needs=dict(cuda=True,host='beast'),execution_environment=original['execution_environment'],config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p)for p,n in[(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=9,logical_branches=9,logical_checkpoint_replays=9,max_local_work_items=9,checkpoint_policy='Consume original sealed predictions;9serial remaining native suffix/VAE futures; no new numerical predictor, encodes, noise or retries.'),
        resource_exception=dict(ram_mb=10000,vram_mb=10500,disk_gb=2,calibration_job='job-ec8810b4b5d4',measured_peak_rss_mb=8172.8,measured_peak_vram_mb=9274,
            replication_job='job-3b50f8b6e026',replication_peak_vram_mb=9266,reason='Same resident noencoder native B1 lane without NumPy weight bank/caches.10.5GB VRAM reservation leaves1.226GB above measured9.274GB; guard ceiling11.55GB. Independent resource review supports this sizing. NVMe output;2700worker/3300outer finite cap.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'],counts=runpy.run_path(str(worker))['expected_counts']())


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if(run/'submission.json').exists()or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already admitted or changed stage')
    cmd=worker_command('flux_response_validate.py');cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    resource=r['resource_exception']
    job=guarded_mrun_launch(cmd,request=r,model=r['model'],task='forward',experiment='mdb-flux-response-prepared',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],ram_mb=resource['ram_mb'],vram_mb=resource['vram_mb'],disk_gb=resource['disk_gb'],cpu_threads=4,est_wall_s=2400,timeout_s=3300,retry_on_kill=False,preflight=True,priority=70,
        note='Reviewed finite continuation from timeout: consume5unchanged sealed forecasts;9serial RGB futures,481 native transitions,18 Acts,40 grouped receipts,436 blocks,4 new targets.1load0enc/noise/newpredictions; NVMe store. Groups25/28 retain all transition validation and commit cuts. New predicted futures precede corresponding targets. Retain original timeout/partial work. Stop2700worker/3300outer; no retries.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job),run=str(run))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('source',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve())if a.submit else stage(a.run.resolve(),a.source.resolve()),sort_keys=True))
