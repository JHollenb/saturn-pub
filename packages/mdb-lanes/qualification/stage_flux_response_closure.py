"""Stage one finite two-target receiver-by-dose response-closure assay."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,study):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']
    read=lambda p:json.loads(p.read_text()); collection=study/'collection-04'; data=collection/'extracted/binding-minimal-result'
    old=read(data/'plan.json'); prior_request=read(collection/'request.json'); inventory=read(data/'files.json')
    rows={(row['id'],row['mode']):row for row in read(data/'rows.json')}
    if read(study/'verification-v1.json')['mechanics_status']!='verified': raise ValueError('verified source required')
    h['bind_environment'](prior_request['execution_environment']); files={}
    def add(source,destination):
        p=data/source
        if sha(p)!=inventory[source.as_posix()]: raise ValueError('source changed')
        item=dict(source=source.as_posix(),sha256=sha(p),bytes=p.stat().st_size)
        if destination in files and files[destination]['sha256']!=item['sha256']: raise ValueError('source collision')
        files[destination]=item
    def cut(fp,prefix):
        name=Path('store/cuts')/(fp+'.json'); record=read(data/prefix/name); add(prefix/name,name.as_posix())
        for page in record['slots'].values():
            name=Path('store/pages')/(page+'.json'); add(prefix/name,name.as_posix())
        for blob in record['physical_blobs']:
            name=Path('store/blobs')/(blob+'.safetensors'); add(prefix/name,name.as_posix())
    for fp in old['parents'].values(): cut(fp,Path('source'))
    history={}
    for name in ('half','half.image1'):
        history[name]={k:rows[name,'native']['target'][k] for k in ('entry','exit')}
        for fp in history[name].values(): cut(fp,Path())
    worker=here/'flux_response_closure.py'; api=runpy.run_path(str(worker))
    config=dict(schema='mdb-response-closure-plan-v1',**{k:old[k] for k in ('model','model_path','checkpoint_sha256','parents')},cases=api['cases'](),
        history=history,input_root=read(data/'summary.json')['output_root']+'/binding-minimal-result',input_files=files,
        output_scratch='/mnt/big/mdb/research',worker_wall_s=750,directory_custody=True,source_verification_sha256=sha(study/'verification-v1.json'),
        scope='Two new 3/8 text states; known native half-dose calibration anchors. Absolute vs receiver-specific delta anchoring. No new prompt/image/semantic/general speed claim.')
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',config); api['check_transaction_contract'](run/'transaction-api-preflight')
    extras=[(here/n,n) for n in ('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py','flux_response_reference.py','flux_response_campaign.py')]
    extras.extend([(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')])
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=['binding-minimal-result','source',config['input_root'],config['model_path'],config['output_scratch']]))
    extras.append((pre,pre.name)); raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    request=dict(schema='mdb-response-closure-request-v1',config=config,model=config['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        needs=dict(cuda=True,host='beast'),execution_environment=prior_request['execution_environment'],config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=2,logical_branches=2,logical_checkpoint_replays=2,max_local_work_items=2,
            checkpoint_policy='Seal eight complete forecasts: absolute, receiver-anchored, copy-anchor and other-receiver response for each target. Then two native J1 blocks. No suffix/VAE. Half-dose endpoints are known calibration.'),
        resource_exception=dict(ram_mb=10000,vram_mb=9500,disk_gb=2,calibration_job='job-96b46e23b8c7',measured_peak_rss_mb=8088.1,measured_peak_vram_mb=8424,
            reason='Same checkpoint, B1 and two serial J1 states; prior 37-port diagnostic used8088.1MBRSS/8424MBVRAM. Here no37-port retention; one receiver cache at a time, two response evaluations each. Fixed geometry only.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted or source changed')
    command=worker_command('flux_response_closure.py'); command.insert(1,'PYTHONDONTWRITEBYTECODE=1'); resource=r['resource_exception']
    job=guarded_mrun_launch(command,request=r,model=r['model'],task='forward',experiment='mdb-flux-response-closure',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],
        ram_mb=resource['ram_mb'],vram_mb=resource['vram_mb'],disk_gb=resource['disk_gb'],cpu_threads=4,est_wall_s=300,timeout_s=900,retry_on_kill=False,preflight=True,priority=70,
        note='Receiver-by-dose closure:2new3/8 text states; knownhalf-dose native anchors;absolute/anchored/copy-anchor/cross-receiver full I/T forecasts sealed then2native J1 blocks/4Acts/6receipts. B1CUDA BF16,1load,0suffix/VAE/enc/noise.750worker/900outer,no retry. Purpose:measure state-dependent arithmetic defect before scaling discovery.')
    h['save'](run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job),run=str(run))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('study',type=Path,nargs='?'); p.add_argument('--submit',action='store_true'); a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.study.resolve()),sort_keys=True))
