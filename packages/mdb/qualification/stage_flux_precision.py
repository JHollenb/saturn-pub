"""Stage a two-block, source-bound CUDA precision microscope."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,study):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    from mdb import parameter_artifacts
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text())
    collection=study/'collection-04';data=collection/'extracted/binding-minimal-result';old=read(data/'plan.json');request=read(collection/'request.json');inventory=read(data/'files.json');rows={(r['id'],r['mode']):r for r in read(data/'rows.json')}
    if read(study/'verification-v1.json')['mechanics_status']!='verified':raise ValueError('verified source required')
    h['bind_environment'](request['execution_environment']);files={}
    def add(source,destination):
        source=Path(source).as_posix()
        p=data/source
        if sha(p)!=inventory[source]:raise ValueError('source file differs')
        item=dict(source=source,sha256=inventory[source],bytes=p.stat().st_size)
        if destination in files and files[destination]['sha256']!=item['sha256']:raise ValueError('source collision')
        files[destination]=item
    def cut(fp,prefix):
        name='store/cuts/'+fp+'.json';record=read(data/prefix/name);add(prefix/name,name)
        for p in record['slots'].values():add(prefix/('store/pages/'+p+'.json'),'store/pages/'+p+'.json')
        for p in record['physical_blobs']:add(prefix/('store/blobs/'+p+'.safetensors'),'store/blobs/'+p+'.safetensors')
    for fp in old['parents'].values():cut(fp,Path('source'))
    history={}
    for name in('half','half.image1'):
        history[name]={k:rows[name,'native']['target'][k]for k in('entry','exit')}
        for fp in history[name].values():cut(fp,Path())
    config=dict(schema='mdb-precision-cut-plan-v1',**{k:old[k]for k in('model','model_path','checkpoint_sha256','parents')},input_root=read(data/'summary.json')['output_root']+'/binding-minimal-result',
        input_files=files,history=history,output_scratch='/mnt/big/mdb/research',worker_wall_s=900,directory_custody=True,source_verification_sha256=sha(study/'verification-v1.json'),
        scope='Two already measured receiver inputs; capture new internal ports and diagnose first divergence. Extra QK BF16 boundary is a candidate, not a confirmed fix. No new images or held-out claims.')
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',config);worker=here/'flux_precision_cut.py'
    runpy.run_path(str(worker))['check_transaction_contract'](run/'transaction-api-preflight')
    extras=[(here/n,n)for n in('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py','flux_response_reference.py','flux_response_campaign.py')]
    extras.extend([(Path(parameter_artifacts.__file__),'retrieve_weights.py'),(run/'plan.json','plan.json')])
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=['binding-minimal-result','source',config['input_root'],config['model_path'],config['output_scratch']]))
    extras.append((pre,pre.name));raw=(json.dumps(config,sort_keys=True)+'\n').encode()
    r=dict(schema='mdb-precision-cut-request-v1',config=config,model=config['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        needs=dict(cuda=True,host='beast'),execution_environment=request['execution_environment'],config_bytes_sha256=hashlib.sha256(raw).hexdigest(),source_hashes={n:sha(p)for p,n in[(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=2,logical_branches=2,logical_checkpoint_replays=2,max_local_work_items=2,checkpoint_policy='Seal both local arithmetic variants then2 native joint1 blocks,0native suffix/VAE,0encodes/noise; source replay equality and new internal port capture.'),
        resource_exception=dict(ram_mb=10000,vram_mb=9500,disk_gb=3,calibration_job='job-96b46e23b8c7',measured_peak_rss_mb=8088.1,measured_peak_vram_mb=8424,
            reason='Complete same-lane two-block37-port diagnostic measured8088.1MB peakRSS and8424MBVRAM. Future10GBRAM/9.5GBVRAM retain over1.9GB/1GB headroom plus strict admission ceiling. Four mathematical preparations and both serial native full-port captures are included in that measurement;0VAEforward. Applies only to this fixed two-input geometry, not image-future campaigns. Executed successful request remains13GBRAM/10.5GBVRAM.'))
    r['request_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',r)
    stage_worker(run/'payload',worker=worker,config=config,request=r,config_bytes=raw,additional_files=extras)
    return dict(run=str(run),request_sha256=r['request_sha256'],counts=runpy.run_path(str(worker))['expected_counts']())


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if(run/'submission.json').exists()or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or changed stage')
    command=worker_command('flux_precision_cut.py');command.insert(1,'PYTHONDONTWRITEBYTECODE=1');resources=r['resource_exception']
    job=guarded_mrun_launch(command,request=r,model=r['model'],task='forward',experiment='mdb-flux-precision-cut',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],
        ram_mb=resources['ram_mb'],vram_mb=resources['vram_mb'],disk_gb=resources['disk_gb'],cpu_threads=4,est_wall_s=600,timeout_s=1200,retry_on_kill=False,preflight=True,priority=70,
        note='Klein CUDA BF16 B1 precision cut:2historical receiver inputs,4 sealed mathematical variants then2native joint1 blocks/4Acts/6receipts. Capture norm/affine/QKV/RoPE/attention/MLP; conditional probes explicitly native-input-assisted.0suffix/VAE/enc/noise,1load;900worker/1200outer,no retry.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job),run=str(run))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('study',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args()
    print(json.dumps(submit(a.run.resolve())if a.submit else stage(a.run.resolve(),a.study.resolve()),sort_keys=True))
