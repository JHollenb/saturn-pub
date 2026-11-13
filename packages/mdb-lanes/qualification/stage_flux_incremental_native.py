"""Stage a finite exact-reuse qualification, not a zero-native predictor."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path


def stage(run,response,precision):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes import parameter_artifacts
    here=Path(__file__).resolve().parent; h=runpy.run_path(str(here/'flux_binding_minimal.py')); sha,save=h['sha'],h['save']; read=lambda p:json.loads(p.read_text())
    if read(response/'audit-v1.json')['mechanics_status']!='verified' or read(precision/'verification-v1.json')['mechanics_status']!='verified': raise ValueError('verified parents required')
    roots={key:study/'collection-01/extracted/binding-minimal-result' for key,study in (('targets',response),('prefixes',precision))}
    inventories={key:read(path/'files.json') for key,path in roots.items()}; files={}
    def add(key,name):
        name=Path(name); p=roots[key]/name; digest=inventories[key][name.as_posix()]
        if sha(p)!=digest: raise ValueError('source changed')
        files[key+'/'+name.as_posix()]=dict(root=key,source=name.as_posix(),sha256=digest,bytes=p.stat().st_size)
    def cut(key,fp):
        name=Path('store/cuts')/(fp+'.json'); record=read(roots[key]/name); add(key,name)
        for page in record['slots'].values(): add(key,Path('store/pages')/(page+'.json'))
        for blob in record['physical_blobs']: add(key,Path('store/blobs')/(blob+'.safetensors'))
    add('prefixes','parameter-artifacts.json')
    native=read(roots['targets']/'native.json'); old=read(roots['prefixes']/'native.json'); plan=read(roots['targets']/'plan.json'); cases=[]
    keys=['image.'+key for key in ('norm1','modulated','q_projection','k_projection','v_projection','q_normalized','k_normalized')]
    for spec in plan['cases']:
        name=spec['id']; anchor=spec['anchor']; stages={k:old[anchor]['stages'][k] for k in keys}
        for record in stages.values(): add('prefixes',record['path'])
        cut('prefixes',old[anchor]['entry']); cut('targets',native[name]['entry']); cut('targets',native[name]['exit'])
        cases.append(dict(id=name,entry=native[name]['entry'],exit=native[name]['exit'],prefix_entry=old[anchor]['entry'],stages=stages))
    prior=read(response/'collection-01/request.json'); h['bind_environment'](prior['execution_environment'])
    config=dict(schema='mdb-incremental-native-plan-v1',**{k:plan[k] for k in ('model','model_path','checkpoint_sha256')},cases=cases,input_files=files,
        input_roots={key:read(path/'summary.json')['output_root']+'/binding-minimal-result' for key,path in roots.items()},output_scratch='/mnt/big/mdb/research',directory_custody=True,
        source_audits=dict(response=sha(response/'audit-v1.json'),precision=sha(precision/'verification-v1.json')))
    run.mkdir(parents=True,exist_ok=False); save(run/'plan.json',config); worker=here/'flux_incremental_native.py'
    extras=[(here/n,n) for n in ('flux_binding_minimal.py','flux_family_custody.py','flux_reentry_reference.py')]
    extras.append((Path(parameter_artifacts.__file__),'retrieve_weights.py'))
    pre=run/'preflight.v1.json'; save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[worker.name,*[n for _,n in extras],pre.name],path_token_allowlist=['binding-minimal-result','source',*config['input_roots'].values(),config['model_path'],config['output_scratch']]))
    extras.append((pre,pre.name)); request=dict(schema='mdb-incremental-native-request-v1',config=config,model=config['model'],task_family='forward',backend='diffusers',device='cuda',dtype='bfloat16',batch_size=1,height=512,width=512,
        needs=dict(cuda=True,host='beast'),execution_environment=prior['execution_environment'],source_hashes={n:sha(p) for p,n in [(worker,worker.name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1,logical_operations=4,logical_branches=4,logical_checkpoint_replays=4,max_local_work_items=4,
            checkpoint_policy='Two incremental J1 blocks with actual six-module image-prefix cache bypass and two full current-profile replays. Four continue receipts,0Acts/suffix/images/enc/noise. Compare complete typed state to archived targets; known-target qualification.'),
        resource_exception=dict(ram_mb=8500,vram_mb=9500,disk_gb=2,cpu_threads=4,est_wall_s=180,calibration_job='job-83c0610d9c67',measured_peak_rss_mb=7105.3,measured_peak_vram_mb=8282,
            reason='Executed two-receiver incremental/full qualification measured7105.3MBRSS/8282MBVRAM. Multiple live typed-cut closures and source-prefix arrays increased RAM beyond the earlier response-study estimate. Future8.5GBRAM/9.5GBVRAM retain1.39GB/1.22GB headroom before admission ceilings. Executed6.5GB request remains immutable; it completed below its7.15GB ceiling. Fixed B1/512/two-receiver geometry only.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(); save(run/'request.json',request)
    stage_worker(run/'payload',worker=worker,config=config,request=request,additional_files=extras); return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py'))); r=json.loads((run/'request.json').read_text()); h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()): raise ValueError('already submitted or stage changed')
    resources={k:v for k,v in r['resource_exception'].items() if k in {'ram_mb','vram_mb','disk_gb','cpu_threads','est_wall_s'}}
    job=guarded_mrun_launch(worker_command('flux_incremental_native.py'),request=r,model=r['model'],task='forward',experiment='mdb-flux-incremental-native',payload=run/'payload',prefer_host='beast',gpu=True,needs=r['needs'],
        timeout_s=600,retry_on_kill=False,preflight=True,priority=70,**resources,
        note='Exact incremental execution qualification:2known3/8 receivers;2image-prefix-cache J1 calls plus2full replay controls. Six image GEMM bodies actually skipped; text/fullSDPA/MLP native. Compare complete state/fingerprint; input+kwargs+layout equality checked.1load,4receipts,0Acts/suffix/VAE/enc/noise. Fixed-order timings diagnostic,600souter,no retry.')
    h['save'](run/'submission.json',dict(job_id=str(job))); return dict(job_id=str(job),run=str(run))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('response',type=Path,nargs='?'); p.add_argument('precision',type=Path,nargs='?'); p.add_argument('--submit',action='store_true'); a=p.parse_args()
    print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.response.resolve(),a.precision.resolve()),sort_keys=True))
