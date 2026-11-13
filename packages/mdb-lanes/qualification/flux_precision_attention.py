"""Model-free, near-data attention diagnostic on already observed operands."""
import argparse
import hashlib
import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload=Path.cwd();request=json.loads(Path(os.environ['SATURN_DEBUG_REQUEST']).read_text());config=request['config']
    for name,expected in request['source_hashes'].items():
        if hashlib.sha256((payload/name).read_bytes()).hexdigest()!=expected:raise ValueError('source differs')
    for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    from mdb_lanes.attention_partition import dense_attention
    from mdb_lanes.tensor_archive import load_tensor_archive
    root=Path(config['input_root']);read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,expected in config['inputs'].items():
        if sha(root/name)!=expected:raise ValueError('observed operand identity differs')
    native=read(root/'native.json');r=runpy.run_path(str(payload/'flux_reentry_reference.py'));metric=runpy.run_path(str(payload/'analyze_flux_precision.py'))['error']
    scratch=Path(config['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');save(out/'plan.json',config);save(out/'request.json',request)
    started=time.monotonic();outputs={};actual={};reports={}
    for name in('half','half.image1'):
        stages=native[name]['stages']
        def operand(label):
            return np.concatenate([load_tensor_archive(root,stages[role+'.'+label])for role in('text','image')],axis=1).transpose(0,2,1,3)
        q,k,v=[operand(label)for label in('q_positioned','k_positioned','v_projection')]
        output=r['bf16'](dense_attention(q,k,v).transpose(0,2,1,3).reshape(1,-1,q.shape[1]*q.shape[-1]));del q,k,v
        actual[name]=load_tensor_archive(root,stages['attention.raw']);outputs[name]=output
        path=out/(name+'.dense.npy')
        with path.open('xb')as f:np.save(f,output,allow_pickle=False)
        reports[name]=dict(error=metric(output,actual[name]),path=path.name,sha256=sha(path),shape=list(output.shape),dtype=str(output.dtype))
    contrast=metric(outputs['half.image1']-outputs['half'],actual['half.image1']-actual['half'])
    result=dict(schema='mdb-native-operand-attention-diagnostic-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],
        reports=reports,receiver_contrast=contrast,seconds=time.monotonic()-started,model_loads=0,native_forwards=0,native_transitions=0,
        scope='Post-hoc mathematical attention from observed native positioned Q/K and raw V. Isolates dense-reference/native-attention arithmetic, not an independent output forecast or semantic claim.',
        source_summary_sha256=config['inputs']['summary.json'],mechanics_status='pending-independent-audit',terminal_status='unassessed')
    save(out/'summary.json',result);save(out/'files.json',{p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=result['job_id'],request_sha256=result['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,source):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.edit_search import file_digest as sha
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));read=lambda p:json.loads(p.read_text())
    contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=2,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=2,extra=dict(task_family='analysis'),checkpoint_policy='One read-only analysis program, two observed QKV cases;2dense mathematical attention diagnostics;0models/nativeforwards/rewinds.')
    runpy.run_path(str(here/'flux_family_custody.py'))['verify'](source)
    prior=read(source/'request.json');h['bind_environment'](prior['execution_environment']);data=source/'extracted/binding-minimal-result';summary=read(data/'summary.json');native=read(data/'native.json');inputs={n:sha(data/n)for n in('native.json','summary.json','files.json')}
    for item in native.values():
        for key,record in item['stages'].items():
            if key=='attention.raw'or key.endswith(('.q_positioned','.k_positioned','.v_projection')):inputs[record['path']]=record['sha256']
    config=dict(input_root=summary['output_root']+'/binding-minimal-result',inputs=inputs,output_scratch='/mnt/big/mdb/research',directory_custody=True,source_job=summary['job_id'])
    run.mkdir(parents=True,exist_ok=False);h['save'](run/'plan.json',config);extras=[(here/n,n)for n in('flux_family_custody.py','flux_reentry_reference.py','analyze_flux_precision.py')]
    pre=run/'preflight.v1.json';h['save'](pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[config['input_root'],config['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name));request=dict(schema='mdb-precision-attention-request-v1',config=config,needs=dict(cpu=True,host='beast'),execution_environment=prior['execution_environment'],task_family='analysis',backend='numpy',device='cpu',dtype='float64',
        source_hashes={n:sha(p)for p,n in[(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=contract,
        resource_exception=dict(ram_mb=1024,vram_mb=0,cpu_threads=4,disk_gb=1,est_wall_s=60,calibration_job='job-005902358e55',measured_peak_rss_mb=543.8,reason='Same two saved QKV sets, CPU NumPy query chunks64; complete audit measured543.8MB peakRSS,14.33s runner.1GB retains480MB headroom plus strict ceiling. No models or GPU allocation.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();h['save'](run/'request.json',request)
    stage_worker(run/'payload',worker=Path(__file__),config=config,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if(run/'submission.json').exists()or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or source changed')
    resources={k:v for k,v in r.get('resource_exception',{}).items()if k in{'ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s'}}
    job=guarded_mrun_launch(worker_command(Path(__file__).name),request=r,model=None,task='analysis',experiment='mdb-precision-attention',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=240,retry_on_kill=False,preflight=True,**resources,
        note='Read-only CPU NumPy attention from two already observed native QKV sets;0model loads/nativeforwards/targets. Query chunks64,4BLASthreads,2outputs;expectedunder60s,240souter,no retries;intent-first resource planning. Host-local immutable source.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('source',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve())if a.submit else stage(a.run.resolve(),a.source.resolve()),sort_keys=True))
