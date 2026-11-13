"""Repackage an already sealed forecast cohort after its JSON publication failed.

No mathematical prediction, model loading or native execution is performed.
The failed originating job and its incomplete cohort remain unchanged.
"""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
H=runpy.run_path(str(HERE/'flux_source_receiver.py'))
read,save,sha=(H[k] for k in ('read','save','sha'))


def stage(origin,run):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    old=read(origin/'request.json');job=read(origin/'job.json')
    if (job['state']!='failed' or job['job_id']!=read(origin/'submission.json')['job_id']
            or job['config']['request_sha256']!=old['request_sha256']
            or job['config']['source_hashes']!=old['source_hashes']):
        raise ValueError('completed failed originating job required')
    log=(origin/'scheduler-finished.log').read_text()
    events=[json.loads(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith('MDB_SOURCE_RECEIVER ')]
    expected={a['id'] for a in H['arms']() if a['kind']!='sham'}
    if ({r['arm'] for r in events if r['kind']=='arm_sealed'}!=expected
            or sum(r['kind']=='predicted_transition' for r in events)!=300
            or events[-1]['counts']['mathematical_transitions']!=300
            or any(r['counts']['native_model_forwards'] for r in events)
            or 'mappingproxy is not JSON serializable' not in log):
        raise ValueError('not the declared post-seal serialization failure')
    run.mkdir(parents=True,exist_ok=False)
    remote=Path(old['config']['output_scratch'])/job['job_id']/'binding-minimal-result'
    script=('import hashlib,json\nfrom pathlib import Path\nr=Path('+repr(str(remote))+')\n'
        'd={}\nfor p in sorted(r.rglob("*")):\n'
        ' if p.is_symlink():raise ValueError("symlink in source")\n'
        ' if p.is_file():d[p.relative_to(r).as_posix()]={"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest(),"bytes":p.stat().st_size}\n'
        'print(json.dumps(d,sort_keys=True))\n')
    inventory=json.loads(subprocess.check_output(['ssh','beast','python3','-I','-'],input=script.encode()))
    save(run/'origin-inventory.json',inventory)
    for src,dst in [('request.json','origin-request.json'),('job.json','origin-job.json'),('scheduler-finished.log','origin-scheduler.log')]:
        shutil.copyfile(origin/src,run/dst)
    cfg=dict(old['config'],phase='recover',origin_root=str(remote),origin_job=job['job_id'],
        origin_request_sha256=old['request_sha256'],max_wall_seconds=120)
    raw=(json.dumps(cfg,sort_keys=True)+'\n').encode()
    files=[(HERE/'flux_source_receiver.py','flux_source_receiver.py'),
        (origin/'payload/flux_family_custody.py','flux_family_custody.py'),
        *[(run/n,n) for n in ('origin-inventory.json','origin-request.json','origin-job.json','origin-scheduler.log')]]
    save(run/'plan.json',cfg)
    pre=run/'preflight.v1.json'
    save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in files],pre.name],
        path_token_allowlist=[cfg['source_root'],cfg['origin_root'],cfg['model_path'],cfg['output_scratch'],'binding-minimal-result']))
    files.append((pre,pre.name))
    request=dict(schema='mdb-source-receiver-recovery-request-v1',config=cfg,model=old['model'],backend='diffusers',
        task_family='analysis',device='cpu',dtype='bfloat16',needs={'host':'beast'},execution_environment=old['execution_environment'],
        config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*files]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=6,logical_branches=6,
            max_local_work_items=16,extra={'task_family':'analysis'},checkpoint_policy='Metadata-only recovery of six original already sealed forecasts; zero new predictions/targets.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    save(run/'request.json',request)
    stage_worker(run/'payload',worker=Path(__file__),config=cfg,request=request,config_bytes=raw,additional_files=files)
    return dict(request_sha256=request['request_sha256'],files=len(inventory))


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    request=read(run/'request.json')
    oldstage=runpy.run_path(str(HERE/'stage_flux_source_receiver.py'));oldstage['bind'](request['execution_environment'])
    command=H['helper']('submit_flux_initial_noise.py')['sealed_worker_command'](request)
    command[-1]=Path(__file__).name;command.insert(1,'CUDA_VISIBLE_DEVICES=')
    if (run/'submission.json').exists() or any(sha(run/'payload'/n)!=v for n,v in request['source_hashes'].items()):
        raise ValueError('changed or previously submitted recovery payload')
    job=guarded_mrun_launch(command,request=request,model=None,task='analysis',experiment='mdb-flux-source-receiver-recover',
        payload=run/'payload',pin='beast',gpu=False,needs=request['needs'],preflight=True,retry_on_kill=False,
        cpu_threads=1,ram_mb=2048,vram_mb=0,disk_gb=4,timeout_s=180,est_wall_s=60,priority=50,
        note='Metadata-only recovery of300completed mathematical transitions/six sealed forecasts from failed job-9877d6c5292c. Verify source inventory/seals and serialize immutable trees;0new math/model/native/VAE calls. Preserve original failure;no retries.')
    save(run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


def worker():
    import numpy as np
    import torch

    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction
    here=Path.cwd();cfg,req=[read(os.environ[k]) for k in ('SATURN_RUN_CONFIG','SATURN_DEBUG_REQUEST')]
    if cfg!=req['config'] or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(here/n)!=v for n,v in req['source_hashes'].items()):
        raise ValueError('recovery identity/device changed')
    origin_request,origin_job=read(here/'origin-request.json'),read(here/'origin-job.json')
    if (origin_request['request_sha256']!=cfg['origin_request_sha256']
            or origin_job['job_id']!=cfg['origin_job'] or origin_job['state']!='failed'
            or origin_job['config']['request_sha256']!=cfg['origin_request_sha256']):
        raise ValueError('originating scheduler/request identity changed')
    started=time.monotonic();attempts=[]
    def deny(module,args):
        attempts.append(type(module).__name__);raise ValueError('no native execution in recovery')
    guard=torch.nn.modules.module.register_module_forward_pre_hook(deny)
    source=Path(cfg['origin_root']);inventory=read(here/'origin-inventory.json')
    out=Path(cfg['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    renamed={'cohort.json':'origin-incomplete-cohort.txt','request.json':'origin-worker-request.json',
        'plan.json':'origin-worker-plan.json','timeline.jsonl':'origin-timeline.jsonl'}
    for name,record in inventory.items():
        if time.monotonic()-started>cfg['max_wall_seconds']:raise RuntimeError('finite recovery budget')
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('unsafe source path')
        src=source/name
        if src.is_symlink() or sha(src)!=record['sha256'] or src.stat().st_size!=record['bytes']:
            raise ValueError('original sealed source changed: '+name)
        target=out/renamed.get(name,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
        if sha(target)!=record['sha256']:raise ValueError('recovery copy changed')
    for n in ('origin-inventory.json','origin-request.json','origin-job.json','origin-scheduler.log'):
        shutil.copyfile(here/n,out/n)
    save(out/'request.json',req);save(out/'plan.json',cfg)
    programs={};seals={};views=[]
    for arm in H['arms']():
        if arm['kind']=='sham':continue
        key=arm['id'];root=out/'transactions'/key
        digest=sha(root/'seal.json');verify_transaction(root,expected_seal_sha256=digest)
        if read(root/'declarations.json')['source_identity']!=cfg['origin_request_sha256']:
            raise ValueError('original transaction source differs')
        programs[key]=load_forecast_tree(root,read(root/'forecasts/program.json')['predictions'])
        if len(programs[key]['records'])!=50:raise ValueError('incomplete original prediction')
        seals[key]=dict(seal_sha256=digest,path=str(root/'seal.json'),forecasts=['program'])
        for record in (programs[key]['input_text'],*[v for r in programs[key]['records'] for v in r['ports'].values()],
                *[v['archive'] for r in programs[key]['records'] for v in r['outputs'].values() if v['archive']]):
            value=H['load_array'](out,record)
            physical=np.load(out/record['path'],allow_pickle=False)
            if list(physical.shape)!=list(record['shape']):
                bits=(value.view(np.uint32)>>16).astype('<u2')
                if bits.tobytes()!=physical.tobytes():raise ValueError('legacy view changed predicted bits')
                views.append(dict(arm=key,path=record['path'],sha256=record['sha256'],
                    physical_shape=list(physical.shape),sealed_logical_shape=list(record['shape']),bit_sequence_exact=True))
    counts=dict(mathematical_transitions=300,native_model_forwards=0,encodes=0,noise_draws=0,images=0)
    save(out/'cohort.json',dict(schema='mdb-source-receiver-cohort-v1',programs=programs,seals=seals,
        sham_reuses='right.base',counts=counts,all_forecasts_before_native=True))
    save(out/'legacy-array-views.json',dict(views=views,changed_forecast_values=0,
        explanation='Original v1 content address omitted shape. Sealed logical views reinterpret the same hash-verified BF16 bit sequence; original files/records/seals remain unchanged. V2 writers bind shape.'))
    summary=dict(schema='mdb-source-receiver-forecast-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        mechanics_status='complete',terminal_status='unassessed',counts=counts,native_module_attempts=attempts,
        output_root=str(out.resolve()),seconds=time.monotonic()-started,originating_forecast_job=cfg['origin_job'],
        originating_request_sha256=cfg['origin_request_sha256'],cohort_sha256=sha(out/'cohort.json'),
        new_work=dict(mathematical_transitions=0,native_model_forwards=0,model_loads=0),
        recovery='Original300transitions and six seals retained byte-exactly; immutable mapping serialization corrected. Original failure unchanged; parameter-cache telemetry was not published.')
    save(out/'summary.json',summary)
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=req['request_sha256'],output_root=str(out.resolve()),
        files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)
    guard.remove()


if __name__=='__main__':
    if os.environ.get('MRUN_JOB_ID'):worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--origin',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.origin.resolve(),a.run.resolve())))
