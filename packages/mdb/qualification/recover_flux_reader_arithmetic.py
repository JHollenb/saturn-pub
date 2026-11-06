"""Publish completed operator arrays after a metadata-only failure; zero replay."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path


def worker():
    h=runpy.run_path('flux_source_receiver.py');read,save,sha=h['read'],h['save'],h['sha']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config'];recovery=c['publication_recovery']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('metadata recovery must hide CUDA')
    if any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('recovery payload changed')
    started=time.monotonic();source=Path(recovery['original_output_root']);inventory=read('partial-inventory.json')
    if any(sha(source/n)!=v for n,v in inventory.items()):raise ValueError('completed original data changed')
    old=read(source/'request.json')
    if old['request_sha256']!=recovery['original_request_sha256']:raise ValueError('wrong original request')
    cases=read(source/'cases.json');timeline=[json.loads(x) for x in (source/'timeline.jsonl').read_text().splitlines()]
    if len(cases)!=8 or len(timeline)!=8 or timeline[-1]['counts']!=c['planned_counts']:raise ValueError('original scientific work incomplete')
    for key,row in cases.items():
        if read(source/'rows'/(key+'.json'))!=row:raise ValueError('case index differs')
        for group in ('native','math'):
            for fields in row[group].values():
                for record in fields.values():h['load_array'](source,record)
        h['load_array'](source,row['source_final'])
    for fields in read(source/'parents.json').values():
        for record in fields.values():h['load_array'](source,record)
    out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    for name in inventory:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name in ('plan.json','request.json'):target=out/('original-'+name)
        os.link(source/name,target)
    save(out/'plan.json',c);save(out/'request.json',req);shutil.copyfile('partial-inventory.json',out/'original-inventory.json')
    env=recovery['original_execution_environment']
    # These getter values were not serialized by the failed original worker.
    # Preserve unknowns rather than fabricating a fresh original measurement.
    save(out/'native-contract.json',dict(device=env['cuda_driver'].split(',')[0],torch_version=env['packages']['torch'],
        cuda_version=None,input_shape=[1,c['text_rows'],c['width']],dtype='bfloat16',default_bf16_reduced_reduction=None,
        fp16_reduced_reduction=None,bf16_accumulation=None,tf32=False,deterministic=True,inference_mode=True,
        cublas_workspace=':4096:8',sdpa='not executed',execution_kind='Torch source-transcribed primitive graph',
        original_module_source_sha256=c['original_module_source_sha256'],
        module_source_sha256={'primitive_graph_file':old['source_hashes']['flux_reader_arithmetic.py']},
        profile_provenance='Declared flags and original recorded package/device metadata. Runtime reduction getters were lost during profile serialization; values remain unknown. Default-versus-opposite lane identities and original arrays are preserved.'))
    summary=dict(schema='mdb-reader-arithmetic-summary-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),counts=timeline[-1]['counts'],counts_authority='Inherited completed work from failed original computation job; no new tensor calculations.',
        original_computation_job=recovery['original_job'],original_computation_state='failed after all scientific rows, during profile serialization',
        original_last_case_seconds=timeline[-1]['seconds'],seconds=time.monotonic()-started,
        new_execution_counts=dict(full_model_loads=0,native_primitives=0,mathematical_affines=0,images=0),
        parameter_reads=None,parameter_cache_peak_bytes=None,mechanics_status='recovered-complete-data',terminal_status='unassessed',scope=c['scope'])
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=req['request_sha256'],output_root=str(out),
        files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,original):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));read,save,sha=h['read'],h['save'],h['sha']
    old=read(original/'request.json');job=read(original/'job.json')
    if job['state']!='failed':raise ValueError('expected original metadata failure')
    root=Path(old['config']['output_scratch'])/job['job_id']/'binding-minimal-result'
    code='import hashlib,json\nfrom pathlib import Path\np=Path('+repr(str(root))+')\nprint(json.dumps({f.relative_to(p).as_posix():hashlib.file_digest(f.open("rb"),"sha256").hexdigest() for f in sorted(p.rglob("*")) if f.is_file()}))\n'
    result=subprocess.run(['ssh','beast','python3','-I','-'],input=code,text=True,capture_output=True,check=True)
    inventory=json.loads(result.stdout)
    environment=[json.loads(x.removeprefix('MDB_EXECUTION_ENVIRONMENT=')) for x in (original/'worker.log').read_text().splitlines() if x.startswith('MDB_EXECUTION_ENVIRONMENT=')]
    if len(environment)!=1:raise ValueError('missing original environment record')
    cfg=dict(old['config'],scope=old['config']['scope']+' Metadata-only publication recovery: original calculations inherited unchanged; reduction getter metadata unavailable.',
        publication_recovery=dict(original_job=job['job_id'],original_request_sha256=old['request_sha256'],original_output_root=str(root),
            original_job_record_sha256=sha(original/'job.json'),original_execution_environment=environment[0]))
    run.mkdir(parents=True,exist_ok=False);save(run/'partial-inventory.json',inventory);save(run/'plan.json',cfg)
    names=['flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py','analyze_flux_reader_arithmetic.py']
    extras=[(original/'payload'/n,n) for n in names]+[(run/'partial-inventory.json','partial-inventory.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[str(root),cfg['input_root'],cfg['model_path'],cfg['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    request=dict(schema='mdb-reader-arithmetic-publication-recovery-v1',config=cfg,execution_environment=old['execution_environment'],
        needs=dict(cpu=True,host='beast'),task_family='analysis',backend='stdlib-and-numpy',device='cpu',dtype='preserved-BF16-bits',
        source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},
        resource_exception=dict(ram_mb=1024,vram_mb=0,cpu_threads=1,disk_gb=1,est_wall_s=30,
            calibration_job='job-c63fc7cf3ca3',measured_peak_rss_mb=756.7,
            reason='Earlier completed metadata-only BF16-array publication measured756.7MBRSS/0VRAM and7.13s runner. This worker keeps one decoded archive at a time; largest paired archive adds under80MB transient decoding versus prior fields.1GB plus guard ceiling; no Torch import, model, or retained operator tensors.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=1,logical_branches=1,
            logical_checkpoint_replays=0,max_local_work_items=1,extra=dict(task_family='analysis'),
            checkpoint_policy='Metadata-only publication of completed failed-job arrays; verify hashes/shapes, hardlink unchanged bytes, append honest metadata.0tensor calculations/nativeprimitives/models/images;no retries.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request)
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](request['execution_environment'])
    stage_worker(run/'payload',worker=Path(__file__),config=cfg,request=request,additional_files=extras)
    return dict(request_sha256=request['request_sha256'],original_files=len(inventory))


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));r=h['read'](run/'request.json')
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('changed/already submitted recovery')
    cmd=worker_command(Path(__file__).name);cmd.insert(1,'CUDA_VISIBLE_DEVICES=');i=cmd.index('python');cmd.insert(i+1,'-B')
    os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-reader-arithmetic-publication',payload=run/'payload',
        gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=180,retry_on_kill=False,preflight=True,
        **{k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')},
        note='Metadata-only recovery of all8completed reader-arithmetic cases from failed job-486f616ef636. Verify every original file/reference; hardlink unchanged arrays; publish missing profile with unknown getters explicit.0native/math/model/image calls;CUDA hidden;no retries.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--original',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.original.resolve())))
