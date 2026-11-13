"""Bounded-memory near-data audit and compact MDB computation catalog."""
import argparse
import ast
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path


def worker():
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='2'
    h=runpy.run_path('flux_source_receiver.py');read,save,sha=h['read'],h['save'],h['sha']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('audit isolation/source binding differs')
    root,source=Path(c['native_root']),Path(c['source_root'])
    if sha(root/'files.json')!=c['native_manifest_sha256'] or sha(source/'files.json')!=c['source_manifest_sha256']:raise ValueError('evidence inventory changed')
    def measurement(path):
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='measure')
        return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    if measurement(Path('analyze_flux_reader_arithmetic.py'))!=measurement(Path('frozen-reader-auditor.py')):raise ValueError('frozen numerical measurement changed')
    started=time.monotonic();out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False)
    result=runpy.run_path('analyze_flux_reader_arithmetic.py')['audit'](root,source,out/'audit.json',lazy_arrays=True)
    published=runpy.run_path('publish_flux_reader_arithmetic.py')['publish'](root,source,out/'audit.json',out/'catalog')
    if time.monotonic()-started>c['max_wall_seconds']:raise RuntimeError('finite saved-only audit budget exceeded')
    for name in ('native-contract.json','cases.json','parents.json','summary.json'):
        shutil.copyfile(root/name,out/('source-'+name))
    for name in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/name,out/name)
    save(out/'external-evidence.json',dict(host='beast',roots={'native':str(root),'reader':str(source)},
        manifests={'native':read(root/'files.json'),'reader':read(source/'files.json')},
        scope='Full tensors remain on Beast. Compact reports are local; retrieve referenced files for a full raw-data re-audit.'))
    shutil.copyfile('retrieve_reader_evidence.py',out/'retrieve_evidence.py')
    save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(schema='mdb-reader-arithmetic-compact-audit-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),mechanics_status='verified',instrument_status=result['instrument_status'],terminal_status='unassessed',
        original_computation_job=read(root/'summary.json')['original_computation_job'],publication_job=read(root/'summary.json')['job_id'],
        measurements_unchanged_ast_sha256=measurement(Path('analyze_flux_reader_arithmetic.py')),records=published['occurrences'],
        new_execution_counts=dict(full_models=0,native_primitives=0,mathematical_affines=0,images=0),seconds=time.monotonic()-started))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),
        files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,publication,original):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));read,save,sha=h['read'],h['save'],h['sha']
    job=read(publication/'job.json');prior=read(publication/'request.json')
    if job['state']!='succeeded':raise ValueError('completed metadata publication required')
    root=Path(prior['config']['output_scratch'])/job['job_id']/'binding-minimal-result'
    code='import hashlib,json\nfrom pathlib import Path\np=Path('+repr(str(root))+')\nprint(json.dumps(dict(sha=hashlib.sha256((p/"files.json").read_bytes()).hexdigest(),summary=json.loads((p/"summary.json").read_text()))))\n'
    remote=json.loads(subprocess.run(['ssh','beast','python3','-I','-'],input=code,text=True,capture_output=True,check=True).stdout)
    cfg=dict(native_root=str(root),source_root=prior['config']['input_root'],native_manifest_sha256=remote['sha'],
        source_manifest_sha256=prior['config']['source_files_sha256'],output_scratch='/mnt/big/mdb/research',directory_custody=True,max_wall_seconds=180,
        scope='Read-only audit of completed inherited operator arrays, using unchanged frozen numerical measurements and a lazy verified loader. Publish16known-target equation records and compact metadata; full tensors stay near compute. No new native or reference forward calculations.')
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',cfg)
    names=['analyze_flux_reader_arithmetic.py','publish_flux_reader_arithmetic.py','flux_source_receiver.py','flux_reentry_reference.py','flux_family_custody.py','retrieve_reader_evidence.py']
    extras=[(here/n,n) for n in names]+[(original/'payload/analyze_flux_reader_arithmetic.py','frozen-reader-auditor.py')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[str(root),cfg['source_root'],cfg['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req=dict(schema='mdb-reader-arithmetic-near-data-audit-v1',config=cfg,execution_environment=prior['execution_environment'],needs=dict(cpu=True,host='beast'),
        task_family='analysis',backend='numpy',device='cpu',dtype='float64',source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=17,logical_branches=1,logical_checkpoint_replays=0,
            max_local_work_items=32,extra=dict(task_family='analysis'),checkpoint_policy='One saved-only bounded-memory audit plus16catalog comparisons. Lazy archive references; full geometry preserved.0models/native/referenceforwards/images;no retries.'),
        resource_exception=dict(ram_mb=2048,vram_mb=0,cpu_threads=2,disk_gb=1,est_wall_s=90,
            reason='Lazy loader retains archive descriptors only. Largest9.44M-element paired field: fourF64 corners plus fourcontrasts, with one prior contrast group, under1.1GB transient;2GB includes interpreter/reduction buffers. No2.1GB corpus hydration, Torch import or model.'))
    req['request_sha256']=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',req)
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](req['execution_environment']);stage_worker(run/'payload',worker=Path(__file__),config=cfg,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'])


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));r=h['read'](run/'request.json')
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('changed/already submitted audit')
    cmd=worker_command(Path(__file__).name);cmd.insert(1,'CUDA_VISIBLE_DEVICES=');cmd.insert(cmd.index('python')+1,'-B');os.environ.setdefault('MRUN_URL','http://192.168.68.68:9025')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-reader-arithmetic-near-data-audit',payload=run/'payload',gpu=False,
        needs=r['needs'],prefer_host='beast',timeout_s=300,retry_on_kill=False,preflight=True,
        **{k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')},
        note='Near-data saved-only operator audit: keep2.1GB raw tensors remote; lazy loader, frozen measurement AST,16MDBcatalog records.0models/native/referenceforwards/images;CPU2/RAM2GB,90sestimate/180sworker/300souter,no retries. Compact custody report plus standalone exact evidence retriever.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--publication',type=Path);p.add_argument('--original',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.publication.resolve(),a.original.resolve())))
