"""Near-data independent CPU forecast replay and compact fresh-image publication."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def publish(collection,audit_path,out):
    from mdb.equation_comparison import EquationCatalog,compare_equations
    from mdb.computation_atlas import ComputationAtlas
    from mdb.tensor_archive import load_tensor_archive
    data=collection/'extracted/binding-minimal-result'; request=read(collection/'request.json'); c=request['config']; audit=read(audit_path)
    forecasts=read(data/'forecasts.json'); native=read(data/'native.json'); fresh=read(data/'fresh-parent.json')
    out.mkdir(parents=True,exist_ok=False); catalog=EquationCatalog(out/'catalog'); atlas=ComputationAtlas(out/'atlas'); records=[]; contexts={}; limits={}
    for name in ('row4heavy','row11heavy'):
        actual={role:load_tensor_archive(data,native[name]['stages'][role+'.exit']) for role in ('image','text')}
        for mode in ('old','new'):
            predicted={role:load_tensor_archive(data,forecasts[name][mode][role]['exit']) for role in ('image','text')}
            binding=dict(model_identity=read(data/'receipts.json')[0]['body']['model_identity'],
                source_identity=request['source_hashes'],input_identity=native[name]['entry'],
                execution=dict(backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',native_consumer='CUDA BF16 original FLUX suffix/VAE'),
                frontier=dict(kind='fresh-receiver-shared-calibration',receiver=fresh['parent'],calibration=native['half']['entry'],scope=c['scope']),horizon=dict(equation='joint1 complete image/text exits',consumer='own-state remaining denoiser and image'))
            context=dict(**binding,clock=dict(step=2,joint=1),seed=c['seed'],case=name,method=mode)
            refs=[dict(path=str(p),sha256=sha(p)) for p in (audit_path,data/'request.json',data/'fresh-parent.json',data/'calibration.json')]
            row=catalog.record(law_id='flux.joint1.mlp.integration.'+('anchored-first' if mode=='old' else 'direct-first')+'.v1',route_id='flux.step2.joint1.time-gated-mlp.v1',alias='cmp.flux.fresh-mlp.'+name+'.'+mode,
                binding=binding,report=compare_equations(predicted,actual,max_elements=5_000_000),conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,
                readers=['image','text','time-gate','residual-addition','native-image'],evidence=dict(artifacts=refs,measurements=dict(numerical=audit['numerical'][name][mode],image=audit['outcomes'][name][mode]),scope=c['scope']))
            fp=row['fingerprint'];records.append(row);contexts[fp]=context;limits[fp]=[dict(claim_id='fresh-receiver-scope',status='limitations-and-countertrends-retained',scope=c['scope'],measurements=audit['outcomes'][name][mode],provenance=refs)]
    atlas.ingest(out/'catalog',[r['fingerprint'] for r in records],provenance=dict(run_id=audit['custody']['job_id'],artifacts=[dict(path=str(audit_path),sha256=sha(audit_path))]),counterexamples=limits)
    for ctx in contexts.values():
        if len(atlas.find(applicable_to=ctx))!=1 or atlas.find(applicable_to={**ctx,'seed':81120}) or atlas.find(applicable_to={**ctx,'clock':dict(step=2,joint=2)}):raise ValueError('atlas applicability differs')
    result=dict(occurrences=4,exact_queries=4,changed_context_abstentions=8,tensor_bytes_copied=0,scope='Two supplied frozen programs/two measured target states on one fresh calibrated receiver. Applicability supplied, not induced.')
    (out/'provenance.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result


def worker():
    payload=Path.cwd(); request=read(os.environ['SATURN_DEBUG_REQUEST']); c=request['config']
    if read(os.environ['SATURN_RUN_CONFIG'])!=c or any(sha(payload/name)!=digest for name,digest in request['source_hashes'].items()):raise ValueError('audit source changed')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    collection=Path(c['remote_collection']); remote=read(collection/'remote-custody.json')
    if sha(collection/'remote-custody.json')!=c['remote_record_sha256'] or remote['verification']['request_sha256']!=c['native_request_sha256']:raise ValueError('native collection identity differs')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']; out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
    save(out/'request.json',request);save(out/'plan.json',c)
    audit=runpy.run_path(str(payload/'analyze_flux_fresh_mlp_future.py'))['analyze'](collection,out/'audit-v1.json')
    if audit['custody']!=remote['verification']:raise ValueError('complete independent custody differs')
    publication=publish(collection,out/'audit-v1.json',out/'index-v1');data=collection/'extracted/binding-minimal-result';copied={}
    def copy(p,name):
        q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        if sha(p)!=sha(q):raise ValueError('projection copy differs')
        copied[name]=dict(sha256=sha(q),bytes=q.stat().st_size,remote_path=str(p))
    copy(collection/'remote-custody.json','remote-custody.json')
    for name in ('request.json','plan.json','summary.json','fresh-parent.json','calibration.json','forecasts.json','native.json','rows.json','receipts.json','image-forecasts.json','timeline.jsonl','parameter-artifacts.json','retrieve_weights.py','prediction-costs.json','source-binding.json'):
        copy(data/name,'native-proof/'+name)
    for directory in ('write-transaction','image-transaction'):
        for p in sorted((data/directory).rglob('*')):
            if p.is_file():copy(p,'native-proof/'+p.relative_to(data).as_posix())
    for p in sorted(data.glob('*.png')):copy(p,'images/'+p.name)
    save(out/'projection.json',dict(schema='mdb-compact-scientific-projection-v1',remote_collection=str(collection),remote_record_sha256=c['remote_record_sha256'],native_request_sha256=c['native_request_sha256'],copied=copied,
        scope='Partial reports/9images/metadata/index only; native-proof references address remote scientific arrays/store. Full archive/log/source/wheels/raw states verified remotely, retained for authenticated exact retrieval.'))
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>32*1024**2:raise ValueError('compact projection exceeded bound')
    if any(name in sys.modules for name in ('torch','diffusers','transformers')):raise ValueError('CPU audit imported model framework')
    summary=dict(schema='mdb-fresh-mlp-remote-audit-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),scientific_custody=audit['custody'],
        audit_sha256=sha(out/'audit-v1.json'),projection_sha256=sha(out/'projection.json'),seconds=time.monotonic()-started,mechanics_status='verified',instrument_status=audit['instrument_status'],terminal_status='unassessed',primary=audit['primary'],publication=publication,counts=dict(model_loads=0,native_calls=0,scientific_images=9,independent_forecast_fields=72))
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,pointer,native_run):
    from mdb.payload import stage_worker
    from mdb.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent; base=runpy.run_path(str(here/'flux_binding_minimal.py'));save=base['save']
    record=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer);previous=read(native_run/'request.json');base['bind_environment'](previous['execution_environment'])
    if previous['request_sha256']!=record['verification']['request_sha256'] or read(native_run/'submission.json')['job_id']!=record['verification']['job_id']:raise ValueError('audit source job differs')
    run.mkdir(parents=True,exist_ok=False);c=dict(remote_collection=record['collection'],remote_record_sha256=read(pointer)['record_sha256'],native_request_sha256=previous['request_sha256'],output_scratch='/mnt/big/mdb/research',directory_custody=True)
    save(run/'plan.json',c)
    names=('flux_binding_minimal.py','flux_family_custody.py','analyze_flux_fresh_mlp_future.py','analyze_flux_mlp_forecast.py','analyze_flux_binding_minimal.py','summarize_context_window.py','flux_source_receiver.py')
    extras=[(here/name,name) for name in names]+[(run/'plan.json','plan.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[name for _,name in extras],pre.name],path_token_allowlist=[c['remote_collection'],previous['config']['source_collection'],previous['config']['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    request=dict(schema='mdb-fresh-mlp-audit-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-audit-of-BF16',execution_environment=previous['execution_environment'],source_hashes={name:sha(p) for p,name in [(Path(__file__),Path(__file__).name),*extras]},
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=3,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=12,
            extra=dict(task_family='analysis'),checkpoint_policy='Full native custody, typed source/consumer lineage, two seals, independent3old forecasts+6replacementMLPs/72fields,9images/pairedresponse/regions,4atlasoccurrences.0native/framework calls, compactprojectiononly.'),
        resource_exception=dict(ram_mb=5120,cpu_threads=4,disk_gb=1,est_wall_s=120,calibration_job='job-97b4b84230b1',measured_peak_rss_mb=3069.9,
            reason='This exact separate72field/9image audit measured84.02sprocess/3069.9MBRSS on job-97b4b84230b1. Future5GBRAM retains headroom;4CPU,120sestimate/600outer,noretries. Executed8GBrequest remains immutable. No model load/VRAM.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras)
    return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb.payload import worker_command
    from mdb.job_support import guarded_mrun_launch
    base=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=read(run/'request.json');base['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(sha(run/'payload'/name)!=digest for name,digest in r['source_hashes'].items()):raise ValueError('changed or submitted audit')
    cmd=worker_command(Path(__file__).name);cmd.insert(1,'CUDA_VISIBLE_DEVICES=');cmd.insert(cmd.index('python')+1,'-B')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-flux-fresh-mlp-independent-audit',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,
        **{key:r['resource_exception'][key] for key in ('ram_mb','cpu_threads','disk_gb','est_wall_s')},note='CPU fullrawcustody+typedlineage/twoseals. Replay3frozen oldforecasts plus6directfirstMLPs,72fieldbitchecks;9actualPNGs, ownP0pairedRGB/target/collateral,4records.0native/framework/VRAM;5GB/4CPU from measured3069.9MB,120sestimate/600outer,noretries. Compactprojectiononly.')
    base['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('pointer',type=Path,nargs='?');p.add_argument('native_run',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.pointer.resolve(),a.native_run.resolve()),sort_keys=True))
