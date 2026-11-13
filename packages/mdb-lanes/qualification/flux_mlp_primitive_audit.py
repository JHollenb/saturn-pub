"""Near-data independent audit and compact searchable MLP diagnostic projection."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import time
from pathlib import Path


def publish(collection,audit_path,output):
    from mdb_lanes.computation_atlas import ComputationAtlas
    from mdb_lanes.equation_comparison import EquationCatalog,compare_equations
    from mdb_lanes.tensor_archive import load_tensor_archive
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    audit=read(audit_path);data=collection/'extracted/binding-minimal-result';request=read(collection/'request.json');c=request['config'];rows=read(data/'rows.json')
    if audit['mechanics_status']!='verified':raise ValueError('verified custody required')
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/'catalog');atlas=ComputationAtlas(output/'atlas');records=[];contexts={};limits={}
    sources={**request['source_hashes'],**{'wheel:'+k:v for k,v in read(collection/'payload/payload-seal.json')['wheels'].items()}};refs=[dict(path=str(p),sha256=sha(p)) for p in (audit_path,collection/'request.json',data/'parameter-artifacts.json',data/'native-contract.json')]
    laws=dict(paired='anchored-affine.first',silu='bf16-silu',activation='bf16-swiglu',product='bf16-product',mlp='anchored-affine.output')
    for role,row in rows.items():
        for port,law in laws.items():
            target='activation' if port=='product' else port
            binding=dict(model_identity=c['model_identity'],source_identity=sources,input_identity=c['native_entry'],execution=dict(backend='numpy',device='cpu',dtype='float64-with-BF16-boundaries',native_target=audit['native_contract']),frontier=dict(kind='post-hoc-native-operands',scope='Known-input diagnostic; actual target operands supplied separately to each primitive. Not a prospective forecast.'),horizon=dict(equation=role+'.'+port,consumer='local native primitive return'))
            context=dict(**binding,clock=dict(step=2,joint=1),case=c['case'],role=role,port=port,anchor=c['anchor'])
            row_record=catalog.record(law_id='flux.mlp.'+law+'.v1',route_id='flux.step2.joint1.'+role+'.mlp.v1',alias='cmp.flux.mlp-primitive.'+role+'.'+port,binding=binding,report=compare_equations({'return':load_tensor_archive(data,row['math']['local'][port])},{'return':load_tensor_archive(data,row['native'][target])},max_elements=20_000_000),conditions=[dict(field=k,op='eq',value=v) for k,v in context.items()],tested_context=context,readers=[role,'MLP'],evidence=dict(artifacts=refs,measurements=audit['reports'][role]['local'][port],source_native_states=1,additional_native_targets=0,selection='Supplied generic primitive IDs; exact applicability and native operand assistance recorded. No family induction.'))
            fp=row_record['fingerprint'];records.append(row_record);contexts[fp]=context;limits[fp]=[dict(claim_id='diagnostic-only',status='limitation-retained',scope=audit['scope'],measurements=audit['reports'][role]['local'][port],provenance=refs)]
    atlas.ingest(output/'catalog',[row['fingerprint'] for row in records],provenance=dict(run_id=audit['custody']['job_id'],artifacts=refs),counterexamples=limits)
    for context in contexts.values():
        if len(atlas.find(applicable_to=context))!=1 or atlas.find(applicable_to={**context,'clock':dict(step=2,joint=2)}) or atlas.find(applicable_to={**context,'input_identity':'unknown'}):raise ValueError('exact applicability/abstention differs')
    result=dict(schema='mdb-mlp-primitives-publication-v1',supplied_generic_laws=len(set(laws.values())),equation_occurrences=len(records),native_mlp_calls_in_source=2,new_native_calls_in_publication=0,exact_queries=10,changed_context_abstentions=20,tensor_bytes_copied=0,scope='Same-input primitive diagnostic records, not prospective predictions, induced families, or new image edits.')
    with (output/'provenance.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result


def worker():
    payload=Path.cwd();read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest();request=read(Path(os.environ['SATURN_DEBUG_REQUEST']));c=request['config']
    if read(Path(os.environ['SATURN_RUN_CONFIG']))!=c or any(sha(payload/n)!=v for n,v in request['source_hashes'].items()):raise ValueError('audit payload differs')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    collection=Path(c['remote_collection']);remote=read(collection/'remote-custody.json')
    if sha(collection/'remote-custody.json')!=c['remote_record_sha256'] or remote['verification']['job_id']!=c['native_job_id'] or remote['verification']['request_sha256']!=c['native_request_sha256']:raise ValueError('remote primitive custody differs')
    scratch=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID'];out=scratch/'binding-minimal-result';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();save=lambda p,v:p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');save(out/'plan.json',c);save(out/'request.json',request)
    audit=runpy.run_path(str(payload/'analyze_flux_mlp_primitives.py'))['analyze'](collection,out/'audit-v1.json')
    if audit['custody']!=remote['verification'] or audit['custody']['archive_sha256']!=c['native_archive_sha256']:raise ValueError('full primitive custody differs')
    publication=publish(collection,out/'audit-v1.json',out/'index-v1');data=collection/'extracted/binding-minimal-result';copied={}
    def copy(p,name):
        q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);digest=sha(q)
        if digest!=sha(p):raise ValueError('projection changed bytes')
        copied[name]=dict(sha256=digest,bytes=q.stat().st_size,remote_path=str(p))
    copy(collection/'remote-custody.json','remote-custody.json')
    for name in ('request.json','plan.json','summary.json','rows.json','row-image.json','row-text.json','source-binding.json','native-contract.json','timeline.jsonl','parameter-artifacts.json','retrieve_weights.py'):copy(data/name,'native-proof/'+name)
    save(out/'projection.json',dict(schema='mdb-compact-scientific-projection-v1',remote_collection=str(collection),remote_record_sha256=c['remote_record_sha256'],native_job_id=c['native_job_id'],native_request_sha256=c['native_request_sha256'],archive_sha256=c['native_archive_sha256'],copied=copied,scope='Partial projection: independent reports, metadata and searchable diagnostic records only. Full native intermediate tensors and original archive/log remain on Beast; native-proof array references are remote. Authenticated member retrieval uses the source native pointer.'))
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>32*1024**2:raise ValueError('compact audit projection exceeded bound')
    if any(n in sys.modules for n in ('torch','diffusers','transformers')):raise ValueError('CPU audit imported model framework')
    summary=dict(schema='mdb-mlp-primitives-remote-audit-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=request['request_sha256'],output_root=str(scratch),scientific_native_job_id=c['native_job_id'],scientific_custody=audit['custody'],audit_sha256=sha(out/'audit-v1.json'),projection_sha256=sha(out/'projection.json'),seconds=time.monotonic()-started,mechanics_status=audit['mechanics_status'],instrument_status=audit['instrument_status'],terminal_status='unassessed',counts=dict(model_loads=0,native_calls=0,images=0),publication=publication)
    save(out/'summary.json',summary);save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=summary['job_id'],request_sha256=request['request_sha256'],output_root=str(out),files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,pointer,native_run):
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.job_support import one_lease_scheduling_contract
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_binding_minimal.py'));sha,save=h['sha'],h['save'];read=lambda p:json.loads(p.read_text());record=runpy.run_path(str(here/'collect_flux_remote_directory.py'))['checked_pointer'](pointer);prior=read(native_run/'request.json')
    if record['verification']['request_sha256']!=prior['request_sha256'] or record['verification']['job_id']!=read(native_run/'submission.json')['job_id']:raise ValueError('primitive audit source mismatch')
    h['bind_environment'](prior['execution_environment']);run.mkdir(parents=True,exist_ok=False)
    c=dict(remote_collection=record['collection'],remote_record_sha256=read(pointer)['record_sha256'],native_job_id=record['verification']['job_id'],native_request_sha256=prior['request_sha256'],native_archive_sha256=record['verification']['archive_sha256'],output_scratch='/mnt/big/mdb/research',directory_custody=True,scope='Independent CPU audit of two native MLP calls. Recalculate target/anchor equations and8owned writes; join previous C, retain local/propagated/response/signed-error comparisons. No native/framework execution; raw arrays remain remote.')
    save(run/'plan.json',c);names=('analyze_flux_mlp_primitives.py','flux_family_custody.py','flux_source_receiver.py','flux_attention_anchor.py');extras=[(here/name,name) for name in names]+[(run/'plan.json','plan.json'),(pointer,'source-remote-pointer.json')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],path_token_allowlist=[c['remote_collection'],prior['config']['source_collection'],str(Path(prior['config']['source_audit']).parent),prior['config']['previous_mlp_root'],prior['config']['model_path'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name));request=dict(schema='mdb-mlp-primitives-remote-audit-request-v1',config=c,needs=dict(cpu=True,host='beast'),task_family='analysis',backend='numpy',device='cpu',dtype='float64-audit-of-BF16',execution_environment=prior['execution_environment'],source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=12,logical_branches=1,logical_checkpoint_replays=0,max_local_work_items=12,extra=dict(task_family='analysis'),checkpoint_policy='Full remote primitive custody and independently calculated target/anchor fields,8owned MLP writes,previous conditional C join,10catalog records.0native/frameworkcalls. Preserve signed powers/countertrends; numerical errors are measurements. Partial<32MBprojection only.'),resource_exception=dict(ram_mb=6144,cpu_threads=4,disk_gb=1,est_wall_s=120,reason='Measured native-plus-NumPy current primitive lane job-f9eebf9b9efb peaked3058MBRSS. Independent NumPy audit additionally holds paired response float64 arrays up to18.9Mscalars and equation catalog comparisons. Sequential per-role weights/metrics,6GBCPU-only finite bound; no model frameworks/forwards. Expectedunder3min,600outer,noretries.'))
    request['request_sha256']=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',request);stage_worker(run/'payload',worker=Path(__file__),config=c,request=request,additional_files=extras);return dict(run=str(run),request_sha256=request['request_sha256'])


def submit(run):
    from mdb_lanes.payload import worker_command
    from mdb_lanes.job_support import guarded_mrun_launch
    h=runpy.run_path(str(Path(__file__).with_name('flux_binding_minimal.py')));r=json.loads((run/'request.json').read_text());h['bind_environment'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('already submitted or source changed')
    resource={k:r['resource_exception'][k] for k in ('ram_mb','cpu_threads','disk_gb','est_wall_s')};cmd=worker_command(Path(__file__).name);cmd.insert(cmd.index('python')+1,'-B')
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-flux-mlp-primitives-independent-audit',payload=run/'payload',gpu=False,needs=r['needs'],prefer_host='beast',timeout_s=600,retry_on_kill=False,preflight=True,**resource,note='CPU-only independent audit on full original-source MLP evidence.12mathematical field/owned calculations,2priorC joins plusanchors,10exact-applicability diagnostic atlasrecords.0native/frameworkcalls/images; signed local/propagated/response errors.6GB/4threads,under3min/600outer,noretries;fullrawstaysBeast,compactprojectiononly.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('pointer',type=Path,nargs='?');p.add_argument('native_run',type=Path,nargs='?');p.add_argument('--submit',action='store_true');a=p.parse_args();print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.pointer.resolve(),a.native_run.resolve()),sort_keys=True))
