"""Near-data saved-only attention audit, catalog publication and compact custody."""
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path


def audit(root, source, out):
    import numpy as np

    from mdb.equation_comparison import EquationCatalog, compare_equations
    here=Path(__file__).resolve().parent
    h=runpy.run_path(str(here/'flux_source_receiver.py'));read,save,sha,load=[h[k] for k in ('read','save','sha','load_array')]
    measure=runpy.run_path(str(here/'analyze_flux_reader_arithmetic.py'))['measure']
    c,s,r=[read(root/n) for n in ('plan.json','summary.json','request.json')]
    if r['config']!=c or s['request_sha256']!=r['request_sha256'] or sha(source/'files.json')!=c['source_files_sha256']:
        raise ValueError('source/request lineage differs')
    if any(sha(root/n)!=v for n,v in read(root/'files.json').items()):raise ValueError('native output changed')
    if any(sha(source/n)!=v['sha256'] or (source/n).stat().st_size!=v['bytes'] for n,v in c['inputs'].items()):raise ValueError('native operand/target changed')
    rows=read(root/'cases.json');profile=read(root/'native-contract.json')
    out.mkdir(parents=True,exist_ok=False)
    errors={};parity={};effects={};geometry={}
    def head(key,lane):return load(root,rows[key]['heads'][lane])
    def observed(key,field='head'):return load(source,rows[key]['source_observed'][field])
    def tail(key,lane,field='text'):return load(root,rows[key]['tails'][lane][field])
    for key,row in rows.items():
        actual=observed(key)
        errors[key]=dict(heads={lane:measure(actual,head(key,lane)) for lane in row['heads']},
            tails={lane:{f:measure(observed(key,f),tail(key,lane,f)) for f in fields} for lane,fields in row['tails'].items()},
            previous_head=measure(actual,load(source,row['source_forecast']['head'])))
        parity[key]=dict(head={lane:a['sha256']==row['source_observed']['head']['sha256'] for lane,a in row['heads'].items()},
            tails={lane:{f:a['sha256']==row['source_observed'][f]['sha256'] for f,a in fields.items()} for lane,fields in row['tails'].items()})
        geometry[key]={lane:measure(head(key,'auto-text'),head(key,lane)) for lane in ('auto-full-zero','auto-full-repeat')}
    def contrasts(get,receiver):
        b,k,v,d=[get(f'{receiver}.k{ki}v{vi}').astype(np.float64) for ki,vi in ((0,0),(1,0),(0,1),(1,1))]
        return dict(key=k-b,value=v-b,interaction=d-k-v+b,total=d-b)
    for receiver in ('left','right'):
        actual=contrasts(observed,receiver); effects[receiver]=dict(heads={},tails={})
        lanes=set.intersection(*[set(rows[f'{receiver}.k{ki}v{vi}']['heads']) for ki,vi in ((0,0),(1,0),(0,1),(1,1))])
        for lane in sorted(lanes):
            predicted=contrasts(lambda key:head(key,lane),receiver)
            effects[receiver]['heads'][lane]={name:measure(actual[name],predicted[name]) for name in actual}
        actual=contrasts(lambda key:observed(key,'text'),receiver)
        for lane in ('auto-full-zero','math-text'):
            predicted=contrasts(lambda key:tail(key,lane),receiver)
            effects[receiver]['tails'][lane]={name:measure(actual[name],predicted[name]) for name in actual}
    measured=dict(sdpa=sum(not lane.startswith('staged-') for row in rows.values() for lane in row['heads']),
        staged_matmuls=2*sum(lane.startswith('staged-') for row in rows.values() for lane in row['heads']),
        staged_softmax=sum(lane.startswith('staged-') for row in rows.values() for lane in row['heads']),
        linear=3*sum(len(row['tails']) for row in rows.values()),layernorm=sum(len(row['tails']) for row in rows.values()),swiglu=sum(len(row['tails']) for row in rows.values()),
        full_models=0,blocks=0,attention_modules=0,images=0)
    if len(rows)!=8 or s['counts']!=measured:raise ValueError('execution evidence census differs')
    result=dict(schema='mdb-reader-attention-audit-v1',job=s['job_id'],mechanics_status='verified',terminal_status='unassessed',
        instrument_status='source-matched' if all(p['head']['auto-full-zero'] and all(p['tails']['auto-full-zero'].values()) for p in parity.values()) else 'source-mismatch-observed',
        trend_status='observed',scope=c['scope'],counts=s['counts'],native_contract=profile,errors=errors,archive_parity=parity,query_geometry=geometry,effects=effects,
        native_manifest_sha256=sha(root/'files.json'),source_manifest_sha256=sha(source/'files.json'),
        auditor_sha256=sha(Path(__file__)),measurement_sha256=sha(here/'analyze_flux_reader_arithmetic.py'),
        limitations=['Known-input numerical diagnosis; no fresh prediction or raw-prompt/image claim.',
            'Full-shape query controls use synthetic image queries; actual original image Q was not saved.',
            'Profiler records current selected ATen operators; original historical kernel selection was not captured.',
            'Two known receivers, one clock/seed; applicability is supplied, not induced.'])
    save(out/'audit.json',result)
    catalog=EquationCatalog(out/'catalog/equations');ids=[]
    def record(key,lane,p,n,kind):
        context=dict(model='flux2-klein-4b-distilled',source_job=s['job_id'],case=key,lane=lane,
            frontier='observed positioned text Q/full K/V at step3/joint0',horizon=1,
            numerical=dict(device=profile['device'],dtype='bfloat16',bf16_reduced_reduction=profile['default_bf16_reduced_reduction']))
        item=catalog.record(law_id='law.flux.reader.qkv-to-text.native.v1' if kind=='source-return' else 'law.flux.reader.attention-numerical-effect.v1',
            route_id='route.flux.image-KV.text-projection-residual-mlp.v1',alias='cmp.flux.reader-attention.'+key+'.'+lane,
            binding=dict(model_identity=context['model'],input_identity=c['parents'],source_identity=r['request_sha256'],execution=context['numerical'],
                frontier=dict(step=3,phase='joint',block=0),horizon=1),report=compare_equations(p,n,max_elements=20_000_000),
            conditions=[dict(field=f,op='eq',value=v) for f,v in context.items()],tested_context=context,
            readers=['joint0.text-head','joint0.text-projection','joint0.text-mlp'],
            evidence=dict(audit_sha256=sha(out/'audit.json'),role='known-target numerical diagnosis',comparison=kind,new_fitted_parameters=0,terminal_status='unassessed'))
        wrong={**context,'numerical':{**context['numerical'],'bf16_reduced_reduction':not profile['default_bf16_reduced_reduction']}}
        if len(catalog.find(applicable_to=context))!=1 or catalog.find(applicable_to=wrong):raise ValueError('catalog applicability differs')
        if key=='right.k1v1':
            save(out/'catalog/context-right.json',context);save(out/'catalog/context-wrong-policy.json',wrong)
        ids.append(item['fingerprint'])
    for key in rows:
        record(key,'auto-full-zero',dict(head=head(key,'auto-full-zero'),**{f:tail(key,'auto-full-zero',f) for f in rows[key]['tails']['auto-full-zero']}),
            {f:observed(key,f) for f in ('head','after_attention','mlp','text')},'source-return')
    for receiver in ('left','right'):
        for lane in sorted(effects[receiver]['heads']):
            keys=[f'{receiver}.k{ki}v{vi}' for ki,vi in ((0,0),(1,1))]
            p=[head(key,lane).astype(np.float64) for key in keys];n=[observed(key).astype(np.float64) for key in keys]
            record(receiver,lane,{'effect':p[1]-p[0]},{'effect':n[1]-n[0]},'joint K/V head effect')
        p=[tail(f'{receiver}.k{ki}v{vi}','math-text').astype(np.float64) for ki,vi in ((0,0),(1,1))]
        n=[observed(f'{receiver}.k{ki}v{vi}','text').astype(np.float64) for ki,vi in ((0,0),(1,1))]
        record(receiver,'math-tail',{'effect':p[1]-p[0]},{'effect':n[1]-n[0]},'joint K/V text effect')
    save(out/'catalog/provenance.json',dict(records=len(ids),fingerprints=ids,matching_queries=len(ids),wrong_policy_abstentions=len(ids),new_model_calls=0))
    return result,len(ids)


def worker():
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='2'
    h=runpy.run_path('flux_source_receiver.py');read,save,sha=h['read'],h['save'],h['sha']
    req=read(os.environ['SATURN_DEBUG_REQUEST']);c=req['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(sha(Path(n))!=v for n,v in req['source_hashes'].items()):raise ValueError('audit binding/isolation differs')
    root,source=Path(c['native_root']),Path(c['source_root'])
    if sha(root/'files.json')!=c['native_manifest_sha256'] or sha(source/'files.json')!=c['source_manifest_sha256']:raise ValueError('input inventories changed')
    started=time.monotonic();out=Path(c['output_scratch'])/os.environ['MRUN_JOB_ID']/'binding-minimal-result'
    result,n=audit(root,source,out)
    for name in ('cases.json','native-contract.json','summary.json'):shutil.copyfile(root/name,out/('source-'+name))
    for name in ('parameter-artifacts.json','retrieve_weights.py'):shutil.copyfile(root/name,out/name)
    save(out/'external-evidence.json',dict(host='beast',roots=dict(native=str(root),reader=str(source)),
        manifests=dict(native=read(root/'files.json'),reader=read(source/'files.json')),scope='Full tensors remain on Beast; retrieve individual checksum-bound files for raw re-audit.'))
    shutil.copyfile('retrieve_reader_evidence.py',out/'retrieve_evidence.py')
    save(out/'plan.json',c);save(out/'request.json',req)
    save(out/'summary.json',dict(schema='mdb-reader-attention-compact-v1',job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],
        output_root=str(out),source_job=result['job'],seconds=time.monotonic()-started,records=n,mechanics_status='verified',instrument_status=result['instrument_status'],terminal_status='unassessed',
        new_execution_counts=dict(full_models=0,native_primitives=0,mathematical_forwards=0,images=0)))
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('MDB_DIRECTORY_READY '+json.dumps(dict(job_id=os.environ['MRUN_JOB_ID'],request_sha256=req['request_sha256'],output_root=str(out),
        files_manifest_sha256=sha(out/'files.json'),summary_sha256=sha(out/'summary.json')),sort_keys=True),flush=True)


def stage(run,study):
    from mdb.job_support import one_lease_scheduling_contract, scheduler_api
    from mdb.payload import stage_worker
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));read,save,sha=h['read'],h['save'],h['sha']
    prior=read(study/'request.json');jid=read(study/'submission.json')['job_id']
    _,raw,_=scheduler_api().request('GET','/api/jobs/'+jid,timeout_s=20);job=json.loads(raw)
    if job['state']!='succeeded':raise ValueError('completed native operator job required')
    save(study/'job.json',job)
    root=Path(prior['config']['output_scratch'])/jid/'binding-minimal-result'
    code='import hashlib,json\nfrom pathlib import Path\np=Path('+repr(str(root))+')\nprint(json.dumps(dict(sha=hashlib.sha256((p/"files.json").read_bytes()).hexdigest(),files=sum(1 for x in p.rglob("*") if x.is_file()),bytes=sum(x.stat().st_size for x in p.rglob("*") if x.is_file()))))\n'
    info=json.loads(subprocess.run(['ssh','beast','python3','-I','-'],input=code,text=True,capture_output=True,check=True).stdout)
    c=dict(native_root=str(root),source_root=prior['config']['input_root'],native_manifest_sha256=info['sha'],
        source_manifest_sha256=prior['config']['source_files_sha256'],raw_files=info['files'],raw_bytes=info['bytes'],
        output_scratch='/mnt/big/mdb/research',directory_custody=True,scope='Saved-only numerical comparisons and supplied-applicability catalog; zero new model/native/reference forward calculations.')
    run.mkdir(parents=True,exist_ok=False);save(run/'plan.json',c)
    extras=[(here/n,n) for n in ('flux_source_receiver.py','analyze_flux_reader_arithmetic.py','flux_family_custody.py','retrieve_reader_evidence.py')]
    pre=run/'preflight.v1.json';save(pre,dict(schema_version='mrun-preflight-config-v1',hashed_roots=[Path(__file__).name,*[n for _,n in extras],pre.name],
        path_token_allowlist=[str(root),c['source_root'],c['output_scratch'],'binding-minimal-result']))
    extras.append((pre,pre.name))
    req=dict(schema='mdb-reader-attention-audit-request-v1',config=c,execution_environment=prior['execution_environment'],needs=dict(cpu=True,host='beast'),
        task_family='analysis',backend='numpy',device='cpu',dtype='float64',source_hashes={n:sha(p) for p,n in [(Path(__file__),Path(__file__).name),*extras]},
        resource_exception=dict(ram_mb=1536,vram_mb=0,cpu_threads=2,disk_gb=1,est_wall_s=60,
            calibration_job='job-c8a4bc3cc1e4',measured_peak_rss_mb=1328.5,reason='Prior saved-only audit1328.5MB peak included9.44M-element paired fields. Here all metrics use1.57M-element heads/text only, no paired arrays retained, no Torch import.1.5GB covers prior measured peak; raw corpus stays remote.'),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_operations=25,logical_branches=1,logical_checkpoint_replays=0,
            max_local_work_items=32,extra=dict(task_family='analysis'),checkpoint_policy='One near-data full-geometry saved-only audit and24catalog comparisons maximum; no native/model/reference forwards or retries.'))
    req['request_sha256']=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest();save(run/'request.json',req)
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](req['execution_environment']);stage_worker(run/'payload',worker=Path(__file__),config=c,request=req,additional_files=extras)
    return dict(request_sha256=req['request_sha256'],**info)


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command
    here=Path(__file__).resolve().parent;h=runpy.run_path(str(here/'flux_source_receiver.py'));r=h['read'](run/'request.json')
    runpy.run_path(str(here/'stage_flux_source_receiver.py'))['bind'](r['execution_environment'])
    if (run/'submission.json').exists() or any(h['sha'](run/'payload'/n)!=v for n,v in r['source_hashes'].items()):raise ValueError('changed/already submitted audit')
    cmd=worker_command(Path(__file__).name);cmd.insert(1,'CUDA_VISIBLE_DEVICES=');cmd.insert(cmd.index('python')+1,'-B')
    resources={k:r['resource_exception'][k] for k in ('ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s')}
    job=guarded_mrun_launch(cmd,request=r,model=None,task='analysis',experiment='mdb-reader-attention-audit',payload=run/'payload',gpu=False,
        needs=r['needs'],prefer_host='beast',timeout_s=240,retry_on_kill=False,preflight=True,**resources,
        note='Saved-only8case attention head/text audit,4signedK/V contrasts,24catalogrecordsmax; compact custody plus individual array retriever.0native/model/referenceforwards/images;CPU2/RAM1.5GB;60sestimate/240souter,no retries.')
    h['save'](run/'submission.json',dict(job_id=str(job)));return dict(job_id=str(job))


if __name__=='__main__':
    if os.environ.get('SATURN_RESIDENT_WORKER')=='1':worker()
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--study',type=Path);p.add_argument('--submit',action='store_true');a=p.parse_args()
        print(json.dumps(submit(a.run.resolve()) if a.submit else stage(a.run.resolve(),a.study.resolve())))
